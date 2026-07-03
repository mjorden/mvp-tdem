"""
QC and editing for helicopter TDEM data.

Operations (applied in order by run_qc)
----------------------------------------
1. noise_floor_flag    — mask gates at or below system noise floor
2. negative_gate_flag  — mask early-time negatives (cultural / instrument artefact)
3. altitude_flag       — flag soundings outside acceptable bird-height range
4. dem_consistency     — flag soundings where DEM > Elevation (GPS/radar mismatch)
5. along_line_despike  — lateral median filter; kills narrow along-line spikes per gate
6. monotonicity_flag   — flag soundings whose early-time gates are non-monotonic
                         (powerline / cultural EM contamination indicator)

Each operation writes a boolean mask column (True = bad) into a `_qc_*` namespace.
`sounding_mask` combines all flags into a single per-sounding boolean.
Gate masks are stored per-gate in `_qc_gate_<nn>`.

Nothing is dropped here — callers decide what to do with flagged data.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd

from .load import gate_columns, gate_array


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_qc(
    df: pd.DataFrame,
    config: dict,
    *,
    alt_min_m: float = 10.0,
    alt_max_m: float = 80.0,
    despike_window: int = 5,
    despike_threshold: float = 4.0,
    despike_min_gates: int = 3,
    mono_n_early: int = 8,
    mono_max_reversals: int = 1,
) -> pd.DataFrame:
    """
    Run the full QC suite and return df with added _qc_* columns.

    Parameters
    ----------
    df                 : output of load.load_survey()
    config             : sidecar config dict (needs system.system_noise_floor)
    alt_min_m          : flag soundings with bird altitude below this (m AGL)
    alt_max_m          : flag soundings with bird altitude above this (m AGL)
    despike_window     : half-width of median filter window (soundings, per line)
    despike_threshold  : gate deviates if |value - median| > threshold * MAD
    despike_min_gates  : flag sounding only if >= this many gates deviate at once
                         (controls the family-wise false-positive rate across gates)
    mono_n_early       : number of early gates to check for monotonic decay
    mono_max_reversals : tolerated sign-reversals before flagging (1 allows one blip)
    """
    df = df.copy()

    noise_floor = config["system"]["system_noise_floor"]
    gate_cols   = gate_columns(df)

    df = _noise_floor_flag(df, gate_cols, noise_floor)
    df = _negative_gate_flag(df, gate_cols)
    df = _altitude_flag(df, alt_min_m, alt_max_m)
    df = _dem_consistency_flag(df)
    df = _along_line_despike(df, gate_cols, despike_window, despike_threshold,
                             despike_min_gates, noise_floor=noise_floor)
    df = _monotonicity_flag(df, gate_cols, mono_n_early, mono_max_reversals)

    df["sounding_mask"] = _combine_sounding_flags(df)

    _print_summary(df, gate_cols)
    return df


# ---------------------------------------------------------------------------
# Individual QC steps
# ---------------------------------------------------------------------------

def _noise_floor_flag(df: pd.DataFrame, gate_cols: list[str], noise_floor: float) -> pd.DataFrame:
    """
    Mask individual gates at or below the system noise floor.

    noise_floor is in the same units as the data (V/(A·m⁴)).
    Writes _qc_gate_<nn> = True where gate is at/below noise floor.
    """
    for col in gate_cols:
        idx   = col.split("_")[-1]
        flag  = col_to_flag(col)
        vals  = df[col].to_numpy(dtype=float)
        df[flag] = np.where(np.isnan(vals), True, vals < noise_floor)
    return df


def _negative_gate_flag(df: pd.DataFrame, gate_cols: list[str]) -> pd.DataFrame:
    """
    Flag soundings that have any negative value in the early gates (first half).

    Late-time sign reversals can be geophysically real (inductive limit);
    early negatives are almost always cultural noise or instrument artefact.
    """
    n_early = max(1, len(gate_cols) // 2)
    early   = gate_cols[:n_early]
    arr     = df[early].to_numpy(dtype=float)
    df["_qc_neg_early"] = np.any(arr < 0, axis=1)
    return df


def _altitude_flag(df: pd.DataFrame, alt_min_m: float, alt_max_m: float) -> pd.DataFrame:
    """
    Flag soundings where the bird altitude (DEM) is outside the acceptable range.

    Too low  → cultural noise, terrain clearance hazard, footprint distortion.
    Too high → weak late-time signal, increased noise, reduced depth of investigation.
    """
    dem = df["dem"].to_numpy(dtype=float)
    df["_qc_alt_low"]  = dem < alt_min_m
    df["_qc_alt_high"] = dem > alt_max_m
    return df


def _dem_consistency_flag(
    df: pd.DataFrame,
    jump_threshold_m: float = 15.0,
    window: int = 7,
) -> pd.DataFrame:
    """
    Flag GPS/radar inconsistency via abrupt jumps in derived ground elevation.

    Ground elevation = GPS ellipsoid elevation − radar AGL. Its absolute value
    is NOT diagnostic: negative ellipsoid ground heights are physically valid
    (geoid undulation reaches ±100 m; sub-sea-level terrain exists) (#3).
    What IS diagnostic of a GPS dropout or radar lock-on artefact is ground
    elevation jumping tens of metres between adjacent soundings — terrain
    doesn't do that at 50 m station spacing, but a re-converging GPS or a
    radar bouncing off canopy does.

    Flags soundings whose ground elevation deviates from the along-line
    rolling median by more than jump_threshold_m.
    """
    ground = (df["elevation"].to_numpy(dtype=float)
              - df["dem"].to_numpy(dtype=float))
    flag = np.zeros(len(df), dtype=bool)

    lines = df["line"].unique() if "line" in df.columns else [None]
    for line_id in lines:
        pos = np.where(df["line"] == line_id)[0] if line_id is not None else np.arange(len(df))
        vals = ground[pos]
        if len(vals) < window:
            continue
        med = _rolling_median(vals, window // 2)
        flag[pos] = np.abs(vals - med) > jump_threshold_m

    df["_qc_dem_mismatch"] = flag
    return df


def _along_line_despike(
    df: pd.DataFrame,
    gate_cols: list[str],
    window: int,
    threshold: float,
    min_gates: int = 3,
    noise_floor: float = 0.0,
    rel_floor: float = 0.01,
) -> pd.DataFrame:
    """
    Per-gate lateral median filter along each flight line.

    For each gate and each line, compute a running median ± MAD over a window
    of (2*window + 1) soundings; a gate "deviates" when its distance from the
    running median exceeds threshold * MAD. A sounding is flagged only when
    >= min_gates deviate simultaneously — a real cultural hit contaminates
    many gates at once, whereas single-gate exceedances at 3–4 MAD are
    expected by chance across 30 gates.

    The MAD is floored at a physically meaningful scale (#7):
        mad >= rel_floor * |running median| + noise_floor
    ASCII deliverables quantize late-time gates to 3–4 significant figures,
    so the raw MAD can collapse to exactly 0 across whole stretches; an
    absolute-epsilon fallback would then flag every sounding that differs
    at all from its median.
    """
    n_deviating = np.zeros(len(df), dtype=int)

    lines = df["line"].unique() if "line" in df.columns else [None]

    for line_id in lines:
        pos = np.where(df["line"] == line_id)[0] if line_id is not None else np.arange(len(df))
        if len(pos) < 2 * window + 1:
            continue

        for col in gate_cols:
            vals = df[col].to_numpy(dtype=float)[pos]
            med  = _rolling_median(vals, window)
            mad  = _rolling_mad(vals, med, window)
            # physically-scaled floor (#7): never below rel_floor of the local
            # signal level nor the system noise floor
            mad  = np.maximum(mad, rel_floor * np.abs(med) + noise_floor)
            # 1.4826 converts MAD to a consistent σ estimate for Gaussian noise
            n_deviating[pos] += (np.abs(vals - med) > threshold * 1.4826 * mad).astype(int)

    df["_qc_spike"] = n_deviating >= min_gates
    return df


def _monotonicity_flag(
    df: pd.DataFrame,
    gate_cols: list[str],
    n_early: int,
    max_reversals: int,
) -> pd.DataFrame:
    """
    Flag soundings whose early-time gate sequence is non-monotonically decaying.

    Healthy TDEM decays monotonically in log-space. Multiple reversals in the
    early gates indicate powerline coupling, cultural EM, or system noise.
    One reversal is tolerated (mono_max_reversals=1) to allow for a single
    gate-level noise blip.
    """
    early_cols = gate_cols[:n_early]
    arr        = np.log1p(np.abs(df[early_cols].to_numpy(dtype=float)))
    # diff < 0 → decaying (good); diff > 0 → rising (reversal)
    diffs      = np.diff(arr, axis=1)
    reversals  = np.sum(diffs > 0, axis=1)
    df["_qc_nonmono"] = reversals > max_reversals
    return df


# ---------------------------------------------------------------------------
# Combination and summary
# ---------------------------------------------------------------------------

def _combine_sounding_flags(df: pd.DataFrame) -> pd.Series:
    """OR all per-sounding _qc_* boolean columns (excluding per-gate flags)."""
    sounding_flag_cols = [
        c for c in df.columns
        if c.startswith("_qc_") and not c.startswith("_qc_gate_")
    ]
    if not sounding_flag_cols:
        return pd.Series(False, index=df.index)
    return df[sounding_flag_cols].any(axis=1)


def _print_summary(df: pd.DataFrame, gate_cols: list[str]) -> None:
    n_total    = len(df)
    n_bad      = df["sounding_mask"].sum()
    gate_flags = [col_to_flag(c) for c in gate_cols]
    gate_flags = [f for f in gate_flags if f in df.columns]
    n_gate_bad = df[gate_flags].sum().sum() if gate_flags else 0

    print(
        f"[qc] {n_total} soundings — "
        f"{n_bad} flagged ({100*n_bad/n_total:.1f}%) | "
        f"{n_gate_bad} individual gate values below noise floor"
    )
    flag_cols = [c for c in df.columns if c.startswith("_qc_") and not c.startswith("_qc_gate_")]
    for col in flag_cols:
        n = df[col].sum()
        if n > 0:
            print(f"  {col}: {n}")


# ---------------------------------------------------------------------------
# Convenience helpers for callers
# ---------------------------------------------------------------------------

def col_to_flag(gate_col: str) -> str:
    """sfz_07 → _qc_gate_07"""
    return f"_qc_gate_{gate_col.split('_', 1)[-1]}"


def good_soundings(df: pd.DataFrame) -> pd.DataFrame:
    """Return only soundings that passed all QC checks."""
    if "sounding_mask" not in df.columns:
        raise RuntimeError("Run run_qc() before calling good_soundings().")
    return df[~df["sounding_mask"]].reset_index(drop=True)


def good_gate_array(df: pd.DataFrame) -> np.ndarray:
    """
    Return (n_soundings, n_gates) array with per-gate flagged values set to NaN.
    Operates on all soundings (including flagged ones) — filter with good_soundings() first.
    """
    from .load import gate_columns
    gate_cols  = gate_columns(df)
    arr        = df[gate_cols].to_numpy(dtype=float).copy()
    gate_flags = [col_to_flag(c) for c in gate_cols]
    for i, flag in enumerate(gate_flags):
        if flag in df.columns:
            arr[df[flag].to_numpy(), i] = np.nan
    return arr


# ---------------------------------------------------------------------------
# Low-level rolling statistics (no pandas rolling to avoid alignment edge cases)
# ---------------------------------------------------------------------------

def _rolling_median(vals: np.ndarray, half_window: int) -> np.ndarray:
    n   = len(vals)
    out = np.empty(n)
    for i in range(n):
        lo = max(0, i - half_window)
        hi = min(n, i + half_window + 1)
        out[i] = np.nanmedian(vals[lo:hi])
    return out


def _rolling_mad(vals: np.ndarray, median: np.ndarray, half_window: int) -> np.ndarray:
    n   = len(vals)
    out = np.empty(n)
    for i in range(n):
        lo = max(0, i - half_window)
        hi = min(n, i + half_window + 1)
        out[i] = np.nanmedian(np.abs(vals[lo:hi] - median[i]))
    return out
