"""
QC and editing for helicopter TDEM data.

Operations (applied in order by run_qc)
----------------------------------------
1. noise_floor_flag    — mask gates noise-dominated in magnitude (|v| ≤ k·floor)
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
    noise_floor_k: float = 2.0,
    despike_window: int = 5,
    despike_threshold: float = 4.0,
    despike_min_gates: int = 3,
    mono_n_early: int = 8,
    mono_max_reversals: int = 1,
    mono_min_rise_frac: float = 0.25,
) -> pd.DataFrame:
    """
    Run the full QC suite and return df with added _qc_* columns.

    Works for any input index — flags are computed positionally per line and
    the caller's index is preserved (#24), so subsetting before QC is safe.

    Parameters
    ----------
    df                 : output of load.load_survey()
    config             : sidecar config dict (needs system.system_noise_floor)
    alt_min_m          : flag soundings with bird altitude below this (m AGL)
    alt_max_m          : flag soundings with bird altitude above this (m AGL)
    noise_floor_k      : gate is noise-dominated when |value| <= this multiple
                         of the system noise floor (#13)
    despike_window     : half-width of median filter window (soundings, per line)
    despike_threshold  : gate deviates if |value - median| > threshold * sigma,
                         with sigma = 1.4826 * MAD — threshold is in true
                         Gaussian-sigma units (#31)
    despike_min_gates  : flag sounding only if >= this many gates deviate at once
                         (controls the family-wise false-positive rate across gates)
    mono_n_early       : number of early gates to check for monotonic decay
    mono_max_reversals : tolerated reversals before flagging (1 allows one blip)
    mono_min_rise_frac : a reversal counts only when the amplitude rises by more
                         than this fraction between consecutive gates (#14)
    """
    df = df.copy()

    noise_floor = config["system"]["system_noise_floor"]
    gate_cols   = gate_columns(df)

    df = _noise_floor_flag(df, gate_cols, noise_floor, k=noise_floor_k)
    df = _negative_gate_flag(df, gate_cols)
    df = _altitude_flag(df, alt_min_m, alt_max_m)
    df = _dem_consistency_flag(df)
    df = _along_line_despike(df, gate_cols, despike_window, despike_threshold,
                             despike_min_gates, noise_floor=noise_floor)
    df = _monotonicity_flag(df, gate_cols, mono_n_early, mono_max_reversals,
                            noise_floor=noise_floor,
                            min_rise_frac=mono_min_rise_frac)

    df["sounding_mask"] = _combine_sounding_flags(df)

    _print_summary(df, gate_cols)
    return df


# ---------------------------------------------------------------------------
# Individual QC steps
# ---------------------------------------------------------------------------

def _noise_floor_flag(
    df: pd.DataFrame, gate_cols: list[str], noise_floor: float, k: float = 2.0
) -> pd.DataFrame:
    """
    Mask individual gates that are noise-dominated: |value| <= k * noise_floor.

    noise_floor is in the same units as the data (V/(A·m⁴)). Testing magnitude
    rather than signed value (#13) means geophysically real late-time negatives
    are treated exactly like positives of the same amplitude — early negatives
    remain _negative_gate_flag's job — and gates at S/N up to k are masked as
    noise-dominated rather than only those literally at/below the floor.
    NaN gates are not flagged here: they are already unusable and stay NaN in
    good_gate_array() regardless.
    Writes _qc_gate_<nn> = True where the gate is noise-dominated.
    """
    for col in gate_cols:
        flag = col_to_flag(col)
        vals = df[col].to_numpy(dtype=float)
        df[flag] = np.abs(vals) <= k * noise_floor
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

    for pos in _line_positions(df):
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

    For each gate and each line, compute a running median over a window of
    (2*window + 1) soundings and a robust scale estimate
        sigma = 1.4826 * MAD
    (the Gaussian consistency constant, #31 — so threshold is in true sigma
    units: threshold=4 is a 4-sigma test, not the 2.7-sigma test raw MAD
    gives). A gate "deviates" when |value - median| > threshold * sigma.
    A sounding is flagged only when >= min_gates deviate simultaneously — a
    real cultural hit contaminates many gates at once, whereas single-gate
    exceedances at 3-4 sigma are expected by chance across 30 gates.

    sigma is floored at a physically meaningful scale (#7):
        sigma >= rel_floor * |running median| + noise_floor
    ASCII deliverables quantize late-time gates to 3–4 significant figures,
    so the raw MAD can collapse to exactly 0 across whole stretches; an
    absolute-epsilon fallback would then flag every sounding that differs
    at all from its median.
    """
    n_deviating = np.zeros(len(df), dtype=int)

    for pos in _line_positions(df):
        if len(pos) < 2 * window + 1:
            continue

        for col in gate_cols:
            vals = df[col].to_numpy(dtype=float)[pos]
            med  = _rolling_median(vals, window)
            mad  = _rolling_mad(vals, med, window)
            sigma = 1.4826 * mad  # Gaussian consistency constant (#31)
            # physically-scaled floor (#7): never below rel_floor of the local
            # signal level nor the system noise floor
            sigma = np.maximum(sigma, rel_floor * np.abs(med) + noise_floor)
            n_deviating[pos] += (np.abs(vals - med) > threshold * sigma).astype(int)

    df["_qc_spike"] = n_deviating >= min_gates
    return df


def _monotonicity_flag(
    df: pd.DataFrame,
    gate_cols: list[str],
    n_early: int,
    max_reversals: int,
    noise_floor: float = 0.0,
    min_rise_frac: float = 0.25,
) -> pd.DataFrame:
    """
    Flag soundings whose early-time gate sequence is non-monotonically decaying.

    Healthy TDEM decays monotonically in log-space; powerline coupling and
    cultural EM produce large upward excursions. The test (#14) is on
    log(|x| + noise_floor) — a true log (log1p was a no-op at gate amplitudes
    of 1e-7..1e-12), with the floor keeping noise-dominated wiggles compressed
    — and a rise only counts as a reversal when the amplitude grows by more
    than min_rise_frac between consecutive gates, so a fractional noise
    uptick is not weighted like a 10x powerline hit. NaN gates are bridged by
    carrying the last finite value forward, so a dummy gate can neither hide
    nor fake a reversal. Sign flips are invisible to |x|; early negatives are
    _negative_gate_flag's job.
    One reversal is tolerated (mono_max_reversals=1) to allow for a single
    gate-level noise blip.
    """
    early = df[gate_cols[:n_early]].to_numpy(dtype=float)
    with np.errstate(divide="ignore"):
        log_amp = np.log(np.abs(early) + noise_floor)
    # bridge NaN gates: each diff compares consecutive *finite* gates (#14)
    filled = pd.DataFrame(log_amp).ffill(axis=1).to_numpy()
    diffs  = np.diff(filled, axis=1)
    # diff < 0 → decaying (good); a reversal must be a material rise
    reversals = np.sum(diffs > np.log1p(min_rise_frac), axis=1)
    df["_qc_nonmono"] = reversals > max_reversals
    return df


def _line_positions(df: pd.DataFrame) -> list[np.ndarray]:
    """
    Positional (iloc) index arrays, one per flight line, in file order.

    Positional throughout (#24): index *labels* are never used to address
    positional numpy arrays, so run_qc works on any input index — including
    frames subset by the caller before QC.
    """
    if "line" not in df.columns:
        return [np.arange(len(df))]
    line_vals = df["line"].to_numpy()
    return [np.flatnonzero(line_vals == lid) for lid in pd.unique(line_vals)]


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
        f"{n_gate_bad} individual gate values noise-dominated"
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
