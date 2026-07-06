"""
QC and editing for helicopter TDEM data.

Operations (applied in order by run_qc)
----------------------------------------
1. noise_floor_flag    — mask gates whose *amplitude* |x| is below the system
                         noise floor (#55: signed comparison would mask real
                         late-time negatives; "below noise floor" is an
                         amplitude concept, not a signed one)
2. negative_gate_flag  — flag early-*time* negatives (cultural / instrument
                         artefact); late-time negatives are preserved (IP /
                         chargeability effects, #44). Split is by gate TIME, not
                         index (#66.6), so it doesn't drift with gate count/spacing
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
    neg_early_max_ms: float = 0.2,
    neg_min_early: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the full QC suite and return df with added _qc_* columns.

    Parameters
    ----------
    df                 : output of load.load_survey()
    config             : sidecar config dict (needs system.system_noise_floor
                         and gate_times_ms for the time-based negative split)
    alt_min_m          : flag soundings with bird altitude below this (m AGL)
    alt_max_m          : flag soundings with bird altitude above this (m AGL)
    despike_window     : half-width of median filter window (soundings, per line)
    despike_threshold  : gate deviates if |value - median| > threshold * σ
    despike_min_gates  : flag sounding only if >= this many gates deviate at once
                         (controls the family-wise false-positive rate across gates;
                         scaled down for soundings with many NaN gates, #66.2)
    mono_n_early       : number of early gates to check for monotonic decay
    mono_max_reversals : tolerated reversals before flagging (1 allows one blip)
    neg_early_max_ms   : gates earlier than this (by TIME, #66.6) are "early-time";
                         negatives there are treated as cultural. Later negatives
                         are preserved (IP / chargeability effects, #44)
    neg_min_early      : escalate the per-sounding negative flag only when at
                         least this many early gates are negative (#66.6)
    verbose            : print the QC summary to stdout (#38). The same summary
                         is always attached to the returned df.attrs["qc_summary"]
                         (see qc_summary()), regardless of this flag.

    Returns
    -------
    df with _qc_* flag columns, a combined `sounding_mask`, and a programmatic
    summary dict in `df.attrs["qc_summary"]`.
    """
    df = df.copy()

    noise_floor   = config["system"]["system_noise_floor"]
    gate_times_ms = config.get("gate_times_ms")
    gate_cols     = gate_columns(df)

    df = _noise_floor_flag(df, gate_cols, noise_floor)
    df = _negative_gate_flag(df, gate_cols, gate_times_ms,
                             early_max_ms=neg_early_max_ms, min_early=neg_min_early)
    df = _altitude_flag(df, alt_min_m, alt_max_m)
    df = _dem_consistency_flag(df)
    df = _along_line_despike(df, gate_cols, despike_window, despike_threshold,
                             despike_min_gates, noise_floor=noise_floor)
    df = _monotonicity_flag(df, gate_cols, mono_n_early, mono_max_reversals,
                            noise_floor=noise_floor)

    df["sounding_mask"] = _combine_sounding_flags(df)

    # summary travels with the data (#38); prints are just a verbose view of it
    summary = qc_summary(df)
    df.attrs["qc_summary"] = summary
    if verbose:
        _print_summary(summary)
    return df


# ---------------------------------------------------------------------------
# Individual QC steps
# ---------------------------------------------------------------------------

def _noise_floor_flag(df: pd.DataFrame, gate_cols: list[str], noise_floor: float) -> pd.DataFrame:
    """
    Mask individual gates whose amplitude is below the system noise floor.

    noise_floor is in the same units as the data (V/(A·m⁴)). "Below the noise
    floor" is an AMPLITUDE statement, so the test is on |value| (#55): a signed
    `value < noise_floor` test would flag every negative gate — including the
    real late-time negatives that `_negative_gate_flag` deliberately preserves —
    and `good_gate_array()` would then NaN them out, silently contradicting the
    negative-gate policy. Sign handling is routed exclusively through
    `_negative_gate_flag`.

    Writes _qc_gate_<nn> = True where |gate| < noise_floor (strict, #13) or NaN.
    """
    for col in gate_cols:
        flag  = col_to_flag(col)
        vals  = df[col].to_numpy(dtype=float)
        df[flag] = np.where(np.isnan(vals), True, np.abs(vals) < noise_floor)
    return df


def _negative_gate_flag(
    df: pd.DataFrame,
    gate_cols: list[str],
    gate_times_ms: Sequence[float] | None = None,
    early_max_ms: float = 0.2,
    min_early: int = 1,
) -> pd.DataFrame:
    """
    Flag soundings with negative values in the EARLY-TIME gates.

    Late-time sign reversals can be geophysically real (IP / chargeability
    effects — NOT the "inductive limit", which is the high-frequency in-phase
    asymptote and produces no off-time sign reversal, #44); early negatives are
    almost always cultural noise or instrument artefact.

    The early/late boundary is defined by gate TIME, not gate index (#66.6):
    an index-based `gate_cols[:n//2]` split silently moves the physical boundary
    whenever the gate count or spacing changes. When `gate_times_ms` is supplied
    (run_qc always supplies it), gates with t < `early_max_ms` are "early".
    When it is absent (direct unit-test calls), the split falls back to the
    first half by index for backward compatibility.

    The per-sounding `_qc_neg_early` escalates only when at least `min_early`
    early gates are negative (#66.6 — with min_early>1 a single noise-driven
    negative need not condemn the whole sounding).
    """
    arr = df[gate_cols].to_numpy(dtype=float)
    neg = arr < 0

    if gate_times_ms is not None and len(gate_times_ms) == len(gate_cols):
        t = np.asarray(gate_times_ms, dtype=float)
        early_mask = t < early_max_ms
        if not early_mask.any():          # all gates later than the cutoff
            early_mask[0] = True          # keep at least the first gate "early"
    else:
        n_early = max(1, len(gate_cols) // 2)
        early_mask = np.zeros(len(gate_cols), dtype=bool)
        early_mask[:n_early] = True

    n_neg_early = np.sum(neg[:, early_mask], axis=1)
    df["_qc_neg_early"] = n_neg_early >= max(1, min_early)
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

    The σ estimate is floored at a physically meaningful scale (#7, #66.1):
        sigma = max(1.4826 * mad,  rel_floor * |running median| + noise_floor)
    The 1.4826 MAD→σ consistency constant is applied to the RAW mad BEFORE the
    floor (#66.1): the previous `max(mad, floor) * 1.4826` inflated the physical
    floor itself by 1.4826× (~48% looser wherever the floor binds), contradicting
    the documented meaning of `rel_floor`/`noise_floor`. ASCII deliverables
    quantize late-time gates to 3–4 significant figures, so the raw MAD can
    collapse to exactly 0 across whole stretches; the floor prevents flagging
    every sounding that differs at all from its median.

    NaN handling (#66.2): NaN gates can never satisfy the deviation test, so a
    sounding with several NaN gates would need proportionally more of its
    surviving gates to deviate before hitting a fixed min_gates — QC would be
    systematically more lenient on the worst data. The per-sounding threshold is
    therefore scaled by the fraction of valid gates:
        effective_min = max(2, round(min_gates * n_valid / n_gates))
    so a half-NaN sounding is judged on a proportionally lower bar.
    """
    n_deviating = np.zeros(len(df), dtype=int)
    arr_all = df[gate_cols].to_numpy(dtype=float)
    n_valid = np.isfinite(arr_all).sum(axis=1)          # per sounding

    lines = df["line"].unique() if "line" in df.columns else [None]

    for line_id in lines:
        pos = np.where(df["line"] == line_id)[0] if line_id is not None else np.arange(len(df))
        if len(pos) < 2 * window + 1:
            # too short to build a rolling window — cannot check (#66.5).
            # A False flag here means "not checked", not "clean".
            if line_id is not None and len(pos) > 0:
                warnings.warn(
                    f"[qc] line {line_id}: {len(pos)} soundings < despike window "
                    f"{2 * window + 1}; along-line despike not applied to this line.",
                    stacklevel=2,
                )
            continue

        for col in gate_cols:
            vals = df[col].to_numpy(dtype=float)[pos]
            med  = _rolling_median(vals, window)
            mad  = _rolling_mad(vals, med, window)
            # 1.4826 → σ on the RAW mad, THEN floor (#66.1)
            sigma = np.maximum(1.4826 * mad, rel_floor * np.abs(med) + noise_floor)
            # NaN vals compare False, so they never add to n_deviating
            n_deviating[pos] += (np.abs(vals - med) > threshold * sigma).astype(int)

    # scale the per-sounding trigger by valid-gate fraction (#66.2)
    n_gates = max(1, len(gate_cols))
    eff_min = np.maximum(2, np.round(min_gates * n_valid / n_gates)).astype(int)
    eff_min = np.minimum(eff_min, min_gates)            # never stricter than default
    df["_qc_spike"] = n_deviating >= eff_min
    return df


def _monotonicity_flag(
    df: pd.DataFrame,
    gate_cols: list[str],
    n_early: int,
    max_reversals: int,
    noise_floor: float = 0.0,
    min_valid_early: int = 3,
) -> pd.DataFrame:
    """
    Flag soundings whose early-time gate sequence is non-monotonically decaying.

    Healthy TDEM decays monotonically in log-space. Multiple reversals in the
    early gates indicate powerline coupling, cultural EM, or system noise.

    Three defects in the previous implementation are fixed (#14, #66.3):

    * `log1p(|x|) ≈ |x|` to machine precision for gate amplitudes 1e-7..1e-12,
      so the advertised log-space transform was a no-op. Use
      `log(|x| + noise_floor)` — genuine log spacing, and the additive floor
      keeps the smallest gates finite and comparable.
    * The reversal size was ignored: a +0.1 % noise uptick counted the same as a
      10× powerline hit. A reversal now counts only if the log-space rise
      exceeds `rev_tol` (a noise-scaled tolerance), so sub-noise wobble is
      ignored.
    * `abs()` before diff hid sign flips (+a → −a with |·| unchanged reads as
      flat). Sign changes within the early window are counted as reversals
      directly — that IS the powerline signature the check targets.

    NaN gates break the diff chain; adjacent NaNs previously under-counted
    reversals (NaN comparisons are False) so contaminated soundings passed as
    monotonic. Soundings with fewer than `min_valid_early` valid early gates are
    flagged (cannot be verified monotonic).
    """
    early = df[gate_cols[:n_early]].to_numpy(dtype=float)
    n = len(df)

    # log-space magnitude with an additive floor so 1e-12 gates stay finite
    log_mag = np.log(np.abs(early) + max(noise_floor, 1e-300))
    # noise-scaled tolerance: a rise smaller than this is treated as flat
    rev_tol = np.log1p(0.05)   # ignore < ~5% log-space upticks

    reversals = np.zeros(n, dtype=int)
    n_valid   = np.isfinite(early).sum(axis=1)

    for i in range(n):
        finite = np.isfinite(early[i])
        if finite.sum() < 2:
            continue
        seq_log  = log_mag[i][finite]
        seq_sign = np.sign(early[i][finite])
        d = np.diff(seq_log)
        # (a) amplitude rise beyond tolerance; (b) sign flip between neighbours
        amp_rev  = d > rev_tol
        sign_rev = np.diff(seq_sign) != 0
        reversals[i] = int(np.sum(amp_rev | sign_rev))

    nonmono = reversals > max_reversals
    # can't verify monotonicity without enough valid early gates → flag
    nonmono |= n_valid < min_valid_early
    df["_qc_nonmono"] = nonmono
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


def qc_summary(df: pd.DataFrame) -> dict:
    """
    Build a programmatic QC summary (#38): counts that travel WITH the data
    instead of only reaching stdout. Also attached to df.attrs["qc_summary"] by
    run_qc, so batch callers can flag high-failure lines and tests can assert on
    drop counts without capturing prints.
    """
    from .load import gate_columns
    gate_cols  = gate_columns(df)
    n_total    = len(df)
    n_flagged  = int(df["sounding_mask"].sum()) if "sounding_mask" in df.columns else 0
    gate_flags = [f for f in (col_to_flag(c) for c in gate_cols) if f in df.columns]
    n_gate_bad = int(df[gate_flags].to_numpy().sum()) if gate_flags else 0
    per_flag = {
        c: int(df[c].sum())
        for c in df.columns
        if c.startswith("_qc_") and not c.startswith("_qc_gate_")
    }
    return {
        "n_total": n_total,
        "n_flagged": n_flagged,
        "flagged_frac": (n_flagged / n_total) if n_total else 0.0,
        "n_gate_below_floor": n_gate_bad,
        "per_flag": per_flag,
    }


def _print_summary(summary: dict) -> None:
    n_total, n_flagged = summary["n_total"], summary["n_flagged"]
    frac = 100 * summary["flagged_frac"]
    # the per-gate flag is set for both sub-floor AND NaN gates, so label it
    # honestly rather than "below noise floor" alone (#68.6)
    print(
        f"[qc] {n_total} soundings — "
        f"{n_flagged} flagged ({frac:.1f}%) | "
        f"{summary['n_gate_below_floor']} individual gates flagged (below floor or NaN)"
    )
    for col, n in summary["per_flag"].items():
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
    present    = [f in df.columns for f in gate_flags]
    # #41: a gate missing its per-gate flag AFTER run_qc (which creates one for
    # every gate) is a flag/gate desync — silently skipping it would degrade a
    # bad gate to "unflagged". Warn if some flags exist but not all.
    if any(present) and not all(present):
        missing = [f for f, p in zip(gate_flags, present) if not p]
        warnings.warn(
            f"[qc] good_gate_array: {len(missing)} gate(s) lack a _qc_gate_ flag "
            f"({missing[:3]}…) — flag/gate desync; those gates are treated as "
            "unflagged. Did run_qc() see the same gate columns?",
            stacklevel=2,
        )
    for i, flag in enumerate(gate_flags):
        if present[i]:
            arr[df[flag].to_numpy(), i] = np.nan
    return arr


# ---------------------------------------------------------------------------
# Low-level rolling statistics (no pandas rolling to avoid alignment edge cases)
# ---------------------------------------------------------------------------

def _rolling_median(vals: np.ndarray, half_window: int) -> np.ndarray:
    n   = len(vals)
    out = np.empty(n)
    # an all-NaN window yields NaN (intended); silence numpy's empty-slice noise (#66.2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i in range(n):
            lo = max(0, i - half_window)
            hi = min(n, i + half_window + 1)
            out[i] = np.nanmedian(vals[lo:hi])
    return out


def _rolling_mad(vals: np.ndarray, median: np.ndarray, half_window: int) -> np.ndarray:
    n   = len(vals)
    out = np.empty(n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i in range(n):
            lo = max(0, i - half_window)
            hi = min(n, i + half_window + 1)
            out[i] = np.nanmedian(np.abs(vals[lo:hi] - median[i]))
    return out
