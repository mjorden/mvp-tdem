"""
Physics normalization: logged volts → V/(A·m⁴).

    v_norm = v_logged / rx_gain / rx_coil_area_m2 / tx_moment

The Tx moment is per-sounding when a Tx-current stream exists
(I × n_turns × loop_area, current interpolated onto sounding times);
otherwise the nominal moment from instrument.yaml. Which path was taken
is recorded so the sidecar provenance can say `moment: measured|nominal`.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .merge import interp_with_gaps
from .stack import stacked_gate_columns


def calibrate(
    soundings: pd.DataFrame,
    instrument: dict,
    txcur_df: pd.DataFrame | None = None,
    max_gap_s: float = 2.0,
    window_s: float = 0.0,
) -> tuple[pd.DataFrame, str]:
    """
    Normalize stacked gate values (and stds) in place-of-copy.

    Returns (calibrated df, moment_mode) where moment_mode is
    "measured" or "nominal".

    window_s : stack-window duration. When > 0 the Tx moment for each sounding
               is the current AVERAGED over its window rather than a single
               interpolated instant (#65.1): a 5 Hz monitor sampled once per
               0.48 s window aliases battery sag / generator ripple straight into
               gate amplitudes. The current is also validated (#65.2): a
               non-positive value means the monitor logged signed bipolar current
               (catastrophically aliased at 5 Hz) and raises; a mean far from
               nominal warns.
    """
    rx = instrument["rx"]
    tx = instrument["tx"]
    denom_fixed = rx["gain"] * rx["coil_area_m2"]
    nominal_current = tx["moment_nominal_am2"] / (tx["n_turns"] * tx["loop_area_m2"])

    if txcur_df is not None and len(txcur_df):
        if "t_utc" not in txcur_df.columns:
            raise ValueError("Tx-current frame has no t_utc — run timesync.apply_clock first")
        cur_t = txcur_df["t_utc"].to_numpy()
        cur_i = txcur_df["current_a"].to_numpy()
        centre = soundings["t_utc"].to_numpy()

        current = interp_with_gaps(centre, cur_t, cur_i, max_gap_s=max_gap_s)
        # window average where enough samples exist (#65.1); interp value (already
        # computed) remains the fallback for sparsely-sampled windows and gaps
        if window_s > 0:
            hw = window_s / 2.0
            n_windowed = 0
            for j, c in enumerate(centre):
                in_win = np.abs(cur_t - c) <= hw
                if in_win.sum() >= 2:
                    current[j] = cur_i[in_win].mean()
                    n_windowed += 1
            # #6: if almost no window held >=2 samples, the de-aliasing was a
            # silent no-op (monitor too slow for this window / high base freq)
            frac = n_windowed / max(len(centre), 1)
            if frac < 0.5:
                warnings.warn(
                    f"[calibrate] Tx-current window averaging fell back to single-"
                    f"instant interpolation for {100*(1-frac):.0f}% of soundings "
                    f"(monitor too sparse for the {window_s*1000:.0f} ms window); "
                    "current is NOT de-aliased for those.",
                    stacklevel=2,
                )

        current = _sanitize_current(current, nominal_current)

        # fall back to nominal current inside Tx-log gaps rather than
        # losing the sounding — the EM data itself is fine
        has_gaps = np.isnan(current).any()
        current = np.where(np.isnan(current), nominal_current, current)
        moment = current * tx["n_turns"] * tx["loop_area_m2"]
        moment_mode = "measured_with_gaps" if has_gaps else "measured"
    else:
        moment = np.full(len(soundings), float(tx["moment_nominal_am2"]))
        moment_mode = "nominal"

    out = soundings.copy()
    scale = 1.0 / (denom_fixed * moment)
    for col in stacked_gate_columns(out):
        out[col] = out[col] * scale
        out[f"gate_std_{col[-2:]}"] = out[f"gate_std_{col[-2:]}"] * scale
    return out, moment_mode


def _sanitize_current(current: np.ndarray, nominal_a: float,
                      max_bad_frac: float = 0.2) -> np.ndarray:
    """
    Sanity-check the measured Tx current (#65.2) and return a cleaned copy.

    Readers stay dumb; the physics check lives here. Two regimes are
    distinguished (#5-review — a single bad sample must not abort a good flight):

    * SYSTEMATIC (> max_bad_frac of samples non-positive): the monitor is logging
      SIGNED bipolar current (aliased at ~5 Hz), not the plateau magnitude — this
      cannot be normalized by, so raise.
    * ISOLATED (a few non-positive samples: a glitch or a brief Tx-off dropout):
      mask those to NaN and warn; the existing gap path fills them with nominal,
      preserving the otherwise-good EM data.

    A surviving median far from the nominal plateau only warns.
    """
    finite_mask = np.isfinite(current)
    finite = current[finite_mask]
    if finite.size == 0:
        return current

    nonpos_frac = float(np.count_nonzero(finite <= 0)) / finite.size
    if nonpos_frac > max_bad_frac:
        raise ValueError(
            f"{100*nonpos_frac:.0f}% of Tx current samples are non-positive — the "
            "monitor is logging signed bipolar current, not the plateau magnitude. "
            "A signed log cannot be normalized by; rectify it upstream."
        )

    current = current.astype(float, copy=True)
    bad = finite_mask & (current <= 0)
    if bad.any():
        warnings.warn(
            f"[calibrate] masked {int(bad.sum())} non-positive Tx current "
            "value(s) (glitch/dropout) → nominal fallback for those soundings.",
            stacklevel=2,
        )
        current[bad] = np.nan

    good = current[np.isfinite(current) & (current > 0)]
    if good.size:
        ratio = float(np.median(good)) / nominal_a
        if not 0.8 <= ratio <= 1.2:
            warnings.warn(
                f"[calibrate] median Tx current is {ratio:.2f}× nominal "
                f"({np.median(good):.1f} A vs {nominal_a:.1f} A) — outside ±20%. "
                "Check the current monitor or the nominal moment in instrument.yaml.",
                stacklevel=2,
            )
    return current
