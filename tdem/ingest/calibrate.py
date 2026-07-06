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
            for j, c in enumerate(centre):
                in_win = np.abs(cur_t - c) <= hw
                if in_win.sum() >= 2:
                    current[j] = cur_i[in_win].mean()

        _validate_current(current, nominal_current)

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


def _validate_current(current: np.ndarray, nominal_a: float) -> None:
    """
    Sanity-check the measured Tx current (#65.2). Readers stay dumb; the physics
    check lives here.

    A finite non-positive current is a hard error: the monitor is logging signed
    bipolar current (aliased at ~5 Hz), so interpolation yields near-zero or
    negative moments and calibrated gates explode or flip sign. A mean far from
    the nominal plateau only warns — it may be a real generator problem, or the
    nominal in instrument.yaml may be stale.
    """
    finite = current[np.isfinite(current)]
    if finite.size == 0:
        return
    if np.any(finite <= 0):
        n_bad = int(np.count_nonzero(finite <= 0))
        raise ValueError(
            f"Tx current has {n_bad} non-positive value(s) — the monitor is "
            "logging signed bipolar current, not the plateau magnitude. Rectify "
            "it upstream; a 5 Hz signed log cannot be normalized by."
        )
    ratio = float(np.median(finite)) / nominal_a
    if not 0.8 <= ratio <= 1.2:
        warnings.warn(
            f"[calibrate] median Tx current is {ratio:.2f}× nominal "
            f"({np.median(finite):.1f} A vs {nominal_a:.1f} A) — outside ±20%. "
            "Check the current monitor or the nominal moment in instrument.yaml.",
            stacklevel=2,
        )
