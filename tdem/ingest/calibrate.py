"""
Physics normalization: logged volts → V/(A·m⁴).

    v_norm = v_logged / rx_gain / rx_coil_area_m2 / tx_moment

The Tx moment is per-sounding when a Tx-current stream exists
(I × n_turns × loop_area, current interpolated onto sounding times);
otherwise the nominal moment from instrument.yaml. Which path was taken
is recorded so the sidecar provenance can say `moment: measured|nominal`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .merge import interp_with_gaps
from .stack import stacked_gate_columns


def calibrate(
    soundings: pd.DataFrame,
    instrument: dict,
    txcur_df: pd.DataFrame | None = None,
    max_gap_s: float = 2.0,
) -> tuple[pd.DataFrame, str]:
    """
    Normalize stacked gate values (and stds) in place-of-copy.

    Returns (calibrated df, moment_mode) where moment_mode is "measured",
    "measured (N of M nominal-filled)" when Tx-log gaps forced nominal
    fallback for some soundings (#35), or "nominal".
    """
    rx = instrument["rx"]
    tx = instrument["tx"]
    denom_fixed = rx["gain"] * rx["coil_area_m2"]

    if txcur_df is not None and len(txcur_df):
        if "t_utc" not in txcur_df.columns:
            raise ValueError("Tx-current frame has no t_utc — run timesync.apply_clock first")
        current = interp_with_gaps(
            soundings["t_utc"].to_numpy(), txcur_df["t_utc"].to_numpy(),
            txcur_df["current_a"].to_numpy(), max_gap_s=max_gap_s,
        )
        # fall back to nominal current inside Tx-log gaps rather than
        # losing the sounding — the EM data itself is fine
        nominal_current = tx["moment_nominal_am2"] / (tx["n_turns"] * tx["loop_area_m2"])
        n_filled = int(np.isnan(current).sum())
        current = np.where(np.isnan(current), nominal_current, current)
        moment = current * tx["n_turns"] * tx["loop_area_m2"]
        if n_filled:
            # moment errors map 1:1 into amplitude — provenance must not
            # claim "measured" for soundings normalized by the nominal (#35)
            print(f"[calibrate] Tx-current gaps: {n_filled} of {len(soundings)} "
                  "soundings nominal-filled")
            moment_mode = f"measured ({n_filled} of {len(soundings)} nominal-filled)"
        else:
            moment_mode = "measured"
    else:
        moment = np.full(len(soundings), float(tx["moment_nominal_am2"]))
        moment_mode = "nominal"

    out = soundings.copy()
    scale = 1.0 / (denom_fixed * moment)
    for col in stacked_gate_columns(out):
        idx = col.removeprefix("gate_")  # full index string, not col[-2:] (#41)
        out[col] = out[col] * scale
        out[f"gate_std_{idx}"] = out[f"gate_std_{idx}"] * scale
    return out, moment_mode
