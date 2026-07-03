"""
Receiver-clock → GPS-time alignment.

GPS time is the master clock. The receiver's oscillator has unknown offset
and drift, so we fit t_utc = offset + rate * t_rx over the flight's sync
pairs (GPS time messages latched against the receiver clock) and apply the
model to every t_rx-stamped stream (EM, altimeter, Tx current).

A bad fit is an error, not a warning: gate timing errors of a fraction of
a gate width corrupt the decay, so residuals above max_residual_ms mean
the sync stream itself is broken and the flight needs inspection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClockModel:
    offset_s: float          # t_utc at t_rx = 0
    rate: float              # d(t_utc)/d(t_rx); 1 + drift
    max_residual_s: float    # worst |fit residual| over the sync pairs
    n_pairs: int

    def to_utc(self, t_rx: np.ndarray) -> np.ndarray:
        return self.offset_s + self.rate * np.asarray(t_rx, dtype=float)


def fit_clock(sync_df: pd.DataFrame, max_residual_ms: float = 1.0) -> ClockModel:
    """Fit the linear clock model from sync pairs; raise if residuals exceed spec."""
    if len(sync_df) < 2:
        raise ValueError(f"Need >= 2 sync pairs to fit the clock model, got {len(sync_df)}")

    t_rx  = sync_df["t_rx"].to_numpy(dtype=float)
    t_utc = sync_df["t_utc"].to_numpy(dtype=float)

    # Centre both axes before the fit — unix-epoch t_utc (~1.8e9 s) in the
    # normal equations otherwise costs float64 precision on the drift term.
    t0, u0 = t_rx.mean(), t_utc.mean()
    rate, c = np.polyfit(t_rx - t0, t_utc - u0, 1)
    offset = u0 + c - rate * t0

    resid = t_utc - (offset + rate * t_rx)
    worst = float(np.abs(resid).max())
    if worst > max_residual_ms / 1000.0:
        raise ValueError(
            f"Clock fit residual {worst*1000:.3f} ms exceeds {max_residual_ms} ms over "
            f"{len(sync_df)} sync pairs — sync stream is inconsistent (dropped PPS? "
            "garbled time messages?). Inspect sync_*.log before trusting this flight."
        )
    return ClockModel(offset_s=float(offset), rate=float(rate),
                      max_residual_s=worst, n_pairs=len(sync_df))


def apply_clock(df: pd.DataFrame, model: ClockModel, t_col: str = "t_rx") -> pd.DataFrame:
    """Return df with a t_utc column derived from t_col via the clock model."""
    out = df.copy()
    out["t_utc"] = model.to_utc(out[t_col].to_numpy())
    return out
