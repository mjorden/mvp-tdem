"""
Receiver-clock → GPS-time alignment.

GPS time is the master clock. The receiver's oscillator has unknown offset
and drift, so we fit a piecewise-linear model over the flight's sync pairs
(GPS time messages latched against the receiver clock) and apply the model
to every t_rx-stamped stream (EM, altimeter, Tx current).

Robustness notes (#64):
- MAD-based outlier rejection tolerates a small fraction of garbled serial
  GPS messages before fitting the model.
- A piecewise-linear (rather than global linear) model handles temperature-
  dependent oscillator drift over long flights without forcing an error on
  thermally-curved data.
- The 1 ms residual spec presumes PPS-quality clock latching.  NMEA-over-
  serial arrival jitter is 10–100 ms; only a hardware PPS latch achieves
  ≤1 ms.  Inspect hardware configuration if residuals exceed spec on real
  flights.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


class ClockModel:
    """Piecewise-linear receiver-clock → GPS-time mapping.

    Parameters
    ----------
    t_rx_knots    : receiver-clock times of the cleaned sync pairs (sorted)
    t_utc_knots   : corresponding GPS times
    max_residual_s: worst |residual from a global linear fit on clean pairs|
    n_pairs       : number of clean sync pairs used
    """

    __slots__ = ("t_rx_knots", "t_utc_knots", "max_residual_s", "n_pairs")

    def __init__(
        self,
        t_rx_knots: np.ndarray,
        t_utc_knots: np.ndarray,
        max_residual_s: float,
        n_pairs: int,
    ) -> None:
        self.t_rx_knots   = np.asarray(t_rx_knots, dtype=float)
        self.t_utc_knots  = np.asarray(t_utc_knots, dtype=float)
        self.max_residual_s = float(max_residual_s)
        self.n_pairs      = int(n_pairs)

    def to_utc(self, t_rx: np.ndarray) -> np.ndarray:
        """Apply the piecewise-linear clock model with linear edge extrapolation.

        np.interp clamps out-of-range values to the boundary knot, which
        makes the converted timestamps non-monotone when streams extend
        slightly beyond the sync range.  Linear extrapolation from the
        nearest two knots preserves strict monotonicity.
        """
        t = np.atleast_1d(np.asarray(t_rx, dtype=float))
        y = np.interp(t, self.t_rx_knots, self.t_utc_knots)

        if self.t_rx_knots.size >= 2:
            rate_lo = ((self.t_utc_knots[1] - self.t_utc_knots[0]) /
                       (self.t_rx_knots[1] - self.t_rx_knots[0]))
            lo = t < self.t_rx_knots[0]
            y[lo] = self.t_utc_knots[0] + rate_lo * (t[lo] - self.t_rx_knots[0])

            rate_hi = ((self.t_utc_knots[-1] - self.t_utc_knots[-2]) /
                       (self.t_rx_knots[-1] - self.t_rx_knots[-2]))
            hi = t > self.t_rx_knots[-1]
            y[hi] = self.t_utc_knots[-1] + rate_hi * (t[hi] - self.t_rx_knots[-1])

        return y


def fit_clock(
    sync_df: pd.DataFrame,
    max_residual_ms: float = 1.0,
    max_outlier_frac: float = 0.20,
) -> ClockModel:
    """Fit the piecewise-linear clock model from sync pairs.

    Procedure
    ---------
    1. Fit a global linear model to all pairs (centred for float64 precision).
    2. Reject pairs whose |residual| exceeds max(3·MAD, max_residual_ms/2 ms)
       as garbled messages.  Hard-error if more than ``max_outlier_frac`` of
       pairs are rejected — the sync stream itself is broken.
    3. Check the worst residual from a global linear fit on the *clean* pairs
       against ``max_residual_ms``; hard-error if exceeded (the device may have
       a hardware fault beyond expected thermal drift).
    4. Return a ``ClockModel`` whose ``to_utc()`` uses piecewise-linear
       interpolation through the clean pairs (handles thermal drift).

    Parameters
    ----------
    max_residual_ms  : residual spec for the *cleaned* global linear fit (ms).
    max_outlier_frac : fraction of pairs that may be rejected as outliers
                       before the stream is considered broken (0–1).
    """
    if len(sync_df) < 2:
        raise ValueError(
            f"Need >= 2 sync pairs to fit the clock model, got {len(sync_df)}"
        )

    t_rx  = sync_df["t_rx"].to_numpy(dtype=float)
    t_utc = sync_df["t_utc"].to_numpy(dtype=float)

    # Step 1: initial global linear fit (centred for float64 precision)
    t0, u0 = t_rx.mean(), t_utc.mean()
    rate0, c0 = np.polyfit(t_rx - t0, t_utc - u0, 1)
    offset0 = u0 + c0 - rate0 * t0
    resid0  = t_utc - (offset0 + rate0 * t_rx)

    # Step 2: MAD-based outlier rejection — tolerates garbled serial messages.
    # Compare each residual's deviation FROM THE MEDIAN (not from zero), so
    # the outlier's influence on the fit doesn't cause good pairs to fail.
    med0   = np.median(resid0)
    mad    = np.median(np.abs(resid0 - med0))
    thresh = max(3.0 * mad, max_residual_ms / 2000.0)  # never tighter than half-spec
    clean  = np.abs(resid0 - med0) <= thresh
    n_out  = int((~clean).sum())

    if n_out > max_outlier_frac * len(sync_df):
        raise ValueError(
            f"Too many outlier sync pairs ({n_out}/{len(sync_df)} = "
            f"{n_out/len(sync_df):.0%} > {max_outlier_frac:.0%}) — sync "
            "stream is inconsistent (hardware fault? wrong file?). "
            "Inspect sync_*.log before trusting this flight."
        )
    if clean.sum() < 2:
        raise ValueError(
            "Fewer than 2 clean sync pairs after outlier rejection — "
            "cannot fit clock model."
        )

    t_rx_c  = t_rx[clean]
    t_utc_c = t_utc[clean]

    # Step 3: residual check on clean pairs (global linear fit)
    t0c, u0c = t_rx_c.mean(), t_utc_c.mean()
    rate_c, cc = np.polyfit(t_rx_c - t0c, t_utc_c - u0c, 1)
    offset_c   = u0c + cc - rate_c * t0c
    resid_c    = t_utc_c - (offset_c + rate_c * t_rx_c)
    worst      = float(np.abs(resid_c).max())

    if worst > max_residual_ms / 1000.0:
        raise ValueError(
            f"Clock fit residual {worst*1000:.3f} ms exceeds {max_residual_ms} ms "
            f"over {int(clean.sum())} clean sync pairs "
            f"({n_out} outlier(s) rejected) — device may have a hardware fault "
            "beyond expected thermal drift. Inspect sync_*.log."
        )

    if n_out:
        warnings.warn(
            f"[timesync] Rejected {n_out} garbled sync pair(s) "
            f"({n_out/len(sync_df):.0%}) as outliers before fitting clock model.",
            stacklevel=2,
        )

    # Step 4: sort clean pairs by t_rx (guarantee monotone for np.interp)
    order   = np.argsort(t_rx_c)
    t_rx_c  = t_rx_c[order]
    t_utc_c = t_utc_c[order]

    return ClockModel(
        t_rx_knots=t_rx_c,
        t_utc_knots=t_utc_c,
        max_residual_s=worst,
        n_pairs=int(clean.sum()),
    )


def apply_clock(df: pd.DataFrame, model: ClockModel, t_col: str = "t_rx") -> pd.DataFrame:
    """Return df with a t_utc column derived from t_col via the clock model."""
    out = df.copy()
    out["t_utc"] = model.to_utc(out[t_col].to_numpy())
    return out
