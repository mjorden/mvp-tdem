"""
Interpolate nav/altimeter streams onto sounding centre times.

Linear interpolation, but never across a source gap wider than max_gap_s
and never beyond the source's time range — those queries return NaN.
Soundings with NaN position are dropped at emit time (with a count);
inventing positions by extrapolation is worse than losing the sounding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def interp_with_gaps(
    t_query: np.ndarray,
    t_src: np.ndarray,
    v_src: np.ndarray,
    max_gap_s: float,
) -> np.ndarray:
    """np.interp with NaN outside the source range or inside gaps > max_gap_s."""
    t_query = np.asarray(t_query, dtype=float)
    t_src   = np.asarray(t_src, dtype=float)
    v_src   = np.asarray(v_src, dtype=float)
    if len(t_src) < 2:
        raise ValueError(f"Need >= 2 source samples to interpolate, got {len(t_src)}")
    if np.any(np.diff(t_src) <= 0):
        raise ValueError("Source timestamps must be strictly increasing")

    out = np.interp(t_query, t_src, v_src)

    inside = (t_query >= t_src[0]) & (t_query <= t_src[-1])
    # width of the bracketing source interval for each query
    hi  = np.clip(np.searchsorted(t_src, t_query, side="right"), 1, len(t_src) - 1)
    gap = t_src[hi] - t_src[hi - 1]
    out[~inside | (gap > max_gap_s)] = np.nan
    return out


def merge_nav(
    soundings: pd.DataFrame,
    gps_df: pd.DataFrame,
    alt_df: pd.DataFrame,
    max_gap_s: float = 2.0,
) -> pd.DataFrame:
    """Attach lat, lon, h_ell (from GPS) and agl_m (from altimeter) to soundings."""
    if "t_utc" not in alt_df.columns:
        raise ValueError("Altimeter frame has no t_utc — run timesync.apply_clock first")

    out = soundings.copy()
    tq  = out["t_utc"].to_numpy()
    tg  = gps_df["t_utc"].to_numpy()
    ta  = alt_df["t_utc"].to_numpy()

    for col in ("lat", "lon", "h_ell"):
        out[col] = interp_with_gaps(tq, tg, gps_df[col].to_numpy(), max_gap_s)
    out["agl_m"] = interp_with_gaps(tq, ta, alt_df["agl_m"].to_numpy(), max_gap_s)
    return out
