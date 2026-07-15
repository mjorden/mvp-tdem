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


def despike_stream(
    v: np.ndarray,
    half_window: int = 4,
    k_sigma: float = 6.0,
    min_dev: float = 3.0,
) -> np.ndarray:
    """
    Median-replace isolated spikes in a 1-D stream (#71.2).

    Radar altimeters spike over canopy/water lock-ons; `merge_nav` interpolates
    the RAW stream and the result feeds inversion geometry (`bird_height_m`),
    so a single spike directly corrupts that sounding's modeled height. A
    sample deviating from the rolling median by more than k_sigma robust sigmas
    AND more than min_dev (metres, absolute) is replaced BY the rolling median —
    narrow spikes are healed; real terrain-following trends move the median
    with them and stay untouched.

    The absolute min_dev term matters on quiet data: with near-zero noise the
    MAD collapses and a pure σ test would 'despike' sub-metre wobble; a radar
    lock-on artefact is tens to hundreds of metres, so 3 m is conservative.
    Edge windows are edge-padded so the rolling median is unbiased at stream
    boundaries.
    """
    v = np.asarray(v, dtype=float)
    n = len(v)
    if n < 2 * half_window + 1:
        return v.copy()
    padded = np.pad(v, half_window, mode="edge")
    med = np.empty(n)
    for i in range(n):
        med[i] = np.nanmedian(padded[i:i + 2 * half_window + 1])
    resid = v - med
    mad = np.nanmedian(np.abs(resid))
    sigma = max(1.4826 * mad, 1e-6)
    spikes = (np.abs(resid) > k_sigma * sigma) & (np.abs(resid) > min_dev)
    out = v.copy()
    out[spikes] = med[spikes]
    if spikes.any():
        print(f"[merge] despiked {int(spikes.sum())} altimeter sample(s) "
              f"(> {k_sigma}σ and > {min_dev} m from rolling median)")
    return out


def merge_nav(
    soundings: pd.DataFrame,
    gps_df: pd.DataFrame,
    alt_df: pd.DataFrame,
    max_gap_s: float = 2.0,
    despike_alt: bool = True,
) -> pd.DataFrame:
    """Attach lat, lon, h_ell (from GPS) and agl_m (from altimeter) to soundings.

    The altimeter stream is median-despike'd before interpolation (#71.2)
    unless despike_alt=False.
    """
    if "t_utc" not in alt_df.columns:
        raise ValueError("Altimeter frame has no t_utc — run timesync.apply_clock first")

    out = soundings.copy()
    tq  = out["t_utc"].to_numpy()
    tg  = gps_df["t_utc"].to_numpy()
    ta  = alt_df["t_utc"].to_numpy()

    agl = alt_df["agl_m"].to_numpy()
    if despike_alt:
        agl = despike_stream(agl)

    for col in ("lat", "lon", "h_ell"):
        out[col] = interp_with_gaps(tq, tg, gps_df[col].to_numpy(), max_gap_s)
    out["agl_m"] = interp_with_gaps(tq, ta, agl, max_gap_s)
    return out
