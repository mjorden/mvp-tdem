"""
Positions → survey geometry: projection, elevations, line ids, fiducials.

- Easting/Northing : pyproj, WGS84 lat/lon → survey EPSG
- Elevation        : ellipsoidal height − geoid offset (survey config;
                     a fixed per-area offset is fine at MVP accuracy)
- DEM              : the altimeter height AGL itself. Repo convention:
                     despite the name, the DEM column is bird height —
                     qc._altitude_flag range-checks it directly, invert
                     uses it as bird_height_m, and ground elevation is
                     derived downstream as Elevation − DEM
- LINE             : operator line log (time ranges) when present;
                     heading-break auto-segmentation as fallback
- FID              : deciseconds since flight start — monotonic, unique,
                     maps back to time
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import Transformer


def project(soundings: pd.DataFrame, epsg: int) -> pd.DataFrame:
    out = soundings.copy()
    tf = Transformer.from_crs(4326, epsg, always_xy=True)
    easting, northing = tf.transform(out["lon"].to_numpy(), out["lat"].to_numpy())
    out["easting"], out["northing"] = easting, northing
    return out


def elevations(soundings: pd.DataFrame, geoid_offset_m: float) -> pd.DataFrame:
    out = soundings.copy()
    out["elevation"] = out["h_ell"] - geoid_offset_m
    out["dem"]       = out["agl_m"]
    return out


def assign_fids(soundings: pd.DataFrame) -> pd.DataFrame:
    out = soundings.copy()
    t0 = out["t_utc"].min()
    out["fid"] = np.round((out["t_utc"] - t0) * 10).astype(int)
    if out["fid"].duplicated().any():
        raise ValueError("Duplicate FIDs — soundings closer than 0.1 s apart?")
    return out


def assign_lines(
    soundings: pd.DataFrame,
    lines_df: pd.DataFrame | None = None,
    *,
    heading_win: int = 5,
    turn_threshold_deg: float = 45.0,
    min_line_soundings: int = 10,
) -> pd.DataFrame:
    """
    Attach a `line` column. Soundings outside any line (in turns, before/
    after the survey block) get line = -1 and are dropped at emit time.
    """
    out = soundings.copy()
    if lines_df is not None and len(lines_df):
        line = np.full(len(out), -1, dtype=int)
        t = out["t_utc"].to_numpy()
        for row in lines_df.itertuples():
            line[(t >= row.t_start_utc) & (t <= row.t_end_utc)] = int(row.line)
        out["line"] = line
        return out
    return _lines_from_heading(out, heading_win, turn_threshold_deg, min_line_soundings)


def _lines_from_heading(out, win, turn_threshold_deg, min_line_soundings):
    """
    Fallback segmentation: a sounding is 'on line' when the local heading
    (smoothed over ±win soundings) stays close to the segment's running
    heading. Runs shorter than min_line_soundings are treated as turns.
    Lines are numbered 1000, 1010, 1020, ... in flight order.
    """
    dx = np.gradient(out["easting"].to_numpy())
    dy = np.gradient(out["northing"].to_numpy())
    heading = np.degrees(np.arctan2(dx, dy))  # deg from north, ±180

    # unwrap so smoothing/differencing doesn't jump at ±180
    heading = np.degrees(np.unwrap(np.radians(heading)))
    kernel  = np.ones(2 * win + 1) / (2 * win + 1)
    smooth  = np.convolve(heading, kernel, mode="same")
    turning = np.abs(np.gradient(smooth)) > turn_threshold_deg / (2 * win + 1)

    line = np.full(len(out), -1, dtype=int)
    line_id, start = 1000, None
    for i, is_turn in enumerate(list(turning) + [True]):  # sentinel closes last run
        if not is_turn and start is None:
            start = i
        elif is_turn and start is not None:
            if i - start >= min_line_soundings:
                line[start:i] = line_id
                line_id += 10
            start = None

    out["line"] = line
    n_off = int((line == -1).sum())
    if n_off:
        print(f"[geometry] {n_off} soundings outside detected lines (turns) — "
              "will be dropped at emit")
    return out
