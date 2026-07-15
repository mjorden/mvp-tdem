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


def apply_layback(
    soundings: pd.DataFrame,
    layback_m: float = 0.0,
    drop_m: float = 0.0,
    epsg: int | None = None,
    win: int = 5,
) -> pd.DataFrame:
    """
    Static lever-arm / layback correction (#70): GPS antenna → bird position.

    The GPS antenna rides on the aircraft; the EM bird trails `layback_m`
    BEHIND it along the flight track and hangs `drop_m` BELOW it. Without this
    correction every sounding is geolocated at the antenna: the along-track
    error flips sign with heading (adjacent lines flown in opposite directions
    mis-tie by 2×layback ≈ 40–60 m) and the vertical offset contaminates the
    derived ground elevation — the radar AGL is measured at the BIRD, so
    `elevation − dem` mixed an antenna height with a bird height.

    The track direction is the smoothed (±win soundings) gradient of the
    projected positions, computed per contiguous valid block (nav gaps break
    the smoothing, same policy as the heading segmentation). Hover / zero-speed
    stretches inherit the nearest moving direction. When `epsg` is given,
    lat/lon are re-derived from the corrected easting/northing so the emitted
    deliverable stays internally consistent (the #95 harness cross-checks them).

    This is the STATIC correction: constant cable geometry per survey. Dynamic
    cable swing / bird attitude needs an attitude stream (drop-in via the
    ingest STREAMS registry) and remains future work.
    """
    if layback_m == 0.0 and drop_m == 0.0:
        return soundings
    out = soundings.copy()

    if drop_m:
        # h_ell is the ANTENNA ellipsoid height; the bird flies drop_m lower.
        # elevations() downstream then derives the BIRD elevation, consistent
        # with the bird-measured radar AGL.
        out["h_ell"] = out["h_ell"] - drop_m

    if layback_m:
        east  = out["easting"].to_numpy(dtype=float)
        north = out["northing"].to_numpy(dtype=float)
        ux, uy = _track_direction(east, north, win)
        out["easting"]  = east  - layback_m * ux
        out["northing"] = north - layback_m * uy
        if epsg is not None and {"lat", "lon"} <= set(out.columns):
            inv = Transformer.from_crs(epsg, 4326, always_xy=True)
            lon, lat = inv.transform(out["easting"].to_numpy(),
                                     out["northing"].to_numpy())
            out["lon"], out["lat"] = lon, lat
    return out


def _track_direction(east: np.ndarray, north: np.ndarray, win: int) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed unit track-direction per sounding; NaN-gap-safe, hover-filled."""
    n = len(east)
    ux = np.full(n, np.nan)
    uy = np.full(n, np.nan)

    valid = np.isfinite(east) & np.isfinite(north)
    padded_v = np.concatenate([[False], valid, [False]])
    dv = np.diff(padded_v.astype(int))
    kernel = np.ones(2 * win + 1) / (2 * win + 1)

    for s, e in zip(np.where(dv == 1)[0], np.where(dv == -1)[0]):
        if e - s < 2:
            continue
        dx = np.gradient(east[s:e])
        dy = np.gradient(north[s:e])
        if e - s > 2 * win + 1:      # smooth only when the block is long enough
            dx = np.convolve(np.pad(dx, win, mode="edge"), kernel, mode="valid")
            dy = np.convolve(np.pad(dy, win, mode="edge"), kernel, mode="valid")
        norm = np.hypot(dx, dy)
        moving = norm > 1e-6         # hover: direction undefined → fill below
        ux[s:e] = np.where(moving, dx / np.where(moving, norm, 1.0), np.nan)
        uy[s:e] = np.where(moving, dy / np.where(moving, norm, 1.0), np.nan)

    # hover stretches inherit the nearest moving direction
    ux = pd.Series(ux).ffill().bfill().to_numpy()
    uy = pd.Series(uy).ffill().bfill().to_numpy()
    return np.nan_to_num(ux), np.nan_to_num(uy)


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

    NaN positions (nav gaps) are treated as mandatory turn boundaries so that
    np.unwrap's cumulative propagation never poisons headings downstream of a
    gap.  Each contiguous block of valid (non-NaN) positions is processed
    independently.  Edge-replicated padding is used instead of zero-padding so
    the first/last `win` soundings of each block don't get spurious turn flags.
    """
    east  = out["easting"].to_numpy()
    north = out["northing"].to_numpy()
    n     = len(east)

    valid = np.isfinite(east) & np.isfinite(north)
    # NaN positions are always turns; valid ones are filled in per block below
    turning = ~valid.copy()

    # find contiguous valid blocks — process each independently
    padded_v = np.concatenate([[False], valid, [False]])
    dv = np.diff(padded_v.astype(int))
    block_starts = np.where(dv == 1)[0]
    block_ends   = np.where(dv == -1)[0]

    thresh = turn_threshold_deg / (2 * win + 1)
    kernel = np.ones(2 * win + 1) / (2 * win + 1)

    for s, e in zip(block_starts, block_ends):
        if e - s < 2:
            continue
        dx = np.gradient(east[s:e])
        dy = np.gradient(north[s:e])
        h  = np.degrees(np.unwrap(np.radians(np.degrees(np.arctan2(dx, dy)))))
        # edge-replicated padding avoids zero-pad artifacts at block boundaries
        smooth   = np.convolve(np.pad(h, win, mode="edge"), kernel, mode="valid")
        seg_turn = np.abs(np.gradient(smooth)) > thresh
        # gap boundaries are always turns — don't let a block merge across a gap
        seg_turn[0] = seg_turn[-1] = True
        turning[s:e] = seg_turn

    line = np.full(n, -1, dtype=int)
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
