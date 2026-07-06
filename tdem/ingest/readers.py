"""
Per-stream parsers for raw flight logs. The only format-specific layer —
a firmware or hardware change means a new/updated reader, nothing else.

v0 reference log formats (ASCII, '#' comment lines, comma-separated)
--------------------------------------------------------------------
em_*.log     seq, t_rx, polarity, g00..gNN
             t_rx = receiver-clock seconds; polarity ±1; gate values in
             logged volts (pre-calibration, sign follows Tx polarity)
sync_*.log   t_rx, t_utc
             GPS time messages latched against the receiver clock (~1 Hz);
             t_utc = unix seconds
gps_*.log    t_utc, lat, lon, h_ell
             h_ell = ellipsoidal height (m)
alt_*.log    t_rx, agl_m
             altimeter shares the receiver clock (same DAQ)
txcur_*.log  t_rx, current_a          (optional)
lines_*.csv  line, t_start_utc, t_end_utc   (optional operator line log)

All times are numeric seconds. Readers are dumb: parse, coerce, count and
skip unparseable rows. No physics, no dropping of parseable data.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

_STREAM_GLOBS = {
    "em":    "em_*.log",
    "sync":  "sync_*.log",
    "gps":   "gps_*.log",
    "alt":   "alt_*.log",
    "txcur": "txcur_*.log",
    "lines": "lines_*.csv",
}

_REQUIRED_STREAMS = ("em", "sync", "gps", "alt")


def discover_flight(flight_dir: str | Path) -> dict[str, list[Path]]:
    """
    Map stream name → sorted list of matching files in a flight directory.

    Raises if a required stream (em, sync, gps, alt) has no files.
    """
    flight_dir = Path(flight_dir)
    if not flight_dir.is_dir():
        raise FileNotFoundError(f"Flight directory not found: {flight_dir}")

    found = {
        name: sorted(Path(p) for p in glob.glob(str(flight_dir / pattern)))
        for name, pattern in _STREAM_GLOBS.items()
    }
    missing = [s for s in _REQUIRED_STREAMS if not found[s]]
    if missing:
        raise FileNotFoundError(
            f"Flight {flight_dir.name}: no files for required stream(s) {missing}. "
            f"Expected globs: {[_STREAM_GLOBS[s] for s in missing]}"
        )
    return found


def _read_stream(paths: list[Path], required_cols: list[str], sort_by: str) -> pd.DataFrame:
    """Concatenate one stream's files, coerce numerics, skip bad rows with a count."""
    frames = []
    for path in paths:
        lines = [l for l in path.read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.strip().startswith("#")]
        from io import StringIO
        df = pd.read_csv(StringIO("\n".join(lines)), skipinitialspace=True,
                         on_bad_lines="skip")
        # on_bad_lines="skip" drops malformed rows SILENTLY; count them (#68.3).
        # Every dropped EM half-cycle mis-phases later stack windows (#56), so a
        # silent loss is not acceptable — the first data line is the header.
        n_expected = max(len(lines) - 1, 0)
        n_skipped = n_expected - len(df)
        if n_skipped > 0:
            print(f"[readers] {path.name}: skipped {n_skipped} malformed rows "
                  f"(wrong field count) of {n_expected}")
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name}: missing columns {missing}; has {list(df.columns)}")
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    bad = df[required_cols].isna().any(axis=1)
    if bad.any():
        print(f"[readers] {paths[0].name} (+{len(paths)-1}): skipped {bad.sum()} unparseable rows")
        df = df[~bad]

    return df.sort_values(sort_by).reset_index(drop=True)


def read_em(paths: list[Path]) -> pd.DataFrame:
    """EM stream: t_rx, polarity, gate columns g00..gNN (order preserved)."""
    df = _read_stream(paths, ["t_rx", "polarity"], sort_by="t_rx")
    gate_cols = [c for c in df.columns if c.startswith("g") and c[1:].isdigit()]
    if not gate_cols:
        raise ValueError("EM log has no gate columns (expected g00, g01, ...)")
    if not df["polarity"].isin([-1, 1]).all():
        raise ValueError("EM log polarity column must be +1 or -1")
    return df


def read_sync(paths: list[Path]) -> pd.DataFrame:
    return _read_stream(paths, ["t_rx", "t_utc"], sort_by="t_rx")


def read_gps(paths: list[Path]) -> pd.DataFrame:
    return _read_stream(paths, ["t_utc", "lat", "lon", "h_ell"], sort_by="t_utc")


def read_alt(paths: list[Path]) -> pd.DataFrame:
    return _read_stream(paths, ["t_rx", "agl_m"], sort_by="t_rx")


def read_txcur(paths: list[Path]) -> pd.DataFrame:
    return _read_stream(paths, ["t_rx", "current_a"], sort_by="t_rx")


def read_lines(paths: list[Path]) -> pd.DataFrame:
    df = _read_stream(paths, ["line", "t_start_utc", "t_end_utc"], sort_by="t_start_utc")
    if (df["t_end_utc"] <= df["t_start_utc"]).any():
        raise ValueError("lines log: t_end_utc must be > t_start_utc on every row")
    return df


def em_gate_columns(em_df: pd.DataFrame) -> list[str]:
    """Ordered raw gate column names (g00, g01, ...)."""
    return sorted(c for c in em_df.columns if c.startswith("g") and c[1:].isdigit())
