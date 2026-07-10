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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Stream registry (#87)
# ---------------------------------------------------------------------------
# Adding a new instrument stream (attitude, laser altimeter, binary EM, …) is:
# write a read_<stream>() below and register it here — discover_flight and the
# pipeline's generic read-and-clock loop pick it up; no orchestration edits.
# `needs_clock` marks streams stamped on the receiver clock (t_rx) that must be
# converted to t_utc via the fitted clock model. `required` aborts discovery
# when absent. NOTE: the pipeline's *stage wiring* (stack needs "em", merge
# needs "gps"/"alt") still refers to the core streams by name — the registry
# makes ingestion of a new stream drop-in, not the physics that consumes it.

@dataclass(frozen=True)
class StreamSpec:
    glob: str
    reader: "Callable[[list[Path]], pd.DataFrame]"
    required: bool = False
    needs_clock: bool = False


def _registry() -> dict[str, StreamSpec]:
    # built lazily so the spec can reference reader functions defined below
    return {
        "em":    StreamSpec("em_*.log",    read_em,    required=True,  needs_clock=True),
        "sync":  StreamSpec("sync_*.log",  read_sync,  required=True,  needs_clock=False),
        "gps":   StreamSpec("gps_*.log",   read_gps,   required=True,  needs_clock=False),
        "alt":   StreamSpec("alt_*.log",   read_alt,   required=True,  needs_clock=True),
        "txcur": StreamSpec("txcur_*.log", read_txcur, required=False, needs_clock=True),
        "lines": StreamSpec("lines_*.csv", read_lines, required=False, needs_clock=False),
    }


STREAMS: dict[str, StreamSpec] = {}     # populated at module bottom


def discover_flight(flight_dir: str | Path) -> dict[str, list[Path]]:
    """
    Map stream name → sorted list of matching files in a flight directory.

    Driven by the STREAMS registry; raises if a required stream has no files.
    """
    flight_dir = Path(flight_dir)
    if not flight_dir.is_dir():
        raise FileNotFoundError(f"Flight directory not found: {flight_dir}")

    found = {
        name: sorted(Path(p) for p in glob.glob(str(flight_dir / spec.glob)))
        for name, spec in STREAMS.items()
    }
    missing = [s for s, spec in STREAMS.items() if spec.required and not found[s]]
    if missing:
        raise FileNotFoundError(
            f"Flight {flight_dir.name}: no files for required stream(s) {missing}. "
            f"Expected globs: {[STREAMS[s].glob for s in missing]}"
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


# populate the registry now that the reader functions exist
STREAMS.update(_registry())
