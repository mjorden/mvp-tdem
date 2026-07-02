"""
Ingest helicopter TDEM field deliverables.

Expected inputs
---------------
- CSV  : flat ASCII table (Geosoft XYZ export or equivalent)
- JSON : sidecar metadata file (system params, gate times, column map)

EM response columns are in V/(A·m⁴) — voltage normalized by Tx moment.
Gate times live in the JSON sidecar, not in the CSV.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_survey(csv_path: str | Path, config_path: str | Path) -> tuple[pd.DataFrame, dict]:
    """
    Load one survey CSV + its JSON sidecar.

    Returns
    -------
    df     : tidy DataFrame with standardised column names (see _RENAME_COLS)
             plus one column per gate: sfz_00, sfz_01, ...
    config : parsed sidecar dict (system params, gate_times_ms, inversion defaults)
    """
    csv_path    = Path(csv_path)
    config_path = Path(config_path)

    with open(config_path) as f:
        config = json.load(f)

    raw = _read_csv(csv_path)
    df  = _apply_column_map(raw, config["column_map"])
    df  = _validate(df, config)
    df  = _clean(df, config)

    return df, config


def load_line(df: pd.DataFrame, line_id: int | str) -> pd.DataFrame:
    """Return all soundings on a single flight line."""
    mask = df["line"] == line_id
    if not mask.any():
        raise ValueError(f"Line {line_id!r} not found. Available: {sorted(df['line'].unique())}")
    return df[mask].reset_index(drop=True)


def gate_columns(df: pd.DataFrame) -> list[str]:
    """Return ordered list of sfz_* column names present in df."""
    return sorted(c for c in df.columns if re.match(r"^sfz_\d+$", c))


def gate_array(df: pd.DataFrame) -> np.ndarray:
    """Return (n_soundings, n_gates) array of EM response values [V/(A·m⁴)]."""
    return df[gate_columns(df)].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_RENAME_COLS = {
    "line":      "line",
    "fiducial":  "fiducial",
    "easting":   "easting",
    "northing":  "northing",
    "elevation": "elevation",
    "dem":       "dem",
    "latitude":  "latitude",
    "longitude": "longitude",
}


def _read_csv(path: Path) -> pd.DataFrame:
    """Read ASCII CSV, skipping Geosoft-style comment lines (start with '/')."""
    lines = path.read_text().splitlines()
    data_lines = [l for l in lines if not l.strip().startswith("/")]
    from io import StringIO
    return pd.read_csv(StringIO("\n".join(data_lines)), sep=r"[\s,]+", engine="python")


def _apply_column_map(raw: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """
    Rename raw columns to standardised names and extract gate columns.

    Supports three gate column naming conventions operators use:
      bracket     : SFz[0] .. SFz[N-1]
      underscore  : SFz_00 .. SFz_N-1
      zero_padded : SFz00  .. SFzN-1
    """
    rename = {}
    for std, key in _RENAME_COLS.items():
        raw_col = col_map.get(key)
        if raw_col and raw_col in raw.columns:
            rename[raw_col] = std

    df = raw.rename(columns=rename)

    prefix = col_map.get("sfz_prefix", "SFz")
    n      = col_map.get("sfz_n", 30)
    fmt    = col_map.get("sfz_format", "bracket")

    gate_raw = _gate_column_names(prefix, n, fmt)
    missing  = [c for c in gate_raw if c not in raw.columns]
    if missing:
        raise ValueError(
            f"Expected gate columns not found in CSV: {missing[:5]}{'...' if len(missing) > 5 else ''}\n"
            f"CSV columns: {list(raw.columns)}"
        )

    for i, raw_col in enumerate(gate_raw):
        df[f"sfz_{i:02d}"] = pd.to_numeric(raw[raw_col], errors="coerce")

    return df


def _gate_column_names(prefix: str, n: int, fmt: str) -> list[str]:
    if fmt == "bracket":
        return [f"{prefix}[{i}]" for i in range(n)]
    if fmt == "underscore":
        width = len(str(n - 1))
        return [f"{prefix}_{i:0{width}d}" for i in range(n)]
    if fmt == "zero_padded":
        width = len(str(n - 1))
        return [f"{prefix}{i:0{width}d}" for i in range(n)]
    raise ValueError(f"Unknown sfz_format: {fmt!r}. Use 'bracket', 'underscore', or 'zero_padded'.")


def _validate(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    required = ["easting", "northing", "elevation", "dem"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after column_map: {missing}")

    gate_times = config.get("gate_times_ms", [])
    n_gates    = config["column_map"].get("sfz_n", 30)
    if len(gate_times) != n_gates:
        raise ValueError(
            f"gate_times_ms has {len(gate_times)} entries but sfz_n = {n_gates}. "
            "These must match."
        )

    return df


def _clean(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Replace dummy fill values, drop dead soundings, coerce coordinate types."""
    DUMMY_VALUES = {-9999.0, -99999.0, 999999.0, 9999999.0, 1e31}

    gate_cols = gate_columns(df)

    for col in gate_cols:
        df[col] = df[col].where(~df[col].isin(DUMMY_VALUES))

    df = df[df["dem"] > 0].copy()

    all_nan = df[gate_cols].isna().all(axis=1)
    n_dropped = all_nan.sum()
    if n_dropped > 0:
        print(f"[load] Dropped {n_dropped} soundings with all-NaN gates.")
    df = df[~all_nan].reset_index(drop=True)

    for col in ["easting", "northing", "elevation", "dem"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
