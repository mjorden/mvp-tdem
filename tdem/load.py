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
             plus one column per gate: sfz_00, sfz_01, ... — and, when the
             sidecar maps sfz_std_prefix, one sfz_std_NN per gate (#33)
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
    """
    Return all soundings on a single flight line.

    Ids are compared as strings on both sides (#19), so '--line 1000' matches
    whether the frame carries int, string, or mixed line ids.
    """
    mask = df["line"].astype(str) == str(line_id)
    if not mask.any():
        avail = sorted(df["line"].astype(str).unique())
        raise ValueError(f"Line {line_id!r} not found. Available: {avail}")
    return df[mask].reset_index(drop=True)


_GATE_COL_RE = re.compile(r"^sfz_(\d+)$")


def gate_index(col: str) -> int:
    """Parse the gate index out of an sfz_* column name: 'sfz_07' → 7 (#41)."""
    m = _GATE_COL_RE.match(col)
    if not m:
        raise ValueError(f"Not a gate column: {col!r}")
    return int(m.group(1))


def gate_columns(df: pd.DataFrame) -> list[str]:
    """
    Return sfz_* column names ordered numerically by gate index.

    Numeric, not lexicographic (#41): past 99 gates a string sort yields
    sfz_09, sfz_10, sfz_100, sfz_101, ..., sfz_11 — silently pairing wrong
    gate values with gate_times_ms.
    """
    return sorted((c for c in df.columns if _GATE_COL_RE.match(c)), key=gate_index)


_GATE_STD_COL_RE = re.compile(r"^sfz_std_(\d+)$")


def gate_std_columns(df: pd.DataFrame) -> list[str]:
    """
    Return sfz_std_* columns (measured per-gate std, #33) in numeric gate
    order. Empty when the deliverable carries no std columns.
    """
    return sorted((c for c in df.columns if _GATE_STD_COL_RE.match(c)),
                  key=lambda c: int(_GATE_STD_COL_RE.match(c).group(1)))


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


_GEOSOFT_LINE_RE = re.compile(r"^(Line|Tie)\s+(\S+)\s*$", re.IGNORECASE)


def _read_csv(path: Path) -> pd.DataFrame:
    """
    Read an ASCII deliverable — flat CSV or Geosoft XYZ (#2).

    Geosoft XYZ specifics handled here:
    - the column-name header is itself a '/'-prefixed comment line — the last
      comment line before the first data row is used as the header
    - 'Line 1000' / 'Tie 2005' records separate blocks and are the ONLY place
      the line id lives; they are captured and forward-filled into a
      __geosoft_line column
    - '*' dummies survive as NaN via per-column to_numeric coercion later

    A file is treated as Geosoft when it contains Line/Tie records or starts
    with a '/' comment; otherwise it goes straight to pandas as a flat table.
    """
    lines = path.read_text().splitlines()

    is_geosoft = any(_GEOSOFT_LINE_RE.match(l.strip()) for l in lines[:200]) \
        or (lines and lines[0].lstrip().startswith("/"))

    from io import StringIO
    if not is_geosoft:
        # Sniff the delimiter (#30): a regex separator cannot represent empty
        # fields — sep=r"[\s,]+" silently shifts every value after a blank
        # cell one column left, landing gate amplitudes in neighbouring gates.
        # Comma files get a literal comma so empty fields parse as NaN in
        # place; the whitespace regex is reserved for space-delimited files,
        # where an empty field is unrepresentable to begin with.
        header_line = next((l for l in lines if l.strip()), "")
        sep = "," if "," in header_line else r"\s+"
        return pd.read_csv(StringIO("\n".join(lines)), sep=sep,
                           engine="python", skipinitialspace=True)

    header: list[str] | None = None
    current_line: str | None = None
    rows: list[list[str]] = []
    line_ids: list[str | None] = []
    n_bad = 0

    for raw in lines:
        s = raw.strip()
        if not s:
            continue
        if s.startswith("/"):
            tokens = s.lstrip("/").split()
            if tokens and not rows:
                header = tokens          # last comment before data wins
            continue
        m = _GEOSOFT_LINE_RE.match(s)
        if m:
            current_line = m.group(2)
            continue
        tokens = re.split(r"[\s,]+", s)
        if header and len(tokens) != len(header):
            n_bad += 1                   # short/garbled record — skip, count
            continue
        rows.append(tokens)
        line_ids.append(current_line)

    if header is None:
        raise ValueError(
            f"{path.name}: Geosoft-style file but no '/'-comment header line found "
            "before the data. Cannot determine column names."
        )
    if not rows:
        raise ValueError(f"{path.name}: no data rows parsed.")
    if n_bad:
        print(f"[load] {path.name}: skipped {n_bad} malformed records "
              f"(token count != {len(header)} columns)")

    df = pd.DataFrame(rows, columns=header)
    if any(lid is not None for lid in line_ids):
        df["__geosoft_line"] = line_ids
    return df


def _apply_column_map(raw: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """
    Rename raw columns to standardised names and extract gate columns.

    Supports three gate column naming conventions operators use:
      bracket     : SFz[0] .. SFz[N-1]
      underscore  : SFz_00 .. SFz_N-1
      zero_padded : SFz00  .. SFzN-1

    Emits exactly the documented schema (#36): the standardised id columns
    plus one sfz_NN column per gate. Raw gate columns and unmapped raw
    channels (e.g. ingest-emitted SFz_std[i]) are dropped — one variable,
    one column.
    """
    rename = {}
    for std, key in _RENAME_COLS.items():
        raw_col = col_map.get(key)
        if raw_col and raw_col in raw.columns:
            rename[raw_col] = std

    df = raw.rename(columns=rename)

    # Geosoft XYZ: line ids come from 'Line NNNN' separator records (#2),
    # which take precedence over any mapped line column
    if "__geosoft_line" in df.columns:
        df["line"] = df.pop("__geosoft_line")

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

    extracted = []
    for i, raw_col in enumerate(gate_raw):
        name = f"sfz_{i:02d}"
        df[name] = pd.to_numeric(raw[raw_col], errors="coerce")
        extracted.append(name)

    # measured per-gate std, when the deliverable carries it (#33)
    std_prefix = col_map.get("sfz_std_prefix")
    if std_prefix:
        std_raw = _gate_column_names(std_prefix, n, fmt)
        std_missing = [c for c in std_raw if c not in raw.columns]
        if std_missing:
            raise ValueError(
                f"sfz_std_prefix is configured but std columns are missing from the CSV: "
                f"{std_missing[:5]}{'...' if len(std_missing) > 5 else ''}"
            )
        for i, raw_col in enumerate(std_raw):
            name = f"sfz_std_{i:02d}"
            df[name] = pd.to_numeric(raw[raw_col], errors="coerce")
            extracted.append(name)

    # documented schema only (#36): standardised ids + one column per gate
    keep = [std for std in _RENAME_COLS if std in df.columns] + extracted
    return df[keep]


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

    # Gate times must fit inside the waveform off-time (#1): a bipolar square
    # wave at base frequency f has a half-period of 1/(2f); subtracting the
    # on-time leaves the measurable off-time window.
    sysc = config.get("system", {})
    f    = sysc.get("tx_frequency_hz")
    if f and gate_times:
        on_time_ms  = sysc.get("tx_on_time_us", 0) / 1000.0
        off_time_ms = 1000.0 / (2.0 * f) - on_time_ms
        if max(gate_times) >= off_time_ms:
            raise ValueError(
                f"Latest gate ({max(gate_times)} ms) is outside the off-time window "
                f"({off_time_ms:.3f} ms for tx_frequency_hz={f}, "
                f"on_time={on_time_ms} ms). These gates are physically impossible — "
                "fix gate_times_ms or tx_frequency_hz in the sidecar."
            )

    return df


# Common ASCII dummy sentinels. Matched with tolerance (#16); anything with
# |x| >= 1e30 (Geosoft double-dummy ±1e32 etc.) is also treated as a dummy.
_DUMMY_SENTINELS = np.array([-9999.0, -99999.0, -999999.0, 999999.0, 9999999.0])
_DUMMY_HUGE = 1e30


# Coordinate/geometry channels: sentinel replacement must be exact here (#29)
_COORD_COLS = ("easting", "northing", "elevation", "dem", "latitude", "longitude")


def _replace_dummies(series: pd.Series, *, tolerant: bool = True) -> pd.Series:
    """
    NaN out dummy sentinels and |x| >= 1e30 values.

    tolerant=True matches sentinels within rtol=1e-6 — catches
    -9999.0000001-style float dirt on data channels, where sentinels
    (>= 1e3 in magnitude) are far from physics-scale values (~1e-12..1e3).
    Coordinate channels must use tolerant=False (#29): the same tolerance is
    a ±1 m window around northing 999,999 and ±10 m around 9,999,999 — real
    positions there would be silently NaN'd. Exporters write coordinate
    sentinels exactly, so exact equality is the right test.
    """
    vals = series.to_numpy(dtype=float)
    bad = np.abs(vals) >= _DUMMY_HUGE
    for s in _DUMMY_SENTINELS:
        if tolerant:
            bad |= np.isclose(vals, s, rtol=1e-6, atol=0.0)
        else:
            bad |= vals == s
    return series.where(~bad)


def _clean(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Coerce numerics, replace dummy fill values, drop dead soundings.

    Order matters (#8): coercion FIRST (Geosoft '*' → NaN via to_numeric),
    then dummy replacement on all numeric channels (#16), then filters —
    every dropped row is reported.
    """
    gate_cols = gate_columns(df)

    # 1. coerce every non-id column to numeric ('*' and junk → NaN)
    numeric_cols = [c for c in df.columns if c not in ("line", "fiducial")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2. dummy sentinels → NaN on ALL numeric channels, not just gates;
    #    coordinates use exact matching so real positions near a sentinel
    #    value survive (#29)
    for col in numeric_cols:
        df[col] = _replace_dummies(df[col], tolerant=col not in _COORD_COLS)

    # 3. drop rows with unusable altimetry (NaN or non-positive radar height);
    #    reported, since dem is required for inversion geometry
    bad_dem = ~(df["dem"] > 0)          # NaN > 0 is False → included here
    if bad_dem.any():
        print(f"[load] Dropped {int(bad_dem.sum())} soundings with missing/"
              f"non-positive dem (radar altimeter).")
    df = df[~bad_dem].copy()

    # 4. drop dead soundings (every gate NaN)
    all_nan = df[gate_cols].isna().all(axis=1)
    if all_nan.any():
        print(f"[load] Dropped {int(all_nan.sum())} soundings with all-NaN gates.")
    df = df[~all_nan].reset_index(drop=True)

    return df
