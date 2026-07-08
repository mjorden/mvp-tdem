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


def _gate_index(col: str) -> int:
    """Parse the trailing integer gate index from an sfz_* / sfz_std_* column."""
    return int(col.rsplit("_", 1)[-1])


def gate_columns(df: pd.DataFrame) -> list[str]:
    """
    Return sfz_* column names present in df, ordered by NUMERIC gate index (#41).

    A plain `sorted()` is lexicographic: with ≥100 gates it would order
    sfz_09, sfz_10, sfz_100, sfz_101, …, sfz_11, silently pairing gate values
    with the wrong `gate_times_ms` entries and desyncing per-gate QC flags.
    Real systems deliver 20–60 gates so this is hardening, but the numeric sort
    makes the ordering correct for any gate count.
    """
    cols = [c for c in df.columns if re.match(r"^sfz_\d+$", c)]
    return sorted(cols, key=_gate_index)


def gate_std_columns(df: pd.DataFrame) -> list[str]:
    """Return sfz_std_* column names present in df, ordered by numeric index (#41)."""
    cols = [c for c in df.columns if re.match(r"^sfz_std_\d+$", c)]
    return sorted(cols, key=_gate_index)


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
    # utf-8-sig strips a leading BOM (#A16) — otherwise "﻿LINE" becomes the
    # first header token and the column_map lookup for the first column fails
    lines = path.read_text(encoding="utf-8-sig").splitlines()

    is_geosoft = any(_GEOSOFT_LINE_RE.match(l.strip()) for l in lines[:200]) \
        or (lines and lines[0].lstrip().startswith("/"))

    from io import StringIO
    if not is_geosoft:
        return pd.read_csv(StringIO("\n".join(lines)), sep=",")

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

    # sfz_n is REQUIRED (#43.2): a silent default of 30 defaulted BOTH gate
    # extraction and the gate_times length cross-check in tandem, so an omitted
    # sfz_n surfaced as a confusing "gate columns not found" instead of naming
    # the real problem. prefix/format keep sensible defaults.
    if "sfz_n" not in col_map:
        raise ValueError(
            "column_map.sfz_n is required (number of gate columns) and must equal "
            "len(gate_times_ms). Add it to the sidecar's column_map."
        )
    prefix = col_map.get("sfz_prefix", "SFz")
    n      = col_map["sfz_n"]
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

    # drop original gate columns now duplicated by the sfz_* ones
    df = df.drop(columns=[c for c in gate_raw if c in df.columns])

    # optionally load per-gate standard-deviation columns if present, in the
    # SAME naming convention as the gates (#A15): a third-party underscore/zero-
    # padded deliverable names its std columns to match, so hardcoding "bracket"
    # silently ignored their measured uncertainty. Our own emit uses bracket
    # gates too, so fmt=bracket there — this stays consistent.
    std_raw = _gate_column_names(f"{prefix}_std", n, fmt)
    if all(c in raw.columns for c in std_raw):
        for i, raw_col in enumerate(std_raw):
            df[f"sfz_std_{i:02d}"] = pd.to_numeric(raw[raw_col], errors="coerce")
        df = df.drop(columns=[c for c in std_raw if c in df.columns])

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
    n_gates    = config["column_map"]["sfz_n"]   # required; validated in _apply_column_map
    if len(gate_times) != n_gates:
        raise ValueError(
            f"gate_times_ms has {len(gate_times)} entries but sfz_n = {n_gates}. "
            "These must match."
        )

    # gate times must be positive and strictly increasing (#68.5): a mis-ordered
    # or non-positive table silently mispairs every gate with the wrong time
    gt = np.asarray(gate_times, dtype=float)
    if gt.size and (gt[0] <= 0 or np.any(np.diff(gt) <= 0)):
        raise ValueError(
            "gate_times_ms must be positive and strictly increasing (ms after "
            f"turnoff); got {gate_times}."
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
        # #63: the sidecar only carries gate CENTRES; a centre in the last 10%
        # of the off-time means any realistic gate width integrates into the
        # next turn-on ramp. Warn — the hard check above uses centres only.
        if max(gate_times) > 0.9 * off_time_ms:
            print(
                f"[load] WARNING: latest gate centre ({max(gate_times)} ms) is within "
                f"10% of the off-time end ({off_time_ms:.3f} ms) — a finite gate "
                "window there overlaps the next turn-on ramp."
            )

    return df


# Common ASCII dummy sentinels. Matched with tolerance (#16); anything with
# |x| >= 1e30 (Geosoft double-dummy ±1e32 etc.) is also treated as a dummy.
_DUMMY_SENTINELS = np.array([-9999.0, -99999.0, -999999.0, 999999.0, 9999999.0])
_DUMMY_HUGE = 1e30


def _replace_dummies(series: pd.Series) -> pd.Series:
    """NaN out dummy sentinels (tolerant match) and |x| >= 1e30 values."""
    vals = series.to_numpy(dtype=float)
    bad = np.abs(vals) >= _DUMMY_HUGE
    for s in _DUMMY_SENTINELS:
        # abs tolerance of 0.5 catches float-dirt like -9999.0000001 without
        # matching real coordinates near sentinel magnitudes (e.g. northing 999,999 m)
        bad |= np.isclose(vals, s, rtol=0.0, atol=0.5)
    return series.where(~bad)


def _clean(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Coerce numerics, replace dummy fill values, drop dead soundings.

    Order matters (#8): coercion FIRST (Geosoft '*' → NaN via to_numeric),
    then dummy replacement on all numeric channels (#16), then filters —
    every dropped row is reported.
    """
    gate_cols = gate_columns(df)

    # 0. normalize the line-id dtype (#68.4): Geosoft parsing yields STRING line
    #    ids, a flat CSV yields ints, so load_line(df, 2000) failed on Geosoft
    #    files with a confusing "not found". If every id is integer-valued, store
    #    as int; otherwise leave the (string) ids untouched.
    if "line" in df.columns:
        line_num = pd.to_numeric(df["line"], errors="coerce")
        if line_num.notna().all() and np.all(line_num == np.round(line_num)):
            df["line"] = line_num.astype("int64")

    # 1. coerce every non-id column to numeric ('*' and junk → NaN)
    numeric_cols = [c for c in df.columns if c not in ("line", "fiducial")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2. dummy sentinels → NaN on ALL numeric channels, not just gates
    for col in numeric_cols:
        df[col] = _replace_dummies(df[col])

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

    # 5. units plausibility guard (#A1): the pipeline assumes moment-normalized
    #    dB/dt in V/(A·m⁴), typically ~1e-6 (early) to ~1e-13 (late). A deliverable
    #    in nT/s, ppm, µV, or un-normalized volts would load silently and invert to
    #    garbage. A very loose amplitude window (13 decades) only trips on gross
    #    unit mismatches, never on real data — warn, don't drop.
    if gate_cols and len(df):
        gvals = np.abs(df[gate_cols].to_numpy(dtype=float))
        finite = gvals[np.isfinite(gvals) & (gvals > 0)]
        if finite.size:
            med = float(np.median(finite))
            if not (1e-16 <= med <= 1e-3):
                import warnings
                warnings.warn(
                    f"[load] median gate amplitude {med:.2e} is outside the physical "
                    "range for moment-normalized dB/dt (~1e-13..1e-6 V/(A·m⁴)). The "
                    "data may be in different units (nT/s, ppm, µV, un-normalized) — "
                    "the pipeline assumes V/(A·m⁴) and will invert to wrong "
                    "resistivities. Verify units before trusting results.",
                    stacklevel=2,
                )

    return df
