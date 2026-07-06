"""
Write the canonical deliverable pair consumed by tdem.load.load_survey():

- survey.csv    : LINE, FID, Easting, Northing, Elevation, DEM, Latitude,
                  Longitude, SFz[0]..SFz[n-1], SFz_std[0]..
- sidecar.json  : generated from instrument.yaml + the survey YAML — never
                  hand-written — plus an `ingest` provenance block.

This is the one place ingest drops data, and it says so: soundings with
NaN position (nav gaps) or line == -1 (turns) are counted and removed.
Remaining NaN gate values are written as the -9999 dummy that
load._clean() already recognizes (empty CSV fields would break its
whitespace-or-comma separator).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .stack import stacked_gate_columns

_DUMMY = -9999.0


def emit_survey(
    soundings: pd.DataFrame,
    instrument: dict,
    survey_cfg: dict,
    out_csv: str | Path,
    out_config: str | Path,
    provenance: dict,
) -> tuple[Path, Path]:
    """Write survey.csv + sidecar.json; returns their paths."""
    out_csv, out_config = Path(out_csv), Path(out_config)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_config.parent.mkdir(parents=True, exist_ok=True)

    df = soundings.copy()

    # RAW_INGEST.md promises these counts in the provenance block (#67.2) — record
    # them, not just print them, so a delivered sidecar carries what was dropped
    no_pos = df[["easting", "northing", "elevation", "dem"]].isna().any(axis=1)
    n_dropped_nav = int(no_pos.sum())
    if n_dropped_nav:
        print(f"[emit] Dropped {n_dropped_nav} soundings with no position (nav gaps)")
        df = df[~no_pos]
    off_line = df["line"] == -1
    n_dropped_offline = int(off_line.sum())
    if n_dropped_offline:
        print(f"[emit] Dropped {n_dropped_offline} soundings outside lines (turns)")
        df = df[~off_line]
    if df.empty:
        raise ValueError("No soundings left after dropping nav gaps and turns")

    provenance = {**provenance,
                  "n_dropped_nav": n_dropped_nav,
                  "n_dropped_offline": n_dropped_offline,
                  "n_soundings_emitted": len(df)}

    gate_cols = stacked_gate_columns(df)
    out = pd.DataFrame({
        "LINE":      df["line"].astype(int),
        "FID":       df["fid"].astype(int),
        "Easting":   df["easting"].round(2),
        "Northing":  df["northing"].round(2),
        "Elevation": df["elevation"].round(2),
        "DEM":       df["dem"].round(2),
        "Latitude":  df["lat"].round(7),
        "Longitude": df["lon"].round(7),
    })
    for i, col in enumerate(gate_cols):
        out[f"SFz[{i}]"]     = df[col].fillna(_DUMMY).map(lambda v: f"{v:.6e}")
        out[f"SFz_std[{i}]"] = df[f"gate_std_{col[-2:]}"].fillna(_DUMMY).map(lambda v: f"{v:.6e}")
    out = out.fillna(_DUMMY)

    out.to_csv(out_csv, index=False)

    sidecar = build_sidecar(instrument, survey_cfg, n_gates=len(gate_cols),
                            provenance=provenance)
    with open(out_config, "w") as f:
        json.dump(sidecar, f, indent=2)

    print(f"[emit] {len(out)} soundings, {out['LINE'].nunique()} lines → {out_csv}")
    return out_csv, out_config


def build_sidecar(instrument: dict, survey_cfg: dict, n_gates: int, provenance: dict) -> dict:
    """Assemble the sidecar dict in the exact shape configs/example.json documents."""
    rx, tx = instrument["rx"], instrument["tx"]
    gate_times = instrument["gate_times_ms"]
    if len(gate_times) != n_gates:
        raise ValueError(
            f"instrument.yaml has {len(gate_times)} gate_times_ms but the EM stream "
            f"produced {n_gates} gates — instrument config does not match the data."
        )

    return {
        "survey": {
            "name":   survey_cfg["survey"]["name"],
            "system": instrument["instrument"]["name"],
            "date":   survey_cfg["survey"].get("date", ""),
            "epsg":   survey_cfg["survey"]["epsg"],
        },
        "column_map": {
            "line": "LINE", "fiducial": "FID",
            "easting": "Easting", "northing": "Northing",
            "elevation": "Elevation", "dem": "DEM",
            "latitude": "Latitude", "longitude": "Longitude",
            "sfz_prefix": "SFz", "sfz_n": n_gates, "sfz_format": "bracket",
        },
        "system": {
            "tx_moment_am2":      tx["moment_nominal_am2"],
            "tx_frequency_hz":    tx["frequency_hz"],
            "tx_waveform":        tx["waveform"],
            "tx_on_time_us":      tx["on_time_us"],
            "tx_ramp_off_us":     tx.get("ramp_off_us", 0.0),
            "rx_coil_area_m2":    rx["coil_area_m2"],
            "tx_geometry":        tx["geometry"],
            "tx_loop_radius_m":   tx["loop_radius_m"],
            "rx_dz_m":            rx.get("dz_m", 0.0),
            "rx_orientation":     rx.get("orientation", "Z"),
            "system_noise_floor": instrument["noise_floor"],
        },
        "gate_times_ms": gate_times,
        "inversion": survey_cfg["inversion"],
        "ingest": provenance,
    }


def file_digest(path: str | Path) -> str:
    """Short sha256 for provenance."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
