"""
Orchestration: flight directory → soundings frame; survey → CSV + sidecar.

    readers → timesync → stack → calibrate → merge → geometry → emit

Each stage lives in its own module and is unit-tested there; this file
only chains them and gathers provenance.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from . import readers
from .calibrate import calibrate
from .emit import emit_survey, file_digest
from .geometry import assign_fids, assign_lines, elevations, project
from .merge import merge_nav
from .timesync import apply_clock, fit_clock


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def ingest_flight(flight_dir: str | Path, instrument: dict, survey_cfg: dict) -> tuple[pd.DataFrame, dict]:
    """
    Run one flight through readers → geometry (no FIDs/emit — those are
    survey-level). Returns (soundings df, per-flight provenance dict).
    """
    flight_dir = Path(flight_dir)
    ing   = survey_cfg.get("ingest", {})
    n_stack   = ing.get("n_stack", 24)
    trim_frac = ing.get("trim_frac", 0.1)
    max_gap_s = ing.get("max_gap_s", 2.0)

    import warnings

    paths = readers.discover_flight(flight_dir)

    sync_df = readers.read_sync(paths["sync"])
    clock = fit_clock(sync_df,
                      max_residual_ms=instrument.get("clock", {}).get("max_residual_ms", 1.0))

    em  = apply_clock(readers.read_em(paths["em"]), clock)
    alt = apply_clock(readers.read_alt(paths["alt"]), clock)
    gps = readers.read_gps(paths["gps"])
    txcur = (apply_clock(readers.read_txcur(paths["txcur"]), clock)
             if paths["txcur"] else None)
    lines_df = readers.read_lines(paths["lines"]) if paths["lines"] else None

    # GPS-vs-UTC leap-second diagnostic (#64): if the sync log and GPS position
    # log disagree by ~18–19 s their time standards differ (GPS ≠ UTC).
    if "t_utc" in gps.columns:
        dt = sync_df["t_utc"].median() - gps["t_utc"].median()
        if 14.0 <= abs(dt) <= 22.0:
            warnings.warn(
                f"[timesync] Sync stream and GPS position stream differ by "
                f"{dt:+.1f} s — possible GPS/UTC leap-second mismatch "
                f"(current offset is ~18 s).  Verify both streams use the "
                "same time standard or ~540 m position errors will result.",
                stacklevel=2,
            )

    df = stack_and_locate(em, gps, alt, txcur, lines_df, instrument, survey_cfg,
                          n_stack=n_stack, trim_frac=trim_frac, max_gap_s=max_gap_s)

    provenance = {
        "flight": flight_dir.name,
        "sources": {p.name: file_digest(p) for ps in paths.values() for p in ps},
        "n_soundings": len(df),
        "n_stack": n_stack,
        "trim_frac": trim_frac,
        "time_sync": {
            "mode": "gps_messages",
            "model": "piecewise_linear",
            "max_residual_ms": clock.max_residual_s * 1000,
            "n_pairs": clock.n_pairs,
        },
        "moment": df.attrs["moment_mode"],
        "line_assignment": "operator_log" if lines_df is not None else "heading_auto",
    }
    return df, provenance


def stack_and_locate(em, gps, alt, txcur, lines_df, instrument, survey_cfg,
                     *, n_stack, trim_frac, max_gap_s) -> pd.DataFrame:
    """The format-agnostic core: time-synced frames in, located soundings out."""
    from .stack import stack_soundings

    df = stack_soundings(em, n_stack=n_stack, trim_frac=trim_frac,
                         tx_frequency_hz=instrument["tx"].get("frequency_hz"))
    df, moment_mode = calibrate(df, instrument, txcur, max_gap_s=max_gap_s)
    df = merge_nav(df, gps, alt, max_gap_s=max_gap_s)
    df = project(df, epsg=survey_cfg["survey"]["epsg"])
    df = elevations(df, geoid_offset_m=survey_cfg["survey"].get("geoid_offset_m", 0.0))
    df = assign_lines(df, lines_df)
    df.attrs["moment_mode"] = moment_mode
    return df


def ingest_survey(
    flight_dirs: list[str | Path],
    instrument_path: str | Path,
    survey_path: str | Path,
    out_csv: str | Path,
    out_config: str | Path,
) -> tuple[Path, Path]:
    """All flights of a survey → one canonical CSV + generated sidecar."""
    instrument = load_yaml(instrument_path)
    survey_cfg = load_yaml(survey_path)

    frames, flights_prov = [], []
    for d in flight_dirs:
        df, prov = ingest_flight(d, instrument, survey_cfg)
        frames.append(df)
        flights_prov.append(prov)

    df = pd.concat(frames, ignore_index=True).sort_values("t_utc").reset_index(drop=True)
    df = assign_fids(df)

    provenance = {
        "ingest_version": "0.1",
        "instrument_config": str(instrument_path),
        "survey_config": str(survey_path),
        "flights": flights_prov,
    }
    return emit_survey(df, instrument, survey_cfg, out_csv, out_config, provenance)
