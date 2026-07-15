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
from .geometry import apply_layback, assign_fids, assign_lines, elevations, project
from .merge import merge_nav
from .timesync import apply_clock, fit_clock


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
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

    # sync is special: it DEFINES the clock the others are converted with
    sync_df = readers.STREAMS["sync"].reader(paths["sync"])
    clock = fit_clock(sync_df,
                      max_residual_ms=instrument.get("clock", {}).get("max_residual_ms", 1.0))

    # every other stream is read + clock-converted generically from the
    # registry (#87): a newly registered stream lands in `frames` with no
    # edits here. Stage wiring below still names the core streams.
    frames: dict[str, pd.DataFrame | None] = {}
    for name, spec in readers.STREAMS.items():
        if name == "sync":
            continue
        if not paths[name]:
            frames[name] = None
            continue
        df_s = spec.reader(paths[name])
        frames[name] = apply_clock(df_s, clock) if spec.needs_clock else df_s

    em, gps, alt = frames["em"], frames["gps"], frames["alt"]
    txcur, lines_df = frames["txcur"], frames["lines"]

    _assert_time_standards_agree(sync_df, gps, ing.get("allow_time_offset", False))
    gps_quality = _gps_quality_summary(gps)         # #71.3 (None if no quality cols)

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
        "tow": instrument.get("tow") or "NOT SPECIFIED (antenna == bird assumed)",
        "gps_quality": gps_quality or "no quality columns in GPS log",
        "line_assignment": "operator_log" if lines_df is not None else "heading_auto",
    }
    return df, provenance


def _gps_quality_summary(gps, max_hdop: float = 5.0, min_fix: int = 3,
                         bad_frac_warn: float = 0.05) -> dict | None:
    """
    GPS quality metadata check (#71.3): a float-solution segment is otherwise
    indistinguishable from RTK. Readers pass every column through, so if the
    GPS log carries hdop / fix / nsat they land here; warn when a meaningful
    fraction of the stream is degraded, and record the stats in provenance.
    Returns None when no quality columns exist (quality unknown — also recorded).
    """
    import warnings
    cols = [c for c in ("hdop", "fix", "nsat") if c in getattr(gps, "columns", [])]
    if not cols:
        return None
    out: dict = {}
    n = len(gps)
    if "hdop" in cols:
        frac = float((gps["hdop"] > max_hdop).mean())
        out["hdop_median"] = float(gps["hdop"].median())
        out["frac_hdop_gt_max"] = round(frac, 4)
        if frac > bad_frac_warn:
            warnings.warn(f"[gps] {100*frac:.1f}% of GPS samples have HDOP > "
                          f"{max_hdop} — degraded positioning.", stacklevel=2)
    if "fix" in cols:
        frac = float((gps["fix"] < min_fix).mean())
        out["frac_fix_below_min"] = round(frac, 4)
        if frac > bad_frac_warn:
            warnings.warn(f"[gps] {100*frac:.1f}% of GPS samples have fix type < "
                          f"{min_fix} (float/autonomous, not RTK/DGPS).", stacklevel=2)
    if "nsat" in cols:
        out["nsat_min"] = int(gps["nsat"].min())
    return out


def _assert_time_standards_agree(sync_df, gps, allow_time_offset: bool,
                                 tol_s: float = 5.0) -> None:
    """
    GPS-vs-UTC leap-second check (#64/#G6): if the sync log and the GPS position
    log disagree by seconds their time standards differ (GPS ≠ UTC), producing a
    uniform ~540 m along-track shift. Too severe to ship on a warning a batch log
    can bury, so it HARD-ERRORS by default; ingest.allow_time_offset=true overrides
    once verified.
    """
    import warnings
    if "t_utc" not in getattr(gps, "columns", []):
        return
    dt = float(sync_df["t_utc"].median() - gps["t_utc"].median())
    if abs(dt) < tol_s:                       # sub-tolerance ⇒ ordinary clock jitter
        return
    msg = (f"[timesync] Sync stream and GPS position stream differ by {dt:+.1f} s "
           "— GPS/UTC time-standard mismatch (leap-second offset ~18 s). Maps to "
           f"~{abs(dt)*30:.0f} m along-track position error at survey speed.")
    if allow_time_offset:
        warnings.warn(msg + " Proceeding (allow_time_offset=true).", stacklevel=2)
    else:
        raise ValueError(
            msg + " Fix the time standards, or set ingest.allow_time_offset: true "
            "in the survey config to override."
        )


def stack_and_locate(em, gps, alt, txcur, lines_df, instrument, survey_cfg,
                     *, n_stack, trim_frac, max_gap_s) -> pd.DataFrame:
    """The format-agnostic core: time-synced frames in, located soundings out."""
    from .stack import stack_soundings

    f_tx = instrument["tx"].get("frequency_hz")
    df = stack_soundings(em, n_stack=n_stack, trim_frac=trim_frac, tx_frequency_hz=f_tx)
    # stack window spans n_stack half-cycles → n_stack/(2·f) seconds; averaging the
    # Tx current over it de-aliases a slow monitor (#65.1)
    window_s = n_stack / (2.0 * f_tx) if f_tx else 0.0
    df, moment_mode = calibrate(df, instrument, txcur, max_gap_s=max_gap_s, window_s=window_s)
    df = merge_nav(df, gps, alt, max_gap_s=max_gap_s)
    epsg = survey_cfg["survey"]["epsg"]
    df = project(df, epsg=epsg)

    # static layback / lever-arm (#70): antenna position → bird position.
    # An instrument config WITHOUT a tow block gets a loud reminder — the
    # correction is unconditional on real flights (~20-30 m horizontal,
    # ~15-25 m vertical); explicit zeros (rigid boom / simulated bird) silence it.
    tow = instrument.get("tow")
    if tow is None:
        import warnings
        warnings.warn(
            "[geometry] instrument.yaml has no `tow` block — positions assume the "
            "GPS antenna is AT the bird. Set tow.layback_m / tow.drop_m (explicit "
            "zeros for a rigid boom) before the first real flight (#70).",
            stacklevel=2,
        )
    else:
        df = apply_layback(df, layback_m=tow.get("layback_m", 0.0),
                           drop_m=tow.get("drop_m", 0.0), epsg=epsg)

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
        "git_commit": _git_commit(),                       # #67.3
        # hash the configs, not just their paths (#67.1): the sidecar physics
        # block is GENERATED from instrument.yaml, so an after-the-fact yaml edit
        # would otherwise be undetectable
        "instrument_config": str(instrument_path),
        "instrument_config_sha256": file_digest(instrument_path),
        "survey_config": str(survey_path),
        "survey_config_sha256": file_digest(survey_path),
        "flights": flights_prov,
    }
    return emit_survey(df, instrument, survey_cfg, out_csv, out_config, provenance)


def _git_commit() -> str | None:
    """Best-effort short git commit for provenance (#67.3); None if unavailable."""
    import subprocess
    try:
        out = subprocess.run(
            ["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None
