"""
Generate a synthetic raw flight directory for testing ingest.

Writes every v0 stream (em/sync/gps/alt/txcur/lines) with the real
nuisances ingest must undo: receiver-clock offset + drift, alternating
Tx polarity, per-half-cycle noise and occasional sferic spikes, Tx
current variation, turns between lines, and different sample rates per
stream.

Decays are analytic power laws, NOT the SimPEG forward — this generator
tests ingest (does raw-in equal truth-out?), not inversion physics;
scripts/generate_synthetic.py remains the physics-true survey generator.

Returns the truth table so tests can compare ingested SFz values against
what was encoded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

# true receiver-clock model the generator encodes (and timesync must recover)
CLOCK_OFFSET_S = 1_750_000_000.0 - 500.0 * (1 + 2e-6)
CLOCK_RATE     = 1 + 2e-6
T_RX_START     = 500.0


def _t_utc(t_rx):
    return CLOCK_OFFSET_S + CLOCK_RATE * np.asarray(t_rx)


def generate_flight(
    out_dir: str | Path,
    instrument: dict,
    survey_cfg: dict,
    *,
    n_lines: int = 2,
    n_stations: int = 20,
    turn_s: float = 12.0,
    noise_frac: float = 0.03,
    spike_frac: float = 0.005,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Write a raw flight dir; return truth DataFrame:
    t_utc, easting, northing, elevation, dem, agl, line, sfz_00..
    (sfz in V/(A·m⁴), noise-free).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    tx, rx = instrument["tx"], instrument["rx"]
    gates_ms = np.array(instrument["gate_times_ms"])
    n_gates  = len(gates_ms)
    hc_rate  = 2 * tx["frequency_hz"]                       # half-cycles / s
    n_stack  = survey_cfg["ingest"]["n_stack"]
    station_s = n_stack / hc_rate                           # flight time per station
    nominal_current = tx["moment_nominal_am2"] / (tx["n_turns"] * tx["loop_area_m2"])

    # ---- flight path in projected coords: N-S lines, E-W turn legs ----
    epsg = survey_cfg["survey"]["epsg"]
    to_ll = Transformer.from_crs(epsg, 4326, always_xy=True)
    speed = 30.0                                            # m/s
    station_spacing = speed * station_s

    truth_rows, line_log = [], []
    t_rx = T_RX_START
    for li in range(n_lines):
        line_id  = 1000 + 10 * li
        x_line   = 500_000 + li * 200
        t_start  = _t_utc(t_rx)
        for j in range(n_stations):
            # serpentine: odd lines flown south
            y = 4_900_000 + (j if li % 2 == 0 else n_stations - 1 - j) * station_spacing
            z_ground = 1400 + 5 * np.sin(y / 500)
            agl      = 35 + 3 * np.sin(y / 300)
            in_body  = n_stations // 3 <= j < 2 * n_stations // 3
            # background: fast power-law decay; conductor: elevated late time
            amp, p = (2e-10, 1.8) if in_body else (1e-10, 2.5)
            sfz = amp * gates_ms ** (-p) * (35.0 / agl) ** 3

            truth_rows.append({
                "t_rx_start": t_rx, "t_utc": _t_utc(t_rx + station_s / 2),
                "easting": x_line, "northing": y,
                "elevation": z_ground + agl, "dem": agl, "agl": agl,
                "line": line_id, **{f"sfz_{k:02d}": sfz[k] for k in range(n_gates)},
            })
            t_rx += station_s
        line_log.append((line_id, t_start, _t_utc(t_rx)))
        t_rx += turn_s                                      # EM keeps logging in the turn

    truth = pd.DataFrame(truth_rows)
    t_rx_end = t_rx

    # ---- em log: n_stack half-cycles per station, plus turns ----------
    em_rows = []
    seq = 0
    tx_current = lambda trx: nominal_current * (1 + 0.003 * np.sin(_t_utc(trx) / 60))
    for row in truth.itertuples():
        sfz = np.array([getattr(row, f"sfz_{k:02d}") for k in range(n_gates)])
        for h in range(n_stack):
            trx = row.t_rx_start + h / hc_rate
            pol = 1 if seq % 2 == 0 else -1
            moment = tx_current(trx) * tx["n_turns"] * tx["loop_area_m2"]
            v = sfz * (1 + rng.normal(0, noise_frac, n_gates))
            v += rng.normal(0, instrument["noise_floor"], n_gates)
            if rng.random() < spike_frac:                   # sferic hit
                v *= 20
            v_logged = pol * v * moment * rx["gain"] * rx["coil_area_m2"]
            em_rows.append([seq, trx, pol] + list(v_logged))
            seq += 1
    em_cols = ["seq", "t_rx", "polarity"] + [f"g{k:02d}" for k in range(n_gates)]
    _write(out_dir / "em_001.log", em_cols, em_rows, "raw EM log v0")

    # ---- sync: 1 Hz GPS time messages, jitter well under the 1 ms spec
    t_sync = np.arange(T_RX_START, t_rx_end, 1.0)
    _write(out_dir / "sync_001.log", ["t_rx", "t_utc"],
           np.column_stack([t_sync, _t_utc(t_sync) + rng.normal(0, 2e-5, len(t_sync))]),
           "GPS time messages")

    # ---- gps 5 Hz / alt 10 Hz: interpolate the truth track -------------
    geoid = survey_cfg["survey"].get("geoid_offset_m", 0.0)
    tt = truth["t_utc"].to_numpy()
    for name, rate, cols in [("gps", 5.0, None), ("alt", 10.0, None)]:
        t_rx_s = np.arange(T_RX_START, t_rx_end, 1 / rate)
        t_u = _t_utc(t_rx_s)
        e = np.interp(t_u, tt, truth["easting"])
        n = np.interp(t_u, tt, truth["northing"])
        if name == "gps":
            elev = np.interp(t_u, tt, truth["elevation"])
            lon, lat = to_ll.transform(e, n)
            _write(out_dir / "gps_001.log", ["t_utc", "lat", "lon", "h_ell"],
                   np.column_stack([t_u, lat, lon, elev + geoid]), "GPS 5 Hz")
        else:
            agl = np.interp(t_u, tt, truth["agl"])
            _write(out_dir / "alt_001.log", ["t_rx", "agl_m"],
                   np.column_stack([t_rx_s, agl]), "altimeter 10 Hz")

    # ---- tx current 5 Hz ------------------------------------------------
    t_rx_c = np.arange(T_RX_START, t_rx_end, 0.2)
    _write(out_dir / "txcur_001.log", ["t_rx", "current_a"],
           np.column_stack([t_rx_c, tx_current(t_rx_c)]), "Tx current monitor")

    # ---- operator line log ----------------------------------------------
    _write(out_dir / "lines_001.csv", ["line", "t_start_utc", "t_end_utc"],
           [(l, f"{a:.3f}", f"{b:.3f}") for l, a, b in line_log], "operator line log")

    return truth


def _write(path: Path, cols: list[str], rows, comment: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# mvp-tdem synthetic {comment}\n")
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(_fmt(v) for v in r) + "\n")


def _fmt(v) -> str:
    if isinstance(v, str):
        return v
    if float(v) == int(v) and abs(float(v)) < 1e7:
        return str(int(v))
    # full float64 precision — unix-epoch t_utc needs ~16 significant
    # digits to keep sub-millisecond resolution for the clock fit
    return f"{float(v):.17g}"
