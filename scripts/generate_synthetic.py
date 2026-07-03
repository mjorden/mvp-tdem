"""
Generate a synthetic helicopter TDEM survey CSV for testing.

Simulates 3 flight lines (N-S traverses) with a conductive body in the centre.
Output columns match the 'bracket' convention: SFz[0]..SFz[19].

To avoid an inverse crime (#26), truth is generated on a fine mesh spaced
independently of the inversion's log-spaced 0.5-300 m / 30-layer mesh, the
conductor boundaries sit between the inversion's layer interfaces, and the
recorded DEM channel carries altimeter error relative to the height actually
simulated. Noise is 3% multiplicative plus additive Gaussian at the sidecar's
declared system_noise_floor (#32), so gates below the floor are
noise-dominated (negative values included) exactly as in real data.

This is a regression fixture for the pipeline's bookkeeping, not physics
validation — the truth still comes from the same SimPEG operator family the
inversion uses.

Usage
-----
    python scripts/generate_synthetic.py --out data/synthetic_survey.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from tdem.forward import TDEMForward, layer_depths


N_GATES    = 20
N_LINES    = 3
N_STATIONS = 80   # soundings per line
LINE_SPACING_M = 200
STATION_SPACING_M = 50

# 20 gates, all inside the 16 ms off-time of a 25 Hz / 4 ms-on-time
# bipolar waveform (#1) — matches configs/example.json
GATE_TIMES_MS = [
    0.0084, 0.0124, 0.0184, 0.0272, 0.0404, 0.0600, 0.0888,
    0.1312, 0.1944, 0.2880, 0.4268, 0.6320, 0.9360, 1.3860,
    2.0524, 3.0400, 4.5012, 6.6680, 9.8760, 14.624
]


# Truth mesh (#26): deliberately NOT the inversion's log-spaced
# 0.5-300 m / 30-layer mesh — 1 m uniform cells through the zone of
# interest, log-widening below, basal half-space from 400 m (145 layers).
TRUE_THICKNESSES = np.concatenate([
    np.full(120, 1.0),
    np.diff(np.logspace(np.log10(120.0), np.log10(400.0), 25)),
])

# Conductor (#26): boundaries at 24 and 77 m fall between the inversion's
# nearest layer interfaces (24.31 m and 76.17 m), so the true model is not
# exactly representable in the inversion's model space.
BODY_TOP_M     = 24.0
BODY_BOT_M     = 77.0
BODY_RHO       = 5.0
BACKGROUND_RHO = 200.0

# Noise (#32): multiplicative measurement noise plus an additive floor at
# the sidecar's declared system_noise_floor (configs/example.json).
REL_NOISE   = 0.03
NOISE_FLOOR = 1e-12  # V/(A*m^4)

# Concentric-loop geometry (VTEM-style, #6); 25 Hz bipolar square wave
# matching the example.json sidecar (#22).
_FWD = TDEMForward(GATE_TIMES_MS, TRUE_THICKNESSES,
                   tx_geometry="concentric_loop", tx_loop_radius_m=13.0,
                   waveform="bipolar_square", base_frequency_hz=25.0,
                   on_time_ms=4.0)


def sounding_response(rho_layers: np.ndarray, bird_height_m: float) -> np.ndarray:
    """Moment-normalized |dB/dt| for a layered model at the given bird height."""
    return _FWD.predict(rho_layers, bird_height_m)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synthetic_survey.csv")
    args = parser.parse_args()

    rng = np.random.default_rng(42)

    rows = []
    fid  = 1000

    for i, line_id in enumerate([1000, 2000, 3000]):
        x_centre = 500000 + i * LINE_SPACING_M
        for j in range(N_STATIONS):
            northing = 4900000 + j * STATION_SPACING_M
            easting  = x_centre
            elev     = 1450 + rng.normal(0, 2)
            # Truth is simulated at the actual bird height; the recorded DEM
            # channel carries altimeter error, so the inversion never sees
            # perfect geometry (#26)
            h_true   = 35 + rng.normal(0, 1)
            dem      = h_true + rng.normal(0, 0.5)

            # Buried conductor (5 ohm-m, 24-77 m depth) between
            # northing 4901500-4902500 on all lines; 200 ohm-m background
            in_body = 4901500 < northing < 4902500
            rho_layers = np.full(_FWD.n_layers, BACKGROUND_RHO)
            if in_body:
                z = layer_depths(_FWD.thicknesses)
                rho_layers[(z >= BODY_TOP_M) & (z < BODY_BOT_M)] = BODY_RHO

            clean  = sounding_response(rho_layers, h_true)
            # No clamp: gates whose true amplitude sits below the floor go
            # noise-dominated, negatives included, as in real data (#32)
            signal = clean * (1.0 + rng.normal(0, REL_NOISE, size=len(clean))) \
                     + rng.normal(0, NOISE_FLOOR, size=len(clean))

            row = {
                "LINE":     line_id,
                "FID":      fid,
                "Easting":  round(easting, 2),
                "Northing": round(northing, 2),
                "Elevation": round(elev, 2),
                "DEM":      round(dem, 2),
                "Latitude": round(44.2 + j * 0.0004, 6),
                "Longitude": round(-119.4 + i * 0.002, 6),
            }
            for k, val in enumerate(signal):
                row[f"SFz[{k}]"] = f"{val:.6e}"

            rows.append(row)
            fid += 1

    df  = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    # ASCII only — cp1252 consoles on Windows choke on arrows/multiplication signs
    print(f"Wrote {len(df)} soundings ({N_LINES} lines x {N_STATIONS} stations) -> {out}")


if __name__ == "__main__":
    main()
