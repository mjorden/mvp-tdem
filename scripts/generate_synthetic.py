"""
Generate a synthetic helicopter TDEM survey CSV for testing.

Simulates 3 flight lines (N-S traverses) with a conductive body in the centre.
Output columns match the 'bracket' convention: SFz[0]..SFz[19].

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

from tdem.forward import TDEMForward, layer_thicknesses


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


# Real SimPEG 1D forward — same physics the inversion uses, so the
# synthetic survey is a proper end-to-end test of the pipeline.
# Concentric-loop geometry (VTEM-style, #6).
_FWD = TDEMForward(GATE_TIMES_MS, layer_thicknesses(0.5, 300, 30),
                   tx_geometry="concentric_loop", tx_loop_radius_m=13.0)


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
            dem      = 35  + rng.normal(0, 1)

            # Buried conductor (5 Ω·m, 20–80 m depth) between
            # northing 4901500–4902500 on all lines; 200 Ω·m background
            in_body = 4901500 < northing < 4902500
            rho_layers = np.full(_FWD.n_layers, 200.0)
            if in_body:
                from tdem.forward import layer_depths
                z = layer_depths(_FWD.thicknesses)
                rho_layers[(z >= 20) & (z <= 80)] = 5.0

            signal  = sounding_response(rho_layers, dem)
            noise   = signal * rng.normal(0, 0.03, size=len(signal))
            signal  = np.maximum(signal + noise, 1e-16)

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
    print(f"Wrote {len(df)} soundings ({N_LINES} lines × {N_STATIONS} stations) → {out}")


if __name__ == "__main__":
    main()
