"""
Generate a synthetic helicopter TDEM survey CSV for testing.

Simulates 3 flight lines (N-S traverses) with a conductive body in the centre.
Output columns match the 'bracket' convention: SFz[0]..SFz[19].

Usage
-----
    python scripts/generate_synthetic.py --out data/synthetic_survey.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from tdem.forward import forward_from_config


N_LINES    = 3
N_STATIONS = 80   # soundings per line
LINE_SPACING_M = 200
STATION_SPACING_M = 50

# Build the forward straight from the sidecar the data will be inverted with,
# so gates, mesh, geometry, and waveform (#22/#52: bipolar train + turn-off
# ramp) stay in lockstep with configs/example.json. Note this keeps the
# operator identical to the inversion's — the inverse-crime caveat is #26.
_CONFIG = json.loads(
    (Path(__file__).parent.parent / "configs" / "example.json").read_text())
_FWD = forward_from_config(_CONFIG)


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
