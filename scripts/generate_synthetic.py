"""
Generate a synthetic helicopter TDEM survey CSV for testing.

Simulates 3 flight lines (N-S traverses) with a conductive body in the centre.
Output columns match the 'bracket' convention: SFz[0]..SFz[29].

Usage
-----
    python scripts/generate_synthetic.py --out data/synthetic_survey.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


N_GATES    = 30
N_LINES    = 3
N_STATIONS = 80   # soundings per line
LINE_SPACING_M = 200
STATION_SPACING_M = 50

GATE_TIMES_MS = [
    0.0084, 0.0124, 0.0184, 0.0272, 0.0404, 0.0600, 0.0888,
    0.1312, 0.1944, 0.2880, 0.4268, 0.6320, 0.9360, 1.3860,
    2.0524, 3.0400, 4.5012, 6.6680, 9.8760, 14.624, 21.664,
    32.076, 47.520, 70.380, 104.23, 154.44, 228.76, 338.80,
    501.96, 743.40
]


def halfspace_response(t_ms: np.ndarray, rho: float, moment: float = 420000) -> np.ndarray:
    """
    Approximate 1D half-space dB/dt response in V/(A·m⁴) using late-time approximation.
    Not a rigorous forward model — for synthetic data shape only.
    """
    t = t_ms * 1e-3
    mu0 = 4 * np.pi * 1e-7
    sigma = 1.0 / rho
    # late-time asymptotic dB/dt for a magnetic dipole over a half-space
    response = (sigma ** 1.5) * (mu0 ** 2.5) / (20 * np.pi ** 1.5) * t ** (-2.5)
    return response / moment   # normalise by Tx moment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synthetic_survey.csv")
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    t   = np.array(GATE_TIMES_MS)

    rows = []
    fid  = 1000

    for i, line_id in enumerate([1000, 2000, 3000]):
        x_centre = 500000 + i * LINE_SPACING_M
        for j in range(N_STATIONS):
            northing = 4900000 + j * STATION_SPACING_M
            easting  = x_centre
            elev     = 1450 + rng.normal(0, 2)
            dem      = 35  + rng.normal(0, 1)

            # Conductive body between northing 4901500–4902500 on all lines
            in_body = 4901500 < northing < 4902500
            rho = 5.0 if in_body else 200.0

            signal  = halfspace_response(t, rho)
            noise   = signal * rng.normal(0, 0.05, size=len(t))
            signal  = np.maximum(signal + noise, 1e-14)

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
