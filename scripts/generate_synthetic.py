"""
Generate a synthetic helicopter TDEM survey CSV for testing.

Simulates 3 flight lines (N-S traverses) with a conductive body in the centre.
Output columns match the 'bracket' convention: SFz[0]..SFz[19].

Usage
-----
    python scripts/generate_synthetic.py --out data/synthetic_survey.csv
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from tdem.forward import forward_from_config, layer_depths


N_LINES    = 3
N_STATIONS = 80   # soundings per line
LINE_SPACING_M = 200
STATION_SPACING_M = 50

# The acquisition parameters (gate times, waveform #22/#52, tx/rx geometry,
# noise floor) legitimately match configs/example.json — those describe the real
# survey. What must NOT match is the DISCRETIZATION: generating on the exact mesh
# the inversion uses, with the body painted onto the inversion's own layer
# interfaces, is an inverse crime (#26) — the true model is then exactly
# representable in the inversion's model space and the data come from the same
# discrete operator being inverted, so the "end-to-end test" can only fail on
# bookkeeping bugs and is structurally blind to discretization / mesh-adequacy /
# geometry error.
#
# So we generate the truth on a MUCH finer, independently-spaced mesh and define
# the conductor by DEPTH (its 20 m / 80 m boundaries fall between the inversion's
# coarse interfaces, not on them). The inversion still uses the coarse
# example.json mesh.
_CONFIG = json.loads(
    (Path(__file__).parent.parent / "configs" / "example.json").read_text())

_TRUTH_CONFIG = copy.deepcopy(_CONFIG)
_TRUTH_CONFIG["inversion"]["n_layers"]    = 160    # ~4× the inversion's 36
_TRUTH_CONFIG["inversion"]["depth_min_m"] = 0.3    # independent spacing
_TRUTH_CONFIG["inversion"]["depth_max_m"] = 800.0
_FWD_TRUTH = forward_from_config(_TRUTH_CONFIG)
_Z_TRUTH   = layer_depths(_FWD_TRUTH.thicknesses)

# A separate coarse operator, ONLY to report where the inversion's interfaces
# fall — so the printout can confirm the body boundaries miss them (#26).
_Z_INV = layer_depths(forward_from_config(_CONFIG).thicknesses)

_NOISE_FLOOR = _CONFIG["system"]["system_noise_floor"]

from pyproj import Transformer

_TO_WGS84 = Transformer.from_crs(
    f"EPSG:{_CONFIG['survey']['epsg']}", "EPSG:4326", always_xy=True)


def truth_model(in_body: bool) -> np.ndarray:
    """Fine-mesh resistivity model: 200 Ω·m background, optional 5 Ω·m body 20–80 m."""
    rho = np.full(_FWD_TRUTH.n_layers, 200.0)
    if in_body:
        rho[(_Z_TRUTH >= 20.0) & (_Z_TRUTH <= 80.0)] = 5.0
    return rho


def sounding_response(rho_layers: np.ndarray, bird_height_m: float) -> np.ndarray:
    """Moment-normalized dB/dt for a fine-mesh model at the given bird height."""
    return _FWD_TRUTH.predict(rho_layers, bird_height_m)


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
            in_body    = 4901500 < northing < 4902500
            rho_layers = truth_model(in_body)

            clean = sounding_response(rho_layers, dem)
            # Noise (#32): 3% multiplicative AND additive Gaussian at the declared
            # system_noise_floor. The additive term is what makes late gates below
            # the floor behave realistically (S/N → O(1) and lower), instead of
            # carrying pristine 3% noise many decades under the stated floor.
            # No 1e-16 clamp: near-floor and sign-changing late gates are physical
            # and the loader/QC/inversion now handle them (#13/#55/#53).
            mult = clean * rng.normal(0, 0.03, size=len(clean))
            addn = rng.normal(0, _NOISE_FLOOR, size=len(clean))
            signal = clean + mult + addn

            # lat/lon must be the true inverse projection of easting/northing
            # under the declared EPSG — the acceptance harness (#95) cross-checks
            # them, and the old hand-rolled placeholders were ~190 km off
            lon, lat = _TO_WGS84.transform(easting, northing)
            row = {
                "LINE":     line_id,
                "FID":      fid,
                "Easting":  round(easting, 2),
                "Northing": round(northing, 2),
                "Elevation": round(elev, 2),
                "DEM":      round(dem, 2),
                "Latitude": round(lat, 6),
                "Longitude": round(lon, 6),
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

    # #26: confirm the truth mesh is independent of the inversion mesh and the
    # body boundaries fall BETWEEN the inversion's interfaces (not on them).
    near = lambda z: float(np.min(np.abs(_Z_INV - z)))
    print(f"Truth mesh: {_FWD_TRUTH.n_layers} layers (inversion uses "
          f"{len(_Z_INV)}); body top 20 m is {near(20.0):.2f} m from the nearest "
          f"inversion interface, base 80 m is {near(80.0):.2f} m — off-grid, so "
          f"the truth is not exactly representable in the inversion model space.")


if __name__ == "__main__":
    main()
