"""
Single-sounding inversion against known ground truth.

Forward-models a known layered earth, adds 3% noise, inverts it back, and
plots the fit + recovered model. The clearest way to see what alpha_s /
alpha_z / rho_initial do — edit them below and rerun.

Run from the repo root:

    python examples/03_single_sounding_inversion.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from tdem.forward import forward_from_config, layer_depths
from tdem.invert import invert_sounding
from tdem.visualize import plot_sounding_fit

REPO = Path(__file__).parent.parent
OUT = REPO / "output" / "examples"

with open(REPO / "configs/example.json") as f:
    config = json.load(f)

fwd = forward_from_config(config)
inv = config["inversion"]
bird_height = 35.0  # m AGL

# --- Ground truth: 200 ohm-m background, 5 ohm-m conductor at 20-80 m -------
z = layer_depths(fwd.thicknesses)
rho_true = np.full(fwd.n_layers, 200.0)
rho_true[(z >= 20) & (z <= 80)] = 5.0

rng = np.random.default_rng(0)
d_clean = fwd.predict(rho_true, bird_height)
d_obs = np.maximum(d_clean * (1 + rng.normal(0, 0.03, size=len(d_clean))), 1e-16)

# --- Invert ------------------------------------------------------------------
rho, rms, converged = invert_sounding(
    fwd, d_obs, bird_height,
    noise_floor=config["system"]["system_noise_floor"],
    rho_initial=inv["rho_initial"],
    rho_min=inv["rho_min"],
    rho_max=inv["rho_max"],
    alpha_s=inv["alpha_s"],   # reference-model damping — raise to pin to rho_initial
    alpha_z=inv["alpha_z"],   # vertical smoothing — raise for blockier-to-smoother models
    max_iter=inv["max_iter"],
)
print(f"converged={converged}, log-space RMS={rms:.3f}")

# --- Compare recovered vs. true ----------------------------------------------
d_pred = fwd.predict(rho, bird_height)
fig = plot_sounding_fit(
    d_obs, d_pred, config["gate_times_ms"], rho, layer_depths(fwd.thicknesses),
    OUT / "single_sounding_fit.png",
    title=f"Synthetic sounding — RMS {rms:.3f}",
)

in_body = (z >= 20) & (z <= 80)
print(f"recovered rho in conductor window: {np.median(rho[in_body]):.0f} ohm-m (true 5)")
print(f"recovered rho in background:       {np.median(rho[~in_body]):.0f} ohm-m (true 200)")
print("note: smoothing (alpha_z) blurs sharp boundaries — that's the Occam trade-off")
