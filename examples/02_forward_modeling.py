"""
Forward-modeling sandbox: how does a buried conductor change the decay?

Builds the survey's forward operator from the sidecar config and predicts
moment-normalized dB/dt for a uniform halfspace vs. the same halfspace with
a conductive layer — useful for gate-time sanity checks and survey design.

Run from the repo root:

    python examples/02_forward_modeling.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tdem.forward import forward_from_config, layer_depths

REPO = Path(__file__).parent.parent
OUT = REPO / "output" / "examples"

with open(REPO / "configs/example.json", encoding="utf-8") as f:
    config = json.load(f)

# One TDEMForward per survey: gate times and layer geometry are fixed,
# only bird height and the trial model vary per call.
fwd = forward_from_config(config)
t_ms = np.asarray(config["gate_times_ms"])
bird_height = 35.0  # m AGL

# 200 ohm-m background vs. the same with a 5 ohm-m layer at 20-80 m —
# the same body scripts/generate_synthetic.py buries in the synthetic survey.
z = layer_depths(fwd.thicknesses)
rho_bg = np.full(fwd.n_layers, 200.0)
rho_cond = rho_bg.copy()
rho_cond[(z >= 20) & (z <= 80)] = 5.0

d_bg = fwd.predict(rho_bg, bird_height)
d_cond = fwd.predict(rho_cond, bird_height)

fig, ax = plt.subplots(figsize=(7, 6))
ax.loglog(t_ms, d_bg, "o-", label="200 Ω·m halfspace")
ax.loglog(t_ms, d_cond, "s-", label="+ 5 Ω·m layer, 20–80 m")
ax.axhline(config["system"]["system_noise_floor"], color="0.5", ls="--",
           label="system noise floor")
ax.set_xlabel("Time after turnoff (ms)")
ax.set_ylabel("|dB/dt|  (V/(A·m⁴))")
ax.set_title("Effect of a buried conductor on the TDEM decay")
ax.legend()
ax.grid(True, which="both", alpha=0.3)

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "forward_conductor_vs_halfspace.png", dpi=200, bbox_inches="tight")
print(f"wrote {OUT / 'forward_conductor_vs_halfspace.png'}")

# The conductor slows the decay: mid/late gates come up hot relative to
# background. Print the max ratio so you can see it without the plot.
ratio = d_cond / d_bg
print(f"max conductor/background ratio: {ratio.max():.1f}x at t = {t_ms[ratio.argmax()]} ms")
