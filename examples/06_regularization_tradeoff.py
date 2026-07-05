"""
Occam regularization sweep — the smooth-vs-blocky trade-off.

Inverts the same synthetic sounding six times across a log-spaced grid of
alpha_z values (0.01 → 100), plotting all recovered models side-by-side.
This is the clearest way to see what the smoothing weight does and why
the default of 1.0 is a reasonable starting point.

Run from the repo root:

    python examples/06_regularization_tradeoff.py
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
from tdem.invert import invert_sounding

REPO = Path(__file__).parent.parent
OUT  = REPO / "output" / "examples"
OUT.mkdir(parents=True, exist_ok=True)

with open(REPO / "configs/example.json") as f:
    config = json.load(f)

fwd = forward_from_config(config)
inv = config["inversion"]
z   = layer_depths(fwd.thicknesses)
t_ms = np.asarray(config["gate_times_ms"])
bird_height = 35.0

# ---------------------------------------------------------------------------
# Ground truth: 200 ohm-m background with a 5 ohm-m conductor at 20–80 m
# ---------------------------------------------------------------------------
rho_true = np.full(fwd.n_layers, 200.0)
rho_true[(z >= 20) & (z <= 80)] = 5.0

rng    = np.random.default_rng(42)
d_obs  = fwd.predict(rho_true, bird_height)
d_obs *= (1 + rng.normal(0, 0.03, size=len(d_obs)))  # 3% noise
d_obs  = np.maximum(d_obs, config["system"]["system_noise_floor"])

# ---------------------------------------------------------------------------
# Sweep alpha_z over two decades either side of the default (1.0)
# ---------------------------------------------------------------------------
alpha_z_values = [0.01, 0.1, 1.0, 5.0, 20.0, 100.0]
results: list[tuple[float, np.ndarray, float, bool]] = []

print(f"{'alpha_z':>10}  {'chi':>6}  {'converged':>10}  "
      f"{'conductor median (Ω·m)':>24}")
print("-" * 58)

for alpha_z in alpha_z_values:
    rho, chi, converged = invert_sounding(
        fwd, d_obs, bird_height,
        noise_floor=config["system"]["system_noise_floor"],
        rho_initial=inv["rho_initial"],
        rho_min=inv["rho_min"],
        rho_max=inv["rho_max"],
        alpha_s=inv["alpha_s"],
        alpha_z=alpha_z,         # ← the variable we're sweeping
        max_iter=inv["max_iter"],
        rel_error=inv.get("rel_error", 0.05),
        chi_target=inv.get("chi_target", 1.0),
    )
    results.append((alpha_z, rho, chi, converged))
    conductor_med = float(np.median(rho[(z >= 20) & (z <= 80)]))
    print(f"{alpha_z:>10.2f}  {chi:>6.2f}  {str(converged):>10}  "
          f"{conductor_med:>24.0f}")

# ---------------------------------------------------------------------------
# Plot: all recovered models side-by-side
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, len(results), figsize=(14, 6), sharey=True)

for ax, (alpha_z, rho, chi, converged) in zip(axes, results):
    # Staircase plot
    stair_rho, stair_z = [], []
    z_edges = np.concatenate([z, [z[-1] * 1.4]])
    for k in range(len(rho)):
        stair_rho += [rho[k], rho[k]]
        stair_z   += [z_edges[k], z_edges[k + 1]]

    ax.semilogx(stair_rho, stair_z, color="steelblue", lw=1.5, label="recovered")
    ax.semilogx(rho_true, z, color="tab:red", lw=1, ls="--", alpha=0.6, label="true")

    # Highlight conductor depth window
    ax.axhspan(20, 80, alpha=0.1, color="tab:red")

    ax.invert_yaxis()
    ax.set_xlim(0.5, 2e4)
    ax.set_ylim(bottom=min(z[-1] * 1.3, 400))
    ax.set_title(
        f"α_z = {alpha_z}\nchi={chi:.2f}  {'✓' if converged else '✗'}",
        fontsize=9,
    )
    ax.set_xlabel("Ω·m", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, which="both", alpha=0.2)

axes[0].set_ylabel("Depth (m)")
axes[0].legend(fontsize=7, loc="lower right")

# Shade the default alpha_z panel
default_idx = alpha_z_values.index(1.0)
axes[default_idx].set_facecolor("#f0f8ff")
axes[default_idx].set_title(
    axes[default_idx].get_title() + "\n← default", fontsize=9
)

fig.suptitle(
    "Occam regularization sweep — varying alpha_z (alpha_s fixed at 1e-4)\n"
    "Red shading = true conductor, 5 Ω·m at 20–80 m.  Blue = default panel.",
    fontsize=10,
)
fig.tight_layout()

out_path = OUT / "regularization_sweep.png"
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"\nwrote {out_path}")

print("""
Interpretation guide
--------------------
alpha_z = 0.01  Very low smoothing: model is allowed to be rough.
                Conductor boundary is sharper but oscillations appear.
alpha_z = 0.1   Low smoothing: still noisy but conductor visible.
alpha_z = 1.0   Default: smooth model, conductor well-resolved but blurred.
                chi ≈ 1 — data fit to within assigned errors.
alpha_z = 5–20  Over-smoothed: chi > 1, data are NOT being fit.
alpha_z = 100   Severely over-smoothed: model barely varies with depth.

Key insight: the Occam cooling loop halts when chi <= chi_target (1.0 by
default). At high alpha_z the loop can't reach chi = 1 at any cooling
step, so the inversion accepts a rough fit. Monitoring chi tells you
whether the smoothing is too aggressive for your data quality.
""")
