"""
Visualization for stitched 1D TDEM inversion results.

plot_section()  — 2D resistivity cross-section for one flight line
                  (pcolormesh, log-colour, elevation-referenced, RMS strip)
plot_decays()   — observed decay curves per sounding, coloured by line position
plot_sounding_fit() — observed vs predicted decay for a single sounding

All figures are matplotlib; savefig-ready for report output. Interactive
Plotly versions are a Phase 2 item.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from .invert import LineResult
from .load import gate_columns


# ---------------------------------------------------------------------------
# Resistivity section
# ---------------------------------------------------------------------------

def plot_section(
    result: LineResult,
    out_path: str | Path | None = None,
    *,
    rho_min: float | None = None,
    rho_max: float | None = None,
    max_depth: float | None = None,
    cmap: str = "turbo_r",   # reversed: red = conductive (EM convention)
    title: str | None = None,
) -> plt.Figure:
    """
    Stitched resistivity section for one line: distance vs elevation, colour = log10(rho).

    Each sounding is drawn as a column of layer cells hung from its ground
    elevation, so topography is honoured. An RMS misfit strip is drawn above.
    """
    if not result.soundings:
        raise ValueError("LineResult has no soundings to plot.")

    S = result.soundings
    n = len(S)

    # along-line distances (from first sounding)
    x0, y0 = S[0].easting, S[0].northing
    dist = np.array([np.hypot(s.easting - x0, s.northing - y0) for s in S])

    # column edges midway between soundings
    edges = np.empty(n + 1)
    if n > 1:
        edges[1:-1] = 0.5 * (dist[:-1] + dist[1:])
        edges[0] = dist[0] - (edges[1] - dist[0])
        edges[-1] = dist[-1] + (dist[-1] - edges[-2])
    else:
        edges[:] = [dist[0] - 25.0, dist[0] + 25.0]

    depths = S[0].depths                      # top of each layer
    bottom = depths[-1] * 1.4                 # nominal bottom for basal half-space cell
    z_edges_rel = np.concatenate([depths, [bottom]])
    if max_depth is not None:
        z_edges_rel = np.minimum(z_edges_rel, max_depth)

    rho = np.column_stack([s.rho for s in S])          # (n_layers, n_soundings)
    elev = np.array([s.elevation for s in S])
    rms = np.array([s.rms for s in S])

    vmin = rho_min or max(np.nanpercentile(rho, 2), 1e-2)
    vmax = rho_max or np.nanpercentile(rho, 98)
    if vmax <= vmin:
        vmax = vmin * 10

    fig, (ax_rms, ax) = plt.subplots(
        2, 1, figsize=(12, 6.5), height_ratios=[1, 8], sharex=True,
        gridspec_kw={"hspace": 0.06},
    )

    # per-column pcolormesh so each column hangs from its own ground elevation
    pc = None
    for j in range(n):
        xx = edges[j:j + 2]
        zz = elev[j] - z_edges_rel                      # elevation of layer edges
        X, Z = np.meshgrid(xx, zz)
        pc = ax.pcolormesh(
            X, Z, rho[:, j:j + 1],
            norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cmap, shading="flat",
        )

    ax.plot(dist, elev, color="k", lw=1.2)              # ground surface
    ax.set_xlabel("Distance along line (m)")
    ax.set_ylabel("Elevation (m)")
    ax.set_xlim(edges[0], edges[-1])

    cb = fig.colorbar(pc, ax=ax, pad=0.01, aspect=30)
    cb.set_label("Resistivity (Ω·m)")

    # RMS strip
    ax_rms.bar(dist, rms, width=np.diff(edges), color="0.5")
    ax_rms.set_ylabel("RMS", fontsize=8)
    ax_rms.tick_params(labelsize=7)
    ax_rms.set_title(title or f"Line {result.line} — stitched 1D resistivity section",
                     fontsize=11)

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"[viz] wrote {out_path}")
    return fig


# ---------------------------------------------------------------------------
# Decay curves
# ---------------------------------------------------------------------------

def plot_decays(
    df_line,
    gate_times_ms,
    out_path: str | Path | None = None,
    *,
    title: str | None = None,
) -> plt.Figure:
    """
    All observed decay curves for a line, coloured by along-line position.
    Quick QC view: cultural noise and dead gates stand out immediately.
    """
    cols = gate_columns(df_line)
    t = np.asarray(gate_times_ms, dtype=float)
    data = df_line[cols].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("viridis")
    n = len(df_line)
    for i in range(n):
        ax.loglog(t, np.abs(data[i]), color=cmap(i / max(n - 1, 1)), alpha=0.4, lw=0.8)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("|dB/dt|  (V/(A·m⁴))")
    ax.set_title(title or "Observed decays (colour = along-line position)")
    ax.grid(True, which="both", alpha=0.3)

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"[viz] wrote {out_path}")
    return fig


def plot_sounding_fit(
    d_obs: np.ndarray,
    d_pred: np.ndarray,
    gate_times_ms,
    rho: np.ndarray,
    depths: np.ndarray,
    out_path: str | Path | None = None,
    *,
    title: str | None = None,
) -> plt.Figure:
    """Two-panel diagnostic: decay fit (left) and recovered 1D model (right)."""
    t = np.asarray(gate_times_ms, dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    use = np.isfinite(d_obs) & (d_obs > 0)
    ax1.loglog(t[use], d_obs[use], "ko", ms=5, label="observed")
    ax1.loglog(t, d_pred, "r-", lw=1.5, label="predicted")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("|dB/dt|  (V/(A·m⁴))")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)

    # stairs plot of the model
    z = np.concatenate([depths, [depths[-1] * 1.4]])
    ax2.stairs(rho, z, orientation="horizontal", color="tab:blue", lw=1.5)
    ax2.set_xscale("log")
    ax2.invert_yaxis()
    ax2.set_xlabel("Resistivity (Ω·m)")
    ax2.set_ylabel("Depth (m)")
    ax2.grid(True, which="both", alpha=0.3)

    if title:
        fig.suptitle(title)

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"[viz] wrote {out_path}")
    return fig
