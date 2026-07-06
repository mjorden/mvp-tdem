"""
Visualization for stitched 1D TDEM inversion results.

plot_section()  — 2D resistivity cross-section for one flight line
                  (pcolormesh, log-colour, elevation-referenced, chi misfit strip)
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
    elevation, so topography is honoured. All cells are rendered as a SINGLE
    PolyCollection (#20: one artist, not one pcolormesh per sounding — matters for
    thousand-sounding lines) while keeping exact per-column hanging.

    Cells below each sounding's depth of investigation are alpha-faded and a DOI
    line is drawn (#12): the deep model there is the regularization prior bent by
    the data, NOT a resolved result, and must not be interpreted as geology.

    A chi misfit strip is drawn above (chi ~ 1 = data fit to within assigned
    errors). The elevation axis is GPS/ellipsoid height, which differs from
    orthometric (map / borehole-collar) elevation by the local geoid undulation.
    """
    if not result.soundings:
        raise ValueError("LineResult has no soundings to plot.")

    from matplotlib.collections import PolyCollection
    from .invert import along_line_distance, basal_depth_bottom

    S = result.soundings
    n = len(S)

    dist = along_line_distance([s.easting for s in S], [s.northing for s in S])

    # column edges midway between soundings
    edges = np.empty(n + 1)
    if n > 1:
        edges[1:-1] = 0.5 * (dist[:-1] + dist[1:])
        edges[0] = dist[0] - (edges[1] - dist[0])
        edges[-1] = dist[-1] + (dist[-1] - edges[-2])
    else:
        edges[:] = [dist[0] - 25.0, dist[0] + 25.0]

    depths = S[0].depths                      # top of each layer
    z_edges_rel = np.concatenate([depths, [basal_depth_bottom(depths)]])
    if max_depth is not None:
        z_edges_rel = np.minimum(z_edges_rel, max_depth)

    rho = np.column_stack([s.rho for s in S])          # (n_layers, n_soundings)
    elev = np.array([s.elevation for s in S])
    chi = np.array([s.chi for s in S])
    n_layers = len(depths)

    # #20: an explicitly-passed rho_min/rho_max must be honoured, not treated as
    # "unset" (the old `rho_min or …` swallowed a deliberate 0.0). The colour
    # scale is logarithmic, so vmin is floored to a positive value — a ≤0 request
    # is meaningless for log resistivity and would break LogNorm.
    vmin = rho_min if rho_min is not None else max(np.nanpercentile(rho, 2), 1e-2)
    vmax = rho_max if rho_max is not None else np.nanpercentile(rho, 98)
    vmin = max(vmin, 1e-2)
    if vmax <= vmin:
        vmax = vmin * 10

    fig, (ax_rms, ax) = plt.subplots(
        2, 1, figsize=(12, 6.5), height_ratios=[1, 8], sharex=True,
        gridspec_kw={"hspace": 0.06},
    )

    # assemble every layer cell as one polygon; a single PolyCollection draws
    # them all in one artist while each column still hangs from its own elevation
    verts, vals = [], []
    for j in range(n):
        z_top = elev[j] - z_edges_rel[:-1]
        z_bot = elev[j] - z_edges_rel[1:]
        for k in range(n_layers):
            verts.append([
                (edges[j], z_bot[k]), (edges[j + 1], z_bot[k]),
                (edges[j + 1], z_top[k]), (edges[j], z_top[k]),
            ])
            vals.append(rho[k, j])
    pc = PolyCollection(verts, array=np.array(vals),
                        norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cmap)
    ax.add_collection(pc)
    ax.set_ylim(float((elev - z_edges_rel[-1]).min()), float(elev.max()) + 5)

    ax.plot(dist, elev, color="k", lw=1.2)              # ground surface

    # #12: fade below the depth of investigation and draw the DOI line
    doi = np.array([s.doi_m if s.doi_m is not None else np.nan for s in S])
    if np.isfinite(doi).any():
        doi_elev = elev - doi
        ax.fill_between(dist, doi_elev, ax.get_ylim()[0],
                        color="white", alpha=0.55, step="mid", zorder=2.5)
        ax.plot(dist, doi_elev, color="0.25", lw=0.9, ls=":",
                zorder=3, label="depth of investigation")
        ax.legend(loc="lower right", fontsize=7, framealpha=0.7)

    ax.set_xlabel("Distance along line (m)")
    ax.set_ylabel("Elevation (m, GPS/ellipsoid)")   # #20: not orthometric
    ax.set_xlim(edges[0], edges[-1])

    cb = fig.colorbar(pc, ax=ax, pad=0.01, aspect=30)
    cb.set_label("Resistivity (Ω·m)")

    # chi misfit strip (chi ~ 1 = fit to errors)
    ax_rms.bar(dist, chi, width=np.diff(edges), color="0.5")
    ax_rms.axhline(1.0, color="tab:red", lw=0.8, ls="--")
    ax_rms.set_ylabel("chi", fontsize=8)
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

    # stairs plot of the model (shared basal-cell rule, #39)
    from .invert import basal_depth_bottom
    z = np.concatenate([depths, [basal_depth_bottom(depths)]])
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
