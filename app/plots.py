"""
Plotly figure builders for the TDEM web interface.
Mirrors the matplotlib outputs in tdem/visualize.py but interactive.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tdem.invert import LineResult
from tdem.load import gate_columns


# ---------------------------------------------------------------------------
# Resistivity section
# ---------------------------------------------------------------------------

def plotly_section(
    result: LineResult,
    *,
    rho_min: float | None = None,
    rho_max: float | None = None,
    max_depth: float | None = None,
    n_dist: int = 400,
    n_elev: int = 200,
) -> go.Figure:
    """
    Interactive resistivity cross-section rasterized onto a regular grid.

    Returns a Plotly figure with a heatmap (log resistivity, turbo_r palette)
    and a chi misfit strip above.
    """
    S = result.soundings
    if not S:
        return go.Figure()

    n = len(S)
    east = np.array([s.easting for s in S])
    north = np.array([s.northing for s in S])
    dx = np.diff(east, prepend=east[0])
    dy = np.diff(north, prepend=north[0])
    dist = np.cumsum(np.hypot(dx, dy))

    edges = np.empty(n + 1)
    if n > 1:
        edges[1:-1] = 0.5 * (dist[:-1] + dist[1:])
        edges[0] = dist[0] - (edges[1] - dist[0])
        edges[-1] = dist[-1] + (dist[-1] - edges[-2])
    else:
        edges[:] = [dist[0] - 25.0, dist[0] + 25.0]

    depths = S[0].depths
    bottom = depths[-1] * 1.4
    z_edges_rel = np.concatenate([depths, [bottom]])
    if max_depth is not None:
        z_edges_rel = np.minimum(z_edges_rel, max_depth)

    rho_all = np.column_stack([s.rho for s in S])   # (n_layers, n_soundings)
    elev = np.array([s.elevation for s in S])
    chi = np.array([s.chi for s in S])
    n_layers = len(depths)

    vmin = rho_min if rho_min else max(float(np.nanpercentile(rho_all, 2)), 1e-2)
    vmax = rho_max if rho_max else float(np.nanpercentile(rho_all, 98))
    if vmax <= vmin:
        vmax = vmin * 10

    # Rasterize onto a regular (distance, elevation) grid
    dist_ax = np.linspace(edges[0], edges[-1], n_dist)
    elev_ax = np.linspace(min(elev) - float(z_edges_rel[-1]), max(elev) + 5, n_elev)
    grid = np.full((n_elev, n_dist), np.nan)

    for di, d in enumerate(dist_ax):
        j = int(np.argmin(np.abs(dist - d)))
        for zi, z in enumerate(elev_ax):
            depth_below = elev[j] - z
            if depth_below < 0 or depth_below >= z_edges_rel[-1]:
                continue
            k = int(np.searchsorted(z_edges_rel[1:], depth_below))
            k = min(k, n_layers - 1)
            rho_val = rho_all[k, j]
            if rho_val > 0:
                grid[zi, di] = np.log10(rho_val)

    log_vmin, log_vmax = np.log10(vmin), np.log10(vmax)

    # Turbo_r colorscale approximated as reversed turbo
    colorscale = "turbo_r"

    # Tick values for the colorbar (decades)
    decade_min = int(np.floor(log_vmin))
    decade_max = int(np.ceil(log_vmax))
    tick_vals = list(range(decade_min, decade_max + 1))
    tick_text = [f"10<sup>{v}</sup>" for v in tick_vals]

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.15, 0.85],
        shared_xaxes=True,
        vertical_spacing=0.04,
    )

    # Chi misfit strip
    fig.add_trace(
        go.Bar(
            x=dist,
            y=chi,
            marker_color="gray",
            name="chi misfit",
            hovertemplate="dist=%{x:.0f} m<br>chi=%{y:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="red", row=1, col=1)

    # Resistivity heatmap
    fig.add_trace(
        go.Heatmap(
            z=grid,
            x=dist_ax,
            y=elev_ax,
            zmin=log_vmin,
            zmax=log_vmax,
            colorscale=colorscale,
            colorbar=dict(
                title="Ω·m",
                tickvals=tick_vals,
                ticktext=tick_text,
                len=0.85,
                y=0.4,
            ),
            hovertemplate=(
                "dist=%{x:.0f} m<br>elev=%{y:.0f} m<br>"
                "ρ=10<sup>%{z:.2f}</sup> Ω·m<extra></extra>"
            ),
            name="resistivity",
        ),
        row=2, col=1,
    )

    # Ground surface line
    fig.add_trace(
        go.Scatter(
            x=dist,
            y=elev,
            mode="lines",
            line=dict(color="black", width=1.5),
            name="ground surface",
            hoverinfo="skip",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        title=f"Line {result.line} — stitched 1D resistivity section",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=60, b=40),
    )
    fig.update_yaxes(title_text="chi", row=1, col=1)
    fig.update_xaxes(title_text="Distance along line (m)", row=2, col=1)
    fig.update_yaxes(title_text="Elevation (m)", row=2, col=1)

    return fig


# ---------------------------------------------------------------------------
# Decay curves
# ---------------------------------------------------------------------------

def plotly_decays(
    df_line,
    gate_times_ms: list[float],
    *,
    qc_df=None,
) -> go.Figure:
    """
    All observed decay curves for a line, coloured by along-line position.
    Flagged soundings (if qc_df provided) are drawn dashed in red.
    """
    cols = gate_columns(df_line)
    t = np.asarray(gate_times_ms, dtype=float)
    data = df_line[cols].to_numpy(dtype=float)
    n = len(df_line)

    flagged = np.zeros(n, dtype=bool)
    if qc_df is not None and "sounding_mask" in qc_df.columns:
        flagged = qc_df["sounding_mask"].to_numpy()

    # Map along-line position to a colour using viridis
    import plotly.colors as pc
    colors = pc.sample_colorscale("viridis", [i / max(n - 1, 1) for i in range(n)])

    fig = go.Figure()
    for i in range(n):
        y = np.abs(data[i])
        valid = np.isfinite(y) & (y > 0)
        if not valid.any():
            continue
        fig.add_trace(
            go.Scatter(
                x=t[valid],
                y=y[valid],
                mode="lines",
                line=dict(
                    color="rgba(200,0,0,0.5)" if flagged[i] else colors[i],
                    dash="dash" if flagged[i] else "solid",
                    width=0.8,
                ),
                name=f"snd {i}",
                showlegend=False,
                hovertemplate=f"sounding {i}<br>t=%{{x:.3f}} ms<br>|dB/dt|=%{{y:.2e}}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Observed decay curves (colour = along-line position; red dashed = QC-flagged)",
        xaxis=dict(title="Time (ms)", type="log"),
        yaxis=dict(title="|dB/dt|  (V/(A·m⁴))", type="log"),
        height=450,
        margin=dict(t=50, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Single-sounding fit
# ---------------------------------------------------------------------------

def plotly_sounding_fit(
    d_obs: np.ndarray,
    d_pred: np.ndarray,
    gate_times_ms: list[float],
    rho: np.ndarray,
    depths: np.ndarray,
    title: str = "",
) -> go.Figure:
    """Observed vs predicted decay + recovered 1D model — two-panel diagnostic."""
    t = np.asarray(gate_times_ms, dtype=float)
    use = np.isfinite(d_obs) & (d_obs > 0)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Decay fit", "1D model"))

    fig.add_trace(
        go.Scatter(
            x=t[use], y=d_obs[use],
            mode="markers",
            marker=dict(color="black", size=5),
            name="observed",
            hovertemplate="t=%{x:.3f} ms<br>obs=%{y:.2e}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t, y=d_pred,
            mode="lines",
            line=dict(color="red", width=1.5),
            name="predicted",
            hovertemplate="t=%{x:.3f} ms<br>pred=%{y:.2e}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Staircase model
    z_edges = np.concatenate([depths, [depths[-1] * 1.4]])
    stair_rho, stair_z = [], []
    for k in range(len(rho)):
        stair_rho += [rho[k], rho[k]]
        stair_z += [z_edges[k], z_edges[k + 1]]

    fig.add_trace(
        go.Scatter(
            x=stair_rho, y=stair_z,
            mode="lines",
            line=dict(color="steelblue", width=1.5),
            name="model",
            hovertemplate="ρ=%{x:.1f} Ω·m<br>depth=%{y:.0f} m<extra></extra>",
        ),
        row=1, col=2,
    )

    fig.update_xaxes(type="log", title_text="Time (ms)", row=1, col=1)
    fig.update_yaxes(type="log", title_text="|dB/dt|", row=1, col=1)
    fig.update_xaxes(type="log", title_text="Resistivity (Ω·m)", row=1, col=2)
    fig.update_yaxes(autorange="reversed", title_text="Depth (m)", row=1, col=2)
    fig.update_layout(title=title, height=420, margin=dict(t=60, b=40))

    return fig
