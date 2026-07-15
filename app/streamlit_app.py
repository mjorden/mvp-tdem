"""
TDEM Survey Processor — Streamlit web interface.

Launch with:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Ensure the project root is on the path when run from the app/ dir
sys.path.insert(0, str(Path(__file__).parent.parent))

from tdem.load import load_survey, load_line, gate_columns
from tdem.qc import run_qc
from tdem.invert import invert_line
from app.plots import plotly_section, plotly_decays, plotly_sounding_fit


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="TDEM Survey Processor",
    page_icon="📡",
    layout="wide",
)

st.title("📡 TDEM Survey Processor")
st.caption("Load → QC → Invert → Visualize airborne TDEM data in the browser.")


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _clear_downstream(from_step: str) -> None:
    """Invalidate cached results when upstream data changes."""
    steps = ["loaded", "qc", "inversion"]
    idx = steps.index(from_step)
    for s in steps[idx:]:
        for key in [f"{s}_df", f"{s}_config", f"{s}_result"]:
            st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# Sidebar — data upload
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Data")
    csv_file = st.file_uploader("Survey CSV", type=["csv", "xyz", "txt"])
    json_file = st.file_uploader("Config JSON sidecar", type=["json"])

    if csv_file and json_file:
        if st.button("Load data", type="primary"):
            _clear_downstream("loaded")
            with st.spinner("Loading…"):
                try:
                    with (
                        tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf_csv,
                        tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf_json,
                    ):
                        tf_csv.write(csv_file.read())
                        tf_json.write(json_file.read())
                        csv_path = tf_csv.name
                        json_path = tf_json.name

                    df, config = load_survey(csv_path, json_path)
                    st.session_state["loaded_df"] = df
                    st.session_state["loaded_config"] = config
                    st.success(f"Loaded {len(df):,} soundings.")
                except Exception as exc:
                    st.error(f"Load failed: {exc}")

    # Line selector — only after data is loaded
    if "loaded_df" in st.session_state:
        df_all = st.session_state["loaded_df"]
        lines = sorted(df_all["line"].unique())
        selected_line = st.selectbox("Flight line", lines)
        st.session_state["selected_line"] = selected_line

        st.divider()
        st.header("Processing")

        if st.button("Run QC"):
            _clear_downstream("qc")
            with st.spinner("Running QC…"):
                try:
                    df_line = load_line(df_all, selected_line)
                    config = st.session_state["loaded_config"]
                    qc_df = run_qc(df_line, config)
                    st.session_state["qc_df"] = qc_df
                    st.session_state["qc_line"] = selected_line
                    n_flagged = int(qc_df["sounding_mask"].sum())
                    st.success(f"{n_flagged}/{len(qc_df)} soundings flagged.")
                except Exception as exc:
                    st.error(f"QC failed: {exc}")

        run_inv_disabled = "qc_df" not in st.session_state
        if st.button("Run inversion", disabled=run_inv_disabled):
            with st.spinner("Inverting (this may take a minute)…"):
                try:
                    qc_df = st.session_state["qc_df"]
                    config = st.session_state["loaded_config"]
                    result = invert_line(qc_df, config, verbose=False)
                    st.session_state["inversion_result"] = result
                    st.session_state["inversion_line"] = selected_line
                    n_ok = sum(s.converged for s in result.soundings)
                    st.success(
                        f"Inverted {len(result.soundings)} soundings "
                        f"({n_ok} converged)."
                    )
                except Exception as exc:
                    st.error(f"Inversion failed: {exc}")


# ---------------------------------------------------------------------------
# Main panel — tabs
# ---------------------------------------------------------------------------

tab_load, tab_qc, tab_section, tab_decays, tab_download, tab_help = st.tabs(
    ["Load", "QC", "Section", "Decays", "Download", "Help"]
)


# ── Load tab ────────────────────────────────────────────────────────────────

with tab_load:
    if "loaded_df" not in st.session_state:
        st.info("Upload a CSV and JSON sidecar in the sidebar, then click **Load data**.")
    else:
        df_all = st.session_state["loaded_df"]
        config = st.session_state["loaded_config"]
        lines = sorted(df_all["line"].unique())
        gate_cols = gate_columns(df_all)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Soundings", f"{len(df_all):,}")
        col2.metric("Lines", len(lines))
        col3.metric("Gates", len(gate_cols))
        col4.metric("Gate time range",
                    f"{config['gate_times_ms'][0]:.2f}–"
                    f"{config['gate_times_ms'][-1]:.2f} ms")

        st.subheader("Lines")
        line_summary = (
            df_all.groupby("line")
            .agg(n_soundings=("fiducial", "count"))
            .reset_index()
        )
        st.dataframe(line_summary, use_container_width=True, hide_index=True)

        st.subheader("System parameters")
        sys_params = config.get("system", {})
        st.json(sys_params, expanded=False)


# ── QC tab ──────────────────────────────────────────────────────────────────

with tab_qc:
    if "qc_df" not in st.session_state:
        st.info("Select a line and click **Run QC** in the sidebar.")
    else:
        qc_df = st.session_state["qc_df"]
        config = st.session_state["loaded_config"]

        flag_cols = [c for c in qc_df.columns
                     if c.startswith("_qc_") and not c.startswith("_qc_gate_")]
        gate_flag_cols = [c for c in qc_df.columns if c.startswith("_qc_gate_")]

        n_total = len(qc_df)
        n_flagged = int(qc_df["sounding_mask"].sum())

        col1, col2, col3 = st.columns(3)
        col1.metric("Soundings", n_total)
        col2.metric("Flagged", n_flagged,
                    delta=f"{100*n_flagged/n_total:.1f}%",
                    delta_color="inverse")
        n_gate_bad = int(qc_df[gate_flag_cols].sum().sum()) if gate_flag_cols else 0
        col3.metric("Gate values below noise floor", n_gate_bad)

        st.subheader("Per-check flag counts")
        _LABEL = {
            "_qc_neg_early": "Negative early gates",
            "_qc_alt_low": "Altitude too low",
            "_qc_alt_high": "Altitude too high",
            "_qc_dem_mismatch": "DEM/GPS mismatch",
            "_qc_spike": "Along-line spike",
            "_qc_nonmono": "Non-monotonic decay",
        }
        rows = []
        for col in flag_cols:
            n = int(qc_df[col].sum())
            rows.append({
                "Check": _LABEL.get(col, col),
                "Flagged": n,
                "Pct": f"{100*n/n_total:.1f}%",
            })
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Sounding flag map")
        # Show flag booleans as a colour-coded table
        display = qc_df[["fiducial"] + flag_cols + ["sounding_mask"]].copy()
        display.columns = (
            ["Fiducial"]
            + [_LABEL.get(c, c) for c in flag_cols]
            + ["Any flag"]
        )

        def _colour_bool(val):
            if val is True or val == 1:
                return "background-color: #ffcccc"
            if val is False or val == 0:
                return "background-color: #ccffcc"
            return ""

        st.dataframe(
            display.style.applymap(_colour_bool,
                                   subset=display.columns[1:]),
            use_container_width=True,
            height=350,
        )


# ── Section tab ──────────────────────────────────────────────────────────────

with tab_section:
    if "inversion_result" not in st.session_state:
        st.info("Run QC then click **Run inversion** in the sidebar.")
    else:
        result = st.session_state["inversion_result"]

        with st.expander("Display options", expanded=False):
            col1, col2, col3 = st.columns(3)
            rho_min_ui = col1.number_input("ρ min (Ω·m)", value=1.0, min_value=0.01)
            rho_max_ui = col2.number_input("ρ max (Ω·m)", value=10000.0, min_value=1.0)
            max_depth_ui = col3.number_input(
                "Max depth (m)", value=0.0,
                help="0 = auto (1.4 × deepest layer top)"
            )

        fig = plotly_section(
            result,
            rho_min=rho_min_ui if rho_min_ui > 0 else None,
            rho_max=rho_max_ui if rho_max_ui > 0 else None,
            max_depth=max_depth_ui if max_depth_ui > 0 else None,
        )
        st.plotly_chart(fig, use_container_width=True)

        if result.soundings:
            st.subheader("Sounding fit inspector")
            fids = [s.fiducial for s in result.soundings]
            sel_fid = st.selectbox("Fiducial", fids)
            idx = next(i for i, s in enumerate(result.soundings)
                       if s.fiducial == sel_fid)
            s = result.soundings[idx]
            config = st.session_state["loaded_config"]
            qc_df = st.session_state["qc_df"]
            gate_cols = gate_columns(qc_df)
            d_obs = qc_df.loc[qc_df["fiducial"] == sel_fid, gate_cols].to_numpy(float).ravel()

            from tdem.forward import forward_from_config
            fwd = forward_from_config(config)
            bird_h = float(qc_df.loc[qc_df["fiducial"] == sel_fid, "dem"].iloc[0])
            d_pred = fwd.predict(s.rho, bird_h)
            fit_fig = plotly_sounding_fit(
                d_obs, d_pred,
                config["gate_times_ms"],
                s.rho, s.depths,
                title=f"Fiducial {sel_fid}  chi={s.chi:.2f}  converged={s.converged}",
            )
            st.plotly_chart(fit_fig, use_container_width=True)


# ── Decays tab ───────────────────────────────────────────────────────────────

with tab_decays:
    if "qc_df" not in st.session_state:
        st.info("Run QC first to see decay curves.")
    else:
        qc_df = st.session_state["qc_df"]
        config = st.session_state["loaded_config"]
        fig = plotly_decays(qc_df, config["gate_times_ms"], qc_df=qc_df)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Colour = along-line position (viridis).  "
            "Red dashed = QC-flagged sounding.  "
            "Click legend entries to isolate traces."
        )


# ── Download tab ─────────────────────────────────────────────────────────────

with tab_download:
    if "inversion_result" not in st.session_state:
        st.info("Run an inversion first to enable downloads.")
    else:
        result = st.session_state["inversion_result"]
        df_model = result.to_frame()

        st.subheader("Model CSV")
        st.dataframe(df_model.head(50), use_container_width=True)

        csv_bytes = df_model.to_csv(index=False).encode()
        st.download_button(
            label="⬇ Download model CSV",
            data=csv_bytes,
            file_name=f"line_{result.line}_model.csv",
            mime="text/csv",
        )

        st.subheader("Section PNG")
        try:
            import plotly.io as pio
            fig = plotly_section(result)
            png_bytes = pio.to_image(fig, format="png", width=1600, height=700, scale=2)
            st.download_button(
                label="⬇ Download section PNG",
                data=png_bytes,
                file_name=f"line_{result.line}_section.png",
                mime="image/png",
            )
        except Exception:
            st.info(
                "PNG export requires `kaleido`. "
                "Install it with: `pip install kaleido`"
            )


# ── Help tab ─────────────────────────────────────────────────────────────────

with tab_help:
    st.header("How to use this app")

    st.markdown("""
This app runs the full airborne TDEM processing pipeline in your browser:
**upload data → quality control → 1D inversion → interactive results**.

No local Python installation is needed once the app is running. Results can be
downloaded as CSV or PNG.
""")

    st.divider()

    # ── Workflow ──────────────────────────────────────────────────────────────
    with st.expander("Step-by-step workflow", expanded=True):
        st.markdown("""
**1. Upload files (sidebar)**

You need two files per survey:

| File | Format | What it contains |
|------|--------|-----------------|
| Survey CSV | `.csv`, `.xyz`, or `.txt` | One row per sounding: coordinates, bird height, and one column per time gate (dB/dt in V/(A·m⁴)) |
| Config JSON | `.json` | Column name mapping, system parameters, gate times, and inversion defaults |

The JSON sidecar tells the pipeline which CSV column is which — the same CSV
can be processed with different sidecars for different system configurations.

**2. Click "Load data"**

The loader standardises column names, replaces dummy fill values (−9999 etc.)
with NaN, and validates that gate times are consistent with the system
parameters. Any problems are shown as an error message.

**3. Select a flight line and click "Run QC"**

Six QC checks are applied (see the QC tab and the *QC checks* section below).
Nothing is dropped — every sounding is flagged or not flagged. The inversion
skips flagged soundings automatically.

**4. Click "Run inversion"**

Each sounding on the selected line is inverted independently for a layered
resistivity model, then stitched into a 2D section. This can take
30 seconds to a few minutes depending on line length.

**5. Explore results**

- **Section tab** — interactive resistivity cross-section; click the Sounding
  Fit Inspector to see observed vs. predicted decay for any fiducial
- **Decays tab** — all observed decay curves; red dashed = QC-flagged
- **Download tab** — model CSV and section PNG
""")

    # ── Input format ──────────────────────────────────────────────────────────
    with st.expander("Input file format"):
        st.markdown("""
### Survey CSV

A flat ASCII table with one row per sounding. Accepted separators: comma or
whitespace. Geosoft XYZ format (lines prefixed with `/`, `Line NNNN` records)
is also supported.

**Required columns** (names configured in the JSON sidecar):

| Standard name | Typical CSV name | Description |
|---------------|-----------------|-------------|
| `easting` | `Easting` | UTM easting, m |
| `northing` | `Northing` | UTM northing, m |
| `elevation` | `Elevation` | GPS ellipsoid elevation of the bird, m |
| `dem` | `DEM` | Radar altimeter — bird height above ground (m AGL) |

**Gate columns** — one column per time gate. Three naming conventions are
supported via `column_map.sfz_format` in the JSON sidecar:

| Format | Example columns |
|--------|----------------|
| `bracket` | `SFz[0]`, `SFz[1]`, … `SFz[19]` |
| `underscore` | `SFz_00`, `SFz_01`, … `SFz_19` |
| `zero_padded` | `SFz00`, `SFz01`, … `SFz19` |

Values must be in **V/(A·m⁴)** — moment-normalised dB/dt. If your data
are in different units, apply the conversion before uploading.

### JSON sidecar

See `configs/README.md` in the repository for the full schema. The minimum
required sections are `column_map`, `system` (with `system_noise_floor`),
`gate_times_ms`, and `inversion`. Use `configs/example.json` as a starting
template.

**Critical constraint:** `len(gate_times_ms)` must equal
`column_map.sfz_n`, and the last gate time must fall inside the
off-time window `1/(2 × tx_frequency_hz) − tx_on_time_us/1000`.
Both are validated at load time.
""")

    # ── QC checks ─────────────────────────────────────────────────────────────
    with st.expander("QC checks — what each flag means"):
        st.markdown("""
QC flags are boolean columns added to the data frame. `True` = problem detected.
**No data is ever dropped** — flags guide the inversion and let you investigate.

| Flag | Column | What triggers it | Typical cause |
|------|--------|-----------------|---------------|
| Negative early gates | `_qc_neg_early` | Any gate in the first half has a negative value | Cultural EM, powerline, or instrument artefact |
| Altitude too low | `_qc_alt_low` | Bird height (DEM) < `alt_min_m` (default 10 m) | Terrain clearance issue; footprint distortion |
| Altitude too high | `_qc_alt_high` | Bird height > `alt_max_m` (default 80 m) | Weak late-time signal; increased noise |
| DEM/GPS mismatch | `_qc_dem_mismatch` | Derived ground elevation jumps > 15 m between adjacent soundings | GPS dropout, radar canopy lock-on |
| Along-line spike | `_qc_spike` | ≥ 3 gates simultaneously deviate from the local median by > 4 MAD | Cultural source (powerline, fence, vehicle) |
| Non-monotonic decay | `_qc_nonmono` | > 1 reversal in the first 8 early gates (log-space) | Powerline coupling, cultural EM |
| Noise floor (per-gate) | `_qc_gate_NN` | Gate value ≤ `system_noise_floor` | Late-time noise; below detection limit |

The combined flag `sounding_mask` is the logical OR of all per-sounding flags.
Soundings with `sounding_mask = True` are skipped by the inversion.

**Per-gate noise-floor flags** (`_qc_gate_00` … `_qc_gate_NN`) are gated
separately from the sounding-level flags. The inversion excludes individual
bad gates from the misfit rather than skipping the whole sounding.
""")

    # ── Inversion parameters ──────────────────────────────────────────────────
    with st.expander("Inversion parameters — what they mean"):
        st.markdown("""
The inversion minimises:

```
‖ W_d (ln d_obs − ln d_pred(m)) ‖²
  + α_s ‖ m − m_ref ‖²
  + α_z ‖ W_z D m ‖²
```

where `m = log₁₀(resistivity)` per layer, `W_d` weights by data error,
and `D` is a first-difference operator scaled by layer thickness.

| Parameter | JSON key | Effect |
|-----------|----------|--------|
| **α_s** (alpha_s) | `inversion.alpha_s` | Damps the model toward `rho_initial`. Increase if the background is well-constrained. Default: 1e-4 |
| **α_z** (alpha_z) | `inversion.alpha_z` | Vertical first-difference smoothing. Higher = smoother model; lower = sharper layer boundaries. Default: 1.0 |
| **rho_initial** | `inversion.rho_initial` | Starting model and reference for α_s damping (Ω·m). Default: 100 |
| **rho_min / rho_max** | `inversion.rho_min/max` | Hard bounds on recovered resistivity (Ω·m). Default: 1–10 000 |
| **chi_target** | `inversion.chi_target` | Occam loop stops when chi ≤ this. chi = 1 means data fit to within assigned errors. Default: 1.0 |
| **rel_error** | `inversion.rel_error` | Relative data error fraction used in W_d. Default: 0.05 (5%) |
| **n_layers** | `inversion.n_layers` | Number of layers including the basal half-space. Default: 36 |
| **depth_max_m** | `inversion.depth_max_m` | Depth where the basal half-space starts (m). Should be ~1.5× the estimated depth of investigation. Default: 600 |

### Interpreting chi (misfit)

- **chi ≈ 1** — data fit to within assigned errors. Healthy.
- **chi < 0.5** — over-fitting. The model is chasing noise. Consider increasing `rel_error` or `alpha_z`.
- **chi > 2** — under-fitting. The model cannot explain the data. Possible causes: too-high `alpha_z`, too-tight `rho_min/max`, data quality issues.

The chi strip at the top of the Section plot is the per-sounding misfit.
The red dashed line at chi = 1 is the target.

### Warm-start stitching

Soundings are inverted in along-line order. Each sounding is initialised from
its neighbour's recovered model (warm start) rather than from the flat
`rho_initial`. This produces a smoother section at low cost. A warm-started
fit with chi > 2 is automatically retried from the cold reference model; the
better of the two is kept.
""")

    # ── Reading the section ───────────────────────────────────────────────────
    with st.expander("Reading the section plot"):
        st.markdown("""
### Colour convention

The section uses the **reversed turbo** palette — a standard EM convention:

| Colour | Resistivity | Interpretation |
|--------|-------------|----------------|
| Red / orange | Low (conductive) | Clay, graphite, sulphides, saline water, conductor |
| Yellow / green | Intermediate | Mixed lithology |
| Blue / purple | High (resistive) | Fresh crystalline rock, dry sand, limestone |

Hover over any cell to see the exact resistivity, elevation, and distance.

### Chi strip (top panel)

The bar chart above the section shows the per-sounding chi misfit. The red
dashed line is chi = 1 (target). Bars above the line indicate soundings where
the model could not fit the data to within assigned errors — usually due to
QC-flagged gates, severe cultural noise, or geometrically complex geology.

### Sounding Fit Inspector

Select any fiducial from the dropdown to see:

- **Left panel** — observed (black dots) vs. predicted (red line) decay curve.
  A good fit means the points lie on the line across all gate times.
- **Right panel** — recovered 1D resistivity model. The staircase is the model;
  remember Occam smoothing blurs sharp boundaries.

### Depth of Investigation (DOI)

The section is plotted to `depth_max_m` from the JSON sidecar. Resistivity
below the actual depth of investigation (DOI) is unconstrained and reflects
the prior (`rho_initial`). A rough guide to DOI:

```
DOI (m) ≈ 500 × √(ρ_background × t_last_gate)
```

where `ρ` is in Ω·m and `t` is in seconds. For a 200 Ω·m background and
14.6 ms last gate, DOI ≈ 340 m.
""")

    # ── Common issues ─────────────────────────────────────────────────────────
    with st.expander("Troubleshooting common issues"):
        st.markdown("""
**"Expected gate columns not found in CSV"**

The `sfz_prefix`, `sfz_n`, and `sfz_format` in the JSON sidecar don't match
your CSV column names. Open the CSV and check the exact gate column names,
then update the sidecar.

**"gate_times_ms has N entries but sfz_n = M"**

The number of entries in `gate_times_ms` must equal `column_map.sfz_n`.
Count your gate columns and update both fields to match.

**"Latest gate is outside the off-time window"**

The last gate time in `gate_times_ms` is ≥ 1/(2×f) − on_time. This is a
physically impossible gate — either the gate times are wrong or
`tx_frequency_hz` / `tx_on_time_us` are incorrect in the sidecar.

**All soundings are QC-flagged**

Check the flag breakdown in the QC tab. Common causes:
- `_qc_alt_high` everywhere: your survey altitude is > 80 m — increase `alt_max_m`
  in the QC defaults (currently only editable in code; see `tdem/qc.py`)
- `_qc_neg_early` everywhere: data may already be absolute-valued, or the
  moment normalisation sign convention is inverted

**Inversion fails or produces chi >> 1 everywhere**

- Verify the system geometry (`tx_geometry`, `tx_loop_radius_m`) matches the
  acquisition hardware
- Check that `system_noise_floor` is in the same units as the data (V/(A·m⁴))
- Try increasing `rel_error` to 0.10 to allow for larger data uncertainty

**PNG download shows "kaleido required"**

Install the optional Plotly image renderer: `pip install kaleido`.
It is included in the `[ui]` optional dependency group.
""")

    # ── Known limitations ─────────────────────────────────────────────────────
    with st.expander("⚠️ Known limitations — how to read results", expanded=False):
        st.markdown("""
This pipeline is a good **anomaly-finder** and a not-yet-trustworthy
**absolute-measurement instrument**. Keep three limits in mind:

1. **IP / chargeable ground is censored, not inverted.** The inversion drops
   negative late-time gates before fitting, so over clay-rich or chargeable
   cover, deep conductive features are suspect — the censoring biases
   surviving late gates high (spuriously conductive basements).
2. **Below-DOI cells show the prior, not a result.** The faded region of the
   section (and `below_doi` in the model CSV) marks cells whose resistivity is
   essentially the starting model bent by regularization. Do not pick
   "basement contacts" beneath the DOI line.
3. **Positions and depths lack layback/lever-arm correction.** On a real
   flight the bird trails ~20–30 m behind and ~15–25 m below the GPS antenna;
   adjacent lines flown in opposite directions can mis-tie by ~40–50 m, and
   the vertical offset shifts depth-to-conductor. Synthetic data is
   unaffected; real surveys need the correction first.

Also: chi ≈ 1 means "fit to assigned errors under the regularization," not a
strict statistical reduced-χ² — a handful of gates cannot uniquely constrain
30 layers.
""")

    st.divider()
    st.caption(
        "Source code: [github.com/mjorden/mvp-tdem](https://github.com/mjorden/mvp-tdem)  ·  "
        "Config schema: `configs/README.md`  ·  "
        "Examples: `examples/README.md`"
    )
