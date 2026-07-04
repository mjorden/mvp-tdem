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

tab_load, tab_qc, tab_section, tab_decays, tab_download = st.tabs(
    ["Load", "QC", "Section", "Decays", "Download"]
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
