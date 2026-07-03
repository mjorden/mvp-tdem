"""
End-to-end library-API walkthrough: load → QC → invert → plot one flight line.

This is the same flow as scripts/process_line.py, but step by step so you can
inspect the intermediate objects. Uses the checked-in synthetic survey.

Run from the repo root:

    python examples/01_end_to_end.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tdem.load import load_survey, load_line, gate_columns
from tdem.qc import run_qc, good_soundings
from tdem.forward import forward_from_config
from tdem.invert import invert_line
from tdem.visualize import plot_section, plot_decays

REPO = Path(__file__).parent.parent
OUT = REPO / "output" / "examples"

# --- 1. Load: CSV + JSON sidecar → tidy DataFrame ---------------------------
df, config = load_survey(REPO / "data/synthetic_survey.csv", REPO / "configs/example.json")
print(f"{len(df)} soundings, lines: {sorted(df['line'].unique())}")
print(f"gate columns: {gate_columns(df)[:3]} ... ({len(gate_columns(df))} total)")

# --- 2. QC: adds _qc_* flag columns and sounding_mask; drops nothing --------
df = run_qc(df, config)
print(f"{len(good_soundings(df))} of {len(df)} soundings pass all QC checks")

# --- 3. Pick one line and plot its raw decays (quick QC view) ---------------
df_line = load_line(df, 2000)
plot_decays(df_line, config["gate_times_ms"], OUT / "line_2000_decays.png",
            title="Line 2000 — observed decays")

# --- 4. Invert: per-sounding 1D, warm-started along the line ----------------
# forward_from_config builds one TDEMForward for the whole survey; reuse it
# across lines so the per-bird-height simulation cache pays off.
fwd = forward_from_config(config)
result = invert_line(df_line, config, fwd=fwd)

# --- 5. Outputs: stitched section PNG + long-format model CSV ---------------
plot_section(result, OUT / "line_2000_section.png")
model = result.to_frame()
model.to_csv(OUT / "line_2000_model.csv", index=False)

# The synthetic survey has a 5 Ω·m conductor at 24–77 m depth between
# northing 4901500–4902500 — it should show up as a red (conductive) lens
# in the section, against the 200 Ω·m background.
depth_win = (model.depth_top > 24) & (model.depth_top < 77)
on_body = model.northing.between(4901500, 4902500)
print(f"median rho at 24–77 m depth: "
      f"{model[depth_win & on_body].rho.median():.0f} ohm-m over the conductor (true 5), "
      f"{model[depth_win & ~on_body].rho.median():.0f} ohm-m off it (true 200)")
print(f"outputs in {OUT}")
