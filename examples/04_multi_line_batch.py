"""
Batch-process every line in the survey.

Builds one TDEMForward (shared across lines — the per-height sim cache pays
off) then loops over all lines: load → QC → invert → plot + CSV.  Finally
writes a merged multi-line model CSV that covers the full survey.

Run from the repo root:

    python examples/04_multi_line_batch.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from tdem.forward import forward_from_config
from tdem.invert import invert_line
from tdem.load import gate_columns, load_line, load_survey
from tdem.qc import run_qc
from tdem.visualize import plot_section

REPO = Path(__file__).parent.parent
OUT  = REPO / "output" / "examples"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load the full survey once
# ---------------------------------------------------------------------------
df, config = load_survey(
    REPO / "data/synthetic_survey.csv",
    REPO / "configs/example.json",
)
lines = sorted(df["line"].unique())
print(f"Survey: {len(df)} soundings across {len(lines)} lines: {lines}")
print(f"Gates: {len(gate_columns(df))}")

# ---------------------------------------------------------------------------
# Build the forward operator once — reuse across all lines.
# Simulations are cached per bird height (rounded to 0.1 m), so the first
# sounding on each line pays the construction cost; subsequent soundings at
# the same height are instant.
# ---------------------------------------------------------------------------
fwd = forward_from_config(config)

# ---------------------------------------------------------------------------
# Process each line
# ---------------------------------------------------------------------------
all_frames: list[pd.DataFrame] = []

for line_id in lines:
    print(f"\n── Line {line_id} ────────────────────────────────")

    # Slice and QC
    df_line = load_line(df, line_id)
    df_qc   = run_qc(df_line, config)

    n_flagged = int(df_qc["sounding_mask"].sum())
    print(f"  QC: {n_flagged}/{len(df_qc)} soundings flagged")

    # Invert (verbose=True shows per-sounding chi)
    result = invert_line(df_qc, config, fwd=fwd, verbose=False)

    n_ok = sum(s.converged for s in result.soundings)
    chis = [s.chi for s in result.soundings]
    median_chi = sorted(chis)[len(chis) // 2] if chis else float("nan")
    print(f"  Inversion: {len(result.soundings)} soundings, "
          f"{n_ok} converged, median chi={median_chi:.2f}")

    # Section plot
    fig_path = OUT / f"line_{line_id}_section.png"
    plot_section(result, fig_path)

    # Per-line model CSV
    model_df = result.to_frame()
    model_df.to_csv(OUT / f"line_{line_id}_model.csv", index=False)

    all_frames.append(model_df)

# ---------------------------------------------------------------------------
# Merge into a single survey-wide model CSV
# ---------------------------------------------------------------------------
merged = pd.concat(all_frames, ignore_index=True)
merged_path = OUT / "all_lines_model.csv"
merged.to_csv(merged_path, index=False)

print(f"\n{'─'*50}")
print(f"Survey-wide model: {len(merged)} rows ({len(merged.columns)} columns)")
print(f"Written to: {merged_path}")

# ---------------------------------------------------------------------------
# Quick sanity check: conductor recovery across lines
# ---------------------------------------------------------------------------
print("\nConductor recovery summary (depth 20–80 m, northing 4901500–4902500):")
depth_win  = (merged["depth_top"] > 20) & (merged["depth_top"] < 80)
on_body    = merged["northing"].between(4_901_500, 4_902_500)
off_body   = ~on_body

for line_id in lines:
    on_line = merged["line"] == line_id
    in_win  = merged[on_line & depth_win & on_body]["rho"]
    off_win = merged[on_line & depth_win & off_body]["rho"]
    print(f"  Line {line_id}: "
          f"conductor window median {in_win.median():.0f} Ω·m  "
          f"(background {off_win.median():.0f} Ω·m, true: 5 vs 200)")
