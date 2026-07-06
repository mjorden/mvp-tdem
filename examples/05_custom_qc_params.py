"""
QC parameter tuning — compare default vs. custom settings.

Surveys flown at non-standard altitudes, over rugged terrain, or with noisy
receivers need QC thresholds tailored to the acquisition conditions. This
example shows how to adjust each check and see the effect on flag counts.

The parameters demonstrated here are exaggerated to show the mechanism;
real adjustments are usually smaller.

Run from the repo root:

    python examples/05_custom_qc_params.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from tdem.load import load_line, load_survey
from tdem.qc import good_soundings, run_qc

REPO = Path(__file__).parent.parent

df, config = load_survey(
    REPO / "data/synthetic_survey.csv",
    REPO / "configs/example.json",
)
df_line = load_line(df, 2000)

print("QC parameter comparison — Line 2000")
print("=" * 60)

# ---------------------------------------------------------------------------
# Helper: summarise one QC run
# ---------------------------------------------------------------------------
_LABELS = {
    "_qc_neg_early":   "Negative early gates",
    "_qc_alt_low":     "Altitude too low",
    "_qc_alt_high":    "Altitude too high",
    "_qc_dem_mismatch":"DEM/GPS mismatch",
    "_qc_spike":       "Along-line spike",
    "_qc_nonmono":     "Non-monotonic decay",
}

def summarise(df_qc: pd.DataFrame, label: str) -> None:
    flag_cols = [c for c in df_qc.columns
                 if c.startswith("_qc_") and not c.startswith("_qc_gate_")]
    n_total   = len(df_qc)
    n_flagged = int(df_qc["sounding_mask"].sum())
    print(f"\n{label}")
    print(f"  {n_flagged}/{n_total} soundings flagged ({100*n_flagged/n_total:.1f}%)")
    for col in flag_cols:
        n = int(df_qc[col].sum())
        if n:
            print(f"  ├ {_LABELS.get(col, col)}: {n}")
    n_good = len(good_soundings(df_qc))
    print(f"  → {n_good} soundings passed to inversion")


# ---------------------------------------------------------------------------
# 1. Default parameters
# ---------------------------------------------------------------------------
df_default = run_qc(df_line, config)
summarise(df_default, "Default parameters")

# ---------------------------------------------------------------------------
# 2. Mountain / low-altitude survey
#    Bird must fly closer to terrain → lower alt_min_m.
#    Also tighten despike to catch turbulence-induced spikes.
# ---------------------------------------------------------------------------
df_mountain = run_qc(
    df_line, config,
    alt_min_m=5.0,          # allow bird down to 5 m AGL (default 10)
    alt_max_m=60.0,         # still reject very high passes
    despike_threshold=3.0,  # tighter: flag at 3 MAD instead of 4
)
summarise(df_mountain, "Mountain survey (low alt, tighter despike)")

# ---------------------------------------------------------------------------
# 3. Noisy receiver / late-gate-dominated survey
#    Relax monotonicity (late-time negatives are geophysically real in some
#    settings) and widen the despike window to smooth over longer-wavelength
#    cultural contamination.
# ---------------------------------------------------------------------------
df_noisy = run_qc(
    df_line, config,
    mono_n_early=4,          # check only the first 4 gates for monotonicity (default 8)
    mono_max_reversals=2,    # allow up to 2 reversals before flagging (default 1)
    despike_window=7,        # wider lateral filter window (default 5)
    despike_min_gates=5,     # require more simultaneous deviating gates (default 3)
)
summarise(df_noisy, "Noisy receiver (relaxed monotonicity, wider despike)")

# ---------------------------------------------------------------------------
# 4. Very permissive — pass almost everything to the inversion.
#    Useful when you want the inversion to see all data and decide for itself,
#    e.g. during exploratory data quality assessment.
# ---------------------------------------------------------------------------
df_permissive = run_qc(
    df_line, config,
    alt_min_m=0.0,
    alt_max_m=200.0,
    despike_threshold=10.0,
    despike_min_gates=10,
    mono_max_reversals=5,
)
summarise(df_permissive, "Permissive (minimal flagging)")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Summary: soundings passed to inversion")
print("-" * 40)
n = len(df_line)
for label, df_qc in [
    ("Default",      df_default),
    ("Mountain",     df_mountain),
    ("Noisy recv.",  df_noisy),
    ("Permissive",   df_permissive),
]:
    n_good = len(good_soundings(df_qc))
    print(f"  {label:<14}  {n_good}/{n}  ({100*n_good/n:.0f}%)")

print("""
Notes:
  · run_qc() never drops rows — it only adds flag columns.
  · Flagged soundings are skipped by invert_line() via sounding_mask.
  · good_soundings(df) returns the unflagged subset when you need it.
  · good_gate_array(df) returns the gate data with flagged values set to NaN.
""")
