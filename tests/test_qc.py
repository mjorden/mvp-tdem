"""Unit tests for tdem/qc.py."""

import numpy as np
import pandas as pd
import pytest

from tdem.qc import (
    run_qc,
    good_soundings,
    good_gate_array,
    _noise_floor_flag,
    _negative_gate_flag,
    _altitude_flag,
    _dem_consistency_flag,
    _along_line_despike,
    _monotonicity_flag,
)
from tdem.load import gate_columns


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_GATES = 10

GATE_TIMES = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12]

NOISE_FLOOR = 1e-12


def _make_config(noise_floor=NOISE_FLOOR):
    return {
        "system": {"system_noise_floor": noise_floor},
        "gate_times_ms": GATE_TIMES,
        "column_map": {"sfz_n": N_GATES},
    }


def _make_sounding(n=20, rho=100.0, line=1000, add_spike_at=None, noise_scale=0.0):
    """Build a minimal DataFrame of synthetic soundings."""
    t     = np.array(GATE_TIMES) * 1e-3
    mu0   = 4 * np.pi * 1e-7
    sigma = 1.0 / rho
    decay = (sigma ** 1.5) * (mu0 ** 2.5) / (20 * np.pi ** 1.5) * t ** (-2.5)

    rows = []
    for i in range(n):
        row = {
            "line":      line,
            "fiducial":  1000 + i,
            "easting":   500000.0 + i * 50,
            "northing":  4900000.0,
            "elevation": 1450.0,
            "dem":       35.0,
        }
        signal = decay.copy()
        if noise_scale:
            rng = np.random.default_rng(i)
            signal = signal + noise_scale * signal * rng.standard_normal(N_GATES)
        for k, v in enumerate(signal):
            row[f"sfz_{k:02d}"] = max(v, 1e-15)
        rows.append(row)

    df = pd.DataFrame(rows)

    if add_spike_at is not None:
        gate_cols = gate_columns(df)
        for col in gate_cols:
            df.loc[add_spike_at, col] = df[col].max() * 1000

    return df


# ---------------------------------------------------------------------------
# noise_floor_flag
# ---------------------------------------------------------------------------

def test_noise_floor_flags_weak_late_gates():
    df        = _make_sounding(n=5)
    gate_cols = gate_columns(df)
    config    = _make_config(noise_floor=1e-6)  # high floor → late gates should be flagged
    df        = _noise_floor_flag(df, gate_cols, noise_floor=1e-6)

    flag_cols = [c for c in df.columns if c.startswith("_qc_gate_")]
    # At least one late gate should be flagged across all soundings
    assert df[flag_cols].any().any(), "Expected some gates flagged below noise floor"


def test_noise_floor_passes_strong_signal():
    df        = _make_sounding(n=5)
    gate_cols = gate_columns(df)
    df        = _noise_floor_flag(df, gate_cols, noise_floor=1e-30)  # impossibly low floor

    flag_cols = [c for c in df.columns if c.startswith("_qc_gate_")]
    assert not df[flag_cols].any().any(), "No gates should be flagged with noise floor=1e-30"


# ---------------------------------------------------------------------------
# negative_gate_flag
# ---------------------------------------------------------------------------

def test_negative_early_gate_flagged():
    df = _make_sounding(n=5)
    df.loc[2, "sfz_02"] = -1e-8   # inject negative into early gate
    df = _negative_gate_flag(df, gate_columns(df))
    assert df.loc[2, "_qc_neg_early"], "Sounding with negative early gate should be flagged"
    assert not df.loc[0, "_qc_neg_early"], "Clean sounding should not be flagged"


def test_negative_late_gate_not_flagged():
    df = _make_sounding(n=5)
    # Negative in the very last gate (late-time; physically possible) — should not trigger
    last = gate_columns(df)[-1]
    df.loc[0, last] = -1e-12
    df = _negative_gate_flag(df, gate_columns(df))
    assert not df.loc[0, "_qc_neg_early"]


# ---------------------------------------------------------------------------
# altitude_flag
# ---------------------------------------------------------------------------

def test_altitude_too_low():
    df = _make_sounding(n=3)
    df.loc[1, "dem"] = 5.0   # below alt_min_m=10
    df = _altitude_flag(df, alt_min_m=10.0, alt_max_m=80.0)
    assert df.loc[1, "_qc_alt_low"]
    assert not df.loc[0, "_qc_alt_low"]


def test_altitude_too_high():
    df = _make_sounding(n=3)
    df.loc[0, "dem"] = 120.0  # above alt_max_m=80
    df = _altitude_flag(df, alt_min_m=10.0, alt_max_m=80.0)
    assert df.loc[0, "_qc_alt_high"]
    assert not df.loc[1, "_qc_alt_high"]


# ---------------------------------------------------------------------------
# dem_consistency_flag
# ---------------------------------------------------------------------------

def test_gps_dropout_jump_flagged():
    """#3: abrupt jump in derived ground elevation → flagged."""
    df = _make_sounding(n=20)
    df.loc[10, "elevation"] = 1450.0 - 60.0   # 60 m GPS dropout at one sounding
    df = _dem_consistency_flag(df)
    assert df.loc[10, "_qc_dem_mismatch"]
    assert not df.loc[0, "_qc_dem_mismatch"]


def test_negative_ellipsoid_ground_not_flagged():
    """#3: uniformly negative ground ellipsoid height (coastal geoid) is valid."""
    df = _make_sounding(n=20)
    df["elevation"] = -5.0   # ground at -40 m ellipsoid height, bird at 35 m AGL
    df = _dem_consistency_flag(df)
    assert not df["_qc_dem_mismatch"].any(), \
        "Negative ellipsoid ground height must not be flagged (geoid undulation)"


# ---------------------------------------------------------------------------
# along_line_despike
# ---------------------------------------------------------------------------

def test_spike_detected():
    df = _make_sounding(n=20, add_spike_at=10)
    df = _along_line_despike(df, gate_columns(df), window=5, threshold=4.0, min_gates=3)
    assert df.loc[10, "_qc_spike"], "Injected spike sounding should be flagged"


def test_no_false_positives_clean_data():
    df = _make_sounding(n=20)
    df = _along_line_despike(df, gate_columns(df), window=5, threshold=4.0, min_gates=3)
    # Clean uniform decay — no soundings should be spiked
    assert df["_qc_spike"].sum() == 0, "No spikes expected in clean monotonic data"


# ---------------------------------------------------------------------------
# monotonicity_flag
# ---------------------------------------------------------------------------

def test_non_monotonic_flagged():
    df = _make_sounding(n=5)
    gate_cols = gate_columns(df)
    # Inject multiple reversals in early gates of sounding 0
    df.loc[0, gate_cols[1]] = df.loc[0, gate_cols[0]] * 10   # up
    df.loc[0, gate_cols[3]] = df.loc[0, gate_cols[2]] * 10   # up again
    df = _monotonicity_flag(df, gate_cols, n_early=8, max_reversals=1)
    assert df.loc[0, "_qc_nonmono"], "Non-monotonic sounding should be flagged"


def test_monotonic_not_flagged():
    df = _make_sounding(n=5)
    df = _monotonicity_flag(df, gate_columns(df), n_early=8, max_reversals=1)
    assert not df["_qc_nonmono"].any()


# ---------------------------------------------------------------------------
# run_qc / integration
# ---------------------------------------------------------------------------

def test_run_qc_returns_sounding_mask():
    df     = _make_sounding(n=20)
    config = _make_config()
    df_qc  = run_qc(df, config)
    assert "sounding_mask" in df_qc.columns
    assert df_qc["sounding_mask"].dtype == bool


def test_good_soundings_excludes_flagged():
    df = _make_sounding(n=20)
    df.loc[5, "dem"] = 5.0   # too low → will be flagged
    config  = _make_config()
    df_qc   = run_qc(df, config)
    df_good = good_soundings(df_qc)
    assert 5 not in df_good["fiducial"].values - 1000


def test_good_gate_array_shape():
    df     = _make_sounding(n=10)
    config = _make_config()
    df_qc  = run_qc(df, config)
    arr    = good_gate_array(df_qc)
    assert arr.shape == (10, N_GATES)


def test_good_gate_array_nan_at_noise_floor():
    df     = _make_sounding(n=5)
    config = _make_config(noise_floor=1e-6)   # high floor → late gates → NaN
    df_qc  = run_qc(df, config)
    arr    = good_gate_array(df_qc)
    assert np.any(np.isnan(arr)), "Flagged gates should be NaN in gate array"


def test_run_qc_non_range_index():
    """#24: flags must be identical whether or not the caller subset the frame
    first — label-vs-positional indexing must not misassign or IndexError."""
    df_a = _make_sounding(n=20, line=1000)
    df_b = _make_sounding(n=20, line=2000, add_spike_at=10)
    df_b.loc[15, "elevation"] = 1450.0 - 60.0  # GPS dropout jump
    df = pd.concat([df_a, df_b], ignore_index=True)
    config = _make_config()

    # subset keeps labels 20..39 — previously IndexError or silent misflags
    sub = df[df["line"] == 2000]
    qc_sub = run_qc(sub, config)
    assert list(qc_sub.index) == list(range(20, 40)), "caller's index preserved"

    qc_reset = run_qc(sub.reset_index(drop=True), config)
    flag_cols = [c for c in qc_sub.columns if c.startswith("_qc_") or c == "sounding_mask"]
    np.testing.assert_array_equal(
        qc_sub[flag_cols].to_numpy(), qc_reset[flag_cols].to_numpy(),
        "flags must not depend on the input index",
    )
    assert qc_sub.loc[30, "_qc_spike"], "spike at position 10 of the subset"
    assert qc_sub.loc[35, "_qc_dem_mismatch"], "GPS jump at position 15 of the subset"


def test_despike_threshold_is_sigma_calibrated():
    """#31: with sigma = 1.4826*MAD, threshold=2 is a true 2-sigma test.
    Raw MAD would make it a 1.35-sigma test and flag the majority of clean
    noisy soundings at min_gates=3; calibrated, only a few percent flag."""
    rng = np.random.default_rng(7)
    df = _make_sounding(n=300, noise_scale=0.03)
    # extra independent per-sounding jitter so windows aren't degenerate
    gate_cols = gate_columns(df)
    for col in gate_cols:
        df[col] *= 1.0 + 0.03 * rng.standard_normal(len(df))
    df = _along_line_despike(df, gate_cols, window=5, threshold=2.0,
                             min_gates=3, rel_floor=0.0)
    frac = df["_qc_spike"].mean()
    assert frac < 0.10, (
        f"{frac:.0%} of clean soundings spiked at threshold=2 — "
        "threshold is not sigma-calibrated (raw-MAD behavior flags >50%)"
    )


def test_noise_floor_uses_magnitude():
    """#13: large-|v| late-time negatives are not noise-flagged; small-|v|
    values are, regardless of sign."""
    df = _make_sounding(n=3)
    df.loc[0, "sfz_09"] = -1e-9    # real IP-style negative, well above floor
    df.loc[1, "sfz_09"] = -5e-13   # sub-floor negative
    df.loc[2, "sfz_09"] = 5e-13    # sub-floor positive
    df = _noise_floor_flag(df, gate_columns(df), noise_floor=1e-12, k=2.0)
    assert not df.loc[0, "_qc_gate_09"], "strong negative is signal, not noise"
    assert df.loc[1, "_qc_gate_09"] and df.loc[2, "_qc_gate_09"]


def test_noise_floor_nan_not_conflated():
    """#13: NaN is not 'below floor' — it stays NaN in the gate array anyway."""
    df = _make_sounding(n=3)
    df.loc[1, "sfz_05"] = np.nan
    df = _noise_floor_flag(df, gate_columns(df), noise_floor=1e-12, k=2.0)
    assert not df.loc[1, "_qc_gate_05"]


def test_monotonicity_reversal_across_nan_detected():
    """#14: a NaN gate must not hide a genuine rise on either side of it."""
    df = _make_sounding(n=3)
    cols = gate_columns(df)
    # two rises, each bridged by a NaN: v2=NaN with v3 = 10*v1, v5=NaN with v6 = 10*v4
    df.loc[0, cols[2]] = np.nan
    df.loc[0, cols[3]] = df.loc[0, cols[1]] * 10
    df.loc[0, cols[5]] = np.nan
    df.loc[0, cols[6]] = df.loc[0, cols[4]] * 10
    df = _monotonicity_flag(df, cols, n_early=8, max_reversals=1,
                            noise_floor=NOISE_FLOOR)
    assert df.loc[0, "_qc_nonmono"], "rises bridged by NaN gates must be caught"
    assert not df.loc[1, "_qc_nonmono"]


def test_monotonicity_ignores_sub_noise_upticks():
    """#14: two +0.5% noise blips are not powerline hits — must not flag."""
    df = _make_sounding(n=3)
    cols = gate_columns(df)
    df.loc[0, cols[2]] = df.loc[0, cols[1]] * 1.005
    df.loc[0, cols[5]] = df.loc[0, cols[4]] * 1.005
    df = _monotonicity_flag(df, cols, n_early=8, max_reversals=1,
                            noise_floor=NOISE_FLOOR)
    assert not df.loc[0, "_qc_nonmono"], "sub-noise upticks must not count as reversals"


def test_quantized_gates_no_mass_spikes():
    """#7: heavily quantized (identical) gate values must not mass-flag."""
    df = _make_sounding(n=30)
    gate_cols = gate_columns(df)
    # quantize the last 4 gates to a single repeated value (ASCII rounding)
    for col in gate_cols[-4:]:
        df[col] = float(f"{df[col].iloc[0]:.3g}")
    # tiny per-sounding jitter on one of them — sub-noise, must NOT spike
    df[gate_cols[-1]] += np.linspace(0, 1e-18, 30)
    df = _along_line_despike(df, gate_cols, window=5, threshold=4.0,
                             min_gates=3, noise_floor=NOISE_FLOOR)
    assert df["_qc_spike"].sum() == 0, "Quantized flat gates must not mass-flag"
