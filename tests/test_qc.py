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
