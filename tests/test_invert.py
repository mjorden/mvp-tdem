"""Tests for tdem/invert.py — synthetic recovery tests (small meshes for speed)."""

import numpy as np
import pandas as pd
import pytest

from tdem.forward import TDEMForward, layer_thicknesses
from tdem.invert import (
    invert_sounding, invert_line, LineResult, _log_floored, _dlog_floored,
)

GATE_TIMES_MS = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
N_LAYERS = 8
BIRD = 35.0


@pytest.fixture(scope="module")
def fwd():
    return TDEMForward(GATE_TIMES_MS, layer_thicknesses(2.0, 150, N_LAYERS))


# ---------------------------------------------------------------------------
# invert_sounding
# ---------------------------------------------------------------------------

def test_recover_halfspace(fwd):
    """Noise-free half-space data should invert back to ~the true resistivity."""
    rho_true = 50.0
    d_obs = fwd.predict(np.full(N_LAYERS, rho_true), BIRD)
    rho, chi, ok, doi_m, rho_sd = invert_sounding(fwd, d_obs, BIRD, rho_initial=200.0, max_iter=40)
    assert ok
    assert chi < 1.0  # noise-free data must fit to well within assigned errors
    # TDEM equivalence: resistive structure is weakly resolved, so allow a
    # generous factor-3 band around the geometric mean rather than exact recovery
    gm = 10 ** np.mean(np.log10(rho))
    assert rho_true / 3 < gm < rho_true * 3, f"recovered {gm:.1f} vs true {rho_true}"
    # DOI must be a finite positive depth (can reach depth_max for noise-free data)
    assert doi_m > 0
    # rho_sd must be per-layer multiplicative factors >= 1
    assert rho_sd.shape == rho.shape
    assert np.all(rho_sd >= 1.0)


def test_conductor_detected(fwd):
    """Two-decade contrast: conductive earth inverts more conductive than resistive earth."""
    d_cond = fwd.predict(np.full(N_LAYERS, 10.0), BIRD)
    d_res = fwd.predict(np.full(N_LAYERS, 1000.0), BIRD)
    rho_c, *_ = invert_sounding(fwd, d_cond, BIRD)
    rho_r, *_ = invert_sounding(fwd, d_res, BIRD)
    # conductors are well resolved by TDEM; resistors suffer equivalence —
    # require clear separation (factor 3+) rather than the full 100x contrast
    assert np.median(rho_c) < np.median(rho_r) / 3


def test_nan_gates_excluded(fwd):
    """Masked gates shouldn't break the inversion."""
    d_obs = fwd.predict(np.full(N_LAYERS, 100.0), BIRD)
    d_obs[5:] = np.nan  # keep only 5 early gates
    rho, chi, ok, *_ = invert_sounding(fwd, d_obs, BIRD)
    assert np.all(np.isfinite(rho))


def test_too_few_gates_raises(fwd):
    d_obs = np.full(len(GATE_TIMES_MS), np.nan)
    d_obs[0] = 1e-8
    with pytest.raises(ValueError, match="usable gates"):
        invert_sounding(fwd, d_obs, BIRD)


def test_bounds_respected(fwd):
    d_obs = fwd.predict(np.full(N_LAYERS, 5.0), BIRD)
    rho, *_ = invert_sounding(fwd, d_obs, BIRD, rho_min=20.0, rho_max=500.0)
    assert np.all(rho >= 20.0 - 1e-6)
    assert np.all(rho <= 500.0 + 1e-6)


# ---------------------------------------------------------------------------
# #53: sign-safe log transform (no zero-gradient cliff on negative predictions)
# ---------------------------------------------------------------------------

def test_log_floored_matches_log_well_above_eps():
    x = np.array([1e-9, 1e-6, 1e-3])
    eps = 1e-3 * x
    assert np.allclose(_log_floored(x, eps), np.log(x))


def test_log_floored_is_c1_continuous_at_eps():
    eps = np.array([1e-9])
    # value continuity: both branches meet at x == eps
    assert np.isclose(_log_floored(eps, eps)[0], np.log(eps)[0])
    # slope continuity: derivative is 1/eps on both sides of eps
    assert np.isclose(_dlog_floored(eps, eps)[0], 1.0 / eps[0])


def test_dlog_floored_nonzero_gradient_for_negative_pred():
    """The core of #53: a non-positive prediction must NOT give zero gradient."""
    eps = np.array([1e-9, 1e-9])
    x = np.array([-3e-9, 0.0])          # negative / zero prediction
    g = _dlog_floored(x, eps)
    assert np.all(g == 1.0 / eps)       # finite, large, positive — pushes pred up
    assert np.all(g > 0)


def test_negative_gate_excluded_from_gate_count(fwd):
    """#59: n_gates_used counts USED gates (finite & positive), not merely finite."""
    d = fwd.predict(np.full(N_LAYERS, 100.0), BIRD)
    df = _make_line_df(fwd, [100.0])
    gcols = [f"sfz_{k:02d}" for k in range(N_LAYERS)]
    df.loc[0, gcols[-1]] = -abs(d[-1])          # one finite-but-negative gate
    result = invert_line(df, _config(), fwd=fwd, verbose=False)
    assert result.soundings[0].n_gates_used == N_LAYERS - 1


# ---------------------------------------------------------------------------
# invert_line
# ---------------------------------------------------------------------------

def _make_line_df(fwd, rhos, line_id=1000):
    """Synthetic line: one sounding per entry of rhos (uniform half-spaces)."""
    rows = []
    for i, rho in enumerate(rhos):
        d = fwd.predict(np.full(N_LAYERS, rho), BIRD)
        row = {
            "line": line_id,
            "fiducial": 1000 + i,
            "easting": 500000.0,
            "northing": 4900000.0 + i * 50,
            "elevation": 1450.0,
            "dem": BIRD,
        }
        for k, v in enumerate(d):
            row[f"sfz_{k:02d}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def _config():
    return {
        "system": {"system_noise_floor": 0.0},
        "gate_times_ms": GATE_TIMES_MS,
        "column_map": {"sfz_n": len(GATE_TIMES_MS)},
        "inversion": {
            "n_layers": N_LAYERS, "depth_min_m": 2.0, "depth_max_m": 150,
            "rho_initial": 100.0, "rho_min": 1.0, "rho_max": 10000.0,
            "alpha_s": 1e-4, "alpha_z": 1.0, "max_iter": 30,
            "use_noise_floor": False,
        },
    }


def test_invert_line_anomaly_localized(fwd):
    """Conductor mid-line should appear only in the middle soundings."""
    rhos = [200, 200, 5, 5, 200, 200]
    df = _make_line_df(fwd, rhos)
    result = invert_line(df, _config(), fwd=fwd, verbose=False)
    assert len(result.soundings) == 6
    med = [np.median(s.rho) for s in result.soundings]
    assert med[2] < med[0] / 5, "conductor sounding should be much more conductive"
    assert med[3] < med[5] / 5


def test_invert_line_skips_masked(fwd):
    df = _make_line_df(fwd, [100, 100, 100])
    df["sounding_mask"] = [False, True, False]
    result = invert_line(df, _config(), fwd=fwd, verbose=False)
    assert len(result.soundings) == 2
    fids = [s.fiducial for s in result.soundings]
    assert 1001 not in fids


def test_to_frame_layout(fwd):
    df = _make_line_df(fwd, [100, 50])
    result = invert_line(df, _config(), fwd=fwd, verbose=False)
    frame = result.to_frame()
    assert len(frame) == 2 * N_LAYERS
    assert {"distance", "depth_top", "rho", "chi", "doi_m", "rho_sd"} <= set(frame.columns)
    # second sounding is 50 m along the line
    assert frame[frame["fiducial"] == 1001]["distance"].iloc[0] == pytest.approx(50.0)
    # doi_m is constant per sounding (repeated across layers)
    for fid in frame["fiducial"].unique():
        sub = frame[frame["fiducial"] == fid]
        assert sub["doi_m"].nunique() == 1
        assert (sub["doi_m"] > 0).all()
    # rho_sd values are per-layer multiplicative factors >= 1
    assert (frame["rho_sd"] >= 1.0).all()


def test_ground_elevation_computed(fwd):
    df = _make_line_df(fwd, [100])
    result = invert_line(df, _config(), fwd=fwd, verbose=False)
    # ground = GPS elevation - bird height
    assert result.soundings[0].elevation == pytest.approx(1450.0 - BIRD)


# ---------------------------------------------------------------------------
# Occam cooling / regularization (#10, #11, #18)
# ---------------------------------------------------------------------------

def test_cooling_stops_at_smoothest_fitting_model(fwd):
    """#10: noisy data should be fit to ~chi_target, not far below (no overfit)."""
    rng = np.random.default_rng(7)
    d_true = fwd.predict(np.full(N_LAYERS, 100.0), BIRD)
    d_obs = d_true * (1 + 0.05 * rng.standard_normal(len(d_true)))
    rho, chi, ok, *_ = invert_sounding(fwd, d_obs, BIRD, rel_error=0.05,
                                       chi_target=1.0, max_iter=40)
    # cooling accepts the FIRST (smoothest) stage reaching target — chi should
    # land near 1, not grossly below (which would mean fitting noise)
    assert chi <= 1.0
    assert chi > 0.2, f"chi={chi:.2f} — suspiciously overfit despite cooling"


def test_reference_model_decoupled_from_start(fwd):
    """#18: rho_ref pins the damping target; warm start must not change the objective."""
    d_obs = fwd.predict(np.full(N_LAYERS, 100.0), BIRD)
    weird_start = np.full(N_LAYERS, 3000.0)
    rho_a, chi_a, *_ = invert_sounding(fwd, d_obs, BIRD,
                                       rho_initial=weird_start, rho_ref=100.0,
                                       max_iter=40)
    rho_b, chi_b, *_ = invert_sounding(fwd, d_obs, BIRD,
                                       rho_initial=100.0, rho_ref=100.0,
                                       max_iter=40)
    # same objective → both should fit; recovered models comparable in the
    # well-resolved shallow half
    assert chi_a <= 1.0 and chi_b <= 1.0
    gm_a = 10 ** np.mean(np.log10(rho_a[:4]))
    gm_b = 10 ** np.mean(np.log10(rho_b[:4]))
    assert gm_a / gm_b < 3 and gm_b / gm_a < 3


def test_stale_warm_start_reset(fwd):
    """#18: after a long QC gap, warm start resets to the cold model."""
    rhos = [5.0] * 2 + [200.0] * 8
    df = _make_line_df(fwd, rhos)
    mask = [False, False] + [True] * 6 + [False, False]   # 6-sounding gap
    df["sounding_mask"] = mask
    result = invert_line(df, _config(), fwd=fwd, verbose=False)
    # soundings after the gap must invert fine (started cold, not from the conductor)
    post_gap = [s for s in result.soundings if s.fiducial >= 1008]
    assert len(post_gap) == 2
    assert all(s.chi <= 2.0 for s in post_gap)
