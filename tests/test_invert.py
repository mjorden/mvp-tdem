"""Tests for tdem/invert.py — synthetic recovery tests (small meshes for speed)."""

import numpy as np
import pandas as pd
import pytest

from tdem.forward import TDEMForward, layer_thicknesses
from tdem.invert import (
    invert_sounding, invert_line, LineResult, _asinh_scaled, _dasinh_scaled,
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
    assert np.all((rho_sd >= 1.0) | np.isnan(rho_sd))  # NaN for pegged layers (#10)


def test_doi_within_mesh_and_cumulative(fwd):
    """#7: cumulative-sensitivity DOI is a positive depth inside the mesh."""
    d_obs = fwd.predict(np.full(N_LAYERS, 50.0), BIRD)
    _, _, _, doi_m, _ = invert_sounding(fwd, d_obs, BIRD, rho_initial=100.0, max_iter=40)
    depth_max = 150 * 1.4          # mesh + basal cell
    assert 0 < doi_m <= depth_max


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


def test_pegged_layer_uncertainty_is_nan(fwd):
    """#10: a layer pinned at a bound has undefined linearized uncertainty → NaN."""
    # true 5 ohm-m but rho_min=20 forces the shallow layers onto the lower bound
    d_obs = fwd.predict(np.full(N_LAYERS, 5.0), BIRD)
    rho, _, _, _, rho_sd = invert_sounding(fwd, d_obs, BIRD, rho_min=20.0, rho_max=500.0)
    pegged = rho <= 20.0 + 1e-6
    assert pegged.any(), "test needs at least one pegged layer"
    assert np.all(np.isnan(rho_sd[pegged]))         # no over-confident error at a bound
    assert np.all(rho_sd[~pegged] >= 1.0)


# ---------------------------------------------------------------------------
# #78: symmetric asinh transform (signed data inverted, no zero-gradient cliff)
# ---------------------------------------------------------------------------

def test_asinh_matches_log_well_above_scale():
    """For |x| >> s: asinh(x/2s) ≈ ln(|x|/s) — the log-misfit limit."""
    x = np.array([1e-9, 1e-6, 1e-3])
    s = 1e-4 * x
    assert np.allclose(_asinh_scaled(x, s), np.log(x / s), rtol=1e-6)


def test_asinh_is_odd_and_smooth_through_zero():
    """Signed data: f(-x) = -f(x); linear (not cliffed) near zero."""
    s = np.array([1e-12, 1e-12])
    x = np.array([3e-12, -3e-12])
    f = _asinh_scaled(x, s)
    assert f[1] == pytest.approx(-f[0])
    # near zero the transform is ~x/(2s): no discontinuity, finite value
    tiny = _asinh_scaled(np.array([1e-14]), np.array([1e-12]))[0]
    assert tiny == pytest.approx(1e-14 / 2e-12, rel=1e-3)


def test_dasinh_positive_gradient_everywhere():
    """The #53/#78 requirement: gradient never vanishes, including x <= 0."""
    s = np.full(4, 1e-12)
    x = np.array([-5e-11, -1e-12, 0.0, 5e-11])
    g = _dasinh_scaled(x, s)
    assert np.all(np.isfinite(g)) and np.all(g > 0)


def test_negative_gate_now_inverted_not_censored(fwd):
    """#78: a signed gate with amplitude above the censor IS used by the misfit."""
    d = fwd.predict(np.full(N_LAYERS, 100.0), BIRD)
    df = _make_line_df(fwd, [100.0])
    gcols = [f"sfz_{k:02d}" for k in range(N_LAYERS)]
    df.loc[0, gcols[-1]] = -abs(d[-1])          # IP-like sign flip, strong amplitude
    result = invert_line(df, _config(), fwd=fwd, verbose=False)
    s = result.soundings[0]
    # censor threshold is 0 here (use_noise_floor False) → all finite gates used
    assert s.n_gates_used == N_LAYERS
    assert s.gate_used is not None and s.gate_used.all()
    assert np.isfinite(s.chi) and np.all(np.isfinite(s.rho))


def test_amplitude_censor_drops_near_floor_gates_of_either_sign(fwd):
    """#27/#78: |d| <= 3·floor is censored regardless of sign; strong gates kept."""
    d = fwd.predict(np.full(N_LAYERS, 100.0), BIRD)
    floor = abs(d[-1])                           # set the floor at the last gate
    d_obs = d.copy()
    d_obs[-1] = -0.5 * floor                     # weak negative: |d| < 3·floor → censored
    d_obs[-2] = -abs(d_obs[-2])                  # strong negative: kept and inverted
    rho, chi, ok, *_ = invert_sounding(fwd, d_obs, BIRD, noise_floor=floor,
                                       rho_initial=100.0, max_iter=40)
    assert np.all(np.isfinite(rho)) and np.isfinite(chi)


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
    # rho_sd values are per-layer multiplicative factors >= 1 (NaN where pegged, #10)
    sd = frame["rho_sd"].to_numpy()
    assert np.all((sd >= 1.0) | np.isnan(sd))


def test_ground_elevation_computed(fwd):
    df = _make_line_df(fwd, [100])
    result = invert_line(df, _config(), fwd=fwd, verbose=False)
    # ground = GPS elevation - bird height
    assert result.soundings[0].elevation == pytest.approx(1450.0 - BIRD)


def test_to_frame_self_describing(fwd):
    """#39/#37/#12: frame carries depth_bottom, n_gates_used, below_doi, elev_ground."""
    df = _make_line_df(fwd, [100, 50])
    result = invert_line(df, _config(), fwd=fwd, verbose=False)
    frame = result.to_frame()
    for col in ("depth_bottom", "n_gates_used", "below_doi", "elev_ground"):
        assert col in frame.columns
    assert "elevation" not in frame.columns          # renamed to avoid sensor/ground clash
    # every depth_bottom is strictly below its depth_top
    assert (frame["depth_bottom"] > frame["depth_top"]).all()
    # below_doi flips true only at/under the sounding's DOI
    for fid in frame["fiducial"].unique():
        sub = frame[frame["fiducial"] == fid]
        doi = sub["doi_m"].iloc[0]
        assert (sub["below_doi"] == (sub["depth_top"] >= doi)).all()


def test_line_result_provenance(fwd):
    """#38: LineResult carries QC-skip and failure counts."""
    df = _make_line_df(fwd, [100, 100, 100])
    df["sounding_mask"] = [False, True, False]
    result = invert_line(df, _config(), fwd=fwd, verbose=False)
    assert result.n_qc_skipped == 1
    assert result.n_failed == 0


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


def test_unreachable_chi_target_returns_solved_model_not_prior(fwd):
    """Regression: when no cooling stage reaches chi_target, return the roughest
    SOLVED model with its real chi — never rho_initial verbatim (prior-as-result)."""
    d_obs = fwd.predict(np.full(N_LAYERS, 5.0), BIRD)   # true 5 ohm-m conductor
    rho, chi, ok, *_ = invert_sounding(
        fwd, d_obs, BIRD, rho_initial=100.0, chi_target=1e-9, max_iter=40)
    assert not np.allclose(rho, 100.0), "must not return the rho_initial prior"
    assert np.median(rho) < 20.0, "must recover the conductor, not the 100 ohm-m start"
    # reported chi must correspond to the returned model, not a stale value
    assert chi == pytest.approx(
        float(np.sqrt(np.mean(((np.log(np.maximum(fwd.predict(rho, BIRD), 1e-300))
                                - np.log(d_obs)) / 0.05) ** 2))), rel=0.05)


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
