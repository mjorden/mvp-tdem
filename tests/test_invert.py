"""Tests for tdem/invert.py — synthetic recovery tests (small meshes for speed)."""

import numpy as np
import pandas as pd
import pytest

from tdem.forward import TDEMForward, layer_thicknesses
from tdem.invert import invert_sounding, invert_line, LineResult

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
    rho, chi, ok = invert_sounding(fwd, d_obs, BIRD, rho_initial=200.0, max_iter=40)
    assert ok
    assert chi < 1.0  # noise-free data must fit to well within assigned errors
    # TDEM equivalence: resistive structure is weakly resolved, so allow a
    # generous factor-3 band around the geometric mean rather than exact recovery
    gm = 10 ** np.mean(np.log10(rho))
    assert rho_true / 3 < gm < rho_true * 3, f"recovered {gm:.1f} vs true {rho_true}"


def test_conductor_detected(fwd):
    """Two-decade contrast: conductive earth inverts more conductive than resistive earth."""
    d_cond = fwd.predict(np.full(N_LAYERS, 10.0), BIRD)
    d_res = fwd.predict(np.full(N_LAYERS, 1000.0), BIRD)
    rho_c, _, _ = invert_sounding(fwd, d_cond, BIRD)
    rho_r, _, _ = invert_sounding(fwd, d_res, BIRD)
    # conductors are well resolved by TDEM; resistors suffer equivalence —
    # require clear separation (factor 3+) rather than the full 100x contrast
    assert np.median(rho_c) < np.median(rho_r) / 3


def test_nan_gates_excluded(fwd):
    """Masked gates shouldn't break the inversion."""
    d_obs = fwd.predict(np.full(N_LAYERS, 100.0), BIRD)
    d_obs[5:] = np.nan  # keep only 5 early gates
    rho, chi, ok = invert_sounding(fwd, d_obs, BIRD)
    assert np.all(np.isfinite(rho))


def test_too_few_gates_raises(fwd):
    d_obs = np.full(len(GATE_TIMES_MS), np.nan)
    d_obs[0] = 1e-8
    with pytest.raises(ValueError, match="usable gates"):
        invert_sounding(fwd, d_obs, BIRD)


def test_bounds_respected(fwd):
    d_obs = fwd.predict(np.full(N_LAYERS, 5.0), BIRD)
    rho, _, _ = invert_sounding(fwd, d_obs, BIRD, rho_min=20.0, rho_max=500.0)
    assert np.all(rho >= 20.0 - 1e-6)
    assert np.all(rho <= 500.0 + 1e-6)


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
    assert {"distance", "depth_top", "rho", "chi"} <= set(frame.columns)
    # second sounding is 50 m along the line
    assert frame[frame["fiducial"] == 1001]["distance"].iloc[0] == pytest.approx(50.0)


def test_ground_elevation_computed(fwd):
    df = _make_line_df(fwd, [100])
    result = invert_line(df, _config(), fwd=fwd, verbose=False)
    # ground = GPS elevation - bird height
    assert result.soundings[0].elevation == pytest.approx(1450.0 - BIRD)
