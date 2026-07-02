"""Tests for tdem/forward.py — geometry helpers + physics sanity checks."""

import json
from pathlib import Path

import numpy as np
import pytest

from tdem.forward import (
    TDEMForward,
    forward_from_config,
    layer_depths,
    layer_thicknesses,
)

GATE_TIMES_MS = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]


# ---------------------------------------------------------------------------
# Layer geometry
# ---------------------------------------------------------------------------

def test_layer_thicknesses_count():
    thk = layer_thicknesses(0.5, 300, 30)
    assert len(thk) == 29  # n_layers - 1; last layer is half-space


def test_layer_thicknesses_positive_and_increasing():
    thk = layer_thicknesses(0.5, 300, 30)
    assert np.all(thk > 0)
    # log spacing → thicknesses grow with depth
    assert thk[-1] > thk[0]


def test_layer_thicknesses_rejects_single_layer():
    with pytest.raises(ValueError):
        layer_thicknesses(0.5, 300, 1)


def test_layer_depths_starts_at_zero():
    thk = layer_thicknesses(0.5, 300, 10)
    d = layer_depths(thk)
    assert d[0] == 0.0
    assert len(d) == len(thk) + 1


# ---------------------------------------------------------------------------
# Forward physics (slower — real SimPEG simulations)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fwd():
    thk = layer_thicknesses(1.0, 200, 12)
    return TDEMForward(GATE_TIMES_MS, thk)


def test_predict_shape_and_positive(fwd):
    rho = np.full(fwd.n_layers, 100.0)
    d = fwd.predict(rho, bird_height_m=35.0)
    assert d.shape == (len(GATE_TIMES_MS),)
    assert np.all(d > 0)


def test_predict_decays_with_time(fwd):
    """Off-time dB/dt over a uniform half-space must decay monotonically."""
    rho = np.full(fwd.n_layers, 100.0)
    d = fwd.predict(rho, bird_height_m=35.0)
    assert np.all(np.diff(d) < 0), f"Response should decay: {d}"


def test_conductor_stronger_late_time(fwd):
    """A conductive earth holds eddy currents longer → larger late-time signal."""
    d_cond = fwd.predict(np.full(fwd.n_layers, 10.0), bird_height_m=35.0)
    d_res = fwd.predict(np.full(fwd.n_layers, 1000.0), bird_height_m=35.0)
    # compare the last few gates
    assert np.all(d_cond[-3:] > d_res[-3:]), "Conductor should dominate late time"


def test_higher_bird_weaker_signal(fwd):
    """More altitude → weaker coupling → smaller response at every gate."""
    rho = np.full(fwd.n_layers, 100.0)
    d_low = fwd.predict(rho, bird_height_m=30.0)
    d_high = fwd.predict(rho, bird_height_m=60.0)
    assert np.all(d_low > d_high)


def test_predict_log_matches_predict(fwd):
    rho = np.full(fwd.n_layers, 50.0)
    d1 = fwd.predict(rho, bird_height_m=35.0)
    d2 = fwd.predict_log(np.log10(rho), bird_height_m=35.0)
    np.testing.assert_allclose(d1, d2, rtol=1e-12)


def test_predict_wrong_layer_count_raises(fwd):
    with pytest.raises(ValueError, match="layer resistivities"):
        fwd.predict(np.full(fwd.n_layers + 3, 100.0), bird_height_m=35.0)


def test_simulation_cache_reused(fwd):
    rho = np.full(fwd.n_layers, 100.0)
    fwd.predict(rho, bird_height_m=42.0)
    fwd.predict(rho, bird_height_m=42.04)  # rounds to same 0.1 m key
    assert 42.0 in fwd._sim_cache
    assert len([k for k in fwd._sim_cache if abs(k - 42.0) < 0.2]) == 1


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------

def test_forward_from_config():
    config_path = Path(__file__).parent.parent / "configs" / "example.json"
    config = json.loads(config_path.read_text())
    fwd = forward_from_config(config)
    assert fwd.n_layers == config["inversion"]["n_layers"]
    assert len(fwd.gate_times_s) == len(config["gate_times_ms"])
    assert fwd.tx_rx_separation_m == config["system"]["tx_rx_separation_m"]
