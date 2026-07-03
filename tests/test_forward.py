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


def test_layer_thicknesses_reach_depth_max():
    """#9: the basal half-space must start exactly at depth_max."""
    thk = layer_thicknesses(0.5, 300, 30)
    assert np.isclose(thk.sum(), 300.0)


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
    return TDEMForward(GATE_TIMES_MS, thk)  # default: concentric_loop, r=13 m


def test_concentric_vs_offset_dipole_differ_early_time():
    """#6: finite-loop and point-dipole geometries must differ at early gates."""
    thk = layer_thicknesses(1.0, 200, 12)
    rho = np.full(12, 100.0)
    d_loop = TDEMForward(GATE_TIMES_MS, thk, tx_geometry="concentric_loop",
                         tx_loop_radius_m=13.0).predict(rho, 35.0)
    d_dip = TDEMForward(GATE_TIMES_MS, thk, tx_geometry="offset_dipole",
                        tx_rx_separation_m=13.0).predict(rho, 35.0)
    # both positive-decaying, but not identical (esp. early time)
    assert np.all(d_loop > 0) and np.all(d_dip > 0)
    assert not np.allclose(d_loop[:3], d_dip[:3], rtol=1e-3, atol=0.0)


def test_offset_dipole_zero_separation_warns_and_clamps():
    """#21: silent clamp is now a warning."""
    thk = layer_thicknesses(1.0, 200, 8)
    with pytest.warns(UserWarning, match="clamped to 1 m"):
        f = TDEMForward(GATE_TIMES_MS, thk, tx_geometry="offset_dipole",
                        tx_rx_separation_m=0.0)
    assert f.tx_rx_separation_m == 1.0


def test_unknown_geometry_raises():
    thk = layer_thicknesses(1.0, 200, 8)
    with pytest.raises(ValueError, match="tx_geometry"):
        TDEMForward(GATE_TIMES_MS, thk, tx_geometry="triangle")


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
# Bipolar waveform superposition (#22)
# ---------------------------------------------------------------------------

# example.json system: 25 Hz bipolar, 4 ms on-time → 16 ms off-time
BIPOLAR = dict(waveform="bipolar_square", base_frequency_hz=25.0, on_time_ms=4.0)


def test_bipolar_bias_grows_late_time():
    """#22: prior half-cycles suppress the late-time response; the single
    step-off overpredicts, worst at the last gate (−57% at 14.624 ms for the
    example system, verified against Ward & Hohmann in the review)."""
    gates = [0.4268, 1.386, 2.0524, 3.04, 4.5012, 6.668, 9.876, 14.624]
    thk = layer_thicknesses(0.5, 300, 30)
    rho = np.full(30, 100.0)
    d_step = TDEMForward(gates, thk).predict(rho, 35.0)
    d_bip = TDEMForward(gates, thk, **BIPOLAR).predict(rho, 35.0)
    bias = (d_bip - d_step) / d_step
    assert np.all(bias < 0), "bipolar response must sit below single step-off"
    assert np.all(np.diff(bias) < 0), "bias magnitude must grow with gate time"
    assert -0.62 < bias[-1] < -0.52, f"last-gate bias {bias[-1]:.3f}, expected ~-0.57"
    assert -0.06 < bias[0] < 0.0, f"first-gate bias {bias[0]:.3f}, expected ~-0.004"


def test_bipolar_converged_by_default_periods():
    """The superposition series must be converged at the default n_periods."""
    gates = [1.0, 4.0, 14.624]
    thk = layer_thicknesses(1.0, 200, 10)
    rho = np.full(10, 100.0)
    d4 = TDEMForward(gates, thk, **BIPOLAR).predict(rho, 35.0)
    d8 = TDEMForward(gates, thk, **{**BIPOLAR, "n_periods": 8}).predict(rho, 35.0)
    # truncation error at n_periods=4 is ~1.5e-4 on the last gate — three
    # orders below the 5% assigned data error
    np.testing.assert_allclose(d4, d8, rtol=5e-4)


def test_bipolar_requires_waveform_params():
    thk = layer_thicknesses(1.0, 200, 8)
    with pytest.raises(ValueError, match="base_frequency_hz"):
        TDEMForward(GATE_TIMES_MS, thk, waveform="bipolar_square")
    with pytest.raises(ValueError, match="Unknown waveform"):
        TDEMForward(GATE_TIMES_MS, thk, waveform="trapezoid")


def test_bipolar_rejects_gates_outside_off_time():
    thk = layer_thicknesses(1.0, 200, 8)
    with pytest.raises(ValueError, match="off-time"):
        TDEMForward([0.5, 17.0], thk, **BIPOLAR)  # off-time is 16 ms


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------

def test_forward_from_config():
    config_path = Path(__file__).parent.parent / "configs" / "example.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    fwd = forward_from_config(config)
    assert fwd.n_layers == config["inversion"]["n_layers"]
    assert len(fwd.gate_times_s) == len(config["gate_times_ms"])
    assert fwd.tx_geometry == config["system"]["tx_geometry"]
    assert fwd.tx_loop_radius_m == config["system"]["tx_loop_radius_m"]
    # #22: the sidecar declares a bipolar square wave — the forward must use it
    assert fwd.waveform == "bipolar_square"
    assert fwd.base_frequency_hz == config["system"]["tx_frequency_hz"]
    assert fwd.on_time_ms == config["system"]["tx_on_time_us"] / 1000.0
