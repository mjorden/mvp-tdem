"""Tests for tdem/forward.py — geometry helpers, physics sanity checks, and
the Ward & Hohmann analytic accuracy benchmark (#54)."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import erf

from tdem.forward import (
    TDEMForward,
    forward_from_config,
    layer_depths,
    layer_thicknesses,
)

GATE_TIMES_MS = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]

MU0 = 4e-7 * np.pi


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
# Analytic accuracy benchmark (#54) — Ward & Hohmann (1988) central loop
#
# These are the only tests that constrain amplitude, units, and physical
# constants end-to-end: a factor-of-mu0, factor-of-2*pi, or wrong-moment
# error passes every qualitative test above but misses here by orders of
# magnitude.
# ---------------------------------------------------------------------------

def wh_central_loop_dbzdt(t_s, rho, radius_m):
    """
    Exact |dBz/dt| at the centre of a circular loop lying on the surface of a
    uniform half-space, per unit Tx moment, positive-decaying convention.

    Ward & Hohmann (1988), eq. 4.69-4.70: with theta = sqrt(mu0*sigma/(4t)),

        dhz/dt = -(I / (mu0 sigma a^3)) *
                 [3 erf(theta*a) - (2/sqrt(pi)) theta*a (3 + 2 theta^2 a^2)
                  exp(-theta^2 a^2)]

    Converting to B (x mu0) and normalizing by the moment m = I*pi*a^2 gives
    units of V/(A*m^4), matching TDEMForward.predict(). Exact at all times
    (not just the late-time asymptote).
    """
    sigma = 1.0 / rho
    ta = np.sqrt(MU0 * sigma / (4.0 * np.asarray(t_s, dtype=float))) * radius_m
    bracket = (3.0 * erf(ta)
               - (2.0 / np.sqrt(np.pi)) * ta * (3.0 + 2.0 * ta**2) * np.exp(-ta**2))
    return bracket / (sigma * np.pi * radius_m**5)


def wh_late_time_asymptote(t_s, rho):
    """
    W&H eq. 4.70 late-time limit per unit moment:

        |dBz/dt| -> sigma^(3/2) mu0^(5/2) / (20 pi^(3/2) t^(5/2))

    Valid for t >> mu0*sigma*a^2; pins the published coefficient and the
    sigma^(3/2) t^(-5/2) scaling independently of the full expression.
    """
    sigma = 1.0 / rho
    return sigma**1.5 * MU0**2.5 / (20.0 * np.pi**1.5 * np.asarray(t_s, dtype=float)**2.5)


BENCH_GATES_MS = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8])


@pytest.fixture(scope="module")
def bench_fwd():
    """Uniform-model forward at zero height — mesh layering is irrelevant for a
    uniform half-space (basal half-space has the same rho), so this isolates
    the source/receiver geometry, transforms, and unit chain."""
    thk = layer_thicknesses(1.0, 200, 12)
    return TDEMForward(BENCH_GATES_MS, thk, tx_geometry="concentric_loop",
                       tx_loop_radius_m=13.0)


# Measured agreement (SimPEG 0.25.2, default 81-pt time filter): 0.24% at
# 10 ohm-m, 1.4% at 100, 2.0% at 1000, drift concentrated at late time /
# high rho — consistent with DLF time-filter error (#63). Tolerances leave
# modest headroom; tightening them further requires the 201-pt filter.
@pytest.mark.parametrize("rho,rtol", [(10.0, 0.005), (100.0, 0.02), (1000.0, 0.03)])
def test_halfspace_matches_ward_hohmann_exact(bench_fwd, rho, rtol):
    """#54: full-curve amplitude agreement with the exact analytic solution."""
    d = bench_fwd.predict(np.full(bench_fwd.n_layers, rho), bird_height_m=0.0)
    expected = wh_central_loop_dbzdt(BENCH_GATES_MS * 1e-3, rho, 13.0)
    np.testing.assert_allclose(d, expected, rtol=rtol)


def test_late_time_asymptote_coefficient(bench_fwd):
    """#54: the published sigma^(3/2) mu0^(5/2) / (20 pi^(3/2) t^(5/2))
    coefficient, checked where the asymptote is essentially exact
    (10 ohm-m at 12.8 ms: t/(mu0*sigma*a^2) ~ 6e6)."""
    rho = 10.0
    d = bench_fwd.predict(np.full(bench_fwd.n_layers, rho), bird_height_m=0.0)
    expected = wh_late_time_asymptote(BENCH_GATES_MS[-1] * 1e-3, rho)
    np.testing.assert_allclose(d[-1], expected, rtol=0.01)


def test_late_time_slope_is_minus_five_halves(bench_fwd):
    """#54: log-log decay slope -> -5/2 over a uniform half-space."""
    d = bench_fwd.predict(np.full(bench_fwd.n_layers, 100.0), bird_height_m=0.0)
    t_s = BENCH_GATES_MS * 1e-3
    slope = np.polyfit(np.log(t_s[-4:]), np.log(d[-4:]), 1)[0]
    assert abs(slope - (-2.5)) < 0.05, f"late-time slope {slope:.3f}, expected -2.5"


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------

def test_forward_from_config():
    config_path = Path(__file__).parent.parent / "configs" / "example.json"
    config = json.loads(config_path.read_text())
    fwd = forward_from_config(config)
    assert fwd.n_layers == config["inversion"]["n_layers"]
    assert len(fwd.gate_times_s) == len(config["gate_times_ms"])
    assert fwd.tx_geometry == config["system"]["tx_geometry"]
    assert fwd.tx_loop_radius_m == config["system"]["tx_loop_radius_m"]
