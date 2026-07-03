"""Unit tests for the ingest stages (tdem/ingest/*)."""

import numpy as np
import pandas as pd
import pytest

from tdem.ingest.calibrate import calibrate
from tdem.ingest.geometry import assign_fids, assign_lines, elevations
from tdem.ingest.merge import interp_with_gaps, merge_nav
from tdem.ingest.stack import stack_soundings
from tdem.ingest.timesync import ClockModel, apply_clock, fit_clock

T0 = 1_750_000_000.0


# ---------------------------------------------------------------------------
# timesync
# ---------------------------------------------------------------------------

def _sync_df(offset=T0, rate=1 + 2e-6, n=120, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t_rx = np.arange(n, dtype=float) + 500.0
    return pd.DataFrame({"t_rx": t_rx,
                         "t_utc": offset + rate * t_rx + rng.normal(0, jitter, n)})


def test_fit_clock_recovers_offset_and_drift():
    m = fit_clock(_sync_df())
    # 0.1 ms on the offset: an order below the residual spec, and above
    # float64 quantization at unix-epoch magnitudes
    assert m.offset_s == pytest.approx(T0, abs=1e-4)
    assert m.rate == pytest.approx(1 + 2e-6, abs=1e-9)


def test_fit_clock_tolerates_jitter_within_spec():
    m = fit_clock(_sync_df(jitter=1e-4))  # 0.1 ms jitter, 1 ms spec
    assert m.max_residual_s < 1e-3


def test_fit_clock_rejects_bad_sync_stream():
    df = _sync_df()
    df.loc[50, "t_utc"] += 0.5  # a garbled time message
    with pytest.raises(ValueError, match="residual"):
        fit_clock(df)


def test_fit_clock_needs_two_pairs():
    with pytest.raises(ValueError, match=">= 2 sync pairs"):
        fit_clock(_sync_df(n=1))


def test_apply_clock():
    m = ClockModel(offset_s=T0, rate=1.0, max_residual_s=0, n_pairs=2)
    out = apply_clock(pd.DataFrame({"t_rx": [1.0, 2.0]}), m)
    assert list(out["t_utc"]) == [T0 + 1.0, T0 + 2.0]


# ---------------------------------------------------------------------------
# stack
# ---------------------------------------------------------------------------

def _em_df(n_hc=50, n_gates=3, value=10.0):
    pol = np.where(np.arange(n_hc) % 2 == 0, 1, -1)
    df = pd.DataFrame({"t_rx": np.arange(n_hc) / 50.0, "polarity": pol})
    df["t_utc"] = T0 + df["t_rx"]
    for k in range(n_gates):
        df[f"g{k:02d}"] = value * pol  # sign follows Tx polarity, as logged
    return df


def test_stack_polarity_alignment():
    """Alternating-sign raw values must stack to the positive response."""
    out = stack_soundings(_em_df(), n_stack=25)
    assert len(out) == 2
    assert np.allclose(out[["gate_00", "gate_01", "gate_02"]], 10.0)


def test_stack_trimmed_mean_rejects_spike():
    df = _em_df()
    df.loc[3, ["g00", "g01", "g02"]] *= 20  # sferic hit on one half-cycle
    out = stack_soundings(df, n_stack=25, trim_frac=0.1)
    assert np.allclose(out.loc[0, ["gate_00", "gate_01", "gate_02"]], 10.0)


def test_stack_keeps_spread_and_drops_partial_window():
    out = stack_soundings(_em_df(n_hc=60), n_stack=25)  # 60 = 2 full + 10
    assert len(out) == 2
    assert "gate_std_00" in out.columns
    assert (out["n_used"] == 25).all()


def test_stack_timestamp_is_window_centre():
    out = stack_soundings(_em_df(), n_stack=25)
    assert out.loc[0, "t_utc"] == pytest.approx(T0 + np.mean(np.arange(25)) / 50)


def test_stack_requires_t_utc():
    with pytest.raises(ValueError, match="t_utc"):
        stack_soundings(_em_df().drop(columns="t_utc"), n_stack=25)


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------

_INSTRUMENT = {
    "rx": {"gain": 1000.0, "coil_area_m2": 100.0},
    "tx": {"moment_nominal_am2": 400_000, "n_turns": 4, "loop_area_m2": 500.0},
}


def _stacked(v=4.0e7):
    # 4e7 logged volts / (gain 1e3 · area 1e2 · moment 4e5) = 1e-3 V/(A·m⁴)
    return pd.DataFrame({"t_utc": [T0], "n_used": [25],
                         "gate_00": [v], "gate_std_00": [v / 100]})


def test_calibrate_nominal_moment():
    out, mode = calibrate(_stacked(), _INSTRUMENT, txcur_df=None)
    assert mode == "nominal"
    # 4e10 / 1000 / 100 / 400000 = 1e-3
    assert out.loc[0, "gate_00"] == pytest.approx(1e-3)
    assert out.loc[0, "gate_std_00"] == pytest.approx(1e-5)


def test_calibrate_measured_moment():
    # measured current 10% above nominal (200 A nominal = 4e5/(4*500))
    txcur = pd.DataFrame({"t_utc": [T0 - 1, T0 + 1], "current_a": [220.0, 220.0]})
    out, mode = calibrate(_stacked(), _INSTRUMENT, txcur_df=txcur)
    assert mode == "measured"
    assert out.loc[0, "gate_00"] == pytest.approx(1e-3 / 1.1)


def test_calibrate_falls_back_to_nominal_current_in_gaps():
    txcur = pd.DataFrame({"t_utc": [T0 + 100, T0 + 101], "current_a": [220.0, 220.0]})
    out, mode = calibrate(_stacked(), _INSTRUMENT, txcur_df=txcur)  # sounding outside range
    assert mode == "measured"
    assert out.loc[0, "gate_00"] == pytest.approx(1e-3)


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

def test_interp_linear_inside():
    v = interp_with_gaps(np.array([1.5]), np.array([1.0, 2.0]), np.array([10.0, 20.0]), 2.0)
    assert v[0] == pytest.approx(15.0)


def test_interp_nan_outside_range_and_in_gaps():
    t_src = np.array([0.0, 1.0, 8.0, 9.0])  # 7 s hole in the middle
    v = interp_with_gaps(np.array([-1.0, 0.5, 4.0, 10.0]), t_src, t_src * 2, max_gap_s=2.0)
    assert np.isnan(v[0])          # before range
    assert v[1] == pytest.approx(1.0)
    assert np.isnan(v[2])          # inside the hole
    assert np.isnan(v[3])          # after range


def test_merge_nav_attaches_all_columns():
    s = pd.DataFrame({"t_utc": [T0 + 0.5]})
    gps = pd.DataFrame({"t_utc": [T0, T0 + 1], "lat": [44.0, 44.001],
                        "lon": [-119.0, -119.0], "h_ell": [1400.0, 1402.0]})
    alt = pd.DataFrame({"t_utc": [T0, T0 + 1], "agl_m": [30.0, 40.0]})
    out = merge_nav(s, gps, alt)
    assert out.loc[0, "lat"] == pytest.approx(44.0005)
    assert out.loc[0, "h_ell"] == pytest.approx(1401.0)
    assert out.loc[0, "agl_m"] == pytest.approx(35.0)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def test_elevations_and_dem_convention():
    """Repo convention: the DEM column is bird height AGL, not ground elevation."""
    out = elevations(pd.DataFrame({"h_ell": [1380.0], "agl_m": [35.0]}),
                     geoid_offset_m=-20.0)
    assert out.loc[0, "elevation"] == pytest.approx(1400.0)
    assert out.loc[0, "dem"] == pytest.approx(35.0)


def test_assign_lines_from_operator_log():
    s = pd.DataFrame({"t_utc": [T0 + 1, T0 + 5, T0 + 11]})
    lines = pd.DataFrame({"line": [1000, 1010],
                          "t_start_utc": [T0, T0 + 10],
                          "t_end_utc": [T0 + 8, T0 + 20]})
    out = assign_lines(s, lines)
    assert list(out["line"]) == [1000, 1000, 1010]


def test_assign_lines_heading_fallback_segments_l_shape():
    # 40 soundings north, 40 east — two lines with a corner
    n = 40
    east  = np.concatenate([np.full(n, 0.0), np.arange(n) * 15.0])
    north = np.concatenate([np.arange(n) * 15.0, np.full(n, n * 15.0)])
    s = pd.DataFrame({"t_utc": T0 + np.arange(2 * n) * 0.5,
                      "easting": east, "northing": north})
    out = assign_lines(s, None)
    ids = [l for l in out["line"].unique() if l != -1]
    assert len(ids) == 2
    # the straight middles of each leg are on-line
    assert (out.loc[10:25, "line"] == ids[0]).all()
    assert (out.loc[n + 10 : n + 25, "line"] == ids[1]).all()


def test_assign_fids_monotonic_deciseconds():
    out = assign_fids(pd.DataFrame({"t_utc": [T0, T0 + 0.5, T0 + 1.0]}))
    assert list(out["fid"]) == [0, 5, 10]
