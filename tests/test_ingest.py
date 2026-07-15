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
    # piecewise model passes through knots; verify to_utc is accurate within
    # the sync range (sync df has t_rx in 500..619 with rate 1+2e-6)
    t_test = np.array([510.0, 560.0, 610.0])
    expected = T0 + (1 + 2e-6) * t_test
    assert np.allclose(m.to_utc(t_test), expected, atol=1e-6)


def test_fit_clock_tolerates_jitter_within_spec():
    m = fit_clock(_sync_df(jitter=1e-4))  # 0.1 ms jitter, 1 ms spec
    assert m.max_residual_s < 1e-3


def test_fit_clock_rejects_single_outlier_transparently():
    """A single garbled message must be silently rejected, not abort (#64)."""
    import warnings
    df = _sync_df()
    df.loc[50, "t_utc"] += 0.5  # one garbled message among 120 good pairs
    with warnings.catch_warnings(record=True):
        m = fit_clock(df)  # must NOT raise
    assert m.n_pairs == 119   # one pair rejected


def test_fit_clock_rejects_bad_sync_stream():
    """Too many outliers (> default 20%) must still error (#64)."""
    df = _sync_df()
    df.loc[30:60, "t_utc"] += 0.5  # 31/120 ≈ 26% garbled
    with pytest.raises(ValueError, match="outlier"):
        fit_clock(df)


def test_fit_clock_needs_two_pairs():
    with pytest.raises(ValueError, match=">= 2 sync pairs"):
        fit_clock(_sync_df(n=1))


def test_clock_extrapolation_capped():
    """#G5: samples far outside the sync range get NaN, not stale-rate extrapolation."""
    import warnings
    m = ClockModel(t_rx_knots=np.array([500.0, 600.0]),
                   t_utc_knots=T0 + np.array([500.0, 600.0]),
                   max_residual_s=0.0, n_pairs=2)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = m.to_utc(np.array([605.0, 5000.0]), max_extrap_s=10.0)
    assert np.isfinite(out[0])          # 5 s past last knot → within margin
    assert np.isnan(out[1])             # 4400 s past → NaN
    assert any("outside the GPS-sync range" in str(x.message) for x in w)


def test_leap_second_hard_errors():
    """#G6: a seconds-scale sync-vs-GPS offset raises unless explicitly allowed."""
    from tdem.ingest.pipeline import _assert_time_standards_agree
    sync = pd.DataFrame({"t_utc": [T0, T0 + 1]})
    gps = pd.DataFrame({"t_utc": [T0 - 18.0, T0 - 17.0]})   # ~18 s leap offset
    with pytest.raises(ValueError, match="time-standard mismatch"):
        _assert_time_standards_agree(sync, gps, allow_time_offset=False)
    import warnings
    with warnings.catch_warnings(record=True) as w:      # override proceeds w/ warning
        warnings.simplefilter("always")
        _assert_time_standards_agree(sync, gps, allow_time_offset=True)
    assert any("Proceeding" in str(x.message) for x in w)
    # aligned standards: no raise
    _assert_time_standards_agree(sync, pd.DataFrame({"t_utc": [T0, T0 + 1]}), False)


def test_apply_clock():
    import numpy as np
    t_rx = np.array([0.0, 100.0])
    m = ClockModel(t_rx_knots=t_rx, t_utc_knots=T0 + t_rx,
                   max_residual_s=0.0, n_pairs=2)
    out = apply_clock(pd.DataFrame({"t_rx": [1.0, 2.0]}), m)
    assert out["t_utc"].tolist() == pytest.approx([T0 + 1.0, T0 + 2.0])


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
    out = stack_soundings(_em_df(), n_stack=24)
    assert len(out) == 2
    assert np.allclose(out[["gate_00", "gate_01", "gate_02"]], 10.0)


def test_stack_rejects_odd_n_stack():
    """#56: odd n_stack cannot balance +/- half-cycles → DC leak; must error."""
    with pytest.raises(ValueError, match="EVEN"):
        stack_soundings(_em_df(), n_stack=25)


def test_stack_pair_differencing_cancels_dc_offset():
    """#56: a constant receiver DC offset b must cancel exactly via pairing."""
    df = _em_df()
    b = 3.0                                   # additive DC on every half-cycle
    for c in ["g00", "g01", "g02"]:
        df[c] = df[c] + b
    out = stack_soundings(df, n_stack=24)
    assert np.allclose(out[["gate_00", "gate_01", "gate_02"]], 10.0)


def test_stack_dc_cancels_on_misaligned_balanced_window():
    """#2: a balanced but phase-misaligned window (+,+,-,-) must still cancel DC
    and NOT report an inflated SE — pairing is keyed on polarity, not position."""
    import warnings
    pol = np.array([1, 1, -1, -1] * 3, dtype=float)   # balanced, mis-phased
    S, b = 10.0, 5.0                                   # response + receiver DC
    v = pol * S + b
    df = pd.DataFrame({"t_rx": np.arange(len(pol)) / 50.0, "polarity": pol,
                       "t_utc": T0 + np.arange(len(pol)) / 50.0, "g00": v})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = stack_soundings(df, n_stack=len(pol), trim_frac=0.1)
    assert out.loc[0, "gate_00"] == pytest.approx(10.0)     # DC cancelled
    assert out.loc[0, "gate_std_00"] < 0.1                  # SE not inflated by ±b
    assert len(w) == 0                                       # balanced → no warning


def test_stack_se_is_floored_not_zero():
    """#3: a perfectly clean gate (MAD == 0) must not report se == 0."""
    out = stack_soundings(_em_df(value=10.0), n_stack=24)
    assert (out["gate_std_00"] > 0).all()


def test_stack_warns_on_genuinely_unbalanced_window():
    """#2: unequal +/- counts (DC truly can't cancel) must warn."""
    import warnings
    pol = np.array([1, 1, 1, -1] * 6, dtype=float)   # 3:1 imbalance per group
    df = pd.DataFrame({"t_rx": np.arange(len(pol)) / 50.0, "polarity": pol,
                       "t_utc": T0 + np.arange(len(pol)) / 50.0, "g00": pol * 10.0})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        stack_soundings(df, n_stack=len(pol))
    assert any("unequal" in str(x.message) for x in w)


def test_stack_trimmed_mean_rejects_spike():
    df = _em_df()
    df.loc[3, ["g00", "g01", "g02"]] *= 20  # sferic hit on one half-cycle
    out = stack_soundings(df, n_stack=24, trim_frac=0.1)
    assert np.allclose(out.loc[0, ["gate_00", "gate_01", "gate_02"]], 10.0)


def test_stack_keeps_spread_and_drops_partial_window():
    out = stack_soundings(_em_df(n_hc=60), n_stack=24)  # 60 = 2 full + 12
    assert len(out) == 2
    assert "gate_std_00" in out.columns
    assert (out["n_used"] == 24).all()


def test_stack_timestamp_centre_with_half_cycle_offset():
    """#68.2: logged stamps are half-cycle STARTS; centre adds 0.5/(2·f)."""
    out = stack_soundings(_em_df(), n_stack=24, tx_frequency_hz=25.0)
    expected = T0 + np.mean(np.arange(24)) / 50 + 0.5 / (2 * 25.0)
    assert out.loc[0, "t_utc"] == pytest.approx(expected)


def test_stack_requires_t_utc():
    with pytest.raises(ValueError, match="t_utc"):
        stack_soundings(_em_df().drop(columns="t_utc"), n_stack=24)


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
    assert mode == "measured_with_gaps"
    assert out.loc[0, "gate_00"] == pytest.approx(1e-3)


def test_calibrate_rejects_signed_bipolar_current():
    """#65.2: a SYSTEMATIC non-positive current means a signed monitor log — error."""
    txcur = pd.DataFrame({"t_utc": [T0 - 1, T0 + 1], "current_a": [220.0, -220.0]})
    with pytest.raises(ValueError, match="non-positive"):
        calibrate(_stacked(), _INSTRUMENT, txcur_df=txcur)


def test_sanitize_current_masks_isolated_glitch_not_raise():
    """#5: a few bad samples (glitch/dropout) are masked+warned, not fatal."""
    from tdem.ingest.calibrate import _sanitize_current
    import warnings
    c = np.array([200.0] * 9 + [-5.0])          # 1 of 10 non-positive
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = _sanitize_current(c, 200.0)
    assert np.isnan(out[-1]) and np.allclose(out[:9], 200.0)
    assert any("masked" in str(x.message) for x in w)


def test_sanitize_current_raises_when_systematic():
    from tdem.ingest.calibrate import _sanitize_current
    with pytest.raises(ValueError, match="non-positive"):
        _sanitize_current(np.array([200.0, -200.0, 200.0, -200.0]), 200.0)


def test_calibrate_warns_current_far_from_nominal():
    """#65.2: current well outside ±20% of nominal warns but proceeds."""
    import warnings
    txcur = pd.DataFrame({"t_utc": [T0 - 1, T0 + 1], "current_a": [400.0, 400.0]})  # 2× nominal
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        calibrate(_stacked(), _INSTRUMENT, txcur_df=txcur)
    assert any("nominal" in str(x.message) for x in w)


def test_calibrate_window_average_de_aliases_current():
    """#65.1/#12: current is time-weighted-averaged over the window, not sampled once."""
    # symmetric edge dips a centre-instant sample (~210) would miss; evenly spaced
    # so the trapezoidal mean is exactly nominal 200 A while a plain sample mean
    # would be 195 A — the test therefore also pins the time-weighting (#12)
    txcur = pd.DataFrame({"t_utc": [T0 - 0.15, T0 - 0.05, T0 + 0.05, T0 + 0.15],
                          "current_a": [180.0, 210.0, 210.0, 180.0]})
    out, _ = calibrate(_stacked(), _INSTRUMENT, txcur_df=txcur, window_s=0.48)
    # trapezoidal mean = (0.5·180 + 210 + 210 + 0.5·180)/3 = 200 A = nominal → 1e-3
    assert out.loc[0, "gate_00"] == pytest.approx(1e-3, rel=1e-6)


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

def test_apply_layback_shifts_bird_behind_and_below():
    """#70: flying north, the bird is layback_m SOUTH of the antenna and drop_m lower."""
    from tdem.ingest.geometry import apply_layback
    n = 20
    df = pd.DataFrame({
        "easting":  np.full(n, 500_000.0),
        "northing": 4_900_000.0 + np.arange(n) * 50.0,     # due north track
        "h_ell":    np.full(n, 1450.0),
    })
    out = apply_layback(df, layback_m=30.0, drop_m=20.0)
    assert np.allclose(out["easting"], 500_000.0, atol=1e-6)
    assert np.allclose(out["northing"], df["northing"] - 30.0)   # behind = south
    assert np.allclose(out["h_ell"], 1430.0)                     # below


def test_apply_layback_flips_with_heading():
    """#70: the along-track shift reverses on a reciprocal heading — the source
    of the ~2x layback line-to-line mis-tie the correction removes."""
    from tdem.ingest.geometry import apply_layback
    n = 20
    east = 500_000.0 + np.arange(n) * 50.0
    df_e = pd.DataFrame({"easting": east, "northing": np.full(n, 4.9e6),
                         "h_ell": np.full(n, 1450.0)})
    df_w = pd.DataFrame({"easting": east[::-1], "northing": np.full(n, 4.9e6),
                         "h_ell": np.full(n, 1450.0)})
    out_e = apply_layback(df_e, layback_m=30.0)
    out_w = apply_layback(df_w, layback_m=30.0)
    assert np.allclose(out_e["easting"], df_e["easting"] - 30.0)  # flying E → bird W
    assert np.allclose(out_w["easting"], df_w["easting"] + 30.0)  # flying W → bird E


def test_apply_layback_zero_is_identity():
    from tdem.ingest.geometry import apply_layback
    df = pd.DataFrame({"easting": [1.0], "northing": [2.0], "h_ell": [3.0]})
    out = apply_layback(df, layback_m=0.0, drop_m=0.0)
    assert out.equals(df)


def test_apply_layback_reprojects_latlon():
    """#70/#95: corrected easting/northing must stay consistent with lat/lon."""
    from pyproj import Transformer
    from tdem.ingest.geometry import apply_layback
    n = 10
    east = np.full(n, 500_000.0)
    north = 4_900_000.0 + np.arange(n) * 50.0
    inv = Transformer.from_crs(32611, 4326, always_xy=True)
    lon, lat = inv.transform(east, north)
    df = pd.DataFrame({"easting": east, "northing": north,
                       "lat": lat, "lon": lon, "h_ell": np.full(n, 1450.0)})
    out = apply_layback(df, layback_m=30.0, epsg=32611)
    tf = Transformer.from_crs(4326, 32611, always_xy=True)
    ex, ny = tf.transform(out["lon"].to_numpy(), out["lat"].to_numpy())
    assert np.allclose(ex, out["easting"], atol=0.01)
    assert np.allclose(ny, out["northing"], atol=0.01)


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
