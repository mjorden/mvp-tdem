"""Tests for tdem/load.py validators, Geosoft XYZ parsing, and cleaning."""

import numpy as np
import pandas as pd
import pytest

from tdem.load import _apply_column_map, _clean, _read_csv, _validate, load_survey


def _df():
    return pd.DataFrame({
        "easting": [1.0], "northing": [1.0], "elevation": [100.0], "dem": [35.0],
    })


def _config(gates, f=25, on_us=4000):
    return {
        "column_map": {"sfz_n": len(gates)},
        "gate_times_ms": gates,
        "system": {"tx_frequency_hz": f, "tx_on_time_us": on_us},
    }


def test_gates_within_off_time_pass():
    # 16 ms off-time at 25 Hz / 4 ms on-time
    _validate(_df(), _config([0.1, 1.0, 10.0, 15.9]))


def test_gates_beyond_off_time_rejected():
    """#1: gates outside the off-time window are physically impossible."""
    with pytest.raises(ValueError, match="off-time"):
        _validate(_df(), _config([0.1, 1.0, 21.7]))


def test_gate_count_mismatch_rejected():
    cfg = _config([0.1, 1.0])
    cfg["column_map"]["sfz_n"] = 5
    with pytest.raises(ValueError, match="must match"):
        _validate(_df(), cfg)


def test_no_frequency_skips_offtime_check():
    cfg = _config([0.1, 743.0], f=None)
    _validate(_df(), cfg)  # no tx_frequency_hz → can't check, don't crash


def test_missing_sfz_n_raises_clear_error():
    """#43.2: an omitted sfz_n must name the real problem, not default to 30."""
    with pytest.raises(ValueError, match="sfz_n is required"):
        _apply_column_map(pd.DataFrame({"E": [1.0]}), {"easting": "E"})


def test_implausible_gate_amplitude_warns():
    """#A1: gate values far outside V/(A·m⁴) range warn about likely wrong units."""
    import warnings
    from tdem.load import _clean
    df = pd.DataFrame({
        "line": [1000, 1000], "fiducial": [1, 2],
        "easting": [5e5, 5e5], "northing": [4.9e6, 4.9e6],
        "elevation": [100.0, 100.0], "dem": [35.0, 35.0],
        # amplitudes ~1e3 — clearly not moment-normalized dB/dt
        "sfz_00": [1200.0, 1300.0], "sfz_01": [800.0, 900.0],
    })
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _clean(df, {"column_map": {"sfz_n": 2}})
    assert any("different units" in str(x.message) for x in w)


def test_non_monotonic_gate_times_rejected():
    """#68.5: a mis-ordered gate-time table mispairs every gate → hard error."""
    with pytest.raises(ValueError, match="strictly increasing"):
        _validate(_df(), _config([0.1, 0.4, 0.2, 1.0]))


def test_nonpositive_gate_times_rejected():
    with pytest.raises(ValueError, match="strictly increasing"):
        _validate(_df(), _config([0.0, 0.4, 1.0]))


# ---------------------------------------------------------------------------
# Geosoft XYZ parsing (#2)
# ---------------------------------------------------------------------------

GEOSOFT_FILE = """\
/ Survey exported from Oasis montaj
/ FID Easting Northing Elevation DEM SFz[0] SFz[1] SFz[2]
Line 1000
100 500000.0 4900000.0 145.2 35.1 2.3e-8 1.1e-8 5.0e-9
101 500000.0 4900050.0 145.5 34.9 2.2e-8 1.0e-8 4.8e-9
Line 2000
200 500200.0 4900000.0 146.0 36.0 2.5e-8 1.2e-8 5.5e-9
201 500200.0 4900050.0 * 35.8 2.4e-8 1.1e-8 5.1e-9
Tie 9000
900 500100.0 4900025.0 145.8 35.5 2.4e-8 1.1e-8 5.2e-9
"""


def _write(tmp_path, text, name="survey.xyz"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_geosoft_header_from_comment(tmp_path):
    df = _read_csv(_write(tmp_path, GEOSOFT_FILE))
    assert list(df.columns[:5]) == ["FID", "Easting", "Northing", "Elevation", "DEM"]
    assert len(df) == 5


def test_geosoft_line_records_forward_filled(tmp_path):
    df = _read_csv(_write(tmp_path, GEOSOFT_FILE))
    assert list(df["__geosoft_line"]) == ["1000", "1000", "2000", "2000", "9000"]


def test_geosoft_star_dummy_coerced(tmp_path):
    cfg = {
        "column_map": {
            "fiducial": "FID", "easting": "Easting", "northing": "Northing",
            "elevation": "Elevation", "dem": "DEM",
            "sfz_prefix": "SFz", "sfz_n": 3, "sfz_format": "bracket",
        },
        "gate_times_ms": [0.1, 1.0, 10.0],
        "system": {"tx_frequency_hz": 25, "tx_on_time_us": 4000},
    }
    import json
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    df, _ = load_survey(_write(tmp_path, GEOSOFT_FILE), cfg_path)
    # '*' elevation row survives load (dem is fine) with elevation NaN — no crash (#8)
    assert len(df) == 5
    assert df["elevation"].isna().sum() == 1
    # line ids came from the separator records, normalized to int so a Geosoft
    # file and a flat CSV both answer load_line(df, 2000) (#68.4)
    assert set(df["line"]) == {1000, 2000, 9000}


def test_flat_csv_still_works(tmp_path):
    flat = "A,B\n1,2\n3,4\n"
    df = _read_csv(_write(tmp_path, flat, "flat.csv"))
    assert list(df.columns) == ["A", "B"]
    assert len(df) == 2


def test_geosoft_malformed_row_skipped(tmp_path):
    bad = GEOSOFT_FILE + "junk row\n"
    df = _read_csv(_write(tmp_path, bad))
    assert len(df) == 5  # junk (2 tokens vs 8 columns) skipped, not NaN-padded


# ---------------------------------------------------------------------------
# _clean dummy handling (#8, #16)
# ---------------------------------------------------------------------------

def _clean_df():
    return pd.DataFrame({
        "line": [1, 1, 1, 1],
        "easting": [1.0, 2.0, 3.0, 4.0],
        "northing": [1.0, 1.0, 1.0, 1.0],
        "elevation": [100.0, -9999.0000001, 100.0, 100.0],
        "dem": [35.0, 35.0, np.nan, 1e32],
        "sfz_00": [1e-8, 1e-8, 1e-8, 1e-8],
        "sfz_01": [1e-9, -9999.0, 1e-9, 1e-9],
    })


def test_dummy_tolerant_match_all_columns():
    df = _clean(_clean_df(), {})
    # -9999.0000001 in elevation (non-gate column) caught by tolerant match
    assert df["elevation"].isna().sum() == 1


def test_nan_and_huge_dem_dropped_with_report(capsys):
    df = _clean(_clean_df(), {})
    out = capsys.readouterr().out
    # NaN dem row and 1e32-dummy dem row both dropped, and reported
    assert len(df) == 2
    assert "Dropped 2 soundings" in out


def test_gate_dummy_still_naned():
    df = _clean(_clean_df(), {})
    assert df["sfz_01"].isna().sum() == 1
