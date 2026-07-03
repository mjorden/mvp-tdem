"""Tests for tdem/load.py validators, Geosoft XYZ parsing, and cleaning."""

import numpy as np
import pandas as pd
import pytest

from tdem.load import (
    _clean,
    _read_csv,
    _validate,
    gate_columns,
    gate_index,
    load_line,
    load_survey,
)


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
    p.write_text(text)
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
    cfg_path.write_text(json.dumps(cfg))
    df, _ = load_survey(_write(tmp_path, GEOSOFT_FILE), cfg_path)
    # '*' elevation row survives load (dem is fine) with elevation NaN — no crash (#8)
    assert len(df) == 5
    assert df["elevation"].isna().sum() == 1
    # line ids came from the separator records
    assert set(df["line"]) == {"1000", "2000", "9000"}


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


def test_coordinate_float_dirt_not_matched():
    """#29: coordinate channels use exact sentinel matching — the tolerant
    window NaN'd real positions near 999,999 / 9,999,999 m."""
    df = _clean(_clean_df(), {})
    # -9999.0000001 in elevation is NOT a sentinel under exact matching
    assert df["elevation"].isna().sum() == 0


def test_coordinate_near_sentinel_survives():
    """#29: northing within 1 m of the 999999 sentinel is a real position."""
    df = _clean_df()
    df["northing"] = [999998.5, 999999.5, 9999995.0, 4900000.0]
    out = _clean(df, {})
    assert out["northing"].notna().all(), \
        "coordinates near (not equal to) a sentinel must survive"


def test_coordinate_exact_sentinel_still_dummy():
    df = _clean_df()
    df.loc[0, "northing"] = 999999.0
    out = _clean(df, {})
    assert np.isnan(out.loc[0, "northing"])


def test_gate_float_dirt_still_tolerant():
    """#29: data channels keep the tolerant match for round-trip float dirt."""
    df = _clean_df()
    df.loc[0, "sfz_00"] = -9999.0000001
    out = _clean(df, {})
    assert np.isnan(out.loc[0, "sfz_00"])


def test_nan_and_huge_dem_dropped_with_report(capsys):
    df = _clean(_clean_df(), {})
    out = capsys.readouterr().out
    # NaN dem row and 1e32-dummy dem row both dropped, and reported
    assert len(df) == 2
    assert "Dropped 2 soundings" in out


def test_gate_dummy_still_naned():
    df = _clean(_clean_df(), {})
    assert df["sfz_01"].isna().sum() == 1


# ---------------------------------------------------------------------------
# Flat-CSV delimiter sniffing (#30)
# ---------------------------------------------------------------------------

def test_flat_csv_empty_field_not_shifted(tmp_path):
    """#30: a blank cell must become NaN in place, not shift later columns."""
    flat = "A,B,C\n1,,3\n4,5,6\n"
    df = _read_csv(_write(tmp_path, flat, "flat.csv"))
    assert np.isnan(df.loc[0, "B"]), "empty field should be NaN"
    assert df.loc[0, "C"] == 3, "value after the empty field must not shift left"
    assert df.loc[1, "B"] == 5


def test_space_delimited_flat_file(tmp_path):
    flat = "A B C\n1 2 3\n4  5\t6\n"
    df = _read_csv(_write(tmp_path, flat, "flat.txt"))
    assert list(df.columns) == ["A", "B", "C"]
    assert df.loc[1, "B"] == 5


# ---------------------------------------------------------------------------
# Loader output schema (#36)
# ---------------------------------------------------------------------------

def test_loader_emits_documented_schema(tmp_path):
    """#36: exactly the standardised columns + one sfz_NN per gate — no raw
    gate columns, no unmapped raw channels, no orphan SFz_std."""
    import json
    csv = ("LINE,FID,Easting,Northing,Elevation,DEM,Junk,SFz[0],SFz[1],SFz_std[0],SFz_std[1]\n"
           "1000,1,500000,4900000,145.0,35.0,7,2.3e-8,1.1e-8,1e-9,5e-10\n")
    cfg = {
        "column_map": {
            "line": "LINE", "fiducial": "FID", "easting": "Easting",
            "northing": "Northing", "elevation": "Elevation", "dem": "DEM",
            "sfz_prefix": "SFz", "sfz_n": 2, "sfz_format": "bracket",
        },
        "gate_times_ms": [0.1, 1.0],
        "system": {"tx_frequency_hz": 25, "tx_on_time_us": 4000},
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    df, _ = load_survey(_write(tmp_path, csv, "flat.csv"), cfg_path)
    assert list(df.columns) == ["line", "fiducial", "easting", "northing",
                                "elevation", "dem", "sfz_00", "sfz_01"]


# ---------------------------------------------------------------------------
# Gate identity (#41) and line-id matching (#19)
# ---------------------------------------------------------------------------

def test_gate_columns_numeric_order():
    """#41: sfz_100 must sort after sfz_11, not between sfz_10 and sfz_11."""
    df = pd.DataFrame({f"sfz_{i}": [0.0] for i in (100, 9, 10, 101, 11)})
    assert gate_columns(df) == ["sfz_9", "sfz_10", "sfz_11", "sfz_100", "sfz_101"]


def test_gate_index_parses_and_rejects():
    assert gate_index("sfz_07") == 7
    assert gate_index("sfz_100") == 100
    with pytest.raises(ValueError, match="Not a gate column"):
        gate_index("gate_07")


def test_load_line_matches_across_types():
    """#19: '--line 1000' must match int, string, and mixed line ids."""
    df = pd.DataFrame({"line": [1000, 2000, 1000], "sfz_00": [1.0, 2.0, 3.0]})
    assert len(load_line(df, "1000")) == 2
    assert len(load_line(df, 1000)) == 2
    df_str = df.assign(line=df["line"].astype(str))
    assert len(load_line(df_str, 1000)) == 2
    with pytest.raises(ValueError, match="not found"):
        load_line(df, "L99")
