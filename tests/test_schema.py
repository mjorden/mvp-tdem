"""Tests for tdem/schema.py — sidecar validation gate (#87, #83)."""

import json
import warnings
from pathlib import Path

import pytest

from tdem.schema import SCHEMA_VERSION, validate_sidecar


def _example_config():
    p = Path(__file__).parent.parent / "configs" / "example.json"
    return json.loads(p.read_text(encoding="utf-8"))


def test_example_sidecar_validates_cleanly():
    """The shipped example must pass without warnings (units + version declared)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # any warning fails the test
        validate_sidecar(_example_config())


def test_future_schema_version_rejected():
    cfg = _example_config()
    cfg["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="newer than this code"):
        validate_sidecar(cfg)


def test_missing_schema_version_treated_as_v1():
    cfg = _example_config()
    del cfg["schema_version"]
    validate_sidecar(cfg)                        # no raise


def test_wrong_units_rejected():
    """#83: a non-canonical units declaration must fail loudly — no conversion exists."""
    cfg = _example_config()
    cfg["system"]["units"] = "nT/s"
    with pytest.raises(ValueError, match="units"):
        validate_sidecar(cfg)


def test_unit_spelling_variants_accepted():
    cfg = _example_config()
    for spelling in ("V/(A.m^4)", "v/(a.m4)", "V/(A·m⁴)"):
        cfg["system"]["units"] = spelling
        validate_sidecar(cfg)                    # no raise


def test_missing_units_warns():
    cfg = _example_config()
    del cfg["system"]["units"]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_sidecar(cfg)
    assert any("units not declared" in str(x.message) for x in w)


def test_bfield_quantity_rejected():
    cfg = _example_config()
    cfg["system"]["response_quantity"] = "B"
    with pytest.raises(ValueError, match="response_quantity"):
        validate_sidecar(cfg)


def test_unknown_waveform_rejected_at_validation():
    """#87: the enum check fires at the door, not deep in forward_from_config."""
    cfg = _example_config()
    cfg["system"]["tx_waveform"] = "vtem_trapezoid"
    with pytest.raises(ValueError, match="tx_waveform"):
        validate_sidecar(cfg)


def test_unknown_geometry_and_format_rejected():
    cfg = _example_config()
    cfg["system"]["tx_geometry"] = "big_loop"
    with pytest.raises(ValueError, match="tx_geometry"):
        validate_sidecar(cfg)
    cfg = _example_config()
    cfg["column_map"]["sfz_format"] = "camelCase"
    with pytest.raises(ValueError, match="sfz_format"):
        validate_sidecar(cfg)


def test_non_z_orientation_rejected():
    cfg = _example_config()
    cfg["system"]["rx_orientation"] = "X"
    with pytest.raises(ValueError, match="rx_orientation"):
        validate_sidecar(cfg)


def test_inversion_typo_warns():
    """#87: a misspelled inversion key silently no-ops downstream — warn here."""
    cfg = _example_config()
    cfg["inversion"]["alpha_zz"] = 2.0           # typo of alpha_z
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_sidecar(cfg)
    assert any("alpha_zz" in str(x.message) for x in w)


def test_load_survey_runs_validation(tmp_path):
    """Integration: load_survey rejects a wrong-units sidecar before reading data."""
    from tdem.load import load_survey
    cfg = _example_config()
    cfg["system"]["units"] = "ppm"
    cfg_path = tmp_path / "bad.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    csv_path = Path(__file__).parent.parent / "data" / "synthetic_survey.csv"
    with pytest.raises(ValueError, match="units"):
        load_survey(csv_path, cfg_path)
