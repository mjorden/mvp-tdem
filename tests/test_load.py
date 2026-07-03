"""Tests for tdem/load.py validators."""

import pandas as pd
import pytest

from tdem.load import _validate


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
