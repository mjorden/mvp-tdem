"""
End-to-end ingest round trip:

    synthetic raw flight → ingest_survey → load_survey → run_qc

Verifies the whole chain: clock sync, polarity stacking, calibration with
measured Tx current, nav merge, projection/elevations, line assignment,
and that the emitted pair is a valid deliverable for the existing
pipeline with values matching the encoded truth.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from tdem.ingest import ingest_survey
from tdem.ingest.synthetic import generate_flight
from tdem.load import gate_array, load_survey
from tdem.qc import run_qc

REPO = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def ingested(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("raw_flight")
    with open(REPO / "configs/instrument.yaml") as f:
        instrument = yaml.safe_load(f)
    with open(REPO / "configs/survey_example.yaml") as f:
        survey_cfg = yaml.safe_load(f)

    truth = generate_flight(tmp / "F01", instrument, survey_cfg,
                            n_lines=2, n_stations=16, seed=7)
    csv_path, config_path = ingest_survey(
        [tmp / "F01"], REPO / "configs/instrument.yaml",
        REPO / "configs/survey_example.yaml",
        tmp / "survey.csv", tmp / "sidecar.json",
    )
    df, config = load_survey(csv_path, config_path)
    return truth, df, config


def test_all_stations_survive(ingested):
    truth, df, _ = ingested
    assert len(df) == len(truth)
    assert sorted(df["line"].unique()) == sorted(truth["line"].unique())


def test_positions_match_truth(ingested):
    truth, df, _ = ingested
    # sort both the same way; serpentine lines make northing non-monotonic
    t = truth.sort_values("t_utc").reset_index(drop=True)
    d = df.sort_values("fiducial").reset_index(drop=True)
    assert np.allclose(d["easting"], t["easting"], atol=1.0)
    assert np.allclose(d["northing"], t["northing"], atol=1.0)
    assert np.allclose(d["elevation"], t["elevation"], atol=0.5)
    assert np.allclose(d["dem"], t["dem"], atol=0.5)


def test_gate_values_match_truth_within_noise(ingested):
    truth, df, config = ingested
    t = truth.sort_values("t_utc").reset_index(drop=True)
    d = df.sort_values("fiducial").reset_index(drop=True)

    got  = gate_array(d)
    want = t[[c for c in t.columns if c.startswith("sfz_")]].to_numpy()

    # only judge gates comfortably above the noise floor; with 3%
    # per-half-cycle noise stacked over 25, a 5% tolerance is generous
    floor = config["system"]["system_noise_floor"]
    strong = want > 10 * floor
    assert strong.sum() > strong.size * 0.5
    rel = np.abs(got[strong] - want[strong]) / want[strong]
    assert np.median(rel) < 0.02
    assert np.quantile(rel, 0.95) < 0.05


def test_sidecar_is_generated_and_valid(ingested):
    _, _, config = ingested
    assert config["column_map"]["sfz_format"] == "bracket"
    assert len(config["gate_times_ms"]) == config["column_map"]["sfz_n"]
    ing = config["ingest"]
    assert ing["flights"][0]["moment"] == "measured"
    assert ing["flights"][0]["line_assignment"] == "operator_log"
    assert ing["flights"][0]["time_sync"]["max_residual_ms"] < 1.0
    assert ing["flights"][0]["sources"]  # provenance hashes present


def test_qc_runs_clean_on_ingested_survey(ingested):
    _, df, config = ingested
    out = run_qc(df, config)
    # altitude ~35 m AGL and a consistent GPS/altimeter pair: nothing
    # should be flagged wholesale
    assert not out["_qc_alt_low"].any()
    assert not out["_qc_alt_high"].any()
    assert out["_qc_dem_mismatch"].mean() < 0.1
