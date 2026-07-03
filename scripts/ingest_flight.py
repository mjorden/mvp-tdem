"""
Raw flight logs → canonical survey CSV + generated sidecar.

Usage
-----
    python scripts/ingest_flight.py \
        --flight data/flights/F20260702_01 [--flight data/flights/F20260702_02 ...] \
        --instrument configs/instrument.yaml \
        --survey configs/survey_example.yaml \
        --out-csv data/survey.csv --out-config data/survey_sidecar.json

The output pair feeds straight into the existing pipeline:

    python scripts/process_line.py --csv data/survey.csv \
        --config data/survey_sidecar.json --all-lines
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click

from tdem.ingest import ingest_survey


@click.command()
@click.option("--flight", "flights", multiple=True, required=True,
              type=click.Path(exists=True, file_okay=False),
              help="Flight directory (repeatable, one per flight)")
@click.option("--instrument", required=True, type=click.Path(exists=True),
              help="Instrument hardware YAML (configs/instrument.yaml)")
@click.option("--survey", required=True, type=click.Path(exists=True),
              help="Survey YAML (name, EPSG, ingest + inversion params)")
@click.option("--out-csv", default="data/survey.csv", type=click.Path())
@click.option("--out-config", default="data/survey_sidecar.json", type=click.Path())
def main(flights, instrument, survey, out_csv, out_config):
    csv_path, config_path = ingest_survey(list(flights), instrument, survey,
                                          out_csv, out_config)
    print(f"[ingest] sidecar → {config_path}")
    print("[ingest] next: python scripts/process_line.py "
          f"--csv {csv_path} --config {config_path} --all-lines")


if __name__ == "__main__":
    main()
