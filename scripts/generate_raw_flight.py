"""
Generate a synthetic RAW flight directory (em/sync/gps/alt/txcur/lines
logs) for exercising the ingest stage end-to-end.

Analytic decays, not SimPEG — this tests ingest bookkeeping (clock sync,
stacking, calibration, merge, geometry), not inversion physics. For a
physics-true survey CSV use scripts/generate_synthetic.py.

Usage
-----
    python scripts/generate_raw_flight.py --out data/flights/F_SYNTH_01
    python scripts/ingest_flight.py --flight data/flights/F_SYNTH_01 \
        --instrument configs/instrument.yaml --survey configs/survey_example.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tdem.ingest.pipeline import load_yaml
from tdem.ingest.synthetic import generate_flight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/flights/F_SYNTH_01")
    parser.add_argument("--instrument", default="configs/instrument.yaml")
    parser.add_argument("--survey", default="configs/survey_example.yaml")
    parser.add_argument("--n-lines", type=int, default=2)
    parser.add_argument("--n-stations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    truth = generate_flight(
        args.out, load_yaml(args.instrument), load_yaml(args.survey),
        n_lines=args.n_lines, n_stations=args.n_stations, seed=args.seed,
    )
    print(f"Wrote raw flight ({len(truth)} stations, "
          f"{truth['line'].nunique()} lines) → {args.out}")


if __name__ == "__main__":
    main()
