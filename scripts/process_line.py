"""
End-to-end pipeline: load → QC → invert → plot, for one or all flight lines.

Usage
-----
    python scripts/process_line.py --csv data/survey.csv --config configs/example.json --line 1000
    python scripts/process_line.py --csv data/survey.csv --config configs/example.json --all-lines
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click

from tdem.load import load_survey, load_line
from tdem.qc import run_qc
from tdem.invert import invert_line
from tdem.forward import forward_from_config
from tdem.visualize import plot_section, plot_decays


@click.command()
@click.option("--csv", "csv_path", required=True, type=click.Path(exists=True),
              help="Survey CSV deliverable")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True),
              help="JSON sidecar (system params, gate times, column map)")
@click.option("--line", "line_id", default=None, help="Flight line to process")
@click.option("--all-lines", is_flag=True, help="Process every line in the survey")
@click.option("--out-dir", default="output", type=click.Path(), help="Output directory")
def main(csv_path, config_path, line_id, all_lines, out_dir):
    if not line_id and not all_lines:
        raise click.UsageError("Provide --line <id> or --all-lines")

    out_dir = Path(out_dir)
    df, config = load_survey(csv_path, config_path)
    print(f"[main] loaded {len(df)} soundings, "
          f"{df['line'].nunique() if 'line' in df.columns else '?'} lines")

    df = run_qc(df, config)
    fwd = forward_from_config(config)   # shared across lines (per-height sim cache)

    if all_lines:
        line_ids = sorted(df["line"].unique())
    else:
        # load_line compares ids as strings (#19) — no coercion needed
        line_ids = [line_id]

    for lid in line_ids:
        df_line = load_line(df, lid)
        print(f"[main] line {lid}: {len(df_line)} soundings")

        plot_decays(
            df_line, config["gate_times_ms"],
            out_dir / f"line_{lid}_decays.png",
            title=f"Line {lid} — observed decays",
        )

        result = invert_line(df_line, config, fwd=fwd)
        if not result.soundings:
            print(f"[main] line {lid}: no soundings survived — skipping outputs")
            continue

        plot_section(result, out_dir / f"line_{lid}_section.png")
        model_csv = out_dir / f"line_{lid}_model.csv"
        result.to_frame().to_csv(model_csv, index=False)
        print(f"[main] wrote {model_csv}")


if __name__ == "__main__":
    main()
