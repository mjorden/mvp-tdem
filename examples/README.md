# Examples

Runnable walkthroughs of the library API. All run from the repo root against the checked-in synthetic survey and sidecar (`data/synthetic_survey.csv` + `configs/example.json`); no extra data needed. Outputs go to `output/examples/`.

```bash
python examples/01_end_to_end.py
python examples/02_forward_modeling.py
python examples/03_single_sounding_inversion.py
```

| Example | What it shows |
|---------|---------------|
| [01_end_to_end.py](01_end_to_end.py) | The full pipeline as library calls: `load_survey` → `run_qc` → `invert_line` → `plot_section`, with the intermediate DataFrames exposed. Same flow as the `process_line.py` CLI. |
| [02_forward_modeling.py](02_forward_modeling.py) | Forward physics only: predict decays for a halfspace vs. a buried conductor, against the system noise floor. Useful for gate-time sanity checks and survey design. |
| [03_single_sounding_inversion.py](03_single_sounding_inversion.py) | Invert one synthetic sounding against known ground truth and plot fit + recovered model. The place to build intuition for `alpha_s`, `alpha_z`, and the Occam smoothing trade-off. |

If you'd rather not `pip install -e .`, each script inserts the repo root on `sys.path`, so a plain `python examples/<name>.py` works either way.
