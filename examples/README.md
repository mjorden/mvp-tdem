# Examples

Runnable walkthroughs of the library API. All examples run from the **repo
root** against the checked-in synthetic survey — no extra data required.
Outputs land in `output/examples/`.

```bash
# Run any example directly:
python examples/01_end_to_end.py
python examples/02_forward_modeling.py
python examples/03_single_sounding_inversion.py
python examples/04_multi_line_batch.py
python examples/05_custom_qc_params.py
python examples/06_regularization_tradeoff.py
```

Each script inserts the repo root on `sys.path`, so `pip install -e .` is
optional — a plain `python examples/<name>.py` works from a fresh clone.

---

## Example index

### [01_end_to_end.py](01_end_to_end.py) — Full pipeline as library calls

The same flow as `scripts/process_line.py --line 2000`, but with each step
exposed so you can inspect the intermediate objects.

**Steps covered:** `load_survey` → `run_qc` → `invert_line` → `plot_section`

**Key things to notice:**
- `df` after `load_survey` has standardised column names (`sfz_00…sfz_19`,
  `easting`, `northing`, `elevation`, `dem`) regardless of what the CSV called them
- `run_qc` adds `_qc_*` flag columns without dropping any rows —
  `good_soundings(df)` applies the mask when you want it
- `invert_line` returns a `LineResult` whose `soundings` list carries
  `rho`, `depths`, `chi`, and `converged` per sounding
- The final print confirms conductor recovery: ~15 Ω·m over the body vs ~200 Ω·m background

**Outputs:** `output/examples/line_2000_decays.png`, `line_2000_section.png`, `line_2000_model.csv`

---

### [02_forward_modeling.py](02_forward_modeling.py) — Forward physics sandbox

Build the survey's forward operator from the sidecar and predict dB/dt for
two models side-by-side: a uniform 200 Ω·m halfspace and the same background
with a 5 Ω·m conductor at 20–80 m. Useful for:

- Checking whether your gate times have enough signal above the noise floor
- Understanding how conductor depth and thickness affect which gates are anomalous
- Survey design (what conductor would this system detect at 50 m depth?)

**Key things to notice:**
- `forward_from_config(config)` builds one `TDEMForward` for the whole survey —
  reuse it across lines; per-height simulations are cached internally
- `fwd.predict(rho, bird_height_m)` returns V/(A·m⁴) for a given layered model
- The conductor shows up most strongly in mid/late gates (2–10 ms here)
- The noise floor line shows which gates are reliable — anything below it will
  be QC-flagged in real data

**Outputs:** `output/examples/forward_conductor_vs_halfspace.png`

---

### [03_single_sounding_inversion.py](03_single_sounding_inversion.py) — Inversion intuition

Forward-model a known sounding, add 3% noise, invert it, and compare the
recovered model against ground truth. The place to build intuition for the
two regularization parameters before running a full survey.

**Key things to notice:**
- `invert_sounding` returns `(rho, chi, converged)` — chi ≈ 1 means the
  data are fit to within assigned errors
- Increasing `alpha_z` produces smoother (more gradational) models; decreasing
  it allows sharper layer boundaries at the cost of more oscillation
- `alpha_s` damps the model toward `rho_initial` — useful when you have
  reliable a-priori information about background resistivity
- Occam smoothing means the recovered conductor is blurred: expect 10–30 Ω·m
  at the 20–80 m window even though the true value is 5 Ω·m

**Experiment:** edit `alpha_z` in the script (try 0.1 and 10.0) and re-run
to see the smooth-vs-blocky trade-off directly.

**Outputs:** `output/examples/single_sounding_fit.png`

---

### [04_multi_line_batch.py](04_multi_line_batch.py) — Batch all lines

Process every line in the survey in a loop, reusing one `TDEMForward` across
all lines (the per-height simulation cache pays off). Produces a section PNG
and model CSV per line, plus a merged multi-line model DataFrame.

**Key things to notice:**
- Building `fwd = forward_from_config(config)` once and passing it to
  each `invert_line(..., fwd=fwd)` is significantly faster than letting each
  call build its own
- The merged DataFrame can be used for map-view products (easting/northing vs
  rho at a target depth)
- Skipped / failed soundings are present in the DataFrame but marked
  `converged=False`; filter before use

**Outputs:** `output/examples/line_*_section.png`, `output/examples/all_lines_model.csv`

---

### [05_custom_qc_params.py](05_custom_qc_params.py) — QC parameter tuning

Shows how to adjust the six QC checks for a non-default survey: lower bird
altitude (mountain flying), noisy late gates, or a different acceptable
altitude window. Prints a side-by-side comparison of flag counts under
default vs. custom parameters.

**Key things to notice:**
- All `run_qc` keyword arguments have sensible defaults (`alt_min_m=10`,
  `alt_max_m=80`, `despike_threshold=4.0`, etc.) — only override what you
  need
- `good_soundings(df)` and `good_gate_array(df)` apply the combined mask;
  you never have to filter manually
- Nothing is dropped by QC — the inversion sees the full line; `sounding_mask`
  tells it which soundings to skip

---

### [06_regularization_tradeoff.py](06_regularization_tradeoff.py) — Occam smoothing sweep

Inverts the same synthetic sounding six times across a grid of `alpha_z`
values (0.01 → 100) and plots all recovered models side-by-side. The clearest
demonstration of the smooth-vs-blocky trade-off that is the core of Occam
inversion.

**Key things to notice:**
- At high `alpha_z` (≥ 10) the model is featureless — over-smoothed, chi > 1
- At low `alpha_z` (≤ 0.1) the model oscillates — under-constrained
- The default `alpha_z = 1.0` is a reasonable middle ground for this system
- `chi_target = 1.0` means the inversion halts the Occam cooling loop as soon
  as the misfit drops to the noise level — it never over-fits

**Outputs:** `output/examples/regularization_sweep.png`

---

## Tips

- All examples use `output/examples/` — create it or let the scripts create it
  automatically (`Path.mkdir(parents=True, exist_ok=True)` is called everywhere)
- The synthetic survey is small enough that the full batch runs in under a
  minute on a laptop; real surveys with hundreds of soundings per line will
  take longer (roughly 1–5 s per sounding depending on gate count and
  iteration budget)
- To use your own data, replace the CSV and JSON paths and ensure the
  `column_map` in your JSON matches your CSV headers — see
  [configs/README.md](../configs/README.md) for the full schema
