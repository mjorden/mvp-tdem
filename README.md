# mvp-tdem

Airborne (helicopter) time-domain electromagnetic (TDEM) data processing and 1D inversion pipeline: generic CSV deliverable in, stitched 2D resistivity sections per flight line out.

> **Note:** This repo will eventually be transferred to the `lubricate-ai` GitHub organization. The name `mvp-tdem` is temporary.

## What it does

```
survey CSV + JSON sidecar
        │
        ▼
  load  ──► standardised DataFrame (sfz_00..sfz_NN gate columns)
        │
        ▼
  qc    ──► _qc_* flag columns + per-sounding sounding_mask (nothing dropped)
        │
        ▼
  invert ──► per-sounding Occam-style 1D inversion, warm-started along-line
        │     (SimPEG 1D layered forward, scipy TRF least squares)
        ▼
  visualize ──► stitched resistivity section PNG + decay plots + model CSV
```

**Stack:** Python ≥ 3.10, SimPEG, empymod, numpy/scipy/pandas, matplotlib, click.

## Install

```bash
pip install -e ".[dev]"
```

## Quickstart (synthetic data)

The repo ships with a synthetic 3-line survey (`data/synthetic_survey.csv`, lines `1000`/`2000`/`3000`, buried 5 Ω·m conductor at 20–80 m depth) generated with the same SimPEG physics the inversion uses.

```bash
# Process one flight line end-to-end
python scripts/process_line.py \
    --csv data/synthetic_survey.csv \
    --config configs/example.json \
    --line 2000

# Or every line in the survey
python scripts/process_line.py \
    --csv data/synthetic_survey.csv \
    --config configs/example.json \
    --all-lines
```

Outputs land in `output/`:

| File | Contents |
|------|----------|
| `line_<id>_section.png` | Stitched resistivity cross-section (elevation-referenced, chi misfit strip on top, below-DOI cells faded) |
| `line_<id>_decays.png` | All observed decay curves, coloured by along-line position (quick QC view) |
| `line_<id>_model.csv` | Self-describing long-format model: one row per (sounding, layer) with `rho`, `depth_top`, `depth_bottom`, `chi`, `doi_m`, `below_doi`, `n_gates_used`, `rho_sd`, `elev_ground`, `distance` |

To regenerate the synthetic survey (e.g. after changing gate times):

```bash
python scripts/generate_synthetic.py --out data/synthetic_survey.csv
```

For library-API usage (no CLI), see [examples/](examples/).

## Web UI

A Streamlit browser interface wraps the full pipeline — upload a CSV + JSON sidecar, run QC and inversion, and explore results interactively without touching the CLI.

```bash
pip install -e ".[ui]"
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in a browser.  The sidebar walks you through upload → QC → inversion; results appear in the **QC**, **Section**, **Decays**, and **Download** tabs.

## Project layout

```
mvp-tdem/
├── tdem/
│   ├── load.py          # CSV + JSON sidecar ingestion, column mapping, cleaning
│   ├── qc.py            # 6 QC checks → _qc_* flags; nothing dropped, callers decide
│   ├── forward.py       # SimPEG 1D TDEM forward wrapper (TDEMForward, per-height sim cache)
│   ├── invert.py        # Occam-style per-sounding inversion + along-line warm-start stitching
│   ├── visualize.py     # plot_section / plot_decays / plot_sounding_fit (matplotlib, Agg)
│   └── ingest/          # Raw instrument logs → canonical CSV + sidecar (docs/RAW_INGEST.md):
│                        #   readers → timesync → stack → calibrate → merge → geometry → emit
├── configs/
│   ├── example.json     # Survey sidecar: column map, system params, gate times, inversion params
│   ├── instrument.yaml  # Own-instrument hardware config (ingest generates sidecars from it)
│   ├── survey_example.yaml  # Per-survey ingest + inversion params
│   └── README.md        # Full sidecar schema reference
├── docs/
│   └── RAW_INGEST.md    # Raw-ingest design and v0 log formats
├── examples/            # Runnable library-API walkthroughs (see examples/README.md)
├── scripts/
│   ├── process_line.py  # CLI: load → QC → invert → plot, one line or --all-lines
│   ├── ingest_flight.py # CLI: raw flight dir(s) → survey.csv + generated sidecar
│   ├── generate_synthetic.py  # Synthetic 3-line survey via the real SimPEG forward
│   └── generate_raw_flight.py # Synthetic RAW flight dir for exercising ingest
├── data/
│   └── synthetic_survey.csv   # Checked-in synthetic survey (see data/README.md)
├── tests/               # pytest suite (load / qc / forward / invert)
└── output/              # Generated plots and model CSVs (gitignored)
```

## Input format

Two files per survey:

1. **CSV** — flat ASCII table (Geosoft XYZ export or equivalent; `/`-prefixed comment lines are skipped, whitespace or comma delimited). One row per sounding: line, fiducial, coordinates, bird height, and one column per time gate.
2. **JSON sidecar** — everything the CSV doesn't carry: column name mapping, system parameters (geometry, waveform, noise floor), gate-centre times, and inversion defaults. Schema documented in [configs/README.md](configs/README.md).

Three gate-column naming conventions are supported via `column_map.sfz_format`: `bracket` (`SFz[0]`), `underscore` (`SFz_00`), `zero_padded` (`SFz00`).

### Raw instrument logs

For our own instrument the CSV + sidecar pair is *generated*, not hand-written: `scripts/ingest_flight.py` turns a flight directory of raw logger streams (EM half-cycles, GPS, altimeter, Tx current, operator line log) into the canonical pair — clock sync, polarity stacking, calibration to V/(A·m⁴), nav merge, projection, and line assignment included. Design and v0 log formats: [docs/RAW_INGEST.md](docs/RAW_INGEST.md).

```bash
python scripts/generate_raw_flight.py --out data/flights/F_SYNTH_01   # synthetic raw flight
python scripts/ingest_flight.py --flight data/flights/F_SYNTH_01 \
    --instrument configs/instrument.yaml --survey configs/survey_example.yaml
```

## Data conventions

- EM response columns are **moment-normalized dB/dt in V/(A·m⁴)**, positive-decaying convention. Gate times are **milliseconds after turnoff** and live in the sidecar, not the CSV.
- After loading, gates are standardised to `sfz_00, sfz_01, ...`; coordinates to `easting/northing/elevation` (GPS) and `dem` (radar altimeter = bird height AGL).
- Dummy fill values (−9999 etc.) become NaN; all-NaN soundings are dropped at load time.
- QC writes boolean flags (`True` = bad): per-sounding `_qc_neg_early`, `_qc_alt_low/high`, `_qc_dem_mismatch`, `_qc_spike`, `_qc_nonmono`, combined into `sounding_mask`; per-gate noise-floor flags in `_qc_gate_<nn>`. **QC never drops data** — `qc.good_soundings(df)` and `qc.good_gate_array(df)` apply the flags.
- Inversion works on `m = log10(resistivity)` with a log-space data misfit; NaN gates are simply excluded from each sounding's misfit.

## Design notes

- **Forward:** one `TDEMForward` per survey — gate times and layer geometry are fixed; simulations are cached per bird height (rounded to 0.1 m). Geometry is either `concentric_loop` (VTEM-style, Rx at loop centre — exact, no offset clamp) or `offset_dipole` (point dipole + horizontal Rx offset; offsets < 1 m are clamped to avoid Hankel-transform NaNs). When the sidecar declares `bipolar_square` with a base frequency and on-time, the forward models the full bipolar pulse train (#22) and the finite turn-off ramp (#52); an ideal step-off is the fallback. Gates are evaluated at their centre instant — finite-window integration (VTEM/SkyTEM window tables) is Phase 2 (~1% bias, see `configs/README.md`).
- **Inversion:** Occam-style — log-space data misfit + `alpha_s` reference-model damping + `alpha_z` vertical first-difference smoothing, solved bounded with scipy Trust Region Reflective. Layer interfaces are fixed and log-spaced (`depth_min_m` → `depth_max_m`, `n_layers`; the last layer is a basal half-space starting at `depth_max_m`).
- **Stitching:** soundings are inverted in along-line order, each warm-started from its neighbour's solution — cheap lateral continuity. The cold reference model is always also run and the better-fitting of the two is kept, so results don't depend on flight direction (#62); the misfit is the error-normalized `chi` (chi ≈ 1 = fit to assigned errors). True 2D/LCI regularization is Phase 2.
- **Sections:** `plot_section` hangs each sounding's layer column from its own ground elevation (GPS elevation − bird height), so topography is honoured. Colormap is reversed turbo — red = conductive, per EM convention.

## Known limitations — how to read a section

The pipeline is a good **anomaly-finder** and a not-yet-trustworthy **absolute-measurement instrument**. Three limits travel with every section (#94):

1. **IP / chargeable ground is censored, not inverted.** QC and the forward deliberately preserve signed late-time data, but the inversion's misfit currently drops all negative gates before fitting (#78). Over clay-rich or chargeable cover, deep conductive features are suspect: the censoring biases surviving late gates high → spuriously conductive basements.
2. **Below-DOI cells show the prior, not a result.** The section plot fades them and `below_doi` flags them in the model CSV, but the resistivity *value* there is essentially `rho_initial` bent by regularization — do not pick "basement contacts" beneath the DOI line. `rho_sd` can also look confident on prior-pinned deep layers (#81).
3. **Positions and depths lack layback/lever-arm correction** (#70). On a real flight the bird trails ~20–30 m behind and hangs ~15–25 m below the GPS antenna; along-track error flips sign with heading (adjacent lines mis-tie ~40–50 m) and the vertical offset propagates into depth-to-conductor. Must be resolved before interpreting real (non-synthetic) surveys.

Also worth knowing: `chi ≈ 1` means "fit to the assigned errors under the regularization," not a strict reduced-χ² (few gates constrain many layers, #81); and warm-start stitching retains a small flight-direction dependence (#82).

## Running tests

```bash
python3 -m pytest tests/ -v
```

Covers the loader (column-map conventions, validation, cleaning), all six QC checks, forward-model physics (halfspace behaviour, geometry, sign convention), and inversion (recovery of known models, warm-start guard).

## Roadmap

- **Phase 1 (MVP — current):** loader + QC, per-sounding 1D inversion with analytic-Jacobian DOI and per-layer uncertainty, DOI-faded stitched 2D section plot, bipolar-train forward, Streamlit web UI, synthetic end-to-end test
- **Phase 2:** batch all lines with map-view products, real system waveform tables (finite gate-window integration), lateral (LCI/SCI-style) regularization, finite-window forward, VTK export
