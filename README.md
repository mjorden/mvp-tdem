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

The repo ships with a synthetic 3-line survey (`data/synthetic_survey.csv`, lines `1000`/`2000`/`3000`, buried 5 Ω·m conductor at 24–77 m depth) generated on an independent fine truth mesh with realistic multiplicative + noise-floor noise (#26, #32).

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
| `line_<id>_section.png` | Stitched resistivity cross-section (elevation-referenced, RMS strip on top) |
| `line_<id>_decays.png` | All observed decay curves, coloured by along-line position (quick QC view) |
| `line_<id>_model.csv` | Long-format model: one row per (sounding, layer) with `rho`, `depth_top`, `rms`, `distance` |

To regenerate the synthetic survey (e.g. after changing gate times):

```bash
python scripts/generate_synthetic.py --out data/synthetic_survey.csv
```

For library-API usage (no CLI), see [examples/](examples/).

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

- **Forward:** one `TDEMForward` per survey — gate times and layer geometry are fixed; simulations are cached per bird height (rounded to 0.1 m). Geometry is either `concentric_loop` (VTEM-style, Rx at loop centre — exact, no offset clamp) or `offset_dipole` (point dipole + horizontal Rx offset; offsets < 1 m are clamped to avoid Hankel-transform NaNs). When the sidecar declares `tx_waveform: "bipolar_square"`, the predicted off-time transient superposes the current turn-off with all prior alternating-polarity transitions (four time-shifted step-off terms per period, converged in ~2 periods) — a single ideal step-off overpredicts late gates by up to 57% at 25 Hz (#22). Finite turn-off ramps (VTEM trapezoid, SkyTEM dual-moment) are Phase 2.
- **Inversion:** Occam-style — log-space data misfit + `alpha_s` reference-model damping + `alpha_z` vertical first-difference smoothing, solved bounded with scipy Trust Region Reflective. Layer interfaces are fixed and log-spaced (`depth_min_m` → `depth_max_m`, `n_layers`; the last layer is a basal half-space starting at `depth_max_m`).
- **Stitching:** soundings are inverted in along-line order, each warm-started from its neighbour's solution — cheap lateral continuity. A warm-started fit with RMS > 0.3 is retried from the cold reference model and the better fit is kept (warm-start trap guard). True 2D/LCI regularization is Phase 2.
- **Sections:** `plot_section` hangs each sounding's layer column from its own ground elevation (GPS elevation − bird height), so topography is honoured. Colormap is reversed turbo — red = conductive, per EM convention.

## Running tests

```bash
python3 -m pytest tests/ -v
```

Covers the loader (column-map conventions, validation, cleaning), all six QC checks, forward-model physics (halfspace behaviour, geometry, sign convention), and inversion (recovery of known models, warm-start guard).

## Roadmap

- **Phase 1 (MVP — current):** loader + QC, per-sounding 1D inversion, stitched 2D section plot, synthetic end-to-end test
- **Phase 2:** batch all lines with map-view products, real system waveforms, DOI estimation, per-gate uncertainty, lateral (LCI-style) regularization, interactive Plotly sections, VTK export
