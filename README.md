# mvp-tdem

Airborne time-domain electromagnetic (TDEM) data processing and inversion pipeline.

> **Note:** This repo will eventually be transferred to the `lubricate-ai` GitHub organization.

## Overview

Processes helicopter-borne TDEM data (generic CSV input) through QC, 1D layered-earth inversion per sounding, and stitched 2D resistivity sections per flight line.

**Stack:** Python 3.10+, SimPEG, empymod, pandas, matplotlib/plotly.

## Project layout

```
mvp-tdem/
├── tdem/
│   ├── load.py          # CSV ingestion, column mapping, waveform metadata
│   ├── qc.py            # Gate editing, despiking, noise floor
│   ├── forward.py       # SimPEG 1D TDEM forward wrapper
│   ├── invert.py        # Per-sounding inversion → stitched 2D section
│   └── visualize.py     # Resistivity sections, sounding curves, DOI overlay
├── configs/
│   └── example.json     # Column map + inversion params for a survey
├── tests/
├── scripts/
│   └── process_line.py  # CLI: process one flight line end-to-end
├── output/              # Generated plots and model CSVs (gitignored)
├── pyproject.toml
└── README.md
```

## Quickstart

```bash
pip install -e ".[dev]"
python scripts/process_line.py --config configs/example.json --line L01
```

## Phases

- **Phase 1 (MVP):** Loader + QC, 1D inversion, stitched 2D section plot
- **Phase 2:** Batch all lines, DOI estimation, uncertainty per gate, VTK export
