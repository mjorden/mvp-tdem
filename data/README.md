# Data

`synthetic_survey.csv` is a checked-in synthetic helicopter TDEM survey used by the tests, examples, and quickstart. It pairs with `configs/example.json`.

**Contents:** 3 N–S flight lines (`1000`, `2000`, `3000`, 200 m apart), 80 soundings per line at 50 m spacing. A 5 Ω·m conductor at 24–77 m depth crosses all lines between northing 4901500–4902500, in a 200 Ω·m background. To avoid an inverse crime (#26), truth is computed on a fine mesh spaced independently of the inversion's (1 m cells through the zone of interest; conductor boundaries off the inversion's layer interfaces), with the 25 Hz bipolar waveform (#22), altimeter error on the recorded `DEM` channel, and 3% multiplicative noise plus additive noise at the declared 1e-12 floor (#32) — gates below the floor are noise-dominated, negatives included, as in real data. It is a regression fixture for the pipeline's bookkeeping, not physics validation.

**Columns:** `LINE, FID, Easting, Northing, Elevation, DEM, Latitude, Longitude, SFz[0]..SFz[19]` — the `bracket` gate-naming convention, values in V/(A·m⁴).

Regenerate (e.g. after changing gate times — keep `configs/example.json` in sync):

```bash
python scripts/generate_synthetic.py --out data/synthetic_survey.csv
```

Real survey deliverables should **not** be committed here — point `process_line.py --csv` at them wherever they live.
