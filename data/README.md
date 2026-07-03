# Data

`synthetic_survey.csv` is a checked-in synthetic helicopter TDEM survey used by the tests, examples, and quickstart. It pairs with `configs/example.json`.

**Contents:** 3 N–S flight lines (`1000`, `2000`, `3000`, 200 m apart), 80 soundings per line at 50 m spacing. A 5 Ω·m conductor at 20–80 m depth crosses all lines between northing 4901500–4902500, in a 200 Ω·m background. Responses are computed with the same SimPEG forward the inversion uses (concentric-loop VTEM-style geometry, 20 gates from 8.4 µs to 14.6 ms), plus 3% multiplicative noise — so processing this survey is a true end-to-end test.

**Columns:** `LINE, FID, Easting, Northing, Elevation, DEM, Latitude, Longitude, SFz[0]..SFz[19]` — the `bracket` gate-naming convention, values in V/(A·m⁴).

Regenerate (e.g. after changing gate times — keep `configs/example.json` in sync):

```bash
python scripts/generate_synthetic.py --out data/synthetic_survey.csv
```

Real survey deliverables should **not** be committed here — point `process_line.py --csv` at them wherever they live.
