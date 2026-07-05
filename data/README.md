# Data

`synthetic_survey.csv` is a checked-in synthetic helicopter TDEM survey used by the tests, examples, and quickstart. It pairs with `configs/example.json`.

**Contents:** 3 N–S flight lines (`1000`, `2000`, `3000`, 200 m apart), 80 soundings per line at 50 m spacing. A 5 Ω·m conductor at 20–80 m depth crosses all lines between northing 4901500–4902500, in a 200 Ω·m background. Responses are computed with the same SimPEG forward the inversion uses (concentric-loop VTEM-style geometry, 20 gates from 8.4 µs to 14.6 ms), plus 3% multiplicative noise — so processing this survey is a true end-to-end test.

**Columns:** `LINE, FID, Easting, Northing, Elevation, DEM, Latitude, Longitude, SFz[0]..SFz[19]` — the `bracket` gate-naming convention, values in V/(A·m⁴).

Regenerate (e.g. after changing gate times — keep `configs/example.json` in sync):

```bash
python scripts/generate_synthetic.py --out data/synthetic_survey.csv
```

Real survey deliverables should **not** be committed here — point `process_line.py --csv` at them wherever they live.

---

## Synthetic earth model

| Parameter | Value |
|-----------|-------|
| Background resistivity | 200 Ω·m |
| Conductor resistivity | 5 Ω·m |
| Conductor depth (top) | 20 m |
| Conductor depth (bottom) | 80 m |
| Conductor northing extent | 4 901 500 – 4 902 500 m N |
| Bird height | 35 m AGL (constant, all soundings) |
| Projection | UTM zone 11N (EPSG 32611) |

The conductor is a horizontal tabular body crossing all three lines at the same
northing position. It responds on all lines at the same along-line positions,
so the three recovered sections should look identical — a useful consistency
check.

## CSV column reference

| CSV column | Standard name | Description |
|------------|--------------|-------------|
| `LINE` | `line` | Flight line ID (`1000`, `2000`, `3000`) |
| `FID` | `fiducial` | Sounding sequence number |
| `Easting` | `easting` | UTM easting, m |
| `Northing` | `northing` | UTM northing, m |
| `Elevation` | `elevation` | GPS ellipsoid elevation of the bird, m |
| `DEM` | `dem` | Radar altimeter — bird height above ground (m AGL) |
| `Latitude` | `latitude` | WGS84 latitude |
| `Longitude` | `longitude` | WGS84 longitude |
| `SFz[0]` … `SFz[19]` | `sfz_00` … `sfz_19` | Moment-normalised dB/dt, V/(A·m⁴) |

Gate-centre times live in `configs/example.json → gate_times_ms` — **not in
the CSV**. The pair must stay in sync: if you add or remove gates, update
`sfz_n` and `gate_times_ms` together.

## Noise model

3% Gaussian multiplicative noise is applied to each gate value. Gates that
fall below the system noise floor (1 × 10⁻¹² V/(A·m⁴)) are clipped to that
floor. This produces a handful of late-time QC-flaggable gates, exercising
the noise-floor and monotonicity checks during testing.

## Expected inversion results

A healthy inversion of this dataset should show:

- ~150–250 Ω·m background outside the conductor window
- 10–30 Ω·m anomaly at 20–80 m depth over northing 4 901 500 – 4 902 500
  (Occam smoothing blurs sharp contacts; exact 5 Ω·m is not expected at the
  layer boundary — that would require a blocky inversion)
- chi ≈ 1 for the majority of soundings
- converged = True for ≥ 95% of soundings

A recovered median of ~15 Ω·m in the conductor window is healthy for a
36-layer Occam inversion at the default smoothing weights.
