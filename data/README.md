# Data

`synthetic_survey.csv` is a checked-in synthetic helicopter TDEM survey used by the tests, examples, and quickstart. It pairs with `configs/example.json`.

**Contents:** 3 N–S flight lines (`1000`, `2000`, `3000`, 200 m apart), 80 soundings per line at 50 m spacing. A 5 Ω·m conductor at 20–80 m depth crosses all lines between northing 4901500–4902500, in a 200 Ω·m background (concentric-loop VTEM-style geometry, 20 gates from 8.4 µs to 14.6 ms).

The truth is forward-modelled on a **finer, independently-spaced 160-layer mesh** (the inversion uses 36 layers), with the conductor boundaries falling *between* the inversion's interfaces — so recovering it is not an inverse crime (#26): the pipeline must overcome a genuine discretization mismatch, not just re-solve its own discrete operator. Noise is 3% multiplicative **plus** additive Gaussian at the declared floor (#32), so near-floor late gates behave realistically instead of carrying pristine noise many decades below the floor.

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
check. The truth model itself lives on the 160-layer generator mesh; the depths
above are the physical body boundaries, which do **not** coincide with the
inversion's coarser layer tops.

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

Each gate carries **3% multiplicative** Gaussian noise **plus additive**
Gaussian noise at the system noise floor (1 × 10⁻¹² V/(A·m⁴)). There is no
`1e-16` clamp — near-floor and sign-changing late gates are physical and are
handled downstream (per-gate noise-floor flag NaNs |x| < floor; the inversion's
sign-safe misfit copes with the rest). This exercises the noise-floor,
negative-gate, and monotonicity checks realistically; on the shipped survey
about 10% of soundings pick up a QC flag.

## Expected inversion results

Running `examples/01_end_to_end.py` (line 2000) against the shipped data gives:

- **~200 Ω·m background** outside the conductor window (recovered ≈ 203, true 200)
- **~5 Ω·m at 20–80 m depth** over northing 4 901 500 – 4 902 500 (recovered
  median ≈ 5, true 5) — the extra data support from the independent-mesh setup
  resolves the body well; a lower-signal survey would blur it more
- **median chi ≈ 0.8** (data fit to within assigned errors)
- most soundings invert; a fraction are QC-skipped where additive-floor noise
  trips the early-negative or noise-floor checks — as with a real survey

Exact numbers shift with QC thresholds and smoothing weights; the point is that
the conductor is recovered at close to its true resistivity and depth despite
the discretization mismatch.
