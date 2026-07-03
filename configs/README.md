# Survey sidecar config reference

Each survey CSV is paired with a JSON sidecar that carries everything the CSV doesn't: column mapping, system parameters, gate times, and inversion defaults. `configs/example.json` is a complete working example (it matches `data/synthetic_survey.csv`).

Keys starting with `_` are comments — ignored by the loader.

## `survey` — provenance metadata

Free-form; not used by the pipeline, but keep it filled in for report traceability.

| Key | Example | Notes |
|-----|---------|-------|
| `name` | `"example_survey"` | Survey identifier |
| `system` | `"VTEM"` | Acquisition system |
| `date` | `"2024-06-15"` | Flight date |
| `contractor` | `"Geotech Ltd."` | |
| `epsg` | `32611` | CRS of easting/northing |

## `column_map` — CSV column names → standard names

Maps the operator's column headers to the pipeline's standard names. A mapped column that's absent from the CSV is silently skipped for the optional ones; `easting`, `northing`, `elevation`, and `dem` are required after mapping.

| Key | Standard meaning |
|-----|------------------|
| `line` | Flight line ID |
| `fiducial` | Sounding sequence number |
| `easting`, `northing` | Projected coordinates (m) |
| `elevation` | GPS/ellipsoid elevation of the bird (m) |
| `dem` | Radar altimeter — bird height above ground (m AGL) |
| `latitude`, `longitude` | Optional geographic coordinates |

Gate columns are described, not listed:

| Key | Example | Notes |
|-----|---------|-------|
| `sfz_prefix` | `"SFz"` | Common prefix of the gate columns |
| `sfz_n` | `20` | Number of gates — **must equal `len(gate_times_ms)`** |
| `sfz_format` | `"bracket"` | `bracket` → `SFz[0]…SFz[19]` · `underscore` → `SFz_00…SFz_19` · `zero_padded` → `SFz00…SFz19` |

## `system` — acquisition system parameters

| Key | Example | Notes |
|-----|---------|-------|
| `tx_moment_am2` | `420000` | Informational (data are already moment-normalized) |
| `tx_frequency_hz` | `25` | Base frequency — used to validate gate times against the off-time window |
| `tx_waveform` | `"bipolar_square"` | Informational in MVP (forward uses step-off) |
| `tx_on_time_us` | `4000` | On-time, µs — enters the off-time check |
| `rx_coil_area_m2` | `99.7` | Informational |
| `tx_geometry` | `"concentric_loop"` | `concentric_loop` (VTEM-style, Rx at loop centre) or `offset_dipole` |
| `tx_loop_radius_m` | `13.0` | `concentric_loop` only |
| `tx_rx_separation_m` | — | `offset_dipole` only; horizontal Tx–Rx offset (values < 1 m are clamped) |
| `rx_dz_m` | `0.0` | Rx vertical offset above the Tx plane (e.g. SkyTEM ≈ 2 m) |
| `rx_orientation` | `"Z"` | Only Z supported in MVP |
| `system_noise_floor` | `1e-12` | V/(A·m⁴) — same units as the gate data. Gates at/below this are QC-flagged, and it feeds the inversion's error model when `inversion.use_noise_floor` is true |

## `gate_times_ms` — gate-centre times

Array of gate-centre times in **milliseconds after waveform turnoff**. Two hard constraints, both enforced at load time:

1. Length must equal `column_map.sfz_n`.
2. The last gate must fit inside the off-time window: `1/(2·tx_frequency_hz) − tx_on_time`. At 25 Hz with 4 ms on-time that's 16 ms — a gate at or beyond that is physically impossible and the loader raises.

## `inversion` — per-survey inversion defaults

Consumed by `tdem.invert.invert_line()` / `invert_sounding()` and `tdem.forward.forward_from_config()`.

| Key | Default in example | Notes |
|-----|--------------------|-------|
| `n_layers` | `30` | Layer count including the basal half-space |
| `depth_min_m` | `0.5` | Depth of the first interface |
| `depth_max_m` | `300` | Depth where the basal half-space starts; interfaces log-spaced between |
| `rho_initial` | `100` | Starting + reference model, Ω·m |
| `rho_min`, `rho_max` | `1`, `10000` | Hard bounds, Ω·m |
| `alpha_s` | `1e-4` | Reference-model damping weight |
| `alpha_z` | `1.0` | Vertical first-difference smoothing weight |
| `max_iter` | `20` | Solver iteration budget per sounding |
| `use_noise_floor` | `true` | Fold `system.system_noise_floor` into the data error model |

## Adding a new survey

1. Copy `example.json` → `<survey>.json`.
2. Set `column_map` to match the CSV headers (check `sfz_format` against the actual gate column names).
3. Fill `system` and `gate_times_ms` from the operator's logistics/deliverable report.
4. Sanity check: `python scripts/process_line.py --csv <survey>.csv --config configs/<survey>.json --line <id>` — the loader will raise immediately on column-map or gate-time mismatches.
