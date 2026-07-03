# Raw ingest design — instrument logs → canonical survey CSV + sidecar

**Status:** design sketch, not implemented.
**Scope:** the stage *upstream* of `tdem/load.py`. Everything downstream
(load → qc → invert → visualize) already works against a canonical
deliverable: a flat CSV (one row per sounding, gated `SFz[i]` columns in
V/(A·m⁴)) plus a JSON sidecar (gate times, system params, column map).
This doc designs the converter that produces that pair from our own
instrument's raw flight logs.

## Problem

The logger writes several independent streams during a flight, at
different rates, on different clocks:

| Stream | Contents | Nominal rate |
|---|---|---|
| EM receiver | gated decay (or raw transient) per Tx half-cycle, Rx clock timestamp | 2 × Tx base freq (e.g. 50/s at 25 Hz) |
| GPS | position + time (NMEA or logger CSV) | 1–10 Hz |
| Altimeter | height AGL (radar or laser) | 1–20 Hz |
| Tx monitor | Tx current per half-cycle (if logged) | with EM stream |
| Attitude (optional) | pitch/roll/yaw | 10–100 Hz |

None of this is directly usable by `load_survey()`. The ingest stage must
time-sync, stack, calibrate, geolocate, segment into lines, and emit the
canonical pair.

## Non-goals

- No QC beyond "is this record parseable" — flagging bad soundings is
  `qc.py`'s job and it already exists. Ingest drops nothing it can parse.
- No inversion-related decisions. The sidecar's `inversion` block is
  copied verbatim from the instrument/survey config.

## Pipeline

```
flight dir ─► 1 readers ─► 2 timesync ─► 3 stack ─► 4 calibrate
                                                        │
survey.csv + sidecar.json ◄─ 7 emit ◄─ 6 geometry ◄─ 5 merge
```

### 0. Flight session layout

One directory per flight, discovered by glob:

```
flights/F20260702_01/
├── em_*.log            # EM receiver stream
├── gps_*.log           # GPS stream
├── alt_*.log           # altimeter stream
├── txcur_*.log         # Tx current (optional)
└── instrument.yaml     # static config for this instrument build
```

`instrument.yaml` is the single source of truth for anything that is a
property of the hardware, not of the flight: gate-centre times, Tx
moment/geometry/waveform, Rx coil area and calibration, noise floor,
clock-sync scheme. The emitted sidecar is *generated* from it — never
hand-written — so the sidecar and the data can't drift apart.

### 1. Readers (`ingest/readers.py`)

One parser per stream type. Each returns a time-indexed `DataFrame`
(UTC). Readers are dumb: parse, coerce types, count and skip unparseable
records (logged, not fatal). No physics here.

New stream formats (a firmware change, a different altimeter) mean a new
reader, nothing else changes — this is the only format-specific layer.

### 2. Time sync (`ingest/timesync.py`)

GPS time is the master clock. The EM/altimeter streams are stamped by
their own oscillators, so:

- If the logger records PPS marks or GPS time messages in the EM stream:
  fit EM-clock → GPS-time as a linear (offset + drift) model over the
  flight; apply it.
- If not: fall back to aligning stream start times and assuming nominal
  sample rate, and record `time_sync: "nominal"` in the sidecar so we
  know positional accuracy is degraded.

Output: every record in every stream carries a `t_utc` column. Residuals
of the drift fit are logged; a residual above ~1 ms is an error, not a
warning (at 30 m/s a 1 s error is a 30 m position error, but gate
timing errors of even a fraction of a gate width corrupt the decay).

### 3. Stacking (`ingest/stack.py`)

Raw half-cycles → soundings. Bipolar square wave means alternating
polarity: flip odd half-cycles, then stack `n_stack` consecutive
half-cycles (config; default ≈ 1 sounding per 0.5 s of flight).

- **Robust stack:** trimmed mean (default trim 20%) per gate across the
  window, to reject sferic hits without a separate despiking pass.
- **Per-gate spread:** also keep the std across the stack window per
  gate → emitted as `SFz_std[i]` columns. `load.py`/`qc.py` don't use
  them yet, but the noise-floor logic wants them eventually
  (`_noise_units` note in the example sidecar anticipates this).
- Sounding timestamp = centre of the stack window.
- If the instrument gates on-board but we ever log full waveforms, gate
  integration happens here too (Phase 2; see below).

### 4. Calibration (`ingest/calibrate.py`)

Physics normalization to the canonical unit, V/(A·m⁴):

1. Apply Rx coil calibration (gain, effective area) from
   `instrument.yaml`.
2. Divide by Tx dipole moment. If the Tx current stream exists, use the
   per-stack measured current × turns × loop area; otherwise use the
   nominal moment from config and record `moment: "nominal"` in the
   sidecar.

### 5. Merge (`ingest/merge.py`)

Interpolate GPS (lat, lon, GPS elevation) and altimeter (height AGL)
onto sounding centre times. Linear interpolation; **never extrapolate**
across gaps longer than a threshold (default 2 s) — soundings inside
such gaps get NaN position and are dropped at emit time with a count.
(NaN coordinates are useless downstream; this is the one place ingest
drops data, and it says so.)

### 6. Geometry (`ingest/geometry.py`)

- Project lat/lon → easting/northing with `pyproj`, target EPSG from
  survey config (already a sidecar field).
- `Elevation` = sensor elevation ASL. GPS gives ellipsoidal height; apply
  geoid correction (config: geoid model or fixed offset per survey area).
- `DEM` = the altimeter height AGL itself. Repo convention: despite the
  name, the DEM column carries bird height — `qc._altitude_flag`
  range-checks it directly and `invert` uses it as `bird_height_m`;
  ground elevation is derived downstream as Elevation − DEM.
- **Line segmentation:** prefer an explicit flight-plan file
  (line id ↔ waypoint pairs). Fallback: automatic — split where heading
  changes > 60° sustained for > 5 s or ground speed drops below survey
  speed (turns), number lines sequentially. Either way every sounding
  gets `LINE`.
- `FID` = seconds since flight start × 10, as an int — monotonic,
  unique, and human-mappable back to time.

### 7. Emit (`ingest/emit.py`)

Write exactly what `load_survey()` consumes:

- `survey.csv` — columns `LINE, FID, Easting, Northing, Elevation, DEM,
  Latitude, Longitude, SFz[0]..SFz[n-1], SFz_std[0]..` One row per
  sounding, all flights of a survey concatenated.
- `sidecar.json` — generated: `column_map` (fixed, since we control the
  CSV), `gate_times_ms` and `system` from `instrument.yaml`, `survey`
  block from a small per-survey YAML (name, EPSG, dates), `inversion`
  defaults copied through. Plus an `ingest` provenance block: source
  files + hashes, n_stack, time-sync mode, moment mode, soundings
  dropped, ingest version.

The existing gate-time-vs-off-time validation in `load._validate` then
acts as a free integration check on every emitted sidecar.

## Module & CLI layout

```
tdem/ingest/
├── __init__.py
├── readers.py     # per-stream parsers → time-indexed frames
├── timesync.py    # clock alignment, drift model
├── stack.py       # polarity, robust stacking, per-gate std
├── calibrate.py   # coil cal + moment normalization → V/(A·m⁴)
├── merge.py       # nav/alt interpolation onto sounding times
├── geometry.py    # projection, elevations, line segmentation
└── emit.py        # canonical CSV + generated sidecar
scripts/ingest_flight.py   # CLI: flight dir(s) → survey.csv + sidecar.json
configs/instrument.yaml    # hardware config (new)
```

Each stage is a pure function frame(s)-in → frame-out, same style as the
existing pipeline, so each gets unit tests against tiny fixture logs.
End-to-end test: synthetic raw logs (extend `generate_synthetic.py` to
emit raw streams instead of the final CSV) → ingest → `load_survey` →
`run_qc` runs clean.

New dependency: `pyproj`. Everything else is already in `pyproject.toml`.

## Phasing

- **Phase 1:** gated-on-instrument path. Readers, timesync, stack,
  calibrate (nominal moment ok), merge, geometry with flight-plan line
  ids, emit. Synthetic-raw round-trip test.
- **Phase 2:** full-waveform gate integration, measured Tx current,
  automatic line detection, GPS-antenna→Rx lever-arm correction,
  attitude/tilt correction of the Z-component, external DEM sampling.

## Open questions (need answers from the instrument side)

1. Exact EM log format — binary or ASCII? One file per flight or rolled?
   Does the receiver gate on-board, or log full waveforms?
2. Is there a PPS/GPS-time marker in the EM stream for clock sync?
3. Is Tx current logged per half-cycle?
4. Altimeter type (radar vs laser) and its output format.
5. Is attitude (pitch/roll) logged? Needed for Phase-2 tilt correction.
6. Geoid handling — fixed offset per survey area acceptable for MVP?
