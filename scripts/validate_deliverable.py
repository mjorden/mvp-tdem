"""
First-contact acceptance harness for an unknown survey deliverable (#95).

When a CSV + sidecar pair arrives in an unfamiliar dialect, run this BEFORE any
processing. It answers "does our pipeline understand this file?" as a checklist
instead of a debugging session: schema, columns, nulls, units plausibility,
gate-time physics, coordinates vs CRS, decay shape, QC flag rate, and one
forward-vs-data overlay at a median sounding.

Usage
-----
    python scripts/validate_deliverable.py --csv survey.csv --config sidecar.json
    python scripts/validate_deliverable.py --csv ... --config ... --json report.json
    python scripts/validate_deliverable.py --csv ... --config ... --skip-forward

Exit codes: 0 = all checks pass, 1 = warnings only, 2 = at least one failure.
Each check reports PASS / WARN / FAIL with a one-line reason; --json writes the
full structured report (CI-friendly).
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


class Report:
    def __init__(self):
        self.checks: list[dict] = []

    def add(self, name: str, status: str, detail: str):
        self.checks.append({"check": name, "status": status, "detail": detail})
        icon = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}[status]
        print(f"  {icon} [{status}] {name}: {detail}")

    @property
    def exit_code(self) -> int:
        if any(c["status"] == FAIL for c in self.checks):
            return 2
        if any(c["status"] == WARN for c in self.checks):
            return 1
        return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--line", default=None,
                    help="line id for the QC/forward spot-checks (default: first line)")
    ap.add_argument("--json", default=None, help="write the structured report here")
    ap.add_argument("--skip-forward", action="store_true",
                    help="skip the (SimPEG-loading) forward overlay check")
    args = ap.parse_args()

    rep = Report()
    print(f"Validating deliverable:\n  csv    = {args.csv}\n  config = {args.config}\n")

    # ── 1. schema + load (units, enums, version, columns, gate times) ────────
    print("— Load & schema —")
    df = config = None
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        try:
            from tdem.load import gate_columns, load_survey
            df, config = load_survey(args.csv, args.config)
            rep.add("load+schema", PASS,
                    f"{len(df):,} soundings, {len(gate_columns(df))} gates loaded")
        except Exception as exc:
            rep.add("load+schema", FAIL, f"{type(exc).__name__}: {exc}")
    for w in wlist:
        rep.add("load warning", WARN, str(w.message).replace("\n", " ")[:160])
    if df is None:
        _finish(rep, args)
        return

    gate_cols = gate_columns(df)

    # ── 2. column coverage & id sanity ───────────────────────────────────────
    print("— Columns & ids —")
    opt_missing = [c for c in ("line", "fiducial", "latitude", "longitude")
                   if c not in df.columns]
    rep.add("optional columns", WARN if opt_missing else PASS,
            f"missing: {opt_missing}" if opt_missing else "line/fiducial/lat/lon all present")
    if "line" in df.columns:
        lines = df["line"].unique()
        rep.add("line ids", PASS, f"{len(lines)} line(s): {sorted(lines)[:8]}"
                + ("…" if len(lines) > 8 else ""))
    if "fiducial" in df.columns and "line" in df.columns:
        nonmono = sum((g["fiducial"].astype(float).diff().dropna() <= 0).sum()
                      for _, g in df.groupby("line"))
        rep.add("fiducial monotonic per line", WARN if nonmono else PASS,
                f"{nonmono} non-increasing step(s)" if nonmono else "strictly increasing")

    # ── 3. null census ────────────────────────────────────────────────────────
    print("— Nulls —")
    gate_nan_frac = float(df[gate_cols].isna().to_numpy().mean())
    status = PASS if gate_nan_frac < 0.05 else (WARN if gate_nan_frac < 0.30 else FAIL)
    rep.add("gate null fraction", status, f"{100*gate_nan_frac:.1f}% of gate values are NaN "
            "(dummies/unparseable) — check null_values/decimal in the sidecar if high")

    # ── 4. units / amplitude / decay-shape plausibility ──────────────────────
    print("— Physics plausibility —")
    g = df[gate_cols].to_numpy(dtype=float)
    finite = g[np.isfinite(g)]
    pos = finite[finite > 0]
    if pos.size:
        med = float(np.median(pos))
        status = PASS if 1e-16 <= med <= 1e-3 else FAIL
        rep.add("amplitude vs V/(A·m⁴)", status,
                f"median |gate| = {med:.2e} " +
                ("(plausible moment-normalized dB/dt)" if status == PASS else
                 "— WRONG UNITS LIKELY (nT/s? ppm? un-normalized?)"))
    neg_frac = float((finite < 0).mean()) if finite.size else 0.0
    rep.add("negative gates", PASS if neg_frac < 0.2 else WARN,
            f"{100*neg_frac:.1f}% negative (late-time IP/bipolar is normal; "
            "a large fraction suggests sign convention or demod issues)")
    # decay slope: median log|amp| vs log t across gates should fall like t^-p
    t = np.asarray(config["gate_times_ms"], dtype=float)
    med_per_gate = np.nanmedian(np.abs(g), axis=0)
    ok_gates = np.isfinite(med_per_gate) & (med_per_gate > 0)
    if ok_gates.sum() >= 4:
        p = -np.polyfit(np.log10(t[ok_gates]), np.log10(med_per_gate[ok_gates]), 1)[0]
        status = PASS if 0.3 <= p <= 4.0 else WARN
        rep.add("decay slope", status,
                f"median |dB/dt| ~ t^-{p:.2f} " +
                ("(physical TDEM decay)" if status == PASS else
                 "— outside the physical band; wrong gate order/times or non-decay data?"))

    # ── 5. coordinates vs declared CRS ────────────────────────────────────────
    print("— Coordinates —")
    east, north = df["easting"].to_numpy(float), df["northing"].to_numpy(float)
    span = float(np.hypot(np.ptp(east), np.ptp(north)))
    rep.add("survey extent", PASS if 10 <= span <= 1e6 else WARN,
            f"{span/1000:.2f} km bounding-diagonal")
    epsg = config.get("survey", {}).get("epsg")
    if epsg and {"latitude", "longitude"} <= set(df.columns):
        try:
            from pyproj import Transformer
            tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
            i = len(df) // 2
            ex, ny = tr.transform(df["longitude"].iloc[i], df["latitude"].iloc[i])
            err = float(np.hypot(ex - east[i], ny - north[i]))
            status = PASS if err < 10 else FAIL
            rep.add("lat/lon vs easting/northing (EPSG)", status,
                    f"projected lat/lon differs from easting/northing by {err:.1f} m "
                    f"under EPSG:{epsg}" + ("" if status == PASS else " — wrong CRS declared?"))
        except Exception as exc:
            rep.add("lat/lon vs EPSG", WARN, f"could not cross-check: {exc}")
    elif epsg:
        rep.add("lat/lon vs EPSG", WARN, "no lat/lon columns — CRS declaration unverified")

    # ── 6. QC spot-check on one line ──────────────────────────────────────────
    print("— QC —")
    try:
        from tdem.load import load_line
        from tdem.qc import run_qc
        line_id = args.line
        if line_id is None and "line" in df.columns:
            line_id = sorted(df["line"].unique())[0]
        df_line = load_line(df, type(df["line"].iloc[0])(line_id)) if line_id is not None else df
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qdf = run_qc(df_line, config, verbose=False)
        s = qdf.attrs["qc_summary"]
        frac = s["flagged_frac"]
        status = PASS if frac < 0.3 else (WARN if frac < 0.7 else FAIL)
        rep.add("QC flag rate", status,
                f"line {line_id}: {s['n_flagged']}/{s['n_total']} flagged "
                f"({100*frac:.0f}%) — per-flag: "
                + ", ".join(f"{k.replace('_qc_','')}={v}" for k, v in s["per_flag"].items() if v))
    except Exception as exc:
        rep.add("QC spot-check", FAIL, f"{type(exc).__name__}: {exc}")
        qdf = None

    # ── 7. forward overlay at a median sounding ──────────────────────────────
    if not args.skip_forward and qdf is not None:
        print("— Forward overlay —")
        try:
            from tdem.forward import forward_from_config
            from tdem.qc import good_soundings
            good = good_soundings(qdf)
            if len(good) == 0:
                rep.add("forward overlay", WARN, "no QC-clean sounding to test")
            else:
                row = good.iloc[len(good) // 2]
                d_obs = row[gate_cols].to_numpy(dtype=float)
                fwd = forward_from_config(config)
                # order-of-magnitude comparison against a mid-range halfspace:
                # catches gross unit/geometry errors without inverting
                pred = fwd.predict(np.full(fwd.n_layers, 100.0), float(row["dem"]))
                m = np.isfinite(d_obs) & (d_obs > 0) & (pred > 0)
                if m.sum() >= 3:
                    ratio = float(np.median(d_obs[m] / pred[m]))
                    status = PASS if 1e-2 <= ratio <= 1e2 else FAIL
                    rep.add("forward overlay", status,
                            f"median obs/pred(100 Ω·m halfspace) = {ratio:.2g} " +
                            ("(within physical range)" if status == PASS else
                             "— orders of magnitude off: units, moment, or geometry wrong"))
                else:
                    rep.add("forward overlay", WARN, "too few comparable gates")
        except Exception as exc:
            rep.add("forward overlay", FAIL, f"{type(exc).__name__}: {exc}")

    _finish(rep, args)


def _finish(rep: Report, args) -> None:
    n = {s: sum(1 for c in rep.checks if c["status"] == s) for s in (PASS, WARN, FAIL)}
    verdict = {0: "ACCEPT", 1: "ACCEPT WITH WARNINGS", 2: "REJECT / INVESTIGATE"}[rep.exit_code]
    print(f"\n{'='*60}\n{verdict}: {n[PASS]} pass, {n[WARN]} warn, {n[FAIL]} fail")
    if args.json:
        Path(args.json).write_text(
            json.dumps({"verdict": verdict, "exit_code": rep.exit_code,
                        "checks": rep.checks}, indent=2), encoding="utf-8")
        print(f"report → {args.json}")
    sys.exit(rep.exit_code)


if __name__ == "__main__":
    main()
