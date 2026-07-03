"""
Per-sounding 1D TDEM inversion → stitched 2D section.

Approach (MVP)
--------------
Occam-style regularized least squares on m = log10(resistivity) per layer:

    minimize  || W_d (log d_obs - log d_pred(m)) ||²
            + alpha_s || m - m_ref ||²
            + alpha_z || D m ||²          (first-difference vertical roughness)

solved with scipy.optimize.least_squares (Trust Region Reflective, bounded).
Log-space data misfit handles the ~6 decades of decay amplitude; relative
data errors become additive in log space.

Line stitching: soundings are inverted in along-line order and each sounding
is warm-started from its neighbour's solution — cheap lateral continuity
without a true 2D regularization (that's Phase 2 / SimPEG LCI territory).

NaN gates (QC-masked or below noise floor) are simply excluded from the
misfit for that sounding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .forward import TDEMForward, forward_from_config, layer_depths
from .load import gate_columns
from .qc import good_gate_array


# ---------------------------------------------------------------------------
# Results containers
# ---------------------------------------------------------------------------

@dataclass
class SoundingResult:
    """Inversion result for one sounding."""
    easting: float
    northing: float
    elevation: float          # ground elevation (GPS elev - bird height), m
    line: object
    fiducial: object
    rho: np.ndarray           # resistivity per layer, ohm·m
    depths: np.ndarray        # depth to top of each layer, m
    chi: float                # error-normalized misfit; chi ~ 1 = fit to errors (#4)
    n_gates_used: int
    converged: bool


@dataclass
class LineResult:
    """Stitched results for one flight line."""
    line: object
    soundings: list[SoundingResult] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        """Long-format DataFrame: one row per (sounding, layer) — plot-ready."""
        rows = []
        for s in self.soundings:
            for i, (d, r) in enumerate(zip(s.depths, s.rho)):
                rows.append({
                    "line": s.line,
                    "fiducial": s.fiducial,
                    "easting": s.easting,
                    "northing": s.northing,
                    "elevation": s.elevation,
                    "layer": i,
                    "depth_top": d,
                    "rho": r,
                    "chi": s.chi,
                    "converged": s.converged,
                })
        df = pd.DataFrame(rows)
        # along-line distance from first sounding
        if len(self.soundings) > 1:
            x0, y0 = self.soundings[0].easting, self.soundings[0].northing
            df["distance"] = np.hypot(df["easting"] - x0, df["northing"] - y0)
        elif len(df):
            df["distance"] = 0.0
        return df


# ---------------------------------------------------------------------------
# Single-sounding inversion
# ---------------------------------------------------------------------------

def invert_sounding(
    fwd: TDEMForward,
    d_obs: np.ndarray,
    bird_height_m: float,
    *,
    noise_floor: float = 0.0,
    rho_initial: float | np.ndarray = 100.0,
    rho_min: float = 1.0,
    rho_max: float = 10000.0,
    alpha_s: float = 1e-4,
    alpha_z: float = 1.0,
    max_iter: int = 20,
    rel_error: float = 0.05,
) -> tuple[np.ndarray, float, bool]:
    """
    Invert one sounding for a layered resistivity model.

    Parameters
    ----------
    d_obs        : (n_gates,) observed moment-normalized dB/dt; NaN = excluded
    bird_height_m: sensor height above ground for this sounding
    rho_initial  : scalar or per-layer starting/reference model (ohm·m)
    rel_error    : assumed relative data error — sets W_d (#17: plumb from
                   config's inversion.rel_error)
    noise_floor  : additional absolute error floor, same units as d_obs

    Returns
    -------
    rho (n_layers,), chi, converged

    chi is the error-normalized misfit (#4):
        chi = sqrt(mean(((ln pred − ln obs) / sd_log)²))
    chi ≈ 1 means the data are fit to within their assigned errors;
    chi >> 1 underfit; chi << 1 overfit (errors overestimated).
    """
    n = fwd.n_layers
    d_obs = np.asarray(d_obs, dtype=float)
    use = np.isfinite(d_obs) & (d_obs > 0)
    if use.sum() < 3:
        raise ValueError(f"Only {use.sum()} usable gates — need at least 3 to invert.")

    m_ref = np.full(n, np.log10(rho_initial)) if np.isscalar(rho_initial) \
        else np.log10(np.asarray(rho_initial, dtype=float))
    lo, hi = np.log10(rho_min), np.log10(rho_max)
    m0 = np.clip(m_ref, lo, hi)

    # log-space data weights: sd(log d) ≈ rel_error + floor/d
    log_d = np.log(d_obs[use])
    sd_log = rel_error + (noise_floor / d_obs[use] if noise_floor > 0 else 0.0)
    w_d = 1.0 / sd_log

    # vertical first-difference operator
    D = np.diff(np.eye(n), axis=0)
    sqrt_as = np.sqrt(alpha_s)
    sqrt_az = np.sqrt(alpha_z)

    def residuals(m):
        pred = fwd.predict_log(m, bird_height_m)
        r_data = w_d * (np.log(np.maximum(pred[use], 1e-300)) - log_d)
        r_smooth = sqrt_az * (D @ m)
        r_ref = sqrt_as * (m - m_ref)
        return np.concatenate([r_data, r_smooth, r_ref])

    result = least_squares(
        residuals,
        m0,
        bounds=(np.full(n, lo), np.full(n, hi)),
        method="trf",
        max_nfev=max_iter * (n + 1),
        x_scale="jac",
    )

    m = result.x
    pred = fwd.predict_log(m, bird_height_m)
    # error-normalized chi (#4) — same weights the misfit was minimized under
    chi = float(np.sqrt(np.mean(
        ((np.log(np.maximum(pred[use], 1e-300)) - log_d) / sd_log) ** 2)))
    return 10.0 ** m, chi, bool(result.success)


# ---------------------------------------------------------------------------
# Line inversion (stitched 1D)
# ---------------------------------------------------------------------------

def invert_line(
    df_line: pd.DataFrame,
    config: dict,
    fwd: TDEMForward | None = None,
    *,
    warm_start: bool = True,
    chi_retry_threshold: float = 2.0,
    verbose: bool = True,
) -> LineResult:
    """
    Invert every sounding on one flight line, in along-line order.

    df_line should already be QC'd (run_qc) — per-gate flags become NaNs via
    good_gate_array(); soundings with sounding_mask=True are skipped.

    Warm-start trap guard: a sounding warm-started from a very different
    neighbour (e.g. stepping off a conductor onto resistive ground) can get
    stuck in a poor local minimum. If a warm-started fit has chi above
    chi_retry_threshold (chi ~ 1 = fit to within errors, so 2 = clearly
    underfit, #4), the sounding is re-inverted from the config's rho_initial
    and the better of the two is kept.
    """
    inv = config["inversion"]
    noise_floor = config["system"].get("system_noise_floor", 0.0) \
        if inv.get("use_noise_floor", True) else 0.0

    if fwd is None:
        fwd = forward_from_config(config)

    data = good_gate_array(df_line) if any(
        c.startswith("_qc_gate_") for c in df_line.columns
    ) else df_line[gate_columns(df_line)].to_numpy(dtype=float)

    skip = df_line["sounding_mask"].to_numpy() if "sounding_mask" in df_line.columns \
        else np.zeros(len(df_line), dtype=bool)

    line_id = df_line["line"].iloc[0] if "line" in df_line.columns else None
    result = LineResult(line=line_id)
    depths = layer_depths(fwd.thicknesses)

    rho_start: float | np.ndarray = inv["rho_initial"]
    n_failed = 0

    for i in range(len(df_line)):
        if skip[i]:
            continue
        row = df_line.iloc[i]
        kwargs = dict(
            bird_height_m=row["dem"],
            noise_floor=noise_floor,
            rho_min=inv["rho_min"],
            rho_max=inv["rho_max"],
            alpha_s=inv["alpha_s"],
            alpha_z=inv["alpha_z"],
            max_iter=inv["max_iter"],
            rel_error=inv.get("rel_error", 0.05),   # #17
        )
        try:
            rho, chi, ok = invert_sounding(fwd, data[i], rho_initial=rho_start, **kwargs)
            # warm-start trap guard: bad fit from a warm start → retry cold
            if chi > chi_retry_threshold and warm_start \
                    and not np.isscalar(rho_start):
                rho2, chi2, ok2 = invert_sounding(
                    fwd, data[i], rho_initial=inv["rho_initial"], **kwargs)
                if chi2 < chi:
                    rho, chi, ok = rho2, chi2, ok2
        except ValueError:
            n_failed += 1
            continue

        result.soundings.append(SoundingResult(
            easting=row["easting"],
            northing=row["northing"],
            elevation=row["elevation"] - row["dem"],  # ground surface
            line=line_id,
            fiducial=row.get("fiducial", i),
            rho=rho,
            depths=depths,
            chi=chi,
            n_gates_used=int(np.isfinite(data[i]).sum()),
            converged=ok,
        ))

        if warm_start:
            rho_start = rho  # lateral continuity: next sounding starts here

    if verbose:
        n_ok = len(result.soundings)
        if n_ok:
            chi_all = [s.chi for s in result.soundings]
            print(
                f"[invert] line {line_id}: {n_ok} soundings inverted, "
                f"{int(skip.sum())} QC-skipped, {n_failed} failed | "
                f"median chi {np.median(chi_all):.2f} (1 = fit to errors)"
            )
        else:
            print(f"[invert] line {line_id}: nothing inverted "
                  f"({int(skip.sum())} QC-skipped, {n_failed} failed)")
    return result
