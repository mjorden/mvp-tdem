"""
Per-sounding 1D TDEM inversion → stitched 2D section.

Approach
--------
Occam-style regularized least squares on m = log10(resistivity) per layer:

    minimize  || W_d (log d_obs - log d_pred(m)) ||²
            + alpha_s || m - m_ref ||²
            + alpha   || W_z D m ||²      (cell-size-weighted vertical roughness)

solved with scipy.optimize.least_squares (Trust Region Reflective, bounded),
with an Occam cooling loop on the roughness multiplier (#10, #60): start smooth
(alpha = alpha_z * 2^cooling_octaves), halve down to a floor of alpha_z until the
error-normalized misfit chi reaches chi_target, then one bisection stage refines
toward the smoothest fitting multiplier — the accepted model is the SMOOTHEST one
found that fits the data, per sounding, regardless of signal amplitude.

W_z scales each first-difference row by 1/sqrt(spacing between layer tops)
(#11) so log-spaced meshes penalize roughness per metre, not per interface —
without it the shallow model is penalized ~100x harder than the deep model.

Log-space data misfit handles the ~6 decades of decay amplitude; relative
data errors become additive in log space.

Line stitching: soundings are inverted in along-line order and each sounding
is warm-STARTED from its neighbour's solution, but the damping reference
m_ref stays pinned to the config's rho_initial (#18) — warm start changes
the initialization, never the objective. A cheap lateral continuity trick;
true 2D/LCI regularization is Phase 2.

NaN gates (QC-masked or below noise floor) are simply excluded from the
misfit for that sounding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .forward import TDEMForward, forward_from_config, layer_depths
from .load import gate_columns, gate_std_columns
from .qc import good_gate_array


# ---------------------------------------------------------------------------
# Sign-safe log transform for the data misfit (#53)
# ---------------------------------------------------------------------------
# The forward deliberately returns SIGNED dB/dt so that sign-changing transients
# (bipolar train / IP) keep a differentiable fold (forward.py, #5). A plain
# log(max(pred, 1e-300)) misfit throws that away: any trial model predicting a
# non-positive gate produces a saturated ~10⁴σ residual whose finite-difference
# gradient is exactly zero, so TRF stalls or wanders (very resistive trials at
# late gates; warm starts stepping off a conductor onto resistive ground —
# precisely the regime the chi>2 cold-retry guard exists to catch).
#
# We replace it with a C1-continuous "floored log": log(x) above a small
# per-gate floor eps, continued linearly below eps with the matching slope 1/eps.
# For any remotely reasonable fit (pred within ~3 decades of the datum) this is
# identical to log(pred); only as pred approaches zero/negative does it switch to
# a finite, gradient-carrying penalty that pushes the iterate back toward
# positive predictions instead of onto a zero-gradient cliff.

def _log_floored(x: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """log(x) for x > eps; log(eps) + (x-eps)/eps for x <= eps (C1-continuous)."""
    safe = np.maximum(x, eps)                       # keep the log branch finite
    return np.where(x > eps, np.log(safe), np.log(eps) + (x - eps) / eps)


def _dlog_floored(x: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """d/dx of _log_floored: 1/x above eps, 1/eps below (never zero)."""
    return np.where(x > eps, 1.0 / np.maximum(x, eps), 1.0 / eps)


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
    # Linearized appraisal from the analytic Jacobian at the accepted model (#58/#12)
    doi_m: float | None = None          # depth of investigation, metres
    rho_sd: np.ndarray | None = None    # per-layer multiplicative uncertainty (10^σ_m)


# Basal half-space cell: the deepest layer has no lower interface in the mesh,
# so its plotted/exported bottom is a convention. Defined once here (#39) and
# reused by visualize.plot_section / plot_sounding_fit instead of a magic 1.4
# duplicated across modules.
BASAL_CELL_FACTOR = 1.4


def basal_depth_bottom(depths: np.ndarray) -> float:
    """Bottom depth of the basal half-space cell for a layer-top array."""
    return float(depths[-1]) * BASAL_CELL_FACTOR


def along_line_distance(eastings: np.ndarray, northings: np.ndarray) -> np.ndarray:
    """Cumulative arc length along the flight line (#25/#39: one canonical rule)."""
    east = np.asarray(eastings, dtype=float)
    north = np.asarray(northings, dtype=float)
    dx = np.diff(east, prepend=east[0] if len(east) else 0.0)
    dy = np.diff(north, prepend=north[0] if len(north) else 0.0)
    return np.cumsum(np.hypot(dx, dy))


@dataclass
class LineResult:
    """Stitched results for one flight line."""
    line: object
    soundings: list[SoundingResult] = field(default_factory=list)
    # provenance so batch callers can flag bad lines without scraping stdout (#38)
    n_qc_skipped: int = 0     # soundings skipped via sounding_mask
    n_failed: int = 0         # soundings that raised during inversion

    def to_frame(self) -> pd.DataFrame:
        """
        Long-format, self-describing DataFrame: one row per (sounding, layer).

        Self-describing means a consumer of the CSV can reconstruct the section
        geometry without the sidecar (#39): every layer carries both its top
        (`depth_top`) and bottom (`depth_bottom`, basal cell included via the
        single BASAL_CELL_FACTOR rule), `n_gates_used` for fit context, and
        `below_doi` marking cells beneath the depth of investigation (#12).

        `elev_ground` is the GROUND-surface elevation (GPS elevation − bird
        height), NOT the sensor elevation the survey CSV calls `elevation`
        (#37) — the two must not be confused when joining tables.
        """
        rows = []
        for s in self.soundings:
            n_layers = len(s.depths)
            for i, (d, r) in enumerate(zip(s.depths, s.rho)):
                d_bot = (float(s.depths[i + 1]) if i < n_layers - 1
                         else basal_depth_bottom(s.depths))
                rows.append({
                    "line": s.line,
                    "fiducial": s.fiducial,
                    "easting": s.easting,
                    "northing": s.northing,
                    "elev_ground": s.elevation,   # ground surface, not sensor (#37)
                    "layer": i,
                    "depth_top": d,
                    "depth_bottom": d_bot,
                    "rho": r,
                    "chi": s.chi,
                    "converged": s.converged,
                    "n_gates_used": s.n_gates_used,
                    "doi_m": s.doi_m,
                    "below_doi": (s.doi_m is not None and d >= s.doi_m),
                    "rho_sd": s.rho_sd[i] if s.rho_sd is not None else None,
                })
        df = pd.DataFrame(rows)
        if len(self.soundings) > 1:
            cum_dist = along_line_distance(
                [s.easting for s in self.soundings],
                [s.northing for s in self.soundings])
            n_layers = len(self.soundings[0].depths)
            df["distance"] = np.repeat(cum_dist, n_layers)
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
    rho_ref: float | np.ndarray | None = None,
    rho_min: float = 1.0,
    rho_max: float = 10000.0,
    alpha_s: float = 1e-4,
    alpha_z: float = 1.0,
    max_iter: int = 20,
    rel_error: float = 0.05,
    chi_target: float = 1.0,
    cooling_octaves: int = 4,
    gate_sd: np.ndarray | None = None,
    min_gates: int = 3,
    censor_factor: float = 3.0,
) -> tuple[np.ndarray, float, bool, float, np.ndarray]:
    """
    Invert one sounding for a layered resistivity model.

    Occam cooling (#10, #60): the roughness multiplier starts at
    alpha_z * 2**cooling_octaves and is halved until chi <= chi_target (the grid
    floor is alpha_z, i.e. 2**0 — NOT alpha_z/2). One bisection stage then
    searches between the last rejected and first accepted multiplier for a
    smoother model that still fits. Each stage warm-starts from the previous
    stage's model, so the accepted model is the smoothest one found that fits
    the data to its assigned errors. If no multiplier reaches chi_target the
    roughest model is returned with chi > chi_target.

    Parameters
    ----------
    d_obs        : (n_gates,) observed moment-normalized dB/dt; NaN = excluded
    bird_height_m: sensor height above ground for this sounding
    rho_initial  : scalar or per-layer STARTING model (ohm·m)
    rho_ref      : scalar or per-layer damping REFERENCE model; defaults to
                   rho_initial. Kept separate (#18) so a warm start changes
                   the initialization but never the objective.
    rel_error    : minimum relative data error floor — sets W_d floor (#17)
    noise_floor  : additional absolute error floor, same units as d_obs
    min_gates    : minimum number of usable gates required to invert (#59).
                   Below this a ValueError is raised rather than returning a
                   model whose depth range is pure regularization.
    censor_factor: gates with amplitude <= censor_factor*noise_floor are dropped
                   from the misfit to avoid the near-floor upward-selection bias
                   (#27). No effect when noise_floor == 0.
    gate_sd      : optional (n_gates,) per-gate absolute standard deviation from
                   stacking (same units as d_obs).  When provided, combined in
                   quadrature with rel_error (floor) and noise_floor:
                   sd_log = sqrt(rel_error² + (gate_sd/d)² + (noise_floor/d)²)
                   This replaces the flat rel_error model when SFz_std columns
                   are present in the survey CSV (Auken & Christiansen 2004).

    Returns
    -------
    rho (n_layers,), chi, converged

    chi is the error-normalized misfit (#4):
        chi = sqrt(mean(((ln pred − ln obs) / sd_log)²))
    chi ≈ 1 means the data are fit to within their assigned errors.
    converged is False only if the final cooling stage exhausted its
    evaluation budget mid-descent (#15).
    """
    n = fwd.n_layers
    d_obs = np.asarray(d_obs, dtype=float)
    # #27: gates whose amplitude is within ~censor_factor× the noise floor are
    # excluded from the misfit entirely. Near the floor only upward noise
    # fluctuations survive the d>0 cut, AND sd_log = rel + floor/d shrinks for
    # those survivors, so the inversion fits systematically inflated late-time
    # data → spuriously conductive basements. Dropping them removes the bias at
    # the cost of a little depth of investigation (which DOI already reports).
    censor = censor_factor * noise_floor if noise_floor > 0 else 0.0
    use = np.isfinite(d_obs) & (d_obs > censor)
    if use.sum() < min_gates:
        raise ValueError(
            f"Only {int(use.sum())} usable gates (> {censor:.2e}) — "
            f"need at least {min_gates} to invert.")

    def _to_log(rho):
        return np.full(n, np.log10(rho)) if np.isscalar(rho) \
            else np.log10(np.asarray(rho, dtype=float))

    m_ref = _to_log(rho_ref if rho_ref is not None else rho_initial)
    lo, hi = np.log10(rho_min), np.log10(rho_max)
    m = np.clip(_to_log(rho_initial), lo, hi)

    # log-space data weights:
    #   sd(log d) ≈ sqrt(rel_error² + (gate_sd/d)² + (noise_floor/d)²)
    # rel_error acts as a floor; gate_sd (from stacking) provides per-gate
    # uncertainty when SFz_std columns are available (#61).
    log_d = np.log(d_obs[use])
    n_use = int(use.sum())
    std_rel = (np.asarray(gate_sd, dtype=float)[use] / d_obs[use]
               if gate_sd is not None else np.zeros(n_use))
    abs_err = (noise_floor / d_obs[use] if noise_floor > 0
               else np.zeros(n_use))
    sd_log = np.sqrt(rel_error**2 + std_rel**2 + abs_err**2)
    w_d = 1.0 / sd_log   # always (n_use,) array — required by analytic jac

    # per-gate prediction floor for the sign-safe log transform (#53): 0.1% of
    # the datum. A prediction below this is already a >3-decade misfit — deep in
    # "awful fit" territory — so the log branch governs every reasonable model
    # and only the non-physical pred→0/negative region gets the linear penalty.
    eps_pred = 1e-3 * d_obs[use]

    # cell-size-weighted vertical first-difference operator (#11):
    # penalize roughness per metre, not per interface — log-spaced meshes
    # otherwise punish the shallow model ~100x harder than the deep model
    D = np.diff(np.eye(n), axis=0)
    dz = np.diff(layer_depths(fwd.thicknesses))          # n-1 spacings = one per D row (#43.4)
    w_z = np.sqrt(np.mean(dz) / dz)                      # normalized: mean weight ~ 1
    Dw = D * w_z[:, None]
    sqrt_as = np.sqrt(alpha_s)

    def _chi(m_):
        pred = fwd.predict_log(m_, bird_height_m)
        return float(np.sqrt(np.mean(
            ((_log_floored(pred[use], eps_pred) - log_d) / sd_log) ** 2)))

    def _solve(m_start, alpha):
        sqrt_a = np.sqrt(alpha)

        def residuals(m_):
            pred = fwd.predict_log(m_, bird_height_m)
            r_data = w_d * (_log_floored(pred[use], eps_pred) - log_d)
            return np.concatenate([r_data, sqrt_a * (Dw @ m_),
                                   sqrt_as * (m_ - m_ref)])

        def jac(m_):
            pred, J_full = fwd.predict_and_jacobian(m_, bird_height_m)
            # d/dm _log_floored(pred) = dlog_floored(pred) * dpred/dm — nonzero
            # even where pred <= 0, so no zero-gradient cliff (#53)
            dfac = _dlog_floored(pred[use], eps_pred)
            J_data = w_d[:, None] * J_full[use] * dfac[:, None]
            return np.vstack([J_data, sqrt_a * Dw, sqrt_as * np.eye(n)])

        return least_squares(
            residuals, m_start,
            jac=jac,
            bounds=(np.full(n, lo), np.full(n, hi)),
            method="trf",
            max_nfev=max_iter,   # analytic jac: ~1 fun eval/iter, not n+1 (#58)
            x_scale="jac",
        )

    # Occam cooling loop (#10): the SMOOTHEST (largest-α) model that reaches
    # chi_target. Coarse descent on a 2× α grid from α_z·2^octaves down to α_z
    # (the floor is α_z, NOT α_z/2 — #60.5), then ONE bisection stage between the
    # last rejected α (larger/smoother, chi>target) and the first accepted α to
    # halve the roughness error for one extra solve (Constable et al. 1987, #60.1).
    # If the very first (smoothest) grid point already fits, there is no larger α
    # on the grid to search toward, so no bisection is done and an even smoother
    # model may exist (#60.3). If no α reaches the target, the roughest model is
    # returned with chi>target (an honest underfit; converged reflects only the
    # solver's own success, #60.6).
    ok = True
    chi = np.inf
    alpha = alpha_z * 2.0 ** cooling_octaves
    prev_alpha = None                # last α that failed the target (larger)
    m_accept = m
    for k in range(cooling_octaves + 1):
        alpha = alpha_z * 2.0 ** (cooling_octaves - k)
        result = _solve(m, alpha)
        m, ok = result.x, bool(result.success)
        chi = _chi(m)
        if chi <= chi_target:
            m_accept = m
            if prev_alpha is not None:          # bisect toward the smoother side (#60.1)
                a_mid = np.sqrt(alpha * prev_alpha)   # geometric mean of the bracket
                res_mid = _solve(m_accept, a_mid)
                chi_mid = _chi(res_mid.x)
                if chi_mid <= chi_target:        # smoother AND still fits → prefer it
                    m_accept, ok, chi, alpha = res_mid.x, bool(res_mid.success), chi_mid, a_mid
            break
        prev_alpha = alpha
    m = m_accept

    # Linearized appraisal from the analytic Jacobian at the accepted model (#58)
    pred_f, J_f = fwd.predict_and_jacobian(m, bird_height_m)
    dfac_f = _dlog_floored(pred_f[use], eps_pred)             # sign-safe (#53)
    J_data_f = w_d[:, None] * J_f[use] * dfac_f[:, None]      # (n_use, n_layers)

    # DOI via column sensitivity (Christiansen & Auken 2012)
    col_sq = np.sum(J_data_f ** 2, axis=0)
    s_norm = col_sq / (col_sq.max() + 1e-300)
    all_depths = layer_depths(fwd.thicknesses)
    sensitive = np.where(s_norm >= 0.2)[0]
    doi_m = float(all_depths[min(sensitive[-1] + 1, len(all_depths) - 1)]) \
        if len(sensitive) else float(all_depths[1])

    # Per-layer multiplicative uncertainty: 10^sqrt(diag(C_m))
    sqrt_a_final = np.sqrt(alpha)
    J_full_f = np.vstack([J_data_f, sqrt_a_final * Dw, sqrt_as * np.eye(n)])
    M = J_full_f.T @ J_full_f
    try:
        rho_sd = 10.0 ** np.sqrt(np.maximum(np.diag(np.linalg.inv(M)), 0.0))
    except np.linalg.LinAlgError:
        rho_sd = np.full(n, np.nan)

    return 10.0 ** m, chi, ok, doi_m, rho_sd


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
    # ensure soundings are in along-line order before warm-start chaining
    if "fiducial" in df_line.columns:
        df_line = df_line.sort_values("fiducial").reset_index(drop=True)

    inv = config["inversion"]
    noise_floor = config["system"].get("system_noise_floor", 0.0) \
        if inv.get("use_noise_floor", True) else 0.0
    # must mirror invert_sounding's censor_factor default (#27/#59) so the
    # reported n_gates_used matches the gates the misfit actually used
    censor_threshold = 3.0 * noise_floor if noise_floor > 0 else 0.0

    if fwd is None:
        fwd = forward_from_config(config)

    data = good_gate_array(df_line) if any(
        c.startswith("_qc_gate_") for c in df_line.columns
    ) else df_line[gate_columns(df_line)].to_numpy(dtype=float)

    # per-gate standard deviations from stacking (#61): used as the signal
    # component of the error model when SFz_std columns are present
    std_cols = gate_std_columns(df_line)
    std_data = df_line[std_cols].to_numpy(dtype=float) if std_cols else None

    skip = df_line["sounding_mask"].to_numpy() if "sounding_mask" in df_line.columns \
        else np.zeros(len(df_line), dtype=bool)

    line_id = df_line["line"].iloc[0] if "line" in df_line.columns else None
    result = LineResult(line=line_id)
    depths = layer_depths(fwd.thicknesses)

    rho_start: float | np.ndarray = inv["rho_initial"]
    n_failed = 0
    n_gap = 0          # consecutive skips/failures since last success (#18)
    MAX_GAP = 5        # beyond this, a warm start is stale — reset to cold

    for i in range(len(df_line)):
        if skip[i]:
            n_gap += 1
            if n_gap > MAX_GAP:
                rho_start = inv["rho_initial"]
            continue
        row = df_line.iloc[i]
        g_sd = std_data[i] if std_data is not None else None
        kwargs = dict(
            bird_height_m=row["dem"],
            noise_floor=noise_floor,
            rho_ref=inv["rho_initial"],             # #18: reference pinned to config
            rho_min=inv["rho_min"],
            rho_max=inv["rho_max"],
            alpha_s=inv["alpha_s"],
            alpha_z=inv["alpha_z"],
            max_iter=inv["max_iter"],
            rel_error=inv.get("rel_error", 0.05),   # #17: floor when gate_sd present
            chi_target=inv.get("chi_target", 1.0),  # #10
            gate_sd=g_sd,                           # #61: per-gate sd from stacking
        )
        try:
            rho, chi, ok, doi_m, rho_sd = invert_sounding(
                fwd, data[i], rho_initial=rho_start, **kwargs)
            # #62: always ALSO run the cold start and keep the better fit — not
            # only when the warm fit trips the chi>threshold trap guard (#15).
            # Processing order tracks flight direction (sort by fiducial), so a
            # warm-only chain smears anomaly edges in the direction of travel and
            # adjacent lines flown opposite ways get opposite-signed edge
            # artifacts. Making the cold (direction-independent) start an
            # unconditional candidate substantially reduces that hysteresis: a
            # sounding keeps the warm result only when it strictly beats the cold
            # one. It is not a full cure — a genuinely better warm minimum is
            # still kept and remains direction-dependent; the real fix is LCI/SCI
            # (Auken & Christiansen 2004), the acknowledged Phase 2 item.
            # (chi_retry_threshold retained for API compatibility; the cold
            # comparison no longer depends on it.)
            if warm_start and not np.isscalar(rho_start):
                rho2, chi2, ok2, doi_m2, rho_sd2 = invert_sounding(
                    fwd, data[i], rho_initial=inv["rho_initial"], **kwargs)
                if chi2 < chi:
                    rho, chi, ok, doi_m, rho_sd = rho2, chi2, ok2, doi_m2, rho_sd2
        except ValueError:
            n_failed += 1
            n_gap += 1
            if n_gap > MAX_GAP:
                rho_start = inv["rho_initial"]
            continue
        n_gap = 0

        result.soundings.append(SoundingResult(
            easting=row["easting"],
            northing=row["northing"],
            elevation=row["elevation"] - row["dem"],  # ground surface
            line=line_id,
            fiducial=row.get("fiducial", i),
            rho=rho,
            depths=depths,
            chi=chi,
            # #59: count gates the misfit actually USED (finite AND above the
            # censoring threshold), not merely finite — negatives and near-floor
            # gates are finite but excluded, so the old count overstated data
            # support and made weakly-constrained soundings look well-resolved.
            n_gates_used=int((np.isfinite(data[i])
                              & (data[i] > censor_threshold)).sum()),
            converged=ok,
            doi_m=doi_m,
            rho_sd=rho_sd,
        ))

        if warm_start:
            rho_start = rho  # lateral continuity: next sounding starts here

    # provenance on the result so batch callers / tests don't scrape stdout (#38)
    result.n_qc_skipped = int(skip.sum())
    result.n_failed = n_failed

    if verbose:
        n_ok = len(result.soundings)
        if n_ok:
            chi_all = [s.chi for s in result.soundings]
            n_nc = sum(1 for s in result.soundings if not s.converged)
            print(
                f"[invert] line {line_id}: {n_ok} soundings inverted, "
                f"{result.n_qc_skipped} QC-skipped, {n_failed} failed"
                + (f", {n_nc} NOT CONVERGED" if n_nc else "")     # #15
                + f" | median chi {np.median(chi_all):.2f} (1 = fit to errors)"
            )
        else:
            print(f"[invert] line {line_id}: nothing inverted "
                  f"({result.n_qc_skipped} QC-skipped, {n_failed} failed)")
    return result
