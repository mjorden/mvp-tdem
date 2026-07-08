"""
1D layered-earth TDEM forward modeling — wrapper around SimPEG's
Simulation1DLayered.

Design notes
------------
- Field data are moment-normalized dB/dt in V/(A·m⁴). SimPEG with a
  unit-moment magnetic dipole source returns dB/dt per unit moment, the
  same normalization — so we simulate with moment=1 and compare directly.
- Sign convention: operators deliver Z-component off-time dB/dt as positive
  decaying values; we NEGATE the simulated response to match (not abs() —
  sign-changing transients must keep their fold, see predict()).
- Layer parameterization: fixed log-spaced interfaces (depth_min→depth_max,
  n_layers); the model vector is log-resistivity per layer. Fixed geometry
  means one mesh for the whole survey — only bird height varies per sounding.
- Waveform (#22, #52): the measured off-time transient is the superposition
  of the final turn-off plus all prior alternating-polarity transitions of
  the bipolar square wave, each with a finite linear turn-off ramp. By
  linearity, for a piecewise-linear current I(τ) that is zero after τ=0,

      d(t) = -∫ I'(τ) · s(t - τ) dτ
           = Σ_ramps  -ΔI · mean of s over u ∈ [t - τ_b, t - τ_a],

  where s(t) is the unit step-off response. We evaluate s ONCE per height
  at every quadrature node (SimPEG's DLF cost is nearly independent of the
  number of output times) and combine in numpy — ~7x faster than driving
  SimPEG with a PiecewiseLinearWaveform, and exact in the r→0 limit.
  Gate times are referenced to the END of the final turn-off ramp.
"""

from __future__ import annotations

import numpy as np

import simpeg.electromagnetics.time_domain as tdem
from simpeg import maps


# ---------------------------------------------------------------------------
# Layer geometry
# ---------------------------------------------------------------------------

def layer_thicknesses(depth_min: float, depth_max: float, n_layers: int) -> np.ndarray:
    """
    Log-spaced layer thicknesses from surface to depth_max.

    Returns (n_layers - 1) thicknesses — SimPEG's convention is that the
    last layer is an infinite half-space, so a model of n_layers
    resistivities pairs with n_layers - 1 thicknesses. The deepest interface
    lands exactly on depth_max, i.e. the half-space starts at depth_max (#9).
    """
    if n_layers < 2:
        raise ValueError("n_layers must be >= 2 (layers + basal half-space)")
    interfaces = np.logspace(np.log10(depth_min), np.log10(depth_max), n_layers - 1)
    return np.diff(np.concatenate([[0.0], interfaces]))


def layer_depths(thicknesses: np.ndarray) -> np.ndarray:
    """Depth to top of each layer (length = n_thicknesses + 1, starts at 0)."""
    return np.concatenate([[0.0], np.cumsum(thicknesses)])


# ---------------------------------------------------------------------------
# Forward simulation
# ---------------------------------------------------------------------------

class TDEMForward:
    """
    Reusable 1D forward operator for one system geometry.

    Build once per survey (gate times fixed); call predict() per sounding
    with that sounding's bird height and trial resistivity model.

    Parameters
    ----------
    gate_times_ms : gate-center times after turnoff, in milliseconds
    thicknesses   : layer thicknesses in m (from layer_thicknesses())
    tx_geometry   : "concentric_loop" (VTEM-style: Rx at centre of a circular
                    Tx loop of tx_loop_radius_m) or "offset_dipole" (point
                    dipole Tx with a horizontal Rx offset, rx_dz vertical offset)
    tx_loop_radius_m   : Tx loop radius, concentric_loop only (#6)
    tx_rx_separation_m : horizontal Tx–Rx offset, offset_dipole only
    rx_dz_m            : Rx vertical offset above the Tx plane (e.g. SkyTEM ~2 m)
    time_filter        : SimPEG cosine/sine DLF filter (#63). Default is the
                         601-pt Key filter: benchmarked ≤0.11% error against
                         the Ward & Hohmann analytic half-space at 10–1000 Ω·m,
                         vs 2.0% for SimPEG's 81-pt default and 8.7% for the
                         201-pt filter (which degrades at high rho).
    tx_frequency_hz    : bipolar base frequency. When given (with
                         tx_on_time_us), the full alternating-polarity pulse
                         train is modeled by superposition (#22).
    tx_on_time_us      : current-on duration per half-cycle, ramps included
    tx_ramp_off_us     : linear turn-off ramp duration (#52). Gate times are
                         relative to the END of the ramp. 0 = ideal step.
    n_prior_periods    : full bipolar periods superposed before the final
                         pulse; contributions decay ~t^(-5/2), 4 is <0.01%
                         from converged at the latest gate.
    """

    N_QUAD = 8   # Gauss-Legendre nodes per ramp integral (in log-time)

    def __init__(
        self,
        gate_times_ms: np.ndarray | list[float],
        thicknesses: np.ndarray,
        tx_geometry: str = "concentric_loop",
        tx_loop_radius_m: float = 13.0,
        tx_rx_separation_m: float = 0.0,
        rx_dz_m: float = 0.0,
        time_filter: str = "key_601_2009",
        tx_frequency_hz: float | None = None,
        tx_on_time_us: float | None = None,
        tx_ramp_off_us: float = 0.0,
        n_prior_periods: int = 4,
    ):
        self.gate_times_s = np.asarray(gate_times_ms, dtype=float) * 1e-3
        self.thicknesses = np.asarray(thicknesses, dtype=float)
        self.n_layers = len(self.thicknesses) + 1
        if tx_geometry not in ("concentric_loop", "offset_dipole"):
            raise ValueError(f"Unknown tx_geometry: {tx_geometry!r}")
        self.tx_geometry = tx_geometry
        self.tx_loop_radius_m = float(tx_loop_radius_m)
        self.rx_dz_m = float(rx_dz_m)
        if tx_geometry == "offset_dipole" and tx_rx_separation_m < 1.0:
            # SimPEG's 1D Hankel transform divides by horizontal Tx–Rx offset;
            # a coincident point dipole (offset=0) produces NaNs. For truly
            # concentric systems use tx_geometry="concentric_loop" instead (#21).
            import warnings
            warnings.warn(
                f"offset_dipole with separation {tx_rx_separation_m} m clamped to 1 m; "
                "use tx_geometry='concentric_loop' for concentric systems"
            )
            tx_rx_separation_m = 1.0
        self.tx_rx_separation_m = float(tx_rx_separation_m)
        self.time_filter = time_filter
        self.tx_frequency_hz = tx_frequency_hz
        self.tx_on_time_s = tx_on_time_us * 1e-6 if tx_on_time_us else None
        self.tx_ramp_off_s = float(tx_ramp_off_us) * 1e-6
        self.n_prior_periods = int(n_prior_periods)
        if tx_frequency_hz is not None and self.tx_on_time_s is None:
            raise ValueError("tx_frequency_hz given without tx_on_time_us — "
                             "cannot place the pulse-train transitions.")
        self._setup_superposition()
        self._sim_cache: dict[float, tdem.Simulation1DLayered] = {}

    # -- waveform superposition (#22, #52) ---------------------------------

    def _transitions(self) -> list[tuple[float, float, float]] | None:
        """
        Linear current ramps (tau_a, tau_b, dI) of the Tx waveform, seconds.

        Current is zero after tau = 0 (the end of the final turn-off ramp —
        the gate-time origin). Returns None for an ideal single step-off.
        Final pulse is positive by convention; polarity alternates backwards.
        """
        r = self.tx_ramp_off_s
        if self.tx_frequency_hz is None:
            return [(-r, 0.0, -1.0)] if r > 0 else None
        half = 1.0 / (2.0 * self.tx_frequency_hz)
        on = self.tx_on_time_s
        out = []
        for k in range(2 * self.n_prior_periods + 1):
            pol = 1.0 if k % 2 == 0 else -1.0
            t_end = -k * half
            out.append((t_end - r, t_end, -pol))            # turn-off ramp
            out.append((t_end - on, t_end - on + r, pol))   # turn-on ramp
        return out

    def _setup_superposition(self) -> None:
        """
        Precompute step-response evaluation times and weights so that
        predict() is  d_i = sum_j W[i, j] * s(t_eval[i, j]),  one row per gate.

        Each ramp (tau_a, tau_b, dI) contributes -dI * mean(s) over
        u in [t - tau_b, t - tau_a]; the mean is Gauss-Legendre in ln(u)
        (s is locally power-law, so log-space quadrature stays accurate even
        when the ramp spans decades of u at the earliest gates). An
        instantaneous step (tau_a == tau_b) is a single node of weight -dI.
        """
        trans = self._transitions()
        if trans is None:
            self._eval_times = None      # pure step-off: sim times = gate times
            return

        gt = self.gate_times_s
        nodes, weights = [], []
        gl_x, gl_w = np.polynomial.legendre.leggauss(self.N_QUAD)
        for tau_a, tau_b, dI in trans:
            u_lo = gt - tau_b
            u_hi = gt - tau_a
            if tau_b - tau_a <= 0.0:                        # instantaneous step
                nodes.append(u_lo[:, None])
                weights.append(np.full((len(gt), 1), -dI))
                continue
            v_lo, v_hi = np.log(u_lo), np.log(u_hi)         # (n_gates,)
            v = 0.5 * (v_hi - v_lo)[:, None] * gl_x + 0.5 * (v_hi + v_lo)[:, None]
            u = np.exp(v)                                   # (n_gates, N_QUAD)
            # mean of s over [u_lo, u_hi]:  (1/(u_hi-u_lo)) * int s(e^v) e^v dv
            w = (0.5 * (v_hi - v_lo)[:, None] * gl_w * u
                 / (u_hi - u_lo)[:, None]) * (-dI)
            nodes.append(u)
            weights.append(w)

        t_flat = np.concatenate(nodes, axis=1)              # (n_gates, n_nodes)
        self._weights = np.concatenate(weights, axis=1)
        self._eval_times, inverse = np.unique(t_flat, return_inverse=True)
        self._eval_inverse = inverse.reshape(t_flat.shape)

    _SIM_CACHE_MAX = 512   # bound the per-height cache (#68.7)

    def _build_simulation(self, bird_height_m: float) -> tdem.Simulation1DLayered:
        """Construct (and cache) a simulation for one bird height (rounded to 0.1 m)."""
        key = round(bird_height_m, 1)
        if key in self._sim_cache:
            return self._sim_cache[key]
        # continuous 20–90 m altimetry at 0.1 m rounding is ~700 distinct sims
        # (#68.7); cap the cache so a long survey can't grow it without bound.
        # Soundings are processed in flight order, so the oldest heights are the
        # least likely to recur — drop the oldest inserted key (FIFO).
        if len(self._sim_cache) >= self._SIM_CACHE_MAX:
            self._sim_cache.pop(next(iter(self._sim_cache)))

        # superposition mode (#22, #52) needs s(t) at every quadrature node;
        # DLF cost is nearly independent of the number of output times
        sim_times = self.gate_times_s if self._eval_times is None else self._eval_times

        if self.tx_geometry == "concentric_loop":
            # Rx at loop centre; SimPEG uses the loop radius as the Hankel
            # offset for central-loop receivers — exact geometry, no clamp (#6)
            rx_location = np.array([[0.0, 0.0, key + self.rx_dz_m]])
            receiver = tdem.receivers.PointMagneticFluxTimeDerivative(
                rx_location, times=sim_times, orientation="z"
            )
            r = self.tx_loop_radius_m
            source = tdem.sources.CircularLoop(
                [receiver],
                location=np.array([0.0, 0.0, key]),
                orientation="z",
                radius=r,
                current=1.0 / (np.pi * r ** 2),  # moment = I·πr² = 1 → output in V/(A·m⁴)
                waveform=tdem.sources.StepOffWaveform(),
            )
        else:  # offset_dipole
            rx_location = np.array([[self.tx_rx_separation_m, 0.0, key + self.rx_dz_m]])
            receiver = tdem.receivers.PointMagneticFluxTimeDerivative(
                rx_location, times=sim_times, orientation="z"
            )
            source = tdem.sources.MagDipole(
                [receiver],
                location=np.array([0.0, 0.0, key]),
                orientation="z",
                moment=1.0,  # unit moment → output is moment-normalized (V/(A·m⁴))
                waveform=tdem.sources.StepOffWaveform(),
            )
        survey = tdem.Survey([source])

        sim = tdem.Simulation1DLayered(
            survey=survey,
            thicknesses=self.thicknesses,
            sigmaMap=maps.ExpMap(nP=self.n_layers),
            time_filter=self.time_filter,
        )
        self._sim_cache[key] = sim
        return sim

    def predict(self, rho_layers: np.ndarray, bird_height_m: float) -> np.ndarray:
        """
        Predict moment-normalized dB/dt at each gate, per unit PEAK Tx moment.

        Parameters
        ----------
        rho_layers    : resistivity per layer, ohm·m (length n_layers)
        bird_height_m : Tx/Rx height above ground (radar altimeter / DEM channel)

        Returns
        -------
        (n_gates,) array in V/(A·m⁴), positive-decaying convention. With the
        bipolar train (#22) the prior opposite-polarity transitions SUBTRACT
        from the step-off response — late gates are substantially smaller
        than a single step-off predicts, and can legitimately approach zero.
        """
        rho_layers = np.asarray(rho_layers, dtype=float)
        if len(rho_layers) != self.n_layers:
            raise ValueError(f"Expected {self.n_layers} layer resistivities, got {len(rho_layers)}")
        sim = self._build_simulation(bird_height_m)
        # sigmaMap is ExpMap → model vector is ln(sigma) = -ln(rho)
        model = np.log(1.0 / rho_layers)
        # SimPEG's z-component off-time dB/dt is negative over a conductive
        # earth; operators deliver it positive-decaying. Negate — do NOT
        # abs(): sign-changing transients (IP effects) must keep their fold
        # so the Jacobian stays differentiable (#5).
        s = -sim.dpred(model)
        if self._eval_times is None:
            return s
        return (self._weights * s[self._eval_inverse]).sum(axis=1)

    def predict_log(self, log10_rho: np.ndarray, bird_height_m: float) -> np.ndarray:
        """predict() but taking log10(resistivity) — the inversion's parameterization."""
        return self.predict(10.0 ** np.asarray(log10_rho, dtype=float), bird_height_m)

    def predict_and_jacobian(
        self, log10_rho: np.ndarray, bird_height_m: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict dB/dt and the analytic Jacobian ∂pred[i]/∂log10_rho[k].

        One SimPEG pass (dpred + getJ) for both. Analytic sensitivity avoids
        the n+1 forward calls per iteration that finite-difference Jacobians
        require and eliminates FD noise near the 1e-300 clamp (#58).

        Chain rule: model = ln(σ) = -ln(10)·log10_rho, so
            ∂pred/∂log10_rho = ∂pred/∂ln_σ · (-ln10)  (scalar per layer)
        For step-off  pred = -dpred_simpeg  →
            ∂pred/∂log10_rho = J_simpeg · ln10
        For the superposition  pred = Σ W[i,j] · s(t_eval[j])  and linearity
        applies node-wise:  J[i,k] = Σ_j W[i,j] · J_simpeg[j,k] · ln10

        Returns
        -------
        pred : (n_gates,) same units/convention as predict()
        J    : (n_gates, n_layers)  ∂pred[i]/∂log10_rho[k]
        """
        log10_rho = np.asarray(log10_rho, dtype=float)
        if len(log10_rho) != self.n_layers:
            raise ValueError(
                f"Expected {self.n_layers} log10-rho values, got {len(log10_rho)}"
            )
        model = np.log(1.0 / 10.0 ** log10_rho)          # ln(sigma)
        sim = self._build_simulation(bird_height_m)
        dpred_raw = sim.dpred(model)                       # sets internal state for getJ
        J_raw = np.asarray(sim.getJ(model))                # (n_eval, n_layers)
        # ∂s/∂log10_rho = J_simpeg · ln10  (s = -dpred_simpeg)
        J_s = J_raw * np.log(10.0)
        if self._eval_times is None:
            pred = -dpred_raw
            J = J_s
        else:
            s = -dpred_raw
            pred = (self._weights * s[self._eval_inverse]).sum(axis=1)
            J = (self._weights[:, :, None] * J_s[self._eval_inverse]).sum(axis=1)
        return pred, J


# ---------------------------------------------------------------------------
# Convenience constructor from sidecar config
# ---------------------------------------------------------------------------

def forward_from_config(config: dict) -> TDEMForward:
    """
    Build a TDEMForward from a survey sidecar config dict (see configs/example.json).

    When the sidecar declares tx_waveform="bipolar_square" with a base
    frequency and on-time, the full pulse train is modeled (#22); the
    turn-off ramp comes from tx_ramp_off_us (#52, 0 = ideal step if absent).
    """
    inv = config["inversion"]
    sysc = config["system"]
    thk = layer_thicknesses(inv["depth_min_m"], inv["depth_max_m"], inv["n_layers"])

    # A2: only bipolar_square activates the pulse-train model; every other value
    # would SILENTLY fall through to an ideal step-off (wrong early gates). Reject
    # unknown waveforms loudly instead — a typo or an unsupported system waveform
    # (VTEM trapezoid, half-sine, …) must fail, not model the wrong physics.
    _SUPPORTED_WAVEFORMS = {"bipolar_square", "step_off", "step", "", None}
    wf = sysc.get("tx_waveform")
    if wf not in _SUPPORTED_WAVEFORMS:
        raise ValueError(
            f"Unsupported tx_waveform {wf!r}. Supported: 'bipolar_square' (pulse "
            "train) or 'step_off'/'step' (ideal step). VTEM-trapezoid / half-sine / "
            "measured-waveform tables are not modeled yet — do not silently degrade."
        )

    # A3: the receiver is hardcoded Z. rx_orientation is in the sidecar but was
    # never wired in, so a non-Z config was silently ignored (Z response labeled X).
    orient = str(sysc.get("rx_orientation", "Z")).upper()
    if orient != "Z":
        raise NotImplementedError(
            f"rx_orientation={orient!r} is not supported — only Z-component dB/dt is "
            "modeled. A non-Z orientation would silently return the Z response."
        )

    bipolar = (wf == "bipolar_square"
               and sysc.get("tx_frequency_hz") and sysc.get("tx_on_time_us"))
    return TDEMForward(
        gate_times_ms=config["gate_times_ms"],
        thicknesses=thk,
        tx_geometry=sysc.get("tx_geometry", "concentric_loop"),
        tx_loop_radius_m=sysc.get("tx_loop_radius_m", 13.0),
        tx_rx_separation_m=sysc.get("tx_rx_separation_m", 0.0),
        rx_dz_m=sysc.get("rx_dz_m", 0.0),
        time_filter=sysc.get("time_filter", "key_601_2009"),
        tx_frequency_hz=sysc["tx_frequency_hz"] if bipolar else None,
        tx_on_time_us=sysc["tx_on_time_us"] if bipolar else None,
        tx_ramp_off_us=sysc.get("tx_ramp_off_us", 0.0),
    )
