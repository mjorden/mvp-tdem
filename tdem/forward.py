"""
1D layered-earth TDEM forward modeling — wrapper around SimPEG's
Simulation1DLayered.

Design notes
------------
- Field data are moment-normalized dB/dt in V/(A·m⁴). SimPEG with a
  unit-moment magnetic dipole source returns dB/dt per unit moment, the
  same normalization — so we simulate with moment=1 and compare directly.
- Sign convention: operators deliver Z-component off-time dB/dt as positive
  decaying values; we take abs() of the simulated response to match.
- Layer parameterization: fixed log-spaced interfaces (depth_min→depth_max,
  n_layers); the model vector is log-resistivity per layer. Fixed geometry
  means one mesh for the whole survey — only bird height varies per sounding.
- Waveform: MVP uses StepOffWaveform. Sidecar gate times are relative to
  turnoff, which matches. Real system waveforms (VTEM trapezoid, SkyTEM
  dual-moment) are a Phase 2 refinement.
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
    resistivities pairs with n_layers - 1 thicknesses.
    """
    if n_layers < 2:
        raise ValueError("n_layers must be >= 2 (layers + basal half-space)")
    interfaces = np.logspace(np.log10(depth_min), np.log10(depth_max), n_layers)
    return np.diff(np.concatenate([[0.0], interfaces]))[: n_layers - 1]


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
    tx_rx_separation_m : horizontal Tx–Rx offset (0 for concentric systems)
    """

    def __init__(
        self,
        gate_times_ms: np.ndarray | list[float],
        thicknesses: np.ndarray,
        tx_rx_separation_m: float = 0.0,
    ):
        self.gate_times_s = np.asarray(gate_times_ms, dtype=float) * 1e-3
        self.thicknesses = np.asarray(thicknesses, dtype=float)
        self.n_layers = len(self.thicknesses) + 1
        # SimPEG's 1D Hankel transform divides by horizontal Tx–Rx offset,
        # so a perfectly coincident geometry (offset=0) produces NaNs. Clamp
        # to 1 m — negligible vs. the system footprint (tens to hundreds of m).
        self.tx_rx_separation_m = max(float(tx_rx_separation_m), 1.0)
        self._sim_cache: dict[float, tdem.Simulation1DLayered] = {}

    def _build_simulation(self, bird_height_m: float) -> tdem.Simulation1DLayered:
        """Construct (and cache) a simulation for one bird height (rounded to 0.1 m)."""
        key = round(bird_height_m, 1)
        if key in self._sim_cache:
            return self._sim_cache[key]

        rx_location = np.array([[self.tx_rx_separation_m, 0.0, key]])
        receiver = tdem.receivers.PointMagneticFluxTimeDerivative(
            rx_location, times=self.gate_times_s, orientation="z"
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
        )
        self._sim_cache[key] = sim
        return sim

    def predict(self, rho_layers: np.ndarray, bird_height_m: float) -> np.ndarray:
        """
        Predict moment-normalized |dB/dt| at each gate.

        Parameters
        ----------
        rho_layers    : resistivity per layer, ohm·m (length n_layers)
        bird_height_m : Tx/Rx height above ground (radar altimeter / DEM channel)

        Returns
        -------
        (n_gates,) array in V/(A·m⁴), positive-decaying convention
        """
        rho_layers = np.asarray(rho_layers, dtype=float)
        if len(rho_layers) != self.n_layers:
            raise ValueError(f"Expected {self.n_layers} layer resistivities, got {len(rho_layers)}")
        sim = self._build_simulation(bird_height_m)
        # sigmaMap is ExpMap → model vector is ln(sigma) = -ln(rho)
        model = np.log(1.0 / rho_layers)
        dpred = sim.dpred(model)
        return np.abs(dpred)

    def predict_log(self, log10_rho: np.ndarray, bird_height_m: float) -> np.ndarray:
        """predict() but taking log10(resistivity) — the inversion's parameterization."""
        return self.predict(10.0 ** np.asarray(log10_rho, dtype=float), bird_height_m)


# ---------------------------------------------------------------------------
# Convenience constructor from sidecar config
# ---------------------------------------------------------------------------

def forward_from_config(config: dict) -> TDEMForward:
    """Build a TDEMForward from a survey sidecar config dict (see configs/example.json)."""
    inv = config["inversion"]
    thk = layer_thicknesses(inv["depth_min_m"], inv["depth_max_m"], inv["n_layers"])
    return TDEMForward(
        gate_times_ms=config["gate_times_ms"],
        thicknesses=thk,
        tx_rx_separation_m=config["system"].get("tx_rx_separation_m", 0.0),
    )
