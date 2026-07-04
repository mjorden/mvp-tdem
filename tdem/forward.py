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
    """

    def __init__(
        self,
        gate_times_ms: np.ndarray | list[float],
        thicknesses: np.ndarray,
        tx_geometry: str = "concentric_loop",
        tx_loop_radius_m: float = 13.0,
        tx_rx_separation_m: float = 0.0,
        rx_dz_m: float = 0.0,
        time_filter: str = "key_601_2009",
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
        self._sim_cache: dict[float, tdem.Simulation1DLayered] = {}

    def _build_simulation(self, bird_height_m: float) -> tdem.Simulation1DLayered:
        """Construct (and cache) a simulation for one bird height (rounded to 0.1 m)."""
        key = round(bird_height_m, 1)
        if key in self._sim_cache:
            return self._sim_cache[key]

        if self.tx_geometry == "concentric_loop":
            # Rx at loop centre; SimPEG uses the loop radius as the Hankel
            # offset for central-loop receivers — exact geometry, no clamp (#6)
            rx_location = np.array([[0.0, 0.0, key + self.rx_dz_m]])
            receiver = tdem.receivers.PointMagneticFluxTimeDerivative(
                rx_location, times=self.gate_times_s, orientation="z"
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
            time_filter=self.time_filter,
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
        # SimPEG's z-component off-time dB/dt is negative over a conductive
        # earth; operators deliver it positive-decaying. Negate — do NOT
        # abs(): sign-changing transients (IP effects) must keep their fold
        # so the Jacobian stays differentiable (#5).
        return -dpred

    def predict_log(self, log10_rho: np.ndarray, bird_height_m: float) -> np.ndarray:
        """predict() but taking log10(resistivity) — the inversion's parameterization."""
        return self.predict(10.0 ** np.asarray(log10_rho, dtype=float), bird_height_m)


# ---------------------------------------------------------------------------
# Convenience constructor from sidecar config
# ---------------------------------------------------------------------------

def forward_from_config(config: dict) -> TDEMForward:
    """Build a TDEMForward from a survey sidecar config dict (see configs/example.json)."""
    inv = config["inversion"]
    sysc = config["system"]
    thk = layer_thicknesses(inv["depth_min_m"], inv["depth_max_m"], inv["n_layers"])
    return TDEMForward(
        gate_times_ms=config["gate_times_ms"],
        thicknesses=thk,
        tx_geometry=sysc.get("tx_geometry", "concentric_loop"),
        tx_loop_radius_m=sysc.get("tx_loop_radius_m", 13.0),
        tx_rx_separation_m=sysc.get("tx_rx_separation_m", 0.0),
        rx_dz_m=sysc.get("rx_dz_m", 0.0),
        time_filter=sysc.get("time_filter", "key_601_2009"),
    )
