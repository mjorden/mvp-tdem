"""
Sidecar schema: version, enumerations, units, and the validator (#87, #83).

This module is the SINGLE SOURCE OF TRUTH for every enumerated sidecar value.
forward.py / load.py import these sets rather than re-typing string literals, so
"add a waveform" or "add a geometry" is: implement it, register it here, done —
and an unknown value fails LOUDLY at the door instead of silently taking a
wrong code path deep in the pipeline (the pre-#87 failure mode: an unrecognized
tx_waveform silently modeled an ideal step-off).

Deliberately dependency-light (no SimPEG import) so load_survey can validate
without dragging in the physics stack.

Versioning policy
-----------------
`schema_version` is written by emit and checked here. A sidecar with NO version
field is treated as version 1 (every sidecar written before the field existed).
A sidecar with a version NEWER than this code understands is rejected — the
right response to a config from the future is a loud error, not a guess.
"""

from __future__ import annotations

import warnings

SCHEMA_VERSION = 1

# -- enumerations (register new values here; implement where noted) ----------

# waveform name -> models a bipolar pulse train (needs tx_frequency_hz + on_time).
# Implementing a new one: tdem/forward.py _transitions().
KNOWN_WAVEFORMS: dict[str, bool] = {
    "bipolar_square": True,
    "step_off": False,
    "step": False,
}

# geometry name -> registered builder lives in tdem/forward.py GEOMETRY_BUILDERS.
KNOWN_GEOMETRIES = ("concentric_loop", "offset_dipole")

KNOWN_SFZ_FORMATS = ("bracket", "underscore", "zero_padded")

KNOWN_ORIENTATIONS = ("Z",)          # only Z-component dB/dt is modeled (see #84)

# -- units (#83) --------------------------------------------------------------
# The entire pipeline works in moment-normalized dB/dt, V/(A·m⁴). The sidecar
# declares it; anything else is rejected (no conversion exists yet — a loud
# failure beats a silent wrong-units inversion). Spelling variants normalize.
RESPONSE_QUANTITY = "dBdt"
CANONICAL_UNITS = "V/(A·m⁴)"

_UNIT_ALIASES = {
    "v/(a·m⁴)", "v/(a.m4)", "v/(a.m^4)", "v/(a*m^4)", "v/(a m4)",
    "v/am4", "v/(am4)", "v/(a·m4)", "v/a·m⁴", "v/(a-m4)",
}
_QUANTITY_ALIASES = {"dbdt", "db/dt", "db_dt", "dbz/dt", "dbzdt"}


def _norm(s: str) -> str:
    return str(s).strip().lower().replace(" ", "")


def validate_sidecar(config: dict, *, source: str = "sidecar") -> None:
    """
    Validate a survey sidecar dict. Raises ValueError on anything that would
    send the pipeline down a wrong path; warns on risky-but-legal omissions.

    Called at the top of load.load_survey(). Tolerant of minimal test/library
    configs: only keys that are PRESENT are checked against the registries —
    hard requirements (column_map.sfz_n, gate_times_ms length, off-time) remain
    enforced where they always were, in load._apply_column_map/_validate.
    """
    # -- version gate ---------------------------------------------------------
    ver = config.get("schema_version", 1)      # absent → written before the field
    if not isinstance(ver, int) or ver < 1:
        raise ValueError(f"{source}: schema_version must be a positive int, got {ver!r}")
    if ver > SCHEMA_VERSION:
        raise ValueError(
            f"{source}: schema_version {ver} is newer than this code understands "
            f"(max {SCHEMA_VERSION}). Update the software; do not guess at a "
            "future schema."
        )

    sysc = config.get("system", {})

    # -- enumerations: present ⇒ must be known --------------------------------
    wf = sysc.get("tx_waveform")
    if wf not in (None, "") and wf not in KNOWN_WAVEFORMS:
        raise ValueError(
            f"{source}: unknown tx_waveform {wf!r}. Known: {sorted(KNOWN_WAVEFORMS)}. "
            "An unrecognized waveform must not silently degrade to a step-off — "
            "register it in tdem/schema.py and implement it in forward._transitions()."
        )
    geom = sysc.get("tx_geometry")
    if geom is not None and geom not in KNOWN_GEOMETRIES:
        raise ValueError(
            f"{source}: unknown tx_geometry {geom!r}. Known: {KNOWN_GEOMETRIES}."
        )
    orient = sysc.get("rx_orientation")
    if orient is not None and str(orient).upper() not in KNOWN_ORIENTATIONS:
        raise ValueError(
            f"{source}: rx_orientation {orient!r} unsupported — only "
            f"{KNOWN_ORIENTATIONS} is modeled (a non-Z config would silently "
            "receive the Z response, #84)."
        )
    fmt = config.get("column_map", {}).get("sfz_format")
    if fmt is not None and fmt not in KNOWN_SFZ_FORMATS:
        raise ValueError(
            f"{source}: unknown sfz_format {fmt!r}. Known: {KNOWN_SFZ_FORMATS}."
        )

    # -- units / quantity (#83): present ⇒ must be canonical; absent ⇒ warn ---
    units = sysc.get("units")
    if units is not None:
        if _norm(units) not in _UNIT_ALIASES:
            raise ValueError(
                f"{source}: system.units is {units!r} but the pipeline works only "
                f"in {CANONICAL_UNITS} (moment-normalized dB/dt). No unit "
                "conversion exists — convert the deliverable upstream, or the "
                "inversion output would be silently wrong."
            )
    else:
        warnings.warn(
            f"[schema] {source}: system.units not declared — ASSUMING "
            f"{CANONICAL_UNITS}. Add 'units' to the sidecar so a differently-"
            "scaled deliverable fails loudly instead of inverting to garbage (#83).",
            stacklevel=2,
        )
    quantity = sysc.get("response_quantity")
    if quantity is not None and _norm(quantity) not in _QUANTITY_ALIASES:
        raise ValueError(
            f"{source}: response_quantity {quantity!r} unsupported — the forward "
            f"models {RESPONSE_QUANTITY} only. B-field data needs a different "
            "receiver (see #84/#85 planning)."
        )

    # -- typo catcher: unknown keys in the inversion block silently no-op -----
    _KNOWN_INVERSION_KEYS = {
        "n_layers", "depth_min_m", "depth_max_m", "rho_initial", "rho_min",
        "rho_max", "alpha_s", "alpha_z", "max_iter", "rel_error", "chi_target",
        "use_noise_floor", "censor_factor", "cooling_octaves",
    }
    inv = config.get("inversion", {})
    unknown = [k for k in inv if not k.startswith("_") and k not in _KNOWN_INVERSION_KEYS]
    if unknown:
        warnings.warn(
            f"[schema] {source}: unknown inversion key(s) {unknown} — these are "
            "silently ignored by the inversion; check for typos.",
            stacklevel=2,
        )
