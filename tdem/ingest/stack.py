"""
Half-cycles → soundings.

The bipolar square wave flips the sign of the secondary response every
half-cycle, so each record is multiplied by its logged polarity before
stacking. Consecutive runs of n_stack half-cycles become one sounding:

- gate value  = trimmed mean across the window (rejects sferic hits
  without a separate despiking pass)
- gate std    = std across the window (kept as SFz_std[i] downstream;
  future per-gate noise-floor input)
- timestamp   = centre of the window

Only full windows are kept; the trailing partial window is dropped with
a log line.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import trim_mean

from .readers import em_gate_columns


def stack_soundings(em_df: pd.DataFrame, n_stack: int, trim_frac: float = 0.1) -> pd.DataFrame:
    """
    Stack polarity-aligned half-cycles into soundings.

    Parameters
    ----------
    em_df     : output of readers.read_em + timesync.apply_clock (needs t_utc)
    n_stack   : half-cycles per sounding
    trim_frac : fraction trimmed from EACH tail of the window before the
                mean (0.1 → middle 80% used)

    Returns
    -------
    DataFrame: t_utc, n_used, gate_00.., gate_std_00..  (volts, still uncalibrated)
    """
    if n_stack < 2:
        raise ValueError(f"n_stack must be >= 2, got {n_stack}")
    if not 0 <= trim_frac < 0.5:
        raise ValueError(f"trim_frac must be in [0, 0.5), got {trim_frac}")
    if "t_utc" not in em_df.columns:
        raise ValueError("EM frame has no t_utc — run timesync.apply_clock first")

    gate_cols = em_gate_columns(em_df)
    n_full    = len(em_df) // n_stack
    if n_full == 0:
        raise ValueError(f"Only {len(em_df)} half-cycles; need at least n_stack={n_stack}")
    dropped = len(em_df) - n_full * n_stack
    if dropped:
        print(f"[stack] Dropped trailing partial window ({dropped} half-cycles)")

    # (n_full, n_stack, n_gates), polarity-aligned
    gates = em_df[gate_cols].to_numpy(dtype=float)[: n_full * n_stack]
    pol   = em_df["polarity"].to_numpy(dtype=float)[: n_full * n_stack]
    gates = (gates * pol[:, None]).reshape(n_full, n_stack, len(gate_cols))
    t     = em_df["t_utc"].to_numpy(dtype=float)[: n_full * n_stack].reshape(n_full, n_stack)

    out = pd.DataFrame({"t_utc": t.mean(axis=1), "n_used": n_stack})
    stacked = trim_mean(gates, proportiontocut=trim_frac, axis=1)
    spread  = gates.std(axis=1, ddof=1)
    for i in range(len(gate_cols)):
        out[f"gate_{i:02d}"]     = stacked[:, i]
        out[f"gate_std_{i:02d}"] = spread[:, i]
    return out


def stacked_gate_columns(df: pd.DataFrame) -> list[str]:
    """gate_NN columns in numeric gate order (#41 — lexicographic breaks past 99)."""
    cols = [c for c in df.columns if c.startswith("gate_") and not c.startswith("gate_std_")]
    return sorted(cols, key=lambda c: int(c.removeprefix("gate_")))
