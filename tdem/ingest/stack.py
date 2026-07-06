"""
Half-cycles → soundings.

The bipolar square wave flips the sign of the secondary response every
half-cycle. Rather than demodulate-then-average the whole window (which cancels
a receiver DC offset only when the window holds exactly n/2 of each polarity —
broken by any dropped EM row or rank-based trimming), consecutive opposite-
polarity half-cycles are differenced into DC-free PAIR estimates first (#56):

    pair = (pol_a·v_a + pol_b·v_b) / 2 = S      (the DC term b·(pol_a+pol_b)/2 = 0)

then the pairs are trimmed-meaned. DC cancels exactly per pair regardless of the
trim, and linear baseline drift cancels to first order.

- gate value  = trimmed mean over the DC-free pair estimates
- gate std    = robust standard ERROR of that estimate,
                1.4826·MAD(pairs)/√n_pairs (#33: NOT the raw single-half-cycle
                std, which is ~√n larger and re-inflated by the very sferics the
                trim rejected — used as a noise model it drove chi ≪ 1)
- timestamp   = window centre, advanced half a half-cycle because the logged
                stamps are half-cycle START times (#68.2)

n_stack must be EVEN so every window pairs cleanly (#56). Only full windows are
kept; the trailing partial window is dropped with a log line.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import trim_mean

from .readers import em_gate_columns


def stack_soundings(
    em_df: pd.DataFrame,
    n_stack: int,
    trim_frac: float = 0.1,
    tx_frequency_hz: float | None = None,
) -> pd.DataFrame:
    """
    Stack polarity-aligned half-cycles into soundings via DC-free pair differencing.

    Parameters
    ----------
    em_df     : output of readers.read_em + timesync.apply_clock (needs t_utc)
    n_stack   : half-cycles per sounding — MUST be even (#56)
    trim_frac : fraction trimmed from EACH tail of the pair estimates before the
                mean (0.1 → middle 80% used)
    tx_frequency_hz : base frequency; when given, the window-centre timestamp is
                advanced by 0.5/(2·f) so it marks the middle of the measurement
                rather than the first half-cycle's start stamp (#68.2)

    Returns
    -------
    DataFrame: t_utc, n_used, gate_00.., gate_std_00..  (volts, still uncalibrated)
    """
    if n_stack < 2:
        raise ValueError(f"n_stack must be >= 2, got {n_stack}")
    if n_stack % 2 != 0:
        raise ValueError(
            f"n_stack must be EVEN for bipolar pair differencing, got {n_stack}. "
            "An odd window cannot hold equal +/- half-cycles, so a receiver DC "
            "offset does not cancel (#56)."
        )
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

    keep    = n_full * n_stack
    n_gates = len(gate_cols)
    n_pairs = n_stack // 2

    gates = em_df[gate_cols].to_numpy(dtype=float)[:keep]
    pol   = em_df["polarity"].to_numpy(dtype=float)[:keep]
    t     = em_df["t_utc"].to_numpy(dtype=float)[:keep].reshape(n_full, n_stack)

    # warn if any window is polarity-unbalanced — a symptom of dropped EM rows
    # mis-phasing the fixed-size windows (#56); DC no longer cancels there
    pol_win = pol.reshape(n_full, n_stack)
    unbalanced = np.count_nonzero(np.abs(pol_win.sum(axis=1)) > 1e-9)
    if unbalanced:
        warnings.warn(
            f"[stack] {unbalanced}/{n_full} windows have unequal +/- half-cycles "
            "(likely dropped EM rows mis-phasing the stack); receiver DC offset "
            "will leak into those soundings' late gates.",
            stacklevel=2,
        )

    # demodulate, then average consecutive half-cycles into DC-free pair estimates
    demod = (gates * pol[:, None]).reshape(n_full, n_pairs, 2, n_gates)
    pairs = demod.mean(axis=2)                       # (n_full, n_pairs, n_gates)

    stacked = trim_mean(pairs, proportiontocut=trim_frac, axis=1)   # (n_full, n_gates)

    # robust standard error of the stacked estimate (#33)
    med = np.median(pairs, axis=1, keepdims=True)
    mad = np.median(np.abs(pairs - med), axis=1)      # (n_full, n_gates)
    se  = 1.4826 * mad / np.sqrt(n_pairs)

    t_centre = t.mean(axis=1)
    if tx_frequency_hz:
        t_centre = t_centre + 0.5 / (2.0 * tx_frequency_hz)   # start-stamp → centre

    out = pd.DataFrame({"t_utc": t_centre, "n_used": n_stack})
    for i in range(n_gates):
        out[f"gate_{i:02d}"]     = stacked[:, i]
        out[f"gate_std_{i:02d}"] = se[:, i]
    return out


def stacked_gate_columns(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if c.startswith("gate_") and not c.startswith("gate_std_"))
