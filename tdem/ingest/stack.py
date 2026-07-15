"""
Half-cycles → soundings.

The bipolar square wave flips the sign of the secondary response every
half-cycle. Rather than demodulate-then-average the whole window (which cancels
a receiver DC offset only when the window holds exactly n/2 of each polarity —
broken by any dropped EM row or rank-based trimming), each + half-cycle is paired
with a − half-cycle (keyed on the logged POLARITY column, #2) into a DC-free PAIR
estimate first (#56):

    pair = (pol_a·v_a + pol_b·v_b) / 2 = S      (the DC term b·(pol_a+pol_b)/2 = 0,
                                                 because the pair is + with −)

then the pairs are trimmed-meaned. Pairing on polarity — not on array position —
means the cancellation survives a mis-phased-but-balanced window; genuinely
unequal +/- counts (DC cannot cancel) are counted and warned.

- gate value  = trimmed mean over the DC-free pair estimates
- gate std    = robust standard ERROR of that estimate,
                1.4826·MAD(pairs)/√n_kept, floored (#33/#3/#9: NOT the raw single-
                half-cycle std, which is ~√n larger and re-inflated by the very
                sferics the trim rejected — used as a noise model it drove chi ≪ 1.
                n_kept is the trimmed count the point estimate actually averages.)
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

    gates = em_df[gate_cols].to_numpy(dtype=float)[:keep]
    pol   = em_df["polarity"].to_numpy(dtype=float)[:keep]
    t     = em_df["t_utc"].to_numpy(dtype=float)[:keep].reshape(n_full, n_stack)

    demod   = (gates * pol[:, None]).reshape(n_full, n_stack, n_gates)
    pol_win = pol.reshape(n_full, n_stack)

    stacked = np.empty((n_full, n_gates))
    se      = np.empty((n_full, n_gates))
    outlier_frac = np.empty(n_full)          # #71.7: pre-stack sferic diagnostic
    n_unbalanced = 0

    for w in range(n_full):
        pw  = pol_win[w]
        pos = np.where(pw > 0)[0]
        neg = np.where(pw < 0)[0]
        if len(pos) == len(neg) and len(pos) > 0:
            # Pair the k-th + half-cycle with the k-th − (keyed on the POLARITY
            # column, not on array position, #2): the pair (pol_a·v_a + pol_b·v_b)/2
            # is DC-free ONLY when pol_a+pol_b=0. Consecutive-position pairing
            # assumed strict alternation, so a balanced-but-mis-phased window
            # (e.g. +,+,-,-, from an even number of dropped rows) previously
            # produced same-polarity pairs that leaked DC while the window-sum
            # balance check stayed silent. Explicit +/- pairing cancels DC for
            # any interleaving.
            pairs_w = 0.5 * (demod[w][pos] + demod[w][neg])   # (n_pairs, n_gates)
        else:
            # genuinely unequal +/- counts: DC cannot cancel — best-effort demod
            # mean, and count it for a warning
            n_unbalanced += 1
            pairs_w = demod[w]

        stacked[w] = trim_mean(pairs_w, proportiontocut=trim_frac, axis=0)
        med = np.median(pairs_w, axis=0)
        mad = np.median(np.abs(pairs_w - med), axis=0)
        # SE of the TRIMMED mean uses the kept count, not all pairs (#9): the
        # point estimate averages n_kept ≈ (1-2·trim_frac)·n_pairs values, so
        # dividing by √n_pairs understated the error of the reported value
        n_kept = max(len(pairs_w) - 2 * int(trim_frac * len(pairs_w)), 1)
        se[w] = 1.4826 * mad / np.sqrt(n_kept)

        # #71.7: fraction of pair samples deviating > 4σ from the window median —
        # a pre-stack sferic/powerline diagnostic. The trim survives ~trim_frac
        # of hits per tail; in high-sferic season more leak through, and this
        # metric (exported as `outlier_frac`, consumed by nothing downstream —
        # it's for humans and provenance) says how hard the trim was working.
        sig = np.maximum(1.4826 * mad, 1e-300)
        outlier_frac[w] = float(np.mean(np.abs(pairs_w - med) > 4.0 * sig))

    # floor the robust SE so a quiet/quantized gate (MAD == 0) does not report a
    # zero standard error — a downstream consumer using it directly as an
    # inverse-variance weight would otherwise divide by zero (#3). Floored at a
    # small fraction of the signal magnitude.
    se = np.maximum(se, 1e-3 * np.abs(stacked))

    if n_unbalanced:
        warnings.warn(
            f"[stack] {n_unbalanced}/{n_full} windows have unequal +/- half-cycles "
            "(dropped EM rows); receiver DC offset will leak into those soundings.",
            stacklevel=2,
        )

    t_centre = t.mean(axis=1)
    if tx_frequency_hz:
        t_centre = t_centre + 0.5 / (2.0 * tx_frequency_hz)   # start-stamp → centre

    high = outlier_frac > 0.10
    if high.any():
        warnings.warn(
            f"[stack] {int(high.sum())}/{n_full} windows have >10% of pre-stack "
            "samples beyond 4σ (sferics/powerline); the trimmed mean may be "
            "leaking interference into those soundings.",
            stacklevel=2,
        )

    out = pd.DataFrame({"t_utc": t_centre, "n_used": n_stack,
                        "outlier_frac": outlier_frac})
    for i in range(n_gates):
        out[f"gate_{i:02d}"]     = stacked[:, i]
        out[f"gate_std_{i:02d}"] = se[:, i]
    return out


def stacked_gate_columns(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if c.startswith("gate_") and not c.startswith("gate_std_"))
