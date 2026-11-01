"""Circular inter-antenna phase features, without a physical AoA claim."""

from __future__ import annotations

import numpy as np


def relative_phase(amplitude, phase, center=False):
    """Return four sin/cos channels relative to receive antenna zero.

    A shared packet/subcarrier phase offset cancels algebraically. This does
    not remove every radio-chain error or recover calibrated propagation phase.
    Missing/zero-power pairs map to the neutral zero vector. Optional circular
    centering uses only the supplied causal window, never future samples.
    """
    a, p = np.asarray(amplitude), np.asarray(phase)
    if a.shape != p.shape or p.ndim != 4 or p.shape[1] != 3:
        raise ValueError("Expected matching [T,3,S,P] amplitude and phase")
    finite = np.isfinite(p) & np.isfinite(a) & (a > 0)
    valid = finite[:, 1:] & finite[:, :1]
    if 1 - float(valid.mean()) > 0.05:
        raise ValueError("More than 5 percent invalid inter-antenna phase pairs")
    delta = np.where(valid, np.nan_to_num(p[:, 1:]) - np.nan_to_num(p[:, :1]), 0.0)
    real, imag = np.cos(delta), np.sin(delta)
    if center:
        count = np.maximum(valid.sum(axis=(0, 3), keepdims=True), 1)
        cr = np.where(valid, real, 0).sum(axis=(0, 3), keepdims=True) / count
        ci = np.where(valid, imag, 0).sum(axis=(0, 3), keepdims=True) / count
        angle = np.arctan2(ci, cr)
        real, imag = np.cos(delta - angle), np.sin(delta - angle)
    return np.concatenate(
        (np.where(valid, real, 0), np.where(valid, imag, 0)), axis=1
    ).astype(np.float32)
