"""Causal packet summaries with explicit MM-Fi log-magnitude/phase units.

These are receiver-relative statistics, not calibrated RSSI, Doppler or AoA.
Original implementation; no upstream model or implementation is imported.
"""

from __future__ import annotations

import numpy as np


def observation_features(rf, *, magnitude_units, phase_units):
    """Return [T,12,114]: 3 magnitude means/SDs and 3 complex coherences.

    Each feature uses only its own frame's ten supplied packet columns.
    The six phase channels use pairs(0,1),(0,2),(1,2). They cancel common
    packet phase offsets. Stationary room/receiver biases can remain.
    """
    if magnitude_units != "log_magnitude_db" or phase_units != "radians":
        raise ValueError("Explicit log_magnitude_db and radians units required")
    rf = np.asarray(rf, dtype=np.float32)
    if rf.ndim != 4 or rf.shape[1:] != (6, 114, 10) or not len(rf):
        raise ValueError("Expected typed RF[T,6,114,10]")
    amplitude, phase = rf[:, :3], rf[:, 3:]
    valid_a, valid_p = np.isfinite(amplitude), np.isfinite(phase)
    if np.any(1 - (valid_a & valid_p).mean((1, 2, 3)) > 0.05):
        raise ValueError("More than 5 percent missing values in a frame")
    med = np.nanmedian(np.where(valid_a, amplitude, np.nan), axis=(2, 3), keepdims=True)
    if not np.isfinite(med).all():
        raise ValueError("An entire receiver link is missing")
    amplitude = np.where(valid_a, amplitude, med)
    magnitude_mean = (amplitude.mean(-1) - 40.0) / 20.0
    magnitude_sd = amplitude.std(-1) / 10.0
    safe_phase = np.where(valid_p, phase, 0.0)
    real, imaginary = [], []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        good = (
            valid_a[:, left] & valid_a[:, right] & valid_p[:, left] & valid_p[:, right]
        )
        delta = safe_phase[:, left] - safe_phase[:, right]
        count = np.maximum(good.sum(-1), 1)
        real.append(np.where(good, np.cos(delta), 0.0).sum(-1) / count)
        imaginary.append(np.where(good, np.sin(delta), 0.0).sum(-1) / count)
    result = np.concatenate(
        (
            magnitude_mean,
            magnitude_sd,
            np.stack(real, axis=1),
            np.stack(imaginary, axis=1),
        ),
        axis=1,
    )
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite observation statistics")
    return result.astype(np.float32)
