"""Explicit log-magnitude/phase preprocessing for MM-Fi v2 research.

Log magnitude is relative to the original receiver IQ scale, not calibrated
RSSI in dBm. Never silently apply this transform to linear-amplitude input.
"""

from __future__ import annotations

import numpy as np


def _shape(x):
    if x.ndim != 4 or x.shape[1:] != (3, 114, 10) or len(x) < 1:
        raise ValueError("Expected [T,3,114,10] CSI")


def prepare_log_magnitude(csi, *, units):
    if units != "log_magnitude_db":
        raise ValueError("Explicit log_magnitude_db units required")
    x = np.asarray(csi, dtype="float32").copy()
    _shape(x)
    valid = np.isfinite(x)
    if 1 - float(valid.mean()) > 0.05:
        raise ValueError("More than 5 percent missing log magnitudes")
    x[~valid] = np.nan
    median = np.nanmedian(x, axis=(0, 2, 3), keepdims=True)
    if not np.isfinite(median).all():
        raise ValueError("Entire receiver link is missing")
    x = np.where(valid, x, median)
    scale = np.median(np.abs(x - median), axis=(0, 2, 3), keepdims=True) * 1.4826
    return np.clip((x - median) / np.maximum(scale, 1.0), -8, 8).astype("float32")


def quotient_features(log_magnitude, phase, *, units):
    """Two log10 magnitude ratios and four circular phase-difference channels.

    All six channels cancel arbitrary common packet gain and phase offsets.
    Static receiver-chain/multipath biases can remain. No physical AoA/ToF is
    estimated. Missing/reference-zero pairs are neutral, never divided by zero.
    """
    if units != "log_magnitude_db":
        raise ValueError("Explicit log_magnitude_db units required")
    a, p = (
        np.asarray(log_magnitude, dtype="float32"),
        np.asarray(phase, dtype="float32"),
    )
    _shape(a)
    if a.shape != p.shape:
        raise ValueError("Phase shape must match log magnitude")
    finite = np.isfinite(a) & np.isfinite(p)
    valid = finite[:, 1:] & finite[:, :1]
    if 1 - float(valid.mean()) > 0.05:
        raise ValueError("More than 5 percent invalid quotient pairs")
    # Replace missing values before subtraction, including +/-infinity.
    a = np.where(finite, a, 0.0)
    p = np.where(finite, p, 0.0)
    ratio = np.clip((a[:, 1:] - a[:, :1]) / 20.0, -2.0, 2.0)
    delta = p[:, 1:] - p[:, :1]
    return np.concatenate(
        (
            np.where(valid, ratio, 0),
            np.where(valid, np.cos(delta), 0),
            np.where(valid, np.sin(delta), 0),
        ),
        axis=1,
    ).astype("float32")
