"""Strict finite numerical readback for saved scientific artifacts.

A Python max reduction can hide NaN differences. Validate every compared array
or scalar before reporting a maximum error, including its exact shape.
"""

from __future__ import annotations

import numpy as np


def numeric_difference(observed, expected, *, name, atol, rtol=0.0):
    """Return a finite maximum absolute error, or reject the saved state."""
    actual, expected = np.asarray(observed), np.asarray(expected)
    if (
        actual.shape != expected.shape
        or not actual.size
        or actual.dtype.kind not in "biufc"
        or expected.dtype.kind not in "biufc"
        or not np.isfinite(actual).all()
        or not np.isfinite(expected).all()
        or not np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False)
    ):
        raise ValueError("Numeric readback mismatch: " + name)
    return float(np.max(np.abs(actual - expected)))
