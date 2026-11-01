"""Cache determinant shapes across lag pairs without changing covariance math.

This optimized path accepts exactly Hermitian covariances from the existing
receiver builder. Tests compare all outputs bit for bit with the original path.
It does not relax missing-vector, SPD, conditioning or history guards.
"""

from __future__ import annotations

import numpy as np

from ghostmesh.covariance_dynamics import (
    LAGS,
    PAIRED_SHRINKAGE,
    determinant_shape,
    hermitian_coordinates,
)


def covariance_features(covariance):
    covariance = np.asarray(covariance, dtype="complex128")
    if (
        covariance.ndim != 4
        or covariance.shape[1:] != (19, 3, 3)
        or len(covariance) <= max(LAGS)
        or not np.array_equal(covariance, covariance.conj().swapaxes(-1, -2))
    ):
        raise ValueError(
            "Expected at least33frames of exact Hermitian19x3x3 covariances"
        )
    shape = determinant_shape(covariance)
    eigenvalues, vectors = np.linalg.eigh(shape)
    tangent = np.einsum(
        "...ij,...j,...kj->...ik",
        vectors,
        np.log(eigenvalues),
        vectors.conj(),
        optimize=False,
    )
    first = max(LAGS)
    paired = []
    for lag in LAGS:
        a, b = shape[first:], shape[first - lag : len(covariance) - lag]
        reference = (a + b) / 2
        a = a + PAIRED_SHRINKAGE * reference
        b = b + PAIRED_SHRINKAGE * reference
        inverse = np.linalg.inv(np.linalg.cholesky(b))
        whitened = np.einsum(
            "...ij,...jk,...lk->...il", inverse, a, inverse.conj(), optimize=False
        )
        whitened = (whitened + whitened.conj().swapaxes(-1, -2)) / 2
        eigenvalues = np.linalg.eigvalsh(whitened)
        if not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= 0):
            raise ValueError("Invalid regularized generalized spectrum")
        values = np.log(eigenvalues)
        paired.append(values - values.mean(-1, keepdims=True))
    covariance_states = [covariance[first:]] + [
        covariance[first - lag : len(covariance) - lag] for lag in LAGS
    ]
    tangent_states = [tangent[first:]] + [
        tangent[first - lag : len(covariance) - lag] for lag in LAGS
    ]
    return {
        "untreated_covariance": hermitian_coordinates(
            np.stack(covariance_states, axis=2)
        ).reshape(len(covariance) - first, -1),
        "log_shape_tangent": hermitian_coordinates(
            np.stack(tangent_states, axis=2)
        ).reshape(len(covariance) - first, -1),
        "congruence_dynamics": np.stack(paired, axis=2).reshape(
            len(covariance) - first, -1
        ),
    }
