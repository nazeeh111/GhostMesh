"""Congruence-invariant receiver-covariance dynamics, as a testable proposal.

If C'=a*G*C*G^H, a>0, determinant normalization removes a and maps every
matrix to the SAME congruence by G/abs(det(G))**(1/3). Thus paired-reference
shrinkage and generalized eigenvalues remain invariant. This assumes one fixed
invertible G per six-subcarrier bin over the paired times. Actual room changes
need not follow that model. This is not calibrated AoA, velocity, or a novelty
claim. Constant covariances give zero dynamics for every static pose, so these
features alone cannot identify an unknown initial pose.
"""

from __future__ import annotations

import numpy as np

LAGS = (1, 2, 4, 8, 16, 32)
PAIRED_SHRINKAGE = 1e-4
CONDITION_LIMIT = 1e10


def receiver_covariances(rf, *, magnitude_units, phase_units):
    """Mean H H^H across6frequencies x10packets for19bins, complete vectors only.

    A missing receiver value drops its entire three-receiver sample, preserving
    Hermitian positive semidefiniteness. More than3missing samples/bin is rejected.
    No future frame, interpolation or fitted data statistic is used.
    """
    if magnitude_units != "log_magnitude_db" or phase_units != "radians":
        raise ValueError("Explicit log_magnitude_db and radians required")
    rf = np.asarray(rf, dtype="float64")
    if rf.ndim != 4 or rf.shape[1:] != (6, 114, 10) or not len(rf):
        raise ValueError("Expected typed RF[T,6,114,10]")
    magnitude, phase = rf[:, :3], rf[:, 3:]
    finite = np.isfinite(magnitude) & np.isfinite(phase)
    complete = finite.all(1).reshape(len(rf), 19, 6, 10)
    counts = complete.sum((2, 3))
    if np.any(counts < 57):
        raise ValueError("More than5percent incomplete three-receiver vectors/bin")
    safe_magnitude = np.where(finite, magnitude, 0)
    safe_phase = np.where(finite, phase, 0)
    with np.errstate(over="raise", invalid="raise"):
        channel = np.exp(safe_magnitude * (np.log(10) / 20) + 1j * safe_phase)
    channel = channel.transpose(0, 2, 3, 1).reshape(len(rf), 19, 6, 10, 3)
    channel = np.where(complete[..., None], channel, 0)
    covariance = np.einsum("tbfpa,tbfpc->tbac", channel, channel.conj(), optimize=False)
    covariance /= counts[..., None, None]
    return (covariance + covariance.conj().swapaxes(-1, -2)) / 2


def determinant_shape(covariance):
    """Remove positive scalar gain; reject numerical singularity without I jitter.

    The exact SPD transform is congruence equivariant up to a shared scalar.
    Its numerical condition-number acceptance guard is NOT invariant under
    arbitrarily ill-conditioned changes of basis; such cases are rejected.
    """
    covariance = np.asarray(covariance, dtype="complex128")
    if covariance.shape[-2:] != (3, 3) or not np.isfinite(covariance).all():
        raise ValueError("Expected finite3x3Hermitian covariance matrices")
    asymmetry = np.max(abs(covariance - covariance.conj().swapaxes(-1, -2)))
    if asymmetry > 1e-10 * max(float(np.max(abs(covariance))), 1e-12):
        raise ValueError("Covariances must be Hermitian")
    eigenvalues = np.linalg.eigvalsh(covariance)
    if np.any(eigenvalues[..., 0] <= 0) or np.any(
        eigenvalues[..., -1] / eigenvalues[..., 0] > CONDITION_LIMIT
    ):
        raise ValueError(
            "Singular or numerically ill-conditioned covariance; no identity jitter applied"
        )
    scale = np.exp(np.log(eigenvalues).mean(-1))
    return covariance / scale[..., None, None]


def relative_log_spectrum(current, past):
    """Three centered log generalized eigenvalues, with equivariant regularization.

    First determinant-normalize each SPD matrix. R=(A+B)/2 transforms by the same
    congruence, so A+epsR and B+epsR do too. Skipping the determinant step would
    break exact invariance to independent frame gains after nonzero shrinkage.
    """
    a, b = determinant_shape(current), determinant_shape(past)
    reference = (a + b) / 2
    a, b = a + PAIRED_SHRINKAGE * reference, b + PAIRED_SHRINKAGE * reference
    inverse = np.linalg.inv(np.linalg.cholesky(b))
    whitened = np.einsum(
        "...ij,...jk,...lk->...il", inverse, a, inverse.conj(), optimize=False
    )
    whitened = (whitened + whitened.conj().swapaxes(-1, -2)) / 2
    eigenvalues = np.linalg.eigvalsh(whitened)
    if not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= 0):
        raise ValueError("Invalid regularized generalized spectrum")
    values = np.log(eigenvalues)
    return values - values.mean(-1, keepdims=True)


def hermitian_coordinates(covariance):
    """Nine real coordinates; sqrt2 off-diagonals preserve Frobenius distances."""
    diagonal = np.diagonal(covariance, axis1=-2, axis2=-1).real
    off = covariance[..., (0, 0, 1), (1, 2, 2)]
    return np.concatenate((diagonal, np.sqrt(2) * off.real, np.sqrt(2) * off.imag), -1)


def covariance_features(covariance):
    """Aligned per-time controls/dynamics with32frame causal startup excluded."""
    covariance = np.asarray(covariance, dtype="complex128")
    if (
        covariance.ndim != 4
        or covariance.shape[1:] != (19, 3, 3)
        or len(covariance) <= max(LAGS)
    ):
        raise ValueError("Expected at least33frames of19three-receiver covariances")
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
    paired = [
        relative_log_spectrum(
            covariance[first:], covariance[first - lag : len(covariance) - lag]
        )
        for lag in LAGS
    ]
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
