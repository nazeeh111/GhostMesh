from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ghostmesh.covariance_dynamics import (
    covariance_features,
    determinant_shape,
    hermitian_coordinates,
    receiver_covariances,
    relative_log_spectrum,
)  # noqa:E402


def complex_normal(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def typed_rf(channel):
    return np.concatenate((20 * np.log10(abs(channel)), np.angle(channel)), axis=1)


def test_nondiagonal_complex_congruence_and_independent_gain_invariance():
    rng = np.random.default_rng(8)
    x, y = complex_normal(rng, (9, 3, 3)), complex_normal(rng, (9, 3, 3))
    a = np.einsum("...ij,...kj->...ik", x, x.conj()) + np.eye(3)
    b = np.einsum("...ij,...kj->...ik", y, y.conj()) + np.eye(3)
    g = complex_normal(rng, (9, 3, 3)) + 2 * np.eye(3)
    transformed_a = np.einsum("...ij,...jk,...lk->...il", g, a, g.conj()) * rng.uniform(
        0.2, 6, (9, 1, 1)
    )
    transformed_b = np.einsum("...ij,...jk,...lk->...il", g, b, g.conj()) * rng.uniform(
        0.2, 6, (9, 1, 1)
    )
    np.testing.assert_allclose(
        relative_log_spectrum(a, b),
        relative_log_spectrum(transformed_a, transformed_b),
        atol=2e-10,
        rtol=2e-10,
    )


def test_common_packet_phase_cancels_and_future_frames_do_not_enter_features():
    rng = np.random.default_rng(9)
    channel = complex_normal(rng, (70, 3, 114, 10))
    shifted = channel * np.exp(1j * rng.normal(size=(70, 1, 114, 10)))
    first = receiver_covariances(
        typed_rf(channel), magnitude_units="log_magnitude_db", phase_units="radians"
    )
    second = receiver_covariances(
        typed_rf(shifted), magnitude_units="log_magnitude_db", phase_units="radians"
    )
    np.testing.assert_allclose(first, second, atol=2e-14, rtol=2e-14)
    full = covariance_features(first)
    prefix = covariance_features(first[:55])
    for name in full:
        np.testing.assert_array_equal(full[name][:23], prefix[name])


def test_singular_or_extremely_ill_conditioned_covariances_are_rejected():
    with pytest.raises(ValueError, match="Singular"):
        determinant_shape(np.ones((3, 3)))
    with pytest.raises(ValueError, match="ill-conditioned"):
        determinant_shape(np.diag([1, 1, 1e-14]))


def test_missing_receiver_drops_complete_vector_and_excess_missing_rejects():
    rng = np.random.default_rng(10)
    channel = complex_normal(rng, (1, 3, 114, 10))
    rf = typed_rf(channel)
    rf[0, 0, 0, 0] = -np.inf
    result = receiver_covariances(
        rf, magnitude_units="log_magnitude_db", phase_units="radians"
    )
    vectors = channel.transpose(0, 2, 3, 1).reshape(1, 19, 60, 3)[0, 0, 1:]
    expected = np.einsum("pi,pj->ij", vectors, vectors.conj()) / 59
    np.testing.assert_allclose(result[0, 0], expected, atol=2e-14)
    rf[0, 0, 0, :4] = -np.inf
    with pytest.raises(ValueError, match="incomplete"):
        receiver_covariances(
            rf, magnitude_units="log_magnitude_db", phase_units="radians"
        )


def test_static_covariances_have_zero_motion_even_for_different_static_poses():
    first = np.tile(np.diag([1, 2, 5]), (45, 19, 1, 1))
    second = np.tile(np.diag([3, 7, 2]), (45, 19, 1, 1))
    np.testing.assert_allclose(
        covariance_features(first)["congruence_dynamics"], 0, atol=1e-14
    )
    np.testing.assert_allclose(
        covariance_features(second)["congruence_dynamics"], 0, atol=1e-14
    )


def test_covariance_control_preserves_the_same_current_and_past_states():
    covariance = np.tile(np.eye(3), (45, 19, 1, 1))
    covariance[:, :, 0, 0] = np.arange(45)[:, None] + 1
    result = covariance_features(covariance)
    expected = hermitian_coordinates(
        np.stack([covariance[index] for index in (32, 31, 30, 28, 24, 16, 0)], axis=1)
    ).reshape(-1)
    np.testing.assert_array_equal(result["untreated_covariance"][0], expected)
    assert result["untreated_covariance"].shape == (13, 19 * 7 * 9)
    assert result["congruence_dynamics"].shape == (13, 19 * 6 * 3)
