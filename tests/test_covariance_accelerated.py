import numpy as np
import pytest

from ghostmesh.covariance_accelerated import covariance_features as accelerated
from ghostmesh.covariance_dynamics import covariance_features, receiver_covariances


def test_cached_determinant_shapes_give_bitexact_original_features():
    rng = np.random.default_rng(141)
    rf = rng.normal(size=(80, 6, 114, 10))
    rf[:, :3] += 40
    covariance = receiver_covariances(
        rf, magnitude_units="log_magnitude_db", phase_units="radians"
    )
    expected = covariance_features(covariance)
    actual = accelerated(covariance)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])
    prefix = accelerated(covariance[:60])
    for name in prefix:
        np.testing.assert_array_equal(prefix[name], actual[name][:28])


def test_accelerated_path_does_not_relax_numerical_or_hermitian_guards():
    covariance = np.tile(np.diag([1, 1, 1e-14]), (40, 19, 1, 1))
    with pytest.raises(ValueError, match="ill-conditioned"):
        accelerated(covariance)
    covariance = np.tile(np.eye(3), (40, 19, 1, 1))
    covariance[0, 0, 0, 1] = 1
    with pytest.raises(ValueError, match="Hermitian"):
        accelerated(covariance)
