import numpy as np
import pytest

from ghostmesh.observation import observation_features


def test_frame_causality_and_common_packet_phase_invariance():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(8, 6, 114, 10)).astype("float32")

    def features(v):
        return observation_features(
            v, magnitude_units="log_magnitude_db", phase_units="radians"
        )

    result = features(x)
    np.testing.assert_array_equal(result[:4], features(x[:4]))
    modified = x.copy()
    modified[:, 3:] += rng.uniform(-3, 3, size=(8, 1, 114, 10))
    np.testing.assert_allclose(features(modified), result, atol=3e-7)
    assert result.shape == (8, 12, 114) and np.isfinite(result).all()


def test_units_are_explicit_and_corrupt_frames_rejected():
    x = np.zeros((3, 6, 114, 10), dtype="float32")
    with pytest.raises(ValueError, match="Explicit"):
        observation_features(
            x, magnitude_units="linear_amplitude", phase_units="radians"
        )
    x[1, 0] = -np.inf
    with pytest.raises(ValueError, match="5 percent"):
        observation_features(
            x, magnitude_units="log_magnitude_db", phase_units="radians"
        )
