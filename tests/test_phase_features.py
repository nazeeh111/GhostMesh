import numpy as np
import pytest
from ghostmesh.phase_features import relative_phase


def test_shared_packet_phase_and_wrap_do_not_change_features():
    r = np.random.default_rng(41)
    amp = r.uniform(0.1, 2, (8, 3, 114, 10))
    phase = r.uniform(-np.pi, np.pi, amp.shape)
    offset = r.uniform(-5, 5, (8, 1, 114, 10))
    expected = relative_phase(amp, phase)
    shifted = (phase + offset + np.pi) % (2 * np.pi) - np.pi
    np.testing.assert_allclose(relative_phase(amp, shifted), expected, atol=2e-6)


def test_centering_removes_constant_receiver_phase_bias():
    r = np.random.default_rng(42)
    amp = np.ones((8, 3, 114, 10))
    phase = r.uniform(-np.pi, np.pi, amp.shape)
    bias = r.uniform(-2, 2, (1, 3, 114, 1))
    np.testing.assert_allclose(
        relative_phase(amp, phase, True),
        relative_phase(amp, phase + bias, True),
        atol=2e-6,
    )


def test_missing_pairs_are_neutral_and_excessive_missing_is_rejected():
    amp = np.ones((8, 3, 114, 10))
    phase = np.zeros_like(amp)
    phase[0, 1, 0, 0] = np.nan
    result = relative_phase(amp, phase)
    assert result[0, 0, 0, 0] == 0 and result[0, 2, 0, 0] == 0
    phase[:, 1] = np.nan
    with pytest.raises(ValueError, match="5 percent"):
        relative_phase(amp, phase)
