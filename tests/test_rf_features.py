import numpy as np
import pytest
from ghostmesh.rf_features import prepare_log_magnitude, quotient_features


def test_db_is_not_linear_amplitude_and_negative_db_is_valid():
    x = np.linspace(-50, 10, 8 * 3 * 114 * 10).reshape(8, 3, 114, 10)
    out = prepare_log_magnitude(x, units="log_magnitude_db")
    assert np.isfinite(out).all() and out.std() > 0.1
    with pytest.raises(ValueError, match="Explicit"):
        prepare_log_magnitude(x, units="linear_amplitude")


def test_complex_quotient_cancels_packet_gain_and_phase():
    r = np.random.default_rng(2027)
    db = r.uniform(-40, 50, (8, 3, 114, 10))
    p = r.uniform(-np.pi, np.pi, db.shape)
    gain = r.uniform(-10, 10, (8, 1, 114, 10))
    angle = r.uniform(-2, 2, (8, 1, 114, 10))
    np.testing.assert_allclose(
        quotient_features(db, p, units="log_magnitude_db"),
        quotient_features(db + gain, p + angle, units="log_magnitude_db"),
        atol=2e-6,
    )


def test_db_offset_invariance_missing_and_finite_neutral_quotient():
    r = np.random.default_rng(2028)
    x = r.uniform(-40, 50, (8, 3, 114, 10))
    p = r.uniform(-2, 2, x.shape)
    bias = r.uniform(-20, 20, (1, 3, 1, 1))
    np.testing.assert_allclose(
        prepare_log_magnitude(x, units="log_magnitude_db"),
        prepare_log_magnitude(x + bias, units="log_magnitude_db"),
        atol=2e-6,
    )
    x[0, 0, 0, 0] = -np.inf
    p[0, 0, 0, 0] = np.inf
    with np.errstate(all="raise"):
        y = quotient_features(x, p, units="log_magnitude_db")
    assert np.isfinite(y).all() and np.count_nonzero(y[0, :, 0, 0]) == 0
