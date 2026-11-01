import numpy as np

from scripts.diagnose_source_alignment import pearson, rms_motion, select_lag


def test_constant_motion_is_undefined_and_lag_obeys_original_frame_clock():
    assert pearson(np.ones(12), np.arange(12)) is None
    x = np.arange(40, dtype=float)[:, None] * np.array([[1, 2]])
    speed = rms_motion(x, 4)
    assert np.isnan(speed[:4]).all()
    np.testing.assert_allclose(speed[4:], np.sqrt(40))


def test_only_train_correlations_choose_offset_and_ties_are_deterministic():
    curve = {
        "-4": dict(train=0.3, validation=-0.9),
        "4": dict(train=0.3, validation=0.99),
        "0": dict(train=0.1, validation=1),
    }
    assert select_lag(curve) == -4
    curve["0"]["validation"] = -100
    assert select_lag(curve) == -4
    curve["0"]["train"] = 0.3
    assert select_lag(curve) == 0
    assert select_lag({"0": dict(train=None, validation=1)}) is None
