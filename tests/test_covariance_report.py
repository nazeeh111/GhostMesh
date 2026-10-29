import numpy as np

from scripts.report_covariance_source_probe import measure


def body():
    pose = np.stack((np.arange(17) / 32, np.zeros(17)), axis=1)
    return pose - pose[[11, 12]].mean(0)


def test_raw_delta_metric_uses_zero_reference_without_record_mean_repair():
    target = np.broadcast_to(body(), (8, 17, 2)).copy()
    target[:, 3, 1] += np.arange(8) / 64
    delta = np.broadcast_to(np.ones((17, 2)) / 128, target.shape).copy()
    record = np.repeat(["one", "two"], 4)
    perfect = measure(target, delta, target, delta, record)
    zero = measure(target, np.zeros_like(delta), target, delta, record)
    wrong_bias = measure(target, -delta, target, delta, record)
    assert perfect["raw_zero_reference_delta_r2"] == 1
    assert zero["raw_zero_reference_delta_r2"] == 0
    assert wrong_bias["raw_zero_reference_delta_r2"] == -3
    assert perfect["relative_nje"] == 0
    assert perfect["relative_pck20"] == 1


def test_static_truth_has_no_artificial_motion_denominator():
    target = np.broadcast_to(body(), (8, 17, 2)).copy()
    delta = np.zeros_like(target)
    report = measure(target, delta, target, delta, np.repeat("one", 8))
    assert report["raw_zero_reference_delta_r2"] is None
    assert report["record_centered_delta_r2"] is None
    assert report["delta_truth_squared_energy"] == 0
