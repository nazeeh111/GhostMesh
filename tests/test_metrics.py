import numpy as np
import pytest

from ghostmesh.metrics import pose_metrics


def test_known_distance_and_bbox_scale():
    target = np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    prediction = target + np.array([0.1, 0.0])
    m = pose_metrics(prediction, target, np.ones((1, 3), bool), threshold=0.08)
    assert m["pck"] == 1
    assert m["normalized_joint_error"] == pytest.approx(0.1 / np.sqrt(2))
    assert (
        pose_metrics(prediction, target, np.ones((1, 3), bool), threshold=0.05)["pck"]
        == 0
    )


def test_invalid_joints_and_degenerate_frames_are_not_successes():
    truth = np.array(
        [
            [[0.0, 0.0], [1.0, 1.0], [np.nan, np.nan]],
            [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
        ]
    )
    prediction = np.nan_to_num(truth)
    prediction[0, 2] = 500
    m = pose_metrics(prediction, truth, np.ones((2, 3), bool))
    assert m["pck"] == 1
    assert m["valid_joints"] == 2
    assert m["excluded_frames"] == 1
    assert m["per_frame_nje"] == [0.0, None]


def test_no_usable_labels_return_none_and_counts():
    m = pose_metrics(
        np.zeros((2, 3, 2)),
        np.zeros((2, 3, 2)),
        np.zeros((2, 3), bool),
        np.zeros((2, 3)),
    )
    assert m["pck"] is None
    assert m["normalized_joint_error"] is None
    assert m["calibration"]["ece"] is None
    assert m["valid_joints"] == 0


def test_confidence_calibration_is_not_pck():
    truth = np.array([[[0.0, 0.0], [1.0, 1.0]]])
    prediction = truth + 0.03
    m = pose_metrics(
        prediction, truth, np.ones((1, 2), bool), np.full((1, 2), np.log(0.05))
    )
    expected = 1 - np.exp(-0.5)
    assert m["calibration"]["ece"] == pytest.approx(1 - expected)
    assert sum(b["count"] for b in m["calibration"]["bins"]) == 2


def test_nonfinite_prediction_cannot_silently_pass():
    truth = np.array([[[0.0, 0.0], [1.0, 1.0]]])
    prediction = truth.copy()
    prediction[0, 1, 0] = np.nan
    with pytest.raises(ValueError, match="Non-finite prediction"):
        pose_metrics(prediction, truth, np.ones((1, 2), bool))
