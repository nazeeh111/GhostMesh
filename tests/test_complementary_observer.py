import copy

import numpy as np
import pytest

from scripts import domain_complementary_observer as study


def test_current_RF_state_reset_causality_and_pure_pose_limit():
    rng = np.random.default_rng(20)
    pose = study.project(rng.normal(size=(8, 17, 2)))
    delta = study.project(rng.normal(size=pose.shape))
    record = np.repeat(["a", "b"], 4)
    ends = np.array([39, 47, 63, 71] * 2)
    out = study.track(pose, delta, record, ends, 0.5, 1.0)
    np.testing.assert_allclose(
        out[[0, 2, 4, 6]], pose[[0, 2, 4, 6]], rtol=0, atol=1e-15
    )
    np.testing.assert_allclose(
        out[1],
        study.project(0.5 * (pose[0] + delta[1]) + 0.5 * pose[1]),
        rtol=0,
        atol=1e-15,
    )
    changed_p, changed_d = pose.copy(), delta.copy()
    changed_p[3:] += 7
    changed_d[3:] -= 9
    np.testing.assert_array_equal(
        study.track(changed_p, changed_d, record, ends, 0.5, 1.0)[:3], out[:3]
    )
    np.testing.assert_allclose(
        study.track(pose, delta, record, ends, 0.0, 2.0), pose, rtol=0, atol=1e-15
    )
    np.testing.assert_array_equal(
        study.track(pose, delta, record, ends, 0.9, 0),
        study.track(pose, delta * 1e6, record, ends, 0.9, 0),
    )
    with pytest.raises(ValueError):
        study.track(pose, delta, record, ends[::-1], 0.5, 1.0)


def test_shared_view_identity_prior_residual_and_weak_signal():
    rng = np.random.default_rng(21)
    x = rng.normal(size=(60, 17, 2)) * 0.01
    matrix, precision, residual = study.view_map(x, x)
    np.testing.assert_allclose(matrix, np.eye(2), rtol=0, atol=1e-15)
    assert residual < 1e-14 and precision > 0
    zero, _, _ = study.view_map(np.zeros_like(x), x)
    np.testing.assert_array_equal(zero, np.eye(2))
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    y = np.einsum("...j,jk->...k", x, rotation)
    matrix, precision, residual = study.view_map(x, y)
    xc = (x - x.mean(0)).reshape(-1, 2)
    yc = (y - y.mean(0)).reshape(-1, 2)
    identity_objective = np.square(xc - yc).sum()
    actual_objective = (
        np.square(np.einsum("nj,jk->nk", xc, matrix) - yc).sum()
        + precision * np.square(matrix - np.eye(2)).sum()
    )
    assert actual_objective < identity_objective and residual < 1e-14
    with pytest.raises(ValueError):
        study.view_map(x, y[:-1])


def reference_meta():
    rows = [
        (f"E01/{person}/{action}", end)
        for person in ("S01", "S02", "S04")
        for action in ("A03", "A07", "A12")
        for end in range(39, 297, 8)
    ]
    rng = np.random.default_rng(22)
    meta = dict(
        record=np.array([r for r, e in rows]), end=np.array([e for r, e in rows])
    )
    meta["pose"] = study.project(rng.normal(size=(len(rows), 17, 2)) * 0.1)
    return meta


def test_calibration_uses_exact_reference_budget_and_no_query_labels_or_predictions():
    meta = reference_meta()
    rng = np.random.default_rng(23)
    pose = study.project(rng.normal(size=meta["pose"].shape))
    delta = study.project(rng.normal(size=pose.shape))
    state = study.calibration(meta, pose, delta, "E01", np.zeros((17, 2)))
    assert state["reference_pairs"].shape == (60, 2)
    assert len(np.unique(state["reference_pairs"])) == 120
    query = np.array(["S04" in r for r in meta["record"]])
    meta["pose"][query] += 1e7
    pose[query] -= 1e7
    delta[query] *= 1e7
    after = study.calibration(meta, pose, delta, "E01", np.zeros((17, 2)))
    for key in state:
        np.testing.assert_array_equal(state[key], after[key])
    assert state["view_residual"] < 1e-10


def test_zero_selection_ignores_all_held_labels_and_never_constructs_calibrator(
    monkeypatch,
):
    rng = np.random.default_rng(24)
    rows = [
        (r, p, end)
        for r, p in [("E01", "S04"), ("E02", "S14"), ("E03", "S24")]
        for end in range(39, 297, 8)
    ]
    meta = dict(
        room=np.array([r for r, p, e in rows]),
        person=np.array([p for r, p, e in rows]),
        record=np.array([f"{r}/{p}/A03" for r, p, e in rows]),
        end=np.array([e for r, p, e in rows]),
        pose=study.project(rng.normal(size=(len(rows), 17, 2)) * 0.1),
    )
    pose = study.project(rng.normal(size=meta["pose"].shape))
    delta = study.project(rng.normal(size=pose.shape)) * 0.01

    def forbidden(*args, **kwargs):
        raise AssertionError("Zero mode accessed reference calibrator")

    monkeypatch.setattr(study, "calibration", forbidden)
    first = study.select(meta, pose, delta, "E02", np.zeros((17, 2)), "zero")
    changed = copy.deepcopy(meta)
    changed["pose"][meta["room"] == "E02"] += 1e9
    second = study.select(changed, pose, delta, "E02", np.zeros((17, 2)), "zero")
    assert first == second
    assert len(study.candidates("calibrated", "full")) == 72
    assert len(study.candidates("zero", "full")) == 9


def test_no_motion_shuffle_and_static_RF_prior_limits():
    rng = np.random.default_rng(25)
    anchor = study.project(rng.normal(size=(17, 2)))
    pose = np.broadcast_to(anchor, (12, 17, 2)).copy()
    delta = study.project(rng.normal(size=pose.shape))
    record = np.repeat(["a", "b"], 6)
    ends = np.tile(np.arange(39, 87, 8), 2)
    meta = dict(record=record, end=ends)
    calibration = study.zero_calibration(anchor)
    pick = np.ones(12, bool)
    out = study.apply(meta, pose, delta, pick, calibration, (0.9, 0.0, 0.0, 0.0, 0.0))
    np.testing.assert_allclose(out, pose, rtol=0, atol=1e-15)
    shuffled = study.shuffled(delta, record, np.arange(12))
    np.testing.assert_array_equal(
        study.apply(meta, pose, delta, pick, calibration, (0.9, 0.0, 0.0, 0.0, 0.0)),
        study.apply(meta, pose, shuffled, pick, calibration, (0.9, 0.0, 0.0, 0.0, 0.0)),
    )
    np.testing.assert_array_equal(pose, np.broadcast_to(anchor, (12, 17, 2)))
