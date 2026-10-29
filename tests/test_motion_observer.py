import numpy as np
import pytest

from scripts.domain_motion_observer import (
    CONFIG,
    apply,
    calibrator,
    reference_ids,
    track,
    track_scores,
    zero_calibrator,
)


def test_streaming_state_has_no_future_dependency_and_resets_on_gaps():
    rng = np.random.default_rng(1)
    delta = rng.normal(size=(8, 17, 2)) * 0.01
    delta -= delta[:, [11, 12]].mean(1, keepdims=True)
    anchor = np.zeros((17, 2))
    record = np.repeat(["a", "b"], 4)
    ends = np.array([39, 47, 63, 71] * 2)
    out = track(delta, record, ends, anchor, 0.9, 1, np.zeros_like(anchor))
    assert not out[[0, 2, 4, 6]].any()
    changed = delta.copy()
    changed[3:] += 2
    np.testing.assert_array_equal(
        track(changed, record, ends, anchor, 0.9, 1, anchor)[:3], out[:3]
    )
    np.testing.assert_allclose(out[:, [11, 12]].mean(1), 0, atol=1e-16)
    np.testing.assert_array_equal(
        track(delta, record, ends, anchor, 0.9, 0, anchor), np.zeros_like(out)
    )
    with pytest.raises(ValueError):
        track(delta, record, ends[::-1], anchor, 0.9, 1, anchor)


def test_fixed120_reference_poses_are_not_query_people_and_bias_uses_only_references():
    rows = [
        (f"E01/{person}/{action}", end)
        for person in ("S01", "S02", "S04")
        for action in ("A03", "A07", "A12")
        for end in range(39, 297, 8)
    ]
    meta = dict(
        record=np.array([r for r, e in rows]), end=np.array([e for r, e in rows])
    )
    rng = np.random.default_rng(2)
    meta["pose"] = rng.normal(size=(len(rows), 17, 2))
    delta = rng.normal(size=meta["pose"].shape)
    pairs = reference_ids(meta, "E01")
    assert pairs.shape == (60, 2) and len(np.unique(pairs)) == 120
    assert np.all(meta["end"][pairs[:, 1]] - meta["end"][pairs[:, 0]] == 8)
    assert all("S04" not in name for name in meta["record"][pairs.ravel()])
    source = np.zeros((17, 2))
    first = calibrator(meta, delta, "E01", source)
    query = np.array(["S04" in name for name in meta["record"]])
    meta["pose"][query] += 1000
    delta[query] -= 1000
    second = calibrator(meta, delta, "E01", source)
    np.testing.assert_array_equal(first["reference"], second["reference"])
    np.testing.assert_array_equal(first["bias"], second["bias"])
    assert len(CONFIG["reference_pairs"]) == 10


def test_zero_mode_calibrator_cannot_access_target_reference_labels():
    anchor = np.arange(34, dtype=float).reshape(17, 2) / 100
    anchor -= anchor[[11, 12]].mean(0, keepdims=True)
    c = zero_calibrator(anchor)
    meta = dict(record=np.array(["x", "x"]), end=np.array([39, 47]))
    delta = np.ones((2, 17, 2)) * 0.01
    # No annotation array is present in this prediction path.
    prediction = apply(meta, delta, np.array([True, True]), c, (0.5, 1, 0, 0))
    assert prediction.shape == (2, 17, 2)
    assert len(c["pairs"]) == 0


def test_metrics_exclude_start_and_gap_deltas_in_every_action_and_handle_static_truth():
    rng = np.random.default_rng(6)
    y = rng.normal(size=(8, 17, 2)) * 0.05
    groups = np.array(["a"] * 4 + ["b"] * 4)
    actions = np.array(["A03"] * 4 + ["A12"] * 4)
    ends = np.array([39, 47, 63, 71] * 2)
    result = track_scores(y, y, np.ones_like(y) * 0.01, groups, actions, ends)
    assert result["derived_delta_valid_windows"] == 4
    assert all(
        r["derived_delta_valid_windows"] == 2 for r in result["per_action"].values()
    )
    assert result["relative_nje"] == 0
    static = np.broadcast_to(y[0], y.shape).copy()
    stationary = track_scores(static, static, np.zeros_like(y), groups, actions, ends)
    assert stationary["delta_zero_r2"] is None
    assert all(r["delta_zero_r2"] is None for r in stationary["per_action"].values())
