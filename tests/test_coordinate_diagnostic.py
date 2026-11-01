"""No dataset/model fits; guards against oracle transformations and weak controls."""

import copy

import numpy as np
import pytest

from scripts.domain_coordinate_diagnostic import (
    align_source_validation,
    balanced_variance,
    canonical_gate,
    coordinates,
    run,
    score,
    source_controls,
)


def test_own_pose_normalization_invariances_and_degenerate_scale_rejection():
    rng = np.random.default_rng(79)
    p = rng.normal(size=(8, 17, 2))
    q, valid = coordinates(p, "unit_body")
    changed = p * rng.uniform(0.1, 5, size=(8, 1, 1)) + rng.normal(size=(8, 1, 2)) * 10
    actual, accepted = coordinates(changed, "unit_body")
    assert valid.all() and accepted.all()
    np.testing.assert_allclose(q, actual, rtol=0, atol=1e-14)
    np.testing.assert_allclose(q.mean(1), 0, atol=1e-15)
    np.testing.assert_allclose(np.square(q).sum(2).mean(1), 1, atol=1e-15)
    _, valid = coordinates(np.ones((2, 17, 2)), "unit_body")
    assert not valid.any()
    q, valid = coordinates(np.ones((2, 17, 2)), "unit_body")
    assert np.isnan(q).all()
    with pytest.raises(ValueError):
        coordinates(p, "unknown")


def test_nonfinite_rows_are_masked_before_centering_with_warnings_as_errors():
    rng = np.random.default_rng(84)
    pose = rng.normal(size=(4, 17, 2))
    expected = pose[:1].copy()
    pose[1, 0, 0] = np.inf
    pose[2, 0, 0] = -np.inf
    pose[3, 0, 0] = np.nan
    for stage in ("native", "centred", "unit_body"):
        actual, valid = coordinates(pose, stage)
        np.testing.assert_array_equal(valid, [True, False, False, False])
        assert np.isnan(actual[1:]).all()
        good, _ = coordinates(expected, stage)
        np.testing.assert_array_equal(actual[:1], good)


def test_proper_rotation_and_mirror_are_global_source_validation_choices():
    rng = np.random.default_rng(80)
    p = rng.normal(size=(12, 17, 2))
    matrix = np.array([[0.0, -1.0], [1.0, 0.0]])
    y = np.einsum("...j,jk->...k", p, matrix)
    fitted = align_source_validation(p[:8], y[:8])
    np.testing.assert_allclose(fitted["proper"]["matrix"], matrix, atol=1e-14)
    assert fitted["proper"]["determinant"] == pytest.approx(1)
    reflected = y * np.array([-1, 1])
    other = align_source_validation(p[:8], reflected[:8])
    assert other["selected"]["kind"] == "reflection_plus_proper"
    assert other["selected"]["source_validation_mse"] < 1e-20
    first = copy.deepcopy(fitted)
    y[8:] += 1e6
    assert align_source_validation(p[:8], y[:8]) == first
    assert np.asarray(fitted["selected"]["matrix"]).shape == (2, 2)


def test_canonical_prior_is_direct_training_mean_and_clock_ignores_query_labels():
    rng = np.random.default_rng(81)
    raw = rng.normal(size=(6, 17, 2))
    target, _ = coordinates(raw, "unit_body")
    train = np.arange(6) < 4
    ids = np.arange(4, 6)
    ends = np.array([39, 39, 47, 47, 39, 55])
    controls, fallback = source_controls(target, ends, train, ids)
    np.testing.assert_array_equal(controls["source_mean"][0], target[:4].mean(0))
    np.testing.assert_array_equal(controls["endpoint_clock"][0], target[:2].mean(0))
    assert fallback == [55]
    transformed_old_mean, _ = coordinates(raw[:4].mean(0, keepdims=True), "unit_body")
    assert not np.allclose(controls["source_mean"][0], transformed_old_mean[0])
    target[~train] += 1e7
    after, _ = source_controls(target, ends, train, ids)
    for name in controls:
        np.testing.assert_array_equal(controls[name], after[name])


def test_balanced_room_variance_is_decomposition_not_camera_causality_claim():
    rng = np.random.default_rng(82)
    base = rng.normal(size=(1, 1, 2, 5, 17, 2))
    p = np.repeat(np.repeat(base, 3, axis=0), 2, axis=1)
    p += np.arange(3)[:, None, None, None, None, None] * np.array([1.0, 2.0])
    native = balanced_variance(p)
    assert native["room_associated_share"] == pytest.approx(1)
    assert native["total_after_action_endpoint_mean_ss"] == pytest.approx(
        native["room_associated_ss"] + native["within_room_person_residual_ss"]
    )
    centred, _ = coordinates(p.reshape(-1, 17, 2), "centred")
    result = balanced_variance(centred.reshape(p.shape))
    assert result["room_associated_ss"] < 1e-25


def test_canonical_continuation_requires_full_coverage_both_strong_controls_and_motion():
    def value(nje, r2=0.1):
        return dict(
            valid=100,
            total=100,
            record_pose_r2=r2,
            metrics=dict(
                normalized_joint_error=nje, pck_curve={"0.05": 0.7, "0.2": 0.9}
            ),
        )

    methods = dict(
        RF=value(0.1),
        RF_shuffle=value(0.3),
        source_mean=value(0.2, 0),
        endpoint_clock=value(0.2, 0),
    )
    assert canonical_gate(methods)
    changed = copy.deepcopy(methods)
    changed["RF"]["valid"] = 99
    assert not canonical_gate(changed)
    changed = copy.deepcopy(methods)
    changed["endpoint_clock"]["metrics"]["normalized_joint_error"] = 0.09
    assert not canonical_gate(changed)
    changed = copy.deepcopy(methods)
    changed["RF"]["record_pose_r2"] = -0.01
    assert not canonical_gate(changed)
    changed = copy.deepcopy(methods)
    changed["RF"]["metrics"]["pck_curve"]["0.05"] = 0.6
    assert not canonical_gate(changed)


def test_invalid_predictions_are_counted_not_silently_hidden_and_output_refused(
    tmp_path,
):
    rng = np.random.default_rng(83)
    y = rng.normal(size=(4, 17, 2))
    p = y.copy()
    p[0] = np.nan
    result = score(p, y, np.repeat("record", 4))
    assert (
        result["total"] == 4
        and result["valid"] == 3
        and result["invalid_predictions"] == 1
    )
    assert result["metrics"]["normalized_joint_error"] == 0
    with pytest.raises(FileExistsError):
        run(None, None, None, tmp_path)
