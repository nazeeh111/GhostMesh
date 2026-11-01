"""Geometry/numerical/null safeguards; no dataset, model fit or GPU required."""

import copy

import numpy as np
import pytest

from ghostmesh.covariance_dynamics import hermitian_coordinates
from scripts.domain_trajectory_geometry_probe import (
    CONFIG,
    accepted_states,
    controls,
    current_covariances,
    gate,
    pose_distances,
    reconstruct,
    residual_ranks,
    rf_distances,
    run,
)


def covariance(seed=3, n=5):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 19, 3, 3)) + 1j * rng.normal(size=(n, 19, 3, 3))
    return x @ x.conj().swapaxes(-1, -2) + np.eye(3)


def test_exact_hermitian_coordinate_inverse_and_last_current_state_extraction():
    c = covariance()
    np.testing.assert_allclose(
        reconstruct(hermitian_coordinates(c)), c, rtol=0, atol=2e-15
    )
    x = np.arange(9576 * 2, dtype=float).reshape(2, -1)
    actual = current_covariances(x)
    expected = reconstruct(x.reshape(2, 8, 19, 7, 9)[:, -1, :, 0, :])
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError):
        current_covariances(x[:, :-1])


def test_covariance_distances_fixed_mixing_and_frame_gain_limits():
    c = covariance()
    g = np.array([[1, 0.2j, 0.1], [0.1, 1.2, -0.3j], [0, 0.2, 0.8]])
    mixed = g @ c @ g.conj().T
    gain = np.arange(1, len(c) + 1)[:, None, None, None]
    for arm in CONFIG["arms"]:
        before, _ = rf_distances(c, arm)
        after, _ = rf_distances(mixed, arm)
        np.testing.assert_allclose(before, after, rtol=1e-10, atol=1e-10)
        assert np.all(np.diag(before) == 0) and np.all(before == before.T)
    normalized, _ = rf_distances(c, CONFIG["arms"][0])
    normalized_gain, _ = rf_distances(mixed * gain, CONFIG["arms"][0])
    np.testing.assert_allclose(normalized, normalized_gain, rtol=1e-10, atol=1e-10)
    raw, _ = rf_distances(c, CONFIG["arms"][1])
    raw_gain, _ = rf_distances(c * gain, CONFIG["arms"][1])
    assert np.max(abs(raw - raw_gain)) > 0.1
    c[0, 0] = np.zeros((3, 3))
    accepted, _ = accepted_states(c)
    assert not accepted[0] and accepted[1:].all()
    with pytest.raises(ValueError, match="no identity jitter"):
        rf_distances(c, CONFIG["arms"][0])


def test_geometry_controls_reject_shared_clock_and_preserve_all_shift_nulls():
    ends = np.arange(39, 297, 8)
    clock = abs(ends[:, None] - ends[None, :]).astype(float)
    result = controls(clock, clock * 2, ends)
    assert result["aligned"]["raw_spearman"] == pytest.approx(1)
    assert result["aligned"]["lag_residual_rank_correlation"] is None
    assert result["residual_null_q95"] is None
    assert len(result["circular_shifts"]) == len(ends) - 1
    rng = np.random.default_rng(4)
    pose = rng.normal(size=(len(ends), 17, 2))
    distance = pose_distances(pose)
    perfect = controls(distance, distance, ends)
    assert perfect["aligned"]["lag_residual_rank_correlation"] == pytest.approx(1)
    assert perfect["residual_above_null_q95"] > 0.5
    translated = pose + rng.normal(size=(len(ends), 1, 2)) * 100
    np.testing.assert_allclose(distance, pose_distances(translated), atol=1e-13, rtol=0)


def test_residualization_uses_actual_lag_groups_when_endpoints_are_missing():
    lags = np.array([8, 24, 32, 16, 24, 8])
    values = np.array([1, 6, 5, 3, 2, 4])
    residual = residual_ranks(values, lags)
    for lag in set(lags):
        assert residual[lags == lag].mean() == pytest.approx(0)
    assert residual[lags == 32][0] == 0


def test_unavailable_records_and_fewer_than_two_thirds_positive_cannot_pass_gate():
    arm = CONFIG["arms"][0]
    rows = [
        dict(
            record=CONFIG["fixed_first_record"] if i == 0 else f"r{i}",
            arms={
                arm: dict(
                    aligned=dict(lag_residual_rank_correlation=0.4 if i == 0 else 0.2),
                    residual_above_null_q95=0.2,
                )
            },
        )
        for i in range(90)
    ]
    assert gate(rows, arm)["passed"]
    unavailable = copy.deepcopy(rows)
    unavailable[-1]["arms"][arm] = dict(status="unavailable", reason="SPD failure")
    assert not gate(unavailable, arm)["passed"]
    for row in rows[-31:]:
        row["arms"][arm]["aligned"]["lag_residual_rank_correlation"] = -0.2
    assert gate(rows, arm)["positive_residuals"] == 59
    assert not gate(rows, arm)["passed"]


def test_existing_output_refused_before_any_input_read(tmp_path):
    with pytest.raises(FileExistsError):
        run(tmp_path / "absent", tmp_path / "absent", tmp_path)
