import inspect

import numpy as np
import pytest

from scripts.domain_masked_motion_diagnostic import (
    ALPHAS,
    checked_product,
    fit_source_rows,
    fit_training,
    lag_pairs,
    motion_scores,
    predict,
    run,
    source_roles,
    stored_npz_array,
    variance_energy,
)


def test_stored_npz_pose_view_is_lazy_and_compressed_or_object_members_refused(
    tmp_path, monkeypatch
):
    pose = np.arange(6 * 17 * 2, dtype=np.float64).reshape(6, 17, 2)
    pose[3:5] = np.nan  # Stand-in held rows must not be inspected or normalized.
    np.savez(tmp_path / "stored.npz", pose=pose)
    np.savez_compressed(tmp_path / "compressed.npz", pose=pose)
    np.savez(tmp_path / "object.npz", pose=np.array([object()], dtype=object))

    def forbidden(*args, **kwargs):
        raise AssertionError("The complete NPZ pose member must not be loaded")

    monkeypatch.setattr(np, "load", forbidden)
    mapped = stored_npz_array(tmp_path / "stored.npz", "pose")
    assert isinstance(mapped, np.memmap) and not mapped.flags.writeable
    np.testing.assert_array_equal(mapped[[0, 1, 2, 5]], pose[[0, 1, 2, 5]])
    with pytest.raises(ValueError, match="ZIP_STORED"):
        stored_npz_array(tmp_path / "compressed.npz", "pose")
    with pytest.raises(ValueError, match="Object"):
        stored_npz_array(tmp_path / "object.npz", "pose")


def test_fixed_effects_and_ordinary_match_independent_direct_normal_equations():
    rng = np.random.default_rng(8)
    record = np.repeat(["a", "b", "c"], 8)
    x = rng.normal(size=(24, 4)) + np.repeat(np.array([0, 10, -6]), 8)[:, None]
    y = rng.normal(size=(24, 3)) + np.repeat(np.array([4, -7, 9]), 8)[:, None]
    models = fit_training(x, y, record)
    wx = np.concatenate(
        [x[record == key] - x[record == key].mean(0) for key in ["a", "b", "c"]]
    )
    wy = np.concatenate(
        [y[record == key] - y[record == key].mean(0) for key in ["a", "b", "c"]]
    )
    scale = np.maximum(np.sqrt((wx * wx).mean(0)), 0.001)
    for name, state in models.items():
        np.testing.assert_allclose(
            state["feature_scale"], scale, rtol=1e-14, atol=1e-15
        )
        design, target = (
            ((x - x.mean(0)) / scale, y - y.mean(0))
            if name == "ordinary"
            else (wx / scale, wy)
        )
        for i, alpha in enumerate(ALPHAS):
            direct = np.linalg.solve(
                design.T @ design + len(x) * alpha * np.eye(4), design.T @ target
            )
            np.testing.assert_allclose(
                state["weights"][i], direct, rtol=2e-12, atol=2e-12
            )
            np.testing.assert_allclose(
                predict(state, x[:2], i),
                (x[:2] - x.mean(0)) / scale @ direct + y.mean(0),
                rtol=2e-12,
                atol=2e-12,
            )


def test_source_fit_coefficients_ignore_validation_and_e02_poison_and_accesses():
    rng = np.random.default_rng(5)
    feature = rng.normal(size=(18, 4))
    target = rng.normal(size=(18, 17, 2))
    metadata = dict(
        room=np.array(["E01"] * 6 + ["E03"] * 6 + ["E02"] * 6),
        record=np.repeat(["a", "b", "c"], 6),
    )
    train = np.arange(18) < 10
    before = fit_source_rows(feature, target, metadata, train)
    feature[~train] = 1e15
    target[~train] = np.nan
    after = fit_source_rows(feature, target, metadata, train)
    for name in before:
        for key in before[name]:
            np.testing.assert_array_equal(before[name][key], after[name][key])

    class Tracked:
        def __init__(self, values):
            self.values = values

        def __getitem__(self, ids):
            assert train[ids].all(), "Forbidden validation/held numerical read"
            return self.values[ids]

    tracked = fit_source_rows(Tracked(feature), Tracked(target), metadata, train)
    for name in before:
        np.testing.assert_array_equal(tracked[name]["weights"], before[name]["weights"])
    bad = train.copy()
    bad[-1] = True
    with pytest.raises(ValueError, match="source TRAIN"):
        fit_source_rows(feature, target, metadata, bad)


def test_predictor_accepts_no_record_labels_or_query_centering():
    assert list(inspect.signature(predict).parameters) == ["state", "x", "alpha_index"]
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=(10, 3)), rng.normal(size=(10, 34))
    state = fit_training(x, y, np.repeat(["a", "b"], 5))["fixed_effects"]
    query = rng.normal(size=(4, 3))
    output = predict(state, query, 0)
    # A query's output is unaffected by any other query in the batch.
    changed = query.copy()
    changed[1:] += 1e6
    np.testing.assert_array_equal(predict(state, changed, 0)[0], output[0])


def test_lag_pairs_are_exact_past8_same_record_and_never_bridge_missing_windows():
    record = np.array(["a", "a", "a", "b", "b"])
    ends = np.array([39, 47, 63, 39, 47])
    current, past = lag_pairs(record, ends)
    np.testing.assert_array_equal(current, [1, 4])
    np.testing.assert_array_equal(past, [0, 3])
    np.testing.assert_array_equal(record[current], record[past])
    assert (ends[current] - ends[past] == 8).all()
    with pytest.raises(ValueError, match="Duplicate"):
        lag_pairs(np.array(["a", "a"]), np.array([39, 39]))


def test_centered_motion_is_only_metric_and_does_not_hide_pose_offset_in_nje():
    rng = np.random.default_rng(7)
    target = rng.normal(size=(8, 17, 2))
    record = np.repeat(["a", "b"], 4)
    ends = np.tile([39, 47, 55, 63], 2)
    prediction = target + np.repeat([10.0, -7.0], 4)[:, None, None]
    metrics = motion_scores(prediction, target, record, ends)
    assert metrics["hip_relative_nje"] > 1
    assert metrics["record_centered_articulation_r2"] == pytest.approx(1)
    assert metrics["lag8_zero_reference_delta_r2"] == pytest.approx(1)
    static = motion_scores(
        np.broadcast_to(target.mean(0), target.shape), target, record, ends
    )
    assert static["record_centered_articulation_r2"] == pytest.approx(0)
    assert static["lag8_zero_reference_delta_r2"] == pytest.approx(0)


def test_zero_dynamic_energy_and_no_lag_pairs_are_unavailable_not_fabricated():
    target = np.tile(np.arange(34).reshape(1, 17, 2), (4, 1, 1)).astype(float)
    result = motion_scores(
        target, target, np.array(["a"] * 4), np.array([39, 47, 63, 79])
    )
    assert result["record_centered_target_squared_energy"] == 0
    assert result["record_centered_articulation_r2"] is None
    assert result["lag8_target_squared_energy"] == 0
    assert result["lag8_zero_reference_delta_r2"] is None
    current, past = lag_pairs(np.array(["a", "b"]), np.array([39, 47]))
    assert len(current) == len(past) == 0


def test_nested_energy_components_have_independent_analytic_values_and_identity():
    values = []
    room = []
    person = []
    record = []
    for r in range(2):
        for p in range(2):
            for a in range(2):
                for t in [-1, 0, 1]:
                    values.append(
                        np.array([10 * (2 * r - 1), 2 * (2 * p - 1)])
                        + (2 * a - 1) * np.array([1, 1])
                        + t * 0.5 * np.array([1, -1])
                    )
                    room.append(str(r))
                    person.append(f"{r}/{p}")
                    record.append(f"{r}/{p}/{a}")
    x = np.asarray(values)
    meta = {
        name: np.array(value)
        for name, value in [("room", room), ("person", person), ("record", record)]
    }
    result = variance_energy(x, meta, np.zeros(2), np.ones(2))
    assert result["components_squared_energy"] == pytest.approx(
        dict(
            between_room=2400,
            person_within_room=96,
            record_within_person=48,
            frame_within_record=8,
        )
    )
    assert result["cohort_centered_total_squared_energy"] == 2552
    assert result["centered_identity_relative_residual"] < 1e-15
    shifted = variance_energy(x + 5, meta, np.zeros(2), np.ones(2))
    assert shifted["train_origin_total_squared_energy"] == 3752
    assert shifted["cohort_centered_total_squared_energy"] == 2552
    assert shifted["common_cohort_mean_shift_squared_energy"] == 1200
    assert shifted["components_squared_energy"]["between_room"] == 2400
    assert shifted["shift_plus_centered_identity_relative_residual"] < 1e-15


def source_metadata():
    values = []
    removed_train = removed_val = 0
    for r in [1, 3]:
        for person in range(1 + (r - 1) * 10, 11 + (r - 1) * 10):
            validation = person in [4, 9, 24, 29]
            for action in range(1, 28):
                for end in range(39, 296, 8):
                    if validation and removed_val < 18:
                        removed_val += 1
                        continue
                    if not validation and removed_train < 132:
                        removed_train += 1
                        continue
                    values.append(
                        (
                            f"E{r:02}",
                            f"S{person:02}",
                            f"E{r:02}/S{person:02}/A{action:02}",
                            end,
                        )
                    )
    values.append(("E02", "S11", "E02/S11/A01", 39))
    return {
        key: np.array([row[i] for row in values])
        for i, key in enumerate(["room", "person", "record", "end"])
    }


def test_roles_match_frozen_source_counts_exclude_e02_and_reject_future_or_overlap():
    meta = source_metadata()
    train, val = source_roles(meta)
    assert (train.sum(), val.sum()) == (14124, 3546)
    assert not (train | val)[meta["room"] == "E02"].any()
    changed = {k: v.copy() for k, v in meta.items()}
    changed["end"][np.flatnonzero(train)[0]] += 1
    with pytest.raises(ValueError, match="canonical"):
        source_roles(changed)
    changed = {k: v.copy() for k, v in meta.items()}
    changed["record"][np.flatnonzero(val)[0]] = changed["record"][
        np.flatnonzero(train)[0]
    ]
    with pytest.raises(ValueError, match="overlap"):
        source_roles(changed)


def test_existing_output_refused_before_any_source_data_read(tmp_path):
    with pytest.raises(FileExistsError):
        run(tmp_path / "absent", tmp_path)


@pytest.mark.parametrize("n", [3, 8, 20, 33, 100])
def test_nonbinary_static_repetitions_have_exact_zero_motion_energy(n):
    base = np.linspace(0.01, 0.99, 34).reshape(17, 2)
    base = base - base[[11, 12]].mean(0)
    target = np.repeat(base[None], n, axis=0)
    result = motion_scores(target, target, np.array(["a"] * n), 39 + 8 * np.arange(n))
    assert result["record_centered_target_squared_energy"] == 0
    assert result["record_centered_articulation_r2"] is None
    assert result["lag8_target_squared_energy"] == 0
    assert result["lag8_zero_reference_delta_r2"] is None


def test_moving_nonbinary_trajectory_retains_analytic_motion_r2_and_constant_baseline_zero():
    base = np.linspace(0.01, 0.99, 34).reshape(17, 2)
    base = base - base[[11, 12]].mean(0)
    velocity = np.linspace(-0.04, 0.07, 34).reshape(17, 2)
    velocity = velocity - velocity[[11, 12]].mean(0)
    motion = np.arange(33)[:, None, None] * velocity
    target = base + motion
    prediction = base + 0.5 * motion
    record = np.array(["a"] * 33)
    ends = 39 + 8 * np.arange(33)
    result = motion_scores(prediction, target, record, ends)
    assert result["record_centered_articulation_r2"] == pytest.approx(0.75, abs=1e-14)
    assert result["lag8_zero_reference_delta_r2"] == pytest.approx(0.75, abs=1e-14)
    static = motion_scores(np.repeat(base[None], 33, axis=0), target, record, ends)
    assert static["record_centered_articulation_r2"] == 0
    assert static["lag8_zero_reference_delta_r2"] == 0


def variance_metadata():
    return dict(
        room=np.array(["a", "a", "b", "b"]),
        person=np.array(["a1", "a1", "b1", "b1"]),
        record=np.array(["r1", "r1", "r2", "r2"]),
    )


@pytest.mark.parametrize("value", [5.0, 0.1])
def test_identical_validation_shift_is_not_fabricated_between_room_variance(value):
    result = variance_energy(
        np.full((4, 2), value), variance_metadata(), np.zeros(2), np.ones(2)
    )
    assert result["cohort_centered_total_squared_energy"] == 0
    assert all(v == 0 for v in result["components_squared_energy"].values())
    assert all(v is None for v in result["component_fractions"].values())
    assert result["cumulative_between_record_fraction"] is None
    assert result["common_cohort_mean_shift_squared_energy"] == pytest.approx(
        8 * value * value
    )
    assert result["train_origin_total_squared_energy"] == pytest.approx(
        8 * value * value
    )
    assert result["common_cohort_mean_shift_fraction_of_train_origin"] == pytest.approx(
        1
    )


def test_common_shift_and_true_room_variation_are_separate_and_normalization_is_fixed():
    meta = variance_metadata()
    values = np.array([[4.0, 5.0], [4.0, 5.0], [6.0, 5.0], [6.0, 5.0]])
    mean = np.array([0.0, 0.0])
    scale = np.array([1.0, 2.0])
    before_mean, before_scale = mean.copy(), scale.copy()
    result = variance_energy(values, meta, mean, scale)
    assert result["cohort_centered_total_squared_energy"] == 4
    assert result["components_squared_energy"]["between_room"] == 4
    assert result["common_cohort_mean_shift_squared_energy"] == 125
    assert result["train_origin_total_squared_energy"] == 129
    # Changing measurement units together with the PRE-EXISTING TRAIN
    # normalization cannot alter normalized descriptive energies.
    factor = np.array([2.0, 3.0])
    offset = np.array([9.0, -7.0])
    affine = variance_energy(
        values * factor + offset, meta, mean * factor + offset, scale * factor
    )
    for key in [
        "cohort_centered_total_squared_energy",
        "common_cohort_mean_shift_squared_energy",
        "train_origin_total_squared_energy",
    ]:
        assert affine[key] == pytest.approx(result[key], abs=1e-12)
    np.testing.assert_array_equal(mean, before_mean)
    np.testing.assert_array_equal(scale, before_scale)


def test_validation_descriptive_centering_cannot_change_training_coefficients_or_predictor():
    rng = np.random.default_rng(7)
    x, y = rng.normal(size=(12, 3)), rng.normal(size=(12, 34))
    record = np.repeat(["a", "b", "c"], 4)
    before = fit_training(x, y, record)
    query = rng.normal(size=(4, 3))
    expected = predict(before["fixed_effects"], query, 0)
    meta = variance_metadata()
    variance_energy(
        query + 50,
        meta,
        before["fixed_effects"]["x_mean"],
        before["fixed_effects"]["feature_scale"],
    )
    after = fit_training(x, y, record)
    for decoder in before:
        for key in before[decoder]:
            np.testing.assert_array_equal(before[decoder][key], after[decoder][key])
    np.testing.assert_array_equal(expected, predict(before["fixed_effects"], query, 0))


def test_checked_blas_product_matches_independent_scalar_and_einsum_for_strided_inputs():
    import math

    rng = np.random.default_rng(19)
    a = rng.normal(size=(13, 9))[:, ::2]
    b = rng.normal(size=(7, 5)).T
    before_a, before_b = a.copy(), b.copy()
    actual = checked_product(a, b)
    scalar = np.array(
        [
            [
                math.fsum(float(x) * float(y) for x, y in zip(row, column))
                for column in b.T
            ]
            for row in a
        ]
    )
    einsum = np.einsum("ik,kj->ij", a, b, optimize=False)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, scalar, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(actual, einsum, rtol=2e-14, atol=2e-14)
    np.testing.assert_array_equal(a, before_a)
    np.testing.assert_array_equal(b, before_b)
    np.testing.assert_allclose(
        checked_product(a.astype(np.float32), b.astype(np.float32)),
        np.einsum(
            "ik,kj->ij",
            a.astype(np.float32).astype(float),
            b.astype(np.float32).astype(float),
        ),
        rtol=2e-14,
        atol=2e-14,
    )


def test_checked_blas_rejects_invalid_inputs_and_genuine_overflow_without_warning_suppression():
    for a, b in [
        (np.ones(3), np.ones((3, 2))),
        (np.ones((2, 3)), np.ones((2, 4))),
        (np.ones((0, 2)), np.ones((2, 3))),
        (np.full((2, 2), np.nan), np.ones((2, 2))),
        (np.ones((2, 2), dtype=complex), np.ones((2, 2))),
    ]:
        with pytest.raises(ValueError):
            checked_product(a, b)
    with pytest.raises(ValueError, match="output"):
        checked_product(np.full((2, 2), 1e308), np.full((2, 2), 2.0))
