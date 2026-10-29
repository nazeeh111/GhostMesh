import numpy as np
import pytest

from scripts.domain_packet_transfer_probe import (
    SLICES,
    build,
    endpoint_control,
    fit,
    fit_map,
    fit_readouts,
    join_labels,
    multiply,
    transform,
    window_features,
)


def test_label_join_uses_same_record_and_exact_covariance_projection():
    rng = np.random.default_rng(61)
    pose = rng.uniform(0.1, 0.9, size=(297, 17, 2))
    ends = np.array([39, 47, 295])
    relative = pose - pose[:, [11, 12]].mean(1, keepdims=True)
    labels = join_labels(
        pose, ends, relative[ends], relative[ends] - relative[ends - 8]
    )
    np.testing.assert_array_equal(labels["pose_full"], pose[ends])
    np.testing.assert_array_equal(labels["delta_full"], pose[ends] - pose[ends - 8])
    np.testing.assert_allclose(
        labels["delta_root"] + labels["delta_relative"],
        labels["delta_full"],
        rtol=0,
        atol=1e-15,
    )
    with pytest.raises(AssertionError):
        join_labels(pose, ends, relative[ends], relative[ends] - relative[ends - 8] + 1)


def test_all_spectral_variants_share_additional_missing_trace_mask():
    raw = np.zeros((297, 6, 114, 10))
    raw[:, :3] = 40
    raw[80:83, 0, 0, :2] = np.nan
    values, keep, rejected = window_features(raw, np.array([39, 47, 87, 95]))
    np.testing.assert_array_equal(keep, [0, 1, 3])
    assert rejected[0]["end"] == 87
    assert all(x.shape == (3, 380) for x in values.values())


def test_feature_maps_fit_only_training_rows_and_reload_exactly():
    rng = np.random.default_rng(62)
    x = rng.normal(size=(256, 12))
    train = np.arange(256) < 192
    for family in ("linear", "rff"):
        z, mapping = fit_map(x, train, family)
        np.testing.assert_array_equal(transform(x, mapping), z)
        changed = x.copy()
        changed[~train] += 1e6
        new, after = fit_map(changed, train, family)
        np.testing.assert_array_equal(new[train], z[train])
        for key in mapping:
            np.testing.assert_array_equal(mapping[key], after[key])


def test_readouts_match_normal_equations_and_held_labels_cannot_change_selection():
    rng = np.random.default_rng(63)
    z = rng.normal(size=(40, 9))
    y = rng.normal(size=(40, 104)) * 0.01
    y[:, :34] += rng.uniform(0.1, 0.9, size=(40, 34))
    train = np.arange(40) < 20
    val = (np.arange(40) >= 20) & (np.arange(40) < 30)
    model, grid = fit_readouts(z, y, train, val)
    x = z[train] - z[train].mean(0)
    target = y[train] - y[train].mean(0)
    for alpha, weight in zip(model["alphas"], model["weights"]):
        direct = np.linalg.solve(
            multiply(x.T, x) + train.sum() * alpha * np.eye(9), multiply(x.T, target)
        )
        np.testing.assert_allclose(weight, direct, rtol=0, atol=1e-12)
    y[30:] += 1e9
    after, grid2 = fit_readouts(z, y, train, val)
    assert grid == grid2
    for key in model:
        np.testing.assert_array_equal(model[key], after[key])


def test_endpoint_control_uses_training_clock_only_and_ignores_query_labels():
    n = 8
    data = {
        name: np.arange(n, dtype=float)[:, None, None]
        * np.ones((n, 1 if name == "delta_root" else 17, 2))
        for name in SLICES
    }
    data["end"] = np.array([39, 39, 47, 47, 39, 47, 55, 55])
    train = np.arange(n) < 4
    ids = np.arange(4, 8)
    before, fallback = endpoint_control(data, train, ids)
    assert fallback == [55]
    for value in before.values():
        np.testing.assert_array_equal(value[:, 0, 0], [0.5, 2.5, 1.5, 1.5])
    for name in SLICES:
        data[name][~train] += 1e9
    after, fallback2 = endpoint_control(data, train, ids)
    assert fallback == fallback2
    for name in SLICES:
        np.testing.assert_array_equal(before[name], after[name])


def test_output_collision_refused_before_reading_sources(tmp_path):
    with pytest.raises(FileExistsError):
        build(None, None, None, tmp_path)
    with pytest.raises(FileExistsError):
        fit(None, None, tmp_path)
