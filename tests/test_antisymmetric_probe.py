import numpy as np
import pytest

from scripts.domain_antisymmetric_probe import (
    CONFIG,
    designs,
    fit_map,
    fit_ridge,
    multiply,
    pair_features,
    swap,
)


def fixture():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(160, 456))
    train = np.arange(160) < 96
    validation = (np.arange(160) >= 96) & (np.arange(160) < 128)
    held = np.arange(160) >= 128
    return rng, x, train, validation, held


def test_swap_is_involution_and_lag8_extraction_preserves_time_and_bins():
    rng, x, *_ = fixture()
    np.testing.assert_array_equal(swap(swap(x)), x)
    full = np.arange(2 * 8 * 19 * 6 * 3).reshape(2, -1)
    expected = full.reshape(2, 8, 19, 6, 3)[:, :, :, 3, :].reshape(2, 456)
    np.testing.assert_array_equal(pair_features(full), expected)
    with pytest.raises(ValueError):
        swap(rng.normal(size=(2, 455)))


def test_symmetric_source_only_stats_and_zero_odd_mean():
    _, x, train, _, held = fixture()
    before = fit_map(x, train)
    source = np.concatenate((x[train], swap(x[train])))
    np.testing.assert_allclose(before["input_mean"], source.mean(0), atol=1e-14)
    np.testing.assert_allclose(
        before["input_scale"], np.maximum(source.std(0), CONFIG["floor"]), atol=1e-14
    )
    x[held] += 100000
    after = fit_map(x, train)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    a, b = designs(x, before)["exact_odd"]
    np.testing.assert_array_equal(a + b, 0)
    assert np.all(before["odd_scale"] >= 0.001)


def test_actual_swapped_input_predictions_negate_and_static_input_is_exact_zero():
    rng, x, train, *_ = fixture()
    mapping = fit_map(x, train)
    w = rng.normal(size=(512, 34))
    original = designs(x, mapping)["exact_odd"][0]
    swapped = designs(swap(x), mapping)["exact_odd"][0]
    np.testing.assert_allclose(
        multiply(original, w), -multiply(swapped, w), atol=1e-12, rtol=1e-12
    )
    static = designs(np.zeros((3, 456)), mapping)["exact_odd"][0]
    np.testing.assert_array_equal(multiply(static, w), np.zeros((3, 34)))


def test_augmented_equations_match_direct_solve_and_ignore_held_labels():
    rng = np.random.default_rng(9)
    a, b = rng.normal(size=(40, 6)), rng.normal(size=(40, 6))
    y = rng.normal(size=(40, 34))
    train, validation, held = (
        np.arange(40) < 20,
        (np.arange(40) >= 20) & (np.arange(40) < 30),
        np.arange(40) >= 30,
    )
    chosen, weights, candidates = fit_ridge(a, b, y, train, validation)
    augmented = np.concatenate((a[train], b[train]))
    labels = np.concatenate((y[train], -y[train]))
    for alpha, weight in zip(CONFIG["alphas"], weights):
        direct = np.linalg.solve(
            multiply(augmented.T, augmented) + 2 * train.sum() * alpha * np.eye(6),
            multiply(augmented.T, labels),
        )
        np.testing.assert_allclose(weight, direct, atol=1e-12, rtol=1e-12)
    y[held] = 1000000
    chosen2, weights2, candidates2 = fit_ridge(a, b, y, train, validation)
    assert chosen == chosen2 and candidates == candidates2
    np.testing.assert_array_equal(weights, weights2)
