import numpy as np
import pytest

from scripts.domain_spectral_shape_probe import (
    fit_map,
    fit_readouts,
    log_power_fractions,
    normalize_cache,
    predict,
    transform,
)
from scripts.domain_packet_transfer_probe import multiply


def test_arbitrary_per_vector_power_gain_cancels_and_power_sums_to_one():
    rng = np.random.default_rng(81)
    x = rng.uniform(-110, 20, size=(9, 2, 19, 10))
    shifts = rng.uniform(-100, 100, size=(9, 2, 19, 1))
    a = log_power_fractions(x.reshape(9, 380)).reshape(x.shape)
    b = log_power_fractions((x + shifts).reshape(9, 380)).reshape(x.shape)
    np.testing.assert_allclose(a, b, rtol=0, atol=7e-14)
    np.testing.assert_allclose((10 ** (a / 10)).sum(-1), 1, rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        log_power_fractions(np.full(380, -120.0)), -10, atol=2e-14
    )


def test_normalization_uses_ten_signed_bands_and_keeps_link_bin_axes_separate():
    x = np.zeros((2, 19, 10))
    base = log_power_fractions(x.reshape(380)).reshape(x.shape)
    x[1, 7, 3] = 10
    after = log_power_fractions(x.reshape(380)).reshape(x.shape)
    changed = np.any(abs(after - base) > 1e-8, axis=-1)
    assert changed.sum() == 1 and changed[1, 7]
    np.testing.assert_allclose(after[1, 7, 3], 10 - 10 * np.log10(19), atol=1e-14)
    np.testing.assert_allclose(after[1, 7, :3], -10 * np.log10(19), atol=1e-14)


def test_other_and_future_examples_cannot_change_current_window_features():
    rng = np.random.default_rng(82)
    x = rng.normal(size=(7, 380)) * 40
    before = log_power_fractions(x)
    changed = x.copy()
    changed[1:] = rng.normal(size=(6, 380)) * 1e4
    np.testing.assert_array_equal(before[0], log_power_fractions(changed)[0])
    np.testing.assert_array_equal(before[0], log_power_fractions(x[0]))
    np.testing.assert_array_equal(
        before, np.vstack([log_power_fractions(row) for row in x])
    )


def test_source_maps_and_selection_ignore_query_features_and_query_labels():
    rng = np.random.default_rng(83)
    x = log_power_fractions(rng.normal(size=(48, 380)))
    train, val = np.arange(48) < 24, (np.arange(48) >= 24) & (np.arange(48) < 36)
    # A small fixture tests the actual shared fitting helpers, not the production width.
    z, mapping = fit_map(x[:, :12], train, "linear")
    y = rng.normal(size=(48, 104)) * 0.01
    y[:, :34] += 0.5
    model, grid = fit_readouts(z, y, train, val)
    x[36:] += 1e6
    _, after_mapping = fit_map(x[:, :12], train, "linear")
    y[36:] += 1e9
    after, after_grid = fit_readouts(z, y, train, val)
    assert grid == after_grid
    for key in mapping:
        np.testing.assert_array_equal(mapping[key], after_mapping[key])
    for key in model:
        np.testing.assert_array_equal(model[key], after[key])


def test_saved_normalization_map_weights_readback_and_direct_system(tmp_path):
    rng = np.random.default_rng(84)
    raw = rng.normal(size=(40, 380)) * 20
    x = log_power_fractions(raw).astype(np.float32)
    train, val = np.arange(40) < 20, (np.arange(40) >= 20) & (np.arange(40) < 30)
    z, mapping = fit_map(x[:, :11], train, "linear")
    y = rng.normal(size=(40, 104)) * 0.01
    y[:, :34] += 0.5
    model, _ = fit_readouts(z, y, train, val)
    centered = z[train] - model["x_mean"]
    rhs = multiply(centered.T, y[train] - model["y_mean"])
    gram = multiply(centered.T, centered)
    for alpha, weight in zip(model["alphas"], model["weights"]):
        direct = np.linalg.solve(gram + train.sum() * alpha * np.eye(11), rhs)
        np.testing.assert_allclose(weight, direct, rtol=0, atol=1e-12)
    np.save(tmp_path / "features.npy", x)
    np.savez(tmp_path / "map.npz", **mapping)
    np.savez(tmp_path / "model.npz", **model)
    with np.load(tmp_path / "map.npz") as f:
        saved_map = dict(f)
    with np.load(tmp_path / "model.npz") as f:
        saved_model = dict(f)
    reloaded = predict(
        transform(np.load(tmp_path / "features.npy")[:, :11], saved_map), saved_model
    )
    for name, value in predict(z, model).items():
        np.testing.assert_array_equal(value, reloaded[name])


def test_invalid_features_and_existing_output_are_refused(tmp_path):
    for value in (np.zeros(379), np.full(380, np.nan), np.full(380, np.inf)):
        with pytest.raises(ValueError):
            log_power_fractions(value)
    with pytest.raises(FileExistsError):
        normalize_cache(None, tmp_path)
