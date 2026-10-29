"""SOFTWARE ONLY: RF invariance, causal EMA startup and known motion traces."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ghostmesh.causal_channel import channel_representations

SPEC = importlib.util.spec_from_file_location(
    "background_probe_test",
    Path(__file__).resolve().parents[1] / "scripts/domain_background_probe.py",
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def rf_fixture():
    rng = np.random.default_rng(2027)
    return np.concatenate(
        [rng.normal(40, 3, (48, 3, 114, 10)), rng.normal(0, 2, (48, 3, 114, 10))],
        axis=1,
    ).astype("float32")


def test_background_is_causal_and_static_control_uses_only_startup():
    x = rf_fixture()
    y = x.copy()
    y[35:, :3] += 10
    y[35:, 3:] += 2
    first, _ = channel_representations(x)
    second, _ = channel_representations(y)
    for key in ["untreated", "ema_residual"]:
        np.testing.assert_array_equal(first[key][:35], second[key][:35])
    np.testing.assert_array_equal(first["startup_static"], second["startup_static"])
    # A change at frame31 affects its residual, because the current frame does not fit its own background.
    y = x.copy()
    y[31, 1] += 10
    changed, _ = channel_representations(y)
    assert not np.allclose(first["ema_residual"][31], changed["ema_residual"][31])


def test_static_channel_residual_and_common_gain_phase_invariance():
    x = rf_fixture()
    static = np.repeat(x[:1], len(x), axis=0)
    # Varying packet columns are retained; a fully static per-subcarrier complex channel cancels.
    static = np.repeat(static[:, :, :, :1], 10, axis=-1)
    representations, _ = channel_representations(static)
    np.testing.assert_allclose(representations["ema_residual"][31:], 0, atol=3e-6)
    y = x.copy()
    y[:, :3] += 15
    y[:, 3:] += 1.7
    other, _ = channel_representations(y)
    first, _ = channel_representations(x)
    for key in first:
        np.testing.assert_allclose(first[key], other[key], atol=2e-5)


def test_invalid_pairs_are_neutral_and_never_nonfinite():
    x = rf_fixture()
    x[5, 0, 0, 0] = -np.inf
    x[5, 3, 1, 0] = np.nan
    representations, quality = channel_representations(x)
    assert quality["invalid_pair_fraction"] > 0
    for value in representations.values():
        assert np.isfinite(value).all()
    assert np.all(representations["untreated"][5, :, 0, 0] == 0)
    assert np.all(representations["ema_residual"][5, :, 0, 0] == 0)


def test_centered_motion_r2_known_traces_and_shuffle():
    y = np.arange(8, dtype=float)[:, None, None] * np.ones((1, 17, 2))
    record = np.array(["SOFTWARE"] * 8)
    mean = np.broadcast_to(y.mean(0), y.shape)
    assert probe.centered_r2(y, y, record)["r2"] == pytest.approx(1)
    assert probe.centered_r2(mean, y, record)["r2"] == pytest.approx(0)
    assert probe.centered_r2(2 * mean - y, y, record)["r2"] == pytest.approx(-3)


def test_rff_scaling_and_bandwidth_never_fit_held_values(tmp_path):
    rng = np.random.default_rng(21)
    x = rng.normal(size=(160, 8)).astype("float32")
    training = np.arange(160) < 140
    y = x.copy()
    y[~training] += 1000
    first = probe.fit_rff(x, training, tmp_path / "first.npz")
    second = probe.fit_rff(y, training, tmp_path / "second.npz")
    np.testing.assert_array_equal(first[training], second[training])
    with np.load(tmp_path / "first.npz") as a, np.load(tmp_path / "second.npz") as b:
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
