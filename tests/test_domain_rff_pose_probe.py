from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

spec = spec_from_file_location("rff_pose_probe", Path(__file__).resolve().parents[1] / "scripts/domain_rff_pose_probe.py")
probe = module_from_spec(spec)
spec.loader.exec_module(probe)


def test_design_uses_source_training_only_and_roundtrips():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(50, 8))
    training = np.arange(len(x)) < 35
    for kernel in ("linear", "rbf512"):
        design, state = probe.fit_design(x, training, kernel)
        np.testing.assert_allclose(probe.apply_design(x, state), design, atol=1e-13)
        changed = x.copy()
        changed[~training] += 200
        other, altered = probe.fit_design(changed, training, kernel)
        assert state.keys() == altered.keys()
        for key in state:
            np.testing.assert_allclose(state[key], altered[key], atol=1e-13)
        np.testing.assert_allclose(design[training], other[training], atol=1e-13)


def test_ridge_fits_known_linear_targets_and_has_small_residual():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(100, 8))
    training = np.arange(len(x)) < 80
    design, _ = probe.fit_design(x, training, "linear")
    target = (probe.multiply(design, rng.normal(size=(8, 34))) + 2).reshape(100, 17, 2).astype("float32")
    alpha, weights, mean, residual = next(probe.ridge_solutions(design, target, training))
    prediction = probe.multiply(design, weights) + mean
    assert alpha == 0.01 and residual < 1e-10
    assert np.array_equal(mean.reshape(17, 2), target[training].mean(0, dtype="float64"))
    assert np.mean((prediction[~training] - target[~training].reshape(20, 34)) ** 2) < 0.01


def test_shuffle_preserves_each_record_without_target_or_interrecord_mixing():
    prediction = np.arange(12 * 17 * 2).reshape(12, 17, 2)
    sequence = np.repeat(["a", "b", "c"], 4)
    shuffled = probe.shuffled_within_record(prediction, sequence)
    assert not np.array_equal(shuffled, prediction)
    for record in set(sequence):
        mask = sequence == record
        assert sorted(shuffled[mask].reshape(-1)) == sorted(prediction[mask].reshape(-1))
