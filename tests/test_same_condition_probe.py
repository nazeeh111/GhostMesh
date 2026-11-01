from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

spec = spec_from_file_location("same_condition_probe", Path(__file__).resolve().parents[1] / "scripts/domain_same_condition_probe.py")
probe = module_from_spec(spec)
spec.loader.exec_module(probe)


def test_time_blocks_have_disjoint_causal_contexts_and_eight_frame_purge():
    blocks = probe.CONFIG["held_time"]["raw_blocks_half_open"]
    supports = []
    for name, expected_count in (("train", 111), ("validation", 45), ("test", 104)):
        end, support = probe.endpoints(*blocks[name])
        assert len(end) == expected_count
        assert np.array_equal(support[:, -1], end)
        supports.append(set(support.ravel()))
    assert not supports[0] & supports[1] and not supports[0] & supports[2] and not supports[1] & supports[2]
    assert min(supports[1]) - max(supports[0]) - 1 == 8
    assert min(supports[2]) - max(supports[1]) - 1 == 8


def test_normalization_and_kernel_bandwidth_fit_training_only():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(50, 10))
    train = np.arange(len(x)) < 30
    z, state = probe.normalize_fit(x, train)
    changed = x.copy()
    changed[~train] += 100
    other, second = probe.normalize_fit(changed, train)
    assert all(np.array_equal(state[key], second[key]) for key in state)
    first_k, _, width1 = probe.kernel_inputs(z, train)
    second_k, _, width2 = probe.kernel_inputs(other, train)
    assert width1 == width2
    for key in first_k:
        np.testing.assert_allclose(first_k[key][train], second_k[key][train], atol=1e-12)


def test_dual_linear_solver_recovers_known_csi_mapping():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(70, 10))
    train = np.arange(len(x)) < 50
    z, _ = probe.normalize_fit(x, train)
    kernel, _, _ = probe.kernel_inputs(z, train)
    target = (probe.multiply(z, rng.normal(size=(10, 34))) + 2).reshape(70, 17, 2)
    regularization, dual, mean, residual = next(probe.dual_solutions(kernel["linear"], target, train))
    prediction = mean + probe.multiply(kernel["linear"], dual)
    assert regularization == 1e-5 and residual < 1e-8
    assert np.mean(abs(prediction[~train] - target[~train].reshape(20, 34))) < 1e-4


def test_lag_control_preserves_current_frame_and_all_values():
    x = np.arange(5 * 8 * 3).reshape(5, 8 * 3)
    altered = probe.lag_permute(x)
    assert not np.array_equal(x, altered)
    assert np.array_equal(x.reshape(5, 8, 3)[:, -1], altered.reshape(5, 8, 3)[:, -1])
    assert np.array_equal(np.sort(x, axis=1), np.sort(altered, axis=1))
