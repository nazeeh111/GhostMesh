from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

spec = spec_from_file_location("observation_positive", Path(__file__).resolve().parents[1] / "scripts/domain_observation_positive_probe.py")
probe = module_from_spec(spec)
spec.loader.exec_module(probe)


def test_zero_phase_ablation_retains_all_magnitude_features_without_mutation():
    frame = np.arange(297 * 12 * 114, dtype="float64").reshape(297, 12, 114)
    original = frame.copy()
    support = np.array([np.arange(8), np.arange(11, 19)])
    full = probe.windows(frame, support, "observation_full").reshape(2, 8, 12, 114)
    ablated = probe.windows(frame, support, "observation_amplitude_zero_phase").reshape(2, 8, 12, 114)
    assert np.array_equal(full[:, :, :6], ablated[:, :, :6])
    assert not ablated[:, :, 6:].any()
    assert np.array_equal(frame, original)


def test_summary_is_prefix_causal_before_window_construction():
    rng = np.random.default_rng(12)
    rf = rng.normal(size=(297, 6, 114, 10)).astype("float32")
    rf[:, :3] += 40
    full = probe.observation_features(rf, magnitude_units="log_magnitude_db", phase_units="radians")
    changed = rf.copy()
    changed[118:, :3] += 100
    changed[118:, 3:] += 2
    other = probe.observation_features(changed, magnitude_units="log_magnitude_db", phase_units="radians")
    assert np.array_equal(full[:118], other[:118])
    assert not np.array_equal(full[118:], other[118:])
