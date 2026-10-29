import numpy as np
import pytest

from ghostmesh.doppler_features import (
    clock_audit,
    complex_quotients,
    signed_band_power,
    spectral_features,
)


def rf_tones(frames=8, frequency=7.5):
    time = np.arange(frames * 10).reshape(frames, 10) / 100.0
    rf = np.zeros((frames, 6, 114, 10))
    rf[:, :3] = 40.0
    rf[:, 4] = 2 * np.pi * frequency * time[:, None, :]
    rf[:, 5] = -2 * np.pi * frequency * time[:, None, :]
    return rf


def quotients(rf):
    return complex_quotients(
        rf, magnitude_units="log_magnitude_db", phase_units="radians"
    )[0]


def test_complex_positive_negative_tones_have_expected_signed_band_and_flat_amplitude_valid():
    raw = rf_tones()
    assert np.std(raw[:, :3]) == 0
    q = quotients(raw)
    power = signed_band_power(q)
    assert power.shape == (2, 19, 10)
    assert power[0, :, 5:].sum() > 1e6 * power[0, :, :5].sum()
    assert power[1, :, :5].sum() > 1e6 * power[1, :, 5:].sum()
    assert np.argmax(power[0, 0]) == 7 and np.argmax(power[1, 0]) == 2
    assert np.isfinite(spectral_features(q)).all()


def test_common_per_packet_gain_and_phase_cancel_before_spectrum():
    raw = rf_tones()
    rng = np.random.default_rng(31)
    changed = raw.copy()
    changed[:, :3] += rng.normal(size=(8, 1, 114, 10)) * 10
    changed[:, 3:] += rng.normal(size=(8, 1, 114, 10)) * 20
    first, second = quotients(raw), quotients(changed)
    np.testing.assert_allclose(first, second, rtol=0, atol=1e-13)
    np.testing.assert_allclose(
        spectral_features(first), spectral_features(second), rtol=0, atol=1e-8
    )
    # A constant per-receiver/subcarrier phase rotates each time trace only.
    constant = raw.copy()
    constant[:, 3:] += rng.normal(size=(1, 3, 114, 1))
    np.testing.assert_allclose(
        spectral_features(first),
        spectral_features(quotients(constant)),
        rtol=0,
        atol=1e-8,
    )


def test_static_complex_signal_has_no_dynamic_power_and_future_frames_do_not_enter_window():
    raw = rf_tones(frames=12)
    expected = spectral_features(quotients(raw[:8]))
    raw[8:] += 1000
    np.testing.assert_array_equal(expected, spectral_features(quotients(raw[:8])))
    static = quotients(
        np.concatenate(
            (np.ones((8, 3, 114, 10)) * 40, np.zeros((8, 3, 114, 10))), axis=1
        )
    )
    np.testing.assert_array_equal(spectral_features(static), np.full(380, -120.0))


def test_missing_pair_rules_are_identical_for_every_packet_order_and_no_flat_guard():
    raw = rf_tones()
    raw[0, 0, 0, :4] = np.nan
    q, receipt = complex_quotients(
        raw, magnitude_units="log_magnitude_db", phase_units="radians"
    )
    assert receipt["missing_pairs"] == 8
    for variant in ("natural", "within_frame_shuffle", "whole_window_shuffle"):
        assert np.isfinite(spectral_features(q, variant)).all()
    raw[0, 0, 0, 4] = np.nan
    q = quotients(raw)
    for variant in ("natural", "within_frame_shuffle", "whole_window_shuffle"):
        with pytest.raises(ValueError, match="5percent"):
            spectral_features(q, variant)
    with pytest.raises(ValueError):
        complex_quotients(rf_tones(), magnitude_units="linear", phase_units="radians")


def test_nominal_clock_gate_detects_a_smooth_tone_but_not_large_frame_phase_resets():
    raw = rf_tones(frames=20, frequency=3.75)
    smooth = clock_audit(quotients(raw))
    assert smooth["plausible_nominal_continuity"]
    broken = rf_tones(frames=20, frequency=0.01)
    broken[:, 4] += np.random.default_rng(32).normal(size=(20, 1, 1)) * 3
    broken[:, 5] += np.random.default_rng(33).normal(size=(20, 1, 1)) * 3
    report = clock_audit(quotients(broken))
    assert not report["plausible_nominal_continuity"]
    assert report["boundary_to_within_median_ratio"] > 10


def test_pose_and_displacement_supports_and_masks_stay_inside_purged_blocks():
    from scripts.domain_packet_time_probe import CONFIG, window_designs

    raw = rf_tones(frames=297, frequency=3.75)
    values, ends, splits, eligible, rejected = window_designs(raw)
    assert not rejected
    assert {
        name: int((splits == name).sum()) for name in ("train", "validation", "test")
    } == {"train": 111, "validation": 45, "test": 104}
    assert {
        name: int(((splits == name) & eligible).sum())
        for name in ("train", "validation", "test")
    } == {"train": 110, "validation": 44, "test": 103}
    for name, (start, stop) in CONFIG["positive_control"][
        "raw_blocks_half_open"
    ].items():
        chosen = ends[splits == name]
        assert chosen.min() - 7 >= start and chosen.max() < stop
        assert (ends[(splits == name) & eligible] - 8).min() >= start
    assert all(
        values[k].shape == (260, 380)
        for k in ("natural", "within_frame_shuffle", "whole_window_shuffle")
    )
    raw[190, 0, 0, :5] = np.nan
    changed, ends2, split2, eligible2, invalid = window_designs(raw)
    assert invalid and all(not end - 7 <= 190 <= end for end in ends2[split2 == "test"])
    assert all(len(value) == len(ends2) for value in changed.values())


def test_kernel_normal_equations_readback_and_validation_selection_ignore_held_labels(
    tmp_path,
):
    from scripts.domain_packet_time_probe import fit_kernel

    rng = np.random.default_rng(41)
    x = rng.normal(size=(14, 9))
    target = rng.normal(size=(14, 17, 2)) * 0.01
    body = rng.uniform(0.2, 0.8, size=(14, 17, 2))
    split = np.array(["train"] * 7 + ["validation"] * 3 + ["test"] * 4)
    before = tmp_path / "before"
    before.mkdir()
    rows, checks = fit_kernel(x, target, body, split, "delta", "synthetic", before)
    for check in checks.values():
        assert check["maximum_normal_equation_residual"] < 1e-6
        assert check["actual_input_prediction_max_abs"] < 1e-9
    target[split == "test"] += 1000
    after = tmp_path / "after"
    after.mkdir()
    replay, _ = fit_kernel(x, target, body, split, "delta", "synthetic", after)
    assert [r["regularization"] for r in rows] == [r["regularization"] for r in replay]
    for kernel in ("linear", "rbf"):
        with (
            np.load(
                before / f"delta-synthetic-{kernel}-model.npz", allow_pickle=False
            ) as a,
            np.load(
                after / f"delta-synthetic-{kernel}-model.npz", allow_pickle=False
            ) as b,
        ):
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
