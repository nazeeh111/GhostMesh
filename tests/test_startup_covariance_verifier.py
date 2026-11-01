"""Independent startup verifier checks on analytic spectra and isolated source data."""

import numpy as np
import pytest

from scripts import verify_startup_covariance_transfer_probe as proof
from scripts.domain_covariance_source_probe import rff_transform
from scripts.domain_reference_covariance_probe import fit_readout


def test_independent_diagonal_generalized_spectrum_matches_closed_form():
    d = np.array([1.0, 2.0, 5.0])[None, None, None, :] * np.exp(
        np.arange(3)[:, None, None, None]
        * np.array([-0.2, 0.1, 0.3])[None, None, None, :]
    )
    d = np.broadcast_to(d, (3, 8, 19, 3)).copy()
    c = np.zeros((3, 8, 19, 3, 3), complex)
    c[..., (0, 1, 2), (0, 1, 2)] = d
    coords = np.zeros((3, 8, 19, 9))
    coords[..., :3] = d
    feature, anchor, volume = proof.independent_features(c, coords, 0)
    shape = d / np.exp(np.log(d).mean(-1))[..., None]
    b = shape[0].mean(0)
    b = b / np.exp(np.log(b).mean(-1))[..., None]
    mid = (shape + b) / 2
    expected = np.sort(np.log((shape + 1e-4 * mid) / (b + 1e-4 * mid)), axis=-1)
    expected -= expected.mean(-1, keepdims=True)
    np.testing.assert_allclose(
        feature["startup_shape"], expected.reshape(3, 456), atol=1e-15
    )
    np.testing.assert_allclose(
        np.diagonal(anchor, axis1=-2, axis2=-1), shape[0].mean(0), atol=1e-15
    )
    np.testing.assert_allclose(volume, np.log(d[0]).mean(-1).mean(0), atol=1e-15)
    np.testing.assert_array_equal(
        feature["startup_frozen_raw"][0], feature["startup_frozen_raw"][2]
    )


def test_independent_decode_reads_only_current_and_restores_complex_hermitian():
    x = np.full((2, 8, 19, 7, 9), np.nan)
    coords = np.array([2.0, 3.0, 4.0, 0.3, 0.4, 0.5, -0.2, 0.1, 0.6])
    x[:, :, :, 0] = coords
    c, observed = proof.decode(x.reshape(2, -1))
    np.testing.assert_allclose(c, c.conj().swapaxes(-1, -2), atol=0)
    np.testing.assert_allclose(c[0, 0, 0, 0, 1], (0.3 - 0.2j) / np.sqrt(2))
    np.testing.assert_array_equal(observed[0, 0, 0], coords)


def test_metric_centers_each_record_prediction_and_teacher_separately():
    rng = np.random.default_rng(161)
    target = rng.normal(size=(8, 17, 2))
    record = np.repeat(["a", "b"], 4)
    pred = target + np.where((record == "a")[:, None, None], 3.0, -5.0)
    metric = proof.metric(pred, target, record)
    assert metric["r2"] == pytest.approx(1) and metric["nje"] > 0
    assert metric["valid_frames"] == 8 and metric["valid_joints"] == 136
    with pytest.raises(ValueError, match="Nonfinite"):
        proof.metric(pred * np.nan, target, record)


def test_roles_are_person_disjoint_and_all_ten_held_people_required():
    room = np.repeat(["E01", "E02", "E03"], 10)
    person = np.array([f"S{i:02}" for i in range(1, 31)])
    data = dict(room=room, person=person)
    masks = proof.roles(data, "E02")
    assert tuple(mask.sum() for mask in masks) == (16, 4, 10)
    assert all(
        not np.any(a & b)
        for a, b in ((masks[0], masks[1]), (masks[0], masks[2]), (masks[1], masks[2]))
    )
    data["person"][10] = "S12"
    with pytest.raises(ValueError, match="roles"):
        proof.roles(data, "E02")


def test_map_verifier_checks_rng_stats_and_bandwidth_without_held_information(tmp_path):
    rng = np.random.default_rng(162)
    feature = rng.normal(size=(170, 7))
    train = np.arange(170) < 140
    path = tmp_path / "map.npz"
    original = rff_transform(feature, train, path)
    state = proof.npz(path)
    actual, diagnostics = proof.check_map(feature, train, state)
    np.testing.assert_allclose(actual, original, atol=1e-13)
    feature[~train] += 1e6
    changed, _ = proof.check_map(feature, train, state)
    np.testing.assert_array_equal(changed[train], actual[train])
    assert diagnostics["feature_mean_difference"] < 1e-14
    bad = {**state, "bandwidth_sample_indices": state["bandwidth_sample_indices"][::-1]}
    with pytest.raises(AssertionError):
        proof.check_map(feature, train, bad)


def test_all_five_direct_equations_and_val_selector_ignore_held_labels():
    rng = np.random.default_rng(163)
    z = rng.normal(size=(28, 512))
    target = rng.normal(size=(28, 17, 2))
    train = np.arange(28) < 16
    val = (np.arange(28) >= 16) & (np.arange(28) < 24)
    room = np.array(["E01"] * 20 + ["E03"] * 8)
    state, _ = fit_readout(z, target, train, val, room)
    grid, residual, difference = proof.check_readout(z, target, train, val, room, state)
    target[~(train | val)] = np.nan
    grid2, residual2, difference2 = proof.check_readout(
        z, target, train, val, room, state
    )
    assert (
        len(grid) == 5
        and grid == grid2
        and residual == residual2
        and difference == difference2
    )
    assert residual < 1e-10 and difference < 1e-10


def test_existing_verification_refused_before_input_access(tmp_path):
    output = tmp_path / "exists.json"
    output.write_text("{}\n")
    with pytest.raises(FileExistsError):
        proof.verify(None, None, None, None, None, output)


def test_sqrt2_raw_coordinate_roundtrip_is_machine_precision_not_bit_identity():
    rng = np.random.default_rng(164)
    coords = rng.normal(size=(2, 8, 19, 9)).astype("float32") * np.float32(10000)
    coords[..., :3] = np.float32(200000)
    x = np.zeros((2, 8, 19, 7, 9), dtype="float32")
    x[:, :, :, 0] = coords
    current, original = proof.decode(x.reshape(2, -1))
    off = current[..., (0, 0, 1), (1, 2, 2)]
    encoded = np.concatenate(
        (
            np.diagonal(current, axis1=-2, axis2=-1).real,
            np.sqrt(2) * off.real,
            np.sqrt(2) * off.imag,
        ),
        axis=-1,
    )
    assert np.max(abs(encoded - original)) > 0
    np.testing.assert_allclose(encoded, original, rtol=4e-15, atol=1e-10)


@pytest.mark.parametrize("bad", [np.nan, np.inf, 1e6])
@pytest.mark.parametrize("shape", [(1, 19, 3, 3), (19,), (4, 17, 2), ()])
def test_full_readback_helper_rejects_nonfinite_or_different_saved_arrays(bad, shape):
    expected = np.ones(shape, dtype=complex if len(shape) == 4 else float)
    observed = np.full_like(expected, bad)
    with pytest.raises(ValueError, match="Numeric readback mismatch"):
        proof.numeric_difference(observed, expected, name="saved artifact", atol=1e-10)
    with pytest.raises(ValueError, match="Numeric readback mismatch"):
        proof.numeric_difference(
            expected.reshape(-1, 1), expected, name="shape", atol=1e-10
        )


@pytest.mark.parametrize(
    "field", ["nje", "pck", "r2", "per_joint_nje", "per_joint_pck"]
)
def test_full_score_comparison_rejects_nonfinite_saved_scalars(field):
    rng = np.random.default_rng(165)
    target = rng.normal(size=(8, 17, 2))
    metric = proof.metric(target + 0.01, target, np.repeat("record", 8))
    expected = dict(
        metrics=dict(
            normalized_joint_error=metric["nje"],
            pck_curve=metric["pck"].copy(),
            valid_frames=8,
            valid_joints=136,
            per_joint=[
                dict(nje=metric["per_joint_nje"][j], pck=metric["per_joint_pck"][j])
                for j in range(17)
            ],
        ),
        record_articulation_r2=metric["r2"],
    )
    if field == "nje":
        expected["metrics"]["normalized_joint_error"] = np.nan
    elif field == "pck":
        expected["metrics"]["pck_curve"]["0.1"] = np.nan
    elif field == "r2":
        expected["record_articulation_r2"] = np.nan
    else:
        expected["metrics"]["per_joint"][3][
            "nje" if field.endswith("nje") else "pck"
        ] = np.nan
    with pytest.raises(ValueError, match="Numeric readback mismatch"):
        proof.compare_score(metric, expected)
