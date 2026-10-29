import numpy as np
import pytest

from ghostmesh.covariance_dynamics import covariance_features, hermitian_coordinates
from ghostmesh.reference_covariance import (
    current_covariances,
    decode_hermitian_coordinates,
    fit_reference,
    reference_features,
)
from scripts.domain_reference_covariance_probe import (
    REF_ACTIONS,
    REF_ENDS,
    build_room_references,
    feature_block,
    fit_readout,
    frozen_gate,
    preflight,
    query_indices,
    reference_indices,
    reference_people,
)


def spd(rng, shape):
    x = rng.normal(size=(*shape, 3, 3)) + 1j * rng.normal(size=(*shape, 3, 3))
    return np.einsum("...ij,...kj->...ik", x, x.conj()) + np.eye(3)


def transform(covariance, gain, g):
    return (
        np.einsum("bij,...bjk,blk->...bil", g, covariance, g.conj())
        * gain[..., None, None]
    )


def test_exact_shared_complex_mixing_and_independent_frame_gain_invariance():
    rng = np.random.default_rng(141)
    calibration, query = spd(rng, (12, 19)), spd(rng, (7, 19))
    groups = np.repeat(REF_ACTIONS, 4)
    g = np.tile(np.diag([1.0, 2.0, 0.7]), (19, 1, 1)).astype("complex128")
    g += 0.1 * (rng.normal(size=g.shape) + 1j * rng.normal(size=g.shape))
    ref_gain = rng.uniform(0.2, 6, (12, 19))
    query_gain = rng.uniform(0.2, 6, (7, 19))
    a = fit_reference(calibration, groups, group_order=REF_ACTIONS)
    b = fit_reference(
        transform(calibration, ref_gain, g), groups, group_order=REF_ACTIONS
    )
    before = reference_features(query, a)
    after = reference_features(transform(query, query_gain, g), b)
    np.testing.assert_allclose(before, after, atol=3e-11, rtol=3e-11)
    full_before = reference_features(query, a, include_volume=True)
    full_after = reference_features(
        transform(query, query_gain, g), b, include_volume=True
    )
    expected = np.log(query_gain) - np.log(ref_gain).mean(0)
    np.testing.assert_allclose(
        full_after[..., -1] - full_before[..., -1], expected, atol=3e-12
    )
    fixed = fit_reference(
        transform(calibration, np.ones((12, 19)), g), groups, group_order=REF_ACTIONS
    )
    np.testing.assert_allclose(
        reference_features(
            transform(query, np.ones((7, 19)), g), fixed, include_volume=True
        ),
        full_before,
        atol=3e-11,
    )


def test_current_encoding_roundtrip_and_correct_state_extraction():
    rng = np.random.default_rng(12)
    current = spd(rng, (2, 8, 19))
    coordinates = hermitian_coordinates(current)
    np.testing.assert_allclose(
        decode_hermitian_coordinates(coordinates), current, atol=3e-15
    )
    full = np.repeat(coordinates[:, :, :, None, :], 7, axis=3)
    full[:, :, :, 1:] = 100
    np.testing.assert_allclose(
        current_covariances(full.reshape(2, -1)), current, atol=3e-15
    )
    with pytest.raises(ValueError):
        decode_hermitian_coordinates(np.full((19, 9), np.nan))


def test_fixed_reference_retains_static_configuration_that_temporal_pairs_lose():
    reference = fit_reference(np.tile(np.eye(3), (6, 19, 1, 1)))
    a = np.tile(np.diag([1.0, 2.0, 5.0]), (45, 19, 1, 1))
    b = np.tile(np.diag([1.0, 3.0, 7.0]), (45, 19, 1, 1))
    np.testing.assert_allclose(
        covariance_features(a)["congruence_dynamics"], 0, atol=1e-14
    )
    np.testing.assert_allclose(
        covariance_features(b)["congruence_dynamics"], 0, atol=1e-14
    )
    assert (
        np.linalg.norm(
            reference_features(a, reference) - reference_features(b, reference)
        )
        > 0.1
    )
    assert np.linalg.norm(reference_features(a, reference)) > 0.1


def test_semantic_anchors_can_distinguish_orientation_lost_by_single_mean():
    diagonals = np.array([[1.0, 2.0, 0.5], [0.5, 1.0, 2.0], [2.0, 0.5, 1.0]])
    calibration = np.stack(
        [np.tile(np.diag(row), (19, 1, 1)) for row in diagonals for _ in range(2)]
    )
    groups = np.repeat(REF_ACTIONS, 2)
    single = fit_reference(calibration)
    multi = fit_reference(calibration, groups, group_order=REF_ACTIONS)
    a = np.tile(np.diag([1.0, 3.0, 0.5]), (1, 19, 1, 1))
    b = a[:, :, [1, 0, 2]][:, :, :, [1, 0, 2]]
    np.testing.assert_allclose(
        reference_features(a, single), reference_features(b, single), atol=1e-14
    )
    assert (
        np.linalg.norm(reference_features(a, multi) - reference_features(b, multi))
        > 0.1
    )


def test_fixed_reference_prefix_causality_and_immutable_enrollment():
    rng = np.random.default_rng(2)
    reference = fit_reference(spd(rng, (9, 19)))
    query = spd(rng, (12, 8, 19))
    before = reference_features(query, reference, include_volume=True)
    query[7:] *= 100
    np.testing.assert_array_equal(
        reference_features(query[:7], reference, include_volume=True), before[:7]
    )
    with pytest.raises(ValueError):
        reference.anchors[0, 0, 0, 0] = 10


def metadata():
    rows = [
        (room, person, action, end)
        for room in ("E01", "E02", "E03")
        for person in reference_people(room)
        for action in REF_ACTIONS
        for end in REF_ENDS
    ]
    return dict(
        room=np.asarray([r[0] for r in rows]),
        person=np.asarray([r[1] for r in rows]),
        action=np.asarray([r[2] for r in rows]),
        end=np.asarray([r[3] for r in rows]),
        record=np.asarray(["/".join(r[:3]) for r in rows]),
        pose=np.zeros((len(rows), 17, 2)),
        delta=np.zeros((len(rows), 17, 2)),
    )


def test_exact_person_disjoint_reference_budget_and_gt_free_operators():
    data = metadata()
    rng = np.random.default_rng(4)
    current = spd(rng, (len(data["pose"]), 8, 19))
    coordinates = hermitian_coordinates(current)
    full = np.repeat(coordinates[:, :, :, None, :], 7, axis=3).reshape(len(current), -1)
    before = build_room_references(full, data)
    data["pose"][:] = 1e6
    data["delta"][:] = -1e6
    after = build_room_references(full, data)
    for room in ("E01", "E02", "E03"):
        indices = reference_indices(data, room)
        assert len(indices) == 120 and len(set(data["person"][indices])) == 2
        assert tuple(before[room]["semantic_reference"].group_counts) == (40, 40, 40)
        for kind in ("global_reference", "semantic_reference"):
            np.testing.assert_array_equal(
                before[room][kind].anchors, after[room][kind].anchors
            )
    bad = dict(data)
    bad["end"] = data["end"].copy()
    bad["end"][0] = -1
    with pytest.raises(ValueError, match="120"):
        reference_indices(bad, "E01")
    with pytest.raises(ValueError):
        reference_people("E04")


def test_held_query_labels_cannot_change_source_readout_or_selection():
    rng = np.random.default_rng(6)
    z = rng.normal(size=(80, 7))
    y = rng.normal(size=(80, 17, 2))
    train = np.arange(80) < 40
    validation = (np.arange(80) >= 40) & (np.arange(80) < 60)
    held = np.arange(80) >= 60
    rooms = np.repeat(["E01", "E03"], 40)
    before, grid = fit_readout(z, y, train, validation, rooms)
    y[held] = 1e6
    after, grid2 = fit_readout(z, y, train, validation, rooms)
    assert grid == grid2
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])


def test_reference_endpoints_and_eight_query_people_are_disjoint():
    data = metadata()
    extra = [
        (room, f"S{i:02}", "A01", 39)
        for room in ("E01", "E02", "E03")
        for i in range(
            int(reference_people(room)[1][1:]) + 1,
            int(reference_people(room)[1][1:]) + 9,
        )
    ]
    for key, position in (("room", 0), ("person", 1), ("action", 2), ("end", 3)):
        data[key] = np.concatenate(
            (data[key], np.asarray([r[position] for r in extra]))
        )
    data["record"] = np.concatenate(
        (data["record"], np.asarray(["/".join(r[:3]) for r in extra]))
    )
    for room in ("E01", "E02", "E03"):
        reference = reference_indices(data, room)
        query = query_indices(data, room)
        assert len(reference) == 120 and len(query) == 8
        assert not set(data["person"][reference]) & set(data["person"][query])
    incomplete = {
        key: value[:-1] for key, value in data.items() if key not in ("pose", "delta")
    }
    with pytest.raises(ValueError, match="eight"):
        query_indices(incomplete, "E03")


def test_raw_current_comparator_does_not_depend_on_room_reference():
    rng = np.random.default_rng(18)
    a = spd(rng, (6, 19))
    b = spd(rng, (6, 19))
    query = spd(rng, (2, 8, 19))
    groups = np.repeat(REF_ACTIONS, 2)
    refs = [
        dict(
            global_reference=fit_reference(cal),
            semantic_reference=fit_reference(cal, groups, group_order=REF_ACTIONS),
        )
        for cal in (a, b)
    ]
    first, second = (feature_block(query, ref) for ref in refs)
    np.testing.assert_array_equal(
        first["raw_current_control"], second["raw_current_control"]
    )
    assert np.max(abs(first["semantic3_shape"] - second["semantic3_shape"])) > 0.01


def test_non_spd_decode_is_rejected_without_identity_repair():
    with pytest.raises(ValueError, match="not SPD"):
        preflight(np.zeros((1, 9576)), {})
    with pytest.raises(ValueError, match="Singular"):
        fit_reference(np.tile(np.diag([1.0, 2.0, -1.0]), (2, 19, 1, 1)))


def test_gate_keeps_fixed_primary_and_both_pck_comparators():
    methods = (
        "semantic3_shape",
        "source_mean",
        "reference120_mean",
        "source_validation_mean_mixture",
        "frozen_temporal_zero_shot",
        "raw_current_control",
        "global_shape",
    )
    rows = []
    for room in ("E01", "E02", "E03"):
        for method in methods:
            rows.append(
                dict(
                    room=room,
                    method=method,
                    aligned=dict(
                        normalized_joint_error=0.1
                        if method == "semantic3_shape"
                        else 0.2,
                        pck_curve={"0.05": 0.7, "0.2": 0.95},
                        record_centered_articulation_r2=0.1,
                    ),
                    shuffled=dict(normalized_joint_error=0.3),
                )
            )
    assert frozen_gate(rows)["passed"]
    primary = next(
        r for r in rows if r["room"] == "E02" and r["method"] == "semantic3_shape"
    )
    primary["aligned"]["normalized_joint_error"] = 0.4
    next(r for r in rows if r["room"] == "E02" and r["method"] == "global_shape")[
        "aligned"
    ]["normalized_joint_error"] = 0
    assert not frozen_gate(rows)["passed"]
    primary["aligned"]["normalized_joint_error"] = 0.1
    next(
        r for r in rows if r["room"] == "E02" and r["method"] == "raw_current_control"
    )["aligned"]["pck_curve"]["0.05"] = 0.9
    assert not frozen_gate(rows)["passed"]
