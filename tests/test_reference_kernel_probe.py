"""Synthetic software protocol fixtures, not wireless accuracy evidence."""

import copy

import numpy as np
import pytest

from ghostmesh.metrics import pose_metrics
from scripts.domain_reference_covariance_probe import (
    REF_ACTIONS,
    REF_ENDS,
    reference_indices,
)
from scripts.domain_reference_kernel_probe import (
    CONFIG,
    METHODS,
    ROOMS,
    candidate_prediction,
    inner_roles,
    label_permutation,
    nje,
    prepare,
    select_outer,
    multiply,
    train_inner,
)


def fixture():
    rng = np.random.default_rng(631)
    rows = [
        (room, f"S{person:02}", f"A{action:02}", end)
        for room in ROOMS
        for person in range((int(room[-2:]) - 1) * 10 + 1, int(room[-2:]) * 10 + 1)
        for action in range(1, 28)
        for end in (
            REF_ENDS
            if person % 10 in (1, 2) and f"A{action:02}" in REF_ACTIONS
            else (39,)
        )
    ]
    data = dict(
        room=np.asarray([r[0] for r in rows]),
        person=np.asarray([r[1] for r in rows]),
        action=np.asarray([r[2] for r in rows]),
        end=np.asarray([r[3] for r in rows]),
        record=np.asarray(["/".join(r[:3]) for r in rows]),
    )
    raw = rng.normal(size=(len(rows), 6))
    body = np.stack((np.linspace(-0.3, 0.3, 17), np.linspace(-0.5, 0.1, 17)), axis=1)
    weights = rng.normal(size=(6, 34)) * 0.012
    y = body + multiply(raw, weights).reshape(-1, 17, 2)
    y -= y[:, [11, 12]].mean(1, keepdims=True)
    data["pose"] = y
    data["delta"] = np.zeros_like(y)
    return raw, data


def test_exact_roles_reference_permutation_and_scalar_nje_contract():
    raw, data = fixture()
    for room in ROOMS:
        train, val = inner_roles(data, room)
        refs = reference_indices(data, room)
        assert (
            len(set(data["person"][train])) == 8 and len(set(data["person"][val])) == 2
        )
        assert set(data["room"][train]) == {room} and not set(
            data["person"][train]
        ) & set(data["person"][val])
        p = label_permutation(data, refs)
        np.testing.assert_array_equal(data["record"][refs[p]], data["record"][refs])
        np.testing.assert_array_equal(np.sort(p), np.arange(120))
        np.testing.assert_array_equal(p, label_permutation(data, refs))
    target = data["pose"][:10]
    prediction = target + 0.01
    expected = pose_metrics(prediction, target, np.ones(target.shape[:2], bool))[
        "normalized_joint_error"
    ]
    assert nje(prediction, target) == expected
    with pytest.raises(ValueError):
        nje(prediction[:1], target)


@pytest.mark.parametrize("held", ROOMS)
def test_outer_room_reference_and_query_perturbation_cannot_change_its_choices_or_source_maps(
    held,
):
    raw, data = fixture()
    before = prepare(raw, data)
    mask = data["room"] == held
    changed = raw.copy()
    changed[mask] = changed[mask] * 3 + 100
    changed_data = copy.deepcopy(data)
    changed_data["pose"][mask] += 1000
    after = prepare(changed, changed_data)
    assert before["choices"][held] == after["choices"][held]
    for source in set(ROOMS) - {held}:
        for group in ("mapping", "state"):
            for key in before["priors"][source][group]:
                np.testing.assert_array_equal(
                    before["priors"][source][group][key],
                    after["priors"][source][group][key],
                )
        assert (
            before["priors"][source]["base_grid"]
            == after["priors"][source]["base_grid"]
        )
    # The affected room's own reference kernel must change, proving the poison
    # exercised actual enrollment/query features rather than a dummy selector.
    assert not np.array_equal(
        before["kernels"][held].input_mean, after["kernels"][held].input_mean
    )


def test_saved_and_pure_inner_maps_are_identical(tmp_path):
    raw, data = fixture()
    written = train_inner(raw, data, "E01", tmp_path)
    rebuilt = train_inner(raw, data, "E01")
    for group in ("mapping", "state"):
        for key in written[group]:
            np.testing.assert_array_equal(written[group][key], rebuilt[group][key])
    np.testing.assert_array_equal(written["prediction"], rebuilt["prediction"])
    assert written["base_grid"] == rebuilt["base_grid"]


def test_selector_rejects_missing_direction_and_does_not_pool_held_room_cases():
    grid = {
        method: [
            dict(alpha=alpha, beta=beta, source_validation_nje=float(i))
            for i, (alpha, beta) in enumerate(
                (a, b)
                for a in (CONFIG["alpha"] if method.endswith("krr") else [None])
                for b in CONFIG["beta"]
            )
        ]
        for method in METHODS
    }
    cases = [
        dict(source=a, target=b, curves=copy.deepcopy(grid))
        for a in ROOMS
        for b in ROOMS
        if a != b
    ]
    selected = select_outer(cases, "E01")
    for case in cases:
        if "E01" in (case["source"], case["target"]):
            for method in METHODS:
                for row in case["curves"][method]:
                    row["source_validation_nje"] = -1000 - row["source_validation_nje"]
    assert select_outer(cases, "E01") == selected
    with pytest.raises(ValueError):
        select_outer(
            [c for c in cases if not (c["source"] == "E02" and c["target"] == "E03")],
            "E01",
        )


def test_no_clipping_or_hidden_mean_in_residual_and_batch_constant_controls():
    mean = np.ones((17, 2)) * 0.2
    base = np.ones((5, 17, 2)) * (-2)
    state = dict(
        residual_krr=np.ones((5, 3, 34)),
        absolute_krr=np.ones((5, 3, 34)),
        permuted_residual_krr=np.ones((5, 3, 34)),
        constant_residual=np.ones(34) * 0.3,
        mean_mixture=np.ones(34) * 0.3,
    )
    zero = np.zeros((5, 3))
    prediction = candidate_prediction(
        "residual_krr", dict(alpha=0.001, beta=1), state, zero, base, mean
    )
    np.testing.assert_array_equal(prediction, base)
    mixed = candidate_prediction(
        "mean_mixture", dict(alpha=None, beta=1), state, zero, base, mean
    )
    assert mixed.shape == (5, 17, 2)
    np.testing.assert_array_equal(mixed, np.ones_like(mixed) * 0.5)
