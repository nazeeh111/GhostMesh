"""Leakage, raw-history and full-dimensional transform contracts."""

import numpy as np
import pytest

from scripts.domain_covariance_source_probe import (
    CONFIG,
    available_examples,
    examples,
    feature_stats,
    fold_masks,
    rff_transform,
    selected_records,
    validate_records,
)


def records():
    return [
        dict(id=f"E{r:02}/S{p:02}/A{a:02}", frames=297, offset=i * 297)
        for i, (r, p, a) in enumerate(
            (r, p, a)
            for r in range(1, 4)
            for p in range((r - 1) * 10 + 1, r * 10 + 1)
            for a in range(1, 28)
        )
    ]


def test_only_complete_ordered_source30_and_disjoint_people_allowed():
    rows = records()
    validate_records(rows)
    for bad in (rows[:-1], rows[::-1], [dict(id="E04/S31/A01", frames=297, offset=0)]):
        with pytest.raises(ValueError):
            validate_records(bad)
    room = np.repeat([r["id"][:3] for r in rows], 33)
    person = np.repeat([r["id"].split("/")[1] for r in rows], 33)
    for held_room in CONFIG["folds"]:
        train, val, held = fold_masks(room, person, held_room)
        assert len(set(person[train])) == 16
        assert len(set(person[val])) == 4
        assert len(set(person[held])) == 10
        assert not set(person[train]) & set(person[val])
        assert not set(person[train]) & set(person[held])
    subset = selected_records(rows, limited_actions=True)
    assert len(subset) == 90
    assert {r["id"].split("/")[2] for r in subset} == {"A03", "A07", "A12"}
    assert len({r["id"].split("/")[1] for r in subset}) == 30
    assert subset[0]["offset"] == 2 * 297
    room = np.repeat([r["id"][:3] for r in subset], 33)
    person = np.repeat([r["id"].split("/")[1] for r in subset], 33)
    train, val, held = fold_masks(
        room, person, "E02", dict(train=1584, validation=396, held=990)
    )
    assert (train.sum(), val.sum(), held.sum()) == (1584, 396, 990)


def test_full_raw_dimensions_and_causal_prefix_identity():
    rng = np.random.default_rng(7)
    rf = rng.normal(size=(297, 6, 114, 10))
    rf[:, :3] += 40
    first, _ = examples(rf)
    rf[100:] += 2
    second, _ = examples(rf)
    for name, dim in CONFIG["raw_dimensions"].items():
        assert first[name].shape == (33, dim)
        np.testing.assert_array_equal(first[name][:8], second[name][:8])


def test_training_only_feature_and_rff_normalizers(tmp_path):
    rng = np.random.default_rng(8)
    x = rng.normal(size=(256, 15))
    train = np.arange(256) < 192
    before = feature_stats(x, train)
    first = rff_transform(x, train, tmp_path / "first.npz")
    x[~train] = 10000
    after = feature_stats(x, train)
    second = rff_transform(x, train, tmp_path / "second.npz")
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(first[train], second[train])
    with (
        np.load(tmp_path / "first.npz", allow_pickle=False) as a,
        np.load(tmp_path / "second.npz", allow_pickle=False) as b,
    ):
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
        assert a["weight"].shape == (15, 512)
        assert set(a["bandwidth_sample_indices"]) <= set(np.flatnonzero(train))


def test_rejected_frame_invalidates_every_history_without_filled_covariances():
    rng = np.random.default_rng(10)
    rf = rng.normal(size=(297, 6, 114, 10))
    rf[:, :3] += 40
    full, _ = examples(rf)
    rf[80, 0, 0, :4] = np.nan
    values, retained, _, rejected = available_examples(rf)
    assert len(rejected) == 1 and rejected[0]["frame"] == 80
    expected = [end for end in CONFIG["endpoints"] if not end - 39 <= 80 <= end]
    np.testing.assert_array_equal(retained, expected)
    keep = np.isin(CONFIG["endpoints"], retained)
    for arm in full:
        np.testing.assert_array_equal(values[arm], full[arm][keep])
