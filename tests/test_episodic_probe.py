"""Source/held role boundaries, episode design, loss and reference-only fitting."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import domain_episodic_probe as e


def config():
    return json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs/episodic-reference-e03-v1.json"
        ).read_text()
    )


def data_fixture():
    rows = []
    for person in range(1, 11):
        for action in range(1, 28):
            for end in range(39, 80, 8):
                rows.append(
                    (
                        "E01",
                        f"S{person:02}",
                        f"A{action:02}",
                        f"E01/S{person:02}/A{action:02}",
                        end,
                    )
                )
    rows += [("E03", "S21", "A03", "E03/S21/A03", 39)]
    data = {
        key: np.array([row[i] for row in rows])
        for i, key in enumerate(("room", "person", "action", "record", "end"))
    }
    data["validation"] = (data["room"] == "E01") & np.isin(
        data["person"], ["S04", "S09"]
    )
    data["train"] = (data["room"] == "E01") & ~data["validation"]
    data["pose"] = (
        np.random.default_rng(4).normal(size=(len(rows), 17, 2)).astype(np.float32)
    )
    data["pose"][data["room"] == "E03"] = np.nan
    return data


def test_episodes_match_roles_people_record_time_and_are_reproducible():
    data = data_fixture()
    episodes, perms = e.make_episodes(data, 3, 2027, 90210)
    other, op = e.make_episodes(data, 3, 2027, 90210)
    np.testing.assert_array_equal(episodes, other)
    np.testing.assert_array_equal(perms, op)
    assert episodes.shape == (3, 48) and data["train"][episodes].all()
    for rows, perm in zip(episodes, perms):
        assert (
            len(set(data["person"][rows[:24]])) == 2
            and len(set(data["person"][rows[24:]])) == 2
        )
        assert not set(data["person"][rows[:24]]) & set(data["person"][rows[24:]])
        np.testing.assert_array_equal(
            data["record"][rows[perm]], data["record"][rows[:24]]
        )
        assert len(e.pair_rows(data["record"][rows], data["end"][rows])) == 34
        assert len(e.pair_rows(data["record"][rows[24:]], data["end"][rows[24:]])) == 16


def test_label_and_rf_guards_refuse_validation_and_held_before_access():
    data = data_fixture()
    with pytest.raises(ValueError, match="role"):
        e.training_labels(data, np.flatnonzero(data["validation"])[:2])
    with pytest.raises(ValueError, match="role"):
        e.training_labels(data, np.flatnonzero(data["room"] == "E03"))
    reader = e.SourceReader(
        np.arange(len(data["train"]))[:, None],
        np.arange(len(data["train"])),
        data["train"],
    )
    with pytest.raises(ValueError, match="role"):
        reader(np.array([-1]))
    with pytest.raises(ValueError, match="role"):
        reader(np.flatnonzero(data["validation"])[:2])


def test_statistics_never_use_validation_or_held_pose_values():
    data = data_fixture()
    cfg = config()

    def reader(ids):
        e.checked_rows(ids, data["train"])
        return np.broadcast_to(
            (np.asarray(ids) % 13)[:, None, None, None] / 13, (len(ids), 8, 12, 114)
        ).astype(np.float32)

    before = e.statistics(data, reader, cfg)
    data["pose"][~data["train"]] = np.nan
    after = e.statistics(data, reader, cfg)
    for key, value in before.items():
        np.testing.assert_array_equal(value, after[key])


def test_non_source_optimizer_rows_and_unrecoverable_budget_are_rejected():
    data = data_fixture()
    data["train"][-1] = True
    with pytest.raises(ValueError, match="E01-only"):
        e.make_episodes(data, 2, 2027, 90210)
    cfg = config()
    cfg["steps"] = 10000
    with pytest.raises(ValueError, match="recovery"):
        e.validate_config(cfg)


def test_source_metadata_cannot_point_at_another_rooms_annotation_frames():
    d = dict(
        room=np.array(["E01"]),
        person=np.array(["S01"]),
        action=np.array(["A01"]),
        record=np.array(["E01/S01/A01"]),
        end=np.array([39]),
        raw_index=np.array([39]),
    )
    records = [dict(id="E01/S01/A01", offset=0), dict(id="E03/S21/A01", offset=160380)]
    e.validate_raw_indices(d, records)
    d["raw_index"][0] = 160419
    with pytest.raises(ValueError, match="frame join"):
        e.validate_raw_indices(d, records)


def test_episode_loss_matches_numpy_equation_and_preserves_time_reversal():
    rng = np.random.default_rng(2)
    p = rng.normal(size=(6, 17, 2))
    y = rng.normal(size=p.shape)
    a = rng.uniform(0.1, 1, (17, 2))
    d = rng.uniform(0.1, 1, (17, 2))
    pairs = np.array([[1, 0], [2, 1], [4, 3], [5, 4]])
    expected = (
        0.5 * np.square((p - y) / a).mean()
        + 0.5
        * np.square(
            ((p[pairs[:, 0]] - p[pairs[:, 1]]) - (y[pairs[:, 0]] - y[pairs[:, 1]])) / d
        ).mean()
    )
    args = [torch.tensor(x) for x in [p, y, pairs, a, d]]
    assert float(e.pose_loss(*args)) == pytest.approx(expected, abs=1e-12)
    args[2] = args[2].flip(1)
    assert float(e.pose_loss(*args)) == pytest.approx(expected, abs=1e-12)


def test_reference_kernel_state_has_no_objects_and_bandwidth_uses_only_references(
    tmp_path,
):
    rng = np.random.default_rng(8)
    x = rng.normal(size=(10, 4))
    q = rng.normal(size=(3, 4))
    y = rng.normal(size=(10, 17, 2))
    _, width = e.raw_kernel(x, q, "rbf")
    _, other = e.raw_kernel(x, q * 100, "rbf")
    assert width == other
    for kind in ["linear", "rbf"]:
        pred, state = e.raw_prediction(x, y, q, kind, 0.1)
        assert pred.shape == (3, 17, 2) and np.isfinite(pred).all()
        path = tmp_path / (kind + ".npz")
        np.savez(path, **state)
        with np.load(path, allow_pickle=False) as z:
            for key in z.files:
                assert z[key].dtype != object


def test_source_reference_selection_is_unique_train_only_and_twenty_per_record():
    # Replace the small sampling fixture with twenty-five endpoints per ref record.
    rows = []
    for person in ["S01", "S02"]:
        for action in ["A03", "A07", "A12"]:
            rows.extend((person, action, end) for end in range(39, 239, 8))
    refs = {
        key: np.array([r[i] for r in rows])
        for i, key in enumerate(("person", "action", "end"))
    }
    refs["train"] = np.ones(len(rows), bool)
    selected = e.references(refs, config())
    assert len(set(selected)) == 120 and refs["train"][selected].all()
    for person in ["S01", "S02"]:
        for action in ["A03", "A07", "A12"]:
            assert (
                (refs["person"][selected] == person)
                & (refs["action"][selected] == action)
            ).sum() == 20
