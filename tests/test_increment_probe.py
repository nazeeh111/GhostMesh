"""Protocol and objective checks that do not consume any MM-Fi examples."""

import numpy as np
import pytest
import torch

from scripts.domain_increment_probe import (
    component_scores,
    index_pairs,
    initial_stats,
    pair_ids,
    previous,
    roles,
    run,
    schedule,
    training_loss,
)
from scripts.domain_meta_probe import windows


def records():
    return [
        dict(id=f"E{room:02}/S{person:02}/A{action:02}", frames=297, offset=i * 297)
        for i, (room, person, action) in enumerate(
            (r, s, a)
            for r in range(1, 4)
            for s in range((r - 1) * 10 + 1, r * 10 + 1)
            for a in range(1, 28)
        )
    ]


def test_both_pair_windows_stay_within_record_and_are_disjoint():
    rows = records()
    index = index_pairs(rows)
    assert len(index) == 29160
    assert set(index["end"]) == set(range(15, 297, 8))
    pairs = pair_ids(index)
    old = previous(pairs)
    for now, past in zip(pairs, old):
        assert now[0] == past[0]
        assert past[1] - 7 >= 0 and now[1] < 297
        assert not set(range(now[1] - 7, now[1] + 1)) & set(
            range(past[1] - 7, past[1] + 1)
        )
    frame = np.arange(594)[:, None]
    np.testing.assert_array_equal(
        windows(frame, rows[:2], np.array([[1, 15]]))[:, :, 0], [range(305, 313)]
    )
    with pytest.raises(ValueError):
        previous(np.array([[0, 7]]))
    with pytest.raises(ValueError):
        index_pairs([dict(id="E04/S31/A01", frames=297, offset=0)])


def test_schedule_same_no_validation_or_held_people():
    ix = index_pairs(records())
    train, val, held = roles(ix)
    assert tuple(int(m.sum()) for m in (train, val, held)) == (15552, 3888, 9720)
    a, b = list(schedule(ix)), list(schedule(ix))
    np.testing.assert_array_equal(a, b)
    assert np.all(train[np.array(a)])
    for ids in a:
        assert len(set(ids)) == 128
        assert len(set(ix["room"][ids])) == 1


def test_increment_loss_has_no_static_offset_shortcut_and_zero_head_not_solution():
    torch.manual_seed(4)
    pred, truth = torch.randn(16, 36, requires_grad=True), torch.randn(16, 36)
    value = training_loss(pred, truth, "increment")
    static_offset = torch.randn(8, 36)
    shifts = torch.cat((static_offset, static_offset))
    torch.testing.assert_close(value, training_loss(pred + shifts, truth, "increment"))
    torch.testing.assert_close(value, training_loss(pred, truth + shifts, "increment"))
    zero = torch.zeros_like(pred, requires_grad=True)
    training_loss(zero, truth, "increment").backward()
    assert zero.grad.abs().sum() > 0
    assert training_loss(pred + shifts, truth, "absolute") != training_loss(
        pred, truth, "absolute"
    )
    with pytest.raises(ValueError):
        training_loss(pred, truth, "other")


def test_zero_increment_control_and_misalignment():
    rng = np.random.default_rng(12)
    y = rng.normal(size=(100, 36))
    ids = np.repeat([0, 1], 50)
    perfect = component_scores(y, y, ids)
    zero = component_scores(np.zeros_like(y), y, ids)
    bad = component_scores(np.roll(y, 40, axis=0), y, ids)
    for key in perfect:
        assert perfect[key]["zero_reference_r2"] == 1
        assert zero[key]["zero_reference_r2"] == 0
        assert zero[key]["record_centered_r2"] == 0
        assert bad[key]["zero_reference_r2"] < 0


def test_stats_do_not_fit_validation_or_target():
    rows = [
        dict(id=f"{r}/{s}/A01", frames=297, offset=i * 297)
        for i, (r, s) in enumerate([("E01", "S01"), ("E01", "S04"), ("E02", "S11")])
    ]
    rng = np.random.default_rng(3)
    frame = rng.normal(size=(891, 12, 114)).astype("float32")
    pose = rng.normal(size=(891, 17, 2)).astype("float32")
    ix = index_pairs(rows)
    before = initial_stats(frame, pose, rows, ix)
    frame[297:] = 1e5
    pose[297:] = 1e5
    after = initial_stats(frame, pose, rows, ix)
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)
    assert (before[-1] >= 0.001).all()
    with pytest.raises(FileExistsError):
        run(None, None, __import__("pathlib").Path(__file__).parent, "cpu")
