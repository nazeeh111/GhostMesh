"""Software-only protocol and label-use checks; no dataset/training runs."""

import numpy as np
import pytest
import torch
from torch import nn
from scripts.domain_meta_probe import (
    CONFIG,
    ENDS,
    episodes,
    index_records,
    loss,
    references,
    run,
    source_stats,
    windows,
)


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


def test_window_endpoints_never_cross_record():
    rows = records()
    ix = index_records(rows)
    assert len(ix) == 29970
    assert set(ix["end"]) == set(range(7, 297, 8))
    small = [
        dict(id="E01/S01/A01", frames=297, offset=0),
        dict(id="E01/S01/A02", frames=297, offset=297),
    ]
    frame = np.arange(594)[:, None]
    actual = windows(frame, small, np.array([[0, 7], [1, 7], [1, 295]]))
    np.testing.assert_array_equal(
        actual[:, :, 0], [[*range(8)], [*range(297, 305)], [*range(585, 593)]]
    )
    with pytest.raises(ValueError):
        index_records([dict(id="E04/S31/A01", frames=297, offset=0)])


def test_references_exact_budget_and_episode_people_disjoint():
    rows = records()
    ix = index_records(rows)
    refs = references(rows, "E02", ["S11", "S12"])
    assert refs.shape == (120, 2)
    assert set(refs[:, 1]) == set(ENDS)
    first, replay = list(episodes(rows, ix, 10)), list(episodes(rows, ix, 10))
    for (s, q), (ss, qq) in zip(first, replay):
        np.testing.assert_array_equal(s, ss)
        np.testing.assert_array_equal(q, qq)
        people = {rows[r]["id"].split("/")[1] for r in s[:, 0]}
        assert len(people) == 2
        assert not people.intersection(ix["person"][q])
        assert not set(ix["person"][q]).intersection({"S04", "S09", "S24", "S29"})
        assert set(ix["room"][q]) <= {"E01", "E03"}
        assert len(q) == CONFIG["query"]


def test_stats_ignore_source_validation_and_target_labels():
    rows = [
        dict(id=f"{room}/{person}/A01", frames=297, offset=i * 297)
        for i, (room, person) in enumerate(
            [("E01", "S01"), ("E01", "S04"), ("E02", "S11")]
        )
    ]
    rng = np.random.default_rng(4)
    frame = rng.normal(size=(891, 12, 114)).astype("float32")
    pose = rng.normal(size=(891, 17, 2)).astype("float32")
    before = source_stats(frame, pose, rows)
    frame[297:] = 10000
    pose[297:] = 10000
    after = source_stats(frame, pose, rows)
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)
    assert np.all(before[1] >= 0.01) and np.all(before[3] >= 0.01)


def test_loss_exact_budgets_and_meta_support_gradients():
    class Small(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(3, 32)
            self.prior = nn.Linear(32, 36)

        def encode(self, x):
            return self.encoder(x)

    torch.manual_seed(2027)
    model = Small()
    x, y = torch.randn(248, 3), torch.randn(248, 36)
    ordinary = loss(model, x, y, "erm")
    torch.testing.assert_close(
        ordinary, ((model.prior(model.encode(x)) - y) ** 2).mean()
    )
    x.requires_grad_()
    meta = loss(model, x, y, "meta")
    meta.backward()
    assert torch.isfinite(x.grad).all() and x.grad[:120].abs().sum() > 0
    assert x.grad[120:].abs().sum() > 0
    with pytest.raises(ValueError):
        loss(model, x, y, "other")


def test_existing_output_refused_before_any_cache_read(tmp_path):
    with pytest.raises(FileExistsError):
        run(tmp_path / "absent", tmp_path / "absent", tmp_path, "cpu")
