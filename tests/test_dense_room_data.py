import numpy as np
import pytest

from scripts import dense_room_data as d


def test_dense_history_view_obeys_delayed_rf_clock_and_rejects_bad_indices():
    moments = np.broadcast_to(np.arange(300)[:, None, None], (300, 12, 114)).copy()
    view = d.HistoryView(moments, np.array([39, 40, 287]))
    x = view[np.array([0, 2])]
    assert x.shape == (2, 32, 12, 114)
    assert x[0, 0, 0, 0] == 12 and x[1, -1, 0, 0] == 291
    with pytest.raises(ValueError):
        view[np.array([0.0])]
    with pytest.raises(ValueError):
        d.HistoryView(moments, np.array([296]))


def test_weighted_dense_normalization_matches_explicit_train_histories(monkeypatch):
    rng = np.random.default_rng(3)
    moments = rng.normal(size=(300, 12, 114)).astype(np.float32)
    centers = np.r_[np.arange(39, 71), np.arange(200, 208)]
    feature = d.HistoryView(moments, centers)
    train = np.arange(40) < 32
    data = dict(
        train=train,
        cache_center=centers,
        record=np.array(["train"] * 32 + ["validation"] * 8),
        end=centers,
        pose=rng.normal(size=(40, 17, 2)).astype(np.float32),
    )
    monkeypatch.setattr(
        d.base,
        "prototypes",
        lambda target, scale, seed: (
            np.zeros((128, 17, 2), np.float32),
            np.full(128, 1 / 128, np.float32),
            np.zeros(len(target), np.int64),
        ),
    )
    stats = d.statistics(
        data, feature, dict(seed=1, pose_scale_floor=0.01, delta_scale_floor=0.01)
    )
    explicit = feature[np.flatnonzero(train)].astype(float).reshape(-1, 12, 114)
    np.testing.assert_allclose(stats["input_mean"], explicit.mean(0), atol=1e-7)
    np.testing.assert_allclose(
        stats["input_scale"], np.maximum(explicit.std(0), 0.05), atol=1e-7
    )
    moments[180:240] += 1000
    after = d.statistics(
        data, feature, dict(seed=1, pose_scale_floor=0.01, delta_scale_floor=0.01)
    )
    np.testing.assert_array_equal(stats["input_mean"], after["input_mean"])
    np.testing.assert_array_equal(stats["input_scale"], after["input_scale"])
