import itertools

import numpy as np
import pytest

from scripts.domain_covariance_neural_probe import (
    CONFIG,
    label_coverage,
    observations,
    run,
    schedule,
    shuffle_indices,
    source_stats,
)


def test_every_bin_lag_eigenvalue_survives_channel_layout_without_projection():
    x = np.arange(2736 * 2, dtype=np.float32).reshape(2, -1)
    actual = observations(x, np.array([0, 1]))
    assert actual.shape == (2, 8, 18, 19)
    for t, bin, lag, eigen in [(0, 0, 0, 0), (7, 18, 5, 2), (4, 11, 3, 1)]:
        for row in range(2):
            assert (
                actual[row, t, lag * 3 + eigen, bin]
                == x[row, t * 342 + bin * 18 + lag * 3 + eigen]
            )
    with pytest.raises(ValueError):
        observations(x[:, :-1], np.array([0]))


def test_source_statistics_ignore_all_validation_held_features_and_labels():
    rng = np.random.default_rng(6)
    feature = rng.normal(size=(20, 2736)).astype(np.float32)
    meta = dict(
        pose=rng.normal(size=(20, 17, 2)), delta=rng.normal(size=(20, 17, 2)) * 0.01
    )
    train = np.arange(20) < 10
    before = source_stats(feature, meta, train)
    feature[~train] += 1e6
    meta["pose"][~train] += 1e9
    meta["delta"][~train] -= 1e9
    after = source_stats(feature, meta, train)
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)
    assert (
        (before[1] >= 0.01).all()
        and (before[3] >= 0.01).all()
        and (before[4] >= 0.001).all()
    )


def test_reproducible_matched_schedule_uses_source_training_only():
    meta = dict(room=np.repeat(["E01", "E02", "E03"], 300))
    train = meta["room"] != "E02"
    train[200:300] = False
    train[800:900] = False
    a = list(itertools.islice(schedule(meta, train), 4))
    b = list(itertools.islice(schedule(meta, train), 4))
    np.testing.assert_array_equal(a, b)
    for ids in a:
        assert len(ids) == CONFIG["batch"] == 128 and train[ids].all()
        assert len(set(ids)) == 128 and len(set(meta["room"][ids])) == 1
    assert CONFIG["steps"] * CONFIG["batch"] * 2 == 256000


def test_unique_endpoint_budget_and_shuffle_preserve_record_membership():
    meta = dict(record=np.array(["r1", "r1", "r2"]), end=np.array([39, 47, 55]))
    count = label_coverage(meta, np.ones(3, bool), np.array([[0, 1, 2]]))
    assert count["train_pose_stat_occurrences"] == 6
    assert count["train_pose_stat_unique_endpoints"] == 5
    assert count["scheduled_unique_endpoint_labels"] == 5
    groups = np.repeat(["a", "b", "c"], 10)
    perm = shuffle_indices(groups)
    np.testing.assert_array_equal(groups[perm], groups)
    np.testing.assert_array_equal(np.sort(perm), np.arange(30))
    np.testing.assert_array_equal(perm, shuffle_indices(groups))


def test_existing_output_refused_before_loading_cache(tmp_path):
    with pytest.raises(FileExistsError):
        run(tmp_path / "absent", tmp_path / "absent", tmp_path, "cpu")
