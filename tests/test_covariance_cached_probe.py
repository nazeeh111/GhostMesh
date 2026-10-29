import numpy as np
import pytest

from scripts.domain_covariance_cached_probe import run, select_ridge


def test_held_labels_cannot_change_source_fit_or_validation_selection():
    rng = np.random.default_rng(71)
    z = rng.normal(size=(80, 9))
    y = rng.normal(size=(80, 68))
    train, val, held = (
        np.arange(80) < 40,
        (np.arange(80) >= 40) & (np.arange(80) < 60),
        np.arange(80) >= 60,
    )
    before = select_ridge(z, y, train, val, held)
    y[held] = 1e8
    after = select_ridge(z, y, train, val, held)
    for i in (0, 1):
        np.testing.assert_array_equal(before[i], after[i])
    assert before[2] == after[2]
    for a, b in zip(before[3], after[3]):
        assert a[:2] == b[:2]
        np.testing.assert_array_equal(a[2], b[2])
        np.testing.assert_array_equal(a[3], b[3])
    for a, b in zip(before[4], after[4]):
        np.testing.assert_array_equal(a, b)


def test_overlapping_roles_and_existing_output_are_rejected(tmp_path):
    z, y = np.ones((10, 2)), np.ones((10, 68))
    mask = np.ones(10, bool)
    with pytest.raises(ValueError, match="partition"):
        select_ridge(z, y, mask, mask, mask)
    with pytest.raises(FileExistsError):
        run(tmp_path / "absent", tmp_path)
