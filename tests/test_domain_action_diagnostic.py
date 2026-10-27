from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

spec = spec_from_file_location("action_diagnostic", Path(__file__).resolve().parents[1] / "scripts/domain_action_diagnostic.py")
diagnostic = module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def test_permutation_preserves_whole_records_and_class_balance():
    sequences = np.repeat([f"E01/S01/A{i:02d}" for i in range(1, 28)], 3)
    subjects = np.repeat("S01", len(sequences))
    labels = np.repeat(np.arange(27), 3)
    actual = diagnostic.record_permutation(labels, subjects, sequences, np.ones(len(labels), bool))
    assert not np.array_equal(actual, labels)
    assert np.array_equal(np.sort(actual), labels)
    assert all(len(set(actual[sequences == r])) == 1 for r in set(sequences))


def test_splits_are_whole_person_disjoint():
    room = np.repeat(["E01", "E02", "E03"], 5)
    subjects = np.asarray([f"S{i:02d}" for start in (1, 11, 21) for i in range(start, start + 5)])
    train, val, held = diagnostic.masks(room, subjects, "E01", "unseen_room")
    assert train.sum() == 8 and val.sum() == 2 and held.sum() == 5
    train, val, held = diagnostic.masks(room, subjects, "E02", "within_room")
    assert train.sum() == 3 and val.sum() == 1 and held.sum() == 1


def test_accuracy_is_record_aware_and_balanced():
    labels = np.array([0, 0, 0, 1])
    sequences = np.array(["a", "a", "a", "b"])
    scores = np.zeros((4, 27))
    scores[:, 0] = 1
    result = diagnostic.accuracy(scores, labels, sequences)
    assert result["accuracy"] == 0.75
    assert result["balanced_accuracy"] == 0.5
    assert result["record_mean_score_accuracy"] == 0.5


def test_preprocessing_fits_training_only():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(40, 8))
    training = np.arange(len(x)) < 30
    z = x.copy()
    z[~training] += 1000
    first = diagnostic.standardize(x, training)
    second = diagnostic.standardize(z, training)
    assert np.array_equal(first[training], second[training])
    first, width1 = diagnostic.rbf_features(first, training)
    second, width2 = diagnostic.rbf_features(second, training)
    assert width1 == width2
    np.testing.assert_allclose(first[training], second[training], atol=1e-14)
