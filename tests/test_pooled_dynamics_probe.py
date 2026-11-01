"""Pooling must preserve causal supports and source-only numeric access."""

import numpy as np
import pytest

from scripts import domain_pooled_dynamics_probe as pooled
from tests.test_conditioned_dynamics_probe import fixture


def test_mean_uses_all_eight_frames_and_no_other_rows():
    history = np.arange(5 * 8 * 64).reshape(5, 512).astype(float)
    history[3:] = np.nan  # Held values must never be read.
    selected = np.array([0, 2])
    output = pooled.PooledHistory(history)[selected, -64:]
    np.testing.assert_array_equal(output, history[selected].reshape(2, 8, 64).mean(1))
    for frame in range(8):
        changed = history.copy()
        changed[0, frame * 64 : (frame + 1) * 64] += 8
        np.testing.assert_array_equal(
            pooled.PooledHistory(changed)[selected, -64:][0], output[0] + 1
        )


def test_pooling_fit_never_reads_held_values_and_keeps_source_models_frozen():
    feature, target, metadata, train, validation = fixture()

    class Tracked:
        def __getitem__(self, ids):
            assert (train | validation)[ids].all()
            return feature[ids]

    first = pooled.p.fit_source_models(
        pooled.PooledHistory(Tracked()), target, metadata, train, validation
    )
    feature[validation] += 1000
    target[validation] -= 1000
    feature[~(train | validation)], target[~(train | validation)] = np.nan, np.nan
    second = pooled.p.fit_source_models(
        pooled.PooledHistory(Tracked()), target, metadata, train, validation
    )
    for key, value in first["mapping"].items():
        np.testing.assert_array_equal(value, second["mapping"][key])
    for key, value in first["models"].items():
        np.testing.assert_array_equal(value["weight"], second["models"][key]["weight"])


def test_pooling_rejects_invalid_views_before_prediction():
    with pytest.raises(ValueError, match="declared"):
        pooled.PooledHistory(np.ones((3, 512)))[np.array([0]), :32]
    with pytest.raises(ValueError, match="Eight"):
        pooled.PooledHistory(np.ones((3, 64)))[np.array([0]), -64:]
    with pytest.raises(ValueError, match="real finite"):
        pooled.PooledHistory(np.ones((3, 512), complex))[np.array([0]), -64:]


def test_pooling_refuses_existing_output_before_input_reads(tmp_path):
    with pytest.raises(FileExistsError):
        pooled.run(tmp_path / "absent", tmp_path)
