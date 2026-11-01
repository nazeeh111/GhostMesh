import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import domain_dense_room_probe as p


def config():
    return json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/cross-room-dense-e03-v1.json"
        ).read_text()
    )


def test_configuration_preserves_both_source_rooms_and_explicit_reference_budgets():
    p.validate_config(config())
    for key, value in [
        ("source_rooms", ["E01", "E03"]),
        ("source_reference_rows_total", 120),
    ]:
        cfg = config()
        cfg[key] = value
        with pytest.raises(ValueError):
            p.validate_config(cfg)


def test_each_room_reference_fit_cannot_consume_other_room_labels():
    rng = np.random.default_rng(3)
    encoded = dict(
        reference_code=rng.normal(size=(240, 128)).astype(np.float32),
        reference_pose=np.zeros((240, 17, 2), np.float32),
        query_code=rng.normal(size=(6, 128)).astype(np.float32),
        query_pose=np.zeros((6, 17, 2), np.float32),
    )
    refs = np.array(["E01"] * 120 + ["E02"] * 120)
    queries = np.array(["E01"] * 3 + ["E02"] * 3)
    target = rng.normal(size=(240, 17, 2)).astype(np.float32)
    before, _ = p.calibrate(encoded, target, refs, queries, config())
    target[:120, :, 0] *= 10
    after, _ = p.calibrate(encoded, target, refs, queries, config())
    np.testing.assert_array_equal(before[3:], after[3:])
    assert not np.allclose(before[:3], after[:3])
    queries[0] = "E03"
    with pytest.raises(ValueError, match="roles"):
        p.calibrate(encoded, target, refs, queries, config())


def test_all_loss_arms_update_finitely_with_identical_rf_and_pose_types():
    torch.manual_seed(21)
    rng = np.random.default_rng(21)
    stats = dict(
        input_mean=np.zeros((12, 114), np.float32),
        input_scale=np.ones((12, 114), np.float32),
        prototypes=rng.normal(size=(128, 17, 2)).astype(np.float32) * 0.1,
        prior=np.full(128, 1 / 128, np.float32),
        pose_scale=np.ones((17, 2), np.float32),
        pose_delta_scale=np.ones((17, 2), np.float32),
    )
    model = p.base.make_model(stats, "cpu")
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    x = torch.randn(32, 32, 12, 114)
    y = torch.randn(32, 17, 2)
    rooms = torch.tensor([0] * 16 + [1] * 16)
    weights = {}
    for arm in p.ARMS:
        model.load_state_dict(initial)
        result = p.train_step(
            model, p.base.optimizer(model, config()), x, y, rooms, stats, arm, config()
        )
        assert np.isfinite(result["loss"])
        weights[arm] = model.filters[0].weight.detach().clone()
    assert not torch.equal(weights["ordinary"], weights["cross_room"])
    assert not torch.equal(weights["cross_room"], weights["teacher_permuted"])
