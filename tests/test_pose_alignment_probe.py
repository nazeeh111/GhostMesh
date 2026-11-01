import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import domain_pose_alignment_probe as p


def config():
    return json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/pose-alignment-e03-v1.json"
        ).read_text()
    )


def test_prototypes_preserve_train_prior_with_actual_positive_cluster_counts():
    rng = np.random.default_rng(2)
    target = rng.normal(size=(320, 17, 2))
    target -= target[:, [11, 12]].mean(1, keepdims=True)
    proto, prior, assignments = p.prototypes(target, np.ones((17, 2)), 2027)
    assert proto.shape == (128, 17, 2) and len(assignments) == len(target)
    assert (prior > 0).all()
    np.testing.assert_allclose(
        np.einsum("k,kjd->jd", prior, proto), target.mean(0), atol=1e-7
    )


def test_train_rf_reader_refuses_query_rows_before_access():
    class Forbidden:
        def __getitem__(self, key):
            raise AssertionError("Role gate must run first")

    reader = p.Reader(Forbidden(), np.array([True, False]))
    with pytest.raises(ValueError, match="explicit source role"):
        reader(np.array([1]))


def test_config_forbids_wrong_room_budget_and_decay():
    p.validate_config(config())
    for key, value in [("source_room", "E03"), ("steps", 0), ("weight_decay", -1)]:
        cfg = config()
        cfg[key] = value
        with pytest.raises(ValueError):
            p.validate_config(cfg)


def test_loss_control_changes_teacher_gradient_with_identical_rf_and_pose_budget():
    torch.manual_seed(11)
    cfg = config()
    stats = dict(
        input_mean=np.zeros((12, 114), dtype=np.float32),
        input_scale=np.ones((12, 114), dtype=np.float32),
        prototypes=np.random.default_rng(3)
        .normal(size=(128, 17, 2))
        .astype(np.float32),
        prior=np.full(128, 1 / 128, dtype=np.float32),
        pose_scale=np.ones((17, 2), dtype=np.float32),
        pose_delta_scale=np.ones((17, 2), dtype=np.float32),
    )
    initial = p.make_model(stats, "cpu").state_dict()
    x = torch.randn(24, 32, 12, 114)
    y = torch.randn(24, 17, 2)
    people = torch.arange(8).repeat_interleave(3)
    values = {}
    for arm in p.ARMS:
        model = p.make_model(stats, "cpu")
        model.load_state_dict(initial)
        result = p.train_step(
            model, p.optimizer(model, cfg), x, y, people, stats, arm, cfg
        )
        assert np.isfinite(result["loss"])
        values[arm] = model.filters[0].weight.detach().clone()
    assert not torch.equal(values["ordinary"], values["cross_person"])
    assert not torch.equal(values["cross_person"], values["teacher_permuted"])
