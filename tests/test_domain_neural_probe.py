"""SOFTWARE TEST ONLY: source split and factorization contracts on tiny tensors."""
import importlib.util
from pathlib import Path

import numpy as np
import torch

SPEC = importlib.util.spec_from_file_location("domain_neural_probe_test", Path(__file__).resolve().parents[1] / "scripts/domain_neural_probe.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_fold_separates_source_people_and_held_development_room():
    arrays = {"room": np.array(["E01", "E01", "E02", "E02", "E03", "E03"]),
              "subject": np.array(["S01", "S04", "S11", "S14", "S21", "S24"])}
    for room in probe.ROOMS:
        train, val, held = probe.masks(arrays, room)
        assert not (train & val).any() and not (held & (train | val)).any()
        assert set(arrays["room"][held]) == {room}
        assert set(arrays["subject"][val]) <= {"S04", "S14", "S24"}
        assert not set(arrays["subject"][train]) & {"S04", "S14", "S24"}
        assert train.sum() == val.sum() == held.sum() == 2


def test_all_arms_start_at_same_training_only_mean_pose():
    torch.manual_seed(11)
    labels = torch.rand(5, 17, 2).numpy().astype(np.float32)
    stats = probe.target_stats(labels)
    moments, dynamic = torch.randn(3, 8, 114), torch.randn(3, 8, 114)
    for arm in probe.ARMS:
        model = probe.Probe(arm, stats)
        out = model(moments, dynamic)
        np.testing.assert_allclose(out["pose"].detach(), np.broadcast_to(labels.mean(0), (3, 17, 2)), atol=1e-7)
        assert torch.isfinite(out["log_sigma"]).all()
        if model.factorized:
            torch.testing.assert_close(out["pose"][:, [11, 12]].mean(1, keepdim=True), out["root"])


def test_absolute_target_translation_cannot_supervise_relative_branch():
    torch.manual_seed(17)
    target = torch.rand(4, 17, 2) * .4 + .2
    model = probe.Probe("factorized_dynamic_nll", probe.target_stats(target.numpy()))
    moments, dynamic = torch.randn(4, 8, 114), torch.randn(4, 8, 114)
    valid = torch.ones(4, 17, dtype=torch.bool)
    gradients = []
    for shift in (0., .1):
        model.zero_grad(set_to_none=True)
        model.loss(model(moments, dynamic), target + shift, valid).backward()
        gradients.append({name: p.grad.clone() for name, p in model.named_parameters()})
    for name in gradients[0]:
        if name.startswith("root_head"):
            continue
        torch.testing.assert_close(gradients[0][name], gradients[1][name], atol=1e-4, rtol=1e-4)
    assert not torch.allclose(gradients[0]["root_head.bias"], gradients[1]["root_head.bias"])
