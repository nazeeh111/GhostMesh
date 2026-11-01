"""Synthetic math, role boundaries and exact recovery, not radio efficacy."""

import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp
import torch
from torch import nn

from ghostmesh.geometric_distillation import (
    GeometricStudent,
    paired_objective,
    relational_objective,
)
from scripts import domain_geometric_distillation_probe as d
from scripts import geometric_checkpoint as recovery


def config():
    value = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs/geometric-distillation-source-v1.json"
        ).read_text()
    )
    value.update(steps=4, checkpoint_every=2)
    return value


def data_fixture():
    rng = np.random.default_rng(88)
    rows = []
    for room, start in [("E01", 1), ("E03", 21)]:
        for person in range(start, start + 10):
            for end in [39, 47, 55]:
                rows.append((room, f"S{person:02}", f"{room}/S{person:02}/A01", end))
    rows.extend([("E02", "S11", "E02/S11/A01", x) for x in [39, 47, 55]])
    data = {
        key: np.array([r[i] for r in rows])
        for i, key in enumerate(["room", "person", "record", "end"])
    }
    val = np.isin(data["person"], ["S04", "S09", "S24", "S29"])
    data.update(
        train=(data["room"] != "E02") & ~val,
        validation=val,
        pose=rng.normal(size=(len(rows), 17, 2)).astype(np.float32),
        geometry=rng.normal(size=(len(rows), 136)).astype(np.float32),
    )
    return data


def test_schedule_balances_distinct_source_people_and_exact_causal_pairs():
    data = data_fixture()
    cfg = config()
    plan = d.make_schedule(data, cfg)
    assert plan.shape == (4, 8, 2) and data["train"][plan].all()
    np.testing.assert_array_equal(plan, d.make_schedule(data, cfg))
    for batch in plan:
        assert len(set(data["person"][batch[:, 0]])) == 8
        assert (data["room"][batch[:, 0]] == "E01").sum() == 4
    np.testing.assert_array_equal(
        data["record"][plan[:, :, 0]], data["record"][plan[:, :, 1]]
    )
    np.testing.assert_array_equal(
        data["end"][plan[:, :, 0]] - data["end"][plan[:, :, 1]], np.full((4, 8), 8)
    )


def test_training_label_reader_and_statistics_refuse_validation_or_record_crossing():
    data = data_fixture()
    cfg = config()
    perm = d.source_permutation(data, cfg["permutation_seed"])
    reader = d.TrainingLabels(data, perm)
    ids = np.flatnonzero(data["train"])[:4]
    y, g = reader(ids, "pose_geometry_permuted")
    np.testing.assert_array_equal(y, data["pose"][ids])
    np.testing.assert_array_equal(g, data["geometry"][perm[ids]])
    with pytest.raises(ValueError, match="outside"):
        reader(np.flatnonzero(data["validation"]), "pose_only")
    before = d.statistics(data, perm, cfg)
    data["pose"][~data["train"]] = np.nan
    data["geometry"][~data["train"]] = np.nan
    after = d.statistics(data, perm, cfg)
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])
    bad = perm.copy()
    bad[ids[0]] = np.flatnonzero(data["validation"])[0]
    with pytest.raises(ValueError, match="TRAIN record"):
        d.statistics(data, bad, cfg)
    bad = perm.copy()
    bad[ids[0]] = bad[ids[1]]
    with pytest.raises(ValueError, match="TRAIN record"):
        d.TrainingLabels(data, bad)


def test_paired_supervision_has_analytic_value_and_endpoint_reversal_invariance():
    torch.manual_seed(4)
    p, y = torch.randn(8, 17, 2), torch.randn(8, 17, 2)
    a, delta = torch.rand(17, 2) + 0.1, torch.rand(17, 2) + 0.1
    expected = (
        0.5 * ((p - y) / a).square().mean()
        + 0.5 * (((p[:4] - p[4:]) - (y[:4] - y[4:])) / delta).square().mean()
    )
    torch.testing.assert_close(paired_objective(p, y, a, delta), expected)
    torch.testing.assert_close(
        paired_objective(
            torch.cat([p[4:], p[:4]]), torch.cat([y[4:], y[:4]]), a, delta
        ),
        expected,
    )


def test_relational_kl_matches_independent_numpy_and_detaches_teacher_labels():
    rng = np.random.default_rng(7)
    code = rng.normal(size=(8, 64))
    target = rng.normal(size=(8, 136))
    room = np.array([0] * 4 + [1] * 4)
    mean = rng.normal(size=136)
    scale = rng.uniform(0.1, 2, 136)
    teacher = (target - mean) / scale
    teacher /= np.linalg.norm(teacher, axis=1, keepdims=True)
    student = code / np.linalg.norm(code, axis=1, keepdims=True)
    mask = room[:, None] != room[None, :]
    tl = -np.square(teacher[:, None] - teacher[None]).sum(-1) / (2 * 0.5**2)
    sl = np.einsum("ik,jk->ij", student, student) / 0.2
    tl[~mask] = sl[~mask] = -np.inf
    tl -= logsumexp(tl, axis=1, keepdims=True)
    sl -= logsumexp(sl, axis=1, keepdims=True)
    expected = np.sum(np.exp(tl[mask]) * (tl[mask] - sl[mask])) / 8
    c = torch.tensor(code, requires_grad=True)
    g = torch.tensor(target, requires_grad=True)
    loss = relational_objective(
        c, g, torch.tensor(mean), torch.tensor(scale), torch.tensor(room)
    )
    assert loss.item() == pytest.approx(expected, rel=1e-12, abs=1e-12)
    loss.backward()
    assert g.grad is None and torch.isfinite(c.grad).all() and c.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="different-room"):
        relational_objective(
            c,
            g,
            torch.tensor(mean),
            torch.tensor(scale),
            torch.zeros(8, dtype=torch.long),
        )


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 512)

    def history(self, x):
        return self.linear(x)


def test_model_inference_has_no_label_metadata_api_and_own_hip_origin():
    assert list(inspect.signature(GeometricStudent.forward).parameters) == [
        "self",
        "histories",
    ]
    model = GeometricStudent(
        ToyEncoder(),
        torch.randn(17, 2),
        torch.ones(17, 2),
        torch.zeros(136),
        torch.ones(136),
    )
    x = torch.randn(4, 8)
    a = model(x)
    assert (
        a["pose"].shape == (4, 17, 2)
        and a["geometry"].shape == (4, 136)
        and a["relation"].shape == (4, 64)
    )
    torch.testing.assert_close(
        a["pose"][:, [11, 12]].mean(1), torch.zeros(4, 2), atol=1e-7, rtol=0
    )
    torch.testing.assert_close(model(x)["pose"], a["pose"], atol=0, rtol=0)


def checkpoint_model():
    torch.manual_seed(15)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(0.3), nn.Linear(8, 2))
    return model, torch.optim.AdamW(model.parameters(), lr=0.003)


def advance(model, opt, x, y, start, stop):
    for i in range(start, stop):
        opt.zero_grad(set_to_none=True)
        loss = (model(x[i]) - y[i]).square().mean()
        loss.backward()
        opt.step()


def test_checkpoint_resume_is_exact_for_model_optimizer_and_stochastic_rng(tmp_path):
    gen = torch.Generator().manual_seed(91)
    x = torch.randn(6, 5, 4, generator=gen)
    y = torch.randn(6, 5, 2, generator=gen)
    uninterrupted, oa = checkpoint_model()
    advance(uninterrupted, oa, x, y, 0, 6)
    rng = torch.get_rng_state().clone()
    first, ob = checkpoint_model()
    advance(first, ob, x, y, 0, 3)
    recovery.save(tmp_path, first, ob, 3, "test-identity", "pose_only", "cpu")
    resumed, oc = checkpoint_model()
    assert (
        recovery.load(tmp_path, resumed, oc, "test-identity", "pose_only", "cpu") == 3
    )
    advance(resumed, oc, x, y, 3, 6)
    for name, value in uninterrupted.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], atol=0, rtol=0)
    assert torch.equal(torch.get_rng_state(), rng)
    for index, state in oa.state_dict()["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(
                value, oc.state_dict()["state"][index][name], atol=0, rtol=0
            )


@pytest.mark.parametrize("damage", ["step", "complex", "moment_shape", "rng_length"])
def test_recovery_schema_errors_reject_before_live_state_mutation(tmp_path, damage):
    model, opt = checkpoint_model()
    x = torch.randn(1, 5, 4)
    y = torch.randn(1, 5, 2)
    advance(model, opt, x, y, 0, 1)
    recovery.save(tmp_path, model, opt, 1, "identity", "pose_only", "cpu")
    path = tmp_path / "step-0001.pt"
    saved = torch.load(path, weights_only=True)
    if damage == "step":
        saved["step"] = 2
    elif damage == "complex":
        key = next(iter(saved["model"]))
        saved["model"][key] = saved["model"][key].to(torch.complex64)
    elif damage == "moment_shape":
        state = next(iter(saved["optimizer"]["state"].values()))
        state["exp_avg"] = torch.ones(1)
    else:
        saved["rng_cpu"] = torch.ones(1, dtype=torch.uint8)
    torch.save(saved, path)
    seal = json.loads((tmp_path / "step-0001.pt.json").read_text())
    seal["checkpoint_sha256"] = recovery.digest(path)
    (tmp_path / "step-0001.pt.json").write_text(json.dumps(seal))
    fresh, newopt = checkpoint_model()
    before = copy.deepcopy(fresh.state_dict())
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        recovery.load(tmp_path, fresh, newopt, "identity", "pose_only", "cpu")
    for name, value in before.items():
        torch.testing.assert_close(value, fresh.state_dict()[name], atol=0, rtol=0)
    assert not newopt.state and torch.equal(rng, torch.get_rng_state())


def test_unsealed_or_wrong_identity_checkpoint_refuses(tmp_path):
    model, opt = checkpoint_model()
    recovery.save(tmp_path, model, opt, 0, "identity", "pose_only", "cpu")
    with pytest.raises(ValueError, match="identity"):
        recovery.load(tmp_path, model, opt, "changed", "pose_only", "cpu")
    (tmp_path / "step-0000.pt.json").unlink()
    with pytest.raises(ValueError, match="Unsealed"):
        recovery.load(tmp_path, model, opt, "identity", "pose_only", "cpu")


def test_completed_marker_without_checkpoint_cannot_silently_restart(tmp_path):
    model, opt = checkpoint_model()
    (tmp_path / "complete.json").write_text('{"step":2000}')
    with pytest.raises(ValueError, match="without a sealed"):
        recovery.load(tmp_path, model, opt, "identity", "pose_only", "cpu")


def test_config_rejects_nonfinite_or_invalid_training_budgets():
    for key, value in [
        ("steps", True),
        ("steps", 0),
        ("batch_pairs", 3),
        ("encoder_lr", float("inf")),
        ("teacher_temperature", 0),
        ("geometry_weight", float("nan")),
    ]:
        c = config()
        c[key] = value
        with pytest.raises(ValueError):
            d.validate_config(c)


def test_nonfinite_loss_refuses_optimizer_update():
    data = data_fixture()
    cfg = config()
    perm = d.source_permutation(data, 90210)
    stats = d.statistics(data, perm, cfg)
    model = GeometricStudent(
        ToyEncoder(),
        *[
            torch.from_numpy(stats[k])
            for k in ["pose_mean", "pose_scale", "geometry_mean", "geometry_scale"]
        ],
    )
    opt = d.make_optimizer(model, cfg)
    tensor_stats = {k: torch.from_numpy(v) for k, v in stats.items()}
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="Nonfinite loss"):
        d.train_step(
            model,
            opt,
            torch.zeros(4, 8),
            torch.full((4, 17, 2), float("nan")),
            torch.zeros(4, 136),
            torch.tensor([0, 1, 0, 1]),
            tensor_stats,
            cfg,
            "pose_only",
        )
    for key, value in before.items():
        torch.testing.assert_close(value, model.state_dict()[key], atol=0, rtol=0)
    assert not opt.state


def test_geometry_supervision_changes_encoder_gradients_while_pose_only_ignores_it():
    data = data_fixture()
    cfg = config()
    perm = d.source_permutation(data, 90210)
    stats = d.statistics(data, perm, cfg)
    model = GeometricStudent(
        ToyEncoder(),
        *[
            torch.from_numpy(stats[k])
            for k in ["pose_mean", "pose_scale", "geometry_mean", "geometry_scale"]
        ],
    )
    with torch.no_grad():
        model.pose_head.weight.normal_(std=0.02)
        model.geometry_head.weight.normal_(std=0.02)
    x = torch.randn(4, 8)
    y = torch.randn(4, 17, 2)
    geometry = torch.randn(4, 136)
    rooms = torch.tensor([0, 1, 0, 1])
    ts = {k: torch.from_numpy(v) for k, v in stats.items()}
    gradients = {}
    from ghostmesh.geometric_distillation import distillation_objective

    for arm in ["pose_only", "pose_geometry"]:
        for changed in [False, True]:
            model.zero_grad(set_to_none=True)
            loss = distillation_objective(
                model(x),
                y,
                geometry + 5 if changed else geometry,
                ts["pose_delta_scale"],
                ts["geometry_delta_scale"],
                model.pose_scale,
                model.geometry_mean,
                model.geometry_scale,
                rooms,
                arm,
            )
            loss.backward()
            gradients[arm, changed] = model.encoder.linear.weight.grad.clone()
    torch.testing.assert_close(
        gradients["pose_only", False], gradients["pose_only", True], atol=0, rtol=0
    )
    assert not torch.allclose(
        gradients["pose_geometry", False], gradients["pose_geometry", True]
    )


def test_infinite_gradient_is_rejected_before_parameter_or_optimizer_changes():
    data = data_fixture()
    cfg = config()
    perm = d.source_permutation(data, 90210)
    stats = d.statistics(data, perm, cfg)
    model = GeometricStudent(
        ToyEncoder(),
        *[
            torch.from_numpy(stats[k])
            for k in ["pose_mean", "pose_scale", "geometry_mean", "geometry_scale"]
        ],
    )
    opt = d.make_optimizer(model, cfg)
    ts = {k: torch.from_numpy(v) for k, v in stats.items()}
    before = copy.deepcopy(model.state_dict())
    model.pose_head.weight.register_hook(lambda g: torch.full_like(g, float("inf")))
    with pytest.raises(RuntimeError, match="non-finite"):
        d.train_step(
            model,
            opt,
            torch.randn(4, 8),
            torch.randn(4, 17, 2),
            torch.randn(4, 136),
            torch.tensor([0, 1, 0, 1]),
            ts,
            cfg,
            "pose_only",
        )
    for key, value in before.items():
        torch.testing.assert_close(value, model.state_dict()[key], atol=0, rtol=0)
    assert not opt.state
