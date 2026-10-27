"""SOFTWARE ONLY: branch/uncertainty isolation, typed units and mode routing."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from ghostmesh.rf_features import prepare_log_magnitude, quotient_features
from ghostmesh.rf_pose import RFPose, loss_components

SPEC = importlib.util.spec_from_file_location(
    "rf_probe_test", Path(__file__).resolve().parents[1] / "scripts/domain_rf_probe.py"
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def fixture():
    torch.manual_seed(2027)
    torch.set_num_threads(1)
    y = torch.rand(3, 17, 2) * 0.4 + 0.2
    model = RFPose()
    model.initialize_from_source(y)
    return model, torch.randn(3, 8, 7, 114, 10), torch.randn(3, 8, 342), y


def test_mean_init_and_uncertainty_cannot_train_geometry():
    model, body, root, y = fixture()
    out = model(body, root)
    np.testing.assert_allclose(
        out["pose"].detach(), np.broadcast_to(y.mean(0), (3, 17, 2)), atol=1e-7
    )
    loss_components(out, y)["uncertainty"].backward()
    for name, param in model.named_parameters():
        if "uncertainty" in name:
            assert param.grad is not None
        else:
            assert param.grad is None


def test_independent_position_network_receives_root_labels_without_body_gradient():
    model, body, root, y = fixture()
    torch.nn.init.normal_(model.root_head.weight, std=0.01)
    terms = loss_components(model(body, root), y)
    terms["root_geometry"].backward()
    assert (
        sum(
            p.grad.abs().sum()
            for p in model.root_encoder.parameters()
            if p.grad is not None
        )
        > 0
    )
    assert model.root_head.weight.grad.abs().sum() > 0
    assert all(
        p.grad is None
        for name, p in model.named_parameters()
        if not name.startswith(("root_encoder", "root_head"))
    )


def test_articulation_gradient_is_independent_of_root_input_and_labels():
    model, body, root, y = fixture()
    torch.nn.init.normal_(model.relative_head.weight, std=0.01)
    terms = loss_components(model(body, root), y)
    (terms["relative_geometry"] + 0.1 * terms["bone"]).backward()
    assert model.packet_encoder[0].weight.grad.abs().sum() > 0
    assert all(
        p.grad is None
        for name, p in model.named_parameters()
        if name.startswith(("root_encoder", "root_head", "root_uncertainty"))
    )
    first = model(body, root)["relative"]
    second = model(body, root + 10)["relative"]
    torch.testing.assert_close(first, second)


def test_composed_pose_clip_retains_raw_outputs_and_rejects_future_window():
    model, body, root, y = fixture()
    with torch.no_grad():
        model.root_head.bias.fill_(2)
    out = model(body, root)
    assert (out["pose"] == 1).all() and (out["raw_pose"] > 1).all()
    with pytest.raises(ValueError, match="causal"):
        model(torch.randn(3, 9, 7, 114, 10), root)


def test_explicit_db_units_negative_values_and_common_gain_phase_cancellation():
    rng = np.random.default_rng(2027)
    a = rng.normal(-10, 3, (8, 3, 114, 10)).astype("float32")
    p = rng.normal(0, 2, a.shape).astype("float32")
    x = prepare_log_magnitude(a, units="log_magnitude_db")
    assert np.isfinite(x).all()
    np.testing.assert_allclose(
        x, prepare_log_magnitude(a + 20, units="log_magnitude_db"), atol=3e-6
    )
    with pytest.raises(ValueError, match="units"):
        prepare_log_magnitude(a, units="linear_amplitude")
    g = rng.normal(0, 3, (8, 1, 114, 10)).astype("float32")
    angle = rng.normal(0, 2, g.shape).astype("float32")
    np.testing.assert_allclose(
        quotient_features(a, p, units="log_magnitude_db"),
        quotient_features(a + g, p + angle, units="log_magnitude_db"),
        atol=3e-6,
    )


def test_four_modes_preserve_typed_channels_and_matching_shapes():
    legacy = np.full((1, 8, 3, 114, 10), -2, dtype="float32")
    db = np.full_like(legacy, 3)
    q = np.stack(
        [np.full((1, 8, 114, 10), k, dtype="float32") for k in range(6)], axis=2
    )
    pose = np.zeros((1, 8, 17, 2), dtype="float32")
    spectrum = np.full((1, 8, 342), 0.5, dtype="float32")
    for mode in probe.MODES:
        body, root, _ = probe.RFSet(legacy, db, q, spectrum, pose, np.array([0]), mode)[
            0
        ]
        assert body.shape == (8, 7, 114, 10) and root.shape == (8, 342)
        np.testing.assert_array_equal(root, spectrum[0])
        if mode == "quotient":
            np.testing.assert_array_equal(body[:, :6], q[0])
            assert not body[:, 6].any()
        elif mode == "db_phase":
            np.testing.assert_array_equal(body[:, 3:], q[0, :, 2:])
        else:
            assert not body[:, 3:].any()


def test_all_time_outputs_are_prefix_causal_with_nonzero_readouts():
    model, body, root, y = fixture()
    for head in (
        model.relative_head,
        model.relative_uncertainty,
        model.root_head,
        model.root_uncertainty,
    ):
        torch.nn.init.normal_(head.weight, std=0.01)
    complete = model(body, root, return_sequence=True)
    prefix = model(body[:, :4], root[:, :4], return_sequence=True)
    changed_body, changed_root = body.clone(), root.clone()
    changed_body[:, 4:] += 50
    changed_root[:, 4:] -= 50
    changed = model(changed_body, changed_root, return_sequence=True)
    for key in complete:
        torch.testing.assert_close(
            complete[key][:, :4], prefix[key], atol=2e-6, rtol=1e-5
        )
        torch.testing.assert_close(
            complete[key][:, :4], changed[key][:, :4], atol=2e-6, rtol=1e-5
        )
        torch.testing.assert_close(model(body, root)[key], complete[key][:, -1])
    dense_target = y[:, None].expand(-1, 8, -1, -1)
    loss = sum(loss_components(complete, dense_target).values())
    assert torch.isfinite(loss)


def test_frame_frontend_and_pointwise_quotient_preserve_prefixes():
    rng = np.random.default_rng(82)
    raw = rng.normal(40, 8, (8, 3, 114, 10)).astype("float32")
    phase = rng.normal(0, 2, raw.shape).astype("float32")
    altered, changed_phase = raw.copy(), phase.copy()
    altered[4:] += 90
    changed_phase[4:] += 20

    def frames(x):
        return np.stack(
            [
                prepare_log_magnitude(x[i : i + 1], units="log_magnitude_db")[0]
                for i in range(len(x))
            ]
        )

    np.testing.assert_array_equal(frames(raw)[:4], frames(altered)[:4])
    np.testing.assert_array_equal(
        quotient_features(raw, phase, units="log_magnitude_db")[:4],
        quotient_features(altered, changed_phase, units="log_magnitude_db")[:4],
    )


def test_uncertainty_clipping_cannot_rescale_geometry_gradients():
    from copy import deepcopy

    model, body, root, y = fixture()
    for head in (model.relative_head, model.root_head):
        torch.nn.init.normal_(head.weight, std=0.01)
    first, second = deepcopy(model), deepcopy(model)
    for i, candidate in enumerate((first, second)):
        terms = loss_components(candidate(body, root), y)
        total = (
            terms["relative_geometry"]
            + terms["root_geometry"]
            + 0.1 * terms["bone"]
            + (0.1 if i == 0 else 1000) * terms["uncertainty"]
        )
        total.backward()
        probe.clip_groups(candidate)
    for (name, p), (other_name, q) in zip(
        first.named_parameters(), second.named_parameters()
    ):
        assert name == other_name
        if "uncertainty" not in name:
            torch.testing.assert_close(p.grad, q.grad)
