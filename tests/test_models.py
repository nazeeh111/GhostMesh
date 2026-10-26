import pytest
import torch

from ghostmesh.models import (
    CSIPose,
    CausalBlock,
    build_model,
    radius_confidence,
    supervised_loss,
)


def test_model_shape_gradients_and_uncertainty():
    torch.manual_seed(3)
    torch.set_num_threads(2)
    model = CSIPose(width=16)
    x = torch.randn(2, 8, 3, 32, 10)
    pred = model(x)
    assert pred["pose"].shape == (2, 17, 2)
    assert pred["log_sigma"].shape == (2, 17)
    assert torch.isfinite(pred["pose"]).all()
    assert ((pred["pose"] >= 0) & (pred["pose"] <= 1)).all()
    loss = supervised_loss(
        pred, torch.rand(2, 17, 2), torch.ones(2, 17, dtype=torch.bool)
    )
    loss.backward()
    assert model.encoder.net[0].weight.grad.abs().sum() > 0
    assert model.temporal_net[0].depthwise.weight.grad.abs().sum() > 0
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_temporal_encoder_has_no_future_dependency():
    model = CSIPose(width=16).eval()
    x = torch.randn(1, 8, 3, 32, 10)
    changed = x.clone()
    changed[:, 5:] = torch.randn_like(changed[:, 5:]) * 50
    with torch.no_grad():
        original, mutated = model.encode_sequence(x), model.encode_sequence(changed)
    torch.testing.assert_close(original[:, :5], mutated[:, :5], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(original[:, -1], mutated[:, -1])


def test_last_frame_encoder_ignores_other_preprocessed_frames():
    model = CSIPose(width=16, temporal=False).eval()
    x = torch.randn(1, 8, 3, 32, 10)
    changed = x.clone()
    changed[:, :-1] = 100
    with torch.no_grad():
        torch.testing.assert_close(model(x)["pose"], model(changed)["pose"])


def test_masks_ignore_nan_labels_and_reject_empty_supervision():
    model = build_model({"model": {"width": 16}})
    prediction = model(torch.rand(1, 8, 3, 16, 4))
    target = torch.rand(1, 17, 2)
    target[:, 0] = float("nan")
    valid = torch.ones(1, 17, dtype=torch.bool)
    loss = supervised_loss(prediction, target, valid)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    with pytest.raises(ValueError, match="no finite"):
        supervised_loss(prediction, target, torch.zeros_like(valid))


def test_confidence_has_an_explicit_radius_and_monotonic_uncertainty():
    confidence = radius_confidence(torch.tensor([-4.0, -2.0, 0.0]), 0.05)
    assert confidence[0] > confidence[1] > confidence[2]
    assert (confidence >= 0).all() and (confidence <= 1).all()
    with pytest.raises(ValueError):
        radius_confidence(torch.zeros(1), 0)


def test_eval_is_deterministic_without_dropout():
    model = CSIPose(width=16).eval()
    x = torch.randn(1, 8, 3, 32, 10)
    with torch.no_grad():
        torch.testing.assert_close(model(x)["pose"], model(x)["pose"], atol=0, rtol=0)


def test_shape_validation():
    with pytest.raises(ValueError):
        CSIPose(width=13)
    with pytest.raises(ValueError):
        CSIPose(width=16)(torch.zeros(1, 8, 2, 16, 10))


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("dilation", [1, 2, 4])
def test_direct_kernel_matches_native_values_gradients_and_state(
    width, dtype, batch, dilation
):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    native = CausalBlock(width, dilation, "native").to(dtype)
    direct = CausalBlock(width, dilation, "direct").to(dtype)
    direct.load_state_dict(native.state_dict(), strict=True)
    x = torch.randn(batch, width, 8, dtype=dtype, requires_grad=True)
    z = x.detach().clone().requires_grad_(True)
    a, b = native(x), direct(z)
    torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-6)
    a.square().mean().backward()
    b.square().mean().backward()
    torch.testing.assert_close(x.grad, z.grad, atol=3e-6, rtol=3e-6)
    for p, q in zip(native.parameters(), direct.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=3e-5, rtol=3e-5)
    assert list(native.state_dict()) == list(direct.state_dict())


def test_packet_moments_retain_antenna_identity_but_ignore_packet_order():
    torch.set_num_threads(1)
    torch.manual_seed(2)
    model = CSIPose(width=32, encoder="packet_moment", temporal_kernel="direct")
    x = torch.randn(2, 8, 3, 114, 10, requires_grad=True)
    with torch.no_grad():
        a = model(x)
        permuted = model(x[..., torch.randperm(10)])
        torch.testing.assert_close(a["pose"], permuted["pose"], atol=1e-6, rtol=1e-6)
        assert not torch.allclose(a["pose"], model(x[:, :, [2, 1, 0]])["pose"])
        changed = x.detach().clone()
        changed[:, 5:] *= 30
        torch.testing.assert_close(
            model.encode_sequence(x)[:, :5],
            model.encode_sequence(changed)[:, :5],
            atol=1e-6,
            rtol=1e-6,
        )
    prediction = model(x)
    supervised_loss(
        prediction, torch.rand(2, 17, 2), torch.ones(2, 17, dtype=torch.bool)
    ).backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
