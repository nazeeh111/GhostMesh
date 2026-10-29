import torch
from torch import nn

from ghostmesh.calibrated_pose import (
    CalibratedPose,
    pack_pose,
    unpack_pose,
    ridge_adapt,
)


def test_pose_factorization_roundtrip_and_translation():
    torch.manual_seed(1)
    pose = torch.rand(5, 17, 2, dtype=torch.float64)
    packed = pack_pose(pose)
    torch.testing.assert_close(unpack_pose(packed), pose)
    shift = torch.tensor([0.2, -0.1], dtype=torch.float64)
    translated = pack_pose(pose + shift)
    torch.testing.assert_close(translated[:, :34], packed[:, :34])
    torch.testing.assert_close(translated[:, 34:], packed[:, 34:] + shift)


def test_ridge_learns_supported_direction_and_retains_prior_elsewhere():
    support = torch.tensor([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    query = torch.tensor([[0.4, 3.0]], dtype=torch.float64)
    prior = nn.Linear(2, 1).double()
    with torch.no_grad():
        prior.weight.copy_(torch.tensor([[0.0, 2.0]], dtype=torch.float64))
        prior.bias.zero_()
    target = 3 * support[:, :1] + 5
    result = ridge_adapt(support, target, query, prior, ridge=1e-9)
    torch.testing.assert_close(
        result, torch.tensor([[12.2]], dtype=torch.float64), atol=1e-8, rtol=1e-8
    )


def test_adaptation_gradients_match_finite_differences():
    torch.manual_seed(3)
    zs = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    ys = torch.randn(5, 2, dtype=torch.float64, requires_grad=True)
    zq = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    prior = nn.Linear(3, 2).double()
    assert torch.autograd.gradcheck(
        lambda a, b, c: ridge_adapt(a, b, c, prior, 0.1),
        (zs, ys, zq),
        atol=1e-5,
        rtol=1e-4,
    )


def test_meta_loss_reaches_encoder_and_prior_without_query_labels_as_inputs():
    torch.manual_seed(4)
    model = CalibratedPose()
    support = torch.randn(6, 8, 12, 114)
    query = torch.randn(3, 8, 12, 114)
    labels = torch.randn(6, 36)
    prediction = model.predict_adapted(support, labels, query)
    prediction.square().mean().backward()
    assert model.spectral[0].weight.grad.abs().sum() > 0
    assert model.prior.weight.grad.abs().sum() > 0
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
    model.eval()
    torch.testing.assert_close(
        model.encode(torch.cat((support, query)))[:6],
        model.encode(support),
        atol=1e-6,
        rtol=1e-5,
    )


def test_temporal_body_has_no_future_dependency():
    torch.manual_seed(5)
    model = CalibratedPose()
    x = torch.randn(2, 8, 64)
    changed = x.clone()
    changed[:, 4:] += 30
    torch.testing.assert_close(model.temporal(x)[:, :4], model.temporal(changed)[:, :4])
