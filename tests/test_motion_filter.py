"""Signal identities and actual model contracts, not sensing-accuracy tests."""

import numpy as np
import pytest
import torch
import copy

from ghostmesh.motion_filter import MotionFilter, packet_moments


def model():
    return MotionFilter(
        torch.zeros(12, 114),
        torch.ones(12, 114),
        torch.zeros(17, 2),
        torch.ones(17, 2),
    ).eval()


def test_packet_moments_cancel_shared_gain_phase_and_packet_reordering():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(8, 6, 114, 10)).astype(np.float32)
    moved = x.copy()
    moved[:, :3] += rng.uniform(-10, 10, (8, 1, 114, 10))
    moved[:, 3:] += rng.uniform(-3, 3, (8, 1, 114, 10))
    expected = packet_moments(x)
    np.testing.assert_allclose(packet_moments(moved), expected, atol=5e-7, rtol=2e-6)
    permutation = rng.permutation(10)
    np.testing.assert_allclose(
        packet_moments(x[..., permutation]), expected, atol=2e-7, rtol=2e-6
    )
    assert expected.shape == (8, 12, 114) and np.isfinite(expected).all()


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((8, 6, 113, 10)),
        np.zeros((0, 6, 114, 10)),
        np.zeros((8, 6, 114, 10), complex),
    ],
)
def test_packet_schema_is_explicit(bad):
    with pytest.raises(ValueError, match="typed RF"):
        packet_moments(bad)


def test_missing_pair_threshold_is_preserved():
    raw = np.ones((8, 6, 114, 10), np.float32)
    raw[:, 0] = np.nan
    with pytest.raises(ValueError, match="invalid quotient"):
        packet_moments(raw)


def test_snapshot_has_no_dependence_on_past_and_body_has_no_root():
    torch.manual_seed(9)
    net = model()
    x = torch.randn(2, 8, 12, 114)
    changed = x.clone()
    changed[:, :-1] = torch.randn_like(changed[:, :-1]) * 100
    with torch.inference_mode():
        torch.testing.assert_close(
            net(x, mode="snapshot"), net(changed, mode="snapshot"), atol=0, rtol=0
        )
        assert not torch.allclose(net(x), net(changed))
        torch.testing.assert_close(
            net(x)[:, [11, 12]].mean(1), torch.zeros(2, 2), atol=2e-7, rtol=0
        )
    x.requires_grad_(True)
    net(x).square().mean().backward()
    assert torch.isfinite(x.grad).all() and (x.grad[:, 0].abs().sum() > 0)
    assert x.grad[:, -1].abs().sum() > 0


def test_permuted_mode_is_the_declared_history_order_and_rejects_bad_statistics():
    torch.manual_seed(10)
    net = model()
    x = torch.randn(2, 8, 12, 114)
    with torch.inference_mode():
        torch.testing.assert_close(
            net(x, mode="history_permuted"),
            net(x[:, [6, 4, 2, 0, 5, 3, 1, 7]]),
            rtol=0,
            atol=0,
        )
    with pytest.raises(ValueError, match="source statistic"):
        MotionFilter(
            torch.zeros(12, 114),
            torch.zeros(12, 114),
            torch.zeros(17, 2),
            torch.ones(17, 2),
        )
    with pytest.raises(ValueError, match="ablation"):
        net(x, mode="unknown")


def test_dense_normalization_prevents_shared_negative_shift_from_killing_signal():
    torch.manual_seed(11)
    net = model()
    shifted = copy.deepcopy(net)
    with torch.no_grad():
        shifted.filters[7].bias -= 100
    x = torch.randn(4, 8, 12, 114, requires_grad=True)
    expected, observed = net(x), shifted(x)
    # Float32 variance with a large offset incurs cancellation. Check bounded
    # coordinate drift and preserved signal, rather than exact shift invariance.
    assert (observed - expected).abs().max() < 0.002
    assert observed.detach().std(0).mean() > 0.9 * expected.detach().std(0).mean()
    assert float(observed.detach().std(0).mean()) > 0.01
    observed.square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0.01
