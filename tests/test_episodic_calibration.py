"""Independent closed-form algebra, gradients and CSI-only query contracts."""

import inspect

import numpy as np
import pytest
import torch

from ghostmesh.episodic_calibration import EpisodicStudent, apply_residual, fit_residual
from ghostmesh.motion_filter import MotionFilter


@pytest.mark.parametrize("n,d", [(5, 9), (9, 5)])
def test_dual_and_primal_solver_match_independent_numpy(n, d):
    rng = np.random.default_rng(1)
    s, q, y, b, z = [
        rng.normal(size=shape) for shape in [(n, d), (3, d), (n, 2), (n, 2), (3, 2)]
    ]
    sn = s / np.linalg.norm(s, axis=1, keepdims=True)
    qn = q / np.linalg.norm(q, axis=1, keepdims=True)
    xm = sn.mean(0)
    residual = y - b
    ym = residual.mean(0)
    x = sn - xm
    w = np.linalg.solve(x.T @ x + (0.1 * n / 24) * np.eye(d), x.T @ (residual - ym))
    expected = z + (qn - xm) @ w + ym
    state = fit_residual(*[torch.tensor(a) for a in [s, y, b]])
    actual = apply_residual(state, torch.tensor(q), torch.tensor(z))
    np.testing.assert_allclose(actual.numpy(), expected, atol=2e-12, rtol=2e-12)


def test_fitted_state_reuses_queries_and_preserves_support_permutation():
    torch.manual_seed(2)
    s, q, y, b, z = [
        torch.randn(*shape, dtype=torch.float64)
        for shape in [(6, 4), (5, 4), (6, 2), (6, 2), (5, 2)]
    ]
    a = fit_residual(s, y, b)
    p = torch.tensor([3, 1, 5, 2, 0, 4])
    other = fit_residual(s[p], y[p], b[p])
    expected = apply_residual(a, q, z)
    torch.testing.assert_close(
        apply_residual(other, q, z), expected, atol=2e-12, rtol=2e-12
    )
    torch.testing.assert_close(
        torch.cat([apply_residual(a, q[:2], z[:2]), apply_residual(a, q[2:], z[2:])]),
        expected,
        atol=2e-12,
        rtol=2e-12,
    )
    assert "target" not in inspect.signature(apply_residual).parameters


def test_gradients_pass_through_exact_solver_but_reference_labels_are_detached():
    torch.manual_seed(3)
    s = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    q = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    y = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    b = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    z = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)

    def f(a, c, d, e):
        return apply_residual(fit_residual(a, y, d), c, e)

    assert torch.autograd.gradcheck(f, (s, q, b, z), eps=1e-6, atol=2e-5, rtol=2e-4)
    f(s, q, b, z).square().sum().backward()
    assert y.grad is None
    for value in (s, q, b, z):
        assert torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), True])
def test_invalid_ridge_is_refused(bad):
    with pytest.raises(ValueError):
        fit_residual(torch.ones(3, 4), torch.ones(3, 2), torch.zeros(3, 2), ridge=bad)


def test_identical_codes_produce_only_the_reference_bias_and_refuse_nonfinite():
    s = torch.ones(4, 3)
    y = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    b = torch.zeros(4, 2)
    state = fit_residual(s, y, b)
    actual = apply_residual(state, torch.ones(2, 3), torch.zeros(2, 2))
    torch.testing.assert_close(actual, y.mean(0).expand(2, -1))
    y[0, 0] = float("nan")
    with pytest.raises(ValueError, match="Finite"):
        fit_residual(s, y, b)


def test_wrapper_initial_zero_shot_matches_source_and_accepts_only_labeled_support():
    torch.manual_seed(4)
    base = MotionFilter(
        torch.zeros(12, 114), torch.ones(12, 114), torch.zeros(17, 2), torch.ones(17, 2)
    ).eval()
    net = EpisodicStudent(base).eval()
    x = torch.randn(5, 8, 12, 114)
    assert sum(p.numel() for p in net.parameters()) == sum(
        p.numel() for p in base.parameters()
    )
    with torch.inference_mode():
        torch.testing.assert_close(net(x)["pose"], base(x), atol=0, rtol=0)
        result = net.calibrate(x[:3], base(x[:3]), x[3:])
        torch.testing.assert_close(result, base(x[3:]), atol=2e-7, rtol=2e-7)
    assert set(inspect.signature(net.calibrate).parameters) == {
        "support_histories",
        "support_poses",
        "query_histories",
        "ridge",
    }


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Native MPS unavailable"
)
def test_native_mps_float32_codes_cross_cpu_double_solve_and_keep_gradients():
    torch.manual_seed(5)
    s = torch.randn(5, 4, device="mps", requires_grad=True)
    q = torch.randn(3, 4, device="mps", requires_grad=True)
    b = torch.randn(5, 2, device="mps", requires_grad=True)
    z = torch.randn(3, 2, device="mps", requires_grad=True)
    y = torch.randn(5, 2, device="mps")
    prediction = apply_residual(fit_residual(s, y, b), q, z)
    assert prediction.device.type == "mps" and prediction.dtype == torch.float32
    prediction.square().mean().backward()
    for value in (s, q, b, z):
        assert torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0
