"""Learn features for labeled-reference adaptation, not a proven sensing system.

Differentiable ridge regression is an established meta-learning building block
(Bertinetto et al., arXiv:1805.08136). These equations and CSI wrapper are original
implementations; no upstream code/weights are used. Only support references have
labels. A query prediction never receives a query pose or its body root.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


def matrix(value, name):
    if (
        not torch.is_tensor(value)
        or value.ndim != 2
        or min(value.shape) == 0
        or not value.is_floating_point()
        or value.is_complex()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("Finite real nonempty matrix required: " + name)
    return value


@dataclass(frozen=True)
class ResidualState:
    code_mean: torch.Tensor
    correction_mean: torch.Tensor
    weights: torch.Tensor
    references: int
    regularization: float


def normalized_cpu(code):
    # Native MPS does not provide every double-precision factorization. Explicit
    # CPU float64 solves retain autograd through device/dtype copies, without
    # global fallback switches, approximate inverses or changing the objective.
    return F.normalize(
        matrix(code, "codes").to(device="cpu").to(dtype=torch.float64), dim=1, eps=1e-6
    )


def fit_residual(
    support_code, support_target, support_base, *, ridge=0.1, reference_count=24
):
    """Fit one support residual map; differentiable in codes and base model.

    Labels are detached training/reference data. Regularization scales with the
    number of references, keeping the per-reference loss convention unchanged.
    The smaller dual/primal positive-definite system is solved exactly; finite
    failures propagate. A fitted map can be reused for streaming queries.
    """
    s = normalized_cpu(support_code)
    target, base = (
        matrix(support_target, "support labels"),
        matrix(support_base, "support predictions"),
    )
    if (
        len(s) < 2
        or target.shape != base.shape
        or len(target) != len(s)
        or isinstance(ridge, bool)
        or not isinstance(ridge, (float, int))
        or not math.isfinite(ridge)
        or ridge <= 0
        or type(reference_count) is not int
        or reference_count <= 0
    ):
        raise ValueError(
            "Matching support rows and positive ridge/reference count required"
        )
    # Separate transfers/casts are required: MPS dispatch can attempt a float64
    # cast on the GPU when both are requested in one .to() operation.
    residual = target.detach().to(device="cpu").to(dtype=torch.float64) - base.to(
        device="cpu"
    ).to(dtype=torch.float64)
    mean, correction = s.mean(0), residual.mean(0)
    x, y = s - mean, residual - correction
    regularization = float(ridge * len(s) / reference_count)
    if len(s) < s.shape[1]:
        system = x @ x.T + regularization * torch.eye(len(s), dtype=x.dtype)
        factor = torch.linalg.cholesky(system)
        weights = x.T @ torch.cholesky_solve(y, factor)
    else:
        system = x.T @ x + regularization * torch.eye(x.shape[1], dtype=x.dtype)
        factor = torch.linalg.cholesky(system)
        weights = torch.cholesky_solve(x.T @ y, factor)
    if not bool(torch.isfinite(weights).all()):
        raise ValueError("Nonfinite fitted reference map")
    return ResidualState(mean, correction, weights, len(s), regularization)


def apply_residual(state, query_code, query_base):
    """CSI-code/base-prediction input only, with no query-label argument."""
    q, base = normalized_cpu(query_code), matrix(query_base, "query predictions")
    if (
        q.shape[0] != len(base)
        or state.code_mean.shape != (q.shape[1],)
        or state.correction_mean.shape != (base.shape[1],)
        or state.weights.shape != (q.shape[1], base.shape[1])
    ):
        raise ValueError("Fitted reference/query dimensions differ")
    correction = (q - state.code_mean) @ state.weights + state.correction_mean
    result = base + correction.to(dtype=base.dtype).to(device=base.device)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Nonfinite calibrated prediction")
    return result


class EpisodicStudent(nn.Module):
    """Expose the original filter representation with an unchanged zero-shot head.

    Initialization copies the supplied source model. No extra trainable
    parameters are added; ordinary and episodic controls can have equal size.
    This wrapper does not call or inspect reference/query person/room/action IDs.
    """

    def __init__(self, source_filter):
        super().__init__()
        self.body = copy.deepcopy(source_filter)

    def forward(self, histories):
        if histories.ndim != 4 or histories.shape[1:] != (8, 12, 114):
            raise ValueError("Causal moment histories[N,8,12,114] required")
        z = (histories - self.body.input_mean) / self.body.input_scale
        z = torch.cat((z, z - z[:, :1]), dim=2).transpose(1, 2)
        for layer in self.body.filters[:-1]:
            z = layer(z)
        pose = (
            self.body.filters[-1](z).reshape(-1, 17, 2) * self.body.pose_scale
            + self.body.pose_mean
        )
        pose = pose - pose[:, [11, 12]].mean(1, keepdim=True)
        return dict(pose=pose, code=z)

    def calibrate(
        self, support_histories, support_poses, query_histories, *, ridge=0.1
    ):
        support, query = self(support_histories), self(query_histories)
        if support_poses.shape != support["pose"].shape:
            raise ValueError("Reference poses must match support histories")
        y = support_poses - support_poses[:, [11, 12]].mean(1, keepdim=True)
        state = fit_residual(
            support["code"], y.flatten(1), support["pose"].flatten(1), ridge=ridge
        )
        pose = apply_residual(state, query["code"], query["pose"].flatten(1)).reshape(
            -1, 17, 2
        )
        return pose - pose[:, [11, 12]].mean(1, keepdim=True)
