"""Learn a CSI observation map that can adapt from labeled reference poses.

Differentiable ridge adaptation has prior art: Bertinetto et al., ICLR2019,
https://arxiv.org/abs/1805.08136. This independently implemented regressor fits
residuals around a source-trained prior. It makes no novelty or 3D claim.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def pack_pose(pose):
    if pose.shape[-2:] != (17, 2):
        raise ValueError("Expected 17 two-dimensional joints")
    root = pose[..., [11, 12], :].mean(-2, keepdim=True)
    return torch.cat(((pose - root).flatten(-2), root.squeeze(-2)), dim=-1)


def unpack_pose(target):
    if target.shape[-1] != 36:
        raise ValueError("Expected 34 relative coordinates plus 2 root coordinates")
    relative = target[..., :34].reshape(*target.shape[:-1], 17, 2)
    relative = relative - relative[..., [11, 12], :].mean(-2, keepdim=True)
    return relative + target[..., None, 34:]


def ridge_adapt(support_z, support_y, query_z, prior_head, ridge=0.1):
    """MAP residual readout; unpenalized intercept, prior weight precision nλ.

    The tiny solve uses CPU float64 for supported, stable autograd on CPU/MPS.
    Transfers remain in the computation graph. Query labels are never inputs.
    """
    if (
        support_z.ndim != 2
        or query_z.ndim != 2
        or support_y.ndim != 2
        or len(support_z) < 2
        or len(support_z) != len(support_y)
        or support_z.shape[1] != query_z.shape[1]
        or ridge <= 0
    ):
        raise ValueError(
            "Compatible support/query features and positive ridge required"
        )
    support_prior, query_prior = prior_head(support_z), prior_head(query_z)
    source_device, source_dtype = query_z.device, query_z.dtype
    # Explicit order matters: MPS cannot create a float64 intermediate.
    zs = support_z.to(device="cpu").to(dtype=torch.float64)
    zq = query_z.to(device="cpu").to(dtype=torch.float64)
    residual = (support_y - support_prior).to(device="cpu").to(dtype=torch.float64)
    zm, rm = zs.mean(0, keepdim=True), residual.mean(0, keepdim=True)
    xc, yc = zs - zm, residual - rm
    matrix = xc.T @ xc + len(zs) * ridge * torch.eye(zs.shape[1], dtype=zs.dtype)
    delta = torch.linalg.solve(matrix, xc.T @ yc)
    adjustment = (
        ((zq - zm) @ delta + rm).to(dtype=source_dtype).to(device=source_device)
    )
    return query_prior + adjustment


class CausalResidual(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv1d(64, 64, 3, dilation=dilation)
        self.norm = nn.LayerNorm(64)

    def forward(self, x):
        y = self.conv(F.pad(x.transpose(1, 2), (2 * self.dilation, 0))).transpose(1, 2)
        return x + F.silu(self.norm(y))


class CalibratedPose(nn.Module):
    """Same encoder/prior architecture for ordinary ERM and episodic training."""

    def __init__(self):
        super().__init__()
        self.spectral = nn.Sequential(
            nn.Conv1d(12, 24, 7, stride=3, padding=3),
            nn.SiLU(),
            nn.Conv1d(24, 32, 3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.frame = nn.Sequential(nn.Linear(32 * 19, 64), nn.LayerNorm(64), nn.SiLU())
        self.temporal = nn.Sequential(*(CausalResidual(d) for d in (1, 2, 4)))
        self.embedding = nn.Sequential(nn.Linear(64, 32), nn.LayerNorm(32))
        self.prior = nn.Linear(32, 36)
        nn.init.zeros_(self.prior.weight)
        nn.init.zeros_(self.prior.bias)
        self.register_buffer("feature_mean", torch.zeros(12, 1))
        self.register_buffer("feature_scale", torch.ones(12, 1))
        self.register_buffer("target_mean", torch.zeros(36))
        self.register_buffer("target_scale", torch.ones(36))

    def fit_source_stats(self, feature_mean, feature_scale, target_mean, target_scale):
        for key, value in (
            ("feature_mean", feature_mean),
            ("feature_scale", feature_scale),
            ("target_mean", target_mean),
            ("target_scale", target_scale),
        ):
            tensor = torch.as_tensor(value, dtype=torch.float32)
            if (
                tensor.shape != getattr(self, key).shape
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError(f"Invalid source-only statistic: {key}")
            if "scale" in key and not torch.all(tensor > 0):
                raise ValueError("Source scales must be positive")
            getattr(self, key).copy_(tensor)

    def encode(self, x):
        if x.ndim != 4 or x.shape[2:] != (12, 114) or not 1 <= x.shape[1] <= 32:
            raise ValueError("Expected CSI observations[N,T<=32,12,114]")
        n, t = x.shape[:2]
        x = (x - self.feature_mean) / self.feature_scale
        x = self.spectral(x.reshape(n * t, 12, 114)).flatten(1)
        x = self.frame(x).reshape(n, t, 64)
        return self.embedding(self.temporal(x)[:, -1])

    def forward(self, x):
        return self.prior(self.encode(x))

    def standardize_targets(self, pose):
        return (pack_pose(pose) - self.target_mean) / self.target_scale

    def decode_targets(self, standardized):
        return unpack_pose(standardized * self.target_scale + self.target_mean)

    def predict_adapted(self, support_x, support_y, query_x, ridge=0.1):
        combined = self.encode(torch.cat((support_x, query_x), dim=0))
        return ridge_adapt(
            combined[: len(support_x)],
            support_y,
            combined[len(support_x) :],
            self.prior,
            ridge,
        )
