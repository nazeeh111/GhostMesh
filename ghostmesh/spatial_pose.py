"""Joint-specific readout of frozen CSI patch tokens.

This module makes no transfer or confidence claim. Its output is hip-relative
2D only. A single pose potential defines both pose and displacement, so an
identical observation cannot produce a nonzero displacement.
"""

from __future__ import annotations

import torch
from torch import nn


class SpatialPose(nn.Module):
    def __init__(self, mean, scale, *, global_pool=False):
        super().__init__()
        if (
            mean.shape != (17, 2)
            or scale.shape != (17, 2)
            or not torch.isfinite(mean).all()
            or not torch.isfinite(scale).all()
            or (scale <= 0).any()
        ):
            raise ValueError("Finite17joint mean and positive scale required")
        self.global_pool = global_pool
        self.register_buffer("target_mean", mean.detach().clone())
        self.register_buffer("target_scale", scale.detach().clone())
        self.query = nn.Parameter(torch.randn(1, 17, 48) * 0.02)
        self.attention = nn.MultiheadAttention(48, 4, dropout=0, batch_first=True)
        self.norm = nn.LayerNorm(48)
        self.output = nn.Sequential(nn.Linear(48, 64), nn.GELU(), nn.Linear(64, 2))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[1:] != (95, 48):
            raise ValueError("Expected causal pooled tokens[N,95,48]")
        if self.global_pool:
            # Same learned parameter count and head; only remove patch detail.
            tokens = tokens.mean(1, keepdim=True).expand(-1, 95, -1)
        queries = self.query.expand(len(tokens), -1, -1)
        attended, _ = self.attention(queries, tokens, tokens, need_weights=False)
        raw = self.output(self.norm(attended + queries))
        pose = raw * self.target_scale + self.target_mean
        return pose - pose[:, [11, 12]].mean(1, keepdim=True)


def pose_objective(
    current, previous, target, past_target, pose_scale, delta_scale, mode
):
    if mode not in {"motion", "absolute"}:
        raise ValueError("Unknown objective")
    if not (
        current.shape == previous.shape == target.shape == past_target.shape
        and current.ndim == 3
        and current.shape[1:] == (17, 2)
        and pose_scale.shape == delta_scale.shape == (17, 2)
    ):
        raise ValueError("Matching17joint endpoint pairs and scales required")
    absolute = 0.5 * (
        ((current - target) / pose_scale).square().mean()
        + ((previous - past_target) / pose_scale).square().mean()
    )
    if mode == "absolute":
        return absolute
    motion = (
        (((current - previous) - (target - past_target)) / delta_scale).square().mean()
    )
    return 0.25 * absolute + 0.75 * motion
