"""Full-matrix log-magnitude pose encoder with source-person regularization.

Independently implemented affine frame normalization, receiver-gain nuisance
augmentation and gradient reversal. The latter is an established idea from
Ganin et al., JMLR2016, not a new general method. Source people supply nuisance
labels during training only; this is not target-data adaptation or identity
recognition at inference. No upstream implementation or weights are copied.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def normalize_amplitude(raw, *, gain_db=None):
    """Normalize each 3x114x10 log-magnitude frame independently to[0,1].

    Missing values use the finite mean of that frame's packet column. No pose,
    preceding/future frame or fitted domain statistics enter the transform.
    Optional receiver offsets are constant over one supplied causal history;
    they approximate receiver-gain variation, not a validated room simulator.
    """
    x = np.asarray(raw)
    if (
        x.dtype.kind not in "iuf"
        or x.ndim != 5
        or x.shape[1:] != (8, 3, 114, 10)
        or not len(x)
    ):
        raise ValueError("Real log-magnitude histories[N,8,3,114,10] required")
    x = x.astype(np.float64, copy=True)
    if gain_db is not None:
        gain = np.asarray(gain_db)
        if (
            gain.dtype.kind not in "iuf"
            or gain.shape != (len(x), 3)
            or not np.isfinite(gain).all()
        ):
            raise ValueError("Finite receiver gains[N,3] required")
        x += gain[:, None, :, None, None]
    finite = np.isfinite(x)
    if (finite.mean(axis=(2, 3, 4)) < 0.95).any():
        raise ValueError("More than5percent missing magnitude in a frame")
    count = finite.sum(axis=(2, 3), keepdims=True)
    if (count == 0).any():
        raise ValueError("Entire packet column is missing")
    mean = np.where(finite, x, 0).sum(axis=(2, 3), keepdims=True) / count
    x = np.where(finite, x, mean)
    minimum = x.min(axis=(2, 3, 4), keepdims=True)
    span = x.max(axis=(2, 3, 4), keepdims=True) - minimum
    if not np.isfinite(span).all() or (span < 1e-6).any():
        raise ValueError("Constant/nonrepresentable amplitude frame")
    return ((x - minimum) / span).astype(np.float32)


class _Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, coefficient):
        ctx.coefficient = coefficient
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.coefficient * gradient, None


def reverse_gradient(value, coefficient):
    if (
        isinstance(coefficient, bool)
        or not isinstance(coefficient, (int, float))
        or not math.isfinite(coefficient)
        or coefficient < 0
    ):
        raise ValueError("Finite nonnegative reversal coefficient required")
    return _Reverse.apply(value, float(coefficient))


class CausalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.padding = 2 * dilation
        self.depth = nn.Conv1d(width, width, 3, dilation=dilation, groups=width)
        self.point = nn.Conv1d(width, width, 1)
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        branch = self.point(F.gelu(self.depth(F.pad(x, (self.padding, 0)))))
        return self.norm((x + branch).transpose(1, 2)).transpose(1, 2)


class AmplitudeBody(nn.Module):
    def __init__(self, pose_mean, pose_scale):
        super().__init__()
        for name, value in (("pose_mean", pose_mean), ("pose_scale", pose_scale)):
            if (
                value.shape != (17, 2)
                or not value.is_floating_point()
                or value.is_complex()
                or not bool(torch.isfinite(value).all())
                or (name.endswith("scale") and bool((value <= 0).any()))
            ):
                raise ValueError("Finite source pose statistic required")
            self.register_buffer(name, value.detach().clone())
        self.frames = nn.Sequential(
            nn.Conv2d(3, 16, (5, 3), stride=2, padding=(2, 1)),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 32, (5, 3), stride=2, padding=(2, 1)),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 48, (3, 3), stride=2, padding=1),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(48 * 15 * 2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.temporal = nn.Sequential(*[CausalBlock(128, d) for d in (1, 2, 4)])
        self.pose = nn.Linear(128, 34)
        nn.init.zeros_(self.pose.weight)
        nn.init.zeros_(self.pose.bias)

    def forward(self, histories):
        if histories.ndim != 5 or histories.shape[1:] != (8, 3, 114, 10):
            raise ValueError("Eight normalized amplitude frames required")
        frames = self.frames(histories.flatten(0, 1)).reshape(len(histories), 8, 128)
        code = self.temporal(frames.transpose(1, 2))[:, :, -1]
        pose = self.pose(code).reshape(-1, 17, 2) * self.pose_scale + self.pose_mean
        pose = pose - pose[:, [11, 12]].mean(1, keepdim=True)
        return pose, code


class AmplitudeStudent(nn.Module):
    def __init__(self, pose_mean, pose_scale, people=8):
        super().__init__()
        if type(people) is not int or people < 2:
            raise ValueError("At least two source nuisance classes required")
        self.body = AmplitudeBody(pose_mean, pose_scale)
        self.nuisance = nn.Sequential(
            nn.Linear(128, 32), nn.GELU(), nn.Linear(32, people)
        )

    def forward(self, histories, *, reversal=0.0):
        pose, code = self.body(histories)
        return dict(
            pose=pose,
            code=code,
            nuisance_logits=self.nuisance(reverse_gradient(code, reversal)),
        )
