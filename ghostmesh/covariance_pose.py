"""Compact causal articulation learning on receiver-congruence invariants.

Inputs are paired covariance log generalized eigenvalues, not raw amplitude.
A second head retrodicts past articulation from the SAME current causal history;
its labels are never inference inputs. Absolute hip position/3D are not predicted.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def root_relative(pose):
    if pose.shape[-2:] != (17, 2):
        raise ValueError("Expected17two-dimensional joints")
    return pose - pose[..., [11, 12], :].mean(-2, keepdim=True)


class CausalBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv1d(64, 64, 3, dilation=dilation)
        self.norm = nn.LayerNorm(64)

    def forward(self, x):
        value = self.conv(F.pad(x.transpose(1, 2), (2 * self.dilation, 0))).transpose(
            1, 2
        )
        return x + F.silu(self.norm(value))


class CovariancePose(nn.Module):
    def __init__(self):
        super().__init__()
        self.spectral = nn.Sequential(
            nn.Conv1d(18, 24, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(24, 32, 3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.frame = nn.Sequential(nn.Linear(32 * 10, 64), nn.LayerNorm(64), nn.SiLU())
        self.temporal = nn.Sequential(*(CausalBlock(d) for d in (1, 2, 4)))
        self.embedding = nn.Sequential(nn.Linear(64, 32), nn.LayerNorm(32))
        self.current = nn.Linear(32, 34)
        self.past = nn.Linear(32, 34)
        for head in (self.current, self.past):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.register_buffer("feature_mean", torch.zeros(18, 1))
        self.register_buffer("feature_scale", torch.ones(18, 1))
        self.register_buffer("pose_mean", torch.zeros(34))
        self.register_buffer("pose_scale", torch.ones(34))
        self.register_buffer("delta_scale", torch.ones(34))

    def fit_source_stats(
        self, feature_mean, feature_scale, pose_mean, pose_scale, delta_scale
    ):
        for key, value in zip(
            ("feature_mean", "feature_scale", "pose_mean", "pose_scale", "delta_scale"),
            (feature_mean, feature_scale, pose_mean, pose_scale, delta_scale),
        ):
            value = torch.as_tensor(value, dtype=torch.float32)
            if (
                value.shape != getattr(self, key).shape
                or not torch.isfinite(value).all()
                or ("scale" in key and not (value > 0).all())
            ):
                raise ValueError("Invalid source statistic: " + key)
            getattr(self, key).copy_(value)

    def encode(self, x):
        if x.ndim != 4 or x.shape[2:] != (18, 19) or not 1 <= x.shape[1] <= 8:
            raise ValueError("Expected covariance observations[N,T<=8,18,19]")
        n, t = x.shape[:2]
        x = (x - self.feature_mean) / self.feature_scale
        x = self.spectral(x.reshape(n * t, 18, 19)).flatten(1)
        x = self.frame(x).reshape(n, t, 64)
        return self.embedding(self.temporal(x)[:, -1])

    def forward(self, x):
        z = self.encode(x)
        return dict(current=self.current(z), past=self.past(z))

    def standardized_target(self, pose):
        return (pose.flatten(-2) - self.pose_mean) / self.pose_scale

    def decode(self, standardized):
        return root_relative(
            (standardized * self.pose_scale + self.pose_mean).reshape(-1, 17, 2)
        )

    def loss(self, prediction, current, past, arm):
        current_loss = F.mse_loss(
            prediction["current"], self.standardized_target(current)
        )
        if arm == "absolute":
            return 0.5 * current_loss + 0.5 * F.mse_loss(
                prediction["past"], self.standardized_target(past)
            )
        if arm != "motion":
            raise ValueError("Unknown objective")
        delta = (
            self.decode(prediction["current"]) - self.decode(prediction["past"])
        ).flatten(-2)
        target = (current - past).flatten(-2)
        return 0.5 * current_loss + 0.5 * F.mse_loss(
            delta / self.delta_scale, target / self.delta_scale
        )
