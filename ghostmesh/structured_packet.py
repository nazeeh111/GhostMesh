"""Small original 2D packet/token/skeleton probe, informed by standard attention
and graph models plus RePos factorization (https://arxiv.org/abs/2607.02986).
No upstream code/weights, 3D output, physical packet-time or novelty claim.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

EDGES = (
    (0, 1),
    (1, 2),
    (3, 5),
    (4, 6),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


class TokenBlock(nn.Module):
    def __init__(self, width=96):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
        )

    def forward(self, x):
        z = self.norm(x)
        x = x + self.attention(z, z, z, need_weights=False)[0]
        return x + self.ff(x)


class SkeletonBlock(nn.Module):
    def __init__(self, width=96):
        super().__init__()
        adjacency = torch.eye(17)
        for a, b in EDGES:
            adjacency[a, b] = adjacency[b, a] = 1
        self.register_buffer("adjacency", adjacency / adjacency.sum(-1, keepdim=True))
        self.self_linear = nn.Linear(width, width)
        self.neighbor_linear = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        neighbors = torch.matmul(self.adjacency, x)
        return x + F.silu(
            self.norm(self.self_linear(x) + self.neighbor_linear(neighbors))
        )


class StructuredPacketPose(nn.Module):
    """Same capacity/readout components for direct and factorized objectives.

    Only the last endpoint of a complete eight-frame past/current window is
    predicted. Input shape never includes future frames or target annotations.
    """

    def __init__(self, factorized=False, width=96, channels=3):
        super().__init__()
        if channels not in {3, 7}:
            raise ValueError(
                "Declare 3 amplitude or 7 amplitude/relative-phase channels"
            )
        self.factorized, self.channels = factorized, channels
        self.cnn = nn.Sequential(
            nn.Conv2d(8 * channels, 32, (5, 3), stride=(3, 2), padding=(1, 1)),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, (3, 3), stride=(2, 1), padding=(1, 1)),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, width, 1),
        )
        self.positions = nn.Parameter(torch.randn(1, 95, width) * 0.02)
        self.tokens = nn.Sequential(TokenBlock(width), TokenBlock(width))
        self.queries = nn.Parameter(torch.randn(1, 17, width) * 0.02)
        self.query_norm, self.token_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.cross = nn.MultiheadAttention(width, 4, batch_first=True)
        self.query_ff = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
        )
        self.skeleton = nn.Sequential(SkeletonBlock(width), SkeletonBlock(width))
        self.joint_head, self.root_head = nn.Linear(width, 3), nn.Linear(width, 3)
        for key, shape in (
            ("absolute_mean", (17, 2)),
            ("absolute_sigma", (17,)),
            ("relative_mean", (17, 2)),
            ("relative_sigma", (17,)),
            ("root_mean", (1, 2)),
            ("root_sigma", (1,)),
        ):
            self.register_buffer(key, torch.zeros(shape))
        for head in (self.joint_head, self.root_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def initialize_from_source(self, pose):
        y = torch.as_tensor(pose, dtype=torch.float32)
        if y.ndim != 3 or y.shape[1:] != (17, 2) or not torch.isfinite(y).all():
            raise ValueError("Finite source [N,17,2] targets required")
        root = y[:, [11, 12]].mean(1, keepdim=True)
        with torch.no_grad():
            for name, value in (
                ("absolute", y),
                ("root", root),
                ("relative", y - root),
            ):
                mean = value.mean(0)
                sigma2 = (value - mean).square().sum(-1).mean(0) / 2
                getattr(self, name + "_mean").copy_(mean)
                getattr(self, name + "_sigma").copy_(0.5 * sigma2.clamp_min(1e-6).log())

    def forward(self, x):
        if x.ndim != 5 or x.shape[1:] != (8, self.channels, 114, 10):
            raise ValueError(
                f"Expected preprocessed [B,8,{self.channels},114,10] causal windows"
            )
        z = self.cnn(x.reshape(len(x), 8 * self.channels, 114, 10))
        z = self.tokens(z.flatten(2).transpose(1, 2) + self.positions)
        q = self.queries.expand(len(x), -1, -1)
        q = (
            q
            + self.cross(
                self.query_norm(q),
                self.token_norm(z),
                self.token_norm(z),
                need_weights=False,
            )[0]
        )
        q = self.skeleton(q + self.query_ff(q))
        root_features = q[:, [11, 12]].mean(1, keepdim=True)
        # Root targets never backpropagate into the relative structure encoder/decoder.
        root = self.root_head(
            root_features.detach() if self.factorized else root_features
        )
        joint = self.joint_head(q)
        if not self.factorized:
            return {
                "pose": self.absolute_mean + joint[..., :2] + root[..., :2],
                "log_sigma": (self.absolute_sigma + joint[..., 2] + root[..., 2]).clamp(
                    -6, 1
                ),
            }
        relative = self.relative_mean + joint[..., :2]
        relative = relative - relative[:, [11, 12]].mean(1, keepdim=True)
        position = self.root_mean + root[..., :2]
        relative_sigma = (self.relative_sigma + joint[..., 2]).clamp(-6, 1)
        root_sigma = (self.root_sigma + root[..., 2]).clamp(-6, 1)
        return {
            "pose": relative + position,
            "log_sigma": 0.5 * torch.logaddexp(2 * relative_sigma, 2 * root_sigma),
            "relative": relative,
            "root": position,
            "relative_log_sigma": relative_sigma,
            "root_log_sigma": root_sigma,
        }


def gaussian_nll(pose, log_sigma, target):
    return (
        0.5 * (pose - target).square().sum(-1) * torch.exp(-2 * log_sigma)
        + 2 * log_sigma
    ).mean()


def structured_loss(out, target, factorized=False):
    if not factorized:
        return gaussian_nll(out["pose"], out["log_sigma"], target)
    scale = torch.linalg.vector_norm(target.amax(1) - target.amin(1), dim=-1).clamp_min(
        1e-6
    )
    root = target[:, [11, 12]].mean(1, keepdim=True)
    relative = target - root

    def term(prediction, sigma, truth):
        residual = (prediction - truth) / scale[:, None, None]
        geometry = F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=0.1)
        return geometry + 0.1 * gaussian_nll(prediction.detach(), sigma, truth)

    geometry = term(out["relative"], out["relative_log_sigma"], relative) + term(
        out["root"], out["root_log_sigma"], root
    )
    a, b = zip(*EDGES)
    predicted_bone = torch.linalg.vector_norm(
        out["relative"][:, a] - out["relative"][:, b], dim=-1
    )
    target_bone = torch.linalg.vector_norm(relative[:, a] - relative[:, b], dim=-1)
    bone = F.smooth_l1_loss(
        predicted_bone / scale[:, None], target_bone / scale[:, None], beta=0.1
    )
    return geometry + 0.1 * bone
