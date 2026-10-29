"""Original 2D packet-attention model with independent articulation/root paths.

CSI quotient inspiration: FarSense https://arxiv.org/abs/1907.03994.
Relative/root separation has prior art including RePos https://arxiv.org/abs/2607.02986.
No copied implementation, calibrated RSSI/AoA, 3D or novelty claim.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .models import CausalBlock

BONES = (
    (0, 1),
    (0, 2),
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
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


class RFPose(nn.Module):
    """Independent causal body/root streams; all-time training or last-only inference."""

    def __init__(self, kernel="native"):
        super().__init__()
        self.packet_encoder = nn.Sequential(
            nn.Linear(798, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
        )
        self.packet_positions = nn.Parameter(torch.randn(1, 10, 64) * 0.02)
        self.packet_attention = nn.Linear(64, 1)
        self.temporal = nn.Sequential(*(CausalBlock(64, d, kernel) for d in (1, 2, 4)))
        self.relative_head = nn.Linear(64, 34)
        self.relative_uncertainty = nn.Linear(64, 17)
        self.root_encoder = nn.Sequential(
            nn.Linear(342, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
        )
        self.root_head, self.root_uncertainty = nn.Linear(64, 2), nn.Linear(64, 1)
        for key, shape in (
            ("relative_mean", (17, 2)),
            ("relative_sigma", (17,)),
            ("root_mean", (1, 2)),
            ("root_sigma", (1,)),
        ):
            self.register_buffer(key, torch.zeros(shape))
        for head in (
            self.relative_head,
            self.relative_uncertainty,
            self.root_head,
            self.root_uncertainty,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def initialize_from_source(self, source_pose):
        y = torch.as_tensor(source_pose, dtype=torch.float32)
        if y.ndim == 4:
            y = y.reshape(-1, 17, 2)
        if y.ndim != 3 or y.shape[1:] != (17, 2) or not torch.isfinite(y).all():
            raise ValueError("Finite source-only [N,17,2] targets required")
        root = y[:, [11, 12]].mean(1, keepdim=True)
        with torch.no_grad():
            for name, value in (("relative", y - root), ("root", root)):
                mean = value.mean(0)
                variance = (value - mean).square().sum(-1).mean(0) / 2
                getattr(self, name + "_mean").copy_(mean)
                getattr(self, name + "_sigma").copy_(
                    0.5 * variance.clamp_min(1e-6).log()
                )

    def forward(self, body, root_input, return_sequence=False):
        if (
            body.ndim != 5
            or not 1 <= body.shape[1] <= 8
            or body.shape[2:] != (7, 114, 10)
            or root_input.shape != (len(body), body.shape[1], 342)
        ):
            raise ValueError(
                "Expected causal body[B,T<=8,7,114,10] and root spectrum[B,T,342]"
            )
        batch, length = body.shape[:2]
        packets = body.permute(0, 1, 4, 2, 3).reshape(batch * length, 10, 798)
        packets = self.packet_encoder(packets) + self.packet_positions
        weights = self.packet_attention(packets).softmax(1)
        frame = (weights * packets).sum(1).reshape(batch, length, 64)
        articulation = self.temporal(frame.transpose(1, 2)).transpose(1, 2)
        relative = self.relative_mean + self.relative_head(articulation).reshape(
            batch, length, 17, 2
        )
        relative = relative - relative[:, :, [11, 12]].mean(2, keepdim=True)
        position = self.root_encoder(root_input)
        root = self.root_mean + self.root_head(position)[:, :, None]
        # Every-time sigma objectives are isolated from both geometry branches.
        relative_sigma = (
            self.relative_sigma + self.relative_uncertainty(articulation.detach())
        ).clamp(-6, 1)
        root_sigma = (self.root_sigma + self.root_uncertainty(position.detach())).clamp(
            -6, 1
        )
        raw = root + relative
        result = {
            "pose": raw.clamp(0, 1),
            "raw_pose": raw,
            "relative": relative,
            "root": root,
            "relative_log_sigma": relative_sigma,
            "root_log_sigma": root_sigma,
            "log_sigma": 0.5 * torch.logaddexp(2 * relative_sigma, 2 * root_sigma),
        }
        return (
            result
            if return_sequence
            else {key: value[:, -1] for key, value in result.items()}
        )


def gaussian(pose, log_sigma, target):
    return (
        0.5 * (pose.detach() - target).square().sum(-1) * torch.exp(-2 * log_sigma)
        + 2 * log_sigma
    ).mean()


def loss_components(prediction, target):
    """Explicit components let software tests inspect gradient boundaries."""
    if target.ndim == 4:
        prediction = {
            key: value.reshape(-1, *value.shape[2:])
            for key, value in prediction.items()
        }
        target = target.reshape(-1, 17, 2)
    scale = torch.linalg.vector_norm(target.amax(1) - target.amin(1), dim=-1).clamp_min(
        1e-6
    )
    root = target[:, [11, 12]].mean(1, keepdim=True)
    relative = target - root

    def geometry(p, y):
        residual = (p - y) / scale[:, None, None]
        return F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=0.1)

    a, b = zip(*BONES)
    predicted_bone = torch.linalg.vector_norm(
        prediction["relative"][:, a] - prediction["relative"][:, b], dim=-1
    )
    target_bone = torch.linalg.vector_norm(relative[:, a] - relative[:, b], dim=-1)
    bone = F.smooth_l1_loss(
        predicted_bone / scale[:, None], target_bone / scale[:, None], beta=0.1
    )
    return {
        "relative_geometry": geometry(prediction["relative"], relative),
        "root_geometry": geometry(prediction["root"], root),
        "bone": bone,
        "uncertainty": gaussian(
            prediction["relative"], prediction["relative_log_sigma"], relative
        )
        + gaussian(prediction["root"], prediction["root_log_sigma"], root),
    }


def rf_loss(prediction, target):
    terms = loss_components(prediction, target)
    return (
        terms["relative_geometry"]
        + terms["root_geometry"]
        + 0.1 * terms["bone"]
        + 0.1 * terms["uncertainty"]
    )
