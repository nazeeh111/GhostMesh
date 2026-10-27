"""Packet-preserving CSI encoder, independently implemented for v2 experiments.

The packet-column order is supplied by MM-Fi. It is not a calibrated physical
time/frequency axis. This module predicts 2D coordinates only.
"""

from __future__ import annotations

import torch
from torch import nn
from .models import CausalBlock


class PacketPose(nn.Module):
    def __init__(self, temporal=True, width=64):
        super().__init__()
        self.temporal = temporal
        self.spatial = nn.Sequential(
            nn.Conv2d(3, 16, (5, 3), stride=(3, 1), padding=(1, 1)),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 32, (3, 3), stride=(2, 1), padding=(1, 1)),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
        )
        self.projection = nn.Sequential(
            nn.Linear(32 * 19, width), nn.LayerNorm(width), nn.SiLU()
        )
        self.context = (
            nn.Sequential(*(CausalBlock(width, d, "direct") for d in (1, 2, 4)))
            if temporal
            else nn.Identity()
        )
        self.head = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 51)
        )

    def initialize_from_source(self, mean, log_sigma):
        mean = torch.as_tensor(mean).clamp(1e-5, 1 - 1e-5)
        sigma = torch.as_tensor(log_sigma)
        if mean.shape != (17, 2) or sigma.shape != (17,):
            raise ValueError("Source statistics must have COCO17 shapes")
        nn.init.zeros_(self.head[-1].weight)
        with torch.no_grad():
            self.head[-1].bias.copy_(
                torch.cat((torch.logit(mean), sigma[:, None]), -1).flatten()
            )

    def forward(self, x):
        if x.ndim != 5 or x.shape[2:] != (3, 114, 10) or x.shape[1] < 1:
            raise ValueError("Expected preprocessed [B,T,3,114,10] CSI")
        if not self.temporal:
            x = x[:, -1:]
        batch, length = x.shape[:2]
        z = self.spatial(x.reshape(batch * length, 3, 114, 10))
        z = self.projection(z.mean(-1).flatten(1)).reshape(batch, length, -1)
        z = self.context(z.transpose(1, 2)).transpose(1, 2)[:, -1]
        raw = self.head(z).reshape(-1, 17, 3)
        return {
            "pose": raw[..., :2].sigmoid(),
            "log_sigma": raw[..., 2].clamp(-6.0, 1.0),
        }
