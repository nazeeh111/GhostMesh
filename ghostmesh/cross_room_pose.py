"""Continuous pose-similarity supervision across different SOURCE rooms.

The forward model remains RF-only. Source-room labels select loss pairs during
training; they never enter the model or select a query pose template.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def cross_room_alignment(code, pose, rooms, pose_scale, *, permutation=None):
    n = len(code)
    if (
        code.ndim != 2
        or pose.shape != (n, 17, 2)
        or rooms.shape != (n,)
        or pose_scale.shape != (17, 2)
        or bool((pose_scale <= 0).any())
    ):
        raise ValueError("Matching SOURCE code/pose/room/scale required")
    mask = rooms[:, None] != rooms[None]
    if not bool(mask.any(1).all()):
        raise ValueError("Each anchor requires another SOURCE room")
    teacher = pose.detach()
    if permutation is not None:
        if (
            permutation.shape != (n,)
            or permutation.dtype != torch.int64
            or not torch.equal(
                permutation.sort().values, torch.arange(n, device=code.device)
            )
            or not torch.equal(rooms[permutation], rooms)
        ):
            raise ValueError("Within-SOURCE-room teacher permutation required")
        teacher = teacher[permutation]
    y = (teacher / pose_scale).flatten(1)
    distance = (y[:, None] - y[None]).square().mean(-1)
    target = (-distance / (2 * 0.5**2)).masked_fill(~mask, -torch.inf).softmax(1)
    z = F.normalize(code, dim=1)
    log_probability = ((z @ z.T) / 0.1).masked_fill(~mask, -torch.inf).log_softmax(1)
    return -(target * log_probability.masked_fill(~mask, 0)).sum(1).mean()
