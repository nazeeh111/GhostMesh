"""Frequency-preserving temporal CSI filters, an unvalidated research model.

Packet moments avoid assuming a calibrated intra-frame packet clock. Inputs
contain receiver-relative log ratios and circular phase differences; arbitrary
common packet gain/phase cancel before aggregation. Static receiver biases can
remain. No physical velocity, angle, range or three-dimensional pose is implied.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ghostmesh.rf_features import quotient_features


def packet_moments(raw):
    """Map typed RF[T,6,114,10] to per-frame moments[T,12,114].

    Each receiver pair retains its own subcarrier indices. Population standard
    deviations use ten packet columns within that frame, never another frame.
    Existing quotient missing-pair guards apply before neutral imputation.
    """
    x = np.asarray(raw)
    if (
        x.dtype.kind not in "iuf"
        or x.ndim != 4
        or x.shape[1:] != (6, 114, 10)
        or len(x) == 0
    ):
        raise ValueError("Real typed RF[T,6,114,10] required")
    q = quotient_features(x[:, :3], x[:, 3:], units="log_magnitude_db")
    return np.concatenate((q.mean(-1), q.std(-1)), axis=1).astype(np.float32)


class MotionFilter(nn.Module):
    """Keep frequency/time locations until the pose readout, without global pooling.

    Both channels contain only a supplied eight-frame causal history. The
    explicit difference is relative to its first frame. The snapshot control
    repeats the current frame eight times with the exact same parameter count.
    """

    def __init__(self, input_mean, input_scale, pose_mean, pose_scale):
        super().__init__()
        for name, value, expected in (
            ("input_mean", input_mean, (12, 114)),
            ("input_scale", input_scale, (12, 114)),
            ("pose_mean", pose_mean, (17, 2)),
            ("pose_scale", pose_scale, (17, 2)),
        ):
            if (
                value.shape != expected
                or value.is_complex()
                or not value.is_floating_point()
                or not bool(torch.isfinite(value).all())
                or (name.endswith("scale") and bool((value <= 0).any()))
            ):
                raise ValueError("Finite real source statistic required: " + name)
            self.register_buffer(name, value.detach().clone())
        self.filters = nn.Sequential(
            nn.Conv2d(24, 24, (3, 7), padding=(1, 3), stride=(1, 2)),
            nn.GroupNorm(6, 24),
            nn.GELU(),
            nn.Conv2d(24, 32, (3, 5), padding=(1, 2), stride=(1, 2)),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 29, 128),
            # Without this normalization, a real source pilot drove almost
            # every dense activation below-5 and killed CSI-dependent gradients.
            # Normalize before GELU so a shared negative shift cannot do that.
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 34),
        )

    def forward(self, histories, *, mode="temporal"):
        if histories.ndim != 4 or histories.shape[1:] != (8, 12, 114):
            raise ValueError("Eight causal moment frames[N,8,12,114] required")
        if mode == "snapshot":
            histories = histories[:, -1:].expand(-1, 8, -1, -1)
        elif mode == "history_permuted":
            histories = histories[:, [6, 4, 2, 0, 5, 3, 1, 7]]
        elif mode != "temporal":
            raise ValueError("Unknown filter ablation")
        z = (histories - self.input_mean) / self.input_scale
        # Time is the first spatial axis; frequency is the second. Packet
        # positions have already been aggregated independently per frame.
        z = torch.cat((z, z - z[:, :1]), dim=2).transpose(1, 2)
        pose = self.filters(z).reshape(-1, 17, 2) * self.pose_scale + self.pose_mean
        return pose - pose[:, [11, 12]].mean(1, keepdim=True)
