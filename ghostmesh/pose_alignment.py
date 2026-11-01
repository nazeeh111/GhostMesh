"""Source-only cross-person pose-similarity supervision for CSI research.

Pose references and person IDs are arguments to a TRAIN loss only. Forward
inference accepts RF history alone. A prototype readout constrains predictions
to convex combinations of supplied TRAIN 2D body poses, not a 3D body model.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PoseAlignedStudent(nn.Module):
    def __init__(self, input_mean, input_scale, prototypes, prior):
        super().__init__()
        for name, value, shape in (
            ("input_mean", input_mean, (12, 114)),
            ("input_scale", input_scale, (12, 114)),
            ("prototypes", prototypes, (128, 17, 2)),
            ("prior", prior, (128,)),
        ):
            if (
                value.shape != shape
                or not value.is_floating_point()
                or not bool(torch.isfinite(value).all())
                or (name in {"input_scale", "prior"} and bool((value <= 0).any()))
            ):
                raise ValueError("Finite TRAIN statistic required: " + name)
            self.register_buffer(name, value.detach().clone())
        self.filters = nn.Sequential(
            nn.Conv2d(24, 24, (5, 7), padding=(2, 3), stride=(2, 2)),
            nn.GroupNorm(6, 24),
            nn.GELU(),
            nn.Conv2d(24, 32, (5, 5), padding=2, stride=2),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 29, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.readout = nn.Linear(128, 128)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, histories):
        if histories.ndim != 4 or histories.shape[1:] != (32, 12, 114):
            raise ValueError("RF histories[N,32,12,114] required")
        absolute = (histories - self.input_mean) / self.input_scale
        dynamic = histories - histories.mean(1, keepdim=True)
        # The entire supplied history is available at the RF input clock.
        # The output timestamp is separately declared four frames earlier.
        scale = dynamic.square().mean(1, keepdim=True).sqrt().clamp_min(0.05)
        x = torch.cat((absolute, dynamic / scale), dim=2).transpose(1, 2)
        code = self.filters(x)
        weights = (self.readout(code) + self.prior.log()).softmax(1)
        pose = (weights @ self.prototypes.flatten(1)).reshape(-1, 17, 2)
        pose = pose - pose[:, [11, 12]].mean(1, keepdim=True)
        return dict(pose=pose, code=code, prototype_weights=weights)


def cross_person_alignment(code, pose, people, pose_scale, *, permutation=None):
    """Match RF-neighbor probabilities to continuous TRAIN-pose similarities.

    Only pairs of different source people contribute. This prevents same-person
    channel identity from satisfying the similarity target. The permutation arm
    changes teacher correspondences within the same person's TRAIN batch, leaving
    the ordinary pose/difference supervision intact. No action class is supplied.
    """
    n = len(code)
    if (
        code.ndim != 2
        or pose.shape != (n, 17, 2)
        or people.shape != (n,)
        or pose_scale.shape != (17, 2)
        or bool((pose_scale <= 0).any())
    ):
        raise ValueError("Matching TRAIN code/pose/person/scale required")
    mask = people[:, None] != people[None]
    if not bool(mask.any(1).all()):
        raise ValueError("Each anchor needs a different source person")
    teacher = pose.detach()
    if permutation is not None:
        if (
            permutation.shape != (n,)
            or permutation.dtype != torch.int64
            or not torch.equal(
                permutation.sort().values, torch.arange(n, device=code.device)
            )
            or not torch.equal(people[permutation], people)
        ):
            raise ValueError("Within-person TRAIN permutation required")
        teacher = teacher[permutation]
    y = (teacher / pose_scale).flatten(1)
    distance = (y[:, None] - y[None]).square().mean(-1)
    target = (-distance / (2 * 0.5**2)).masked_fill(~mask, -torch.inf).softmax(1)
    z = F.normalize(code, dim=1)
    logits = ((z @ z.T) / 0.1).masked_fill(~mask, -torch.inf)
    log_probability = logits.log_softmax(1)
    # Avoid 0*(-inf), preserving exact exclusion and finite gradients.
    return -(target * log_probability.masked_fill(~mask, 0)).sum(1).mean()
