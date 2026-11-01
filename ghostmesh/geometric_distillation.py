"""Source annotation geometry guides CSI representation learning.

The model's only inference input is a causal CSI history. Geometry targets and
room labels are training-only. The decoded skeleton is hip-relative2D; the
auxiliary distances do not reconstruct orientation, chirality or3D coordinates.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GeometricStudent(nn.Module):
    def __init__(self, encoder, pose_mean, pose_scale, geometry_mean, geometry_scale):
        super().__init__()
        expected = {
            "pose_mean": (17, 2),
            "pose_scale": (17, 2),
            "geometry_mean": (136,),
            "geometry_scale": (136,),
        }
        for name, value in (
            ("pose_mean", pose_mean),
            ("pose_scale", pose_scale),
            ("geometry_mean", geometry_mean),
            ("geometry_scale", geometry_scale),
        ):
            if (
                value.shape != expected[name]
                or value.is_complex()
                or not torch.isfinite(value).all()
                or (name.endswith("scale") and (value <= 0).any())
            ):
                raise ValueError("Invalid real finite source statistic: " + name)
            self.register_buffer(name, value.detach().clone())
        self.encoder = encoder
        self.pose_head = nn.Linear(512, 34)
        self.geometry_head = nn.Linear(512, 136)
        self.relation_head = nn.Linear(512, 64)
        for head in (self.pose_head, self.geometry_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, histories):
        feature = self.encoder.history(histories)
        if feature.shape != (len(histories), 512):
            raise ValueError("Expected eight64D causal frame embeddings")
        pose = (
            self.pose_head(feature).reshape(-1, 17, 2) * self.pose_scale
            + self.pose_mean
        )
        pose = pose - pose[:, [11, 12]].mean(1, keepdim=True)
        return dict(
            pose=pose,
            geometry=self.geometry_head(feature) * self.geometry_scale
            + self.geometry_mean,
            relation=F.normalize(self.relation_head(feature), dim=1),
        )


def paired_objective(prediction, target, absolute_scale, delta_scale):
    if (
        prediction.shape != target.shape
        or prediction.ndim < 2
        or len(prediction) % 2
        or absolute_scale.shape != prediction.shape[1:]
        or delta_scale.shape != prediction.shape[1:]
    ):
        raise ValueError("Matching current/past pairs and source scales required")
    current, previous = prediction.chunk(2)
    y, z = target.chunk(2)
    absolute = ((prediction - target) / absolute_scale).square().mean()
    delta = (((current - previous) - (y - z)) / delta_scale).square().mean()
    return 0.5 * absolute + 0.5 * delta


def relational_objective(
    code,
    geometry_target,
    source_mean,
    source_scale,
    rooms,
    *,
    teacher_temperature=0.5,
    student_temperature=0.2,
):
    if (
        code.ndim != 2
        or code.shape[1] != 64
        or geometry_target.shape != (len(code), 136)
        or source_mean.shape != (136,)
        or source_scale.shape != (136,)
        or rooms.shape != (len(code),)
        or rooms.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(
            "64D student codes,136D teacher geometry and integer room labels required"
        )
    if teacher_temperature <= 0 or student_temperature <= 0:
        raise ValueError("Positive teacher/student temperatures required")
    allowed = rooms[:, None] != rooms[None, :]
    if not bool(allowed.any(1).all()):
        raise ValueError("Every training anchor needs a different-room candidate")
    teacher = F.normalize(
        (geometry_target.detach() - source_mean.detach()) / source_scale.detach(), dim=1
    )
    distance = (teacher[:, None] - teacher[None]).square().sum(-1)
    teacher_log = F.log_softmax(
        (-distance / (2 * teacher_temperature**2)).masked_fill(~allowed, -torch.inf),
        dim=1,
    )
    student = F.normalize(code, dim=1)
    student_log = F.log_softmax(
        (student @ student.T / student_temperature).masked_fill(~allowed, -torch.inf),
        dim=1,
    )
    # Index the allowed pairs before multiplication:0*(-inf-(-inf)) is NaN.
    return (
        teacher_log.exp()[allowed] * (teacher_log[allowed] - student_log[allowed])
    ).sum() / len(code)


def distillation_objective(
    result,
    pose_target,
    geometry_target,
    pose_delta_scale,
    geometry_delta_scale,
    source_pose_scale,
    source_geometry_mean,
    source_geometry_scale,
    rooms,
    arm,
    *,
    geometry_weight=0.25,
    relation_weight=0.05,
    teacher_temperature=0.5,
    student_temperature=0.2,
):
    if arm not in {
        "pose_only",
        "pose_geometry",
        "pose_geometry_relational",
        "pose_geometry_permuted",
    }:
        raise ValueError("Unknown source distillation arm")
    pose = paired_objective(
        result["pose"], pose_target, source_pose_scale, pose_delta_scale
    )
    loss = pose
    if arm != "pose_only":
        loss = loss + geometry_weight * paired_objective(
            result["geometry"],
            geometry_target,
            source_geometry_scale,
            geometry_delta_scale,
        )
    if arm in {"pose_geometry_relational", "pose_geometry_permuted"}:
        loss = loss + relation_weight * relational_objective(
            result["relation"],
            geometry_target,
            source_geometry_mean,
            source_geometry_scale,
            rooms,
            teacher_temperature=teacher_temperature,
            student_temperature=student_temperature,
        )
    return loss
