"""Dimensionless distance geometry of supplied17-point annotation arrays.

This is a training-target transform, not a 3D reconstruction or a COCO remap.
It keeps the supplied point indices and removes translation, orthogonal view
transformations (including reflection), and overall positive isotropic scale.
Orientation and chirality are consequently unavailable from this target alone.
"""

from __future__ import annotations

import numpy as np

PAIR_I, PAIR_J = np.triu_indices(17, 1)


def canonical_distances(points):
    raw = np.asarray(points)
    if (
        raw.dtype.kind not in "iuf"
        or raw.ndim < 2
        or raw.shape[-2:] != (17, 3)
        or not raw.size
        or not np.isfinite(raw).all()
    ):
        raise ValueError("Finite real annotation arrays[...,17,3] required")
    with np.errstate(over="ignore", invalid="ignore"):
        x = raw.astype(np.float64, copy=False)
        relative = x - x[..., :1, :]
    if not np.isfinite(relative).all():
        raise ValueError("Point spread is not representable in float64")
    # Remove an arbitrary origin before scaling, preserving small clouds with
    # large offsets. Scale before squares to avoid underflow/overflow in RMS.
    bound = np.max(np.abs(relative), axis=(-2, -1), keepdims=True)
    if (bound <= 0).any():
        raise ValueError("Collapsed point cloud has no defined normalized shape")
    relative = relative / bound
    centered = relative - relative.mean(-2, keepdims=True)
    radius = np.sqrt(np.square(centered).sum(-1).mean(-1, keepdims=True))[..., None]
    if not np.isfinite(radius).all() or (radius <= 0).any():
        raise ValueError("Undefined point-cloud radius")
    normalized = centered / radius
    delta = normalized[..., PAIR_I, :] - normalized[..., PAIR_J, :]
    distances = np.sqrt(np.square(delta).sum(-1))
    if distances.shape != raw.shape[:-2] + (136,) or not np.isfinite(distances).all():
        raise ValueError("Invalid canonical shape descriptor")
    return distances
