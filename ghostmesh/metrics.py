"""Transparent 2D pose metrics with explicit coordinate and scale conventions."""

from __future__ import annotations

import numpy as np


def pose_metrics(
    prediction,
    target,
    valid,
    log_sigma=None,
    threshold: float = 0.2,
    confidence_radius: float = 0.05,
    bins: int = 10,
) -> dict:
    """PCK/NJE use each frame's valid GT bounding-box diagonal.

    Degenerate boxes and frames with fewer than two valid joints are excluded.
    NJE is mean Euclidean distance / GT box diagonal, not an MPJPE in mm.
    Confidence ECE uses success within an absolute normalized-image radius;
    its probability is the predicted isotropic Gaussian radial mass.
    """
    pred, truth = np.asarray(prediction, float), np.asarray(target, float)
    mask = np.asarray(valid, bool).copy()
    if (
        pred.shape != truth.shape
        or pred.ndim != 3
        or pred.shape[-1] != 2
        or mask.shape != pred.shape[:-1]
    ):
        raise ValueError("Expected matching [N,J,2] poses and [N,J] validity")
    if threshold <= 0 or confidence_radius <= 0 or bins < 1:
        raise ValueError("Metric thresholds and bin count must be positive")
    mask &= np.isfinite(truth).all(-1)
    if not np.isfinite(pred[mask]).all():
        raise ValueError("Non-finite prediction on a valid joint")
    lower = np.min(np.where(mask[..., None], truth, np.inf), axis=1)
    upper = np.max(np.where(mask[..., None], truth, -np.inf), axis=1)
    scale = np.linalg.norm(upper - lower, axis=-1)
    usable = (mask.sum(1) >= 2) & np.isfinite(scale) & (scale > 1e-6)
    mask &= usable[:, None]
    safe_truth = np.where(np.isfinite(truth), truth, 0)
    distance = np.linalg.norm(pred - safe_truth, axis=-1)
    error = distance / np.where(usable, scale, 1)[:, None]
    n = int(mask.sum())
    per_frame = [
        float(error[i, mask[i]].mean()) if mask[i].any() else None
        for i in range(len(pred))
    ]
    result = {
        "pck": float((error[mask] <= threshold).mean()) if n else None,
        "pck_threshold": threshold,
        "pck_curve": {
            f"{value:g}": float((error[mask] <= value).mean()) if n else None
            for value in sorted({0.05, 0.1, 0.2, float(threshold)})
        },
        "normalized_joint_error": float(error[mask].mean()) if n else None,
        "normalizer": "per-frame valid ground-truth 2D bounding-box diagonal",
        "coordinate_units": "normalized image coordinates",
        "valid_joints": n,
        "valid_frames": int(usable.sum()),
        "excluded_frames": int((~usable).sum()),
        "per_frame_nje": per_frame,
        "per_joint": [
            {
                "joint": j,
                "count": int(mask[:, j].sum()),
                "pck": float((error[mask[:, j], j] <= threshold).mean())
                if mask[:, j].any()
                else None,
                "nje": float(error[mask[:, j], j].mean()) if mask[:, j].any() else None,
            }
            for j in range(pred.shape[1])
        ],
    }
    if log_sigma is not None:
        ls = np.asarray(log_sigma, float)
        if ls.shape != mask.shape or not np.isfinite(ls[mask]).all():
            raise ValueError("Expected finite [N,J] log_sigma on valid joints")
        confidence = -np.expm1(
            -0.5 * confidence_radius**2 * np.exp(np.clip(-2 * ls[mask], -700, 700))
        )
        correct = (distance[mask] <= confidence_radius).astype(float)
        rows, ece = [], 0.0
        for i in range(bins):
            selected = (confidence >= i / bins) & (
                (confidence < (i + 1) / bins) if i < bins - 1 else (confidence <= 1)
            )
            count = int(selected.sum())
            accuracy = float(correct[selected].mean()) if count else None
            mean_confidence = float(confidence[selected].mean()) if count else None
            if count:
                ece += count / max(n, 1) * abs(accuracy - mean_confidence)
            rows.append(
                {
                    "lower": i / bins,
                    "upper": (i + 1) / bins,
                    "count": count,
                    "accuracy": accuracy,
                    "confidence": mean_confidence,
                }
            )
        result["calibration"] = {
            "ece": float(ece) if n else None,
            "radius": confidence_radius,
            "event": "joint Euclidean error <= radius in normalized image coordinates",
            "bins": rows,
            "status": "evaluated, not post-hoc calibrated",
        }
    return result
