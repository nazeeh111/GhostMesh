"""Independent causal amplitude processing; no test-fitted statistics."""

from __future__ import annotations
import numpy as np


def signal_quality(
    csi: np.ndarray, max_missing: float = 0.05
) -> tuple[bool, list[str]]:
    x = np.asarray(csi)
    reasons = []
    if x.ndim != 4 or min(x.shape, default=0) == 0:
        return False, ["invalid_shape"]
    finite = np.isfinite(x)
    if 1 - finite.mean() > max_missing:
        reasons.append("missing_signal")
    good = x[finite]
    if good.size == 0:
        return False, ["missing_signal"]
    if (good < 0).any():
        reasons.append("negative_amplitude")
    if float(np.std(good)) < 1e-7:
        reasons.append("flat_signal")
    return not reasons, reasons


def prepare_window(csi: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Log amplitude and optional per-link robust scaling using this window only.

    The window ends at the target time. Missing entries <=5% are replaced with
    the same window/link's median. Whole missing or flat signals are rejected.
    This is instance normalization, not adaptation on the test dataset.
    """
    ok, reasons = signal_quality(csi)
    if not ok:
        raise ValueError("Rejected CSI window: " + ", ".join(reasons))
    x = np.asarray(csi, dtype=np.float32).copy()
    x[~np.isfinite(x)] = np.nan
    center = np.nanmedian(x, axis=(0, 2, 3), keepdims=True)
    if not np.isfinite(center).all():
        raise ValueError("Rejected CSI window: entire missing link")
    x = np.where(np.isfinite(x), x, center)
    x = np.log1p(x)
    if normalize:
        center = np.median(x, axis=(0, 2, 3), keepdims=True)
        scale = np.median(np.abs(x - center), axis=(0, 2, 3), keepdims=True) * 1.4826
        x = (x - center) / np.maximum(scale, 0.03)
    return np.clip(x, -8, 8).astype(np.float32)


def spectrum(csi: np.ndarray) -> list[list[float]]:
    """Subcarrier-by-time amplitude map, not a Doppler spectrum."""
    x = np.asarray(csi, dtype=np.float32)
    x = np.nan_to_num(np.log1p(np.maximum(x, 0)), nan=0.0, posinf=0.0, neginf=0.0)
    x = x.mean(axis=(1, 3)).T
    lo, hi = np.percentile(x, [5, 95])
    return np.clip((x - lo) / max(float(hi - lo), 1e-6), 0, 1).round(4).tolist()
