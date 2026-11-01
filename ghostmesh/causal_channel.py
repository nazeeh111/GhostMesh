"""Causal per-subcarrier background removal for receiver-relative CSI quotients.

Original code; standard quotient/static-channel ideas, not a physical Doppler,
AoA, velocity, RSSI/dBm or novelty claim. Invalid samples are neutral and do
not update the background. Frame intervals are nominally 0.1s.
"""

from __future__ import annotations

import numpy as np


def channel_representations(rf, alpha=1 - np.exp(-0.1 / 2), startup=31):
    """Return untreated/residual [T,4,114,10], and fixed-start [4,114,10].

    RF holds three log-magnitude-dB channels then three phase-radian channels.
    Complex ratios use a fixed +/-40dB ratio guard matching the earlier typed
    quotient probe. Subtract the PREVIOUS frame's per-subcarrier background;
    only then update its valid-packet mean. The fixed control freezes the
    background after frames0..30, before the first scored frame31.
    """
    x = np.asarray(rf, dtype=np.float32)
    if (
        x.ndim != 4
        or x.shape[1:] != (6, 114, 10)
        or not 0 < alpha <= 1
        or not 1 <= startup < len(x)
    ):
        raise ValueError(
            "Expected [T>startup,6,114,10] typed RF and valid EMA settings"
        )
    magnitude, phase = x[:, :3], x[:, 3:]
    finite = np.isfinite(magnitude) & np.isfinite(phase)
    valid = finite[:, 1:] & finite[:, :1]
    magnitude = np.where(finite, magnitude, 0)
    phase = np.where(finite, phase, 0)
    difference = (magnitude[:, 1:] - magnitude[:, :1]) / 20
    log_ratio = np.clip(difference, -2, 2)
    delta = phase[:, 1:] - phase[:, :1]
    q = np.where(
        valid, 10.0**log_ratio * (np.cos(delta) + 1j * np.sin(delta)), 0
    ).astype(np.complex64)
    residual = np.zeros_like(q)
    count = valid[0].sum(-1)
    background = q[0].sum(-1) / np.maximum(count, 1)
    background = background.astype(np.complex64)
    frozen = None
    for t in range(len(q)):
        residual[t] = np.where(valid[t], q[t] - background[..., None], 0)
        count = valid[t].sum(-1)
        current = q[t].sum(-1) / np.maximum(count, 1)
        background = np.where(
            count > 0, (1 - alpha) * background + alpha * current, background
        ).astype(np.complex64)
        if t == startup - 1:
            frozen = background.copy()

    def real(z):
        return np.concatenate((z.real, z.imag), axis=-3).astype(np.float32)

    static = np.broadcast_to(frozen[..., None], q.shape[1:])
    return {
        "untreated": real(q),
        "ema_residual": real(residual),
        "startup_static": real(static),
    }, {
        "invalid_pair_fraction": float(1 - valid.mean()),
        "ratio_clipping_fraction": float((np.abs(difference[valid]) > 2).mean())
        if valid.any()
        else None,
        "alpha": float(alpha),
        "startup_frames": startup,
    }
