"""Nominal signed temporal bandpower of complex interreceiver CSI quotients.

The MM-Fi paper describes about100Hz CSI packed into ten-column100ms frames.
Packet timestamps/order are not supplied here. These are conditional nominal
frequency features, not calibrated Doppler, physical velocity, AoA or BVP.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 100.0
BAND_EDGES = (1.25, 2.5, 5.0, 10.0, 20.0, 50.0)
WINDOW = 80
POWER_FLOOR = 1e-12
COLUMN_ORDER = np.random.default_rng(2027).permutation(10)
WHOLE_ORDER = np.random.default_rng(2028).permutation(80)


def complex_quotients(rf, *, magnitude_units, phase_units):
    if magnitude_units != "log_magnitude_db" or phase_units != "radians":
        raise ValueError("Explicit log_magnitude_db and radians required")
    rf = np.asarray(rf, dtype=np.float64)
    if rf.ndim != 4 or rf.shape[1:] != (6, 114, 10) or not len(rf):
        raise ValueError("Expected typed RF[T,6,114,10]")
    amplitude, phase = rf[:, :3], rf[:, 3:]
    finite = np.isfinite(amplitude) & np.isfinite(phase)
    valid = finite[:, 1:] & finite[:, :1]
    a = np.where(finite, amplitude, 0.0)
    p = np.where(finite, phase, 0.0)
    logratio = (a[:, 1:] - a[:, :1]) / 20.0
    clipped = (abs(logratio) > 2) & valid
    value = np.power(10.0, np.clip(logratio, -2.0, 2.0)) * np.exp(
        1j * (p[:, 1:] - p[:, :1])
    )
    value = np.where(valid, value, np.nan + 1j * np.nan)
    return value, dict(
        valid_pairs=int(valid.sum()),
        missing_pairs=int((~valid).sum()),
        clipped_pairs=int(clipped.sum()),
        total_pairs=int(valid.size),
        clip_fraction=float(clipped.sum() / max(valid.sum(), 1)),
    )


def _pair_summary(first, second):
    valid = np.isfinite(first) & np.isfinite(second)
    a, b = first[valid], second[valid]
    if not len(a):
        raise ValueError("No finite neighboring CSI pairs")
    energy = 0.5 * (np.square(abs(a)) + np.square(abs(b)))
    normalized = np.square(abs(b - a)) / np.maximum(energy, 1e-24)
    phase = abs(np.angle(b * a.conj()))
    amplitude = abs(20 * np.log10(abs(b) / abs(a)))
    correlation = np.mean(b * a.conj()) / max(
        float(np.sqrt(np.mean(abs(a) ** 2) * np.mean(abs(b) ** 2))), 1e-24
    )

    def quantiles(value):
        return dict(
            zip(
                ("q25", "median", "q75"),
                [float(v) for v in np.quantile(value, [0.25, 0.5, 0.75])],
            )
        )

    return dict(
        pairs=len(a),
        normalized_complex_step=quantiles(normalized),
        circular_phase_increment_radians=quantiles(phase),
        absolute_log_amplitude_increment_db=quantiles(amplitude),
        complex_lag_correlation_real=float(correlation.real),
        complex_lag_correlation_abs=float(abs(correlation)),
        exact_equal_pair_fraction=float(np.mean(a == b)),
    )


def clock_audit(quotients):
    q = np.asarray(quotients)
    if q.ndim != 4 or q.shape[1:] != (2, 114, 10) or len(q) < 2:
        raise ValueError("At least two quotient frames required")
    shuffled = q[..., COLUMN_ORDER]
    result = dict(
        natural_within_frame=_pair_summary(q[..., :-1], q[..., 1:]),
        natural_frame_boundary=_pair_summary(q[:-1, ..., -1], q[1:, ..., 0]),
        shuffled_within_frame=_pair_summary(shuffled[..., :-1], shuffled[..., 1:]),
        shuffled_frame_boundary=_pair_summary(
            shuffled[:-1, ..., -1], shuffled[1:, ..., 0]
        ),
    )
    within = result["natural_within_frame"]["normalized_complex_step"]["median"]
    boundary = result["natural_frame_boundary"]["normalized_complex_step"]["median"]
    unordered = result["shuffled_within_frame"]["normalized_complex_step"]["median"]
    result["boundary_to_within_median_ratio"] = float(boundary / max(within, 1e-12))
    result["natural_to_shuffled_within_median_ratio"] = float(
        within / max(unordered, 1e-12)
    )
    result["plausible_nominal_continuity"] = bool(
        within < unordered and boundary < 10 * max(within, 1e-12)
    )
    return result


def signed_band_power(quotients, variant="natural"):
    q = np.asarray(quotients, dtype=np.complex128)
    if q.shape != (8, 2, 114, 10):
        raise ValueError("Expected exactly8frames of2x114x10quotients")
    x = q.transpose(1, 2, 0, 3).reshape(2, 114, 80)
    finite = np.isfinite(x)
    counts = finite.sum(-1)
    if np.any(counts < 76):
        raise ValueError("More than5percent missing samples in a quotient trace")
    # Constant causal-window mean fill only, never interpolation or a future frame.
    center = np.where(finite, x, 0.0).sum(-1) / counts
    x = np.where(finite, x - center[..., None], 0.0)
    if variant == "within_frame_shuffle":
        x = x.reshape(2, 114, 8, 10)[..., COLUMN_ORDER].reshape(2, 114, 80)
    elif variant == "whole_window_shuffle":
        x = x[..., WHOLE_ORDER]
    elif variant != "natural":
        raise ValueError("Unknown packet-order variant")
    hann = np.hanning(80)
    frequency = np.fft.fftfreq(80, d=1 / SAMPLE_RATE)
    power = abs(np.fft.fft(x * hann, axis=-1)) ** 2 / (
        SAMPLE_RATE * float(np.square(hann).sum())
    )
    bands = []
    for sign in (-1, 1):
        signed = sign * frequency
        for lower, upper in zip(BAND_EDGES[:-1], BAND_EDGES[1:]):
            pick = (signed >= lower) & (signed < upper) & (abs(frequency) < 50.0)
            bands.append(power[..., pick].sum(-1) * (SAMPLE_RATE / 80))
    # Average POWERS of six adjacent subcarriers; do not average complex phases.
    return np.stack(bands, axis=-1).reshape(2, 19, 6, 10).mean(2)


def spectral_features(quotients, variant="natural"):
    return (
        10 * np.log10(np.maximum(signed_band_power(quotients, variant), POWER_FLOOR))
    ).reshape(380)
