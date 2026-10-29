"""Fixed, consented room-reference covariance features.

Normalize each reference matrix BEFORE averaging. Under C'=a G C G^H,
the normalized calibration and query shapes share the same determinant-one
congruence. The existing paired shrinkage then preserves generalized spectra.
Actual room changes need not obey this model. A fixed anchor restores some
static configuration information, but one or several spectra are not an
injective pose observation. The optional log-volume channel loses invariance
to independent positive frame gains and is reported as a separate ablation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ghostmesh.covariance_dynamics import determinant_shape, relative_log_spectrum


def decode_hermitian_coordinates(coordinates):
    """Invert the original nine-coordinate encoding, with no SPD repair.

    Cached float32 coordinates carry quantization error. Reject a decoded
    non-SPD matrix at the caller; identity jitter would change the gauge claim.
    """
    value = np.asarray(coordinates, dtype="float64")
    if value.shape[-1:] != (9,) or not np.isfinite(value).all():
        raise ValueError("Expected finite nine-coordinate Hermitian encoding")
    covariance = np.zeros((*value.shape[:-1], 3, 3), dtype="complex128")
    covariance[..., (0, 1, 2), (0, 1, 2)] = value[..., :3]
    off = (value[..., 3:6] + 1j * value[..., 6:9]) / np.sqrt(2)
    covariance[..., (0, 0, 1), (1, 2, 2)] = off
    covariance[..., (1, 2, 2), (0, 0, 1)] = off.conj()
    return covariance


def current_covariances(full_features):
    """All eight CURRENT states; discard the six lagged states in this mode."""
    value = np.asarray(full_features)
    if value.ndim != 2 or value.shape[1] != 8 * 19 * 7 * 9:
        raise ValueError("Expected untreated cache features[N,9576]")
    return decode_hermitian_coordinates(value.reshape(-1, 8, 19, 7, 9)[:, :, :, 0])


@dataclass(frozen=True)
class RoomReference:
    anchors: np.ndarray
    mean_log_volume: np.ndarray
    group_names: tuple[str, ...]
    group_counts: tuple[int, ...]

    def __post_init__(self):
        anchors = np.array(self.anchors, dtype="complex128", copy=True)
        volume = np.array(self.mean_log_volume, dtype="float64", copy=True)
        if anchors.shape != (len(self.group_names), 19, 3, 3) or volume.shape != (19,):
            raise ValueError("Reference dimensions differ from declared groups")
        determinant_shape(anchors)
        if not np.isfinite(volume).all() or len(self.group_counts) != len(anchors):
            raise ValueError("Invalid reference volume or group counts")
        anchors.setflags(write=False)
        volume.setflags(write=False)
        object.__setattr__(self, "anchors", anchors)
        object.__setattr__(self, "mean_log_volume", volume)


def fit_reference(covariance, groups=None, *, group_order=None):
    """CSI-only reference fit; ground-truth joints are deliberately not inputs.

    Only endpoint covariances are averaged: one matrix per reference pose and
    frequency bin. An action-group name is enrollment protocol metadata, never
    a query action label. Query matrices do not update this reference.
    """
    covariance = np.asarray(covariance, dtype="complex128")
    if (
        covariance.ndim != 4
        or covariance.shape[1:] != (19, 3, 3)
        or not len(covariance)
    ):
        raise ValueError("Expected calibration covariances[N,19,3,3]")
    shapes = determinant_shape(covariance)
    volume = np.log(np.linalg.eigvalsh(covariance)).mean(-1).mean(0)
    if groups is None:
        groups = np.repeat("all", len(covariance))
        names = ("all",)
    else:
        groups = np.asarray(groups)
        if groups.shape != (len(covariance),):
            raise ValueError("One protocol group is required per reference")
        names = (
            tuple(group_order)
            if group_order is not None
            else tuple(sorted(set(groups)))
        )
        if set(names) != set(groups) or len(set(names)) != len(names):
            raise ValueError(
                "Reference group order must include each observed group once"
            )
    anchors, counts = [], []
    for name in names:
        selected = groups == name
        if not selected.any():
            raise ValueError("Empty reference group")
        anchor = shapes[selected].mean(0)
        anchors.append((anchor + anchor.conj().swapaxes(-1, -2)) / 2)
        counts.append(int(selected.sum()))
    return RoomReference(
        np.stack(anchors), volume, tuple(str(n) for n in names), tuple(counts)
    )


def reference_features(covariance, reference, *, include_volume=False):
    """Per-current-frame descriptors, with no query normalization or state fit."""
    covariance = np.asarray(covariance, dtype="complex128")
    if covariance.ndim < 3 or covariance.shape[-3:] != (19, 3, 3):
        raise ValueError("Expected current covariance[...,19,3,3]")
    spectra = [
        relative_log_spectrum(covariance, anchor) for anchor in reference.anchors
    ]
    if include_volume:
        # A fixed G cancels here. Independent query/reference gains do not.
        log_volume = np.log(np.linalg.eigvalsh(covariance)).mean(-1)
        spectra.append((log_volume - reference.mean_log_volume)[..., None])
    return np.concatenate(spectra, axis=-1)
