"""Restricted same-person/room/action kernel diagnostic, not a transfer model.

The checkpoint stores reference CSI and fitted regression coefficients. Query
inference needs CSI and timestamps only; annotation overlays are separate.
"""

from pathlib import Path
import json

import numpy as np
from scipy.linalg.blas import dgemm

from .rf_features import quotient_features

KIND = "same_condition_kernel_v1"
WIDTH = 8 * 6 * 114 * 10


def design(rf):
    rf = np.asarray(rf)
    if rf.ndim != 4 or rf.shape[1:] != (6, 114, 10) or len(rf) < 8:
        raise ValueError("Expected at least8 typed RF frames[T,6,114,10]")
    features = quotient_features(rf[:, :3], rf[:, 3:], units="log_magnitude_db").astype(
        "float64"
    )
    support = np.arange(7, len(rf))[:, None] - np.arange(7, -1, -1)[None]
    return features[support].reshape(len(support), WIDTH)


def load_model(path):
    with np.load(path, allow_pickle=False) as bundle:
        values = {k: bundle[k] for k in bundle.files}
    if (
        str(values.get("kind")) != KIND
        or str(values.get("magnitude_units")) != "log_magnitude_db"
        or str(values.get("phase_units")) != "radians"
    ):
        raise ValueError("Expected explicitly typed same-condition kernel checkpoint")
    for key in ("input_mean", "input_scale"):
        if (
            key not in values
            or values[key].shape != (WIDTH,)
            or not np.isfinite(values[key]).all()
        ):
            raise ValueError("Invalid kernel normalizer")
    if np.any(values["input_scale"] <= 0):
        raise ValueError("Kernel scale must be positive")
    training = design(values["reference_rf"])
    if values["dual"].shape != (len(training), 34) or values["target_mean"].shape != (
        34,
    ):
        raise ValueError("Kernel coefficients do not match reference contexts")
    if (
        not np.isfinite(values["dual"]).all()
        or not np.isfinite(values["target_mean"]).all()
        or not np.isfinite(values["bandwidth_squared"])
        or values["bandwidth_squared"] <= 0
    ):
        raise ValueError("Invalid fitted kernel")
    values["training_design"] = (training - values["input_mean"]) / values[
        "input_scale"
    ]
    values["training_squared_norm"] = np.square(values["training_design"]).sum(1)
    return values


def predict(model, rf):
    """Causal RF-only inference; returned positions align to input frames7 onward."""
    query = (design(rf) - model["input_mean"]) / model["input_scale"]
    training = model["training_design"]
    cross = dgemm(1.0, query, training.T)
    distance = np.maximum(
        np.square(query).sum(1)[:, None]
        + model["training_squared_norm"][None]
        - 2 * cross,
        0,
    )
    kernel = np.exp(-distance / (2 * float(model["bandwidth_squared"])))
    raw = (model["target_mean"] + dgemm(1.0, kernel, model["dual"])).reshape(-1, 17, 2)
    return raw.clip(0, 1)


def amplitude_map(rf):
    """Causal subcarrier/time map of original log magnitude, not Doppler."""
    amplitude = np.asarray(rf[:, :3], float)
    finite = np.isfinite(amplitude)
    if not finite.any():
        return []
    fill = float(np.median(amplitude[finite]))
    x = np.where(finite, amplitude, fill).mean((1, 3)).T
    lo, hi = np.percentile(x, [5, 95])
    return np.clip((x - lo) / max(float(hi - lo), 1e-6), 0, 1).round(4).tolist()


def infer_replay(
    model, rf, timestamps, *, checkpoint_name, checkpoint_sha256, recording_sha256
):
    from .prepare import EDGES, JOINT_NAMES

    times = np.asarray(timestamps)
    rf = np.asarray(rf)
    if rf.ndim != 4 or rf.shape[1:] != (6, 114, 10) or len(rf) < 1:
        raise ValueError("Expected typed RF[T,6,114,10]")
    if (
        times.shape != (len(rf),)
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("Strictly increasing finite timestamps must align with CSI")
    frames = []
    # Per-window checks allow invalid frames to be withheld without filling gaps.
    for end in range(len(rf)):
        raw = rf[max(0, end - 7) : end + 1]
        pose, status = None, "invalid_signal"
        reasons = []
        if len(raw) < 8:
            reasons.append("Causal8-frame context warming up")
        elif np.any(np.diff(times[end - 7 : end + 1]) > 0.25):
            reasons.append("Timestamp gap exceeds synchronization tolerance")
        else:
            try:
                pose = predict(model, raw)[0].tolist()
                status = "low_confidence"
                reasons.append(
                    "No uncertainty estimator; confidence unavailable. Same-person/room/action diagnostic only."
                )
            except ValueError as exc:
                reasons.append(str(exc))
        frames.append(
            dict(
                time=float(times[end] - times[0]),
                pose=pose,
                confidence=[0.0] * 17,
                status=status,
                reasons=reasons,
                spectrum=amplitude_map(raw),
                latency_ms=None,
            )
        )
    return dict(
        schema_version=1,
        provenance=dict(
            kind="recording_inference",
            dataset="Recorded CSI (domain unverified)",
            subject="unknown",
            environment="unknown",
            sequence="unverified recording",
            split="unverified",
            evaluation_scope="unverified",
            coordinate_system="normalized_image",
            confidence_available=False,
            confidence_definition="Unavailable: kernel regression has no fitted uncertainty estimator; zero array entries are schema placeholders, not scores.",
            checkpoint=checkpoint_name,
            checkpoint_sha256=checkpoint_sha256,
            recording_sha256=recording_sha256,
            license="CC-BY-NC-4.0",
            source="https://github.com/ybhbingo/MMFi_dataset",
            note="Model fitted to111 windows from E01/S01/A12 only. Generalization is unvalidated; input domain is unknown. Dense source references span11.8s. Query CSI only; no annotation files loaded.",
            timestamp_basis="Recording-supplied timestamps",
            context_frames=8,
            training_context_seconds=0.7,
            signal_map="Original log-magnitude subcarrier/time map, not Doppler",
        ),
        joint_names=JOINT_NAMES,
        edges=EDGES,
        frames=frames,
    )


def infer_recording(model_path, recording_path, output):
    from .data import sha256

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    with np.load(recording_path, allow_pickle=False) as source:
        if not {"rf", "timestamps", "magnitude_units", "phase_units"} <= set(
            source.files
        ):
            raise ValueError("RF and timestamps with explicit units required")
        if (
            str(source["magnitude_units"]) != "log_magnitude_db"
            or str(source["phase_units"]) != "radians"
        ):
            raise ValueError("Incompatible recording units")
        rf, timestamps = source["rf"], source["timestamps"]
    model = load_model(model_path)
    replay = infer_replay(
        model,
        rf,
        timestamps,
        checkpoint_name=Path(model_path).name,
        checkpoint_sha256=sha256(model_path),
        recording_sha256=sha256(recording_path),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(replay, allow_nan=False, separators=(",", ":")) + "\n")
    return replay
