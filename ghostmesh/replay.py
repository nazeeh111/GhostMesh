"""Replay export runs the model on CSI only; labels remain a separate overlay."""

from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
import torch
from .data import load_manifest, read_sequence, sha256
from .engine import load_checkpoint
from .models import radius_confidence
from .preprocess import prepare_window, signal_quality, spectrum


def predict_frames(
    csi,
    timestamps,
    model,
    checkpoint,
    *,
    device="cpu",
    max_gap=0.25,
    unseen_environment=False,
    unknown_domain=False,
    confidence_threshold=0.15,
):
    """Annotation-free inference. This function accepts no pose labels."""
    csi, timestamps = np.asarray(csi), np.asarray(timestamps)
    if not np.isfinite(max_gap) or max_gap <= 0 or not 0 <= confidence_threshold <= 1:
        raise ValueError(
            "Positive finite max_gap and confidence threshold in [0,1] required"
        )
    if csi.ndim != 4 or list(csi.shape[1:]) != checkpoint.get("input_shape"):
        raise ValueError(
            "Checkpoint and recording sensor/subcarrier/packet layouts differ"
        )
    if (
        timestamps.shape != (len(csi),)
        or len(csi) < 1
        or not np.isfinite(timestamps).all()
        or (np.diff(timestamps) <= 0).any()
    ):
        raise ValueError("Recording timestamps must align and strictly increase")
    torch.set_num_threads(int(checkpoint["config"].get("threads", 2)))
    data_config = checkpoint["config"].get("data", {})
    window = int(data_config.get("window", 8))
    frames = []
    for end in range(len(csi)):
        start = max(0, end - window + 1)
        raw = csi[start : end + 1]
        ok, reasons = signal_quality(raw)
        if end < window - 1:
            ok = False
            reasons.append("Causal window warming up")
        if (np.diff(timestamps[start : end + 1]) > max_gap).any():
            ok = False
            reasons.append("Timestamp gap exceeds synchronization tolerance")
        pose, confidence, latency = None, [0.0] * model.joints, None
        if ok:
            began = time.perf_counter()
            try:
                x = (
                    torch.from_numpy(
                        prepare_window(raw, data_config.get("normalize", True))
                    )
                    .unsqueeze(0)
                    .to(device)
                )
            except ValueError as exc:
                ok = False
                reasons.append(str(exc))
            if ok:
                with torch.no_grad():
                    prediction = model(x)
                pose = prediction["pose"][0].cpu().numpy().round(6).tolist()
                confidence = (
                    radius_confidence(prediction["log_sigma"], 0.05)[0]
                    .cpu()
                    .numpy()
                    .round(6)
                    .tolist()
                )
                if device == "mps":
                    torch.mps.synchronize()
                if device == "cuda":
                    torch.cuda.synchronize()
                latency = (time.perf_counter() - began) * 1000
        status = "ok"
        if not ok:
            status = "invalid_signal"
        elif unseen_environment:
            status = "out_of_domain"
            reasons.append(
                "Environment absent from training; metadata flag, not learned OOD detection"
            )
        elif unknown_domain:
            status = "out_of_domain"
            reasons.append(
                "Recording domain not verified against training; no automatic OOD detector"
            )
        elif np.mean(confidence) < confidence_threshold:
            status = "low_confidence"
            reasons.append("Mean uncalibrated Gaussian mass below display threshold")
        frames.append(
            {
                "time": float(timestamps[end] - timestamps[0]),
                "pose": pose,
                "confidence": confidence,
                "spectrum": spectrum(raw),
                "status": status,
                "reasons": reasons,
                "latency_ms": latency,
            }
        )
    return frames


def infer_recording(
    recording_path, checkpoint_path, output, *, device="cpu", max_gap=0.25
):
    """Run compatible recorded CSI with no annotation files, camera or manifest."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite replay {output}")
    with np.load(recording_path, allow_pickle=False) as data:
        if not {"csi", "timestamps"} <= set(data.files):
            raise ValueError(
                "Recording requires csi[N,C,S,P] and timestamps[N]; pose labels are not required"
            )
        csi, timestamps = data["csi"], data["timestamps"]
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    frames = predict_frames(
        csi,
        timestamps,
        model,
        checkpoint,
        device=device,
        max_gap=max_gap,
        unknown_domain=True,
    )
    from .prepare import JOINT_NAMES, EDGES

    standard = bool(model.joints == 17)
    provenance = {
        "kind": "synthetic_diagnostic"
        if checkpoint.get("purpose") == "software_check_only"
        else "recording_inference",
        "dataset": "User-supplied recorded CSI",
        "split": "unverified",
        "subject": "unknown",
        "environment": "unknown",
        "sequence": Path(recording_path).name,
        "checkpoint": Path(checkpoint_path).name,
        "checkpoint_sha256": sha256(checkpoint_path),
        "recording_sha256": sha256(recording_path),
        "coordinate_system": "normalized_image",
        "note": "CSI and timestamps only. No annotation files loaded. Domain compatibility unverified; sensor layout enforced.",
        "confidence_definition": "Uncalibrated Gaussian mass within radius .05 in the declared annotation reference plane",
    }
    result = {
        "schema_version": 1,
        "provenance": provenance,
        "joint_names": JOINT_NAMES
        if standard
        else [f"joint_{i}" for i in range(model.joints)],
        "edges": EDGES if standard else [],
        "frames": frames,
    }
    payload = json.dumps(result, allow_nan=False, separators=(",", ":")) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    return {"output": str(output), "frames": len(frames), "provenance": provenance}


def export_replay(
    manifest_path,
    checkpoint_path,
    output,
    *,
    sequence=None,
    split="test",
    device="cpu",
    confidence_threshold=0.15,
):
    path, output = Path(manifest_path), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite replay {output}")
    manifest = load_manifest(path)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    if sha256(path) != checkpoint["manifest_sha256"]:
        raise ValueError("Replay manifest differs from checkpoint's evaluated dataset")
    torch.set_num_threads(int(checkpoint["config"].get("threads", 2)))
    records = [
        r
        for r in manifest["sequences"]
        if r["split"] == split and (sequence is None or r["id"] == sequence)
    ]
    if not records:
        raise ValueError("Requested sequence/split absent")
    row = records[0]
    seq = read_sequence(path.parent / row["path"])
    config = checkpoint["config"]
    window = int(config.get("data", {}).get("window", 8))
    if seq["csi"].shape[1] != model.channels or seq["pose"].shape[1] != model.joints:
        raise ValueError("Checkpoint and recording sensor/skeleton layouts differ")
    if list(seq["csi"].shape[1:]) != checkpoint.get("input_shape"):
        raise ValueError("Checkpoint and recording subcarrier/packet layouts differ")
    source_environments = {
        r["environment"] for r in manifest["sequences"] if r["split"] == "train"
    }
    unseen_environment = row["environment"] not in source_environments
    frames = predict_frames(
        seq["csi"],
        seq["timestamps"],
        model,
        checkpoint,
        device=device,
        max_gap=row.get("max_gap_seconds", 0.25),
        unseen_environment=unseen_environment,
        confidence_threshold=confidence_threshold,
    )
    for end, frame in enumerate(frames):
        frame["ground_truth"] = seq["pose"][end].round(6).tolist()
        frame["ground_truth_valid"] = seq["valid"][end].tolist()
    diagnostic = manifest["provenance"]["kind"] == "synthetic_fixture"
    result = {
        "schema_version": 1,
        "provenance": {
            "kind": "synthetic_diagnostic" if diagnostic else "dataset_inference",
            "dataset": manifest["provenance"].get("dataset"),
            "split": split,
            "subject": row["subject"],
            "environment": row["environment"],
            "sequence": row["id"],
            "checkpoint": str(Path(checkpoint_path).name),
            "checkpoint_sha256": sha256(checkpoint_path),
            "manifest_sha256": sha256(path),
            "coordinate_system": "normalized_image",
            "source": manifest["provenance"].get("source"),
            "license": manifest["provenance"].get("license"),
            "note": "Generated fixture; software validation only"
            if diagnostic
            else "Actual held-out CSI inference. Reference labels are supplied teacher annotations; no camera input to model.",
            "confidence_definition": "Uncalibrated isotropic Gaussian mass inside radius .05 normalized image coordinates",
            "signal_map": "Subcarrier by causal frame amplitude map, not Doppler spectrum",
            "context_seconds": float(np.median(np.diff(seq["timestamps"])))
            * (window - 1)
            if len(seq["timestamps"]) > 1
            else 0.0,
        },
        "joint_names": manifest.get(
            "joint_names", [f"joint_{i}" for i in range(model.joints)]
        ),
        "edges": manifest.get("edges", []),
        "frames": frames,
    }
    payload = json.dumps(result, allow_nan=False, separators=(",", ":")) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    return {
        "output": str(output),
        "frames": len(frames),
        "provenance": result["provenance"],
        "status_counts": {
            s: sum(f["status"] == s for f in frames)
            for s in {f["status"] for f in frames}
        },
    }
