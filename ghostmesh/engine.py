"""Reproducible source-only training, fixed-checkpoint evaluation and inference."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import random
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import WindowDataset, load_manifest
from .metrics import pose_metrics
from .models import build_model, supervised_loss


def _json(path: Path, value: dict):
    content = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_torch_save(value, path):
    temporary = None
    try:
        # A file object keeps PyTorch's archive names independent of this
        # temporary filename, making equal best snapshots byte-reproducible.
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _capture_rng(generator, device):
    py = random.getstate()
    np_state = np.random.get_state()
    result = {
        "torch_cpu": torch.get_rng_state().clone(),
        "loader": generator.get_state().clone(),
        "python": {"version": py[0], "internal": list(py[1]), "gauss": py[2]},
        "numpy": {
            "algorithm": np_state[0],
            "keys": np_state[1].tolist(),
            "position": np_state[2],
            "has_gauss": np_state[3],
            "cached_gauss": np_state[4],
        },
    }
    if device.type == "cuda":
        result["cuda"] = [
            value.cpu().clone() for value in torch.cuda.get_rng_state_all()
        ]
    elif device.type == "mps":
        result["mps"] = torch.mps.get_rng_state().cpu().clone()
    return result


def _restore_rng(state, generator, device):
    torch.set_rng_state(state["torch_cpu"].cpu())
    generator.set_state(state["loader"].cpu())
    py, nr = state["python"], state["numpy"]
    random.setstate((py["version"], tuple(py["internal"]), py["gauss"]))
    np.random.set_state(
        (
            nr["algorithm"],
            np.asarray(nr["keys"], dtype=np.uint32),
            nr["position"],
            nr["has_gauss"],
            nr["cached_gauss"],
        )
    )
    if device.type == "cuda":
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    elif device.type == "mps":
        torch.mps.set_rng_state(state["mps"].cpu())


def _preserve_for_recovery(path, output):
    if path.is_symlink():
        raise ValueError("Managed recovery source must not be a symlink")
    if not path.exists():
        return None
    recovery = output / "recovery"
    if recovery.is_symlink():
        raise ValueError("Recovery directory must not be a symlink")
    recovery.mkdir(exist_ok=True)
    content = path.read_bytes()
    destination = recovery / f"{path.name}.{sha256(content).hexdigest()}.preserved"
    if destination.exists():
        if destination.read_bytes() != content:
            raise ValueError("Refusing to replace different recovery bytes")
    else:
        with destination.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    return str(destination.relative_to(output))


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def validate_protocol(manifest: dict, protocol: str):
    """Never describe a sequence split as independent people or environments."""
    declared = manifest.get("protocol")
    if declared in ("cross_subject", "cross_environment") and declared != protocol:
        raise ValueError(
            f"Config protocol {protocol} does not match manifest protocol {declared}"
        )
    records = manifest["sequences"]
    split_records = {
        s: [r for r in records if r["split"] == s]
        for s in ("train", "validation", "test")
    }
    for split, entries in split_records.items():
        if not entries:
            raise ValueError(f"Protocol requires a nonempty {split} split")
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sequence IDs can leak across splits")
    if protocol in ("cross_subject", "cross_environment"):
        subjects = {
            s: {r["subject"] for r in rows} for s, rows in split_records.items()
        }
        for a, b in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            if subjects[a] & subjects[b]:
                raise ValueError(f"Subjects overlap between {a} and {b}")
    if protocol == "cross_environment":
        source = {
            r["environment"] for s in ("train", "validation") for r in split_records[s]
        }
        target = {r["environment"] for r in split_records["test"]}
        if source & target:
            raise ValueError("Held-out test environments overlap source environments")
    elif protocol not in ("cross_subject", "sequence"):
        raise ValueError(
            "protocol must be cross_subject, cross_environment or sequence"
        )


def _device(name: str) -> torch.device:
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return torch.device(name)


def _sync(device: torch.device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _move(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }


def _dataset(manifest_path, split, config):
    data = config.get("data", {})
    return WindowDataset(
        manifest_path,
        split,
        window=int(data.get("window", 8)),
        normalize=bool(data.get("normalize", True)),
        stride=int(data.get("split_stride", {}).get(split, data.get("stride", 1))),
        cache_windows=bool(data.get("cache_windows", False)),
    )


def _augment(x: torch.Tensor, strength: float) -> torch.Tensor:
    # Per-link gain and sparse subcarrier erasure, applied only to source training.
    b, _, c, s, _ = x.shape
    gain = torch.exp(torch.randn(b, 1, c, 1, 1, device=x.device) * strength)
    keep = (torch.rand(b, 1, 1, s, 1, device=x.device) > min(strength, 0.5)).to(x.dtype)
    return x * gain * keep


def _consistency_loss(prediction, teacher_prediction, mode="raw"):
    """Source-only teacher consistency with optional bounded precision units.

    ``bounded_precision`` divides squared 2D displacement by twice the teacher
    coordinate variance, clamped to [.05**2, .5**2] in normalized coordinates.
    These fixed bounds prevent an uncalibrated teacher from demanding infinite
    precision. They are a loss-scale convention, not a correctness guarantee.
    """
    error2 = (prediction["pose"] - teacher_prediction["pose"].detach()).square().sum(-1)
    if mode == "raw":
        return error2.mean()
    if mode == "bounded_precision":
        variance = torch.exp(2 * teacher_prediction["log_sigma"].detach()).clamp(
            0.05**2, 0.5**2
        )
        return (error2 / (2 * variance)).mean()
    raise ValueError("consistency_mode must be raw or bounded_precision")


@torch.no_grad()
def _collect(model, loader, device, threshold=0.2):
    model.eval()
    poses, targets, validity, sigmas, metadata = [], [], [], [], []
    for batch in loader:
        moved = _move(batch, device)
        out = model(moved["csi"])
        poses.append(out["pose"].cpu().numpy())
        sigmas.append(out["log_sigma"].cpu().numpy())
        targets.append(batch["pose"].numpy())
        validity.append(batch["valid"].numpy())
        for i in range(len(batch["csi"])):
            item = {}
            for key in ("sequence", "subject", "environment", "time", "frame_index"):
                if key in batch:
                    v = batch[key][i]
                    item[key] = v.item() if isinstance(v, torch.Tensor) else v
            metadata.append(item)
    if not poses:
        raise ValueError("No complete usable windows in split")
    arrays = [np.concatenate(v) for v in (poses, targets, validity, sigmas)]
    result = pose_metrics(*arrays[:3], log_sigma=arrays[3], threshold=threshold)
    return result, arrays, metadata


def _purpose(manifest):
    return (
        "software_check_only"
        if manifest.get("provenance", {}).get("kind") == "synthetic_fixture"
        else "dataset_experiment"
    )


def _fingerprint(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _source_fingerprints():
    """Hash the actual runtime modules, independent of Git or documentation."""
    root = Path(__file__).resolve().parent
    return {
        name: _fingerprint(root / name)
        for name in ("engine.py", "models.py", "data.py", "preprocess.py", "metrics.py")
    }


def _runtime_environment():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_cuda_build": torch.version.cuda,
    }


def _coverage(dataset, metric_report=None):
    if hasattr(dataset, "coverage"):
        result = dataset.coverage()
    else:
        # Custom injected datasets must not acquire a fabricated 100% coverage.
        result = {
            "accepted_windows": len(dataset),
            "candidate_windows": None,
            "rejected_windows": None,
            "accepted_fraction": None,
            "note": "This dataset does not expose prefilter coverage accounting",
        }
    if metric_report is not None:
        result = {
            **result,
            "metric_eligible_windows": metric_report["valid_frames"],
            "metric_excluded_windows": metric_report["excluded_frames"],
            "metric_eligible_fraction_of_accepted": metric_report["valid_frames"]
            / len(dataset)
            if len(dataset)
            else None,
            "metric_eligible_fraction_of_candidates": metric_report["valid_frames"]
            / result["candidate_windows"]
            if result.get("candidate_windows")
            else None,
        }
    if hasattr(dataset, "storage_info"):
        result = {**result, "storage": dataset.storage_info()}
    return result


def _evaluation_provenance(checkpoint=None):
    current = _source_fingerprints()
    trained = checkpoint.get("source_sha256") if checkpoint else None
    return {
        "source_sha256": current,
        "runtime_environment": _runtime_environment(),
        "training_source_sha256": trained,
        "source_matches_training": current == trained if trained is not None else None,
        "source_changed_since_training": [
            name
            for name, digest in current.items()
            if trained is not None and trained.get(name) != digest
        ],
        "source_comparison_note": "Fingerprints are evidence, not a model-compatibility proof; historical checkpoints may lack source hashes.",
    }


def train(manifest_path, config_path, output, device="cpu", resume=False) -> dict:
    """Validation-only selection with explicit, exact completed-epoch recovery.

    ``latest.pt`` is the authoritative epoch commit and embeds the selected best
    snapshot. Partial epochs are discarded by restoring all training/RNG state;
    their output bytes are retained where recovery would replace a managed file.
    """
    manifest_path, config_path, output = (
        Path(manifest_path),
        Path(config_path),
        Path(output),
    )
    if output.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite experiment directory: {output}")
    if resume and not output.is_dir():
        raise FileNotFoundError(
            f"Cannot resume a missing experiment directory: {output}"
        )
    manifest = load_manifest(manifest_path)
    config = json.loads(config_path.read_text())
    validate_protocol(manifest, config.get("protocol", "cross_subject"))
    consistency_mode = config.get("training", {}).get("consistency_mode", "raw")
    if consistency_mode not in {"raw", "bounded_precision"}:
        raise ValueError("consistency_mode must be raw or bounded_precision")
    seed = int(config.get("seed", 2027))
    seed_everything(seed)
    torch.set_num_threads(int(config.get("threads", 2)))
    source_fingerprints = _source_fingerprints()
    runtime_environment = _runtime_environment()
    dev = _device(device)
    identity = {
        "schema_version": 1,
        "config": config,
        "config_source_sha256": _fingerprint(config_path),
        "manifest_sha256": _fingerprint(manifest_path),
        "source_sha256": source_fingerprints,
        "runtime_environment": runtime_environment,
        "device": str(dev),
    }
    if resume:
        try:
            saved_identity = json.loads((output / "run_identity.json").read_text())
        except (OSError, ValueError) as exc:
            raise ValueError(
                "Cannot resume without a valid run_identity.json; existing bytes were preserved"
            ) from exc
        if not isinstance(saved_identity, dict):
            raise ValueError("Cannot resume: run_identity.json must contain a mapping")
        if saved_identity != identity:
            changed = [
                key for key in identity if saved_identity.get(key) != identity[key]
            ]
            raise ValueError(
                f"Cannot resume changed experiment identity: {', '.join(changed)}"
            )
        if (output / "config.json").exists() and json.loads(
            (output / "config.json").read_text()
        ) != config:
            raise ValueError(
                "Cannot resume: saved config.json differs from the frozen config"
            )
    else:
        output.mkdir(parents=True)
        _json(output / "run_identity.json", identity)
    if not (output / "config.json").exists():
        _json(output / "config.json", config)

    latest = None
    recovery_files = []
    if resume and (output / "latest.pt").exists():
        try:
            latest = torch.load(
                output / "latest.pt", map_location="cpu", weights_only=True
            )
        except Exception as exc:
            raise ValueError(
                "Cannot resume corrupt latest.pt; it was not overwritten"
            ) from exc
        if (
            not isinstance(latest, dict)
            or latest.get("schema_version") != 1
            or latest.get("kind") != "ghostmesh_completed_epoch"
            or latest.get("run_identity") != identity
        ):
            raise ValueError(
                "Cannot resume: latest.pt identity/schema differs from frozen run"
            )
        recovery_files = list(latest.get("recovery_files", []))
    elif resume:
        # No completed epoch is committed. Repeating the deterministic seed is
        # safe only after the complete original run identity has matched.
        for name in ("best.pt", "history.json", "training_report.json"):
            preserved = _preserve_for_recovery(output / name, output)
            if preserved is not None:
                recovery_files.append(preserved)
    model = build_model(config).to(dev)
    teacher = deepcopy(model).eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    training = config.get("training", {})
    epochs = int(training.get("epochs", 40))
    if epochs < 1:
        raise ValueError("epochs must be positive")
    train_data, val_data = (
        _dataset(manifest_path, "train", config),
        _dataset(manifest_path, "validation", config),
    )
    expected = config.get("model", {})
    for split, dataset in (("train", train_data), ("validation", val_data)):
        if dataset.input_shape[0] != expected.get(
            "channels", 3
        ) or dataset.joints != expected.get("joints", 17):
            raise ValueError(
                f"{split} sensor/skeleton layout does not match model config"
            )
    if tuple(train_data.input_shape) != tuple(val_data.input_shape):
        raise ValueError("Training and validation sensor layouts differ")
    if len(train_data) == 0 or len(val_data) == 0:
        raise ValueError("Training and validation need complete usable windows")
    batch_size = int(training.get("batch_size", 32))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False, num_workers=0
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 0.001)),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    weight = float(training.get("consistency_weight", 0))
    decay = float(training.get("ema_decay", 0.99))
    if weight < 0 or not 0 <= decay < 1:
        raise ValueError("Invalid consistency weight or EMA decay")
    strength = float(training.get("augmentation_strength", 0.0))
    if strength < 0:
        raise ValueError("augmentation_strength must be nonnegative")
    threshold = float(config.get("evaluation", {}).get("pck_threshold", 0.2))
    history, best, best_epoch = [], float("inf"), None
    best_checkpoint, start_epoch, elapsed_before = None, 0, 0.0
    if latest is not None:
        start_epoch = latest.get("completed_epoch")
        history = latest.get("history")
        if (
            type(start_epoch) is not int
            or not 1 <= start_epoch <= epochs
            or not isinstance(history, list)
            or len(history) != start_epoch
            or [row.get("epoch") for row in history] != list(range(1, start_epoch + 1))
        ):
            raise ValueError(
                "Cannot resume: latest.pt completed-epoch/history accounting is invalid"
            )
        if (
            latest.get("input_shape") != list(train_data.input_shape)
            or latest.get("joints") != train_data.joints
        ):
            raise ValueError("Cannot resume: latest.pt sensor/skeleton layout differs")
        best, best_epoch, best_checkpoint = (
            latest["best_score"],
            latest["best_epoch"],
            latest["best_checkpoint"],
        )
        scored = [row for row in history if row["validation_nje"] is not None]
        selected = (
            min(scored, key=lambda row: row["validation_nje"]) if scored else None
        )
        if selected is not None and (
            best != selected["validation_nje"] or best_epoch != selected["epoch"]
        ):
            raise ValueError(
                "Cannot resume: best record is not selected by validation history"
            )
        if selected is None and (best_epoch is not None or best_checkpoint is not None):
            raise ValueError(
                "Cannot resume: best checkpoint lacks a valid validation score"
            )
        if best_checkpoint is not None and (
            best_checkpoint.get("epoch") != best_epoch
            or best_checkpoint.get("selected_by") != "validation_normalized_joint_error"
            or best_checkpoint.get("config") != config
            or best_checkpoint.get("manifest_sha256") != identity["manifest_sha256"]
            or best_checkpoint.get("source_sha256") != source_fingerprints
        ):
            raise ValueError(
                "Cannot resume: embedded best checkpoint disagrees with frozen run"
            )
        model.load_state_dict(latest["model"], strict=True)
        teacher.load_state_dict(latest["teacher"], strict=True)
        optimizer.load_state_dict(latest["optimizer"])
        elapsed_before = float(latest["elapsed_seconds"])
        if not np.isfinite(elapsed_before) or elapsed_before < 0:
            raise ValueError("Cannot resume: completed-epoch elapsed time is invalid")
        # Restore only after reconstruction has finished and before publishing
        # repairs. Atomic serialization itself consumes no training RNG state.
        _restore_rng(latest["rng"], generator, dev)
        if best_checkpoint is not None:
            preserved = _preserve_for_recovery(output / "best.pt", output)
            if preserved is not None:
                recovery_files.append(preserved)
            _atomic_torch_save(best_checkpoint, output / "best.pt")
        if (output / "history.json").exists():
            preserved = _preserve_for_recovery(output / "history.json", output)
            if preserved is not None:
                recovery_files.append(preserved)
        _json(
            output / "history.json", {"epochs": history, "purpose": _purpose(manifest)}
        )
    started = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        model.train()
        losses, joints_seen = 0.0, 0
        for batch in loader:
            b = _move(batch, dev)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(_augment(b["csi"], strength) if strength else b["csi"])
            loss = supervised_loss(prediction, b["pose"], b["valid"])
            if weight:
                with torch.no_grad():
                    teacher_prediction = teacher(b["csi"])
                # Source-only teacher consistency; no validation/test adaptation.
                consistency = _consistency_loss(
                    prediction, teacher_prediction, consistency_mode
                )
                loss = loss + weight * min(1.0, (epoch + 1) / 5) * consistency
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite training loss; no result can be accepted"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            with torch.no_grad():
                for t, s in zip(teacher.parameters(), model.parameters()):
                    t.lerp_(s, 1 - decay)
                for t, s in zip(teacher.buffers(), model.buffers()):
                    t.copy_(s)
            count = len(b["csi"])
            losses += float(loss.detach()) * count
            joints_seen += count
        val, _, _ = _collect(model, val_loader, dev, threshold)
        score = val["normalized_joint_error"]
        row = {
            "epoch": epoch + 1,
            "training_nll_plus_consistency": losses / joints_seen,
            "validation_pck": val["pck"],
            "validation_nje": score,
        }
        history.append(row)
        if score is not None and score < best:
            best, best_epoch = score, epoch + 1
            checkpoint = {
                "schema_version": 1,
                "model": deepcopy(model.state_dict()),
                "config": config,
                "epoch": best_epoch,
                "selected_by": "validation_normalized_joint_error",
                "manifest_sha256": _fingerprint(manifest_path),
                "provenance": manifest.get("provenance", {}),
                "purpose": _purpose(manifest),
                "input_shape": list(train_data.input_shape),
                "joints": train_data.joints,
                "source_sha256": source_fingerprints,
                "runtime_environment": runtime_environment,
                "validation": {k: v for k, v in val.items() if k != "per_frame_nje"},
                "normalization": "causal-window self statistics"
                if config.get("data", {}).get("normalize", True)
                else "none",
            }
            best_checkpoint = checkpoint
        # Commit the full completed-epoch state before publishing its derived
        # best/history artifacts. A crash between these writes is recoverable.
        epoch_state = {
            "schema_version": 1,
            "kind": "ghostmesh_completed_epoch",
            "run_identity": identity,
            "completed_epoch": epoch + 1,
            "model": deepcopy(model.state_dict()),
            "teacher": deepcopy(teacher.state_dict()),
            "optimizer": deepcopy(optimizer.state_dict()),
            "history": deepcopy(history),
            "best_score": best,
            "best_epoch": best_epoch,
            "best_checkpoint": deepcopy(best_checkpoint),
            "rng": _capture_rng(generator, dev),
            "elapsed_seconds": elapsed_before + time.perf_counter() - started,
            "input_shape": list(train_data.input_shape),
            "joints": train_data.joints,
            "recovery_files": sorted(set(recovery_files)),
        }
        _atomic_torch_save(epoch_state, output / "latest.pt")
        if best_checkpoint is not None:
            _atomic_torch_save(best_checkpoint, output / "best.pt")
        _json(
            output / "history.json", {"epochs": history, "purpose": _purpose(manifest)}
        )
        print(
            f"epoch={epoch + 1} loss={row['training_nll_plus_consistency']:.5f} validation_nje={score}",
            flush=True,
        )
    if best_epoch is None:
        raise RuntimeError(
            "Validation has no usable 2D pose boxes; no checkpoint selected"
        )
    if resume and (output / "training_report.json").exists():
        preserved = _preserve_for_recovery(output / "training_report.json", output)
        if preserved is not None:
            recovery_files.append(preserved)
    report = {
        "status": "trained",
        "purpose": _purpose(manifest),
        "seed": seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "validation_nje": best,
        "selection": "validation only; test samples never decoded by training (manifest integrity hashes are verified)",
        "train_windows": len(train_data),
        "validation_windows": len(val_data),
        "training_coverage": _coverage(train_data),
        "validation_coverage": _coverage(val_data),
        "source_sha256": source_fingerprints,
        "runtime_environment": runtime_environment,
        "training_seconds": elapsed_before + time.perf_counter() - started,
        "device": str(dev),
        "timing_note": "Completed training time; excludes discarded partial epochs and interruption downtime",
        "resume": {
            "requested": bool(resume),
            "from_completed_epoch": start_epoch,
            "seed_restart": bool(resume and latest is None),
            "recovery_files": sorted(set(recovery_files)),
        },
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "manifest_sha256": _fingerprint(manifest_path),
        "checkpoint_sha256": _fingerprint(output / "best.pt"),
        "config_source_sha256": identity["config_source_sha256"],
        "config": config,
        "history": history,
    }
    _json(output / "training_report.json", report)
    return report


def load_checkpoint(path, device="cpu"):
    dev = _device(device)
    checkpoint = torch.load(path, map_location=dev, weights_only=True)
    if checkpoint.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    model = build_model(checkpoint["config"]).to(dev)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def benchmark(model, example, device, repeats=30):
    """Batch-one model-only latency excludes preprocessing, I/O and rendering."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    model.eval()
    x = example[:1].to(device)
    for _ in range(5):
        model(x)
    _sync(device)
    values = []
    for _ in range(repeats):
        _sync(device)
        begin = time.perf_counter()
        model(x)
        _sync(device)
        values.append((time.perf_counter() - begin) * 1000)
    return {
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "model_fps": float(1000 / np.median(values)),
        "batch_size": 1,
        "warmup": 5,
        "repeats": repeats,
        "device": str(device),
        "scope": "model forward only; excludes preprocessing, disk, network and rendering",
    }


def evaluate(
    manifest_path, checkpoint_path, split="test", device="cpu", output=None
) -> dict:
    if split not in ("validation", "test"):
        raise ValueError("Evaluation split must be validation or test")
    if output is not None and Path(output).exists():
        raise FileExistsError(f"Refusing to overwrite evaluation report: {output}")
    manifest = load_manifest(manifest_path)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    if _fingerprint(manifest_path) != checkpoint["manifest_sha256"]:
        raise ValueError(
            "Manifest differs from training; create an explicit new protocol experiment"
        )
    validate_protocol(manifest, checkpoint["config"].get("protocol", "cross_subject"))
    dataset = _dataset(manifest_path, split, checkpoint["config"])
    if (
        tuple(dataset.input_shape) != tuple(checkpoint["input_shape"])
        or dataset.joints != checkpoint["joints"]
    ):
        raise ValueError("Evaluation sensor/skeleton layout differs from checkpoint")
    loader = DataLoader(
        dataset,
        batch_size=checkpoint["config"].get("training", {}).get("batch_size", 32),
        shuffle=False,
    )
    dev = _device(device)
    torch.set_num_threads(int(checkpoint["config"].get("threads", 2)))
    threshold = float(
        checkpoint["config"].get("evaluation", {}).get("pck_threshold", 0.2)
    )
    report, arrays, metadata = _collect(model, loader, dev, threshold)
    per_frame = report.pop("per_frame_nje")
    groups = defaultdict(list)
    for i, row in enumerate(metadata):
        for key in ("subject", "environment", "sequence"):
            if key in row:
                groups[(key, row[key])].append(i)
    domains = {}
    for (key, value), indices in groups.items():
        sub = pose_metrics(
            *(a[indices] for a in arrays[:3]),
            log_sigma=arrays[3][indices],
            threshold=threshold,
        )
        sub.pop("per_frame_nje")
        domains.setdefault(key, {})[value] = sub
    ranked = sorted(((v, i) for i, v in enumerate(per_frame) if v is not None))

    def examples(items):
        return [{**metadata[i], "normalized_joint_error": float(v)} for v, i in items]

    purpose = _purpose(manifest)
    report.update(
        {
            "status": "evaluated",
            "purpose": purpose,
            "split": split,
            "protocol": checkpoint["config"].get("protocol", "cross_subject"),
            "generalization_claim": "not established by synthetic fixtures"
            if purpose == "software_check_only"
            else "report measured held-out scores and compare matched baseline before claiming improvement",
            "checkpoint_epoch": checkpoint["epoch"],
            "checkpoint_selected_by": checkpoint["selected_by"],
            "checkpoint_sha256": _fingerprint(checkpoint_path),
            "manifest_sha256": _fingerprint(manifest_path),
            "domains": domains,
            "qualitative_candidates": {
                "lowest_error": examples(ranked[:5]),
                "highest_error": examples(ranked[-5:][::-1]),
            },
            "latency": benchmark(model, dataset[0]["csi"].unsqueeze(0), dev),
            "windows": len(dataset),
            "parameters": sum(p.numel() for p in model.parameters()),
            "coverage": _coverage(dataset, report),
            **_evaluation_provenance(checkpoint),
            "sequence_subjects": {
                r["id"]: r["subject"]
                for r in manifest["sequences"]
                if r["split"] == split
            },
        }
    )
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        _json(Path(output), report)
    return report


def evaluate_mean_pose(manifest_path, config_path, split="test", output=None) -> dict:
    """Fit a no-signal mean pose using training targets only, then evaluate.

    Uses exactly the neural model's complete causal windows and validity masks.
    Training mean/variance are frozen before the held-out split is constructed.
    """
    if split not in ("test", "validation"):
        raise ValueError("Evaluation split must be validation or test")
    if output is not None and Path(output).exists():
        raise FileExistsError(f"Refusing to overwrite evaluation report: {output}")
    manifest = load_manifest(manifest_path)
    config = json.loads(Path(config_path).read_text())
    validate_protocol(manifest, config.get("protocol", "cross_subject"))
    training = _dataset(manifest_path, "train", config)
    labels, masks = [], []
    for index in range(len(training)):
        row = training[index]
        labels.append(row["pose"].numpy())
        masks.append(row["valid"].numpy())
    labels, masks = np.stack(labels), np.stack(masks).astype(bool)
    count = masks.sum(0)
    if (count == 0).any():
        raise ValueError(
            "Mean-pose baseline lacks training labels for at least one joint"
        )
    mean = np.where(masks[..., None], labels, 0).sum(0) / count[:, None]
    residual2 = ((labels - mean) ** 2).sum(-1)
    sigma2 = (np.where(masks, residual2, 0).sum(0) / count / 2).clip(1e-6)
    log_sigma = 0.5 * np.log(sigma2)
    heldout = _dataset(manifest_path, split, config)
    targets, valid, metadata = [], [], []
    for index in range(len(heldout)):
        row = heldout[index]
        targets.append(row["pose"].numpy())
        valid.append(row["valid"].numpy())
        metadata.append(
            {
                key: row[key]
                for key in ("sequence", "subject", "environment", "time", "frame_index")
                if key in row
            }
        )
    target, mask = np.stack(targets), np.stack(valid)
    prediction = np.broadcast_to(mean, target.shape)
    uncertainty = np.broadcast_to(log_sigma, mask.shape)
    threshold = float(config.get("evaluation", {}).get("pck_threshold", 0.2))
    report = pose_metrics(prediction, target, mask, uncertainty, threshold)
    per_frame = report.pop("per_frame_nje")
    domains, groups = {}, defaultdict(list)
    for index, row in enumerate(metadata):
        for key in ("sequence", "subject", "environment"):
            if key in row:
                groups[(key, row[key])].append(index)
    for (key, value), indices in groups.items():
        scores = pose_metrics(
            prediction[indices],
            target[indices],
            mask[indices],
            uncertainty[indices],
            threshold,
        )
        scores.pop("per_frame_nje")
        domains.setdefault(key, {})[value] = scores
    report.update(
        {
            "status": "evaluated",
            "purpose": _purpose(manifest),
            "baseline": "train-mean-pose-no-signal",
            "split": split,
            "protocol": config.get("protocol", "cross_subject"),
            "manifest_sha256": _fingerprint(manifest_path),
            "config_sha256": _fingerprint(config_path),
            "windows": len(heldout),
            "training_windows": len(training),
            "train_mean_pose": mean.tolist(),
            "coverage": _coverage(heldout, report),
            "training_coverage": _coverage(training),
            **_evaluation_provenance(),
            "train_sigma": np.sqrt(sigma2).tolist(),
            "training_joint_counts": count.tolist(),
            "fit": "training targets only; no CSI features, action IDs or held-out labels used to fit predictions",
            "domains": domains,
            "per_frame_nje": per_frame,
            "sequence_subjects": {
                r["id"]: r["subject"]
                for r in manifest["sequences"]
                if r["split"] == split
            },
        }
    )
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        _json(Path(output), report)
    return report


def compare_reports(
    reference: dict, candidate: dict, seed=2027, repeats=2000, cluster="sequence"
) -> dict:
    """Paired block bootstrap of sequence-level means, never independent frames.

    Delta is candidate minus reference. Negative NJE / positive PCK favors the
    candidate. A subject-cluster sensitivity analysis resamples people while
    retaining all their sequence means. This is sampling uncertainty for one
    fitted checkpoint pair, not training-seed variability.
    """
    if repeats < 100 or cluster not in ("sequence", "subject"):
        raise ValueError("Use >=100 bootstrap repeats and sequence or subject clusters")
    for key in (
        "manifest_sha256",
        "split",
        "protocol",
        "pck_threshold",
        "purpose",
        "windows",
    ):
        if reference.get(key) != candidate.get(key):
            raise ValueError(f"Cannot pair reports with different {key}")
    a = reference.get("domains", {}).get("sequence", {})
    b = candidate.get("domains", {}).get("sequence", {})
    if set(a) != set(b) or not a:
        raise ValueError(
            "Paired reports need identical nonempty held-out sequence sets"
        )
    ids = sorted(a)
    if any(a[k]["valid_joints"] != b[k]["valid_joints"] for k in ids):
        raise ValueError("Paired reports have different validity/target coverage")
    ids = [
        k
        for k in ids
        if a[k]["normalized_joint_error"] is not None
        and b[k]["normalized_joint_error"] is not None
    ]
    if len(ids) < 2:
        raise ValueError(
            "At least two usable sequence blocks required for a confidence interval"
        )
    mapping = candidate.get("sequence_subjects") or reference.get("sequence_subjects")
    if cluster == "subject" and not mapping:
        raise ValueError("Subject-cluster analysis requires sequence_subjects metadata")
    groups = defaultdict(list)
    for index, sequence in enumerate(ids):
        groups[mapping[sequence] if cluster == "subject" else sequence].append(index)
    blocks = list(groups.values())
    if len(blocks) < 2:
        raise ValueError("At least two independent clusters required")
    rng = np.random.default_rng(seed)
    indices = [
        np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
        for _ in range(repeats)
    ]
    result = {
        "comparison": "candidate minus reference",
        "cluster": cluster,
        "clusters": len(blocks),
        "sequences": len(ids),
        "bootstrap_repeats": repeats,
        "bootstrap_seed": seed,
        "weighting": "equal weight per sequence mean; frames never independently resampled",
        "ci": "paired percentile bootstrap 95%",
        "limits": "Intervals reflect held-out sampling for fixed checkpoints, not training-seed variability. Sequence blocks can remain correlated within a person; report subject-cluster sensitivity. Small cluster counts give unstable intervals.",
    }
    for key in ("normalized_joint_error", "pck"):
        differences = np.array([b[k][key] - a[k][key] for k in ids])
        bootstrap = np.array([differences[index].mean() for index in indices])
        result[key] = {
            "delta": float(differences.mean()),
            "ci95": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
            "reference_sequence_mean": float(np.mean([a[k][key] for k in ids])),
            "candidate_sequence_mean": float(np.mean([b[k][key] for k in ids])),
        }
    return result
