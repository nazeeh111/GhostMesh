"""Bounded structured 2D representation screen on E01-E03, or synthetic GPU check.

Research execution is explicit; --benchmark opens no dataset or research cache.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.engine import _json, seed_everything  # noqa: E402
from ghostmesh.metrics import pose_metrics  # noqa: E402
from ghostmesh.structured_packet import StructuredPacketPose, structured_loss  # noqa: E402

CONFIG = {
    "epochs": 8,
    "seed": 2027,
    "batch_size": 64,
    "width": 96,
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "training_endpoint_subsample": 4,
    "source_validation_subjects": ["S04", "S14", "S24"],
    "checkpoint_selection": "source validation NJE only",
    "architecture": "stack 8 frames; fixed CNN 114x10->38x5->19x5; 95 tokens; two 4-head attention blocks; 17 cross-attended joint queries; two COCO adjacency refinements",
    "candidate_objective": "bbox-normalized separate relative/root SmoothL1 beta .1 + .1 relative bone-length SmoothL1 + .1 detached-pose Gaussian NLL for each factor",
    "root_gradient": "Root supervision stops at root readout; no root label gradients into structure encoder/decoder",
    "scope": "DEVELOPMENT ONLY E01-E03; 2D supplied teacher targets, no E04/final holdout, no physical packet timing/AoA or SOTA claim",
    "prior_art": "Original implementation informed by standard attention/graph principles and RePos factorization https://arxiv.org/abs/2607.02986",
}


def digest(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def benchmark(out, device):
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(1)
    seed_everything(CONFIG["seed"])
    rows = []
    for channels in (3, 7):
        for factorized in (False, True):
            model = StructuredPacketPose(factorized, channels=channels)
            y = torch.rand(64, 17, 2) * 0.4 + 0.2
            model.initialize_from_source(y)
            model.to(device)
            x, y = torch.randn(64, 8, channels, 114, 10, device=device), y.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            timings = []
            for step in range(8):
                if device == "mps":
                    torch.mps.synchronize()
                started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                loss = structured_loss(model(x), y, factorized)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite synthetic objective")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if device == "mps":
                    torch.mps.synchronize()
                timings.append(time.perf_counter() - started)
            rows.append(
                {
                    "channels": channels,
                    "factorized": factorized,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "first_step_seconds": timings[0],
                    "steady_step_median_seconds": float(np.median(timings[3:])),
                    "steady_step_p95_seconds": float(np.percentile(timings[3:], 95)),
                    "batch_size": 64,
                }
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    _json(
        out,
        {
            "scope": "SOFTWARE ONLY: random tensors, no data or accuracy result",
            "device": device,
            "torch": str(torch.__version__),
            "platform": platform.platform(),
            "rows": rows,
            "source_sha256": {
                str(p.relative_to(ROOT)): digest(p)
                for p in (Path(__file__), ROOT / "ghostmesh/structured_packet.py")
            },
        },
    )
    print(json.dumps(rows, indent=2), flush=True)


class RawSet(Dataset):
    def __init__(self, raw, pose, indices, zero_phase=False, phase=None):
        self.phase = phase
        self.raw, self.pose, self.indices, self.zero_phase = (
            raw,
            pose,
            indices,
            zero_phase,
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        x = self.raw[i].copy()
        if self.phase is not None:
            extra = (
                np.zeros_like(self.phase[i])
                if self.zero_phase
                else self.phase[i].copy()
            )
            x = np.concatenate((x, extra), axis=1)
        elif self.zero_phase:
            x[:, 3:] = 0
        return torch.from_numpy(x.astype(np.float32, copy=False)), torch.from_numpy(
            self.pose[i]
        )


def motion_metrics(prediction, target, sequence):
    """Post-fit analysis: recording centering uses labels, never model inputs."""
    output = {}

    def parts(x):
        root = x[:, [11, 12]].mean(1, keepdims=True)
        return {"absolute": x, "root": root, "relative": x - root}

    pp, yy = parts(prediction.astype(float)), parts(target.astype(float))
    for component in pp:
        centered_p, centered_y, positive, records = [], [], 0, []
        for key in sorted(set(sequence)):
            ix = sequence == key
            p, y = pp[component][ix], yy[component][ix]
            p, y = p - p.mean(0), y - y.mean(0)
            denominator = float((y * y).sum())
            r2 = (
                1 - float(((p - y) ** 2).sum()) / denominator
                if denominator > 0
                else None
            )
            positive += r2 is not None and r2 > 0
            records.append({"sequence": str(key), "r2": r2, "windows": int(ix.sum())})
            centered_p.append(p)
            centered_y.append(y)
        p, y = np.concatenate(centered_p), np.concatenate(centered_y)
        denominator = float((y * y).sum())
        output[component] = {
            "within_record_r2": 1 - float(((p - y) ** 2).sum()) / denominator
            if denominator
            else None,
            "prediction_to_teacher_variance": float((p * p).sum()) / denominator
            if denominator
            else None,
            "alignment_cosine": float(
                (p * y).sum() / np.sqrt((p * p).sum() * denominator)
            )
            if (p * p).sum() > 0 and denominator > 0
            else None,
            "positive_records": positive,
            "records": records,
        }
    return output


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    predictions, targets, sigmas = [], [], []
    for x, y in loader:
        out = model(x.to(device))
        predictions.append(out["pose"].cpu().numpy())
        targets.append(y.numpy())
        sigmas.append(out["log_sigma"].cpu().numpy())
    p, y, sigma = (
        np.concatenate(predictions),
        np.concatenate(targets),
        np.concatenate(sigmas),
    )
    metrics = pose_metrics(p, y, np.ones(y.shape[:2], bool), sigma)
    metrics.pop("per_frame_nje")
    return metrics, p


def run(reference, raw_cache, out, device, zero_phase=False, phase_cache=None):
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(1)
    receipt, raw_receipt = (
        json.loads((reference / "receipt.json").read_text()),
        json.loads((raw_cache / "receipt.json").read_text()),
    )
    if (
        digest(reference / "receipt.json")
        != raw_receipt["reference_cache_receipt_sha256"]
        or digest(raw_cache / "raw.npy") != raw_receipt["raw_sha256"]
    ):
        raise ValueError("Raw/reference cache identity differs")
    a = {}
    for key in ("pose", "valid", "room", "subject", "sequence", "end"):
        path = reference / (key + ".npy")
        if digest(path) != receipt["arrays"][path.name]:
            raise ValueError("Reference array changed")
        a[key] = np.load(path, allow_pickle=False)
    if set(a["room"]) != {"E01", "E02", "E03"} or not a["valid"].all():
        raise ValueError("E01-E03 complete development targets required")
    raw = np.load(raw_cache / "raw.npy", mmap_mode="r", allow_pickle=False)
    if (
        list(raw.shape) != raw_receipt["shape"]
        or raw.shape[:2] != (len(a["pose"]), 8)
        or raw.shape[2] not in {3, 7}
        or raw.shape[3:] != (114, 10)
    ):
        raise ValueError("Unexpected aligned raw cache shape")
    phase, phase_receipt = None, None
    if phase_cache is not None:
        phase_receipt = json.loads((phase_cache / "receipt.json").read_text())
        if (
            digest(reference / "receipt.json")
            != phase_receipt["reference_cache_receipt_sha256"]
            or digest(phase_cache / "phase.npy") != phase_receipt["phase_sha256"]
            or set(phase_receipt["raw_sequence_ordered_file_hashes"])
            != set(receipt["sequences"])
        ):
            raise ValueError("Phase/reference content or sequence alignment differs")
        phase = np.load(phase_cache / "phase.npy", mmap_mode="r", allow_pickle=False)
        if (
            list(phase.shape) != phase_receipt["shape"]
            or phase.shape != (len(a["pose"]), 8, 4, 114, 10)
            or raw.shape[2] != 3
        ):
            raise ValueError("Phase must supply exactly four aligned circular channels")
    if zero_phase and raw.shape[2] != 7 and phase is None:
        raise ValueError("Phase-zero ablation requires seven-channel inputs")
    channels = 7 if phase is not None else raw.shape[2]
    modalities = (
        [("amplitude_zero_phase", True), ("amplitude_relative_phase", False)]
        if phase is not None
        else [("amplitude", zero_phase)]
    )
    source_paths = [
        Path(__file__),
        ROOT / "ghostmesh/structured_packet.py",
        ROOT / "scripts/domain_raw_cache.py",
    ] + [
        ROOT / "ghostmesh" / n
        for n in ("engine.py", "models.py", "data.py", "preprocess.py", "metrics.py")
    ]
    for name, expected in raw_receipt.get("source_sha256", {}).items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or digest(path) != expected:
            raise ValueError("Raw-cache builder source changed")
        source_paths.append(path)
    if phase_receipt is not None:
        for name, expected in phase_receipt["sources"].items():
            path = (ROOT / name).resolve()
            if not path.is_relative_to(ROOT) or digest(path) != expected:
                raise ValueError("Phase-cache builder source changed")
            source_paths.append(path)
    sources = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    config = {
        **CONFIG,
        "device": device,
        "channels": channels,
        "modalities": modalities,
        "amplitude_baseline_note": "Seven-channel zero-phase control has identical parameter count, but weights on its four unused input channels receive no data gradient. Phase channels never receive amplitude log/quality preprocessing.",
        "runtime": {
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    out.mkdir(parents=True)
    _json(
        out / "registration.json",
        {
            "config": config,
            "sources": sources,
            "reference_receipt_sha256": digest(reference / "receipt.json"),
            "reference_arrays_sha256": receipt["arrays"],
            "raw_receipt_sha256": digest(raw_cache / "receipt.json"),
            "raw_sha256": raw_receipt["raw_sha256"],
            "phase_receipt_sha256": digest(phase_cache / "receipt.json")
            if phase_cache is not None
            else None,
            "phase_sha256": phase_receipt["phase_sha256"]
            if phase_receipt is not None
            else None,
        },
    )
    for name, expected in sources.items():
        content = (ROOT / name).read_bytes()
        if sha256(content).hexdigest() != expected:
            raise ValueError("Source drift before fitting")
        destination = out / "executed-source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    results, priors = [], []
    for room in ("E01", "E02", "E03"):
        source = a["room"] != room
        val = source & np.isin(a["subject"], CONFIG["source_validation_subjects"])
        train = source & ~val
        indices, held = np.flatnonzero(train)[::4], ~source
        if set(a["subject"][indices]) & set(a["subject"][val]):
            raise ValueError("Source subject leakage")

        def loader(mask, shuffle=False, omit_phase=False):
            ix = np.flatnonzero(mask) if mask.dtype == bool else mask
            return DataLoader(
                RawSet(raw, a["pose"], ix, omit_phase, phase),
                batch_size=64,
                shuffle=shuffle,
                generator=torch.Generator().manual_seed(CONFIG["seed"]),
            )

        for modality, omit_phase in modalities:
            for factorized in (False, True):
                folder = (
                    out
                    / f"{room}-{modality}-{'factorized' if factorized else 'direct'}"
                )
                folder.mkdir()
                history = []
                try:
                    seed_everything(CONFIG["seed"])
                    model = StructuredPacketPose(factorized, channels=channels)
                    model.initialize_from_source(a["pose"][indices])
                    model.to(device)
                    optimizer = torch.optim.AdamW(
                        model.parameters(), lr=0.001, weight_decay=0.01
                    )
                    train_loader, val_loader = (
                        loader(indices, True, omit_phase),
                        loader(val, omit_phase=omit_phase),
                    )
                    best, chosen, best_epoch, started = (
                        float("inf"),
                        None,
                        None,
                        time.monotonic(),
                    )
                    for epoch in range(1, 9):
                        model.train()
                        losses = []
                        for x, y in train_loader:
                            optimizer.zero_grad(set_to_none=True)
                            loss = structured_loss(
                                model(x.to(device)), y.to(device), factorized
                            )
                            if not torch.isfinite(loss):
                                raise ValueError("Nonfinite objective")
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                            losses.append(float(loss.detach().cpu()))
                        validation, _ = evaluate(model, val_loader, device)
                        score = validation["normalized_joint_error"]
                        history.append(
                            {
                                "epoch": epoch,
                                "training_loss": float(np.mean(losses)),
                                "source_validation": validation,
                            }
                        )
                        if score < best:
                            best, best_epoch = score, epoch
                            chosen = {
                                k: v.detach().cpu().clone()
                                for k, v in model.state_dict().items()
                            }
                        _json(folder / "history.json", {"history": history})
                        print(
                            json.dumps(
                                {
                                    "room": room,
                                    "modality": modality,
                                    "factorized": factorized,
                                    "epoch": epoch,
                                    "source_val_nje": score,
                                }
                            ),
                            flush=True,
                        )
                    model.load_state_dict(chosen)
                    torch.save(
                        {
                            "state_dict": chosen,
                            "config": config,
                            "factorized": factorized,
                            "sources": sources,
                        },
                        folder / "best.pt",
                    )
                    metrics, prediction = evaluate(
                        model, loader(held, omit_phase=omit_phase), device
                    )
                    np.save(
                        folder / "held-development-prediction.npy",
                        prediction,
                        allow_pickle=False,
                    )
                    report = {
                        "room": room,
                        "modality": modality,
                        "factorized": factorized,
                        "selected_epoch": best_epoch,
                        "metrics": metrics,
                        "motion": motion_metrics(
                            prediction, a["pose"][held], a["sequence"][held]
                        ),
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "seconds": time.monotonic() - started,
                        "counts": {
                            "training": len(indices),
                            "source_validation": int(val.sum()),
                            "held_development": int(held.sum()),
                        },
                        "training_indices_sha256": sha256(
                            indices.astype("<i8").tobytes()
                        ).hexdigest(),
                        "held_indices_sha256": sha256(
                            np.flatnonzero(held).astype("<i8").tobytes()
                        ).hexdigest(),
                        "checkpoint_sha256": digest(folder / "best.pt"),
                        "prediction_sha256": digest(
                            folder / "held-development-prediction.npy"
                        ),
                    }
                    _json(folder / "result.json", report)
                    results.append(report)
                    print(
                        json.dumps(
                            {
                                "completed": room,
                                "modality": modality,
                                "factorized": factorized,
                                "nje": metrics["normalized_joint_error"],
                                "relative_motion_r2": report["motion"]["relative"][
                                    "within_record_r2"
                                ],
                            }
                        ),
                        flush=True,
                    )
                except Exception as error:
                    _json(
                        folder / "failure.json",
                        {
                            "type": type(error).__name__,
                            "error": str(error),
                            "history": history,
                        },
                    )
                    raise

        # Static prior fits exactly the same quarter-subsampled source training targets.
        mean = a["pose"][indices].mean(0)
        prior_prediction = np.broadcast_to(mean, a["pose"][held].shape)
        prior = pose_metrics(prior_prediction, a["pose"][held], a["valid"][held])
        prior.pop("per_frame_nje")
        priors.append(
            {
                "room": room,
                "metrics": prior,
                "mean_pose": mean.tolist(),
                "training_windows": len(indices),
                "source_subjects": sorted(set(a["subject"][indices])),
                "fit": "same source-only quarter endpoints; no validation/held labels used for fitting",
            }
        )
        _json(out / "matched-source-mean.json", {"priors": priors})

    if (
        sources != {name: digest(ROOT / name) for name in sources}
        or digest(raw_cache / "raw.npy") != raw_receipt["raw_sha256"]
        or (
            phase_receipt is not None
            and digest(phase_cache / "phase.npy") != phase_receipt["phase_sha256"]
        )
    ):
        raise ValueError("Source/raw/phase input changed during execution")
    _json(
        out / "results.json",
        {
            "config": config,
            "results": results,
            "sources": sources,
            "matched_source_mean": priors,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--raw-cache", type=Path)
    parser.add_argument("--zero-phase", action="store_true")
    parser.add_argument(
        "--phase-cache",
        type=Path,
        help="Aligned separate circular phase cache; runs both zero-phase and actual-phase modalities",
    )
    args = parser.parse_args()
    if args.benchmark:
        benchmark(args.out.resolve(), args.device)
    elif args.reference is None or args.raw_cache is None:
        parser.error("Research execution requires --reference and --raw-cache")
    else:
        run(
            args.reference.resolve(),
            args.raw_cache.resolve(),
            args.out.resolve(),
            args.device,
            args.zero_phase,
            args.phase_cache.resolve() if args.phase_cache is not None else None,
        )
