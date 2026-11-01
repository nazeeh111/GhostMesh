"""Fixed four-mode typed-RF development screen, or software-only MPS benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time
from hashlib import sha256

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.engine import _json, seed_everything  # noqa: E402
from ghostmesh.metrics import pose_metrics  # noqa: E402
from ghostmesh.rf_pose import RFPose, rf_loss  # noqa: E402
from scripts.domain_structured_probe import digest, motion_metrics  # noqa: E402

MODES = ("legacy", "db", "db_phase", "quotient")
INITIAL_MODES = ("legacy", "quotient")
CONFIG = {
    "epochs": 12,
    "seed": 2027,
    "batch_size": 128,
    "learning_rate": 0.0003,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "source_stride": 8,
    "training_endpoint_subsample": 1,
    "temporal_kernel": "native",
    "dense_supervision": "All 8 past/current frame labels per source window; 63936 source frames per fold, 7992 nonoverlapping windows",
    "validation_evaluation": "Last endpoint only, complete 8-frame context",
    "clip_groups": "Three separate norm1 groups: body geometry, root geometry, detached uncertainty heads",
    "source_validation_subjects": ["S04", "S14", "S24"],
    "checkpoint_selection": "source validation NJE only",
    "body_channels": 7,
    "root_channels": 3,
    "architecture": "packet798->128->64, learned ten packet positions/attention, causal TCN64 d1/2/4; completely separate packet-mean342->128->64 root MLP",
    "loss": "bbox-normalized relative/root Huber beta.1 + .1 bone + .1 uncertainty NLL; uncertainty latents/residuals detached",
    "clipping": "Final composed pose clipped to [0,1] for reported metrics; raw predictions/root/relative components and coordinate clipping fraction retained",
    "modes": {
        "legacy": "per-frame legacy log1p(DB)3 + zero4 body; fixed raw DB342 root spectrum",
        "db": "per-frame typed DB3 + zero4 body; same fixed DB342 root",
        "db_phase": "per-frame typed DB3 + four quotient phasors body; same fixed DB342 root",
        "quotient": "two log-magnitude ratios and four phasors + zero1 body; same fixed DB342 root",
    },
    "unused_weights": "legacy/db have four all-zero body input channels; quotient has one. Equal shape/parameter count does not give these weights data gradients.",
    "units": "inferred log magnitude relative to integer receiver I/Q; not RSSI dBm or calibrated AoA",
    "scope": "DEVELOPMENT ONLY E01-E03; no E04/final records; 2D teacher annotations, no novelty/SOTA or causal-motion claim",
    "prior_art": [
        "https://arxiv.org/abs/1907.03994",
        "https://arxiv.org/abs/2607.02986",
    ],
}


class RFSet(Dataset):
    def __init__(
        self,
        legacy,
        db,
        quotient,
        root_spectrum,
        pose_sequence,
        indices,
        mode,
        dense=False,
    ):
        if mode not in MODES:
            raise ValueError("Unknown RF mode")
        self.legacy, self.db, self.quotient = legacy, db, quotient
        self.root_spectrum, self.pose, self.indices, self.mode, self.dense = (
            root_spectrum,
            pose_sequence,
            indices,
            mode,
            dense,
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        amplitude = self.legacy[i] if self.mode == "legacy" else self.db[i]
        body = np.zeros((8, 7, 114, 10), dtype="float32")
        if self.mode == "quotient":
            body[:, :6] = self.quotient[i]
        else:
            body[:, :3] = amplitude
            if self.mode == "db_phase":
                body[:, 3:] = self.quotient[i, :, 2:]
        target = self.pose[i].copy() if self.dense else self.pose[i, -1].copy()
        return (
            torch.from_numpy(body),
            torch.from_numpy(self.root_spectrum[i].copy()),
            torch.from_numpy(target),
        )


def clip_groups(model):
    groups = {"body_geometry": [], "root_geometry": [], "uncertainty": []}
    for name, parameter in model.named_parameters():
        group = (
            "uncertainty"
            if "uncertainty" in name
            else "root_geometry"
            if name.startswith(("root_encoder.", "root_head."))
            else "body_geometry"
        )
        groups[group].append(parameter)
    for parameters in groups.values():
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)


def benchmark(out, device):
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(1)
    seed_everything(CONFIG["seed"])
    y = torch.rand(128, 8, 17, 2) * 0.4 + 0.2
    model = RFPose()
    model.initialize_from_source(y)
    model.to(device)
    body = torch.randn(128, 8, 7, 114, 10, device=device)
    root, y = torch.randn(128, 8, 342, device=device), y.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
    timing = []
    for _ in range(8):
        if device == "mps":
            torch.mps.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = rf_loss(model(body, root, return_sequence=True), y)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite software objective")
        loss.backward()
        clip_groups(model)
        optimizer.step()
        if device == "mps":
            torch.mps.synchronize()
        timing.append(time.perf_counter() - started)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "SOFTWARE ONLY random tensors; no cache/data read",
        "device": device,
        "parameters": sum(p.numel() for p in model.parameters()),
        "batch_size": 128,
        "supervised_frames_per_batch": 1024,
        "first_step_seconds": timing[0],
        "steady_median_seconds": float(np.median(timing[3:])),
        "steady_p95_seconds": float(np.percentile(timing[3:], 95)),
        "torch": str(torch.__version__),
        "source_sha256": {
            str(p.relative_to(ROOT)): digest(p)
            for p in (Path(__file__), ROOT / "ghostmesh/rf_pose.py")
        },
    }
    _json(out, report)
    print(json.dumps(report, indent=2), flush=True)


@torch.no_grad()
def evaluate(model, loader, device, collect=False):
    model.eval()
    ps, raw, ys, sigmas, roots, relatives = [], [], [], [], [], []
    for body, root, target in loader:
        result = model(body.to(device), root.to(device))
        ps.append(result["pose"].cpu().numpy())
        raw.append(result["raw_pose"].cpu().numpy())
        ys.append(target.numpy())
        sigmas.append(result["log_sigma"].cpu().numpy())
        roots.append(result["root"].cpu().numpy())
        relatives.append(result["relative"].cpu().numpy())
    p, r, y, sigma = (
        np.concatenate(ps),
        np.concatenate(raw),
        np.concatenate(ys),
        np.concatenate(sigmas),
    )
    report = pose_metrics(p, y, np.ones(y.shape[:2], bool), sigma)
    report.pop("per_frame_nje")
    report["coordinate_clipping_fraction"] = float(((r < 0) | (r > 1)).mean())
    report["raw_pose_nje"] = pose_metrics(r, y, np.ones(y.shape[:2], bool))[
        "normalized_joint_error"
    ]
    truth_root = y[:, [11, 12]].mean(1, keepdims=True)
    report["raw_root_coordinate_error"] = float(
        np.linalg.norm(np.concatenate(roots) - truth_root, axis=-1).mean()
    )
    report["raw_relative_coordinate_error"] = float(
        np.linalg.norm(np.concatenate(relatives) - (y - truth_root), axis=-1).mean()
    )
    return report, {
        "pose": p,
        "raw_pose": r,
        "root": np.concatenate(roots),
        "relative": np.concatenate(relatives),
    } if collect else None


def run(reference, stream_cache, rf_cache, out, device, modes=INITIAL_MODES):
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(1)
    ref = json.loads((reference / "receipt.json").read_text())
    stream_receipt = json.loads((stream_cache / "receipt.json").read_text())
    rf = json.loads((rf_cache / "receipt.json").read_text())
    ref_sha = digest(reference / "receipt.json")
    if (
        stream_receipt["reference_receipt_sha256"] != ref_sha
        or rf["reference_receipt_sha256"] != ref_sha
    ):
        raise ValueError("Caches do not bind the same endpoint reference")
    arrays = {}
    for key in ("pose", "valid", "room", "subject", "sequence", "end"):
        path = reference / (key + ".npy")
        if digest(path) != ref["arrays"][path.name]:
            raise ValueError("Reference array changed")
        arrays[key] = np.load(path, allow_pickle=False)
    if set(arrays["room"]) != {"E01", "E02", "E03"} or not arrays["valid"].all():
        raise ValueError("Only E01-E03 development targets allowed")
    features = {}
    for key, expected in stream_receipt["array_sha256"].items():
        path = stream_cache / (key + ".npy")
        if digest(path) != expected:
            raise ValueError("Stream array changed")
        features[key] = np.load(path, mmap_mode="r", allow_pickle=False)
    if digest(rf_cache / "quotient.npy") != rf["array_sha256"]["quotient"]:
        raise ValueError("Pointwise quotient array changed")
    features["quotient"] = np.load(
        rf_cache / "quotient.npy", mmap_mode="r", allow_pickle=False
    )
    n = len(arrays["pose"])
    shapes = {
        "legacy": (n, 8, 3, 114, 10),
        "db": (n, 8, 3, 114, 10),
        "quotient": (n, 8, 6, 114, 10),
        "root_spectrum": (n, 8, 342),
        "pose_sequence": (n, 8, 17, 2),
        "valid_sequence": (n, 8, 17),
    }
    if (
        any(features[key].shape != shape for key, shape in shapes.items())
        or not features["valid_sequence"].all()
    ):
        raise ValueError("Dense stream shape/masks differ")
    if not np.array_equal(features["pose_sequence"][:, -1], arrays["pose"]):
        raise ValueError("Dense labels do not align to registered endpoints")
    paths = [
        Path(__file__),
        ROOT / "ghostmesh/rf_pose.py",
        ROOT / "scripts/domain_structured_probe.py",
        ROOT / "ghostmesh/structured_packet.py",
    ] + [
        ROOT / "ghostmesh" / key
        for key in ("models.py", "engine.py", "data.py", "preprocess.py", "metrics.py")
    ]
    for receipt in (stream_receipt, rf):
        for name, expected in receipt["source_sha256"].items():
            path = (ROOT / name).resolve()
            if not path.is_relative_to(ROOT) or digest(path) != expected:
                raise ValueError("Cache-builder source changed")
            paths.append(path)
    sources = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    config = {
        **CONFIG,
        "selected_modes": list(modes),
        "device": device,
        "root_input": "Same finite per-frame original DB342 spectrum, fixed (dB-40)/20, for EVERY mode; not RSSI dBm",
        "runtime": {
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    registration = {
        "config": config,
        "sources": sources,
        "reference_sha256": ref_sha,
        "reference_arrays": ref["arrays"],
        "stream_array_sha256": stream_receipt["array_sha256"],
        "stream_receipt_sha256": digest(stream_cache / "receipt.json"),
        "quotient_sha256": rf["array_sha256"]["quotient"],
        "rf_receipt_sha256": digest(rf_cache / "receipt.json"),
        "unit_audit_finite_samples": rf["finite_complex_samples"],
        "unit_audit_integer_samples": rf["integer_iq_at1e8"],
    }
    out.mkdir(parents=True)
    _json(out / "registration.json", registration)
    for name, expected in sources.items():
        content = (ROOT / name).read_bytes()
        if sha256(content).hexdigest() != expected:
            raise ValueError("Source changed before snapshot")
        destination = out / "executed-source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    results, priors = [], []
    for room in ("E01", "E02", "E03"):
        source = arrays["room"] != room
        validation = source & np.isin(
            arrays["subject"], CONFIG["source_validation_subjects"]
        )
        training, held = source & ~validation, ~source
        indices = np.flatnonzero(training)
        if (
            len(indices) != 7992
            or validation.sum() != 1998
            or held.sum() != 4995
            or set(arrays["subject"][training]) & set(arrays["subject"][validation])
        ):
            raise ValueError("Full-source fold/count mismatch")

        def loader(mask, mode, shuffle=False, dense=False):
            ix = np.flatnonzero(mask) if mask.dtype == bool else mask
            return DataLoader(
                RFSet(
                    features["legacy"],
                    features["db"],
                    features["quotient"],
                    features["root_spectrum"],
                    features["pose_sequence"],
                    ix,
                    mode,
                    dense,
                ),
                batch_size=128,
                shuffle=shuffle,
                generator=torch.Generator().manual_seed(CONFIG["seed"]),
            )

        root_states = None
        for mode in modes:
            folder = out / f"{room}-{mode}"
            folder.mkdir()
            history = []
            try:
                seed_everything(CONFIG["seed"])
                model = RFPose(kernel="native")
                model.initialize_from_source(features["pose_sequence"][indices])
                model.to(device)
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=0.0003, weight_decay=0.01
                )
                train_loader, val_loader = (
                    loader(indices, mode, True, True),
                    loader(validation, mode),
                )
                best, selected, best_epoch, started = (
                    float("inf"),
                    None,
                    None,
                    time.monotonic(),
                )
                for epoch in range(1, 13):
                    model.train()
                    losses = []
                    for body, root, target in train_loader:
                        optimizer.zero_grad(set_to_none=True)
                        loss = rf_loss(
                            model(
                                body.to(device), root.to(device), return_sequence=True
                            ),
                            target.to(device),
                        )
                        if not torch.isfinite(loss):
                            raise ValueError("Nonfinite objective")
                        loss.backward()
                        clip_groups(model)
                        optimizer.step()
                        losses.append(float(loss.detach().cpu()))
                    score, _ = evaluate(model, val_loader, device)
                    nje = score["normalized_joint_error"]
                    history.append(
                        {
                            "epoch": epoch,
                            "training_loss": float(np.mean(losses)),
                            "supervised_frames": len(indices) * 8,
                            "source_validation": score,
                        }
                    )
                    if nje < best:
                        best, best_epoch, selected = (
                            nje,
                            epoch,
                            {
                                key: value.detach().cpu().clone()
                                for key, value in model.state_dict().items()
                            },
                        )
                    _json(folder / "history.json", {"history": history})
                    print(
                        json.dumps(
                            {
                                "room": room,
                                "mode": mode,
                                "epoch": epoch,
                                "source_val_nje": nje,
                            }
                        ),
                        flush=True,
                    )
                # Identical root inputs/targets and independent clip groups should make final root geometry identical across modes.
                final_root = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                    if key.startswith(("root_encoder.", "root_head."))
                }
                root_max_difference = (
                    max(
                        float((final_root[key] - root_states[key]).abs().max())
                        for key in final_root
                    )
                    if root_states is not None
                    else 0.0
                )
                if root_states is None:
                    root_states = final_root
                model.load_state_dict(selected)
                torch.save(
                    {
                        "state_dict": selected,
                        "config": config,
                        "mode": mode,
                        "sources": sources,
                    },
                    folder / "best.pt",
                )
                score, prediction = evaluate(model, loader(held, mode), device, True)
                source_score, source_prediction = evaluate(
                    model, loader(training, mode), device, True
                )
                for prefix, values in (
                    ("held", prediction),
                    ("source", source_prediction),
                ):
                    for key, value in values.items():
                        np.save(
                            folder / (prefix + "-" + key + ".npy"),
                            value,
                            allow_pickle=False,
                        )
                motion = motion_metrics(
                    prediction["pose"], arrays["pose"][held], arrays["sequence"][held]
                )
                report = {
                    "room": room,
                    "mode": mode,
                    "selected_epoch": best_epoch,
                    "metrics": score,
                    "source_metrics": source_score,
                    "motion": motion,
                    "raw_motion": motion_metrics(
                        prediction["raw_pose"],
                        arrays["pose"][held],
                        arrays["sequence"][held],
                    ),
                    "parameters": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                    "counts": {
                        "training_windows": len(indices),
                        "training_frames_per_epoch": len(indices) * 8,
                        "source_validation_endpoints": int(validation.sum()),
                        "held_development_endpoints": int(held.sum()),
                    },
                    "seconds": time.monotonic() - started,
                    "final_root_geometry_max_difference_from_first_mode": root_max_difference,
                    "checkpoint_sha256": digest(folder / "best.pt"),
                    "prediction_sha256": {
                        path.name: digest(path) for path in folder.glob("*.npy")
                    },
                    "training_index_sha256": sha256(
                        indices.astype("<i8").tobytes()
                    ).hexdigest(),
                    "held_index_sha256": sha256(
                        np.flatnonzero(held).astype("<i8").tobytes()
                    ).hexdigest(),
                }
                _json(folder / "result.json", report)
                results.append(report)
                print(
                    json.dumps(
                        {
                            "completed": room,
                            "mode": mode,
                            "nje": score["normalized_joint_error"],
                            "relative_motion_r2": motion["relative"][
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
        mean = features["pose_sequence"][indices].reshape(-1, 17, 2).mean(0)
        prediction = np.broadcast_to(mean, arrays["pose"][held].shape)
        prior = pose_metrics(prediction, arrays["pose"][held], arrays["valid"][held])
        prior.pop("per_frame_nje")
        priors.append(
            {
                "room": room,
                "training_frames": len(indices) * 8,
                "mean_pose": mean.tolist(),
                "metrics": prior,
                "fit": "same dense source-training frames as every mode, no validation/held fitting",
            }
        )
        _json(out / "matched-source-mean.json", {"priors": priors})
    if (
        sources != {key: digest(ROOT / key) for key in sources}
        or digest(stream_cache / "receipt.json")
        != registration["stream_receipt_sha256"]
        or digest(rf_cache / "receipt.json") != registration["rf_receipt_sha256"]
        or any(
            digest(stream_cache / (key + ".npy")) != value
            for key, value in stream_receipt["array_sha256"].items()
        )
        or digest(rf_cache / "quotient.npy") != registration["quotient_sha256"]
    ):
        raise ValueError("Frozen source/cache drift during execution")
    _json(
        out / "results.json",
        {
            "registration": registration,
            "results": results,
            "matched_source_mean": priors,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--stream-cache", type=Path)
    parser.add_argument(
        "--modes", nargs="+", choices=MODES, default=list(INITIAL_MODES)
    )
    parser.add_argument("--rf-cache", type=Path)
    args = parser.parse_args()
    if args.benchmark:
        benchmark(args.out.resolve(), args.device)
    elif any(x is None for x in [args.reference, args.stream_cache, args.rf_cache]):
        parser.error("Data execution requires reference,stream-cache,rf-cache")
    else:
        run(
            args.reference.resolve(),
            args.stream_cache.resolve(),
            args.rf_cache.resolve(),
            args.out.resolve(),
            args.device,
            args.modes,
        )
