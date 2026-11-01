"""Long-context development-only E01-E03 neural room folds; never opens original/E04 arrays."""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.engine import _json, seed_everything  # noqa: E402
from ghostmesh.metrics import pose_metrics  # noqa: E402
from ghostmesh.models import CausalBlock, supervised_loss  # noqa: E402

ROOMS = ("E01", "E02", "E03")
ARMS = ("frame_nll", "frame_huber", "temporal_huber", "factorized_dynamic_huber")
CONFIG = {"epochs": 15, "seed": 2027, "threads": 1, "batch_size": 128,
          "width": 64, "dilations": [1, 2, 4, 8], "learning_rate": .001,
          "weight_decay": .01, "gradient_clip": 1., "feature_scale_floor": .1,
          "source_validation_subjects": ["S04", "S14", "S24"],
          "checkpoint_selection": "source_validation_normalized_joint_error",
          "factorized_loss": "separate relative/root Huber plus detached-pose uncertainty NLL; no absolute-position loss into relative branch",
          "huber_beta": .1, "uncertainty_auxiliary_weight": .1,
          "huber_scale": "coordinate residual divided by target valid-body bbox diagonal, per endpoint",
          "uncertainty_rule": "NLL receives detached predicted poses; uncertainty cannot weight direct pose gradients",
          "root": "mean of COCO left/right hip joints 11 and 12; linear static last-frame root branch",
          "scope": "DEVELOPMENT ONLY: third-room validation within E01-E03, not final holdout; no E04 decoded",
          "prior_art": "Inspired by root/relative separation, not a RePos reproduction or first-ever claim: https://arxiv.org/abs/2607.02986"}


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def load_cache(cache):
    receipt = json.loads((cache / "receipt.json").read_text())
    arrays = {}
    for filename, expected in receipt["arrays"].items():
        path = cache / filename
        if path.parent != cache or digest(path) != expected:
            raise ValueError(f"Cache hash/path changed: {filename}")
        arrays[path.stem] = np.load(path, allow_pickle=False)
    if set(arrays["room"]) != set(ROOMS) or any(not s.startswith(ROOMS) for s in arrays["sequence"]):
        raise ValueError("Cache must contain E01-E03 development rooms exclusively")
    n = len(arrays["pose"])
    shape = arrays["moments"].shape
    if (len(shape) != 3 or n != receipt["windows"] or shape[0] != n
            or shape[1] != receipt["window"] or shape[2] != receipt.get("bins", shape[2] // 6) * 6
            or arrays["dynamic"].shape != shape or arrays["pose"].shape != (n, 17, 2)
            or arrays["valid"].shape != (n, 17) or not arrays["valid"].all()):
        raise ValueError("Unexpected aligned cache shapes or masks")
    if not np.allclose(arrays["dynamic"], arrays["moments"] - arrays["moments"].mean(1, keepdims=True), atol=1e-6):
        raise ValueError("Dynamic cache differs from within-window centering")
    if any(not np.isfinite(arrays[k]).all() for k in ("pose", "moments", "dynamic")):
        raise ValueError("Nonfinite cache arrays")
    return arrays, receipt


def masks(arrays, held_room):
    source = arrays["room"] != held_room
    val = source & np.isin(arrays["subject"], CONFIG["source_validation_subjects"])
    train, held = source & ~val, ~source
    if not train.any() or not val.any() or not held.any():
        raise ValueError("Empty development fold")
    if set(arrays["subject"][train]) & set(arrays["subject"][val]):
        raise ValueError("Source validation subjects overlap training")
    return train, val, held


def target_stats(y):
    root = y[:, [11, 12]].mean(1, keepdims=True)
    relative = y - root
    def stats(a):
        mean = a.mean(0)
        sigma2 = ((a - mean) ** 2).sum(-1).mean(0) / 2
        return torch.tensor(mean), torch.tensor(.5 * np.log(np.maximum(sigma2, 1e-6)))
    return {key: stats(a) for key, a in (("absolute", y), ("root", root), ("relative", relative))}


class Probe(nn.Module):
    def __init__(self, arm, stats, features):
        super().__init__()
        self.arm = arm
        width = CONFIG["width"]
        self.encoder = nn.Sequential(nn.Linear(features, width * 2), nn.LayerNorm(width * 2), nn.SiLU(),
                                     nn.Linear(width * 2, width), nn.LayerNorm(width), nn.SiLU())
        self.temporal = nn.Sequential(*(CausalBlock(width, d, "direct") for d in CONFIG["dilations"])) if not arm.startswith("frame_") else nn.Identity()
        self.head = nn.Linear(width, 51)
        self.factorized = arm == "factorized_dynamic_huber"
        mean, sigma = stats["relative" if self.factorized else "absolute"]
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.copy_(torch.cat([mean, sigma[:, None]], -1).reshape(-1))
        if self.factorized:
            self.root_head = nn.Linear(features, 3)
            nn.init.zeros_(self.root_head.weight)
            with torch.no_grad():
                self.root_head.bias.copy_(torch.cat([stats["root"][0].reshape(-1), stats["root"][1]]))

    def forward(self, moments, dynamic):
        x = dynamic if "dynamic" in self.arm else moments
        if self.arm.startswith("frame_"):
            x = x[:, -1:]
        z = self.encoder(x)
        z = self.temporal(z.transpose(1, 2)).transpose(1, 2)[:, -1]
        raw = self.head(z).reshape(-1, 17, 3)
        pose, sigma = raw[..., :2], raw[..., 2].clamp(-6, 1)
        if not self.factorized:
            return {"pose": pose, "log_sigma": sigma}
        relative = pose - pose[:, [11, 12]].mean(1, keepdim=True)
        root_raw = self.root_head(moments[:, -1])
        root, root_sigma = root_raw[:, None, :2], root_raw[:, None, 2].clamp(-6, 1)
        total_sigma = .5 * torch.logaddexp(2 * sigma, 2 * root_sigma)
        return {"pose": relative + root, "log_sigma": total_sigma,
                "relative": relative, "relative_log_sigma": sigma,
                "root": root, "root_log_sigma": root_sigma}

    def loss(self, prediction, target, valid):
        if self.arm == "frame_nll":
            return supervised_loss(prediction, target, valid)
        lower = torch.where(valid[..., None], target, torch.inf).amin(1)
        upper = torch.where(valid[..., None], target, -torch.inf).amax(1)
        scale = torch.linalg.vector_norm(upper - lower, dim=-1).clamp_min(1e-6)
        if not self.factorized:
            return geometry_uncertainty(prediction["pose"], prediction["log_sigma"], target, valid, scale)
        root = target[:, [11, 12]].mean(1, keepdim=True)
        relative = geometry_uncertainty(prediction["relative"], prediction["relative_log_sigma"], target - root, valid, scale)
        position = geometry_uncertainty(prediction["root"], prediction["root_log_sigma"], root, valid[:, :1], scale)
        return relative + position


def geometry_uncertainty(pose, log_sigma, target, valid, scale):
    residual = (pose - target) / scale[:, None, None]
    geometry = F.smooth_l1_loss(residual[valid], torch.zeros_like(residual[valid]), beta=CONFIG["huber_beta"])
    uncertainty = supervised_loss({"pose": pose.detach(), "log_sigma": log_sigma}, target, valid)
    return geometry + CONFIG["uncertainty_auxiliary_weight"] * uncertainty


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    poses, ys, valid, sigma = [], [], [], []
    root_error, relative_error = [], []
    for moments, dynamic, target, mask in loader:
        out = model(moments, dynamic)
        poses.append(out["pose"].numpy())
        ys.append(target.numpy())
        valid.append(mask.numpy())
        sigma.append(out["log_sigma"].numpy())
        root = target[:, [11, 12]].mean(1, keepdim=True)
        predicted_root = out["pose"][:, [11, 12]].mean(1, keepdim=True)
        root_error.append(torch.linalg.vector_norm(predicted_root - root, dim=-1).numpy())
        relative_error.append(torch.linalg.vector_norm(out["pose"] - predicted_root - (target - root), dim=-1).numpy())
    metrics = pose_metrics(np.concatenate(poses), np.concatenate(ys), np.concatenate(valid), np.concatenate(sigma))
    metrics.pop("per_frame_nje")
    metrics["root_coordinate_error"] = float(np.concatenate(root_error).mean())
    metrics["root_relative_coordinate_error"] = float(np.concatenate(relative_error).mean())
    return metrics


def run(cache, out, selected_arms=ARMS, selected_rooms=ROOMS, seed=2027):
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(1)
    a, receipt = load_cache(cache)
    CONFIG.update({"seed": seed, "window": a["moments"].shape[1], "input_features": a["moments"].shape[2], "bins": receipt.get("bins")})
    source_paths = [Path(__file__), ROOT / "scripts/domain_probe.py"] + [ROOT / "ghostmesh" / name for name in ("engine.py", "models.py", "data.py", "preprocess.py", "metrics.py")]
    source = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    registration = {"config": CONFIG, "arms": list(selected_arms), "held_development_rooms": list(selected_rooms),
                    "source_sha256": source, "cache_receipt_sha256": digest(cache / "receipt.json"),
                    "cache_arrays_sha256": receipt["arrays"], "cache_source_manifest_sha256": receipt["manifest_sha256"],
                    "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__), "platform": platform.platform()}}
    out.mkdir(parents=True)
    _json(out / "registration.json", registration)
    for relative, expected in source.items():
        content = (ROOT / relative).read_bytes()
        if sha256(content).hexdigest() != expected:
            raise ValueError("Source changed before executed-source archive")
        destination = out / "executed-source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    results = []
    for room in selected_rooms:
        train, val, held = masks(a, room)
        stats = target_stats(a["pose"][train])
        normalization = {}
        for key in ("moments", "dynamic"):
            normalization[key] = (a[key][train].mean((0, 1)), np.maximum(a[key][train].std((0, 1)), CONFIG["feature_scale_floor"]))
        np.savez(out / f"{room}-training-stats.npz", **{key + suffix: v for key, pair in normalization.items() for suffix, v in zip(("_mean", "_scale"), pair)})
        def loader(mask, shuffle=False):
            tensors = [torch.from_numpy(((a[k][mask] - normalization[k][0]) / normalization[k][1]).astype(np.float32)) for k in ("moments", "dynamic")]
            tensors += [torch.from_numpy(a["pose"][mask]), torch.from_numpy(a["valid"][mask])]
            return DataLoader(TensorDataset(*tensors), batch_size=CONFIG["batch_size"], shuffle=shuffle,
                              generator=torch.Generator().manual_seed(CONFIG["seed"]))
        for arm in selected_arms:
            started = time.monotonic()
            destination = out / f"{room}-{arm}"
            destination.mkdir()
            seed_everything(CONFIG["seed"])
            model = Probe(arm, stats, CONFIG["input_features"])
            optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
            train_loader, validation_loader = loader(train, True), loader(val)
            history, best, chosen = [], float("inf"), None
            try:
                for epoch in range(1, CONFIG["epochs"] + 1):
                    model.train()
                    losses = []
                    for moments, dynamic, y, valid in train_loader:
                        optimizer.zero_grad(set_to_none=True)
                        loss = model.loss(model(moments, dynamic), y, valid)
                        if not torch.isfinite(loss):
                            raise ValueError("Nonfinite training objective")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip"])
                        optimizer.step()
                        losses.append(float(loss.detach()))
                    validation = evaluate(model, validation_loader)
                    score = validation["normalized_joint_error"]
                    history.append({"epoch": epoch, "training_loss": float(np.mean(losses)), "source_validation": validation})
                    if score < best:
                        best, chosen, best_epoch = score, deepcopy(model.state_dict()), epoch
                    _json(destination / "history.json", {"history": history})
                    print(json.dumps({"room": room, "arm": arm, "epoch": epoch, "source_validation_nje": score}), flush=True)
                model.load_state_dict(chosen)
                torch.save({"state_dict": chosen, "arm": arm, "config": CONFIG,
                            "training_target_stats": stats, "source_sha256": source,
                            "cache_receipt_sha256": registration["cache_receipt_sha256"]}, destination / "best.pt")
                # The third room is scored only after source-only fitting and checkpoint selection.
                held_metrics = evaluate(model, loader(held))
                result = {"room": room, "arm": arm, "status": "completed", "selected_epoch": best_epoch,
                          "source_validation_nje": best, "held_development_room": held_metrics,
                          "counts": {"training": int(train.sum()), "source_validation": int(val.sum()), "held_development": int(held.sum())},
                          "subjects": {k: sorted(set(a["subject"][mask])) for k, mask in (("training", train), ("source_validation", val), ("held_development", held))},
                          "parameters": sum(p.numel() for p in model.parameters()), "seconds": time.monotonic() - started,
                          "coverage": {"total_cached_endpoints": len(a["pose"]), "fold_endpoints": int((train | val | held).sum()),
                                       "training_sequences": len(set(a["sequence"][train])), "source_validation_sequences": len(set(a["sequence"][val])),
                                       "held_development_sequences": len(set(a["sequence"][held])), "window": receipt["window"], "stride": receipt["stride"]},
                          "checkpoint_sha256": digest(destination / "best.pt")}
                _json(destination / "result.json", result)
                results.append(result)
                print(json.dumps(result), flush=True)
            except Exception as exc:
                _json(destination / "failure.json", {"type": type(exc).__name__, "error": str(exc), "history": history})
                raise
    if source != {name: digest(ROOT / name) for name in source} or digest(cache / "receipt.json") != registration["cache_receipt_sha256"]:
        raise ValueError("Source/cache receipt changed during run")
    if any(digest(cache / name) != expected for name, expected in receipt["arrays"].items()):
        raise ValueError("Cache arrays changed during run")
    _json(out / "results.json", {"registration": registration, "results": results})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--rooms", nargs="+", choices=ROOMS, default=list(ROOMS))
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    run(args.cache.resolve(), args.out.resolve(), args.arms, args.rooms, args.seed)
