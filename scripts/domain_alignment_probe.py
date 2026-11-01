"""Source-room continuous pose alignment. Development E01-E03 only."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.engine import _json, seed_everything  # noqa:E402
from ghostmesh.models import CausalBlock  # noqa:E402
from domain_neural_long_probe import (  # noqa:E402
    digest,
    evaluate,
    geometry_uncertainty,
    load_cache,
    masks,
    target_stats,
)

CONFIG = {
    "seed": 2027,
    "epochs": 15,
    "batch_size": 128,
    "threads": 1,
    "width": 64,
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "temperature": 0.1,
    "target_bandwidth": 0.1,
    "positive_neighbors": 4,
    "alignment_weights": [0.0, 0.05],
    "source_validation_subjects": ["S04", "S14", "S24"],
    "selection": "source-validation NJE; held third room scored only after training",
    "scope": "DEVELOPMENT only; continuous source-pose contrast across different source rooms; E04 never decoded",
    "mechanism": "Same relative posture in different source rooms is a soft positive; factorized heads keep root-position gradients out of structure branch. Standard supervised contrastive principles, no novelty claim.",
}


def alignment(z, target, room):
    """Continuous labels provide soft cross-room positives, never action IDs."""
    if len(z) < 2:
        return z.sum() * 0
    root = target[:, [11, 12]].mean(1, keepdim=True)
    scale = torch.linalg.vector_norm(target.amax(1) - target.amin(1), dim=-1).clamp_min(
        1e-6
    )
    posture = ((target - root) / scale[:, None, None]).flatten(1)
    distance = torch.cdist(posture, posture).square() / posture.shape[1]
    cross = room[:, None] != room[None, :]
    usable = cross.any(1)
    if not usable.any():
        return z.sum() * 0
    rank = distance.masked_fill(~cross, torch.inf).argsort(1).argsort(1)
    positive = cross & (rank < CONFIG["positive_neighbors"])
    target_logits = (-distance / (2 * CONFIG["target_bandwidth"] ** 2)).masked_fill(
        ~positive, -1e9
    )
    weights = target_logits.softmax(1)
    normalized = F.normalize(z, dim=-1)
    logits = normalized @ normalized.T / CONFIG["temperature"]
    logits = logits.masked_fill(torch.eye(len(z), dtype=torch.bool), -1e9)
    log_probability = logits.log_softmax(1)
    return -(weights[usable] * log_probability[usable]).sum(1).mean()


class AlignPose(nn.Module):
    def __init__(self, width_in, stats):
        super().__init__()
        w = CONFIG["width"]
        self.encoder = nn.Sequential(
            nn.Linear(width_in * 2, w * 2),
            nn.LayerNorm(w * 2),
            nn.SiLU(),
            nn.Linear(w * 2, w),
            nn.LayerNorm(w),
            nn.SiLU(),
        )
        self.temporal = nn.Sequential(*(CausalBlock(w, d, "direct") for d in (1, 2, 4)))
        self.head = nn.Linear(w, 51)
        self.root_head = nn.Linear(width_in, 3)
        for head, key in ((self.head, "relative"), (self.root_head, "root")):
            nn.init.zeros_(head.weight)
            mean, sigma = stats[key]
            with torch.no_grad():
                head.bias.copy_(torch.cat((mean, sigma[:, None]), -1).flatten())

    def forward(self, moments, dynamic):
        x = torch.cat((moments, dynamic), -1)
        z = self.encoder(x)
        z = self.temporal(z.transpose(1, 2)).transpose(1, 2)[:, -1]
        raw = self.head(z).reshape(-1, 17, 3)
        rel = raw[..., :2]
        rel = rel - rel[:, [11, 12]].mean(1, keepdim=True)
        ls = raw[..., 2].clamp(-6, 1)
        r = self.root_head(moments[:, -1])
        root, rlog = r[:, None, :2], r[:, None, 2].clamp(-6, 1)
        return {
            "pose": rel + root,
            "log_sigma": 0.5 * torch.logaddexp(2 * ls, 2 * rlog),
            "relative": rel,
            "relative_log_sigma": ls,
            "root": root,
            "root_log_sigma": rlog,
            "z": z,
        }

    def loss(self, pred, target, valid):
        scale = torch.linalg.vector_norm(
            target.amax(1) - target.amin(1), dim=-1
        ).clamp_min(1e-6)
        root = target[:, [11, 12]].mean(1, keepdim=True)
        return geometry_uncertainty(
            pred["relative"], pred["relative_log_sigma"], target - root, valid, scale
        ) + geometry_uncertainty(
            pred["root"], pred["root_log_sigma"], root, valid[:, :1], scale
        )


def run(cache, out):
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(1)
    a, receipt = load_cache(cache)
    paths = [
        Path(__file__),
        ROOT / "scripts/domain_neural_long_probe.py",
        ROOT / "scripts/domain_probe.py",
    ] + [
        ROOT / "ghostmesh" / n
        for n in ["engine.py", "models.py", "data.py", "metrics.py", "preprocess.py"]
    ]
    sources = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    config = {**CONFIG, "window": receipt["window"], "features": a["moments"].shape[-1]}
    out.mkdir(parents=True)
    _json(
        out / "registration.json",
        {
            "config": config,
            "sources": sources,
            "cache_receipt_sha256": digest(cache / "receipt.json"),
            "arrays": receipt["arrays"],
        },
    )
    for name in sources:
        dst = out / "executed-source" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((ROOT / name).read_bytes())
    results = []
    for room in ["E01", "E02", "E03"]:
        train, val, held = masks(a, room)
        stats = target_stats(a["pose"][train])
        normal = {
            k: (a[k][train].mean((0, 1)), np.maximum(a[k][train].std((0, 1)), 0.1))
            for k in ["moments", "dynamic"]
        }

        def loader(mask, shuffle=False):
            tensors = [
                torch.from_numpy(
                    ((a[k][mask] - normal[k][0]) / normal[k][1]).astype("float32")
                )
                for k in ["moments", "dynamic"]
            ]
            tensors.extend(
                [
                    torch.from_numpy(a["pose"][mask]),
                    torch.from_numpy(a["valid"][mask]),
                    torch.tensor([int(s[-1]) for s in a["room"][mask]]),
                ]
            )
            return DataLoader(
                TensorDataset(*tensors),
                batch_size=128,
                shuffle=shuffle,
                generator=torch.Generator().manual_seed(CONFIG["seed"]),
            )

        def evaluation(model, mask):
            return evaluate(model, ((m, d, y, v) for m, d, y, v, _ in loader(mask)))

        for weight in CONFIG["alignment_weights"]:
            seed_everything(CONFIG["seed"])
            model = AlignPose(a["moments"].shape[-1], stats)
            opt = torch.optim.AdamW(
                model.parameters(),
                lr=CONFIG["learning_rate"],
                weight_decay=CONFIG["weight_decay"],
            )
            folder = out / f"{room}-align-{weight:g}"
            folder.mkdir()
            history = []
            best = float("inf")
            chosen = None
            started = time.monotonic()
            for epoch in range(1, CONFIG["epochs"] + 1):
                model.train()
                losses = []
                for m, d, y, v, r in loader(train, True):
                    opt.zero_grad(set_to_none=True)
                    pred = model(m, d)
                    loss = model.loss(pred, y, v)
                    if weight:
                        loss = loss + weight * alignment(pred["z"], y, r)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    losses.append(float(loss.detach()))
                score = evaluation(model, val)
                history.append(
                    {
                        "epoch": epoch,
                        "training_loss": float(np.mean(losses)),
                        "source_validation": score,
                    }
                )
                if score["normalized_joint_error"] < best:
                    best = score["normalized_joint_error"]
                    chosen = deepcopy(model.state_dict())
                    best_epoch = epoch
                _json(folder / "history.json", {"history": history})
                print(
                    json.dumps(
                        {
                            "room": room,
                            "weight": weight,
                            "epoch": epoch,
                            "source_val_nje": score["normalized_joint_error"],
                        }
                    ),
                    flush=True,
                )
            model.load_state_dict(chosen)
            torch.save(
                {
                    "state_dict": chosen,
                    "config": config,
                    "weight": weight,
                    "stats": stats,
                    "normal": {
                        k: tuple(torch.from_numpy(b) for b in pair)
                        for k, pair in normal.items()
                    },
                    "sources": sources,
                },
                folder / "best.pt",
            )
            r = {
                "room": room,
                "alignment_weight": weight,
                "selected_epoch": best_epoch,
                "metrics": evaluation(model, held),
                "parameters": sum(p.numel() for p in model.parameters()),
                "counts": {
                    "train": int(train.sum()),
                    "source_validation": int(val.sum()),
                    "held_development": int(held.sum()),
                },
                "seconds": time.monotonic() - started,
                "checkpoint_sha256": digest(folder / "best.pt"),
            }
            _json(folder / "result.json", r)
            results.append(r)
            print(
                json.dumps(
                    {
                        "completed": room,
                        "weight": weight,
                        "nje": r["metrics"]["normalized_joint_error"],
                    }
                ),
                flush=True,
            )
    assert sources == {n: digest(ROOT / n) for n in sources}
    _json(
        out / "results.json", {"config": config, "sources": sources, "results": results}
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    run(args.cache.resolve(), args.out.resolve())
