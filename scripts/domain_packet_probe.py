"""Development-only raw packet structure screen; no E04 or final records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.engine import _json, seed_everything  # noqa:E402
from ghostmesh.metrics import pose_metrics  # noqa:E402
from ghostmesh.packet_model import PacketPose  # noqa:E402
from domain_raw_cache import digest  # noqa:E402

CONFIG = {
    "seed": 2027,
    "epochs": 8,
    "batch_size": 64,
    "width": 64,
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "training_endpoint_subsample": 4,
    "source_validation_subjects": ["S04", "S14", "S24"],
    "checkpoint_selection": "source validation NJE",
    "scope": "DEVELOPMENT E01-E03 only; actual preprocessed CSI with packet axis retained. Unknown original packet timestamps; no physical Doppler claim.",
}


class RawSet(Dataset):
    def __init__(self, raw, pose, indices):
        self.raw, self.pose, self.indices = raw, pose, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        return torch.from_numpy(self.raw[i].copy()), torch.from_numpy(self.pose[i])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps, sigmas = [], [], []
    for x, y in loader:
        result = model(x.to(device))
        ys.append(y.numpy())
        ps.append(result["pose"].cpu().numpy())
        sigmas.append(result["log_sigma"].cpu().numpy())
    y, p, sigma = np.concatenate(ys), np.concatenate(ps), np.concatenate(sigmas)
    m = pose_metrics(p, y, np.ones(y.shape[:2], bool), sigma)
    m.pop("per_frame_nje")
    return m, p


def run(reference, raw_cache, out, device):
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(1)
    r = json.loads((reference / "receipt.json").read_text())
    rawr = json.loads((raw_cache / "receipt.json").read_text())
    assert digest(reference / "receipt.json") == rawr["reference_cache_receipt_sha256"]
    assert digest(raw_cache / "raw.npy") == rawr["raw_sha256"]
    a = {
        k: np.load(reference / (k + ".npy"), allow_pickle=False)
        for k in ["pose", "valid", "room", "subject", "sequence", "end"]
    }
    for k in a:
        assert digest(reference / (k + ".npy")) == r["arrays"][k + ".npy"]
    assert set(a["room"]) == {"E01", "E02", "E03"} and a["valid"].all()
    raw = np.load(raw_cache / "raw.npy", mmap_mode="r", allow_pickle=False)
    assert list(raw.shape) == rawr["shape"]
    sources = {
        str(p.relative_to(ROOT)): digest(p)
        for p in [
            Path(__file__),
            ROOT / "ghostmesh/packet_model.py",
            ROOT / "scripts/domain_raw_cache.py",
        ]
        + [
            ROOT / "ghostmesh" / n
            for n in [
                "models.py",
                "engine.py",
                "data.py",
                "metrics.py",
                "preprocess.py",
            ]
        ]
    }
    out.mkdir(parents=True)
    config = {
        **CONFIG,
        "device": device,
        "runtime": {"torch": str(torch.__version__), "numpy": np.__version__},
    }
    _json(
        out / "registration.json",
        {
            "config": config,
            "source": sources,
            "reference": digest(reference / "receipt.json"),
            "raw_receipt": digest(raw_cache / "receipt.json"),
        },
    )
    for name in sources:
        dst = out / "executed-source" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((ROOT / name).read_bytes())
    results = []
    for room in ["E01", "E02", "E03"]:
        source = a["room"] != room
        val = source & np.isin(a["subject"], CONFIG["source_validation_subjects"])
        train = source & ~val
        held = ~source
        indices = np.flatnonzero(train)[:: CONFIG["training_endpoint_subsample"]]
        assert not (set(a["subject"][indices]) & set(a["subject"][val]))
        y = a["pose"][indices]
        mean = y.mean(0)
        logsigma = 0.5 * np.log(
            np.maximum(np.square(y - mean).sum(-1).mean(0) / 2, 1e-6)
        )

        def loader(which, shuffle=False):
            idx = which if which.dtype != bool else np.flatnonzero(which)
            return DataLoader(
                RawSet(raw, a["pose"], idx),
                batch_size=CONFIG["batch_size"],
                shuffle=shuffle,
                generator=torch.Generator().manual_seed(CONFIG["seed"]),
            )

        for temporal in [False, True]:
            seed_everything(CONFIG["seed"])
            model = PacketPose(temporal, CONFIG["width"])
            model.initialize_from_source(mean, logsigma)
            model.to(device)
            opt = torch.optim.AdamW(
                model.parameters(),
                lr=CONFIG["learning_rate"],
                weight_decay=CONFIG["weight_decay"],
            )
            folder = out / f"{room}-{'temporal' if temporal else 'frame'}"
            folder.mkdir()
            train_loader = loader(indices, True)
            history = []
            best = float("inf")
            chosen = None
            started = time.monotonic()
            for epoch in range(1, CONFIG["epochs"] + 1):
                model.train()
                losses = []
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad(set_to_none=True)
                    pred = model(x)
                    error2 = (pred["pose"] - y).square().sum(-1)
                    ls = pred["log_sigma"]
                    loss = (0.5 * error2 * torch.exp(-2 * ls) + 2 * ls).mean()
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    losses.append(float(loss.detach().cpu()))
                score, _ = evaluate(model, loader(val), device)
                history.append(
                    {
                        "epoch": epoch,
                        "training_loss": float(np.mean(losses)),
                        "source_validation": score,
                    }
                )
                if score["normalized_joint_error"] < best:
                    best = score["normalized_joint_error"]
                    best_epoch = epoch
                    chosen = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }
                _json(folder / "history.json", {"history": history})
                print(
                    json.dumps(
                        {
                            "room": room,
                            "temporal": temporal,
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
                    "temporal": temporal,
                    "sources": sources,
                },
                folder / "best.pt",
            )
            score, pred = evaluate(model, loader(held), device)
            np.save(
                folder / "held-development-prediction.npy", pred, allow_pickle=False
            )
            report = {
                "room": room,
                "temporal": temporal,
                "selected_epoch": best_epoch,
                "metrics": score,
                "parameters": sum(p.numel() for p in model.parameters()),
                "counts": {
                    "train": len(indices),
                    "source_validation": int(val.sum()),
                    "held_development": int(held.sum()),
                },
                "seconds": time.monotonic() - started,
                "checkpoint_sha256": digest(folder / "best.pt"),
            }
            _json(folder / "result.json", report)
            results.append(report)
            print(
                json.dumps(
                    {
                        "completed": room,
                        "temporal": temporal,
                        "nje": score["normalized_joint_error"],
                    }
                ),
                flush=True,
            )
    assert sources == {n: digest(ROOT / n) for n in sources}
    _json(
        out / "results.json", {"config": config, "results": results, "source": sources}
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--raw-cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    a = p.parse_args()
    run(a.reference.resolve(), a.raw_cache.resolve(), a.out.resolve(), a.device)
