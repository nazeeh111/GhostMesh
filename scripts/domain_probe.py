"""Development-only room-transfer probes. E04 is never decoded by this tool."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ghostmesh.preprocess import prepare_window

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def features(csi, bins=19):
    """Causal variable-window moments; retain static and centered dynamic paths."""
    length = len(csi)
    if np.shape(csi) != (length, 3, 114, 10) or length < 1 or 114 % bins:
        raise ValueError(
            f"Expected [T,3,114,10] CSI with bins dividing 114, got {np.shape(csi)}"
        )
    x = prepare_window(csi, normalize=True)
    mean = x.mean(-1).reshape(length, 3, bins, 114 // bins).mean(-1)
    sd = np.sqrt(x.var(-1) + 1e-6).reshape(length, 3, bins, 114 // bins).mean(-1)
    moments = np.stack((mean, sd), axis=-1).reshape(length, -1)
    dynamic = moments - moments.mean(0, keepdims=True)
    return moments.astype(np.float32), dynamic.astype(np.float32)


def build(manifest, out, window=8, bins=19):
    if out.exists():
        raise FileExistsError(out)
    raw = manifest.read_bytes()
    m = json.loads(raw)
    rows = [r for r in m["sequences"] if r["environment"] in {"E01", "E02", "E03"}]
    assert len(rows) == 405
    out.mkdir(parents=True)
    xs, ds, ys, vs, rooms, people, sequences, ends = [], [], [], [], [], [], [], []
    source = {}
    started = time.monotonic()
    for i, r in enumerate(rows):
        p = manifest.parent / r["path"]
        assert sha256(p.read_bytes()).hexdigest() == r["sha256"]
        source[r["id"]] = r["sha256"]
        with np.load(p, allow_pickle=False) as a:
            csi, pose, valid = a["csi"], a["pose"], a["valid"]
        for end in range(window - 1, len(csi), 8):
            moment, dyn = features(csi[end - window + 1 : end + 1], bins=bins)
            xs.append(moment)
            ds.append(dyn)
            ys.append(pose[end])
            vs.append(valid[end])
            rooms.append(r["environment"])
            people.append(r["subject"])
            sequences.append(r["id"])
            ends.append(end)
        if (i + 1) % 45 == 0:
            print(
                json.dumps(
                    {
                        "sequences": i + 1,
                        "total": len(rows),
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    for name, value in {
        "moments": xs,
        "dynamic": ds,
        "pose": ys,
        "valid": vs,
        "room": rooms,
        "subject": people,
        "sequence": sequences,
        "end": ends,
    }.items():
        np.save(out / (name + ".npy"), np.asarray(value), allow_pickle=False)
    receipt = {
        "scope": "Development E01-E03 only; no E04 CSI/labels decoded",
        "manifest_sha256": sha256(raw).hexdigest(),
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "sequences": source,
        "stride": 8,
        "window": window,
        "bins": bins,
        "windows": len(xs),
        "arrays": {
            p.name: sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.glob("*.npy"))
        },
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


def metrics(pred, y, valid):
    finite = valid & np.isfinite(y).all(-1)
    low = np.where(finite[..., None], y, np.inf).min(1)
    high = np.where(finite[..., None], y, -np.inf).max(1)
    scale = np.linalg.norm(high - low, axis=-1)
    mask = finite & (scale[:, None] > 1e-6)
    e = np.linalg.norm(pred - y, axis=-1) / scale[:, None]
    return {
        "nje": float(e[mask].mean()),
        "pck05": float((e[mask] <= 0.05).mean()),
        "pck20": float((e[mask] <= 0.2).mean()),
        "joints": int(mask.sum()),
    }


def ridge(x, y, z, alpha):
    mean = x.mean(0)
    scale = np.maximum(x.std(0), 0.1)
    x = (x - mean) / scale
    z = (z - mean) / scale
    ym = y.mean(0)
    cov = x.T @ x
    cov.flat[:: len(cov) + 1] += alpha * len(x)
    w = cho_solve(
        cho_factor(cov, lower=True, check_finite=False),
        x.T @ (y - ym),
        check_finite=False,
    )
    return z @ w + ym


def probe(cache, out):
    if out.exists():
        raise FileExistsError(out)
    a = {p.stem: np.load(p, allow_pickle=False) for p in cache.glob("*.npy")}
    assert set(a["room"]) == {"E01", "E02", "E03"}
    y = a["pose"]
    assert a["valid"].all()
    xs = {
        "last": a["moments"][:, -1],
        "temporal": a["moments"].reshape(len(y), -1),
        "dynamic": a["dynamic"].reshape(len(y), -1),
        "hybrid": np.concatenate(
            [a["moments"][:, -1], a["dynamic"].reshape(len(y), -1)], axis=1
        ),
    }
    results = []
    for room in ["E01", "E02", "E03"]:
        tr = a["room"] != room
        va = ~tr
        prior = np.broadcast_to(y[tr].mean(0), y[va].shape)
        results.append(
            {
                "held_room": room,
                "method": "source_mean",
                "metrics": metrics(prior, y[va], a["valid"][va]),
            }
        )
        print(json.dumps(results[-1]), flush=True)
        for method, x in xs.items():
            for alpha in [0.001, 0.01, 0.1, 1.0, 10.0]:
                pred = ridge(
                    x[tr].astype("float64"),
                    y[tr].reshape(tr.sum(), -1).astype("float64"),
                    x[va].astype("float64"),
                    alpha,
                ).reshape(-1, 17, 2)
                row = {
                    "held_room": room,
                    "method": method,
                    "alpha": alpha,
                    "metrics": metrics(pred, y[va], a["valid"][va]),
                }
                results.append(row)
                print(json.dumps(row), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "scope": "DEVELOPMENT ONLY: three room folds on E01-E03, no final test results",
                "cache_receipt_sha256": sha256(
                    (cache / "receipt.json").read_bytes()
                ).hexdigest(),
                "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["build", "probe"])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--bins", type=int, default=19)
    args = parser.parse_args()
    if args.mode == "build":
        build(args.manifest, args.cache, args.window, args.bins)
    else:
        probe(args.cache, args.out)
