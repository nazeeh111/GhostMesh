"""One bounded source30 causal-background feature probe; no E04/model scaling."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy.linalg import eigh
from scipy.linalg.blas import dgemm, sgemm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.causal_channel import channel_representations  # noqa:E402
from ghostmesh.metrics import pose_metrics  # noqa:E402

CONFIG = {
    "seed": 2027,
    "permutation_seed": 90210,
    "context": 32,
    "endpoint_stride": 8,
    "ema_tau_seconds": 2.0,
    "nominal_frame_seconds": 0.1,
    "projection_dimensions": 64,
    "rff_dimensions": 512,
    "ridge_alphas": [0.01, 0.1, 1.0, 10.0],
    "scale_floor": 0.001,
    "arms": ["untreated", "ema_residual", "startup_static"],
    "targets": ["root_relative_pose", "lag8_root_relative_displacement"],
    "selection": "Separate source-validation coordinate MSE per target; first alpha wins ties",
    "scope": "DEVELOPMENT E01-E03/source30 only; no E04. No novel RF/Doppler/AoA/physical-velocity claim",
    "continuation_gate": "EMA residual record-centered relative pose R2 and centered displacement R2 positive in every held room; macro pose R2 >=.05; fixed prediction shuffle lowers macro pose R2 >=.05. This is a continuation heuristic, not efficacy/significance proof.",
    "sources": [
        "https://arxiv.org/abs/1907.03994",
        "https://www.cswu.me/papers/mobisys18_widar2.0_paper.pdf",
    ],
    "limitations": "Fixed random projection and RFF are restricted learners. Negative findings cannot prove raw CSI lacks information; EMA can remove useful slow motion; nominal frame timing/within-block packet timing are not independently calibrated.",
}


def digest(path):
    h = sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def multiply(a, b):
    # SciPy BLAS avoids NumPy/Accelerate's spurious matmul FP warnings observed in earlier probes.
    a, b = (
        np.asarray(a, dtype=np.float64, order="F"),
        np.asarray(b, dtype=np.float64, order="F"),
    )
    result = dgemm(1.0, a, b)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite BLAS result")
    return result


def centered_r2(pred, target, record):
    error = denominator = 0.0
    per_record = []
    for key in sorted(set(record)):
        ix = record == key
        p, y = pred[ix], target[ix]
        p, y = p - p.mean(0), y - y.mean(0)
        e, d = float(((p - y) ** 2).sum()), float((y * y).sum())
        error += e
        denominator += d
        per_record.append(
            {
                "record": str(key),
                "r2": 1 - e / d if d > 0 else None,
                "samples": int(ix.sum()),
            }
        )
    return {
        "r2": 1 - error / denominator if denominator > 0 else None,
        "per_record": per_record,
    }


def scores(pose, delta, target, truth_delta, record):
    mask = np.ones(target.shape[:2], bool)
    metric = pose_metrics(pose, target, mask)
    metric.pop("per_frame_nje")
    return {
        "relative_nje": metric["normalized_joint_error"],
        "relative_pck05": metric["pck_curve"]["0.05"],
        "pose_motion": centered_r2(pose, target, record),
        "delta_motion": centered_r2(delta, truth_delta, record),
        "delta_zero_r2": 1
        - float(((delta - truth_delta) ** 2).sum()) / float((truth_delta**2).sum()),
    }


def fit_rff(feature, train, out):
    source = np.asarray(feature[train], dtype=np.float64)
    mean = source.mean(0)
    scale = np.maximum(source.std(0), CONFIG["scale_floor"])
    rng = np.random.default_rng(CONFIG["seed"])
    small = source[rng.choice(len(source), 128, replace=False)]
    small = (small - mean) / scale
    distance = np.maximum(
        (small * small).sum(1)[:, None]
        + (small * small).sum(1)[None, :]
        - 2 * multiply(small, small.T),
        0,
    )
    bandwidth = float(np.median(distance[np.triu_indices(128, 1)]))
    if not bandwidth > 0:
        raise ValueError("Degenerate source bandwidth")
    rng = np.random.default_rng(CONFIG["seed"])
    weight = rng.normal(size=(feature.shape[1], 512)) / np.sqrt(bandwidth)
    offset = rng.uniform(0, 2 * np.pi, 512)
    result = np.empty((len(feature), 512), dtype=np.float64)
    for start in range(0, len(feature), 1024):
        x = (np.asarray(feature[start : start + 1024], dtype=np.float64) - mean) / scale
        result[start : start + len(x)] = np.sqrt(2 / 512) * np.cos(
            multiply(x, weight) + offset
        )
    np.savez(
        out,
        feature_mean=mean,
        feature_scale=scale,
        weight=weight,
        offset=offset,
        bandwidth_squared=bandwidth,
    )
    return result


def run(cache, out):
    if out.exists():
        raise FileExistsError(out)
    started = time.monotonic()
    receipt = json.loads((cache / "receipt.json").read_text())
    records = receipt["records"]
    config = receipt["config"]
    if (
        len(records) != 810
        or set(r["id"][:3] for r in records) != {"E01", "E02", "E03"}
        or len(set(r["id"].split("/")[1] for r in records)) != 30
    ):
        raise ValueError("Only complete810-record source30 cache allowed")
    for key, expected in receipt["arrays"].items():
        if digest(cache / (key + ".npy")) != expected:
            raise ValueError("Source cache changed")
    rf = np.load(cache / "rf.npy", mmap_mode="r", allow_pickle=False)
    pose = np.load(cache / "pose.npy", mmap_mode="r", allow_pickle=False)
    valid = np.load(cache / "valid.npy", mmap_mode="r", allow_pickle=False)
    if (
        rf.shape != (240570, 6, 114, 10)
        or pose.shape != (240570, 17, 2)
        or not valid.all()
    ):
        raise ValueError("Unexpected full-source arrays")
    sources = {
        str(p.relative_to(ROOT)): digest(p)
        for p in [
            Path(__file__),
            ROOT / "ghostmesh/causal_channel.py",
            ROOT / "ghostmesh/metrics.py",
        ]
    }
    registration = {
        "config": CONFIG,
        "source_sha256": sources,
        "cache_receipt_sha256": digest(cache / "receipt.json"),
        "cache_arrays_sha256": receipt["arrays"],
        "source30_config": config,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    out.mkdir(parents=True)
    save(out / "registration.json", registration)
    for name in sources:
        p = out / "executed-source" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((ROOT / name).read_bytes())
    endpoints = np.arange(31, 297, 8)
    n = len(records) * len(endpoints)
    features = {
        key: np.lib.format.open_memmap(
            out / (key + "-features.npy"),
            mode="w+",
            dtype="float32",
            shape=(n, 32 * 64),
        )
        for key in CONFIG["arms"]
    }
    rng = np.random.default_rng(CONFIG["seed"])
    projection = (rng.normal(size=(4 * 114 * 10, 64)) / np.sqrt(4 * 114 * 10)).astype(
        np.float32, order="F"
    )
    np.save(out / "fixed-projection.npy", projection, allow_pickle=False)
    target = []
    delta = []
    room = []
    person = []
    record = []
    ends = []
    quality = []
    for i, row in enumerate(records):
        offset, length = row["offset"], row["frames"]
        assert length == 297 and offset == i * 297
        block = np.asarray(rf[offset : offset + length])
        representations, q = channel_representations(block)
        quality.append({"record": row["id"], **q})
        a = np.asarray(pose[offset : offset + length])
        root = a[:, [11, 12]].mean(1, keepdims=True)
        relative = a - root
        for arm, values in representations.items():
            if arm == "startup_static":
                v = sgemm(1.0, np.asarray(values.reshape(1, -1), order="F"), projection)
                projected = np.broadcast_to(v, (297, 64))
            else:
                projected = sgemm(
                    1.0, np.asarray(values.reshape(297, -1), order="F"), projection
                )
            features[arm][i * 34 : (i + 1) * 34] = np.stack(
                [projected[t - 31 : t + 1].reshape(-1) for t in endpoints]
            )
        target.extend(relative[endpoints])
        delta.extend(relative[endpoints] - relative[endpoints - 8])
        room.extend([row["id"][:3]] * 34)
        person.extend([row["id"].split("/")[1]] * 34)
        record.extend([row["id"]] * 34)
        ends.extend(endpoints.tolist())
        if (i + 1) % 90 == 0:
            print(
                json.dumps(
                    {
                        "features_records": i + 1,
                        "total": 810,
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    for value in features.values():
        value.flush()
    target, delta = (
        np.asarray(target, dtype=np.float64),
        np.asarray(delta, dtype=np.float64),
    )
    room, person, record = np.asarray(room), np.asarray(person), np.asarray(record)
    np.savez(
        out / "targets-and-metadata.npz",
        pose=target,
        delta=delta,
        room=room,
        person=person,
        record=record,
        end=np.asarray(ends),
    )
    save(out / "feature-quality.json", {"quality": quality})
    results = []
    priors = []
    for held_room in ["E01", "E02", "E03"]:
        val = (room != held_room) & np.isin(
            person, config["source_validation_subjects"]
        )
        train = (room != held_room) & ~val
        held = room == held_room
        assert train.sum() == 14688 and val.sum() == 3672 and held.sum() == 9180
        ix = np.flatnonzero(held)
        groups = record[held]
        perm = np.arange(len(ix))
        prng = np.random.default_rng(CONFIG["permutation_seed"])
        for key in sorted(set(groups)):
            g = np.flatnonzero(groups == key)
            perm[g] = prng.permutation(g)
        np.save(out / (held_room + "-permutation.npy"), perm, allow_pickle=False)
        mean = target[train].mean(0)
        static = np.broadcast_to(mean, target[held].shape)
        zero = np.zeros_like(delta[held])
        priors.append(
            {
                "room": held_room,
                "fit": "Source training relative-target mean and zero displacement only",
                "metrics": scores(static, zero, target[held], delta[held], groups),
            }
        )
        y = np.concatenate([target.reshape(n, -1), delta.reshape(n, -1)], axis=1)
        for arm in CONFIG["arms"]:
            tag = held_room + "-" + arm
            z = fit_rff(features[arm], train, out / (tag + "-rff.npz"))
            xm = z[train].mean(0)
            x = z[train] - xm
            ym = y[train].mean(0)
            covariance = multiply(x.T, x)
            eigen, vector = eigh(covariance, check_finite=True)
            right = multiply(vector.T, multiply(x.T, y[train] - ym))
            best = [None, None]
            grid = []
            for alpha in CONFIG["ridge_alphas"]:
                weight = multiply(
                    vector, right / (eigen[:, None] + alpha * train.sum())
                )
                prediction = multiply(z - xm, weight) + ym
                mse = [
                    float(
                        (
                            (
                                prediction[val, j * 34 : (j + 1) * 34]
                                - y[val, j * 34 : (j + 1) * 34]
                            )
                            ** 2
                        ).mean()
                    )
                    for j in range(2)
                ]
                grid.append(
                    {
                        "alpha": alpha,
                        "validation_mse_pose": mse[0],
                        "validation_mse_delta": mse[1],
                    }
                )
                for j in range(2):
                    if best[j] is None or mse[j] < best[j][0]:
                        best[j] = (
                            mse[j],
                            alpha,
                            weight[:, j * 34 : (j + 1) * 34].copy(),
                            prediction[:, j * 34 : (j + 1) * 34].copy(),
                        )
            p, d = (
                best[0][3][held].reshape(-1, 17, 2),
                best[1][3][held].reshape(-1, 17, 2),
            )
            metrics = scores(p, d, target[held], delta[held], groups)
            shuffled = scores(p[perm], d[perm], target[held], delta[held], groups)
            np.savez(
                out / (tag + "-model-prediction.npz"),
                pose=p,
                delta=d,
                pose_weights=best[0][2],
                delta_weights=best[1][2],
                x_mean=xm,
                y_mean=ym,
                held_indices=ix,
            )
            row = {
                "held_room": held_room,
                "arm": arm,
                "pose_alpha": best[0][1],
                "delta_alpha": best[1][1],
                "counts": {
                    "train": int(train.sum()),
                    "validation": int(val.sum()),
                    "held": int(held.sum()),
                },
                "metrics": metrics,
                "shuffled": shuffled,
                "source_validation_grid": grid,
            }
            results.append(row)
            save(out / "progress.json", {"results": results, "static_priors": priors})
            print(
                json.dumps(
                    {
                        "completed": tag,
                        "relative_nje": metrics["relative_nje"],
                        "pose_r2": metrics["pose_motion"]["r2"],
                        "delta_r2": metrics["delta_motion"]["r2"],
                        "shuffled_pose_r2": shuffled["pose_motion"]["r2"],
                    }
                ),
                flush=True,
            )
    if (
        sources != {name: digest(ROOT / name) for name in sources}
        or digest(cache / "receipt.json") != registration["cache_receipt_sha256"]
        or any(
            digest(cache / (key + ".npy")) != value
            for key, value in receipt["arrays"].items()
        )
    ):
        raise ValueError("Frozen inputs changed")
    save(
        out / "results.json",
        {
            "registration": registration,
            "results": results,
            "static_priors": priors,
            "seconds": time.monotonic() - started,
        },
    )
    save(
        out / "artifact_manifest.json",
        {
            "files": {
                p.name: digest(p)
                for p in sorted(out.iterdir())
                if p.is_file() and p.name != "artifact_manifest.json"
            }
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.cache.resolve(), args.out.resolve())
    except Exception as exc:
        if args.out.exists():
            (args.out / "FAILURE.json").write_text(
                json.dumps({"type": type(exc).__name__, "error": str(exc)}, indent=2)
                + "\n"
            )
        raise
