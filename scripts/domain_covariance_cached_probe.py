"""Fit identical covariance controls from an integrity-checked full feature cache.

This separates reusable signal computation from source-only statistical fitting.
No E04, raw-image input or calibration label is accessed by this development run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import domain_covariance_full_cache as cache_tools  # noqa: E402
from scripts.domain_covariance_source_probe import (  # noqa: E402
    CONFIG,
    digest,
    fold_masks,
    multiply,
    prediction_metrics,
    rff_transform,
    save,
)


def load_inputs(cache):
    receipt = cache_tools.read_json(cache / "receipt.json")
    cache_tools.require(
        receipt["kind"] == "full-source-covariance-feature-cache"
        and receipt["fits_performed"] == 0,
        "Expected completed full feature cache",
    )
    cache_tools.study.validate_records(receipt["records"])
    cache_tools.verify_hashes(cache, receipt["arrays_sha256"])
    cache_tools.verify_hashes(
        cache, cache_tools.read_json(cache / "artifact_manifest.json")
    )
    cache_tools.verify_hashes(ROOT, receipt["source_sha256"])
    with np.load(cache / "targets-and-metadata.npz", allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    n = len(data["pose"])
    cache_tools.require(
        n == receipt["logical_rows"] and receipt["nominal_rows"] == 26730,
        "Logical row count differs",
    )
    cache_tools.record_blocks(
        data,
        cache_tools.read_json(cache / "feature-quality.json"),
        [r["id"] for r in receipt["records"]],
    )
    coverage = cache_tools.coverage(data, receipt["records"])
    cache_tools.require(
        coverage == receipt["coverage_per_fold"],
        "Source role coverage differs from cache",
    )
    features = cache_tools.logical_features(cache, n, receipt["nominal_rows"])
    for feature in features.values():
        for start in range(0, n, 256):
            cache_tools.require(
                np.isfinite(feature[start : start + 256]).all(),
                "Nonfinite retained features",
            )
    return receipt, data, features


def select_ridge(z, y, train, val, held):
    """Fit source labels and select on source validation; held labels unused."""
    if z.ndim != 2 or y.shape != (len(z), 68):
        raise ValueError("Expected design[N,D] and two34-coordinate targets")
    masks = (train, val, held)
    if any(
        mask.dtype != bool or mask.shape != (len(z),) or not mask.any()
        for mask in masks
    ) or not np.all(sum(mask.astype(int) for mask in masks) == 1):
        raise ValueError("Training, validation and held roles must partition examples")
    xmean, ymean = z[train].mean(0), y[train].mean(0)
    x, yt = z[train] - xmean, y[train] - ymean
    system = multiply(x.T, x)
    eigen, vector = eigh(system, check_finite=True)
    right = multiply(vector.T, multiply(x.T, yt))
    grid, best, allweights = [], [None, None], []
    for alpha in CONFIG["ridge_alphas"]:
        weight = multiply(vector, right / (eigen[:, None] + alpha * train.sum()))
        allweights.append(weight)
        pred = multiply(z - xmean, weight) + ymean
        mse = [
            float(
                (
                    (pred[val, j * 34 : (j + 1) * 34] - y[val, j * 34 : (j + 1) * 34])
                    ** 2
                ).mean()
            )
            for j in range(2)
        ]
        grid.append(
            dict(
                alpha=alpha,
                validation_mse_pose=mse[0],
                validation_mse_delta=mse[1],
            )
        )
        for j in range(2):
            if best[j] is None or mse[j] < best[j][0]:
                best[j] = (
                    mse[j],
                    alpha,
                    weight[:, j * 34 : (j + 1) * 34].copy(),
                    pred[held, j * 34 : (j + 1) * 34].copy(),
                )
    return xmean, ymean, grid, best, allweights


def run(cache, out):
    if out.exists():
        raise FileExistsError(out)
    started = time.monotonic()
    receipt, data, features = load_inputs(cache)
    n = len(data["pose"])
    target, delta = data["pose"], data["delta"]
    room, person, record, action = (
        data[key] for key in ("room", "person", "record", "action")
    )
    config = dict(
        CONFIG,
        selected_actions=[f"A{i:02}" for i in range(1, 28)],
        records=810,
        cached_shapes=True,
        feature_source="Verified reusable full-source covariance cache",
        scope="FULL27-ACTION SOURCE DEVELOPMENT E01-E03 only; no E04, target calibration labels or new final efficacy claim",
    )
    source_paths = [
        Path(__file__),
        ROOT / "scripts/domain_covariance_full_cache.py",
        ROOT / "scripts/domain_covariance_source_probe.py",
        ROOT / "scripts/verify_covariance_source_probe.py",
        ROOT / "ghostmesh/covariance_dynamics.py",
        ROOT / "ghostmesh/covariance_accelerated.py",
        ROOT / "ghostmesh/metrics.py",
        ROOT / "scripts/domain_background_probe.py",
    ]
    sources = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    registration = dict(
        config=config,
        source_sha256=sources,
        cache_receipt_sha256=receipt["raw_receipt_sha256"],
        cache_arrays_sha256=receipt["raw_arrays_sha256"],
        feature_cache_receipt_sha256=digest(cache / "receipt.json"),
        feature_cache_arrays_sha256=receipt["arrays_sha256"],
        feature_cache_logical_rows=n,
        feature_cache_reusecounts=receipt["reusecounts"],
        source_selection="Training/validation people only; frozen before these full-action fits",
        lineage="The mechanism and motion continuation criterion were motivated by the limited3action development result, including its observed held-room scores; this is not an untouched final hypothesis test",
    )
    out.mkdir(parents=True)
    save(out / "registration.json", registration)
    for path in source_paths:
        destination = out / "executed-source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    for name in ("targets-and-metadata.npz", "feature-quality.json"):
        shutil.copyfile(cache / name, out / name)
    save(
        out / "feature-cache-reference.json",
        dict(
            directory=str(cache.resolve()),
            receipt_sha256=digest(cache / "receipt.json"),
            logical_rows=n,
        ),
    )
    y = np.concatenate((target.reshape(n, 34), delta.reshape(n, 34)), axis=1)
    results, priors = [], []
    nominal_counts = CONFIG["counts_per_fold"]
    for held_room in CONFIG["folds"]:
        train, val, held = fold_masks(room, person, held_room, nominal_counts)
        if any(
            set(action[mask]) != set(config["selected_actions"])
            for mask in (train, val, held)
        ):
            raise ValueError("Every role must retain all declared actions")
        held_indices = np.flatnonzero(held)
        groups, held_actions = record[held], action[held]
        perm = np.arange(held.sum())
        rng = np.random.default_rng(CONFIG["permutation_seed"])
        for key in sorted(set(groups)):
            indices = np.flatnonzero(groups == key)
            perm[indices] = rng.permutation(indices)
        np.save(out / (held_room + "-permutation.npy"), perm, allow_pickle=False)
        static = np.broadcast_to(target[train].mean(0), target[held].shape)
        priors.append(
            dict(
                held_room=held_room,
                metrics=prediction_metrics(
                    static,
                    np.zeros_like(delta[held]),
                    target[held],
                    delta[held],
                    groups,
                    held_actions,
                ),
            )
        )
        for arm in CONFIG["arms"]:
            tag = held_room + "-" + arm
            z = rff_transform(features[arm], train, out / (tag + "-rff.npz"))
            np.save(out / (tag + "-design.npy"), z, allow_pickle=False)
            xmean, ymean, grid, best, allweights = select_ridge(z, y, train, val, held)
            p, d = best[0][3].reshape(-1, 17, 2), best[1][3].reshape(-1, 17, 2)
            np.savez(
                out / (tag + "-model-prediction.npz"),
                pose=p,
                delta=d,
                pose_weights=best[0][2],
                delta_weights=best[1][2],
                x_mean=xmean,
                y_mean=ymean,
                held_indices=held_indices,
                all_weights=np.stack(allweights),
                alphas=np.asarray(CONFIG["ridge_alphas"]),
            )
            metrics = prediction_metrics(
                p, d, target[held], delta[held], groups, held_actions
            )
            shuffled = prediction_metrics(
                p[perm], d[perm], target[held], delta[held], groups, held_actions
            )
            row = dict(
                held_room=held_room,
                arm=arm,
                pose_alpha=best[0][1],
                delta_alpha=best[1][1],
                counts=dict(
                    train=int(train.sum()),
                    validation=int(val.sum()),
                    held=int(held.sum()),
                ),
                metrics=metrics,
                shuffled=shuffled,
                source_validation_grid=grid,
            )
            results.append(row)
            save(out / "progress.json", dict(results=results, static_priors=priors))
            print(
                json.dumps(
                    dict(
                        completed=tag,
                        relative_nje=metrics["relative_nje"],
                        pose_r2=metrics["pose_motion"]["r2"],
                        delta_centered_r2=metrics["delta_motion"]["r2"],
                        delta_zero_r2=metrics["delta_zero_r2"],
                        shuffled_pose_r2=shuffled["pose_motion"]["r2"],
                        seconds=time.monotonic() - started,
                    )
                ),
                flush=True,
            )
    invariant = [r for r in results if r["arm"] == "congruence_dynamics"]
    macro_pose = float(np.mean([r["metrics"]["pose_motion"]["r2"] for r in invariant]))
    shuffle_drop = float(
        np.mean(
            [
                r["metrics"]["pose_motion"]["r2"] - r["shuffled"]["pose_motion"]["r2"]
                for r in invariant
            ]
        )
    )
    gate = dict(
        passed=bool(
            all(
                r["metrics"]["pose_motion"]["r2"] > 0
                and r["metrics"]["delta_zero_r2"] > 0
                for r in invariant
            )
            and macro_pose >= 0.05
            and shuffle_drop >= 0.05
        ),
        macro_pose_r2=macro_pose,
        macro_shuffle_drop=shuffle_drop,
    )
    for name, expected in sources.items():
        if digest(ROOT / name) != expected:
            raise ValueError("Executed source changed")
    if digest(cache / "receipt.json") != registration[
        "feature_cache_receipt_sha256"
    ] or any(
        digest(cache / name) != expected
        for name, expected in receipt["arrays_sha256"].items()
    ):
        raise ValueError("Source inputs changed during experiment")
    save(
        out / "results.json",
        dict(
            registration=registration,
            results=results,
            static_priors=priors,
            continuation_gate=gate,
            seconds=time.monotonic() - started,
        ),
    )
    save(
        out / "artifact_manifest.json",
        {
            str(p.relative_to(out)): digest(p)
            for p in sorted(out.rglob("*"))
            if p.is_file()
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.cache, args.out)
    except BaseException as exc:
        if args.out.exists():
            save(
                args.out / "failure.json", dict(type=type(exc).__name__, error=str(exc))
            )
        raise
