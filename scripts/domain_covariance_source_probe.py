"""Three source-room folds testing full covariance features without projection.

This is a bounded development probe, not the reserved-room evaluation. No E04
record can enter this loader. Static pose remains unidentifiable from paired
covariance invariants alone; relative-pose and displacement results are separate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.covariance_dynamics import (  # noqa: E402
    CONDITION_LIMIT,
    LAGS,
    covariance_features,
    receiver_covariances,
)
from ghostmesh.covariance_accelerated import (  # noqa: E402
    covariance_features as cached_shape_features,
)
from scripts.domain_background_probe import (  # noqa: E402
    digest,
    multiply,
    scores,
)

CONFIG = dict(
    seed=2027,
    permutation_seed=90210,
    context=8,
    endpoint_stride=8,
    endpoints=list(range(39, 297, 8)),
    raw_history=40,
    invalid_history_policy="Reject every canonical endpoint whose forty-frame history includes a decoder-rejected frame. Exact same mask in allarms. No filled covariance enters any retained example. Require at least90percent endpoints and all people per role.",
    maximum_rejected_fraction=0.1,
    covariance_lags=LAGS,
    arms=["untreated_covariance", "log_shape_tangent", "congruence_dynamics"],
    raw_dimensions=dict(
        untreated_covariance=8 * 19 * 7 * 9,
        log_shape_tangent=8 * 19 * 7 * 9,
        congruence_dynamics=8 * 19 * 6 * 3,
    ),
    feature_normalization="Source-training column mean/SD only; SD floor .001",
    rff_dimensions=512,
    projection_dimensions=None,
    rbf_bandwidth="Median squared distance of128 deterministic source-training samples after source normalization; no held data",
    rff_normalization="Source-training RFF column mean/SD only; SD floor .001",
    ridge_alphas=[0.001, 0.01, 0.1, 1.0, 10.0],
    ridge_equation="(Xc.T Xc+n*alpha*I)W=Xc.T(Y-Ytrainmean); unpenalized intercept",
    targets=["root_relative_pose", "lag8_root_relative_displacement"],
    selection="Separate source-validation coordinate MSE per target; first alpha wins ties",
    source_validation_subjects=["S04", "S09", "S14", "S19", "S24", "S29"],
    folds=["E01", "E02", "E03"],
    counts_per_fold=dict(train=14256, validation=3564, held=8910),
    comparison="Every arm uses identical endpoints, forty-frame raw histories, source roles, RFF512 and alpha grid; no preliminary64-dimensional projection",
    continuation_gate="Congruence-dynamics record-centered poseR2 and raw-zero-reference displacementR2 positive in all3held rooms; macro poseR2>=.05; fixed within-record shuffle lowers macro poseR2>=.05. Heuristic only, not statistical significance.",
    scope="DEVELOPMENT ONLY complete source30 E01-E03, no E04, no target calibration labels",
    limits="Assumes fixed complex receiver mixing per frequency bin within compared times; real room physics need not obey this. Pair invariants cannot identify unknown static pose. This is not a complete absolute-pose/calibration product or 3D/Doppler/AoA claim.",
)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate_records(records):
    expected = [
        f"E{room:02}/S{person:02}/A{action:02}"
        for room in range(1, 4)
        for person in range((room - 1) * 10 + 1, room * 10 + 1)
        for action in range(1, 28)
    ]
    if [r["id"] for r in records] != expected or any(
        r["frames"] != 297 or r["offset"] != i * 297 for i, r in enumerate(records)
    ):
        raise ValueError("Only ordered complete810source records E01-E03 are allowed")


def examples(raw):
    covariance = receiver_covariances(
        raw, magnitude_units="log_magnitude_db", phase_units="radians"
    )
    values = covariance_features(covariance)
    ends = np.asarray(CONFIG["endpoints"])
    context = ends[:, None] - max(LAGS) - np.arange(7, -1, -1)[None]
    result = {
        name: feature[context].reshape(len(ends), -1)
        for name, feature in values.items()
    }
    eigenvalues = np.linalg.eigvalsh(covariance)
    condition = float(np.max(eigenvalues[..., -1] / eigenvalues[..., 0]))
    return result, condition


def available_examples(raw, *, feature_function=covariance_features):
    """Return complete contiguous histories only, never a filled covariance."""
    raw = np.asarray(raw)
    if raw.shape != (297, 6, 114, 10):
        raise ValueError("Expected complete297frame typed RF record")
    complete = (np.isfinite(raw[:, :3]) & np.isfinite(raw[:, 3:])).all(1)
    counts = complete.reshape(297, 19, 60).sum(2)
    valid = (counts >= 57).all(1)
    rejected = [
        dict(
            frame=int(index),
            reason="More than5percent incomplete three-receiver vectors/bin",
        )
        for index in np.flatnonzero(~valid)
    ]
    # Batch valid frames once; placeholder rows never enter covariance_features.
    matrices = np.zeros((297, 19, 3, 3), dtype="complex128")
    if valid.any():
        indices = np.flatnonzero(valid)
        covariance = receiver_covariances(
            raw[valid], magnitude_units="log_magnitude_db", phase_units="radians"
        )
        eigenvalues = np.linalg.eigvalsh(covariance)
        positive = (eigenvalues[..., 0] > 0).all(1)
        ratio = np.full_like(eigenvalues[..., 0], np.inf)
        np.divide(
            eigenvalues[..., -1],
            eigenvalues[..., 0],
            out=ratio,
            where=eigenvalues[..., 0] > 0,
        )
        accepted = positive & (ratio <= CONDITION_LIMIT).all(1)
        for index in indices[~accepted]:
            valid[index] = False
            rejected.append(
                dict(
                    frame=int(index),
                    reason="Singular or numerically ill-conditioned covariance; no identity jitter applied",
                )
            )
        matrices[indices[accepted]] = covariance[accepted]
    rejected.sort(key=lambda row: row["frame"])
    retained = np.asarray(
        [end for end in CONFIG["endpoints"] if valid[end - 39 : end + 1].all()],
        dtype=np.int64,
    )
    result = {
        arm: np.empty((len(retained), CONFIG["raw_dimensions"][arm]))
        for arm in CONFIG["arms"]
    }
    condition = 0.0
    padded = np.concatenate(([False], valid, [False])).astype(np.int8)
    starts = np.flatnonzero(np.diff(padded) == 1)
    stops = np.flatnonzero(np.diff(padded) == -1)
    for start, stop in zip(starts, stops):
        selected = (retained >= start + 39) & (retained < stop)
        if not selected.any():
            continue
        covariance = matrices[start:stop]
        values = feature_function(covariance)
        context = (
            retained[selected, None] - start - max(LAGS) - np.arange(7, -1, -1)[None]
        )
        for arm in result:
            result[arm][selected] = values[arm][context].reshape(selected.sum(), -1)
        eigenvalues = np.linalg.eigvalsh(covariance)
        condition = max(
            condition, float(np.max(eigenvalues[..., -1] / eigenvalues[..., 0]))
        )
    return result, retained, condition, rejected


def fold_masks(room, person, held_room, nominal_counts=None):
    nominal_counts = nominal_counts or CONFIG["counts_per_fold"]
    validation = (room != held_room) & np.isin(
        person, CONFIG["source_validation_subjects"]
    )
    train = (room != held_room) & ~validation
    held = room == held_room
    for name, mask in (("train", train), ("validation", validation), ("held", held)):
        if (
            int(mask.sum())
            < (1 - CONFIG["maximum_rejected_fraction"]) * nominal_counts[name]
        ):
            raise ValueError("Too many unavailable source histories: " + name)
    if tuple(len(set(person[x])) for x in (train, validation, held)) != (16, 4, 10):
        raise ValueError("Not all required people retain valid histories")
    if any(
        set(person[a]) & set(person[b])
        for a, b in ((train, validation), (train, held), (validation, held))
    ):
        raise ValueError("Person roles overlap")
    return train, validation, held


def feature_stats(feature, training):
    indices = np.flatnonzero(training)
    total = np.zeros(feature.shape[1], dtype="float64")
    squared = total.copy()
    for start in range(0, len(indices), 128):
        x = np.asarray(feature[indices[start : start + 128]], dtype="float64")
        total += x.sum(0)
        squared += (x * x).sum(0)
    mean = total / len(indices)
    scale = np.maximum(
        np.sqrt(np.maximum(squared / len(indices) - mean * mean, 0)), 0.001
    )
    return mean, scale


def rff_transform(feature, train, out):
    mean, scale = feature_stats(feature, train)
    rng = np.random.default_rng(CONFIG["seed"])
    sample_indices = rng.choice(np.flatnonzero(train), 128, replace=False)
    sample = (np.asarray(feature[sample_indices], dtype="float64") - mean) / scale
    distances = np.maximum(
        (sample * sample).sum(1)[:, None]
        + (sample * sample).sum(1)[None]
        - 2 * multiply(sample, sample.T),
        0,
    )
    bandwidth = float(np.median(distances[np.triu_indices(128, 1)]))
    if bandwidth <= 0 or not np.isfinite(bandwidth):
        raise ValueError("Degenerate source-only RBF bandwidth")
    rng = np.random.default_rng(CONFIG["seed"])
    weight = rng.normal(size=(feature.shape[1], CONFIG["rff_dimensions"])) / np.sqrt(
        bandwidth
    )
    offset = rng.uniform(0, 2 * np.pi, CONFIG["rff_dimensions"])
    result = np.empty((len(feature), CONFIG["rff_dimensions"]), dtype="float64")
    for start in range(0, len(feature), 128):
        x = (np.asarray(feature[start : start + 128], dtype="float64") - mean) / scale
        result[start : start + len(x)] = np.sqrt(2 / CONFIG["rff_dimensions"]) * np.cos(
            multiply(x, weight) + offset
        )
    rff_mean = result[train].mean(0)
    rff_scale = np.maximum(result[train].std(0), 0.001)
    np.savez(
        out,
        feature_mean=mean,
        feature_scale=scale,
        weight=weight,
        offset=offset,
        bandwidth_squared=np.asarray(bandwidth),
        bandwidth_sample_indices=sample_indices,
        rff_mean=rff_mean,
        rff_scale=rff_scale,
    )
    return (result - rff_mean) / rff_scale


def prediction_metrics(p, d, target, truth_delta, record, action):
    result = scores(p, d, target, truth_delta, record)
    result["per_action"] = {
        key: scores(p[ix], d[ix], target[ix], truth_delta[ix], record[ix])
        for key in sorted(set(action))
        for ix in [action == key]
    }
    return result


def selected_records(records, limited_actions=False):
    validate_records(records)
    return [
        r
        for r in records
        if not limited_actions or r["id"].split("/")[2] in ("A03", "A07", "A12")
    ]


def run(cache, out, *, limited_actions=False, cached_shapes=False):
    if out.exists():
        raise FileExistsError(out)
    started = time.monotonic()
    receipt = json.loads((cache / "receipt.json").read_text())
    records = selected_records(receipt["records"], limited_actions)
    nominal_counts = {
        name: people * (3 if limited_actions else 27) * 33
        for name, people in (("train", 16), ("validation", 4), ("held", 10))
    }
    config = dict(
        CONFIG,
        counts_per_fold=nominal_counts,
        selected_actions=["A03", "A07", "A12"]
        if limited_actions
        else [f"A{i:02}" for i in range(1, 28)],
        records=len(records),
        cached_shapes=cached_shapes,
    )
    if limited_actions:
        config["scope"] = (
            "DEVELOPMENT ONLY fixed A03/A07/A12 acrossall30sourcepeople/E01-E03; scope registered beforeanyfit. No otheraction/general efficacy/finaltest claim; no E04/targetcalibration labels. Full810recordcache integrity still verified."
        )
    for key, expected in receipt["arrays"].items():
        if digest(cache / (key + ".npy")) != expected:
            raise ValueError("Source array changed")
    rf, labels, valid = [
        np.load(cache / (name + ".npy"), mmap_mode="r", allow_pickle=False)
        for name in ("rf", "pose", "valid")
    ]
    if (
        rf.shape != (240570, 6, 114, 10)
        or labels.shape != (240570, 17, 2)
        or not valid.all()
    ):
        raise ValueError("Unexpected source array shape/labels")
    source_paths = [
        Path(__file__),
        ROOT / "ghostmesh/covariance_dynamics.py",
        ROOT / "ghostmesh/covariance_accelerated.py",
        ROOT / "ghostmesh/metrics.py",
        ROOT / "scripts/domain_background_probe.py",
    ]
    sources = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    registration = dict(
        config=config,
        source_sha256=sources,
        cache_receipt_sha256=digest(cache / "receipt.json"),
        cache_arrays_sha256=receipt["arrays"],
        software=dict(numpy=np.__version__, python=platform.python_version()),
    )
    out.mkdir(parents=True)
    save(out / "registration.json", registration)
    for source in source_paths:
        destination = out / "executed-source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    ends = np.asarray(CONFIG["endpoints"])
    n = len(records) * len(ends)
    features = {
        arm: np.lib.format.open_memmap(
            out / (arm + "-features.npy"),
            mode="w+",
            dtype="float32",
            shape=(n, CONFIG["raw_dimensions"][arm]),
        )
        for arm in CONFIG["arms"]
    }
    target, delta, rooms, people, names, actions, endpoints, quality = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    cursor = 0
    for i, record in enumerate(records):
        offset = record["offset"]
        values, record_ends, condition, rejected = available_examples(
            np.asarray(rf[offset : offset + 297]),
            feature_function=cached_shape_features
            if cached_shapes
            else covariance_features,
        )
        for arm in features:
            features[arm][cursor : cursor + len(record_ends)] = values[arm]
        cursor += len(record_ends)
        y = np.asarray(labels[offset : offset + 297], dtype="float64")
        relative = y - y[:, [11, 12]].mean(1, keepdims=True)
        target.extend(relative[record_ends])
        delta.extend(relative[record_ends] - relative[record_ends - 8])
        room, person, action = record["id"].split("/")
        rooms.extend([room] * len(record_ends))
        people.extend([person] * len(record_ends))
        names.extend([record["id"]] * len(record_ends))
        actions.extend([action] * len(record_ends))
        endpoints.extend(record_ends.tolist())
        quality.append(
            dict(
                record=record["id"],
                maximum_covariance_condition=condition,
                retained_endpoints=record_ends.tolist(),
                rejected_frames=rejected,
            )
        )
        if (i + 1) % 10 == 0:
            print(
                json.dumps(
                    dict(
                        features_records=i + 1,
                        total=len(records),
                        seconds=time.monotonic() - started,
                    )
                ),
                flush=True,
            )
    for value in features.values():
        value.flush()
    if cursor < (1 - CONFIG["maximum_rejected_fraction"]) * n:
        raise ValueError("More than10percent source histories unavailable")
    n = cursor
    features = {arm: feature[:n] for arm, feature in features.items()}
    target, delta = np.asarray(target), np.asarray(delta)
    room, person, record, action = map(np.asarray, (rooms, people, names, actions))
    np.savez(
        out / "targets-and-metadata.npz",
        pose=target,
        delta=delta,
        room=room,
        person=person,
        record=record,
        action=action,
        end=np.asarray(endpoints),
    )
    save(out / "feature-quality.json", quality)
    y = np.concatenate((target.reshape(n, 34), delta.reshape(n, 34)), axis=1)
    results, priors = [], []
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
            xmean, ymean = z[train].mean(0), y[train].mean(0)
            x, yt = z[train] - xmean, y[train] - ymean
            system = multiply(x.T, x)
            eigen, vector = eigh(system, check_finite=True)
            right = multiply(vector.T, multiply(x.T, yt))
            grid, best, allweights = [], [None, None], []
            for alpha in CONFIG["ridge_alphas"]:
                weight = multiply(
                    vector, right / (eigen[:, None] + alpha * train.sum())
                )
                allweights.append(weight)
                pred = multiply(z - xmean, weight) + ymean
                mse = [
                    float(
                        (
                            (
                                pred[val, j * 34 : (j + 1) * 34]
                                - y[val, j * 34 : (j + 1) * 34]
                            )
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
    if digest(cache / "receipt.json") != registration["cache_receipt_sha256"] or any(
        digest(cache / (key + ".npy")) != expected
        for key, expected in receipt["arrays"].items()
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
    parser.add_argument(
        "--limited-actions",
        action="store_true",
        help="Fixed A03/A07/A12 development scope, declared before fitting",
    )
    parser.add_argument(
        "--cached-shapes",
        action="store_true",
        help="Use bitexact cached determinant-shape optimization",
    )
    args = parser.parse_args()
    try:
        run(
            args.cache,
            args.out,
            limited_actions=args.limited_actions,
            cached_shapes=args.cached_shapes,
        )
    except BaseException as exc:
        if args.out.exists():
            save(
                args.out / "failure.json", dict(type=type(exc).__name__, error=str(exc))
            )
        raise
