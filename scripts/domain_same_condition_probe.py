"""Predeclared source-only same-person/room held-time and held-action probe.

This learning-capability diagnostic is neither room generalization nor a product
calibration evaluation. E04 arrays are forbidden. Models consume CSI only.
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
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.metrics import pose_metrics  # noqa:E402
from ghostmesh.rf_features import quotient_features  # noqa:E402
from scripts.domain_background_probe import multiply  # noqa:E402
from scripts.domain_structured_probe import motion_metrics  # noqa:E402

CONFIG = {
    "seed": 2027,
    "context": 8,
    "records": ["E01/S01/A03", "E01/S01/A07", "E01/S01/A12", "E01/S01/A14"],
    "held_time": {
        "record": "E01/S01/A12",
        "raw_blocks_half_open": {"train": [0, 118], "validation": [126, 178], "test": [186, 297]},
        "boundary_note": "First40%, next20%, last40% with eight raw frames purged after each nominal boundary; every8framecontext lies wholly inside its block",
    },
    "held_action": {
        "train": ["E01/S01/A03", "E01/S01/A07"],
        "validation": ["E01/S01/A12"],
        "test": ["E01/S01/A14"],
    },
    "features": {
        "raw_db": "Full8x3x114x10 original log magnitude/20; same-frame per-link finite-median imputation only; no extra log or packet pooling",
        "quotient": "Full8x6x114x10 clipped interRX log-magnitude ratios and circular phase differences; no frequency/packet pooling",
    },
    "normalization": "Per-column mean/SD fitted on training endpoints only, SD floor.05 in the feature's defined units",
    "kernels": ["linear", "rbf"],
    "rbf_bandwidth": "Median squared Euclidean distance between all source-training endpoints after frozen training normalization; no validation/test fitting",
    "ridge_lambdas": [1e-5, 0.001, 0.1, 10.0],
    "ridge": "Exact dual kernel regression of training-mean-centered34absolute2D targets, (K+lambda*I)A=Y-mean; fixed training mean intercept",
    "selection": "Lowest validation NJE for each feature/kernel; first lambda wins ties. Test metrics never select the model",
    "controls": ["training_reference_mean", "last_training_reference", "CSI_nearest_reference", "within_record_prediction_shuffle", "within_record_input_window_shuffle", "past7_lag_permutation_current_frame_kept"],
    "past_lag_permutation": [6, 4, 2, 0, 5, 3, 1, 7],
    "metric_clipping": "Absolute coordinates clipped[0,1] with raw selected predictions and clipping fractions saved",
    "threads": 1,
    "scope": "Development only; fixed same E01/S01 person and room, no E04, no room/general population/product-calibration claim",
    "lineage": "A12 is reused across these two distinct DEVELOPMENT protocols; no new untouched-test claim. Entire A14 is absent from training/validation in held-action protocol",
}


def digest(path):
    result = sha256()
    with Path(path).open("rb") as stream:
        for value in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(value)
    return result.hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def endpoints(start, stop, context=8):
    if not 0 <= start < stop or stop - start < context:
        raise ValueError("Block must contain a complete causal context")
    ends = np.arange(start + context - 1, stop)
    support = ends[:, None] - np.arange(context - 1, -1, -1)[None]
    if support.min() < start or support.max() >= stop:
        raise ValueError("Context escaped block")
    return ends, support


def normalize_fit(x, training):
    mean = x[training].mean(0)
    scale = np.maximum(x[training].std(0), 0.05)
    transformed = (x - mean) / scale
    if not np.isfinite(transformed).all():
        raise ValueError("Nonfinite normalized CSI")
    return transformed, {"input_mean": mean, "input_scale": scale}


def kernel_inputs(x, training):
    source = x[training]
    cross = multiply(x, source.T)
    distance = np.maximum((x * x).sum(1)[:, None] + (source * source).sum(1)[None] - 2 * cross, 0)
    triangle = distance[training][np.triu_indices(len(source), 1)]
    width = float(np.median(triangle))
    if not np.isfinite(width) or width <= 0:
        raise ValueError("Degenerate training-only RBF width")
    return {"linear": cross / x.shape[1], "rbf": np.exp(-distance / (2 * width))}, distance, width


def kernel_predict_input(x, source, kernel, width):
    cross = multiply(x, source.T)
    if kernel == "linear":
        return cross / x.shape[1]
    distance = np.maximum((x * x).sum(1)[:, None] + (source * source).sum(1)[None] - 2 * cross, 0)
    return np.exp(-distance / (2 * width))


def dual_solutions(kernel, target, training):
    source = kernel[training]
    values, vectors = eigh(source, check_finite=True)
    if values.min() < -1e-7:
        raise ValueError("Kernel is not numerically positive semidefinite")
    y = target[training].astype("float64").reshape(training.sum(), -1)
    mean = y.mean(0)
    right = multiply(vectors.T, y - mean)
    for regularization in CONFIG["ridge_lambdas"]:
        dual = multiply(vectors, right / (values[:, None] + regularization))
        residual = multiply(source, dual) + regularization * dual - (y - mean)
        relative = float(np.linalg.norm(residual) / max(np.linalg.norm(y - mean), 1e-12))
        if not np.isfinite(relative) or relative > 1e-6:
            raise ValueError("Kernel normal-equation residual too large")
        yield regularization, dual, mean, relative


def score(prediction, truth, records):
    result = pose_metrics(prediction, truth, np.ones(truth.shape[:2], bool))
    result.pop("per_frame_nje")
    return {"metrics": result, "motion": motion_metrics(prediction, truth, records)}


def permutations(records):
    result = np.arange(len(records))
    rng = np.random.default_rng(CONFIG["seed"])
    for record in sorted(set(records)):
        indices = np.flatnonzero(records == record)
        result[indices] = rng.permutation(indices)
    return result


def lag_permute(x):
    n = len(x)
    if x.shape[1] % CONFIG["context"]:
        raise ValueError("Unexpected flattened context")
    return x.reshape(n, 8, -1)[:, CONFIG["past_lag_permutation"]].reshape(x.shape)


def record_features(raw):
    magnitude, phase = raw[:, :3].astype("float64"), raw[:, 3:].astype("float64")
    finite = np.isfinite(magnitude)
    medians = np.nanmedian(np.where(finite, magnitude, np.nan), axis=(2, 3), keepdims=True)
    if not np.isfinite(medians).all() or (1 - finite.mean()) > 0.05:
        raise ValueError("Invalid magnitude coverage")
    filled = np.where(finite, magnitude, medians) / 20
    quotient = quotient_features(magnitude, phase, units="log_magnitude_db").astype("float64")
    return {"raw_db": filled, "quotient": quotient}


def data_for_protocol(records, protocol, feature):
    samples, targets, names, ends, supports, labels = [], [], [], [], [], []
    if protocol == "held_time":
        definitions = [(split, CONFIG["held_time"]["record"], *block) for split, block in CONFIG["held_time"]["raw_blocks_half_open"].items()]
    else:
        definitions = [(split, record, 0, 297) for split, ids in CONFIG["held_action"].items() for record in ids]
    for split, record, start, stop in definitions:
        endpoint, support = endpoints(start, stop)
        x = records[record]["features"][feature]
        samples.append(x[support].reshape(len(support), -1))
        targets.append(records[record]["pose"][endpoint])
        names.extend([record] * len(endpoint))
        ends.extend(endpoint)
        supports.append(support + records[record]["offset"])
        labels.extend([split] * len(endpoint))
    return np.concatenate(samples), np.concatenate(targets), np.asarray(names), np.asarray(ends), np.concatenate(supports), np.asarray(labels)


def run(cache, output):
    if output.exists():
        raise FileExistsError(output)
    receipt = json.loads((cache / "receipt.json").read_text())
    if receipt["frames"] != 240570 or len(receipt["records"]) != 810 or any(r["id"].split("/")[0] not in {"E01", "E02", "E03"} for r in receipt["records"]):
        raise ValueError("Only source30 cache is allowed")
    for name, expected in receipt["arrays"].items():
        if digest(cache / (name + ".npy")) != expected:
            raise ValueError("Source cache changed")
    source_paths = [Path(__file__), ROOT / "ghostmesh/metrics.py", ROOT / "ghostmesh/rf_features.py", ROOT / "scripts/domain_background_probe.py", ROOT / "scripts/domain_structured_probe.py"]
    source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in source_paths}
    output.mkdir(parents=True)
    started = time.monotonic()
    registration = {"config": CONFIG, "sources_sha256": source_hashes, "cache_receipt_sha256": digest(cache / "receipt.json"), "cache_arrays_sha256": receipt["arrays"], "numpy": np.__version__, "platform": platform.platform()}
    write(output / "registration.json", registration)
    for path in source_paths:
        destination = output / "executed-source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    raw = np.load(cache / "rf.npy", mmap_mode="r", allow_pickle=False)
    pose = np.load(cache / "pose.npy", mmap_mode="r", allow_pickle=False)
    valid = np.load(cache / "valid.npy", mmap_mode="r", allow_pickle=False)
    records, bindings = {}, {}
    for record in CONFIG["records"]:
        row = next(r for r in receipt["records"] if r["id"] == record)
        offset = row["offset"]
        rf = np.array(raw[offset : offset + 297], dtype="float64")
        y = np.array(pose[offset : offset + 297], dtype="float64")
        if not valid[offset : offset + 297].all() or not np.isfinite(y).all():
            raise ValueError("Complete selected2D targets required")
        records[record] = {"offset": offset, "pose": y, "features": record_features(rf)}
        bindings[record] = {"offset": offset, "ordered_raw_sha256": row["ordered_raw_sha256"], "cached_rf_sha256": sha256(np.ascontiguousarray(raw[offset : offset + 297]).tobytes()).hexdigest(), "cached_pose_sha256": sha256(np.ascontiguousarray(pose[offset : offset + 297]).tobytes()).hexdigest()}
    write(output / "selected-record-bindings.json", bindings)
    rows = []
    for protocol in ("held_time", "held_action"):
        for feature in ("raw_db", "quotient"):
            x, target, names, ends, supports, split = data_for_protocol(records, protocol, feature)
            train, val, test = (split == key for key in ("train", "validation", "test"))
            if any(set(supports[first].ravel()) & set(supports[second].ravel()) for first, second in ((train, val), (train, test), (val, test))):
                raise ValueError("Raw contexts leak across splits")
            if protocol == "held_action" and any(set(names[first]) & set(names[second]) for first, second in ((train, val), (train, test), (val, test))):
                raise ValueError("Held action recording leaked")
            prefix = f"{protocol}-{feature}"
            np.savez_compressed(output / (prefix + "-indices.npz"), records=names, end=ends, raw_support=supports, split=split)
            transformed, normalizer = normalize_fit(x, train)
            kernels, distance, width = kernel_inputs(transformed, train)
            reference_mean = target[train].mean(0)
            nearest = target[train][distance.argmin(1)]
            control_values = {"training_reference_mean": np.broadcast_to(reference_mean, target.shape), "last_training_reference": np.broadcast_to(target[train][-1], target.shape), "CSI_nearest_reference": nearest}
            for control, prediction in control_values.items():
                row = {"protocol": protocol, "feature": feature, "kernel": control, "regularization": None}
                for label, mask in (("train", train), ("validation", val), ("test", test)):
                    row[label] = score(prediction[mask], target[mask], names[mask])
                rows.append(row)
                np.savez_compressed(output / (prefix + "-" + control + "-predictions.npz"), prediction=prediction, target=target)
            for kernel_name, kernel in kernels.items():
                best, candidates, coefficients = None, [], []
                for regularization, dual, mean, residual in dual_solutions(kernel, target, train):
                    raw_prediction = (mean + multiply(kernel, dual)).reshape(-1, 17, 2)
                    prediction = raw_prediction.clip(0, 1)
                    validation = score(prediction[val], target[val], names[val])
                    candidates.append({"regularization": regularization, "validation_nje": validation["metrics"]["normalized_joint_error"], "normal_equation_relative_residual": residual})
                    coefficients.append(dual)
                    if best is None or validation["metrics"]["normalized_joint_error"] < best[0]:
                        best = (validation["metrics"]["normalized_joint_error"], regularization, dual, mean, prediction, raw_prediction)
                _, selected, dual, mean, prediction, raw_prediction = best
                permutation = permutations(names[test])
                input_shuffle_kernel = kernel_predict_input(transformed[test][permutation], transformed[train], kernel_name, width)
                input_shuffled = (mean + multiply(input_shuffle_kernel, dual)).reshape(-1, 17, 2).clip(0, 1)
                predicted_shuffled = prediction[test][permutation]
                shuffle_difference = float(np.max(abs(input_shuffled - predicted_shuffled)))
                if shuffle_difference > 1e-9:
                    raise ValueError("Input/window-output shuffle identity failed")
                lag_raw = lag_permute(x[test])
                lag_x = (lag_raw - normalizer["input_mean"]) / normalizer["input_scale"]
                lag_kernel = kernel_predict_input(lag_x, transformed[train], kernel_name, width)
                lag_prediction = (mean + multiply(lag_kernel, dual)).reshape(-1, 17, 2).clip(0, 1)
                name = prefix + "-" + kernel_name
                np.savez_compressed(output / (name + "-weights.npz"), **normalizer, training_design=transformed[train], training_target=target[train], dual=np.stack(coefficients), lambdas=np.asarray(CONFIG["ridge_lambdas"]), selected_lambda=np.asarray(selected), target_mean=mean, bandwidth_squared=np.asarray(width))
                np.savez_compressed(output / (name + "-predictions.npz"), prediction=prediction, raw_prediction=raw_prediction, target=target, shuffle_indices=permutation, prediction_shuffle=predicted_shuffled, input_window_shuffle=input_shuffled, past_lag_permutation=lag_prediction)
                row = {"protocol": protocol, "feature": feature, "kernel": kernel_name, "regularization": selected, "candidates": candidates, "counts": {"train": int(train.sum()), "validation": int(val.sum()), "test": int(test.sum())}, "held_coordinate_clipping_fraction": float(np.mean((raw_prediction[test] < 0) | (raw_prediction[test] > 1))), "input_output_shuffle_max_difference": shuffle_difference}
                for label, mask in (("train", train), ("validation", val), ("test", test)):
                    row[label] = score(prediction[mask], target[mask], names[mask])
                row["test_controls"] = {"prediction_shuffle": score(predicted_shuffled, target[test], names[test]), "input_window_shuffle": score(input_shuffled, target[test], names[test]), "past_lag_permutation": score(lag_prediction, target[test], names[test])}
                rows.append(row)
                print(json.dumps({"protocol": protocol, "feature": feature, "kernel": kernel_name, "lambda": selected, "test_nje": row["test"]["metrics"]["normalized_joint_error"], "relative_motion_r2": row["test"]["motion"]["relative"]["within_record_r2"]}), flush=True)
            del x, transformed, kernels, distance
    for record, value in bindings.items():
        offset = value["offset"]
        if sha256(np.ascontiguousarray(raw[offset : offset + 297]).tobytes()).hexdigest() != value["cached_rf_sha256"] or sha256(np.ascontiguousarray(pose[offset : offset + 297]).tobytes()).hexdigest() != value["cached_pose_sha256"]:
            raise ValueError("Selected source arrays changed during probe")
    for name, expected in source_hashes.items():
        if digest(ROOT / name) != expected:
            raise ValueError("Source changed during probe")
    write(output / "results.json", {"registration": registration, "results": rows, "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.cache, args.out)
