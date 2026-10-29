"""Check frame summaries against a fixed source-only held-motion positive."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.observation import observation_features  # noqa:E402
from scripts import domain_same_condition_probe as base  # noqa:E402

CONFIG = {
    "record": "E01/S01/A12",
    "context": 8,
    "raw_blocks_half_open": {"train": [0, 118], "validation": [126, 178], "test": [186, 297]},
    "features": ["observation_full", "observation_amplitude_zero_phase"],
    "kernels": ["linear", "rbf"],
    "ridge_lambdas": [1e-5, 0.001, 0.1, 10.0],
    "normalization": "Training-only per-column mean/SDfloor.05; same rules as frozen raw positive",
    "selection": "Lowest validation NJE, first lambda wins ties, separately per feature/kernel",
    "ablation": "Zero channels6:12 of the same12channel frame summary; six allzero channels carry no information",
    "scope": "One predeclared DEVELOPMENT same-person/room/action held-time positive-control extension; no E04/newactions/roomgeneralization/metatraining",
    "continuation_question": "Does compression preserve the existing CSI-dependent held-motion information before using these inputs for meta-training?",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def windows(frame, support, ablation):
    frame = np.array(frame, dtype="float64")
    require(frame.shape == (297, 12, 114), "Expected exactly the fixed source record")
    if ablation == "observation_amplitude_zero_phase":
        frame[:, 6:] = 0
    elif ablation != "observation_full":
        raise ValueError("Unknown observation arm")
    require(support.ndim == 2 and support.shape[1] == 8 and support.min() >= 0 and support.max() < 297, "Unexpected causal supports")
    return frame[support].reshape(len(support), -1)


def run(cache, predecessor, output):
    require(not output.exists(), "Refusing to replace existing results")
    manifest = json.loads((predecessor / "artifact-manifest-final.json").read_text())["sha256"]
    dependency_names = ["registration.json", "verification.json", "results.json", "selected-record-bindings.json", "held_time-raw_db-indices.npz"]
    for name in dependency_names:
        require(base.digest(predecessor / name) == manifest[name], "Prior positive-control evidence changed")
    receipt = json.loads((cache / "receipt.json").read_text())
    old_registration = json.loads((predecessor / "registration.json").read_text())
    old_verification = json.loads((predecessor / "verification.json").read_text())
    require(base.digest(cache / "receipt.json") == old_registration["cache_receipt_sha256"], "Source receipt changed")
    require(old_verification["source_and_actual_cache_hashes_match"] and old_verification["maximum_saved_prediction_difference"] == 0, "Prior actual-array verification failed")
    require(receipt["config"]["environments"] == ["E01", "E02", "E03"], "Source-only cache required")
    row = next(r for r in receipt["records"] if r["id"] == CONFIG["record"])
    offset = row["offset"]
    raw = np.load(cache / "rf.npy", allow_pickle=False, mmap_mode="r")[offset : offset + 297].copy()
    pose = np.load(cache / "pose.npy", allow_pickle=False, mmap_mode="r")[offset : offset + 297].copy()
    binding = json.loads((predecessor / "selected-record-bindings.json").read_text())[CONFIG["record"]]
    require(sha256(raw.tobytes()).hexdigest() == binding["cached_rf_sha256"], "Actual fixed-record RF changed")
    require(sha256(pose.tobytes()).hexdigest() == binding["cached_pose_sha256"], "Actual fixed-record pose changed")
    with np.load(predecessor / "held_time-raw_db-indices.npz", allow_pickle=False) as z:
        support, names, ends, split = (z[key] for key in ("raw_support", "records", "end", "split"))
    require(set(names) == {CONFIG["record"]} and np.array_equal(support[:, -1] - offset, ends), "Fixed-record endpoints differ")
    train, validation, test = (split == key for key in ("train", "validation", "test"))
    require((train.sum(), validation.sum(), test.sum()) == (111, 45, 104), "Frozen split counts differ")
    for label in ("train", "validation", "test"):
        mask = split == label
        expected_end, expected_support = base.endpoints(*CONFIG["raw_blocks_half_open"][label])
        require(np.array_equal(ends[mask], expected_end) and np.array_equal(support[mask] - offset, expected_support), "Purged block boundaries differ")
    frame = observation_features(raw, magnitude_units="log_magnitude_db", phase_units="radians")
    target = pose[ends].astype("float64")
    source_paths = [Path(__file__), ROOT / "ghostmesh/observation.py", ROOT / "scripts/domain_same_condition_probe.py", ROOT / "scripts/domain_background_probe.py", ROOT / "scripts/domain_structured_probe.py", ROOT / "ghostmesh/rf_features.py", ROOT / "ghostmesh/metrics.py"]
    sources = {str(p.relative_to(ROOT)): base.digest(p) for p in source_paths}
    registration = {"config": CONFIG, "sources_sha256": sources, "predecessor_dependencies": {name: base.digest(predecessor / name) for name in dependency_names}, "cache_receipt_sha256": base.digest(cache / "receipt.json"), "fixed_record_binding": binding, "numpy": np.__version__}
    output.mkdir(parents=True)
    started = time.monotonic()
    base.write(output / "registration.json", registration)
    np.save(output / "frame.npy", frame, allow_pickle=False)
    np.savez_compressed(output / "indices.npz", raw_support=support, records=names, end=ends, split=split, target=target)
    for p in source_paths:
        destination = output / "executed-source" / p.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(p.read_bytes())
    results = []
    for feature in CONFIG["features"]:
        x = windows(frame, support - offset, feature)
        transformed, normalization = base.normalize_fit(x, train)
        matrices, distance, width = base.kernel_inputs(transformed, train)
        prior, last = target[train].mean(0), target[train][-1]
        for name, prediction in (("training_reference_mean", np.broadcast_to(prior, target.shape)), ("last_training_reference", np.broadcast_to(last, target.shape)), ("CSI_nearest_reference", target[train][distance.argmin(1)])):
            result = {"feature": feature, "kernel": name}
            for label, mask in (("train", train), ("validation", validation), ("test", test)):
                result[label] = base.score(prediction[mask], target[mask], names[mask])
            results.append(result)
        for kernel, matrix in matrices.items():
            coefficients, candidates, best = [], [], None
            for regularization, dual, mean, residual in base.dual_solutions(matrix, target, train):
                raw_prediction = (mean + base.multiply(matrix, dual)).reshape(-1, 17, 2)
                prediction = raw_prediction.clip(0, 1)
                metric = base.score(prediction[validation], target[validation], names[validation])
                coefficients.append(dual)
                candidates.append({"lambda": regularization, "validation_nje": metric["metrics"]["normalized_joint_error"], "normal_equation_relative_residual": residual})
                if best is None or metric["metrics"]["normalized_joint_error"] < best[0]:
                    best = metric["metrics"]["normalized_joint_error"], regularization, dual, mean, raw_prediction, prediction
            _, selected, dual, mean, raw_prediction, prediction = best
            shuffled = prediction[test][base.permutations(names[test])]
            lag_input = (base.lag_permute(x[test]) - normalization["input_mean"]) / normalization["input_scale"]
            lag_matrix = base.kernel_predict_input(lag_input, transformed[train], kernel, width)
            lag_prediction = (mean + base.multiply(lag_matrix, dual)).reshape(-1, 17, 2).clip(0, 1)
            name = feature + "-" + kernel
            np.savez_compressed(output / (name + "-weights.npz"), **normalization, training_design=transformed[train], dual=np.stack(coefficients), lambdas=np.asarray(CONFIG["ridge_lambdas"]), target_mean=mean, bandwidth_squared=np.asarray(width), selected_lambda=np.asarray(selected))
            np.savez_compressed(output / (name + "-predictions.npz"), prediction=prediction, raw_prediction=raw_prediction, prediction_shuffle=shuffled, past_lag_permutation=lag_prediction)
            result = {"feature": feature, "kernel": kernel, "regularization": selected, "candidates": candidates}
            for label, mask in (("train", train), ("validation", validation), ("test", test)):
                result[label] = base.score(prediction[mask], target[mask], names[mask])
            result["test_controls"] = {"prediction_shuffle": base.score(shuffled, target[test], names[test]), "past_lag_permutation": base.score(lag_prediction, target[test], names[test])}
            results.append(result)
            print(json.dumps({"feature": feature, "kernel": kernel, "lambda": selected, "test_nje": result["test"]["metrics"]["normalized_joint_error"], "test_relative_R2": result["test"]["motion"]["relative"]["within_record_r2"]}), flush=True)
    for name, expected in sources.items():
        require(base.digest(ROOT / name) == expected, "Source changed during experiment")
    base.write(output / "results.json", {"registration": registration, "results": results, "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--predecessor", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.cache, args.predecessor, args.out)
