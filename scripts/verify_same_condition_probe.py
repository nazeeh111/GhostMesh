"""Read back the frozen same-condition experiment from actual cache bytes."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import domain_same_condition_probe as probe  # noqa:E402


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run(cache, output):
    destination = output / "verification.json"
    require(not destination.exists(), "Verification already exists")
    results = json.loads((output / "results.json").read_text())
    registration = results["registration"]
    require(probe.CONFIG == registration["config"], "Current config differs")
    for name, expected in registration["sources_sha256"].items():
        require(probe.digest(ROOT / name) == expected, "Current source changed")
        require(probe.digest(output / "executed-source" / name) == expected, "Source snapshot differs")
    require(probe.digest(cache / "receipt.json") == registration["cache_receipt_sha256"], "Cache receipt changed")
    for name, expected in registration["cache_arrays_sha256"].items():
        require(probe.digest(cache / (name + ".npy")) == expected, "Actual cache array changed")
    rf = np.load(cache / "rf.npy", mmap_mode="r", allow_pickle=False)
    pose = np.load(cache / "pose.npy", mmap_mode="r", allow_pickle=False)
    receipt = json.loads((cache / "receipt.json").read_text())
    bindings = json.loads((output / "selected-record-bindings.json").read_text())
    records, duplicate_checks = {}, {}
    for record, value in bindings.items():
        require(record in probe.CONFIG["records"] and record.startswith("E01/S01/"), "Unexpected decoded record")
        row = next(r for r in receipt["records"] if r["id"] == record)
        offset = row["offset"]
        require(offset == value["offset"] and row["ordered_raw_sha256"] == value["ordered_raw_sha256"], "Raw record binding differs")
        source = np.asarray(rf[offset : offset + 297])
        y = np.asarray(pose[offset : offset + 297])
        require(sha256(source.tobytes()).hexdigest() == value["cached_rf_sha256"], "Selected RF differs")
        require(sha256(y.tobytes()).hexdigest() == value["cached_pose_sha256"], "Selected pose differs")
        hashes = [sha256(np.ascontiguousarray(frame).tobytes()).hexdigest() for frame in source]
        duplicate_checks[record] = {"raw_frames": 297, "unique_raw_RF_frame_hashes": len(set(hashes))}
        records[record] = {"offset": offset, "features": probe.record_features(source), "pose": y.astype("float64")}
    maximum, residual_max, models, coverage = 0.0, 0.0, 0, {}
    duplicate_cross_split = {}
    for protocol in ("held_time", "held_action"):
        for feature in ("raw_db", "quotient"):
            prefix = f"{protocol}-{feature}"
            with np.load(output / (prefix + "-indices.npz"), allow_pickle=False) as z:
                names, ends, support, split = (z[key] for key in ("records", "end", "raw_support", "split"))
            train, val, test = (split == key for key in ("train", "validation", "test"))
            require(not any(set(support[a].ravel()) & set(support[b].ravel()) for a, b in ((train, val), (train, test), (val, test))), "Raw supports overlap")
            target = np.asarray(pose[support[:, -1]], dtype="float64")
            examples = []
            for record, endpoint, indices in zip(names, ends, support):
                offset = records[record]["offset"]
                require(endpoint + offset == indices[-1], "Frame-label endpoint mapping differs")
                require(np.array_equal(indices, np.arange(indices[-1] - 7, indices[-1] + 1)), "Noncausal context")
                examples.append(records[record]["features"][feature][indices - offset].reshape(-1))
            x = np.asarray(examples)
            if feature == "raw_db":
                frame_hashes = {label: {sha256(np.ascontiguousarray(rf[index]).tobytes()).hexdigest() for index in set(support[mask].ravel())} for label, mask in (("train", train), ("validation", val), ("test", test))}
                duplicate_cross_split[protocol] = {"train_validation": len(frame_hashes["train"] & frame_hashes["validation"]), "train_test": len(frame_hashes["train"] & frame_hashes["test"]), "validation_test": len(frame_hashes["validation"] & frame_hashes["test"])}
                # Teacher labels are used only for this coverage audit, never prediction.
                pairwise = np.linalg.norm(target[test, None] - target[train][None], axis=-1).mean(-1)
                nearest = pairwise.argmin(1)
                oracle = target[train][nearest]
                coverage[protocol] = {
                    "scope": "Teacher-only nearest-reference coverage oracle; NOT an inference model or deployable score",
                    "oracle_nearest_reference": probe.score(oracle, target[test], names[test]),
                }
                relative = target - target[:, [11, 12]].mean(1, keepdims=True)
                coverage[protocol]["train_relative_RMS_about_train_mean_pixels"] = float(np.sqrt(np.mean((relative[train] - relative[train].mean(0)) ** 2)) * 640)
                coverage[protocol]["test_relative_RMS_about_test_mean_pixels"] = float(np.sqrt(np.mean((relative[test] - relative[test].mean(0)) ** 2)) * 640)
            for kernel in ("linear", "rbf"):
                row = next(r for r in results["results"] if (r["protocol"], r["feature"], r["kernel"]) == (protocol, feature, kernel))
                name = prefix + "-" + kernel
                with np.load(output / (name + "-weights.npz"), allow_pickle=False) as z:
                    state = {key: z[key] for key in z.files}
                with np.load(output / (name + "-predictions.npz"), allow_pickle=False) as z:
                    stored = {key: z[key] for key in z.files}
                normalized = (x - state["input_mean"]) / state["input_scale"]
                require(np.array_equal(state["input_mean"], x[train].mean(0)), "Normalization mean uses nontraining inputs")
                require(np.array_equal(state["input_scale"], np.maximum(x[train].std(0), 0.05)), "Normalization scale uses nontraining inputs")
                require(np.array_equal(normalized[train], state["training_design"]), "Stored training CSI differs from actual supports")
                require(np.array_equal(target[train], state["training_target"]), "Training labels differ")
                require(np.array_equal(target, stored["target"]), "Stored targets differ")
                require(np.array_equal(target[train].mean(0).reshape(34), state["target_mean"]), "Mean uses nontraining labels")
                matrix = probe.kernel_predict_input(normalized, state["training_design"], kernel, float(state["bandwidth_squared"]))
                if kernel == "linear":
                    source_norm = (normalized[train] ** 2).sum(1)
                    training_distance = np.maximum(source_norm[:, None] + source_norm[None] - 2 * matrix[train] * x.shape[1], 0)
                    expected_width = np.median(training_distance[np.triu_indices(train.sum(), 1)])
                    require(abs(float(expected_width - state["bandwidth_squared"])) < 1e-10, "Bandwidth uses nontraining inputs")
                selected = np.flatnonzero(state["lambdas"] == row["regularization"])
                require(len(selected) == 1, "Selected lambda absent")
                for regularization, dual in zip(state["lambdas"], state["dual"]):
                    residual = probe.multiply(matrix[train], dual) + regularization * dual - (target[train].reshape(train.sum(), 34) - state["target_mean"])
                    residual_max = max(residual_max, float(np.linalg.norm(residual) / np.linalg.norm(target[train].reshape(train.sum(), 34) - state["target_mean"])))
                raw_prediction = (state["target_mean"] + probe.multiply(matrix, state["dual"][selected[0]])).reshape(-1, 17, 2)
                prediction = raw_prediction.clip(0, 1)
                maximum = max(maximum, float(np.max(abs(raw_prediction - stored["raw_prediction"]))), float(np.max(abs(prediction - stored["prediction"]))))
                for label, mask in (("train", train), ("validation", val), ("test", test)):
                    current = probe.score(prediction[mask], target[mask], names[mask])
                    require(current["metrics"] == row[label]["metrics"], "Metrics do not reproduce")
                chosen = min(range(len(row["candidates"])), key=lambda i: row["candidates"][i]["validation_nje"])
                require(row["candidates"][chosen]["regularization"] == row["regularization"], "Validation choice/tie rule differs")
                require(np.array_equal(stored["prediction_shuffle"], stored["prediction"][test][stored["shuffle_indices"]]), "Prediction shuffle changed values")
                models += 1
    require(maximum < 1e-12 and residual_max < 1e-6 and models == 8, "Saved-model verification failed")
    require(not any(count for row in duplicate_cross_split.values() for count in row.values()), "Exact RF frames repeat across splits")
    report = {"models_verified": models, "maximum_saved_prediction_difference": maximum, "maximum_recomputed_normal_equation_relative_residual": residual_max, "source_and_actual_cache_hashes_match": True, "raw_supports_disjoint": True, "frame_and_label_join_verified": True, "duplicate_raw_RF_frames": duplicate_checks, "exact_RF_frame_duplicates_across_splits": duplicate_cross_split, "teacher_coverage_diagnostic": coverage, "verification_script_sha256": probe.digest(Path(__file__)), "artifact_sha256": {str(p.relative_to(output)): probe.digest(p) for p in sorted(output.rglob("*")) if p.is_file()}}
    probe.write(output / "verification.json", report)
    print(json.dumps({key: value for key, value in report.items() if key not in ("artifact_sha256", "teacher_coverage_diagnostic")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.cache, args.out)
