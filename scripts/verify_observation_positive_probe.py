"""Read back compressed-observation positive-control models and cache identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import domain_observation_positive_probe as study  # noqa:E402


def run(cache, frame_cache, output):
    require, base = study.require, study.base
    require(not (output / "verification.json").exists(), "Verification already exists")
    result = json.loads((output / "results.json").read_text())
    reg = result["registration"]
    for name, expected in reg["sources_sha256"].items():
        require(base.digest(ROOT / name) == expected == base.digest(output / "executed-source" / name), "Source differs")
    receipt = json.loads((cache / "receipt.json").read_text())
    require(base.digest(cache / "receipt.json") == reg["cache_receipt_sha256"], "Raw receipt changed")
    record = next(row for row in receipt["records"] if row["id"] == study.CONFIG["record"])
    offset = record["offset"]
    raw = np.load(cache / "rf.npy", mmap_mode="r", allow_pickle=False)[offset : offset + 297]
    features = study.observation_features(raw, magnitude_units="log_magnitude_db", phase_units="radians")
    require(np.array_equal(features, np.load(output / "frame.npy", allow_pickle=False)), "Saved frame features differ")
    compressed_receipt = json.loads((frame_cache / "receipt.json").read_text())
    require(compressed_receipt["base_receipt_sha256"] == reg["cache_receipt_sha256"], "Compressed cache base differs")
    require(compressed_receipt["source_sha256"]["ghostmesh/observation.py"] == reg["sources_sha256"]["ghostmesh/observation.py"], "Frame-cache feature module differs")
    require(base.digest(frame_cache / "frame.npy") == compressed_receipt["arrays"]["frame"], "Actual completed compressed cache changed")
    cached = np.load(frame_cache / "frame.npy", mmap_mode="r", allow_pickle=False)
    require(np.array_equal(features, cached[offset : offset + 297]), "Meta-input cache differs from direct positive-control features")
    with np.load(output / "indices.npz", allow_pickle=False) as z:
        support, names, split, target = (z[name] for name in ("raw_support", "records", "split", "target"))
    require(np.array_equal(target, np.load(cache / "pose.npy", mmap_mode="r", allow_pickle=False)[support[:, -1]].astype("float64")), "Target endpoint join differs")
    train, val, test = (split == key for key in ("train", "validation", "test"))
    maximum = residual_maximum = 0.0
    models = solutions = 0
    for feature in study.CONFIG["features"]:
        x = study.windows(features, support - offset, feature)
        for kernel in study.CONFIG["kernels"]:
            name = feature + "-" + kernel
            row = next(r for r in result["results"] if r["feature"] == feature and r["kernel"] == kernel)
            with np.load(output / (name + "-weights.npz"), allow_pickle=False) as z:
                state = {key: z[key] for key in z.files}
            with np.load(output / (name + "-predictions.npz"), allow_pickle=False) as z:
                saved = {key: z[key] for key in z.files}
            require(np.array_equal(state["input_mean"], x[train].mean(0)), "Normalization mean uses held data")
            require(np.array_equal(state["input_scale"], np.maximum(x[train].std(0), .05)), "Normalization SD uses held data")
            transformed = (x - state["input_mean"]) / state["input_scale"]
            require(np.array_equal(transformed[train], state["training_design"]), "Frozen design differs")
            require(np.array_equal(state["target_mean"], target[train].mean(0).reshape(34)), "Mean target uses held labels")
            matrices, _, width = base.kernel_inputs(transformed, train)
            require(width == float(state["bandwidth_squared"]), "Bandwidth differs from training-only calculation")
            matrix = matrices[kernel]
            index = np.flatnonzero(state["lambdas"] == row["regularization"])
            require(len(index) == 1, "Selected lambda absent")
            for regularization, dual in zip(state["lambdas"], state["dual"]):
                residual = base.multiply(matrix[train], dual) + regularization * dual - (target[train].reshape(train.sum(), 34) - state["target_mean"])
                residual_maximum = max(residual_maximum, float(np.linalg.norm(residual) / np.linalg.norm(target[train].reshape(train.sum(), 34) - state["target_mean"])))
                solutions += 1
            raw_prediction = (state["target_mean"] + base.multiply(matrix, state["dual"][index[0]])).reshape(-1, 17, 2)
            prediction = raw_prediction.clip(0, 1)
            maximum = max(maximum, float(np.max(abs(raw_prediction - saved["raw_prediction"]))), float(np.max(abs(prediction - saved["prediction"]))))
            for label, mask in (("train", train), ("validation", val), ("test", test)):
                require(base.score(prediction[mask], target[mask], names[mask])["metrics"] == row[label]["metrics"], "Metrics differ")
            first = min(range(len(row["candidates"])), key=lambda i: row["candidates"][i]["validation_nje"])
            require(row["candidates"][first]["lambda"] == row["regularization"], "Validation choice/tie differs")
            require(np.array_equal(saved["prediction_shuffle"], prediction[test][base.permutations(names[test])]), "Shuffle differs")
            lag = (base.lag_permute(x[test]) - state["input_mean"]) / state["input_scale"]
            lag_matrix = base.kernel_predict_input(lag, state["training_design"], kernel, width)
            lag_prediction = (state["target_mean"] + base.multiply(lag_matrix, state["dual"][index[0]])).reshape(-1, 17, 2).clip(0, 1)
            maximum = max(maximum, float(np.max(abs(lag_prediction - saved["past_lag_permutation"]))))
            models += 1
    require(maximum == 0 and residual_maximum < 1e-6 and models == 4 and solutions == 16, "Saved-model readback failed")
    report = {"models_verified": models, "dual_solutions_recomputed": solutions, "maximum_saved_prediction_difference": maximum, "maximum_normal_equation_relative_residual": residual_maximum, "fixed_A12_slice_of_completed_source30_observation_cache_matches_direct_features_bitexact": True, "completed_frame_cache_receipt_sha256": base.digest(frame_cache / "receipt.json"), "completed_frame_array_sha256": compressed_receipt["arrays"]["frame"], "training_only_normalization_and_bandwidth_checked": True, "source_and_endpoint_target_join_checked": True, "verification_script_sha256": base.digest(Path(__file__)), "artifacts_sha256": {str(p.relative_to(output)): base.digest(p) for p in sorted(output.rglob("*")) if p.is_file()}}
    base.write(output / "verification.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "artifacts_sha256"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--frame-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.cache, args.frame_cache, args.out)
