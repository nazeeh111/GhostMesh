"""Rebuild every source covariance input and replay all frozen ridge decoders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import domain_covariance_source_probe as study  # noqa: E402


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run(cache, out):
    require(not (out / "verification.json").exists(), "Verification already exists")
    result = json.loads((out / "results.json").read_text())
    reg = result["registration"]
    for name, expected in reg["source_sha256"].items():
        require(
            study.digest(ROOT / name)
            == expected
            == study.digest(out / "executed-source" / name),
            "Executed source mismatch",
        )
    receipt = json.loads((cache / "receipt.json").read_text())
    require(
        study.digest(cache / "receipt.json") == reg["cache_receipt_sha256"],
        "Source receipt mismatch",
    )
    for name, expected in reg["cache_arrays_sha256"].items():
        require(
            study.digest(cache / (name + ".npy")) == expected,
            "Actual source array hash mismatch",
        )
    for name, expected in json.loads(
        (out / "artifact_manifest.json").read_text()
    ).items():
        require(study.digest(out / name) == expected, "Saved artifact hash mismatch")
    with np.load(out / "targets-and-metadata.npz", allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    n = len(data["pose"])
    features = {
        arm: np.load(out / (arm + "-features.npy"), mmap_mode="r", allow_pickle=False)[
            :n
        ]
        for arm in study.CONFIG["arms"]
    }
    rf, pose = [
        np.load(cache / (name + ".npy"), mmap_mode="r", allow_pickle=False)
        for name in ("rf", "pose")
    ]
    quality = json.loads((out / "feature-quality.json").read_text())
    cursor = 0
    record_rows = [
        r
        for r in receipt["records"]
        if r["id"].split("/")[2] in reg["config"]["selected_actions"]
    ]
    for i, row in enumerate(record_rows):
        offset = row["offset"]
        actual, ends, condition, rejected = study.available_examples(
            rf[offset : offset + 297],
            feature_function=study.cached_shape_features
            if reg["config"]["cached_shapes"]
            else study.covariance_features,
        )
        count = len(ends)
        require(
            quality[i]
            == dict(
                record=row["id"],
                maximum_covariance_condition=condition,
                retained_endpoints=ends.tolist(),
                rejected_frames=rejected,
            ),
            "Signal rejection/conditioning report differs",
        )
        require(
            np.array_equal(
                data["record"][cursor : cursor + count], np.repeat(row["id"], count)
            )
            and np.array_equal(data["end"][cursor : cursor + count], ends),
            "Record/endpoint ordering differs",
        )
        y = np.asarray(pose[offset : offset + 297], dtype="float64")
        relative = y - y[:, [11, 12]].mean(1, keepdims=True)
        require(
            np.array_equal(data["pose"][cursor : cursor + count], relative[ends])
            and np.array_equal(
                data["delta"][cursor : cursor + count],
                relative[ends] - relative[ends - 8],
            ),
            "Endpoint target join differs",
        )
        for arm in features:
            require(
                np.array_equal(
                    features[arm][cursor : cursor + count],
                    actual[arm].astype("float32"),
                ),
                "Actual-RF-derived full features differ",
            )
        cursor += count
        if (i + 1) % 10 == 0:
            print(
                json.dumps(
                    dict(raw_records_reconstructed=i + 1, total=len(record_rows))
                ),
                flush=True,
            )
    require(cursor == n, "Unexpected retained example count")
    target, delta = data["pose"], data["delta"]
    y = np.concatenate((target.reshape(n, 34), delta.reshape(n, 34)), axis=1)
    maximum = design_maximum = residual_maximum = 0.0
    models = solutions = 0
    for held_room in study.CONFIG["folds"]:
        train, val, held = study.fold_masks(
            data["room"], data["person"], held_room, reg["config"]["counts_per_fold"]
        )
        groups, actions = data["record"][held], data["action"][held]
        perm = np.load(out / (held_room + "-permutation.npy"), allow_pickle=False)
        rng = np.random.default_rng(study.CONFIG["permutation_seed"])
        expected_perm = np.arange(held.sum())
        for key in sorted(set(groups)):
            ix = np.flatnonzero(groups == key)
            expected_perm[ix] = rng.permutation(ix)
        require(
            np.array_equal(perm, expected_perm), "Within-record permutation mismatch"
        )
        for arm in study.CONFIG["arms"]:
            tag = held_room + "-" + arm
            row = next(
                r
                for r in result["results"]
                if r["held_room"] == held_room and r["arm"] == arm
            )
            with np.load(out / (tag + "-rff.npz"), allow_pickle=False) as z:
                mapping = {key: z[key] for key in z.files}
            feature = features[arm]
            mean, scale = study.feature_stats(feature, train)
            require(
                np.array_equal(mean, mapping["feature_mean"])
                and np.array_equal(scale, mapping["feature_scale"]),
                "Feature normalization not source-training-only",
            )
            rng = np.random.default_rng(study.CONFIG["seed"])
            ix = rng.choice(np.flatnonzero(train), 128, replace=False)
            require(
                np.array_equal(ix, mapping["bandwidth_sample_indices"]),
                "Bandwidth samples differ",
            )
            small = (np.asarray(feature[ix], dtype="float64") - mean) / scale
            distances = np.maximum(
                (small * small).sum(1)[:, None]
                + (small * small).sum(1)[None]
                - 2 * study.multiply(small, small.T),
                0,
            )
            bandwidth = float(np.median(distances[np.triu_indices(128, 1)]))
            require(
                bandwidth == float(mapping["bandwidth_squared"]),
                "Bandwidth not source-only",
            )
            rng = np.random.default_rng(study.CONFIG["seed"])
            expected_weight = rng.normal(size=(feature.shape[1], 512)) / np.sqrt(
                bandwidth
            )
            expected_offset = rng.uniform(0, 2 * np.pi, 512)
            require(
                np.array_equal(expected_weight, mapping["weight"])
                and np.array_equal(expected_offset, mapping["offset"]),
                "Full-dimensional RFF seed/weights differ",
            )
            raw_z = np.empty((n, 512))
            for start in range(0, n, 128):
                x = (
                    np.asarray(feature[start : start + 128], dtype="float64") - mean
                ) / scale
                raw_z[start : start + len(x)] = np.sqrt(2 / 512) * np.cos(
                    study.multiply(x, mapping["weight"]) + mapping["offset"]
                )
            require(
                np.array_equal(raw_z[train].mean(0), mapping["rff_mean"])
                and np.array_equal(
                    np.maximum(raw_z[train].std(0), 0.001), mapping["rff_scale"]
                ),
                "RFF normalization not source-training-only",
            )
            z = (raw_z - mapping["rff_mean"]) / mapping["rff_scale"]
            design = np.load(
                out / (tag + "-design.npy"), mmap_mode="r", allow_pickle=False
            )
            design_maximum = max(design_maximum, float(np.max(abs(z - design))))
            with np.load(
                out / (tag + "-model-prediction.npz"), allow_pickle=False
            ) as stored:
                state = {key: stored[key] for key in stored.files}
            require(
                np.array_equal(state["held_indices"], np.flatnonzero(held)),
                "Query index differs",
            )
            require(
                np.array_equal(state["x_mean"], z[train].mean(0))
                and np.array_equal(state["y_mean"], y[train].mean(0)),
                "Ridge means not source-only",
            )
            x = z[train] - state["x_mean"]
            matrix = study.multiply(x.T, x)
            right = study.multiply(x.T, y[train] - state["y_mean"])
            validation_losses = []
            for alpha, weight in zip(state["alphas"], state["all_weights"]):
                residual = (
                    study.multiply(matrix + train.sum() * alpha * np.eye(512), weight)
                    - right
                )
                residual_maximum = max(
                    residual_maximum,
                    float(np.linalg.norm(residual) / np.linalg.norm(right)),
                )
                prediction = (
                    study.multiply(z[val] - state["x_mean"], weight) + state["y_mean"]
                )
                validation_losses.append(
                    [
                        float(
                            (
                                (
                                    prediction[:, j * 34 : (j + 1) * 34]
                                    - y[val, j * 34 : (j + 1) * 34]
                                )
                                ** 2
                            ).mean()
                        )
                        for j in range(2)
                    ]
                )
                solutions += 1
            for j, name in enumerate(("pose", "delta")):
                alpha_index = int(np.argmin(np.asarray(validation_losses)[:, j]))
                require(
                    float(state["alphas"][alpha_index]) == row[name + "_alpha"],
                    "Target-specific validation-only alpha differs",
                )
                require(
                    np.array_equal(
                        state[name + "_weights"],
                        state["all_weights"][alpha_index, :, j * 34 : (j + 1) * 34],
                    ),
                    "Selected saved weights differ",
                )
            predictions = [
                (
                    study.multiply(z[held] - state["x_mean"], state[name + "_weights"])
                    + state["y_mean"][j * 34 : (j + 1) * 34]
                ).reshape(-1, 17, 2)
                for j, name in enumerate(("pose", "delta"))
            ]
            for name, prediction in zip(("pose", "delta"), predictions):
                maximum = max(maximum, float(np.max(abs(prediction - state[name]))))
            metrics = study.prediction_metrics(
                *predictions, target[held], delta[held], groups, actions
            )
            shuffled = study.prediction_metrics(
                predictions[0][perm],
                predictions[1][perm],
                target[held],
                delta[held],
                groups,
                actions,
            )
            # BLAS block widths can change last-bit rounding; metrics use tolerance.
            require(
                abs(metrics["pose_motion"]["r2"] - row["metrics"]["pose_motion"]["r2"])
                < 1e-10
                and abs(metrics["delta_zero_r2"] - row["metrics"]["delta_zero_r2"])
                < 1e-10
                and abs(
                    shuffled["pose_motion"]["r2"] - row["shuffled"]["pose_motion"]["r2"]
                )
                < 1e-10,
                "Replay metrics/control differ",
            )
            models += 1
            print(json.dumps(dict(verified=tag)), flush=True)
    require(
        design_maximum == 0
        and maximum < 1e-9
        and residual_maximum < 1e-6
        and models == 9
        and solutions == 45,
        "Frozen decoder replay failed",
    )
    report = dict(
        raw_records_reconstructed=len(record_rows),
        retained_examples_verified=n,
        feature_arms_verified=3,
        models_verified=models,
        normal_equations_verified=solutions,
        maximum_RFF_design_difference=design_maximum,
        maximum_saved_prediction_difference=maximum,
        maximum_normal_equation_relative_residual=residual_maximum,
        source_statistics_bandwidth_alpha_selection_verified=True,
        input_and_artifact_hashes_verified=True,
        verification_script_sha256=study.digest(Path(__file__)),
    )
    study.save(out / "verification.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.cache, args.out)
