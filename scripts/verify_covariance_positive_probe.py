"""Reconstruct covariance-gate inputs and check every saved kernel solution."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import domain_covariance_positive_probe as study  # noqa: E402


def run(cache, output):
    require, base = study.require, study.base
    require(not (output / "verification.json").exists(), "Verification already exists")
    result = json.loads((output / "results.json").read_text())
    reg = result["registration"]
    for name, expected in reg["sources_sha256"].items():
        require(
            base.digest(ROOT / name)
            == expected
            == base.digest(output / "executed-source" / name),
            "Executed/live source mismatch",
        )
    require(
        base.digest(cache / "receipt.json") == reg["cache_receipt_sha256"],
        "Cache receipt changed",
    )
    receipt = json.loads((cache / "receipt.json").read_text())
    row = next(r for r in receipt["records"] if r["id"] == study.CONFIG["record"])
    offset = row["offset"]
    raw = np.load(cache / "rf.npy", mmap_mode="r", allow_pickle=False)[
        offset : offset + 297
    ].copy()
    pose = np.load(cache / "pose.npy", mmap_mode="r", allow_pickle=False)[
        offset : offset + 297
    ].copy()
    require(
        sha256(raw.tobytes()).hexdigest()
        == reg["fixed_record_binding"]["cached_rf_sha256"],
        "Selected actual RF bytes changed",
    )
    require(
        sha256(pose.tobytes()).hexdigest()
        == reg["fixed_record_binding"]["cached_pose_sha256"],
        "Selected actual pose bytes changed",
    )
    pose = pose.astype("float64")
    pieces = {name: [] for name in study.CONFIG["features"]}
    target, ends, support, splits, diagnostics = [], [], [], [], {}
    for split, (start, stop) in study.CONFIG["raw_blocks_half_open"].items():
        values, y, e, s, d = study.block_examples(raw, pose, start, stop)
        for feature in pieces:
            pieces[feature].append(values[feature])
        target.append(y)
        ends.extend(e)
        support.append(s + offset)
        splits.extend([split] * len(y))
        diagnostics[split] = d
    target, support, splits = (
        np.concatenate(target),
        np.concatenate(support),
        np.array(splits),
    )
    with np.load(output / "indices.npz", allow_pickle=False) as z:
        for key, value in (
            ("target", target),
            ("raw_support", support),
            ("split", splits),
            ("end", np.asarray(ends)),
        ):
            require(np.array_equal(z[key], value), "Saved index mismatch: " + key)
        names = z["records"]
    require(
        diagnostics
        == json.loads((output / "covariance-condition-diagnostics.json").read_text()),
        "Covariance conditioning diagnostics changed",
    )
    train, val, test = (splits == key for key in ("train", "validation", "test"))
    require(
        not any(
            set(support[a].ravel()) & set(support[b].ravel())
            for a, b in ((train, val), (train, test), (val, test))
        ),
        "Raw split histories overlap",
    )
    maximum = residual_maximum = solver_difference = 0.0
    models = solutions = 0
    for feature, blocks in pieces.items():
        x = np.concatenate(blocks)
        require(
            np.array_equal(
                x, np.load(output / (feature + "-features.npy"), allow_pickle=False)
            ),
            "Saved features differ from actual RF reconstruction",
        )
        for kernel in study.CONFIG["kernels"]:
            name = feature + "-" + kernel
            row = next(
                r
                for r in result["results"]
                if r["feature"] == feature and r["kernel"] == kernel
            )
            with np.load(output / (name + "-weights.npz"), allow_pickle=False) as z:
                state = {key: z[key] for key in z.files}
            with np.load(output / (name + "-predictions.npz"), allow_pickle=False) as z:
                saved = {key: z[key] for key in z.files}
            require(
                np.array_equal(state["input_mean"], x[train].mean(0))
                and np.array_equal(
                    state["input_scale"], np.maximum(x[train].std(0), 0.05)
                ),
                "Normalization is not training-only",
            )
            transformed = (x - state["input_mean"]) / state["input_scale"]
            require(
                np.array_equal(state["training_design"], transformed[train]),
                "Saved training design mismatch",
            )
            y = target[train].reshape(train.sum(), 34)
            require(
                np.array_equal(state["target_mean"], y.mean(0)),
                "Target mean is not training-only",
            )
            matrices, _, width = base.kernel_inputs(transformed, train)
            require(
                width == float(state["bandwidth_squared"]),
                "Bandwidth is not training-only",
            )
            matrix = matrices[kernel]
            validation_nje = []
            for regularization, dual in zip(state["lambdas"], state["dual"]):
                right = y - state["target_mean"]
                system = matrix[train] + regularization * np.eye(train.sum())
                residual = base.multiply(system, dual) - right
                residual_maximum = max(
                    residual_maximum,
                    float(np.linalg.norm(residual) / np.linalg.norm(right)),
                )
                independently_solved = np.linalg.solve(system, right)
                solver_difference = max(
                    solver_difference,
                    float(
                        np.linalg.norm(independently_solved - dual)
                        / max(np.linalg.norm(dual), 1e-12)
                    ),
                )
                p = (
                    (state["target_mean"] + base.multiply(matrix, dual))
                    .reshape(-1, 17, 2)
                    .clip(0, 1)
                )
                validation_nje.append(
                    base.score(p[val], target[val], names[val])["metrics"][
                        "normalized_joint_error"
                    ]
                )
                solutions += 1
            selected = int(np.argmin(validation_nje))
            require(
                float(state["lambdas"][selected]) == row["regularization"]
                and float(state["selected_lambda"]) == row["regularization"],
                "Validation-only lambda selection mismatch",
            )
            for observed, candidate in zip(validation_nje, row["candidates"]):
                require(
                    observed == candidate["validation_nje"],
                    "Saved validation candidate metric mismatch",
                )
            dual = state["dual"][selected]
            raw_prediction = (
                state["target_mean"] + base.multiply(matrix, dual)
            ).reshape(-1, 17, 2)
            prediction = raw_prediction.clip(0, 1)
            maximum = max(
                maximum,
                float(np.max(abs(raw_prediction - saved["raw_prediction"]))),
                float(np.max(abs(prediction - saved["prediction"]))),
            )
            for split, mask in (("train", train), ("validation", val), ("test", test)):
                require(
                    base.score(prediction[mask], target[mask], names[mask])
                    == row[split],
                    "Recomputed pose/motion metric mismatch",
                )
            shuffled = prediction[test][base.permutations(names[test])]
            require(
                np.array_equal(shuffled, saved["prediction_shuffle"]),
                "Shuffle mismatch",
            )
            lag = (base.lag_permute(x[test]) - state["input_mean"]) / state[
                "input_scale"
            ]
            lag_matrix = base.kernel_predict_input(
                lag, state["training_design"], kernel, width
            )
            lag_prediction = (
                (state["target_mean"] + base.multiply(lag_matrix, dual))
                .reshape(-1, 17, 2)
                .clip(0, 1)
            )
            maximum = max(
                maximum,
                float(np.max(abs(lag_prediction - saved["past_lag_permutation"]))),
            )
            for control, p in (
                ("prediction_shuffle", shuffled),
                ("past_lag_permutation", lag_prediction),
            ):
                require(
                    base.score(p, target[test], names[test])
                    == row["test_controls"][control],
                    "Control metric mismatch",
                )
            models += 1
    require(
        maximum == 0
        and residual_maximum < 1e-6
        and solver_difference < 1e-6
        and models == 8
        and solutions == 32,
        "Saved-fit verification failed",
    )
    report = dict(
        models_verified=models,
        independently_solved_systems=solutions,
        maximum_saved_prediction_difference=maximum,
        maximum_normal_equation_relative_residual=residual_maximum,
        maximum_independent_direct_solver_relative_difference=solver_difference,
        raw_RF_and_target_bindings_verified=True,
        forty_frame_histories_disjoint_between_splits=True,
        source_only_normalization_bandwidth_and_lambda_selection_verified=True,
        saved_metrics_and_controls_verified=True,
        verification_script_sha256=base.digest(Path(__file__)),
        artifacts_sha256={
            str(p.relative_to(output)): base.digest(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    )
    base.write(output / "verification.json", report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "artifacts_sha256"}, indent=2
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.cache, args.out)
