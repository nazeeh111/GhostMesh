"""Fixed A12 feasibility gate for covariance congruence dynamics; no E04."""

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
from ghostmesh.covariance_dynamics import (  # noqa: E402
    CONDITION_LIMIT,
    LAGS,
    PAIRED_SHRINKAGE,
    covariance_features,
    receiver_covariances,
)
from scripts import domain_same_condition_probe as base  # noqa:E402

CONFIG = {
    "record": "E01/S01/A12",
    "raw_blocks_half_open": {
        "train": [0, 118],
        "validation": [126, 178],
        "test": [186, 297],
    },
    "feature_context": 8,
    "covariance_lags": list(LAGS),
    "minimum_raw_context": 40,
    "endpoints": {"train": [39, 117], "validation": [165, 177], "test": [225, 296]},
    "counts": {"train": 79, "validation": 13, "test": 72},
    "paired_shrinkage": PAIRED_SHRINKAGE,
    "condition_limit": CONDITION_LIMIT,
    "features": [
        "raw_db_reference",
        "untreated_covariance",
        "log_shape_tangent",
        "congruence_dynamics",
    ],
    "covariance_control_history": "Both covariance controls retain current+lag1/2/4/8/16/32 states before the same8framecontext; same rawtemporalcoverage as invariant arm",
    "raw_db_reference": "Original fullpacket8frameDB reference, restricted to same endpoints; does not use the extra32frames and is not the matched-history invariance ablation",
    "kernels": ["linear", "rbf"],
    "ridge_lambdas": [1e-5, 0.001, 0.1, 10.0],
    "normalization": "Training-only columnmean/SDfloor.05; training-only medianRBFdistance; centered34target kernelRidge",
    "selection": "Per-arm validation NJE; firstlambda ties. Groupkernel also chosen by validation NJE, never test",
    "gate": "Selected groupkernel testrelative motionR2>=.5, alignedminusshuffledR2>=.3, testNJE<=.75training-reference-meanNJE",
    "scope": "Single fixed source-only sameperson/room/action feasibility gate, not roomgeneralization or finaltest",
    "physics_assumption": "Same invertible complex receiver mixing per six-subcarrier bin across each paired time; a proposal that actual rooms may violate",
    "limits": "Pure constant covariances givezero dynamics for every staticpose; cannot identify unknown initialpose; numerical condition acceptance is not invariant under arbitrary ill-conditionedG",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def block_examples(raw, pose, start, stop):
    covariance = receiver_covariances(
        raw[start:stop], magnitude_units="log_magnitude_db", phase_units="radians"
    )
    values = covariance_features(covariance)
    ends = np.arange(start + 39, stop)
    context = (ends - start - max(LAGS))[:, None] - np.arange(7, -1, -1)[None]
    result = {
        name: value[context].reshape(len(ends), -1) for name, value in values.items()
    }
    magnitude = raw[start:stop, :3].astype("float64")
    median = np.nanmedian(
        np.where(np.isfinite(magnitude), magnitude, np.nan), axis=(2, 3), keepdims=True
    )
    require(np.isfinite(median).all(), "Missing magnitude link")
    magnitude = np.where(np.isfinite(magnitude), magnitude, median) / 20
    result["raw_db_reference"] = magnitude[
        (ends - start)[:, None] - np.arange(7, -1, -1)[None]
    ].reshape(len(ends), -1)
    support = ends[:, None] - np.arange(39, -1, -1)[None]
    eigenvalues = np.linalg.eigvalsh(covariance)
    diagnostics = {
        "maximum_covariance_condition": float(
            np.max(eigenvalues[..., -1] / eigenvalues[..., 0])
        ),
        "minimum_covariance_eigenvalue": float(np.min(eigenvalues)),
        "matrices": int(np.prod(covariance.shape[:-2])),
    }
    return result, pose[ends], ends, support, diagnostics


def run(cache, predecessor, output):
    require(not output.exists(), "Refusing to overwrite covariance experiment")
    receipt = json.loads((cache / "receipt.json").read_text())
    old = json.loads((predecessor / "registration.json").read_text())
    require(
        base.digest(cache / "receipt.json") == old["cache_receipt_sha256"],
        "Source cache receipt changed",
    )
    row = next(r for r in receipt["records"] if r["id"] == CONFIG["record"])
    offset = row["offset"]
    raw = np.load(cache / "rf.npy", mmap_mode="r", allow_pickle=False)[
        offset : offset + 297
    ].copy()
    pose = np.load(cache / "pose.npy", mmap_mode="r", allow_pickle=False)[
        offset : offset + 297
    ].astype("float64")
    binding = json.loads((predecessor / "selected-record-bindings.json").read_text())[
        CONFIG["record"]
    ]
    require(
        sha256(raw.tobytes()).hexdigest() == binding["cached_rf_sha256"],
        "Actual selected RF changed",
    )
    require(
        sha256(pose.astype("float32").tobytes()).hexdigest()
        == binding["cached_pose_sha256"],
        "Actual selected pose changed",
    )
    source_paths = [
        Path(__file__),
        ROOT / "ghostmesh/covariance_dynamics.py",
        ROOT / "scripts/domain_same_condition_probe.py",
        ROOT / "scripts/domain_background_probe.py",
        ROOT / "scripts/domain_structured_probe.py",
        ROOT / "ghostmesh/metrics.py",
    ]
    sources = {str(path.relative_to(ROOT)): base.digest(path) for path in source_paths}
    registration = {
        "config": CONFIG,
        "sources_sha256": sources,
        "cache_receipt_sha256": base.digest(cache / "receipt.json"),
        "fixed_record_binding": binding,
        "predecessor_registration_sha256": base.digest(
            predecessor / "registration.json"
        ),
        "predecessor_verification_sha256": base.digest(
            predecessor / "verification.json"
        ),
        "numpy": np.__version__,
    }
    output.mkdir(parents=True)
    started = time.monotonic()
    base.write(output / "registration.json", registration)
    for path in source_paths:
        destination = output / "executed-source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    data, labels, ends, supports, splits, diagnostics = (
        {name: [] for name in CONFIG["features"]},
        [],
        [],
        [],
        [],
        {},
    )
    for split, (start, stop) in CONFIG["raw_blocks_half_open"].items():
        values, y, endpoint, support, diagnostic = block_examples(
            raw, pose, start, stop
        )
        require(
            len(endpoint) == CONFIG["counts"][split],
            "Expected matched warmup-excluded count",
        )
        for name in data:
            data[name].append(values[name])
        labels.append(y)
        ends.extend(endpoint)
        supports.append(support + offset)
        splits.extend([split] * len(y))
        diagnostics[split] = diagnostic
    target, support, splits = (
        np.concatenate(labels),
        np.concatenate(supports),
        np.asarray(splits),
    )
    names = np.repeat(CONFIG["record"], len(target))
    train, val, test = (splits == key for key in ("train", "validation", "test"))
    require(
        not any(
            set(support[a].ravel()) & set(support[b].ravel())
            for a, b in ((train, val), (train, test), (val, test))
        ),
        "Forty-frame raw histories overlap splits",
    )
    np.savez_compressed(
        output / "indices.npz",
        raw_support=support,
        split=splits,
        end=np.asarray(ends),
        target=target,
        records=names,
    )
    base.write(output / "covariance-condition-diagnostics.json", diagnostics)
    results = []
    mean_prediction = np.broadcast_to(target[train].mean(0), target.shape)
    static = {"feature": "training_reference_mean", "kernel": "static"}
    for label, mask in (("train", train), ("validation", val), ("test", test)):
        static[label] = base.score(mean_prediction[mask], target[mask], names[mask])
    results.append(static)
    for feature, pieces in data.items():
        x = np.concatenate(pieces)
        np.save(output / (feature + "-features.npy"), x, allow_pickle=False)
        transformed, normalization = base.normalize_fit(x, train)
        kernels, _, width = base.kernel_inputs(transformed, train)
        for kernel, matrix in kernels.items():
            best, candidates, coefficients = None, [], []
            for regularization, dual, mean, residual in base.dual_solutions(
                matrix, target, train
            ):
                raw_prediction = (mean + base.multiply(matrix, dual)).reshape(-1, 17, 2)
                prediction = raw_prediction.clip(0, 1)
                validation = base.score(prediction[val], target[val], names[val])
                candidates.append(
                    {
                        "lambda": regularization,
                        "validation_nje": validation["metrics"][
                            "normalized_joint_error"
                        ],
                        "normal_equation_relative_residual": residual,
                    }
                )
                coefficients.append(dual)
                if (
                    best is None
                    or validation["metrics"]["normalized_joint_error"] < best[0]
                ):
                    best = (
                        validation["metrics"]["normalized_joint_error"],
                        regularization,
                        dual,
                        mean,
                        raw_prediction,
                        prediction,
                    )
            _, selected, dual, mean, raw_prediction, prediction = best
            shuffled = prediction[test][base.permutations(names[test])]
            lag_x = (
                base.lag_permute(x[test]) - normalization["input_mean"]
            ) / normalization["input_scale"]
            lag_kernel = base.kernel_predict_input(
                lag_x, transformed[train], kernel, width
            )
            lag_prediction = (
                (mean + base.multiply(lag_kernel, dual)).reshape(-1, 17, 2).clip(0, 1)
            )
            name = feature + "-" + kernel
            np.savez_compressed(
                output / (name + "-weights.npz"),
                **normalization,
                training_design=transformed[train],
                dual=np.stack(coefficients),
                lambdas=np.asarray(CONFIG["ridge_lambdas"]),
                target_mean=mean,
                bandwidth_squared=np.asarray(width),
                selected_lambda=np.asarray(selected),
            )
            np.savez_compressed(
                output / (name + "-predictions.npz"),
                prediction=prediction,
                raw_prediction=raw_prediction,
                prediction_shuffle=shuffled,
                past_lag_permutation=lag_prediction,
            )
            result = {
                "feature": feature,
                "kernel": kernel,
                "regularization": selected,
                "candidates": candidates,
            }
            for label, mask in (("train", train), ("validation", val), ("test", test)):
                result[label] = base.score(prediction[mask], target[mask], names[mask])
            result["test_controls"] = {
                "prediction_shuffle": base.score(shuffled, target[test], names[test]),
                "past_lag_permutation": base.score(
                    lag_prediction, target[test], names[test]
                ),
            }
            results.append(result)
            print(
                json.dumps(
                    {
                        "feature": feature,
                        "kernel": kernel,
                        "lambda": selected,
                        "test_nje": result["test"]["metrics"]["normalized_joint_error"],
                        "relative_R2": result["test"]["motion"]["relative"][
                            "within_record_r2"
                        ],
                    }
                ),
                flush=True,
            )
    groups = [r for r in results if r["feature"] == "congruence_dynamics"]
    selected = min(
        groups, key=lambda r: r["validation"]["metrics"]["normalized_joint_error"]
    )
    r2 = selected["test"]["motion"]["relative"]["within_record_r2"]
    shuffle_r2 = selected["test_controls"]["prediction_shuffle"]["motion"]["relative"][
        "within_record_r2"
    ]
    ratio = (
        selected["test"]["metrics"]["normalized_joint_error"]
        / static["test"]["metrics"]["normalized_joint_error"]
    )
    gate = {
        "kernel_chosen_on_validation": selected["kernel"],
        "relative_R2": r2,
        "aligned_minus_shuffled_R2": r2 - shuffle_r2,
        "NJE_fraction_of_static": ratio,
        "passed": bool(r2 >= 0.5 and r2 - shuffle_r2 >= 0.3 and ratio <= 0.75),
    }
    for name, expected in sources.items():
        require(base.digest(ROOT / name) == expected, "Source changed during probe")
    base.write(
        output / "results.json",
        {
            "registration": registration,
            "results": results,
            "continuation_gate": gate,
            "elapsed_seconds": time.monotonic() - started,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--predecessor", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.cache, args.predecessor, args.out)
