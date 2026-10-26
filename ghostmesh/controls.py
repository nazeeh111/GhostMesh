"""Frozen supplemental diagnostics, separate from model selection and training."""

from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .engine import (
    _collect,
    _coverage,
    _dataset,
    _runtime_environment,
    _source_fingerprints,
    load_checkpoint,
)
from .metrics import pose_metrics

FILES = (
    "ghostmesh/controls.py",
    "scripts/run_signal_controls.py",
    "tests/test_controls.py",
    "docs/SIGNAL_CONTROLS.md",
    "configs/signal-controls-v1.json",
    "scripts/report_results.py",
)
SCOPE = (
    "Supplemental descriptive diagnostics of sample-specific CSI-target "
    "correspondence beyond static room pose distributions. Oracle room metadata; "
    "person/action/recording cues may remain. No significance tests, causal motion, "
    "localization, unseen-action or cross-dataset claim."
)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def runtime():
    return {k: v for k, v in _runtime_environment().items() if k != "torch_threads"}


def fit_room_prior(training):
    """Fit joint means/isotropic coordinate variance from training endpoints only."""
    labels, masks, rooms = [], [], []
    for i in range(len(training)):
        row = training[i]
        labels.append(np.asarray(row["pose"], dtype=float))
        masks.append(np.asarray(row["valid"], dtype=bool))
        rooms.append(str(row["environment"]))
    require(bool(labels), "No training endpoints")
    y, valid, rooms = np.stack(labels), np.stack(masks), np.asarray(rooms)
    valid &= np.isfinite(y).all(-1)

    def estimate(indices, fallback=None):
        targets, mask = y[indices], valid[indices]
        counts = mask.sum(0)
        if fallback is None:
            require(bool((counts > 0).all()), "No global training label for a joint")
        mean = np.where(mask[..., None], targets, 0).sum(0) / np.maximum(
            counts[:, None], 1
        )
        residual = np.where(mask[..., None], targets - mean, 0)
        var = (residual**2).sum((0, 2)) / np.maximum(2 * counts, 1)
        if fallback is not None:
            mean[counts == 0] = np.asarray(fallback["mean"])[counts == 0]
            var[counts == 0] = np.exp(
                2 * np.asarray(fallback["log_sigma"])[counts == 0]
            )
        return {
            "mean": mean.tolist(),
            "log_sigma": (0.5 * np.log(var.clip(1e-6))).tolist(),
            "joint_counts": counts.tolist(),
            "windows": int(len(indices)),
        }

    global_fit = estimate(np.arange(len(y)))
    return {
        "global": global_fit,
        "rooms": {
            room: estimate(np.flatnonzero(rooms == room), global_fit)
            for room in sorted(set(rooms))
        },
        "fit": "training labels/masks only; train stride 8; no action conditioning; "
        "oracle environment metadata; unseen room/missing room joint uses global training fit",
    }


def prior_predictions(prior, metadata):
    fits = [
        prior["rooms"].get(str(r["environment"]), prior["global"]) for r in metadata
    ]
    return np.asarray([r["mean"] for r in fits]), np.asarray(
        [r["log_sigma"] for r in fits]
    )


def within_room_permutation(metadata, seed=90210):
    """Uniform permutation of complete accepted windows, independently per room."""
    rooms = np.asarray([str(r["environment"]) for r in metadata])
    rng, indices = np.random.default_rng(seed), np.arange(len(rooms), dtype="<i8")
    for room in sorted(set(rooms)):
        group = np.flatnonzero(rooms == room)
        indices[group] = rng.permutation(group)
    return indices


def permute_predictions(pose, log_sigma, indices):
    require(
        sorted(np.asarray(indices).tolist()) == list(range(len(pose))),
        "Not a bijection",
    )
    require(len(log_sigma) == len(pose), "Pose/uncertainty length mismatch")
    return pose[indices], log_sigma[indices]


def _registration(root, suite, config_path, manifests):
    config, registered = read(config_path), read(suite / "preregistration.json")
    require(
        config == read(root / "configs/signal-controls-v1.json"),
        "Control config differs",
    )
    require(
        config.get("permutation_seed") == 90210
        and config.get("variants")
        == [
            "cohort20-baseline",
            "cohort20-temporal",
            "cohort20-environment-baseline",
            "cohort20-environment-temporal",
        ],
        "Fixed seed/variant set changed",
    )
    require(
        registered.get("profile") == "cohort20" and len(registered["variants"]) == 12,
        "Complete twelve-arm cohort20 registration required",
    )
    require(
        registered["source_sha256"] == _source_fingerprints(), "Runtime source changed"
    )
    require(
        registered["runner_sha256"] == digest(root / "scripts/run_suite.py"),
        "Runner changed",
    )
    require(
        registered["runtime_environment"] == runtime(), "Runtime environment changed"
    )
    selected = {}
    for name in config["variants"]:
        row = next((r for r in registered["variants"] if r["name"] == name), None)
        require(row is not None, "Control variant missing from suite")
        cfg, protocol = row["config"], row["config"]["protocol"]
        require(
            cfg["seed"] == 2027
            and cfg["data"]["split_stride"] == {"train": 8, "validation": 8, "test": 1},
            "Unexpected registered settings",
        )
        require(
            read(suite / "configs" / f"{name}.json") == cfg,
            "Saved suite config changed",
        )
        require(
            digest(manifests[protocol]) == row["manifest_sha256"], "Manifest changed"
        )
        selected[name] = row
    return {
        "schema_version": 1,
        "status": "frozen_before_test",
        "scope": SCOPE,
        "config": config,
        "variants": selected,
        "files_sha256": {f: digest(root / f) for f in FILES},
        "config_sha256": digest(config_path),
        "suite_configs_sha256": {
            name: digest(suite / "configs" / f"{name}.json") for name in selected
        },
        "runtime_source_sha256": _source_fingerprints(),
        "runtime_environment": runtime(),
        "runner_sha256": registered["runner_sha256"],
        "suite_preregistration_sha256": digest(suite / "preregistration.json"),
        "manifests_sha256": {k: digest(v) for k, v in manifests.items()},
        "checkpoint_binding": "At execution only, validation-selected best.pt SHA256 from completed suite; no selection by these diagnostics",
    }


def freeze(root, suite, config_path, manifests, out):
    root, suite, out = Path(root), Path(suite), Path(out)
    require(
        not list(suite.glob("*/test.json")),
        "Late freeze refused: original test report exists",
    )
    require(
        not (suite / "suite_report.json").exists(),
        "Late freeze refused: evaluated suite exists",
    )
    require(not out.exists(), "Refusing to replace frozen control directory")
    plan = _registration(root, suite, config_path, manifests)
    out.mkdir(parents=True)
    from .engine import _json

    _json(out / "plan.json", plan)
    return plan


def evaluation_guard(root, suite, config_path, manifests, out):
    plan = read(Path(out) / "plan.json")
    require(
        plan == _registration(Path(root), Path(suite), config_path, manifests),
        "Frozen plan/code/config/runtime/manifest changed",
    )
    registration = read(Path(suite) / "preregistration.json")
    for row in registration["variants"]:
        report_path = Path(suite) / row["name"] / "training_report.json"
        require(report_path.is_file(), "All twelve training jobs must complete first")
        report = read(report_path)
        require(
            report.get("status") == "trained"
            and report.get("epochs") == 40
            and report.get("config") == row["config"],
            "All twelve training jobs must complete first",
        )
    process = read(Path(suite) / "process.json")
    require(
        process.get("status") == "completed" and process.get("stage") == "evaluated",
        "Main suite must be complete and evaluated first",
    )
    spec = importlib.util.spec_from_file_location(
        "ghostmesh_control_reporter", Path(root) / "scripts/report_results.py"
    )
    reporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reporter)
    validated = reporter.load_suite(
        suite, Path(root) / "docs/cohort-provenance.json", root=root
    )
    require(
        runtime() == plan["runtime_environment"], "Control execution runtime changed"
    )
    return plan, validated


def verify_alignment(metric, reference, windows):
    for key in ("pck", "normalized_joint_error"):
        require(
            metric[key] is not None
            and reference[key] is not None
            and abs(metric[key] - reference[key]) <= 1e-6,
            f"Aligned {key} differs from main suite",
        )
    for key in ("valid_joints", "valid_frames", "excluded_frames"):
        require(metric[key] == reference[key], f"Aligned {key} differs from main suite")
    require(
        windows == reference["windows"], "Aligned window count differs from main suite"
    )


def stable_coverage(coverage):
    # First evaluation builds the cache; replay reuses it. This flag is not coverage.
    result = {**coverage}
    if "storage" in result:
        result["storage"] = {
            k: v for k, v in result["storage"].items() if k != "reused"
        }
    return result


def compact(metric):
    return {k: v for k, v in metric.items() if k != "per_frame_nje"}


@torch.no_grad()
def evaluate_controls(root, suite, config_path, manifests, out):
    root, suite, out = Path(root), Path(suite), Path(out)
    require(
        not any(
            (out / f).exists() for f in ("results.json", "results.csv", "README.md")
        ),
        "Refusing to replace diagnostic results",
    )
    plan, validated = evaluation_guard(root, suite, config_path, manifests, out)
    results, protocols, checkpoints = {}, {}, {}
    for name, row in plan["variants"].items():
        config, protocol = row["config"], row["config"]["protocol"]
        checkpoint_path = suite / name / "best.pt"
        checkpoints[name] = digest(checkpoint_path)
        model, checkpoint = load_checkpoint(checkpoint_path, "cpu")
        require(
            checkpoint["config"] == config
            and checkpoint["selected_by"] == "validation_normalized_joint_error",
            "Checkpoint selection/config changed",
        )
        torch.set_num_threads(int(config["threads"]))
        # Fit before constructing held-out data. Reuse once per protocol.
        if protocol not in protocols:
            training = _dataset(manifests[protocol], "train", config)
            prior = fit_room_prior(training)
            protocols[protocol] = {
                "prior": prior,
                "training_coverage": _coverage(training),
            }
        dataset = _dataset(manifests[protocol], "test", config)
        loader = DataLoader(
            dataset, batch_size=config["training"]["batch_size"], shuffle=False
        )
        threshold = config["evaluation"]["pck_threshold"]
        aligned, arrays, metadata = _collect(
            model, loader, torch.device("cpu"), threshold
        )
        reference = read(suite / name / "test.json")
        verify_alignment(aligned, reference, len(dataset))
        require(
            stable_coverage(_coverage(dataset, aligned))
            == stable_coverage(reference["coverage"]),
            "Aligned endpoint coverage changed",
        )
        pose, target, valid, sigma = arrays
        meta_sha = sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        state = protocols[protocol]
        indices = within_room_permutation(metadata, plan["config"]["permutation_seed"])
        index_sha = sha256(indices.astype("<i8").tobytes()).hexdigest()
        if "metadata_sha256" in state:
            require(
                state["metadata_sha256"] == meta_sha
                and state["permutation_sha256"] == index_sha,
                "Baseline/temporal endpoint order or permutation differs",
            )
        else:
            mean, uncertainty = prior_predictions(state["prior"], metadata)
            state.update(
                {
                    "metadata_sha256": meta_sha,
                    "permutation_sha256": index_sha,
                    "permutation_fixed_points": int(
                        (indices == np.arange(len(indices))).sum()
                    ),
                    "windows": len(dataset),
                    "coverage": _coverage(dataset, aligned),
                    "unseen_rooms_using_global_training_prior": sorted(
                        {r["environment"] for r in metadata}
                        - set(state["prior"]["rooms"])
                    ),
                    "room_prior_metrics": compact(
                        pose_metrics(mean, target, valid, uncertainty, threshold)
                    ),
                }
            )
        shuffled_pose, shuffled_sigma = permute_predictions(pose, sigma, indices)
        results[name] = {
            "protocol": protocol,
            "aligned": compact(aligned),
            "permuted": compact(
                pose_metrics(shuffled_pose, target, valid, shuffled_sigma, threshold)
            ),
            "checkpoint_sha256": checkpoints[name],
            "test_report_sha256": digest(suite / name / "test.json"),
        }
    require(
        read(out / "plan.json") == _registration(root, suite, config_path, manifests),
        "Frozen inputs changed during evaluation",
    )
    require(
        all(
            digest(suite / name / "best.pt") == value
            for name, value in checkpoints.items()
        ),
        "Checkpoint changed during evaluation",
    )
    return {
        "schema_version": 1,
        "status": "evaluated",
        "purpose": "dataset_experiment",
        "scope": SCOPE,
        "plan_sha256": digest(out / "plan.json"),
        "plan": plan,
        "checkpoint_sha256": checkpoints,
        "main_suite_inputs_sha256": validated["inputs_sha256"],
        "protocols": protocols,
        "models": results,
        "permutation": "One fixed seed 90210 per protocol; identical indices for baseline/temporal; "
        "pose/log_sigma move together; target/mask/endpoint order unchanged. "
        "Equivalent to permuting complete input windows for window-independent eval mode; "
        "each causal window retains its internal order. Fixed points retained.",
    }
