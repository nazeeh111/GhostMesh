#!/usr/bin/env python3
"""Freeze a bounded experiment suite, train on source/validation, then test once.

Profiles explicitly distinguish the two-action pilot and 27-action cohort.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.engine import compare_reports, evaluate, evaluate_mean_pose, train  # noqa: E402
from ghostmesh.engine import _source_fingerprints  # noqa: E402
from ghostmesh.engine import _runtime_environment  # noqa: E402
from ghostmesh.engine import _json as atomic_json  # noqa: E402


SUBJECT_VARIANTS = [
    "subset-baseline",
    "subset-temporal",
    "subset-no_consistency",
    "subset-no_normalization",
    "subset-no_augmentation",
    "subset-baseline-seed2028",
    "subset-temporal-seed2028",
]
ENVIRONMENT_VARIANTS = ["subset-environment-baseline", "subset-environment-temporal"]


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    atomic_json(Path(path), value)


def run_suite(
    subject_manifest,
    output,
    environment_manifest=None,
    device="cpu",
    epochs=None,
    only=None,
    resume=False,
    profile="subset",
):
    output = Path(output)
    subject_manifest = Path(subject_manifest).resolve()
    manifests = {"cross_subject": subject_manifest}
    if environment_manifest:
        manifests["cross_environment"] = Path(environment_manifest).resolve()
    if profile not in {"subset", "cohort20"}:
        raise ValueError("Unknown experiment profile")
    names = [
        name.replace("subset-", profile + "-", 1)
        for name in SUBJECT_VARIANTS
        + (ENVIRONMENT_VARIANTS if environment_manifest else [])
    ]
    if profile == "cohort20":
        names.insert(5, "cohort20-raw_consistency")
        if environment_manifest:
            names.extend(
                [
                    "cohort20-environment-baseline-seed2028",
                    "cohort20-environment-temporal-seed2028",
                ]
            )
    if only:
        requested = set(only)
        if requested - set(names):
            raise ValueError(f"Unknown/unavailable variants: {requested - set(names)}")
        names = [name for name in names if name in requested]
    if not names:
        raise ValueError("No experiment variants selected")
    variants = []
    for name in names:
        config = json.loads((ROOT / "configs" / f"{name}.json").read_text())
        if epochs is not None:
            if epochs < 1:
                raise ValueError("epochs must be positive")
            config["training"]["epochs"] = epochs
        variants.append(
            {
                "name": name,
                "config": config,
                "manifest": str(
                    Path(manifests[config["protocol"]].parent.name)
                    / manifests[config["protocol"]].name
                ),
                "manifest_sha256": digest(manifests[config["protocol"]]),
            }
        )
    registration = {
        "schema_version": 1,
        "purpose": "Balanced MM-Fi comparison across 27 actions and 20 subjects"
        if profile == "cohort20"
        else "bounded two-action MM-Fi subset comparison",
        "profile": profile,
        "source_sha256": _source_fingerprints(),
        "runner_sha256": digest(__file__),
        "runtime_environment": {
            key: value
            for key, value in _runtime_environment().items()
            if key != "torch_threads"
        },
        "baseline_description": "Last-frame encoder after the same eight-frame causal-window normalization; not a raw history-free control",
        "primary_hypothesis": "The combined temporal encoder and source-only teacher strategy reduces held-out error against the shared-preprocessing last-frame encoder",
        "primary_comparison": [f"{profile}-baseline", f"{profile}-temporal"],
        "primary_endpoint": "held-out normalized joint error; report PCK@0.2 and subject-cluster sensitivity",
        "device": device,
        "variants": variants,
        "rules": [
            "All architecture, seed and training settings are frozen before this suite's test evaluations.",
            "Checkpoints selected only by validation NJE; test metrics do not select epoch or model.",
            "Train mean pose uses only training labels; same test-window coverage as neural models.",
            "Bootstrap whole sequence blocks; include subject-cluster sensitivity.",
            "Do not infer full 40-subject efficacy or a 10x improvement from cohort/subset metrics.",
        ],
    }
    registration_path = output / "preregistration.json"
    if output.exists():
        if not resume:
            raise FileExistsError(
                f"Suite output exists: {output}; use --resume only for identical preregistration"
            )
        if (
            not registration_path.is_file()
            or json.loads(registration_path.read_text()) != registration
        ):
            raise ValueError(
                "Cannot resume: preregistration or manifest hashes changed"
            )
    else:
        output.mkdir(parents=True)
        save(registration_path, registration)
    configs = output / "configs"
    configs.mkdir(exist_ok=True)

    def assert_frozen(row):
        if (
            _source_fingerprints() != registration["source_sha256"]
            or digest(__file__) != registration["runner_sha256"]
        ):
            raise ValueError("Frozen runtime or runner source changed")
        manifest = manifests[row["config"]["protocol"]]
        if digest(manifest) != row["manifest_sha256"]:
            raise ValueError("Frozen manifest changed")
        config_path = configs / f"{row['name']}.json"
        if (
            config_path.exists()
            and json.loads(config_path.read_text()) != row["config"]
        ):
            raise ValueError("Frozen configuration changed")

    save(
        output / "process.json",
        {"pid": os.getpid(), "status": "running", "stage": "training"},
    )
    # Complete every training job before loading the first test window.
    for row in variants:
        name, manifest = row["name"], manifests[row["config"]["protocol"]]
        assert_frozen(row)
        config_path = configs / f"{name}.json"
        if (
            config_path.exists()
            and json.loads(config_path.read_text()) != row["config"]
        ):
            raise ValueError("Saved frozen config changed")
        if not config_path.exists():
            save(config_path, row["config"])
        destination = output / name
        training_report = destination / "training_report.json"
        if resume and training_report.is_file():
            record = json.loads(training_report.read_text())
            if (
                record.get("config") != row["config"]
                or record.get("config_source_sha256") != digest(config_path)
                or record.get("manifest_sha256") != row["manifest_sha256"]
                or record.get("checkpoint_sha256") != digest(destination / "best.pt")
                or record.get("source_sha256") != registration["source_sha256"]
                or record.get("device") != device
                or {
                    key: value
                    for key, value in record.get("runtime_environment", {}).items()
                    if key != "torch_threads"
                }
                != registration["runtime_environment"]
                or record.get("runtime_environment", {}).get("torch_threads")
                != row["config"].get("threads", 2)
            ):
                raise ValueError(
                    f"Cannot resume modified/incomplete training artifact {name}"
                )
            print(f"verified completed training: {name}", flush=True)
            continue
        print(f"training frozen variant: {name}", flush=True)
        train(
            manifest,
            config_path,
            destination,
            device=device,
            resume=resume and destination.exists(),
        )
    reports = {}
    for row in variants:
        assert_frozen(row)
        name = row["name"]
        result_path = output / name / "test.json"
        if resume and result_path.is_file():
            report = json.loads(result_path.read_text())
            if (
                report.get("manifest_sha256") != row["manifest_sha256"]
                or report.get("checkpoint_sha256") != digest(output / name / "best.pt")
                or report.get("source_sha256") != registration["source_sha256"]
                or report.get("split") != "test"
                or report.get("protocol") != row["config"]["protocol"]
                or {
                    key: value
                    for key, value in report.get("runtime_environment", {}).items()
                    if key != "torch_threads"
                }
                != registration["runtime_environment"]
                or report.get("latency", {}).get("device") != device
            ):
                raise ValueError(f"Cannot resume modified test artifacts {name}")
        else:
            report = evaluate(
                manifests[row["config"]["protocol"]],
                output / name / "best.pt",
                output=result_path,
                device=device,
            )
        reports[name] = report
    mean_reports = {}
    for protocol, manifest in manifests.items():
        if (
            _source_fingerprints() != registration["source_sha256"]
            or digest(__file__) != registration["runner_sha256"]
        ):
            raise ValueError(
                "Frozen runtime or runner source changed before mean-pose evaluation"
            )
        rows = [r for r in variants if r["config"]["protocol"] == protocol]
        if not rows:
            continue
        result_path = output / f"{protocol}-train-mean-pose.json"
        if resume and result_path.is_file():
            mean = json.loads(result_path.read_text())
            if (
                mean.get("manifest_sha256") != digest(manifest)
                or mean.get("split") != "test"
                or mean.get("protocol") != protocol
                or mean.get("config_sha256")
                != digest(configs / f"{rows[0]['name']}.json")
                or mean.get("source_sha256") != registration["source_sha256"]
                or {
                    key: value
                    for key, value in mean.get("runtime_environment", {}).items()
                    if key != "torch_threads"
                }
                != registration["runtime_environment"]
            ):
                raise ValueError("Cannot resume changed mean-pose baseline")
        else:
            mean = evaluate_mean_pose(
                manifest, configs / f"{rows[0]['name']}.json", output=result_path
            )
        mean_reports[protocol] = mean
    pairs = [
        ("subset-baseline", "subset-temporal"),
        ("subset-baseline", "subset-no_consistency"),
        ("subset-baseline-seed2028", "subset-temporal-seed2028"),
        ("subset-no_consistency", "subset-temporal"),
        ("subset-no_normalization", "subset-temporal"),
        ("subset-no_augmentation", "subset-temporal"),
        ("subset-environment-baseline", "subset-environment-temporal"),
    ]
    pairs = [
        (a.replace("subset-", profile + "-", 1), b.replace("subset-", profile + "-", 1))
        for a, b in pairs
    ]
    if profile == "cohort20":
        pairs.append(("cohort20-raw_consistency", "cohort20-temporal"))
        pairs.append(
            (
                "cohort20-environment-baseline-seed2028",
                "cohort20-environment-temporal-seed2028",
            )
        )
    comparisons = []
    for reference, candidate in pairs:
        if reference not in reports or candidate not in reports:
            continue
        comparisons.append(
            {
                "reference": reference,
                "candidate": candidate,
                "sequence_blocks": compare_reports(
                    reports[reference], reports[candidate]
                ),
                "subject_clusters": compare_reports(
                    reports[reference], reports[candidate], cluster="subject"
                ),
            }
        )
    for name, report in reports.items():
        if name in tuple(
            f"{profile}-{suffix}"
            for suffix in (
                "baseline",
                "temporal",
                "environment-baseline",
                "environment-temporal",
            )
        ):
            comparisons.append(
                {
                    "reference": "train-mean-pose-no-signal",
                    "candidate": name,
                    "sequence_blocks": compare_reports(
                        mean_reports[report["protocol"]], report
                    ),
                    "subject_clusters": compare_reports(
                        mean_reports[report["protocol"]], report, cluster="subject"
                    ),
                }
            )
    summary = {
        "preregistration_sha256": digest(registration_path),
        "comparisons": comparisons,
        "results": {
            name: {
                k: report[k]
                for k in (
                    "pck",
                    "normalized_joint_error",
                    "windows",
                    "purpose",
                    "protocol",
                    "latency",
                )
            }
            for name, report in reports.items()
        },
        "limits": "Subset results only. Two training seeds are a sensitivity check, not a stable training-variance confidence interval. Bootstrap does not establish causal mechanism or cross-dataset transfer.",
    }
    save(output / "suite_report.json", summary)
    save(
        output / "process.json",
        {"pid": os.getpid(), "status": "completed", "stage": "evaluated"},
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-manifest", required=True)
    parser.add_argument("--environment-manifest")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument(
        "--epochs",
        type=int,
        help="Freeze a different epoch budget before any suite test evaluation",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        help="Freeze a smaller named subset of configurations before testing",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only identical hashed preregistration and completed artifacts",
    )
    parser.add_argument("--profile", choices=["subset", "cohort20"], default="subset")
    args = parser.parse_args()
    result = run_suite(
        args.subject_manifest,
        args.out,
        args.environment_manifest,
        args.device,
        args.epochs,
        args.only,
        args.resume,
        args.profile,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
