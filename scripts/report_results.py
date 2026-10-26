#!/usr/bin/env python3
"""Render a completed, artifact-bound cohort20 dataset experiment without ML imports.

Usage: python scripts/report_results.py --suite experiments/cohort20-final-v1 \
    --out work/results-v1 --provenance docs/cohort-provenance.json
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("engine.py", "models.py", "data.py", "preprocess.py", "metrics.py")
PROTOCOLS = ("cross_subject", "cross_environment")
THRESHOLDS = ("0.05", "0.1", "0.2")
METHODS = {
    "baseline": "Last-frame encoder",
    "temporal": "Temporal + teacher",
    "no_consistency": "Temporal, no teacher consistency",
    "no_normalization": "Temporal, no normalization",
    "no_augmentation": "Temporal, no augmentation",
    "raw_consistency": "Temporal + raw consistency",
}
COLORS = {
    "Static training mean pose": "#7b8088",
    "Last-frame encoder": "#276a97",
    "Temporal + teacher": "#b75e29",
}


def digest(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    def reject(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    return json.loads(Path(path).read_text(), parse_constant=reject)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name, low=0, high=None):
    require(type(value) in (int, float) and math.isfinite(value), f"Invalid {name}")
    require(value >= low and (high is None or value <= high), f"Out-of-range {name}")
    return value


def expected_variants():
    return {
        f"cohort20-{prefix}{method}{suffix}"
        for prefix in ("", "environment-")
        for method in ("baseline", "temporal")
        for suffix in ("", "-seed2028")
    } | {
        f"cohort20-{method}"
        for method in (
            "no_consistency",
            "no_normalization",
            "no_augmentation",
            "raw_consistency",
        )
    }


def expected_pairs():
    pairs = []
    for prefix in ("", "environment-"):
        for suffix in ("", "-seed2028"):
            pairs.append(
                (
                    f"cohort20-{prefix}baseline{suffix}",
                    f"cohort20-{prefix}temporal{suffix}",
                )
            )
    pairs += [
        (f"cohort20-{method}", "cohort20-temporal")
        for method in (
            "no_consistency",
            "no_normalization",
            "no_augmentation",
            "raw_consistency",
        )
    ]
    pairs += [("cohort20-baseline", "cohort20-no_consistency")]
    pairs += [
        ("train-mean-pose-no-signal", f"cohort20-{prefix}{method}")
        for prefix in ("", "environment-")
        for method in ("baseline", "temporal")
    ]
    return set(pairs)


def runtime_without_threads(record):
    return {
        k: v
        for k, v in record.get("runtime_environment", {}).items()
        if k != "torch_threads"
    }


def validate_metrics(report, context):
    require(
        report.get("purpose") == "dataset_experiment",
        f"{context}: dataset_experiment required",
    )
    require(
        report.get("status") == "evaluated" and report.get("split") == "test",
        f"{context}: evaluated test report required",
    )
    require(
        report.get("pck_threshold") == 0.2, f"{context}: expected PCK threshold 0.2"
    )
    for threshold in THRESHOLDS:
        number(
            report.get("pck_curve", {}).get(threshold),
            f"{context} PCK@{threshold}",
            high=1,
        )
    require(
        report["pck"] == report["pck_curve"]["0.2"], f"{context}: PCK summary mismatch"
    )
    number(report.get("normalized_joint_error"), f"{context} NJE")
    counts = [
        report.get(k)
        for k in ("windows", "valid_frames", "excluded_frames", "valid_joints")
    ]
    require(
        all(type(v) is int and v >= 0 for v in counts)
        and counts[0] > 0
        and counts[3] > 0,
        f"{context}: invalid or empty metric counts",
    )
    require(
        counts[1] + counts[2] == counts[0],
        f"{context}: metric frame accounting mismatch",
    )
    coverage = report.get("coverage", {})
    require(
        coverage.get("accepted_windows") == counts[0],
        f"{context}: accepted coverage mismatch",
    )
    candidate = coverage.get("candidate_windows")
    require(
        type(candidate) is int and candidate >= counts[0],
        f"{context}: candidate coverage required",
    )
    require(
        coverage.get("rejected_windows") == candidate - counts[0],
        f"{context}: rejected coverage mismatch",
    )
    require(
        coverage.get("metric_eligible_windows") == counts[1]
        and coverage.get("metric_excluded_windows") == counts[2],
        f"{context}: eligible coverage mismatch",
    )
    calibration = report.get("calibration", {})
    number(calibration.get("ece"), f"{context} calibration ECE", high=1)
    require(
        calibration.get("radius") == 0.05
        and calibration.get("event")
        == "joint Euclidean error <= radius in normalized image coordinates",
        f"{context}: unknown calibration event",
    )
    require(
        sum(row["count"] for row in calibration.get("bins", [])) == counts[3],
        f"{context}: calibration count mismatch",
    )
    return report


def load_suite(suite, provenance_path, root=None):
    """Read/hash source and JSON artifacts only; never unpickle or evaluate a model."""
    root = Path(root or ROOT).resolve()
    suite, provenance_path = Path(suite).resolve(), Path(provenance_path).resolve()
    inputs = {}

    def record(path):
        path = Path(path).resolve()
        require(path.is_file(), f"Missing required artifact: {path}")
        inputs[str(path)] = digest(path)
        return read(path)

    process = record(suite / "process.json")
    require(
        process.get("status") == "completed" and process.get("stage") == "evaluated",
        "Suite is not complete",
    )
    registration = record(suite / "preregistration.json")
    summary = record(suite / "suite_report.json")
    provenance = record(provenance_path)
    require(
        registration.get("profile") == "cohort20",
        "Only the declared cohort20 profile is supported",
    )
    require(
        summary.get("preregistration_sha256")
        == inputs[str(suite / "preregistration.json")],
        "Preregistration hash mismatch",
    )
    expected_sources = {name: digest(root / "ghostmesh" / name) for name in SOURCES}
    require(
        registration.get("source_sha256") == expected_sources,
        "Frozen runtime source hash mismatch",
    )
    require(
        registration.get("runner_sha256") == digest(root / "scripts" / "run_suite.py"),
        "Frozen runner source hash mismatch",
    )
    for name in SOURCES:
        inputs[str(root / "ghostmesh" / name)] = expected_sources[name]
    inputs[str(root / "scripts" / "run_suite.py")] = registration["runner_sha256"]
    variants = registration.get("variants", [])
    names = [row["name"] for row in variants]
    require(
        len(names) == 12 and set(names) == expected_variants(),
        "Complete twelve-variant cohort20 suite required",
    )
    require(
        set(summary.get("results", {})) == set(names),
        "Suite summary variant set mismatch",
    )
    require(
        len(provenance.get("subjects", [])) == 20
        and len(provenance.get("actions", [])) == 27
        and len(provenance.get("environments", [])) == 4,
        "Provenance must describe 20 people, 27 actions, four environments",
    )
    require(
        provenance.get("dataset", {}).get("kind") == "licensed_dataset",
        "Licensed dataset provenance required",
    )
    provenance_records = {
        r["id"]: (r["frames"], r["prepared_npz_sha256"])
        for r in provenance["sequence_records"]
    }
    manifests, reports, training = {}, {}, {}
    protocol_manifests = {}
    for row in variants:
        name, config = row["name"], row["config"]
        protocol = config["protocol"]
        require(protocol in PROTOCOLS, f"{name}: unknown protocol")
        seed = 2028 if name.endswith("-seed2028") else 2027
        require(config.get("seed") == seed, f"{name}: unexpected seed")
        require(
            protocol
            == ("cross_environment" if "-environment-" in name else "cross_subject"),
            f"{name}: protocol mismatch",
        )
        relative = Path(row["manifest"])
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            "Unsafe manifest path",
        )
        manifest_path = root / "data" / relative
        manifest = manifests.get(protocol)
        if manifest is None:
            manifest = record(manifest_path)
            manifests[protocol] = manifest
            protocol_manifests[protocol] = row["manifest_sha256"]
            require(
                manifest.get("protocol") == protocol
                and manifest.get("provenance", {}).get("kind") == "licensed_dataset",
                "Manifest protocol/provenance mismatch",
            )
            records = {
                r["id"]: (r["frames"], r["sha256"]) for r in manifest["sequences"]
            }
            require(
                records == provenance_records,
                "Manifest cohort differs from provenance receipt",
            )
            if protocol == "cross_subject":
                require(
                    provenance.get("manifest_sha256") == digest(manifest_path),
                    "Provenance manifest binding mismatch",
                )
            test = [r for r in manifest["sequences"] if r["split"] == "test"]
            require(
                len({r["subject"] for r in test})
                == (4 if protocol == "cross_subject" else 5),
                "Unexpected held-out people count",
            )
            require(
                {r["action"] for r in test}
                <= {
                    r["action"] for r in manifest["sequences"] if r["split"] == "train"
                },
                "Report assumes all actions seen in training",
            )
            if protocol == "cross_environment":
                require(
                    {r["environment"] for r in test} == {"E04"},
                    "Expected held-out E04 environment",
                )
        require(
            digest(manifest_path)
            == row["manifest_sha256"]
            == protocol_manifests[protocol],
            f"{name}: manifest hash mismatch",
        )
        config_path = suite / "configs" / f"{name}.json"
        require(
            record(config_path) == config,
            f"{name}: saved config differs from registration",
        )
        train = record(suite / name / "training_report.json")
        test = validate_metrics(record(suite / name / "test.json"), name)
        checkpoint = suite / name / "best.pt"
        require(checkpoint.is_file(), f"{name}: missing checkpoint")
        checkpoint_hash = digest(checkpoint)
        inputs[str(checkpoint)] = checkpoint_hash
        require(
            train.get("status") == "trained"
            and train.get("purpose") == "dataset_experiment",
            f"{name}: completed dataset training required",
        )
        require(
            train.get("config") == config
            and train.get("config_source_sha256") == digest(config_path),
            f"{name}: training config binding mismatch",
        )
        require(
            train.get("epochs") == config["training"]["epochs"]
            and train.get("seed") == seed,
            f"{name}: incomplete training or wrong seed",
        )
        require(
            test.get("checkpoint_selected_by") == "validation_normalized_joint_error"
            and test.get("checkpoint_epoch") == train.get("best_epoch"),
            f"{name}: checkpoint selection mismatch",
        )
        for artifact in (train, test):
            require(
                artifact.get("manifest_sha256") == row["manifest_sha256"]
                and artifact.get("checkpoint_sha256") == checkpoint_hash,
                f"{name}: manifest/checkpoint report binding mismatch",
            )
            require(
                artifact.get("source_sha256") == expected_sources,
                f"{name}: report source binding mismatch",
            )
            require(
                runtime_without_threads(artifact)
                == registration.get("runtime_environment"),
                f"{name}: runtime binding mismatch",
            )
        require(
            test.get("training_source_sha256") == expected_sources
            and test.get("source_matches_training") is True,
            f"{name}: training source mismatch",
        )
        require(test.get("protocol") == protocol, f"{name}: report protocol mismatch")
        require(
            train.get("device")
            == registration.get("device")
            == test.get("latency", {}).get("device"),
            f"{name}: timing device mismatch",
        )
        latency = test["latency"]
        require(
            latency.get("scope")
            == "model forward only; excludes preprocessing, disk, network and rendering"
            and latency.get("batch_size") == 1
            and latency.get("repeats", 0) > 0,
            f"{name}: unknown latency scope",
        )
        for key in ("median_ms", "p95_ms", "model_fps"):
            number(latency.get(key), f"{name} {key}", low=1e-12)
        require(
            latency["p95_ms"] >= latency["median_ms"]
            and math.isclose(
                latency["model_fps"], 1000 / latency["median_ms"], rel_tol=1e-6
            ),
            f"{name}: latency accounting mismatch",
        )
        require(
            set(summary["results"][name])
            == {
                "pck",
                "normalized_joint_error",
                "windows",
                "purpose",
                "protocol",
                "latency",
            },
            f"{name}: incomplete suite summary",
        )
        for key, value in summary["results"][name].items():
            require(
                test.get(key) == value, f"{name}: suite summary differs from test {key}"
            )
        reports[name], training[name] = test, train
    means = {}
    for protocol in PROTOCOLS:
        mean = validate_metrics(
            record(suite / f"{protocol}-train-mean-pose.json"), protocol + " mean pose"
        )
        first = next(row for row in variants if row["config"]["protocol"] == protocol)
        require(
            mean.get("baseline") == "train-mean-pose-no-signal"
            and mean.get("protocol") == protocol
            and mean.get("manifest_sha256") == protocol_manifests[protocol]
            and mean.get("config_sha256")
            == digest(suite / "configs" / f"{first['name']}.json")
            and mean.get("source_sha256") == expected_sources
            and runtime_without_threads(mean)
            == registration.get("runtime_environment"),
            "Mean-pose provenance binding mismatch",
        )
        means[protocol] = mean
    comparisons = summary.get("comparisons", [])
    require(
        len(comparisons) == len(expected_pairs())
        and {(r["reference"], r["candidate"]) for r in comparisons} == expected_pairs(),
        "Incomplete paired comparisons",
    )
    for pair in comparisons:
        candidate = reports[pair["candidate"]]
        reference = (
            means[candidate["protocol"]]
            if pair["reference"] == "train-mean-pose-no-signal"
            else reports[pair["reference"]]
        )
        require(
            reference["manifest_sha256"] == candidate["manifest_sha256"]
            and reference["windows"] == candidate["windows"]
            and reference["valid_joints"] == candidate["valid_joints"]
            and all(
                reference["coverage"].get(k) == candidate["coverage"].get(k)
                for k in (
                    "candidate_windows",
                    "accepted_windows",
                    "rejected_windows",
                    "metric_eligible_windows",
                    "metric_excluded_windows",
                )
            ),
            "Paired coverage mismatch",
        )
        a, b = (
            reference.get("domains", {}).get("sequence", {}),
            candidate.get("domains", {}).get("sequence", {}),
        )
        require(a and set(a) == set(b), "Paired sequence domains differ or are empty")
        require(
            all(a[k]["valid_joints"] == b[k]["valid_joints"] for k in a),
            "Paired sequence valid N mismatch",
        )
        usable = [
            k
            for k in a
            if a[k]["normalized_joint_error"] is not None
            and b[k]["normalized_joint_error"] is not None
        ]
        mapping = candidate.get("sequence_subjects", {})
        require(
            len(usable) >= 2 and all(k in mapping for k in usable),
            "Paired usable sequence metadata missing",
        )
        for key, cluster in (
            ("sequence_blocks", "sequence"),
            ("subject_clusters", "subject"),
        ):
            interval = pair[key]
            count = (
                len(usable)
                if cluster == "sequence"
                else len({mapping[k] for k in usable})
            )
            require(
                interval.get("clusters") == count
                and interval.get("sequences") == len(usable),
                "Paired cluster accounting mismatch",
            )
            require(
                interval.get("cluster") == cluster
                and interval.get("comparison") == "candidate minus reference"
                and interval.get("ci") == "paired percentile bootstrap 95%"
                and interval.get("clusters", 0) >= 2,
                "Unknown paired confidence interval definition",
            )
            for metric in ("pck", "normalized_joint_error"):
                values = interval[metric]
                left = sum(
                    number(
                        a[k][metric],
                        "sequence metric",
                        high=1 if metric == "pck" else None,
                    )
                    for k in usable
                ) / len(usable)
                right = sum(
                    number(
                        b[k][metric],
                        "sequence metric",
                        high=1 if metric == "pck" else None,
                    )
                    for k in usable
                ) / len(usable)
                for field, expected in (
                    ("reference_sequence_mean", left),
                    ("candidate_sequence_mean", right),
                    ("delta", right - left),
                ):
                    require(
                        type(values.get(field)) in (int, float)
                        and math.isclose(
                            values[field], expected, abs_tol=1e-10, rel_tol=1e-10
                        ),
                        "Paired point estimate does not bind to sequence reports",
                    )
                number(abs(values["delta"]), "paired delta")
                require(
                    len(values["ci95"]) == 2
                    and all(
                        type(v) in (int, float) and math.isfinite(v)
                        for v in values["ci95"]
                    )
                    and values["ci95"][0] <= values["ci95"][1],
                    "Invalid paired interval",
                )
    return {
        "suite": suite,
        "registration": registration,
        "summary": summary,
        "provenance": provenance,
        "reports": reports,
        "means": means,
        "training": training,
        "inputs_sha256": inputs,
    }


def method_name(name):
    core = (
        name.removeprefix("cohort20-")
        .removeprefix("environment-")
        .removesuffix("-seed2028")
    )
    return METHODS[core]


def result_rows(data):
    rows = []
    for protocol in PROTOCOLS:
        records = [("train-mean-pose-no-signal", data["means"][protocol])]
        records += [
            (name, r)
            for name, r in data["reports"].items()
            if r["protocol"] == protocol
        ]
        for name, report in records:
            static = name == "train-mean-pose-no-signal"
            coverage, latency = report["coverage"], report.get("latency", {})
            row = {
                "protocol": protocol,
                "variant": name,
                "method": "Static training mean pose" if static else method_name(name),
                "seed": None if static else data["training"][name]["seed"],
                **{f"pck_at_{t}": report["pck_curve"][t] for t in THRESHOLDS},
                "normalized_joint_error": report["normalized_joint_error"],
                "candidate_windows": coverage["candidate_windows"],
                "accepted_windows": report["windows"],
                "rejected_windows": coverage["rejected_windows"],
                "valid_frames": report["valid_frames"],
                "excluded_frames": report["excluded_frames"],
                "valid_joints": report["valid_joints"],
                "accepted_fraction": report["windows"] / coverage["candidate_windows"],
                "metric_eligible_fraction_of_candidates": report["valid_frames"]
                / coverage["candidate_windows"],
                "calibration_ece": report["calibration"]["ece"],
                "calibration_radius": report["calibration"]["radius"],
                "calibration_event": report["calibration"]["event"],
                "median_ms": latency.get("median_ms"),
                "p95_ms": latency.get("p95_ms"),
                "model_fps": latency.get("model_fps"),
                "latency_scope": latency.get("scope"),
                "device": latency.get("device"),
                "parameters": report.get("parameters"),
            }
            rows.append(row)
    return rows


def comparison_rows(data):
    rows = []
    for pair in data["summary"]["comparisons"]:
        for kind in ("sequence_blocks", "subject_clusters"):
            ci = pair[kind]
            for metric in ("normalized_joint_error", "pck"):
                value = ci[metric]
                rows.append(
                    {
                        "protocol": data["reports"][pair["candidate"]]["protocol"],
                        "reference": pair["reference"],
                        "candidate": pair["candidate"],
                        "metric": metric,
                        "cluster": ci["cluster"],
                        "clusters": ci["clusters"],
                        "sequences": ci["sequences"],
                        "delta": value["delta"],
                        "ci95_low": value["ci95"][0],
                        "ci95_high": value["ci95"][1],
                        "reference_sequence_mean": value["reference_sequence_mean"],
                        "candidate_sequence_mean": value["candidate_sequence_mean"],
                        "bootstrap_repeats": ci["bootstrap_repeats"],
                        "bootstrap_seed": ci["bootstrap_seed"],
                        "weighting": ci["weighting"],
                    }
                )
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def f(value, places=4):
    return "N/A" if value is None else f"{value:.{places}f}"


def markdown(data, rows, differences):
    lines = [
        "# GhostMesh: measured cohort20 results",
        "",
        "A bounded MM-Fi dataset experiment covering **27 actions, 20 people and four environments**. "
        "All evaluated actions occur in training. These scores describe 2D annotation prediction on this cohort; "
        "they do not establish full-dataset, 3D, unseen-action or cross-dataset performance.",
        "",
        "## Protocol and interpretation",
        "",
        "- Cross subject: four held-out people; environments remain represented in training.",
        "- Cross environment: E04 with five people. Room and person changes are confounded, so this is a joint room/person transfer test.",
        "- Static prior: training-label mean pose with no CSI input. Last-frame encoder: the same eight-frame causal-window preprocessing and normalization used by the temporal model, then last-frame encoding.",
        "- Temporal + teacher combines temporal encoding and source-only teacher consistency. Primary pairs use seeds 2027 and 2028 in both protocols. Ablations use seed 2027 only.",
        "- PCK is the fraction of valid joints within the named fraction of each frame’s valid ground-truth 2D bounding-box diagonal. NJE is mean joint Euclidean error divided by that diagonal; it is not millimeters.",
        "- Absolute scores below are pooled valid-joint metrics. Paired changes instead give equal weight to sequence means. Their values need not equal differences of pooled scores.",
        "",
        "## Absolute test scores",
        "",
        "No absolute-score confidence intervals are inferred from paired-difference intervals. PCK figure axes start at zero.",
        "",
        "| Protocol | Method | Seed | PCK@.05 | PCK@.1 | PCK@.2 | NJE ↓ |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['protocol']} | {row['method']} | {row['seed'] or 'N/A'} | "
            + " | ".join(
                f(row[key])
                for key in (
                    "pck_at_0.05",
                    "pck_at_0.1",
                    "pck_at_0.2",
                    "normalized_joint_error",
                )
            )
            + " |"
        )
    lines += [
        "",
        "![Absolute scores](primary_metrics.png)",
        "",
        "## Paired changes and sampling uncertainty",
        "",
        "Candidate minus reference; negative NJE and positive PCK@.2 favor the candidate. "
        "Intervals are the suite’s existing paired percentile bootstrap 95% intervals. Sequence blocks and subject-cluster sensitivity "
        "describe held-out sampling for fixed checkpoints, not training-seed variability. Only two training seeds are available; "
        "small numbers of held-out people make subject intervals unstable.",
        "",
        "| Protocol | Candidate minus reference | Metric | Cluster (N) | Delta | 95% interval |",
        "|---|---|---|---|---:|---|",
    ]
    for row in differences:
        lines.append(
            f"| {row['protocol']} | {row['candidate']} minus {row['reference']} | {row['metric']} | {row['cluster']} ({row['clusters']}) | {f(row['delta'])} | [{f(row['ci95_low'])}, {f(row['ci95_high'])}] |"
        )
    lines += [
        "",
        "![Primary paired changes](paired_changes.png)",
        "",
        "## Coverage and valid N",
        "",
        "Candidate windows precede filtering. Accepted windows pass data filtering; eligible frames also have at least two valid joints and a nondegenerate target box. "
        "Metrics exclude unusable frames. Joint N is the denominator of PCK and NJE. Counts refer to overlapping windows, not independent observations.",
        "",
        "| Protocol | Variant | Candidates | Accepted | Rejected | Metric frames | Excluded frames | Valid joints | Eligible / candidates |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['protocol']} | {row['variant']} | "
            + " | ".join(
                str(row[k])
                for k in (
                    "candidate_windows",
                    "accepted_windows",
                    "rejected_windows",
                    "valid_frames",
                    "excluded_frames",
                    "valid_joints",
                )
            )
            + f" | {row['metric_eligible_fraction_of_candidates']:.2%} |"
        )
    lines += [
        "",
        "## Calibration and measured timing",
        "",
        "ECE is expected calibration error for the event **joint Euclidean error ≤ 0.05 in normalized image coordinates**. "
        "This absolute image-radius event differs from bounding-box-normalized PCK@.05. Confidence is isotropic Gaussian radial mass; "
        "ECE is evaluated without post-hoc calibration. Smaller ECE does not by itself imply better pose accuracy.",
        "",
        "Timing is measured batch-one model forward only after warmup. Median, p95 and reciprocal-median FPS exclude preprocessing, "
        "disk, network and rendering. FPS is not end-to-end throughput. The static prior has no neural-forward benchmark.",
        "",
        "| Protocol | Variant | ECE ↓ | Device | Median ms | p95 ms | Model FPS |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['protocol']} | {row['variant']} | {f(row['calibration_ece'])} | {row['device'] or 'N/A'} | {f(row['median_ms'], 3)} | {f(row['p95_ms'], 3)} | {f(row['model_fps'], 1)} |"
        )
    lines += [
        "",
        "## Single-seed ablations",
        "",
        "The ablation plot shows seed 2027 cross-subject scores. Teacher, normalization, augmentation and consistency-loss changes "
        "are diagnostic comparisons, not independent replication or proof of a causal mechanism.",
        "",
        "![Single-seed ablations](ablations.png)",
        "",
        "## Provenance and reproducibility",
        "",
        "The renderer requires completed suite status, all twelve frozen variants, both static priors, exact primary seeds, "
        "and matching source, manifest, config, checkpoint and report bindings. It reads JSON and hashes checkpoint bytes without loading models. "
        "The renderer does not rerun evaluation or recompute the suite’s bootstrap intervals.",
        "",
        f"Dataset receipt: {data['provenance'].get('evidence_basis', 'Not supplied')}",
        "",
        "See `results.csv` for raw values, `comparisons.csv` for every stored paired interval, `results.json` for source report content "
        "and `renderer_manifest.json` for input/output SHA-256 fingerprints and renderer versions.",
        "",
    ]
    return "\n".join(lines)


def plot_results(rows, differences, out, diagnostic=False):
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ghostmesh-mpl-cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "svg.fonttype": "none",
            "savefig.dpi": 200,
            "axes.axisbelow": True,
        }
    )

    def save(fig, name):
        if diagnostic:
            fig.text(
                0.5,
                -0.015,
                "SOFTWARE TEST ONLY · FABRICATED METRICS · NOT RESEARCH RESULTS",
                ha="center",
                color="#a12222",
                fontsize=12,
                weight="bold",
            )
        for ext in ("svg", "png"):
            fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
        plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(15, 8), layout="constrained")
    for i, protocol in enumerate(PROTOCOLS):
        selected = [
            r for r in rows if r["protocol"] == protocol and r["method"] in COLORS
        ]
        for j, key in enumerate(
            ("pck_at_0.05", "pck_at_0.1", "pck_at_0.2", "normalized_joint_error")
        ):
            ax = axes[i, j]
            bars = ax.bar(
                range(len(selected)),
                [r[key] for r in selected],
                width=0.68,
                color=[COLORS[r["method"]] for r in selected],
            )
            for bar, row in zip(bars, selected):
                if row["seed"] == 2028:
                    bar.set_hatch("///")
                    bar.set_edgecolor("white")
                ax.annotate(
                    f(row[key], 3),
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
            labels = [
                (
                    "Static\nprior"
                    if r["seed"] is None
                    else (
                        "Last frame"
                        if r["method"] == "Last-frame encoder"
                        else "Temporal\n+ teacher"
                    )
                    + f"\n{r['seed']}"
                )
                for r in selected
            ]
            ax.set_xticks(range(len(selected)), labels, fontsize=8)
            ax.set_ylim(0, 1.10 if j < 3 else max(r[key] for r in selected) * 1.22)
            if j < 3:
                ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(axis="y", color="#e5e7e9", linewidth=0.6)
            ax.set_title(("PCK@" + THRESHOLDS[j] + " ↑") if j < 3 else "NJE ↓")
            if j == 0:
                ax.set_ylabel(
                    "Cross subject · 4 people"
                    if i == 0
                    else "E04 · 5 people\nroom/person confounded"
                )
    fig.suptitle(
        "GhostMesh · 27 actions, 20 people, four environments\nAbsolute pooled valid-joint scores; individual seeds; no absolute confidence intervals",
        fontsize=13,
    )
    save(fig, "primary_metrics")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), layout="constrained")
    keys = [(protocol, seed) for protocol in PROTOCOLS for seed in (2027, 2028)]
    for ax, metric in zip(axes, ("normalized_joint_error", "pck")):
        for y, (protocol, seed) in enumerate(keys):
            for cluster, offset, color, marker in [
                ("sequence", -0.12, "#276a97", "o"),
                ("subject", 0.12, "#b75e29", "s"),
            ]:
                suffix = "-seed2028" if seed == 2028 else ""
                prefix = "environment-" if protocol == "cross_environment" else ""
                row = next(
                    r
                    for r in differences
                    if r["reference"] == f"cohort20-{prefix}baseline{suffix}"
                    and r["candidate"] == f"cohort20-{prefix}temporal{suffix}"
                    and r["metric"] == metric
                    and r["cluster"] == cluster
                )
                ax.hlines(
                    y + offset, row["ci95_low"], row["ci95_high"], color=color, lw=2
                )
                ax.plot(
                    row["delta"],
                    y + offset,
                    marker=marker,
                    color=color,
                    markersize=5,
                    label=f"{cluster.capitalize()} clusters" if y == 0 else None,
                )
        ax.axvline(0, color="#555555", lw=1, linestyle="--")
        ax.set_yticks(
            range(len(keys)),
            [
                ("Cross subject" if p == "cross_subject" else "E04 transfer")
                + f" · {s}"
                for p, s in keys
            ],
        )
        ax.invert_yaxis()
        ax.set_ylim(3.55, -0.6)
        ax.grid(axis="x", color="#e5e7e9", lw=0.6)
        ax.set_xlabel(
            "Δ NJE (negative favors temporal + teacher)"
            if metric == "normalized_joint_error"
            else "Δ PCK@0.2 (positive favors temporal + teacher)"
        )
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(
        "Primary paired changes · temporal + teacher minus last-frame encoder\nStored 95% paired bootstrap intervals; fixed checkpoints, not training variance",
        fontsize=12,
    )
    save(fig, "paired_changes")
    selected = [
        r for r in rows if r["protocol"] == "cross_subject" and r["seed"] == 2027
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), layout="constrained")
    for ax, key in zip(axes, ("pck_at_0.2", "normalized_joint_error")):
        ax.barh(
            range(len(selected)),
            [r[key] for r in selected],
            color=[COLORS.get(r["method"], "#79899a") for r in selected],
        )
        ax.set_yticks(range(len(selected)), [r["method"] for r in selected])
        ax.invert_yaxis()
        ax.set_xlim(
            0, 1.08 if key == "pck_at_0.2" else max(r[key] for r in selected) * 1.2
        )
        for y, row in enumerate(selected):
            ax.annotate(
                f(row[key], 4),
                (row[key], y),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
            )
        ax.set_xlabel("PCK@0.2 ↑" if key == "pck_at_0.2" else "NJE ↓")
        if key == "pck_at_0.2":
            ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="x", color="#e5e7e9", lw=0.6)
    fig.suptitle(
        "Cross-subject ablations · seed 2027 only\nAbsolute pooled scores; diagnostic comparisons without training-variance intervals",
        fontsize=12,
    )
    save(fig, "ablations")
    return matplotlib.__version__


def render(suite, out, provenance, root=None):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {out}")
    diagnostic = root is not None
    data = load_suite(suite, provenance, root=root)
    rows, differences = result_rows(data), comparison_rows(data)
    # Validate everything before creating any public-facing artifacts.
    out.mkdir(parents=True, exist_ok=False)
    try:
        version = plot_results(rows, differences, out, diagnostic=diagnostic)
        (out / "results.md").write_text(
            (
                "> SOFTWARE TEST ONLY: FABRICATED METRICS; NOT RESEARCH RESULTS.\n\n"
                if diagnostic
                else ""
            )
            + markdown(data, rows, differences)
        )
        write_csv(out / "results.csv", rows)
        write_csv(out / "comparisons.csv", differences)
        machine = {
            "schema_version": 1,
            "purpose": "software_test_only" if diagnostic else "dataset_experiment",
            "rows": rows,
            "paired_comparisons": data["summary"]["comparisons"],
            "test_reports": data["reports"],
            "mean_pose_reports": data["means"],
            "training_reports": data["training"],
            "preregistration": data["registration"],
            "provenance": data["provenance"],
        }
        (out / "results.json").write_text(
            json.dumps(machine, indent=2, allow_nan=False) + "\n"
        )
        # Rehash inputs after rendering, rejecting artifacts that changed mid-read.
        require(
            all(digest(path) == value for path, value in data["inputs_sha256"].items()),
            "An input changed during rendering",
        )
        renderer = Path(__file__).resolve()
        manifest = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "renderer": str(renderer),
            "renderer_sha256": digest(renderer),
            "renderer_python": platform.python_version(),
            "matplotlib": version,
            "command": sys.argv,
            "suite": str(data["suite"]),
            "inputs_sha256": data["inputs_sha256"],
            "outputs_sha256": {p.name: digest(p) for p in sorted(out.iterdir())},
            "diagnostic_software_test_only": diagnostic,
            "scope": "JSON reporting only; no training, evaluation or bootstrap recomputation",
        }
        (out / "renderer_manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
    except Exception:
        # Preserve diagnostic bytes while making failure impossible to mistake for a complete report.
        (out / "RENDER_FAILED.txt").write_text(
            "Reporting failed. This directory is incomplete; do not publish.\n"
        )
        raise
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, required=True, help="New directory; overwriting is refused"
    )
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args()
    try:
        destination = render(args.suite, args.out, args.provenance)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as error:
        parser.exit(2, f"Report refused: {error}\n")
    print(destination / "results.md")


if __name__ == "__main__":
    main()
