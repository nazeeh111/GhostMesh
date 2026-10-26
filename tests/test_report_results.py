"""SOFTWARE TEST ONLY: fabricated metrics, fake checkpoint bytes, no ML execution.

Everything is written under pytest temporary storage. These fixtures deliberately
exercise dataset-report validation; they are never empirical research results.
"""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ghostmesh_reporting_test", PROJECT / "scripts/report_results.py"
)
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def suite(tmp_path):
    root = tmp_path / "SOFTWARE_TEST_ONLY"
    suite = root / "experiments" / "FABRICATED_NOT_RESEARCH"
    source = root / "ghostmesh"
    source.mkdir(parents=True)
    for name in reporter.SOURCES:
        (source / name).write_text("SOFTWARE TEST ONLY: placeholder source\n")
    (root / "scripts").mkdir()
    (root / "scripts" / "run_suite.py").write_text(
        "SOFTWARE TEST ONLY: placeholder runner\n"
    )
    sources = {name: reporter.digest(source / name) for name in reporter.SOURCES}
    subjects = [f"S{base + i:02}" for base in (0, 10, 20, 30) for i in range(1, 6)]
    actions = [f"A{i:02}" for i in range(1, 28)]
    records = []
    for subject in subjects:
        environment = f"E{int(subject[1:]) // 10 + 1:02}"
        for action in actions:
            records.append(
                {
                    "id": f"{environment}/{subject}/{action}",
                    "subject": subject,
                    "environment": environment,
                    "action": action,
                    "frames": 16,
                    "sha256": "a" * 64,
                    "split": "test" if subject.endswith("5") else "train",
                }
            )
    manifests = {}
    for protocol in reporter.PROTOCOLS:
        rows = copy.deepcopy(records)
        if protocol == "cross_environment":
            for row in rows:
                row["split"] = "test" if row["environment"] == "E04" else "train"
        manifest = {
            "schema_version": 1,
            "protocol": protocol,
            "provenance": {"kind": "licensed_dataset", "SOFTWARE_TEST_ONLY": True},
            "sequences": rows,
        }
        path = root / "data" / "test-only-cohort" / (protocol + ".json")
        save(path, manifest)
        manifests[protocol] = path
    provenance = root / "SOFTWARE_TEST_ONLY-provenance.json"
    save(
        provenance,
        {
            "schema_version": 1,
            "dataset": {"kind": "licensed_dataset", "SOFTWARE_TEST_ONLY": True},
            "subjects": subjects,
            "actions": actions,
            "environments": ["E01", "E02", "E03", "E04"],
            "manifest_sha256": reporter.digest(manifests["cross_subject"]),
            "evidence_basis": "SOFTWARE TEST ONLY: all values fabricated",
            "sequence_records": [
                {
                    "id": r["id"],
                    "frames": r["frames"],
                    "prepared_npz_sha256": r["sha256"],
                }
                for r in records
            ],
        },
    )
    runtime = {
        "python": "software-test-only",
        "machine": "fabricated",
        "torch_threads": 1,
    }
    registration = {
        "profile": "cohort20",
        "source_sha256": sources,
        "runner_sha256": reporter.digest(root / "scripts" / "run_suite.py"),
        "runtime_environment": {
            "python": "software-test-only",
            "machine": "fabricated",
        },
        "device": "cpu",
        "variants": [],
        "SOFTWARE_TEST_ONLY": True,
    }
    reports = {}

    def metrics(protocol):
        coverage = {
            "candidate_windows": 12,
            "accepted_windows": 10,
            "rejected_windows": 2,
            "metric_eligible_windows": 9,
            "metric_excluded_windows": 1,
        }
        return {
            "status": "evaluated",
            "purpose": "dataset_experiment",
            "split": "test",
            "protocol": protocol,
            "SOFTWARE_TEST_ONLY": "FABRICATED; NOT RESEARCH",
            "manifest_sha256": reporter.digest(manifests[protocol]),
            "source_sha256": sources,
            "runtime_environment": runtime,
            "pck": 0.4,
            "pck_threshold": 0.2,
            "pck_curve": {"0.05": 0.1, "0.1": 0.2, "0.2": 0.4},
            "normalized_joint_error": 0.25,
            "windows": 10,
            "valid_frames": 9,
            "excluded_frames": 1,
            "valid_joints": 153,
            "coverage": coverage,
            "domains": {
                "sequence": {
                    f"test-sequence-{i}": {
                        "normalized_joint_error": 0.25,
                        "pck": 0.4,
                        "valid_joints": 17,
                    }
                    for i in range(8)
                }
            },
            "sequence_subjects": {
                f"test-sequence-{i}": f"test-subject-{i // 2}" for i in range(8)
            },
            "calibration": {
                "ece": 0.12,
                "radius": 0.05,
                "event": "joint Euclidean error <= radius in normalized image coordinates",
                "bins": [
                    {
                        "count": 153,
                        "lower": 0,
                        "upper": 1,
                        "accuracy": 0.3,
                        "confidence": 0.4,
                    }
                ],
            },
        }

    names = ["cohort20-baseline"] + sorted(
        reporter.expected_variants() - {"cohort20-baseline"}
    )
    for name in names:
        protocol = "cross_environment" if "-environment-" in name else "cross_subject"
        seed = 2028 if name.endswith("seed2028") else 2027
        config = {
            "name": name,
            "protocol": protocol,
            "seed": seed,
            "training": {"epochs": 2},
        }
        config_path = suite / "configs" / f"{name}.json"
        save(config_path, config)
        checkpoint = suite / name / "best.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"SOFTWARE TEST ONLY. NOT A REAL MODEL.")
        train = {
            "status": "trained",
            "purpose": "dataset_experiment",
            "seed": seed,
            "epochs": 2,
            "best_epoch": 1,
            "config": config,
            "config_source_sha256": reporter.digest(config_path),
            "manifest_sha256": reporter.digest(manifests[protocol]),
            "checkpoint_sha256": reporter.digest(checkpoint),
            "source_sha256": sources,
            "runtime_environment": runtime,
            "device": "cpu",
        }
        save(suite / name / "training_report.json", train)
        test = metrics(protocol)
        test.update(
            {
                "checkpoint_sha256": reporter.digest(checkpoint),
                "checkpoint_epoch": 1,
                "checkpoint_selected_by": "validation_normalized_joint_error",
                "training_source_sha256": sources,
                "source_matches_training": True,
                "latency": {
                    "device": "cpu",
                    "median_ms": 2.0,
                    "p95_ms": 3.0,
                    "model_fps": 500.0,
                    "batch_size": 1,
                    "repeats": 30,
                    "scope": "model forward only; excludes preprocessing, disk, network and rendering",
                },
            }
        )
        save(suite / name / "test.json", test)
        reports[name] = test
        registration["variants"].append(
            {
                "name": name,
                "config": config,
                "manifest": f"test-only-cohort/{protocol}.json",
                "manifest_sha256": reporter.digest(manifests[protocol]),
            }
        )
    save(suite / "preregistration.json", registration)
    for protocol in reporter.PROTOCOLS:
        first = next(
            r for r in registration["variants"] if r["config"]["protocol"] == protocol
        )
        mean = metrics(protocol)
        mean.update(
            {
                "baseline": "train-mean-pose-no-signal",
                "config_sha256": reporter.digest(
                    suite / "configs" / f"{first['name']}.json"
                ),
            }
        )
        save(suite / f"{protocol}-train-mean-pose.json", mean)
    comparisons = []
    for reference, candidate in sorted(reporter.expected_pairs()):
        row = {"reference": reference, "candidate": candidate}
        for key, cluster in [
            ("sequence_blocks", "sequence"),
            ("subject_clusters", "subject"),
        ]:
            row[key] = {
                "cluster": cluster,
                "clusters": 8 if cluster == "sequence" else 4,
                "sequences": 8,
                "comparison": "candidate minus reference",
                "ci": "paired percentile bootstrap 95%",
                "bootstrap_seed": 2027,
                "bootstrap_repeats": 2000,
                "weighting": "equal weight per sequence mean",
                "pck": {
                    "delta": 0.0,
                    "ci95": [-0.02, 0.04],
                    "reference_sequence_mean": 0.4,
                    "candidate_sequence_mean": 0.4,
                },
                "normalized_joint_error": {
                    "delta": 0.0,
                    "ci95": [-0.03, 0.01],
                    "reference_sequence_mean": 0.25,
                    "candidate_sequence_mean": 0.25,
                },
            }
        comparisons.append(row)
    summary = {
        "preregistration_sha256": reporter.digest(suite / "preregistration.json"),
        "results": {
            name: {
                key: r[key]
                for key in (
                    "pck",
                    "normalized_joint_error",
                    "windows",
                    "purpose",
                    "protocol",
                    "latency",
                )
            }
            for name, r in reports.items()
        },
        "comparisons": comparisons,
    }
    save(suite / "suite_report.json", summary)
    save(suite / "process.json", {"status": "completed", "stage": "evaluated"})
    return SimpleNamespace(root=root, path=suite, provenance=provenance, tmp=tmp_path)


def change(path, action):
    value = reporter.read(path)
    action(value)
    save(path, value)


def test_complete_bound_json_only_suite(suite):
    data = reporter.load_suite(suite.path, suite.provenance, root=suite.root)
    assert len(data["reports"]) == 12
    rows = reporter.result_rows(data)
    assert len(rows) == 14
    assert len(reporter.comparison_rows(data)) == 52
    assert {r["seed"] for r in rows if r["seed"]} == {2027, 2028}
    assert all(
        r["seed"] == 2027
        for r in rows
        if r["variant"].endswith(
            ("no_consistency", "no_normalization", "no_augmentation", "raw_consistency")
        )
    )


@pytest.mark.parametrize(
    "case",
    [
        "running",
        "missing_test",
        "source",
        "runner",
        "manifest",
        "checkpoint",
        "training_config",
        "summary",
        "purpose",
        "mean_purpose",
        "paired_ci",
        "paired_estimate",
        "paired_count",
        "missing_pair",
        "coverage",
        "calibration",
        "timing",
    ],
)
def test_refuses_unbound_or_incomplete_artifacts(suite, case):
    name = "cohort20-baseline"
    test_path = suite.path / name / "test.json"
    if case == "running":
        change(suite.path / "process.json", lambda r: r.update(status="running"))
    elif case == "missing_test":
        test_path.unlink()
    elif case == "source":
        (suite.root / "ghostmesh" / "models.py").write_text("changed test source")
    elif case == "runner":
        (suite.root / "scripts" / "run_suite.py").write_text("changed runner")
    elif case == "manifest":
        (suite.root / "data" / "test-only-cohort" / "cross_subject.json").write_text(
            "{}"
        )
    elif case == "checkpoint":
        (suite.path / name / "best.pt").write_bytes(b"changed fake bytes")
    elif case == "training_config":
        change(
            suite.path / name / "training_report.json",
            lambda r: r["config"].update(seed=0),
        )
    elif case == "summary":
        change(
            suite.path / "suite_report.json",
            lambda r: r["results"][name].update(pck=0.99),
        )
    elif case == "purpose":
        change(test_path, lambda r: r.update(purpose="software_check_only"))
    elif case == "mean_purpose":
        change(
            suite.path / "cross_subject-train-mean-pose.json",
            lambda r: r.update(purpose="software_check_only"),
        )
    elif case == "paired_ci":
        change(
            suite.path / "suite_report.json",
            lambda r: r["comparisons"][0]["sequence_blocks"]["pck"].update(
                ci95=[0.5, -0.5]
            ),
        )
    elif case == "paired_estimate":
        change(
            suite.path / "suite_report.json",
            lambda r: r["comparisons"][0]["sequence_blocks"]["pck"].update(delta=0.01),
        )
    elif case == "paired_count":
        change(
            suite.path / "suite_report.json",
            lambda r: r["comparisons"][0]["sequence_blocks"].update(clusters=4),
        )
    elif case == "missing_pair":
        change(suite.path / "suite_report.json", lambda r: r["comparisons"].pop())
    elif case == "coverage":
        change(test_path, lambda r: r["coverage"].update(accepted_windows=8))
    elif case == "calibration":
        change(test_path, lambda r: r["calibration"].update(event="PCK"))
    elif case == "timing":
        change(test_path, lambda r: r["latency"].update(scope="end-to-end"))
    out = suite.tmp / "SHOULD_NOT_EXIST"
    with pytest.raises((ValueError, FileNotFoundError)):
        reporter.render(suite.path, out, suite.provenance, root=suite.root)
    assert not out.exists()


def test_refuses_output_overwrite_before_read(suite):
    out = suite.tmp / "preserve"
    out.mkdir()
    (out / "original").write_text("preserve original bytes")
    with pytest.raises(FileExistsError):
        reporter.render(suite.path, out, suite.provenance, root=suite.root)
    assert (out / "original").read_text() == "preserve original bytes"


def test_nonfinite_json_is_refused(tmp_path):
    path = tmp_path / "nonfinite.json"
    path.write_text('{"score": NaN}')
    with pytest.raises(ValueError, match="Non-finite"):
        reporter.read(path)


def test_render_software_diagnostic_only(suite):
    pytest.importorskip("matplotlib")
    out = suite.tmp / "SOFTWARE_TEST_ONLY_DIAGNOSTIC"
    reporter.render(suite.path, out, suite.provenance, root=suite.root)
    assert reporter.read(out / "results.json")["purpose"] == "software_test_only"
    assert "FABRICATED METRICS" in (out / "results.md").read_text()
    assert "confounded" in (out / "results.md").read_text()
    assert "not training-seed variability" in (out / "results.md").read_text()
    manifest = reporter.read(out / "renderer_manifest.json")
    assert manifest["diagnostic_software_test_only"]
    assert manifest["renderer_sha256"] == reporter.digest(
        PROJECT / "scripts" / "report_results.py"
    )
    for name, checksum in manifest["outputs_sha256"].items():
        assert reporter.digest(out / name) == checksum
    for name in ("primary_metrics", "paired_changes", "ablations"):
        assert (out / f"{name}.png").stat().st_size > 1000
        svg = (out / f"{name}.svg").read_text()
        assert "SOFTWARE TEST ONLY" in svg
    # These paired CIs must be copied exactly, not reused as absolute-score errors.
    assert (
        reporter.read(out / "results.json")["paired_comparisons"]
        == reporter.read(suite.path / "suite_report.json")["comparisons"]
    )


def test_storage_cache_metadata_is_not_coverage(suite):
    # Cache reuse/key details can legitimately differ for matched target windows.
    change(
        suite.path / "cohort20-baseline" / "test.json",
        lambda r: r["coverage"].update(
            storage={"cache_key": "test-only-a", "reused": True}
        ),
    )
    change(
        suite.path / "cohort20-temporal" / "test.json",
        lambda r: r["coverage"].update(
            storage={"cache_key": "test-only-b", "reused": False}
        ),
    )
    assert (
        len(
            reporter.load_suite(suite.path, suite.provenance, root=suite.root)[
                "reports"
            ]
        )
        == 12
    )
