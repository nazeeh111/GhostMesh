"""Exercise frozen-suite orchestration without running ML or decoding data."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def suite(tmp_path, monkeypatch):
    project = Path(__file__).resolve().parents[1]
    script = project / "scripts" / "run_suite.py"
    spec = importlib.util.spec_from_file_location("ghostmesh_suite_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    isolated = tmp_path / "project"
    configs = isolated / "configs"
    configs.mkdir(parents=True)
    for source in (project / "configs").glob("*.json"):
        (configs / source.name).write_bytes(source.read_bytes())
    monkeypatch.setattr(module, "ROOT", isolated)
    subject = tmp_path / "data" / "manifest.json"
    subject.parent.mkdir()
    subject.write_text('{"test_owned_manifest":"subject"}\n')
    environment = subject.with_name("cross-environment.json")
    environment.write_text('{"test_owned_manifest":"environment"}\n')
    output = tmp_path / "experiments"
    events = []
    sources = {"models.py": "original-model-source", "data.py": "original-data-source"}
    runtime = {"python": "test-runtime", "machine": "test-machine", "torch_threads": 2}
    runner = {"hash": module.digest(script)}
    original_digest = module.digest

    def digest(path):
        return (
            runner["hash"]
            if Path(path).resolve() == script.resolve()
            else original_digest(path)
        )

    monkeypatch.setattr(module, "digest", digest)
    monkeypatch.setattr(module, "_source_fingerprints", lambda: dict(sources))
    monkeypatch.setattr(module, "_runtime_environment", lambda: dict(runtime))
    hooks = {"after_train": None}

    def train(manifest, config_path, destination, device="cpu", resume=False):
        destination = Path(destination)
        config = json.loads(Path(config_path).read_text())
        events.append(("train", destination.name, resume))
        destination.mkdir(exist_ok=resume)
        checkpoint = destination / "best.pt"
        checkpoint.write_bytes(
            (destination.name + json.dumps(config, sort_keys=True)).encode()
        )
        runtime["torch_threads"] = config.get("threads", 2)
        report = {
            "config": config,
            "manifest_sha256": digest(manifest),
            "config_source_sha256": digest(config_path),
            "checkpoint_sha256": digest(checkpoint),
            "source_sha256": dict(sources),
            "runtime_environment": dict(runtime),
            "device": device,
        }
        module.save(destination / "training_report.json", report)
        if hooks["after_train"] is not None:
            hooks["after_train"](destination.name)
        return report

    def report_for(manifest):
        return {
            "pck": 0.5,
            "normalized_joint_error": 0.3,
            "windows": 5,
            "purpose": "dataset_experiment",
            "manifest_sha256": digest(manifest),
            "protocol": "cross_environment"
            if Path(manifest) == environment
            else "cross_subject",
            "latency": {
                "median_ms": 1.0,
                "scope": "test-owned fake, never a measured benchmark",
            },
            "source_sha256": dict(sources),
            "runtime_environment": dict(runtime),
        }

    def evaluate(manifest, checkpoint, split="test", device="cpu", output=None):
        events.append(("evaluate", Path(checkpoint).parent.name))
        report = {
            **report_for(manifest),
            "checkpoint_sha256": digest(checkpoint),
            "split": split,
            "device": device,
        }
        report["latency"]["device"] = device
        module.save(output, report)
        return report

    def mean(manifest, config_path, split="test", output=None):
        events.append(("mean", report_for(manifest)["protocol"]))
        report = {
            **report_for(manifest),
            "split": split,
            "config_sha256": digest(config_path),
        }
        module.save(output, report)
        return report

    def compare(reference, candidate, **kwargs):
        assert reference["manifest_sha256"] == candidate["manifest_sha256"]
        assert reference["protocol"] == candidate["protocol"]
        return {
            "scope": "test-owned fake",
            "cluster": kwargs.get("cluster", "sequence"),
        }

    monkeypatch.setattr(module, "train", train)
    monkeypatch.setattr(module, "evaluate", evaluate)
    monkeypatch.setattr(module, "evaluate_mean_pose", mean)
    monkeypatch.setattr(module, "compare_reports", compare)
    requested = [
        "subset-baseline",
        "subset-temporal",
        "subset-environment-baseline",
        "subset-environment-temporal",
    ]
    kwargs = {"environment_manifest": environment, "only": requested}
    return SimpleNamespace(
        module=module,
        subject=subject,
        environment=environment,
        output=output,
        events=events,
        sources=sources,
        runtime=runtime,
        runner=runner,
        hooks=hooks,
        requested=requested,
        kwargs=kwargs,
        configs=configs,
    )


def test_every_requested_training_finishes_before_any_heldout_evaluation(suite):
    result = suite.module.run_suite(suite.subject, suite.output, **suite.kwargs)
    first_heldout = next(
        i for i, event in enumerate(suite.events) if event[0] in {"evaluate", "mean"}
    )
    assert first_heldout == len(suite.requested)
    assert {event[1] for event in suite.events[:first_heldout]} == set(suite.requested)
    assert all(
        event[0] == "train" and event[2] is False
        for event in suite.events[:first_heldout]
    )
    assert {event[1] for event in suite.events if event[0] == "evaluate"} == set(
        suite.requested
    )
    assert {event[1] for event in suite.events if event[0] == "mean"} == {
        "cross_subject",
        "cross_environment",
    }
    assert set(result["results"]) == set(suite.requested)


def test_primary_cohort_requests_all_twelve_preregistered_arms(suite):
    expected = {
        "cohort20-baseline",
        "cohort20-temporal",
        "cohort20-no_consistency",
        "cohort20-no_normalization",
        "cohort20-no_augmentation",
        "cohort20-raw_consistency",
        "cohort20-baseline-seed2028",
        "cohort20-temporal-seed2028",
        "cohort20-environment-baseline",
        "cohort20-environment-temporal",
        "cohort20-environment-baseline-seed2028",
        "cohort20-environment-temporal-seed2028",
    }
    result = suite.module.run_suite(
        suite.subject,
        suite.output,
        environment_manifest=suite.environment,
        profile="cohort20",
    )
    first_heldout = next(
        i for i, event in enumerate(suite.events) if event[0] == "evaluate"
    )
    assert first_heldout == 12
    assert {event[1] for event in suite.events[:first_heldout]} == expected
    assert set(result["results"]) == expected


def test_identical_resume_verifies_completed_artifacts_without_retraining_or_retesting(
    suite,
):
    suite.module.run_suite(suite.subject, suite.output, **suite.kwargs)
    frozen = (suite.output / "preregistration.json").read_bytes()
    suite.events.clear()
    suite.module.run_suite(suite.subject, suite.output, resume=True, **suite.kwargs)
    assert suite.events == []
    assert (suite.output / "preregistration.json").read_bytes() == frozen


@pytest.mark.parametrize(
    "changed", ["config", "source", "manifest", "runner", "runtime"]
)
def test_resume_refuses_changed_frozen_inputs_before_running_any_job(suite, changed):
    suite.module.run_suite(suite.subject, suite.output, **suite.kwargs)
    frozen = (suite.output / "preregistration.json").read_bytes()
    checkpoint = suite.output / "subset-baseline" / "best.pt"
    saved_checkpoint = checkpoint.read_bytes()
    suite.events.clear()
    if changed == "config":
        path = suite.configs / "subset-baseline.json"
        value = json.loads(path.read_text())
        value["training"]["epochs"] += 1
        path.write_text(json.dumps(value))
    elif changed == "source":
        suite.sources["models.py"] = "changed model source"
    elif changed == "manifest":
        suite.subject.write_text(suite.subject.read_text() + "\n")
    elif changed == "runner":
        suite.runner["hash"] = "changed runner source"
    else:
        suite.runtime["machine"] = "different machine"
    with pytest.raises(ValueError):
        suite.module.run_suite(suite.subject, suite.output, resume=True, **suite.kwargs)
    assert suite.events == []
    assert (suite.output / "preregistration.json").read_bytes() == frozen
    assert checkpoint.read_bytes() == saved_checkpoint


@pytest.mark.parametrize("changed", ["source", "runner"])
def test_drift_after_final_training_aborts_before_the_first_heldout_evaluation(
    suite, changed
):
    def mutate_after_last_train(name):
        if sum(event[0] == "train" for event in suite.events) == len(suite.requested):
            if changed == "source":
                suite.sources["models.py"] = "changed after the final training call"
            else:
                suite.runner["hash"] = "changed after the final training call"

    suite.hooks["after_train"] = mutate_after_last_train
    with pytest.raises(ValueError, match="source changed"):
        suite.module.run_suite(suite.subject, suite.output, **suite.kwargs)
    assert len(suite.events) == len(suite.requested)
    assert all(event[0] == "train" for event in suite.events)
    assert not list(suite.output.glob("*/test.json"))


def test_resume_refuses_modified_completed_checkpoint(suite):
    suite.module.run_suite(suite.subject, suite.output, **suite.kwargs)
    checkpoint = suite.output / "subset-baseline" / "best.pt"
    checkpoint.write_bytes(b"test-owned modified checkpoint bytes")
    suite.events.clear()
    with pytest.raises(ValueError):
        suite.module.run_suite(suite.subject, suite.output, resume=True, **suite.kwargs)
    assert suite.events == []
    assert checkpoint.read_bytes() == b"test-owned modified checkpoint bytes"


@pytest.mark.parametrize(
    "changed", ["source", "runtime", "threads", "device", "config_hash"]
)
def test_resume_refuses_completed_training_with_changed_saved_provenance(
    suite, changed
):
    suite.module.run_suite(suite.subject, suite.output, **suite.kwargs)
    report_path = suite.output / "subset-baseline" / "training_report.json"
    report = json.loads(report_path.read_text())
    if changed == "source":
        report["source_sha256"]["models.py"] = "copied from a different source revision"
    elif changed == "runtime":
        report["runtime_environment"]["machine"] = "copied from a different machine"
    elif changed == "threads":
        report["runtime_environment"]["torch_threads"] += 1
    elif changed == "device":
        report["device"] = "different-device"
    else:
        report["config_source_sha256"] = "different configuration bytes"
    suite.module.save(report_path, report)
    changed_bytes = report_path.read_bytes()
    suite.events.clear()
    with pytest.raises(ValueError, match="training artifact"):
        suite.module.run_suite(suite.subject, suite.output, resume=True, **suite.kwargs)
    assert suite.events == []
    assert report_path.read_bytes() == changed_bytes


@pytest.mark.parametrize("changed", ["source", "runtime", "latency_device"])
def test_resume_refuses_changed_saved_test_provenance(suite, changed):
    suite.module.run_suite(suite.subject, suite.output, **suite.kwargs)
    report_path = suite.output / "subset-baseline" / "test.json"
    report = json.loads(report_path.read_text())
    if changed == "source":
        report["source_sha256"]["models.py"] = "different evaluation source revision"
    elif changed == "runtime":
        report["runtime_environment"]["machine"] = "different evaluation runtime"
    else:
        report["latency"]["device"] = "different latency measurement device"
    suite.module.save(report_path, report)
    changed_bytes = report_path.read_bytes()
    suite.events.clear()
    with pytest.raises(ValueError, match="test artifacts"):
        suite.module.run_suite(suite.subject, suite.output, resume=True, **suite.kwargs)
    assert suite.events == []
    assert report_path.read_bytes() == changed_bytes


def test_resume_refuses_changed_mean_baseline_config_bytes(suite):
    suite.module.run_suite(suite.subject, suite.output, **suite.kwargs)
    report_path = suite.output / "cross_subject-train-mean-pose.json"
    report = json.loads(report_path.read_text())
    report["config_sha256"] = "different mean-pose fitting configuration"
    suite.module.save(report_path, report)
    changed_bytes = report_path.read_bytes()
    suite.events.clear()
    with pytest.raises(ValueError, match="mean-pose baseline"):
        suite.module.run_suite(suite.subject, suite.output, resume=True, **suite.kwargs)
    assert suite.events == []
    assert report_path.read_bytes() == changed_bytes


@pytest.mark.parametrize("report_kind", ["neural", "mean"])
@pytest.mark.parametrize("field", ["split", "protocol"])
def test_resume_cannot_substitute_validation_or_other_protocol_for_final_test(
    suite, report_kind, field
):
    suite.module.run_suite(suite.subject, suite.output, **suite.kwargs)
    report_path = (
        suite.output / "subset-baseline" / "test.json"
        if report_kind == "neural"
        else suite.output / "cross_subject-train-mean-pose.json"
    )
    report = json.loads(report_path.read_text())
    report[field] = "validation" if field == "split" else "cross_environment"
    suite.module.save(report_path, report)
    changed_bytes = report_path.read_bytes()
    suite.events.clear()
    with pytest.raises(ValueError):
        suite.module.run_suite(suite.subject, suite.output, resume=True, **suite.kwargs)
    assert suite.events == []
    assert report_path.read_bytes() == changed_bytes


def test_resume_passes_resume_only_to_the_unfinished_existing_training_job(
    suite, monkeypatch
):
    original_train = suite.module.train
    partial = suite.output / "subset-temporal"

    def interrupted(manifest, config_path, destination, **kwargs):
        if Path(destination) == partial:
            partial.mkdir()
            (partial / "test-owned-partial.txt").write_text(
                "preserve this partial artifact"
            )
            raise KeyboardInterrupt("test-owned interrupted training")
        return original_train(manifest, config_path, destination, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(suite.module, "train", interrupted)
        with pytest.raises(KeyboardInterrupt):
            suite.module.run_suite(suite.subject, suite.output, **suite.kwargs)
    assert not any(event[0] in {"evaluate", "mean"} for event in suite.events)
    suite.events.clear()
    suite.module.run_suite(suite.subject, suite.output, resume=True, **suite.kwargs)
    trains = [event for event in suite.events if event[0] == "train"]
    assert ("train", "subset-baseline", False) not in trains
    assert ("train", "subset-temporal", True) in trains
    assert all(event[2] is False for event in trains if event[1] != "subset-temporal")
    assert (
        partial / "test-owned-partial.txt"
    ).read_text() == "preserve this partial artifact"
    first_heldout = next(
        i for i, event in enumerate(suite.events) if event[0] == "evaluate"
    )
    assert first_heldout == len(trains)
