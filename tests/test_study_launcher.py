"""SOFTWARE TEST ONLY: mocked children and tiny configs; no training/evaluation."""

import importlib.util
import json
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ghostmesh_study_launcher_test", PROJECT / "scripts/run_study.py"
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def registration(suite):
    (suite / "configs").mkdir(parents=True)
    config = {
        "schema_version": 1,
        "note": "SOFTWARE TEST ONLY",
        "training": {"epochs": 40},
    }
    (suite / "preregistration.json").write_text(
        json.dumps({"variants": [{"name": "fixture", "config": config}]})
    )
    return config


def test_materialization_exact_bytes_preserves_existing_and_rejects_drift(tmp_path):
    config = registration(tmp_path)
    launcher.materialize_configs(tmp_path)
    path = tmp_path / "configs/fixture.json"
    assert (
        path.read_bytes()
        == (json.dumps(config, indent=2, allow_nan=False) + "\n").encode()
    )
    before = path.stat().st_ino
    launcher.materialize_configs(tmp_path)
    assert path.stat().st_ino == before
    path.write_text(json.dumps(config))  # Same semantics, noncanonical bytes.
    with pytest.raises(ValueError, match="differs"):
        launcher.materialize_configs(tmp_path)
    assert path.read_text() == json.dumps(config)
    assert not list((tmp_path / "configs").glob(".registered-*"))


def test_atomic_publish_verifies_race_winner(tmp_path, monkeypatch):
    registration(tmp_path)
    real_link = launcher.os.link

    def race(source, destination):
        destination.write_bytes(source.read_bytes())
        real_link(
            source, destination
        )  # FileExistsError, matching the original runner race.

    monkeypatch.setattr(launcher.os, "link", race)
    launcher.materialize_configs(tmp_path)
    assert (tmp_path / "configs/fixture.json").is_file()


@pytest.fixture
def harness(tmp_path, monkeypatch):
    events = []
    suite, controls = tmp_path / "suite", tmp_path / "controls"

    class Child:
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, **kwargs):
            events.append("wait")
            self.returncode = 0
            return 0

        def send_signal(self, sig):
            events.append(("signal", sig))

    child = Child()

    def popen(command, **kwargs):
        assert "--profile" in command and "cohort20" in command and "cpu" in command
        assert "--resume" not in command and "--epochs" not in command
        events.append("start")
        registration(suite)
        return child

    def freeze(root, output, config, manifests, destination):
        assert (suite / "configs/fixture.json").is_file()
        assert child.returncode is None
        events.append("freeze")
        destination.mkdir()
        (destination / "plan.json").write_text("SOFTWARE TEST ONLY")

    def run(command, **kwargs):
        assert child.returncode == 0
        assert (controls / "plan.json").is_file()
        assert kwargs["check"] is True
        events.append("controls")

    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    monkeypatch.setattr(launcher.subprocess, "run", run)
    monkeypatch.setattr(launcher, "freeze", freeze)
    return SimpleNamespace(
        suite=suite,
        controls=controls,
        child=child,
        events=events,
        args=(
            tmp_path / "subject.json",
            tmp_path / "environment.json",
            suite,
            controls,
        ),
    )


def test_ordering_materialize_freeze_wait_then_controls(harness):
    launcher.launch(*harness.args)
    assert harness.events == ["start", "freeze", "wait", "controls"]


def test_early_child_exit_does_not_freeze_or_evaluate(harness, monkeypatch):
    harness.child.returncode = 7
    with pytest.raises(RuntimeError, match="before controls freeze"):
        launcher.launch(*harness.args)
    assert harness.events == ["start"]
    assert not harness.controls.exists()


def test_freeze_failure_interrupts_only_owned_child_and_preserves_outputs(
    harness, monkeypatch
):
    def fail(*args):
        raise ValueError("SOFTWARE TEST ONLY freeze refusal")

    monkeypatch.setattr(launcher, "freeze", fail)
    with pytest.raises(ValueError, match="freeze refusal"):
        launcher.launch(*harness.args)
    assert harness.events == ["start", ("signal", signal.SIGINT), "wait"]
    assert (harness.suite / "preregistration.json").is_file()
    assert (harness.suite / "configs/fixture.json").is_file()


def test_startup_timeout_cleans_up_owned_child(harness, monkeypatch):
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: harness.child)
    with pytest.raises(TimeoutError, match="registration"):
        launcher.launch(*harness.args, startup_timeout=0)
    assert harness.events == [("signal", signal.SIGINT), "wait"]


def test_existing_destination_refused_before_child_start(harness):
    harness.controls.mkdir()
    with pytest.raises(ValueError, match="new, separate"):
        launcher.launch(*harness.args)
    assert harness.events == []


def test_main_failure_preserves_plan_and_skips_controls(harness, monkeypatch):
    def failed_wait(**kwargs):
        harness.events.append("wait")
        harness.child.returncode = 9
        return 9

    monkeypatch.setattr(harness.child, "wait", failed_wait)
    with pytest.raises(RuntimeError, match="Main suite failed"):
        launcher.launch(*harness.args)
    assert harness.events == ["start", "freeze", "wait"]
    assert (harness.controls / "plan.json").is_file()


def test_failed_controls_does_not_signal_completed_main(harness, monkeypatch):
    def failed_controls(*args, **kwargs):
        raise launcher.subprocess.CalledProcessError(1, "SOFTWARE TEST ONLY")

    monkeypatch.setattr(launcher.subprocess, "run", failed_controls)
    with pytest.raises(launcher.subprocess.CalledProcessError):
        launcher.launch(*harness.args)
    assert harness.events == ["start", "freeze", "wait"]
