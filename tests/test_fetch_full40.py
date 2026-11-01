"""Full acquisition must cover40people and stop on a rejected provider request."""

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

spec = importlib.util.spec_from_file_location(
    "fetch_full40", Path(__file__).resolve().parents[1] / "scripts/fetch_full40.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_dry_run_has_exact_full_cohort_without_execution(monkeypatch, capsys):
    def no_execution(*args, **kwargs):
        pytest.fail("Dry run must never execute the network fetcher")

    monkeypatch.setattr(module.subprocess, "run", no_execution)
    module.main(["--dry-run", "--resume"])
    result = json.loads(capsys.readouterr().out)
    people = []
    for command in result["commands"]:
        selected = command[command.index("--subjects") + 1 : command.index("--actions")]
        people.extend(selected)
        assert len(selected) == 10
        assert command[command.index("--actions") + 1 : command.index("--out")] == [
            f"A{i:02d}" for i in range(1, 28)
        ]
        assert "--resume" in command
    assert sorted(people) == [f"S{i:02d}" for i in range(1, 41)]


def test_provider_failure_stops_before_next_environment(monkeypatch):
    calls = []

    def denied(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(module.subprocess, "run", denied)
    with pytest.raises(subprocess.CalledProcessError):
        module.main(["--inventory"])
    assert len(calls) == 1
