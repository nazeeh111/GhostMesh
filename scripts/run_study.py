#!/usr/bin/env python3
"""Launch a new fixed cohort20 CPU study with controls frozen before testing."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.controls import freeze  # noqa: E402


def materialize_configs(suite):
    registration = json.loads((suite / "preregistration.json").read_text())
    (suite / "configs").mkdir(exist_ok=True)
    for row in registration["variants"]:
        name = row["name"]
        if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
            raise ValueError("Unsafe registered variant name")
        destination = suite / "configs" / f"{name}.json"
        expected = (
            json.dumps(row["config"], indent=2, allow_nan=False) + "\n"
        ).encode()
        # Link publishes complete, fsynced bytes exclusively; never replace a race winner.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, prefix=".registered-", delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(expected)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                pass
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        actual = destination.read_bytes()
        if actual != expected or json.loads(actual) != row["config"]:
            raise ValueError(f"Existing registered config differs: {name}")


def stop_owned(child):
    if child.poll() is not None:
        return
    child.send_signal(signal.SIGINT)
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def launch(
    subject_manifest, environment_manifest, out, controls_out, startup_timeout=120
):
    subject_manifest, environment_manifest = (
        Path(subject_manifest).resolve(),
        Path(environment_manifest).resolve(),
    )
    if os.path.lexists(out) or os.path.lexists(controls_out):
        raise ValueError("Choose new, separate suite and controls destinations")
    out, controls_out = Path(out).resolve(), Path(controls_out).resolve()
    if (
        out == controls_out
        or out in controls_out.parents
        or controls_out in out.parents
    ):
        raise ValueError("Choose new, separate suite and controls destinations")
    manifests = {
        "cross_subject": subject_manifest,
        "cross_environment": environment_manifest,
    }
    common = [
        "--subject-manifest",
        str(subject_manifest),
        "--environment-manifest",
        str(environment_manifest),
    ]
    child = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "scripts/run_suite.py"),
            *common,
            "--profile",
            "cohort20",
            "--device",
            "cpu",
            "--out",
            str(out),
        ],
        cwd=ROOT,
    )
    try:
        deadline = time.monotonic() + startup_timeout
        while True:
            if child.poll() is not None:
                raise RuntimeError(
                    f"Suite exited before controls freeze (exit {child.returncode})"
                )
            if (out / "preregistration.json").is_file() and (out / "configs").is_dir():
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for suite registration/config directory"
                )
            time.sleep(0.05)
        materialize_configs(out)
        freeze(
            ROOT, out, ROOT / "configs/signal-controls-v1.json", manifests, controls_out
        )
        code = child.wait()
        if code:
            raise RuntimeError(
                f"Main suite failed (exit {code}); preserve outputs for original --resume"
            )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/run_signal_controls.py"),
                *common,
                "--suite",
                str(out),
                "--out",
                str(controls_out),
            ],
            cwd=ROOT,
            check=True,
        )
    except BaseException:
        stop_owned(child)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-manifest", required=True)
    parser.add_argument("--environment-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--controls-out", required=True)
    args = parser.parse_args()
    launch(
        args.subject_manifest, args.environment_manifest, args.out, args.controls_out
    )


if __name__ == "__main__":
    main()
