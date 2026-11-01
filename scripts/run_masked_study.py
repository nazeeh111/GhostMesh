"""Run the fixed E02 development pilot in a fresh checkout, with safe ordering.

No data acquisition or publication. Fitting and all four numerical audits must
succeed before the ledger freezes; only then may any query score run. The large
source caches and original covariance baseline are explicit prerequisites.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import verify_masked_results as v  # noqa:E402

RAW = "work/domain-source30-rf-v1"
CACHE = "work/domain-covariance-full-cache-v1"
OLD = "work/domain-covariance-full27-v1"
SOURCE = "work/domain-masked-pretrain-e02-v1"
FULL = "work/domain-masked-fullpose-e02-v1"
BODY_CAL = "work/domain-masked-calibration-e02-v1"
FULL_CAL = "work/domain-masked-fullimage-calibration-e02-v1"
LEDGER = "work/domain-masked-e02-evaluation-freeze-v1.json"


def plan(device="cpu"):
    common = ["--source", SOURCE, "--cache", CACHE]
    full = [*common, "--raw", RAW, "--out", FULL]
    body_cal = [*common, "--out", BODY_CAL]
    full_cal = [
        *common,
        "--raw",
        RAW,
        "--fullpose-source-ready",
        FULL,
        "--out",
        FULL_CAL,
    ]
    source_sha = "sha256:" + SOURCE + "/source-ready.json"
    full_sha = "sha256:" + FULL + "/fullpose-source-ready.json"
    return [
        (
            "fit_source",
            "domain_masked_pretrain_probe.py",
            [
                "--raw",
                RAW,
                "--cache",
                CACHE,
                "--old-probe",
                OLD,
                "--out",
                SOURCE,
                "--device",
                device,
            ],
        ),
        (
            "audit_source",
            "verify_masked_pretrain_probe.py",
            [
                "--experiment",
                SOURCE,
                "--raw",
                RAW,
                "--cache",
                CACHE,
                "--old-probe",
                OLD,
                "--output",
                v.AUDITS["source"],
            ],
        ),
        (
            "fit_root",
            "domain_masked_fullpose_probe.py",
            [*full, "--source-ready-sha256", source_sha],
        ),
        (
            "audit_root",
            "verify_masked_fullpose_probe.py",
            [
                *common,
                "--raw",
                RAW,
                "--experiment",
                FULL,
                "--output",
                v.AUDITS["root"],
                "--fullpose-source-ready-sha256",
                full_sha,
                "--source-ready-sha256",
                source_sha,
            ],
        ),
        (
            "fit_calibrated_body",
            "domain_masked_calibration_probe.py",
            ["fit", *body_cal, "--source-ready-sha256", source_sha],
        ),
        (
            "fit_calibrated_full",
            "domain_masked_fullimage_calibration_probe.py",
            ["fit", *full_cal, "--fullpose-source-ready-sha256", full_sha],
        ),
        (
            "audit_calibrated_body",
            "verify_masked_calibration_probe.py",
            [
                "--mode",
                "body",
                *common,
                "--raw",
                RAW,
                "--experiment",
                BODY_CAL,
                "--expected-ready-sha256",
                "sha256:" + BODY_CAL + "/calibration-ready.json",
                "--output",
                v.AUDITS["body"],
            ],
        ),
        (
            "audit_calibrated_full",
            "verify_masked_calibration_probe.py",
            [
                "--mode",
                "full",
                *common,
                "--raw",
                RAW,
                "--full-source",
                FULL,
                "--experiment",
                FULL_CAL,
                "--expected-ready-sha256",
                "sha256:" + FULL_CAL + "/fullimage-calibration-ready.json",
                "--output",
                v.AUDITS["full"],
            ],
        ),
        ("freeze", "freeze_masked_evaluation.py", ["--output", LEDGER]),
        (
            "score_zero_body",
            "domain_masked_pretrain_probe.py",
            ["--score-frozen-out", SOURCE, "--cache", CACHE],
        ),
        (
            "score_zero_full",
            "domain_masked_fullpose_probe.py",
            [*full, "--score-frozen-sha256", full_sha],
        ),
        (
            "score_calibrated_body",
            "domain_masked_calibration_probe.py",
            ["score", *body_cal],
        ),
        (
            "score_calibrated_full",
            "domain_masked_fullimage_calibration_probe.py",
            ["score", *full_cal],
        ),
        (
            "audit_scores",
            "verify_masked_results.py",
            [
                "--ledger-sha256",
                "sha256:" + LEDGER,
                "--output",
                "work/domain-masked-post-e02-verification-v1.json",
            ],
        ),
    ]


def resolve(root, arguments):
    return [
        v.contracts.digest(root / x.removeprefix("sha256:"))
        if x.startswith("sha256:")
        else x
        for x in arguments
    ]


def run(root, receipt, *, device="cpu", fit_only=False):
    root, receipt = Path(root), Path(receipt)
    if receipt.exists() or receipt.is_symlink():
        raise FileExistsError(receipt)
    for directory in (RAW, CACHE, OLD):
        if not (root / directory).is_dir():
            raise ValueError("Missing explicit prerequisite: " + directory)
    protected = [
        SOURCE,
        FULL,
        BODY_CAL,
        FULL_CAL,
        LEDGER,
        *v.AUDITS.values(),
        "work/domain-masked-post-e02-verification-v1.json",
    ]
    for name in protected:
        if (root / name).exists() or (root / name).is_symlink():
            raise FileExistsError(
                "Preserve existing experiment; use a fresh checkout: " + name
            )
    if any(
        receipt.resolve().is_relative_to((root / name).resolve()) for name in protected
    ):
        raise ValueError("Runner receipt must be outside protected experiment outputs")
    receipt.mkdir(parents=True)
    stages = plan(device)
    if fit_only:
        stages = stages[:9]
    (receipt / "plan.json").write_text(json.dumps(stages, indent=2) + "\n")
    env = os.environ.copy()
    env.update(
        {
            name: "1"
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        }
    )
    for name, script, arguments in stages:
        if name.startswith("score_"):
            # Enforce the actual ledger and all eight hashes at EVERY score,
            # rather than trusting success of a previous subprocess alone.
            v.ledger_contract(root / LEDGER, v.contracts.digest(root / LEDGER), root)
        command = [sys.executable, "-u", "scripts/" + script, *resolve(root, arguments)]
        (receipt / (name + "-command.json")).write_text(
            json.dumps(command, indent=2) + "\n"
        )
        print(json.dumps(dict(stage=name, status="starting")), flush=True)
        completed = subprocess.run(command, cwd=root, env=env, check=False)
        (receipt / (name + "-exit.json")).write_text(
            json.dumps(dict(returncode=completed.returncode)) + "\n"
        )
        if completed.returncode:
            raise RuntimeError("Stage failed; later stages were not run: " + name)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--receipt", type=Path, default=ROOT / "work/masked-study-run-v1"
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(
            json.dumps(
                plan(args.device)[:9] if args.fit_only else plan(args.device), indent=2
            )
        )
    else:
        run(args.root, args.receipt, device=args.device, fit_only=args.fit_only)
