#!/usr/bin/env python3
"""Freeze before first test; evaluate descriptive controls after main suite completion."""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.controls import evaluate_controls, freeze, SCOPE  # noqa: E402
from ghostmesh.engine import _json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--subject-manifest", required=True)
    parser.add_argument("--environment-manifest", required=True)
    parser.add_argument(
        "--config", default=str(ROOT / "configs/signal-controls-v1.json")
    )
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    manifests = {
        "cross_subject": Path(args.subject_manifest).resolve(),
        "cross_environment": Path(args.environment_manifest).resolve(),
    }
    suite, out = Path(args.suite).resolve(), Path(args.out).resolve()
    if args.freeze:
        freeze(ROOT, suite, args.config, manifests, out)
        print(f"Frozen before first main-suite test: {out / 'plan.json'}")
        return
    result = evaluate_controls(ROOT, suite, args.config, manifests, out)
    rows = []
    for name, model in result["models"].items():
        for mode in ("aligned", "permuted"):
            metric = model[mode]
            rows.append(
                {
                    "protocol": model["protocol"],
                    "model": name,
                    "mode": mode,
                    **{
                        k: metric[k]
                        for k in (
                            "pck",
                            "normalized_joint_error",
                            "valid_joints",
                            "valid_frames",
                            "excluded_frames",
                        )
                    },
                }
            )
    for protocol, state in result["protocols"].items():
        metric = state["room_prior_metrics"]
        rows.append(
            {
                "protocol": protocol,
                "model": "source-room-prior-oracle-metadata",
                "mode": "prior",
                **{
                    k: metric[k]
                    for k in (
                        "pck",
                        "normalized_joint_error",
                        "valid_joints",
                        "valid_frames",
                        "excluded_frames",
                    )
                },
            }
        )
    _json(out / "results.json", result)
    with (out / "results.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Supplemental signal controls",
        "",
        SCOPE,
        "",
        "One fixed permutation; descriptive changes only, without p-values or model selection.",
        "",
        "| Protocol | Model | Mode | NJE | PCK@0.2 | Valid frames |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['protocol']} | {row['model']} | {row['mode']} | "
            f"{row['normalized_joint_error']:.6f} | {row['pck']:.6f} | {row['valid_frames']} |"
        )
    lines.extend(
        [
            "",
            "Full metrics, fitted priors, coverage, permutation hashes/fixed points and artifact hashes: results.json.",
            "",
        ]
    )
    (out / "README.md").write_text("\n".join(lines))
    print(json.dumps({"status": result["status"], "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
