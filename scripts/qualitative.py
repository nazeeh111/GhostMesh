#!/usr/bin/env python3
"""Export four report-ranked held-out frames, or plot their saved arrays without Torch."""

from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SELECTION = "First lowest_error and first highest_error in each original seed-2027 temporal test report; extremes, not typical or representative frames."


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def nje(case):
    pairs = [
        (p, t)
        for p, t, valid in zip(case["prediction"], case["reference"], case["valid"])
        if valid
    ]
    require(len(pairs) >= 2, "Fewer than two valid joints")
    scale = math.hypot(
        *(max(t[j] for _, t in pairs) - min(t[j] for _, t in pairs) for j in (0, 1))
    )
    require(scale > 1e-6, "Degenerate reference box")
    return sum(math.dist(p, t) for p, t in pairs) / len(pairs) / scale


def export(args):
    # Only export requires NumPy/PyTorch and the installed GhostMesh package.
    import numpy as np
    from ghostmesh.replay import export_replay

    root, out = args.root.resolve(), args.out
    require(not out.exists(), "Refusing to overwrite cases")
    evidence_path = root / "results/cohort20-v1/renderer_manifest.json"
    evidence, inputs, cases = read(evidence_path), {}, []

    def bind(path):
        path = path.resolve()
        inputs[str(path.relative_to(root))] = digest(path)
        return inputs[str(path.relative_to(root))]

    bind(evidence_path)
    for name, manifest_path in (
        ("cohort20-temporal", args.subject_manifest),
        ("cohort20-environment-temporal", args.environment_manifest),
    ):
        report_path, checkpoint = (
            args.suite / name / "test.json",
            args.suite / name / "best.pt",
        )
        report, manifest = read(report_path), read(manifest_path)
        report_hash, checkpoint_hash, manifest_hash = (
            bind(report_path),
            bind(checkpoint),
            bind(manifest_path),
        )
        frozen = {
            str(Path(k).relative_to(Path(evidence["suite"]).parents[1])): v
            for k, v in evidence["inputs_sha256"].items()
        }
        require(
            all(
                frozen[str(p.resolve().relative_to(root))] == h
                for p, h in (
                    (report_path, report_hash),
                    (checkpoint, checkpoint_hash),
                    (manifest_path, manifest_hash),
                )
            ),
            "Primary renderer input binding differs",
        )
        require(
            report["checkpoint_sha256"] == checkpoint_hash
            and report["manifest_sha256"] == manifest_hash,
            "Report checkpoint/manifest mismatch",
        )
        for module, expected in report["source_sha256"].items():
            require(
                bind(root / "ghostmesh" / module) == expected,
                "Frozen runtime source differs",
            )
        bind(root / "ghostmesh/replay.py")
        for ranking in ("lowest_error", "highest_error"):
            selected = report["qualitative_candidates"][ranking][0]
            row = next(
                r for r in manifest["sequences"] if r["id"] == selected["sequence"]
            )
            sequence_path = manifest_path.parent / row["path"]
            require(
                row["split"] == "test" and bind(sequence_path) == row["sha256"],
                "Test sequence hash/split differs",
            )
            with tempfile.TemporaryDirectory(
                prefix="ghostmesh-qualitative-"
            ) as scratch:
                replay_path = Path(scratch) / "replay.json"
                export_replay(
                    manifest_path,
                    checkpoint,
                    replay_path,
                    sequence=row["id"],
                    split="test",
                    device="cpu",
                )
                replay = read(replay_path)
            index = selected["frame_index"]
            with np.load(sequence_path, allow_pickle=False) as data:
                source_time, start = (
                    float(data["timestamps"][index]),
                    float(data["timestamps"][0]),
                )
            frame = replay["frames"][index]
            require(
                abs(source_time - selected["time"]) < 1e-8
                and abs(frame["time"] - (source_time - start)) < 1e-8,
                "Frame/time correspondence differs",
            )
            case = dict(
                protocol=report["protocol"],
                variant=name,
                ranking=ranking,
                sequence=row["id"],
                frame_index=index,
                time=source_time,
                replay_elapsed_seconds=frame["time"],
                reported_nje=selected["normalized_joint_error"],
                prediction=frame["pose"],
                confidence=frame["confidence"],
                reference=frame["ground_truth"],
                valid=frame["ground_truth_valid"],
                edges=replay["edges"],
                joint_names=replay["joint_names"],
            )
            case["recomputed_nje"] = nje(case)
            require(
                abs(case["recomputed_nje"] - case["reported_nje"]) < 1e-5,
                "Exported NJE differs from original ranked score",
            )
            cases.append(case)
    require(
        all(digest(root / path) == value for path, value in inputs.items()),
        "Input bytes changed during export",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    write(
        out,
        dict(
            schema_version=1,
            selection=SELECTION,
            license="CC BY-NC 4.0; see NOTICE.md and data/LICENSE.md",
            source="MM-Fi, Yang et al., NeurIPS 2023; https://ntu-aiot-lab.github.io/mm-fi",
            coordinate_note="Isotropic 640 x 640 annotation reference plane; source timestamps, zero-based frame indices. References are supplied model-generated 2D annotations, not independently audited ground truth.",
            confidence_note="Uncalibrated isotropic Gaussian mass inside radius .05 in the annotation plane.",
            script_sha256=digest(__file__),
            inputs_sha256=inputs,
            cases=cases,
        ),
    )


def plot(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    data = read(args.cases)
    paths = [args.out / f"qualitative.{suffix}" for suffix in ("svg", "png")] + [
        args.out / "render_manifest.json"
    ]
    require(not any(p.exists() for p in paths), "Refusing to overwrite plot outputs")
    require(
        len(data["cases"]) == 4 and data["script_sha256"] == digest(__file__),
        "Case count/script binding differs",
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 12))
    for ax, case in zip(axes.flat, data["cases"]):
        require(abs(nje(case) - case["reported_nje"]) < 1e-5, "Stored case NJE differs")
        for key, color, style in (
            ("reference", "#C36B27", "--"),
            ("prediction", "#087E83", "-"),
        ):
            pose = case[key]
            for a, b in case["edges"]:
                if case["valid"][a] and case["valid"][b]:
                    ax.plot(
                        [pose[a][0], pose[b][0]],
                        [pose[a][1], pose[b][1]],
                        style,
                        color=color,
                        linewidth=2,
                    )
            valid = [p for p, ok in zip(pose, case["valid"]) if ok]
            ax.scatter(
                [p[0] for p in valid],
                [p[1] for p in valid],
                s=16,
                color=color,
                zorder=3,
            )
        protocol = (
            "Cross subject"
            if case["protocol"] == "cross_subject"
            else "E04 room/person transfer"
        )
        ax.set_title(
            f"{protocol} | {case['ranking'].replace('_', ' ')}\n{case['sequence']} | frame {case['frame_index']} (0-based)\nsource t={case['time']:.1f}s | NJE={case['reported_nje']:.4f}",
            fontsize=10,
        )
        ax.set(
            xlim=(0, 1), ylim=(1, 0), xlabel="x / 640", ylabel="y / 640", aspect="equal"
        )
        ax.grid(alpha=0.18)
    fig.suptitle(
        "GhostMesh | report-ranked held-out pose extremes\nTemporal + teacher, seed 2027 | selected extremes, not typical performance",
        fontsize=15,
        y=0.98,
    )
    fig.legend(
        handles=[
            Line2D(
                [], [], color="#C36B27", linestyle="--", label="Supplied 2D reference"
            ),
            Line2D([], [], color="#087E83", label="CSI model prediction"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.092),
        ncol=2,
    )
    fig.text(
        0.5,
        0.035,
        "Same isotropic annotation plane in all panels. No camera input to model.\nReferences are model-generated annotations; these examples do not establish reliable pose recovery.\nMM-Fi, Yang et al., NeurIPS 2023 | CC BY-NC 4.0 | ntu-aiot-lab.github.io/mm-fi\nDerivative assets: see NOTICE.md and data/LICENSE.md.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(
        left=0.08, right=0.98, bottom=0.17, top=0.85, hspace=0.48, wspace=0.24
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for path in paths[:2]:
        fig.savefig(path, dpi=180)
    plt.close(fig)
    write(
        paths[2],
        dict(
            schema_version=1,
            created_utc=datetime.now(timezone.utc).isoformat(),
            script_sha256=digest(__file__),
            cases_sha256=digest(args.cases),
            matplotlib=matplotlib.__version__,
            outputs_sha256={p.name: digest(p) for p in paths[:2]},
            selection=SELECTION,
            scope="Plots saved case arrays only; no training or evaluation.",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("export")
    for name, default in (
        ("root", ROOT),
        ("suite", ROOT / "experiments/cohort20-final-v1"),
        ("subject-manifest", ROOT / "data/mmfi-cohort20/manifest.json"),
        ("environment-manifest", ROOT / "data/mmfi-cohort20/cross-environment.json"),
    ):
        ex.add_argument(f"--{name}", type=Path, default=default)
    ex.add_argument("--out", type=Path, required=True)
    pl = sub.add_parser("plot")
    pl.add_argument("--cases", type=Path, required=True)
    pl.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    (export if args.command == "export" else plot)(args)
