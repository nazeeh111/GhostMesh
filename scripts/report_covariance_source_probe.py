"""Post-fit breakdowns of the covariance development probe, without refitting.

Reports person/action reversals and a clearly post-hoc motion continuation check.
This script cannot select or alter a fitted checkpoint or decoder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.metrics import pose_metrics  # noqa: E402
from scripts.domain_background_probe import centered_r2, digest  # noqa: E402


def measure(pose, delta, target, truth_delta, record):
    metrics = pose_metrics(pose, target, np.ones(target.shape[:2], dtype=bool))
    denominator = float((truth_delta * truth_delta).sum())
    return dict(
        relative_nje=metrics["normalized_joint_error"],
        relative_pck05=metrics["pck_curve"]["0.05"],
        relative_pck20=metrics["pck_curve"]["0.2"],
        record_centered_relative_pose_r2=centered_r2(pose, target, record)["r2"],
        record_centered_delta_r2=centered_r2(delta, truth_delta, record)["r2"],
        raw_zero_reference_delta_r2=(
            1 - float(((delta - truth_delta) ** 2).sum()) / denominator
        )
        if denominator
        else None,
        delta_truth_squared_energy=denominator,
        windows=len(pose),
        records=len(set(record)),
    )


def run(out):
    if (out / "postfit_breakdowns.json").exists():
        raise FileExistsError("Post-fit breakdown already exists")
    result = json.loads((out / "results.json").read_text())
    verification = json.loads((out / "verification.json").read_text())
    if verification["models_verified"] != 9:
        raise ValueError("Nine frozen models must be verified first")
    with np.load(out / "targets-and-metadata.npz", allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    config = result["registration"]["config"]
    rooms = []
    for room in config["folds"]:
        train = (data["room"] != room) & ~np.isin(
            data["person"], config["source_validation_subjects"]
        )
        held = data["room"] == room
        target, truth_delta, record = (
            data["pose"][held],
            data["delta"][held],
            data["record"][held],
        )
        permutation = np.load(out / (room + "-permutation.npy"), allow_pickle=False)
        methods = dict(
            source_static_prior=(
                np.broadcast_to(data["pose"][train].mean(0), target.shape),
                np.zeros_like(truth_delta),
            )
        )
        for arm in config["arms"]:
            with np.load(
                out / (room + "-" + arm + "-model-prediction.npz"), allow_pickle=False
            ) as z:
                methods[arm] = (z["pose"], z["delta"])
        breakdown = {}
        for name, (pose, delta) in methods.items():
            summary = dict(
                aligned=measure(pose, delta, target, truth_delta, record),
                within_record_shuffle=measure(
                    pose[permutation], delta[permutation], target, truth_delta, record
                ),
            )
            for grouping in ("person", "action"):
                field = data[grouping][held]
                summary["per_" + grouping] = {
                    key: measure(
                        pose[pick],
                        delta[pick],
                        target[pick],
                        truth_delta[pick],
                        record[pick],
                    )
                    for key in sorted(set(field))
                    for pick in [field == key]
                }
            breakdown[name] = summary
        inv = breakdown["congruence_dynamics"]
        prior = breakdown["source_static_prior"]
        reversals = dict(
            people_nonpositive_raw_delta_r2=[
                key
                for key, m in inv["per_person"].items()
                if m["raw_zero_reference_delta_r2"] is None
                or m["raw_zero_reference_delta_r2"] <= 0
            ],
            actions_nonpositive_raw_delta_r2=[
                key
                for key, m in inv["per_action"].items()
                if m["raw_zero_reference_delta_r2"] is None
                or m["raw_zero_reference_delta_r2"] <= 0
            ],
            people_relative_nje_not_better_than_static=[
                key
                for key, m in inv["per_person"].items()
                if m["relative_nje"] >= prior["per_person"][key]["relative_nje"]
            ],
            actions_relative_nje_not_better_than_static=[
                key
                for key, m in inv["per_action"].items()
                if m["relative_nje"] >= prior["per_action"][key]["relative_nje"]
            ],
        )
        rooms.append(dict(held_room=room, methods=breakdown, reversals=reversals))
    posthoc = dict(
        introduced="After observing the first E01 development score, before any full-cohort escalation; NOT the preregistered gate or final efficacy acceptance",
        rule="Congruence-dynamics rawzero-reference deltaR2>0 and better than fixed within-record shuffle in each of all3development rooms",
        passed=bool(
            all(
                row["methods"]["congruence_dynamics"]["aligned"][
                    "raw_zero_reference_delta_r2"
                ]
                is not None
                and row["methods"]["congruence_dynamics"]["within_record_shuffle"][
                    "raw_zero_reference_delta_r2"
                ]
                is not None
                and row["methods"]["congruence_dynamics"]["aligned"][
                    "raw_zero_reference_delta_r2"
                ]
                > 0
                and row["methods"]["congruence_dynamics"]["aligned"][
                    "raw_zero_reference_delta_r2"
                ]
                > row["methods"]["congruence_dynamics"]["within_record_shuffle"][
                    "raw_zero_reference_delta_r2"
                ]
                for row in rooms
            )
        ),
    )
    report = dict(
        scope=config["scope"],
        sources=dict(
            results_sha256=digest(out / "results.json"),
            verification_sha256=digest(out / "verification.json"),
            report_script_sha256=digest(Path(__file__)),
        ),
        normalizers=dict(
            pose="Supplied2D annotation coordinates on declared640x640 isotropic reference plane, hiproot-relative34targets; no perperson/roomtarget alignment or fitted scaling",
            relative_nje="Euclidean joint error divided by the current reference skeleton bounding-box diagonal; roottranslation removed from BOTHprediction andtarget for this articulation diagnostic",
            pck="Fractions of that same reference bounding-box diagonal",
            displacement="Root-relative pose(t)-pose(t-8), in the same coordinate units. Raw deltaR2 uses ZERO as reference:1-SSE/sum(target_delta^2). No targetmean subtraction in that metric",
            centered_r2="Subtract each predicted/target recording mean separately for evaluation only. This oracle centering is not an inference input or DC recovery",
        ),
        matched_neural_control=f"Unavailable in this{len(config['selected_actions'])}action trial. Historical full27action neural runs have different budgets/data and are not paired controls here.",
        frozen_original_gate=result["continuation_gate"],
        posthoc_motion_continuation=posthoc,
        rooms=rooms,
        limits="Source DEVELOPMENT only, one seed, represented3actions acrossnewpeople/rooms. Small positive pooled scores do not establish population significance, consistency perperson/action, unknown static pose, complete absolute reconstruction or calibrated product efficacy. E04 remains sealed.",
    )
    (out / "postfit_breakdowns.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            dict(
                frozen_gate=result["continuation_gate"],
                posthoc_motion_continuation=posthoc,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.out)
