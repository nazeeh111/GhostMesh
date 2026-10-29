"""Source-only coordinate/identifiability audit of frozen pose predictions.

No RF readout is fitted. Source-validation-only constant2D orthogonal transforms
are DIAGNOSTIC, not query-GT alignment. Canonical targets are a different task
and cannot replace the original image-plane acceptance criteria.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.metrics import pose_metrics  # noqa:E402
from scripts.domain_background_probe import centered_r2, digest, multiply  # noqa:E402
from scripts.domain_covariance_source_probe import fold_masks, validate_records  # noqa:E402
from scripts.domain_packet_transfer_probe import predict, transform  # noqa:E402

BONES = [
    ("shoulder_width", 5, 6),
    ("left_upper_arm", 5, 7),
    ("right_upper_arm", 6, 8),
    ("left_forearm", 7, 9),
    ("right_forearm", 8, 10),
    ("left_torso", 5, 11),
    ("right_torso", 6, 12),
    ("hip_width", 11, 12),
    ("left_thigh", 11, 13),
    ("right_thigh", 12, 14),
    ("left_shin", 13, 15),
    ("right_shin", 14, 16),
]
CONFIG = dict(
    scope="SOURCE E01-E03 diagnostic only; no RFmodel fit/RGB/E04, new3D/camera-calibration or original-target efficacy claim",
    label_records=810,
    label_frames=240570,
    pose_shape=[17, 2],
    stages=["native", "centred", "unit_body"],
    centre="Own17-joint centroid, not GTroot supplied to predictions. Centroid decomposition is orthogonal; hip root does not yield the same exact MSE decomposition",
    scale="Own RMS17-joint radius after centering. Guard1e-4 in normalized image units. Never use query GTscale for predictions",
    scale_guard=1e-4,
    labels="All30subjects×27actions×297frames, balanced. Room-associated variance after subtracting globalaction×endpoint teacher means; room/person/view/execution/teacher effects remain confounded",
    bone_ratios="Twelve fixed projected2D limb/torso widths divided by per-pose shoulder-midpoint-to-hip-midpoint length; not physical bone lengths or 3D-invariant quantities",
    baseline="In EACH targetspace, fit TRAINlabel mean and endpoint-only TRAINmeans directly, NOT canonicalize an old mean. Same source-validation rotation/reflection opportunity for every method",
    rf_model="Frozen rawpacket CURRENTfullpose head, natural-linear/RFF choice from its original source-validation selection. No canonical re-selection of RFfamilies/alphas",
    alignment="Unit-body space only: one constant2x2 proper rotation or reflection+proper rotation permethod/perheldfold, min SOURCEVALcoordinateMSE, firstproper tie. ALL3fold choices on disk before heldscores",
    query_labels="Only scoring and descriptive teachergeometry; never used for query-specific centering/scale/rotation of predictions, calibration or state initialization",
    metrics="Original imageNJE/PCK alongside separately labeled new-targetNJE/PCK/record-centeredR2 and prediction-shuffle; no original-target acceptance replacement",
    degeneracy="Report invalid prediction/target counts, no silent worst-frame deletion; continuation requires100percent canonical coverage",
    gate="FrozenRF unit_body+sourceVALselectedorthogonal transform: lower NEWtargetNJE and no-worsePCK.05/.2 than BOTH independently-aligned canonicalTRAINmean and endpoint-only controls in EACH3rooms; positive canonicalrecordR2 and worseRFshuffleNJE inall3. Diagnostic only, not original image-plane acceptance",
    limitations="Centering/scaling/proper2Drotation/reflection do not remove yaw/depth foreshortening. This is not evidence that camera alone caused previous transfer failure. No per-query oracle alignment score is promoted",
)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def coordinates(pose, stage):
    x = np.asarray(pose, dtype=np.float64).copy()
    require(x.ndim == 3 and x.shape[1:] == (17, 2), "Expected poses[N,17,2]")
    valid = np.isfinite(x).all((1, 2))
    # An invalid row must not reach mean/subtraction (inf-inf raises a warning).
    # This is a mask, not a replacement pose: restore NaNs before returning it.
    x[~valid] = 0
    if stage == "native":
        x[~valid] = np.nan
        return x, valid
    if stage not in ("centred", "unit_body"):
        raise ValueError("Unknown coordinate stage")
    x -= x.mean(1, keepdims=True)
    if stage == "unit_body":
        scale = np.sqrt(np.square(x).sum(2).mean(1))
        valid &= scale > CONFIG["scale_guard"]
        x /= np.where(valid, scale, 1)[:, None, None]
    x[~valid] = np.nan
    return x, valid


def rotation(x, y):
    """One proper row-vector rotation, no translation/scale/shear parameter."""
    h = multiply(x.reshape(-1, 2).T, y.reshape(-1, 2))
    u, _, vt = np.linalg.svd(h)
    return multiply(multiply(u, np.diag([1.0, np.linalg.det(multiply(u, vt))])), vt)


def align_source_validation(prediction, target):
    good = np.isfinite(prediction).all((1, 2)) & np.isfinite(target).all((1, 2))
    require(good.any(), "No valid source-validation skeletons for constant rotation")
    x, y = prediction[good], target[good]
    proper = rotation(x, y)
    flip = np.diag([-1.0, 1.0])
    reflected = multiply(flip, rotation(np.einsum("...j,jk->...k", x, flip), y))
    choices = []
    for name, matrix in (("proper", proper), ("reflection_plus_proper", reflected)):
        error = float(np.square(np.einsum("...j,jk->...k", x, matrix) - y).mean())
        choices.append(
            dict(
                kind=name,
                matrix=matrix.tolist(),
                source_validation_mse=error,
                determinant=float(np.linalg.det(matrix)),
            )
        )
    return dict(
        proper=choices[0],
        selected=min(choices, key=lambda v: v["source_validation_mse"]),
        candidates=choices,
        source_validation_valid=int(good.sum()),
        source_validation_total=len(good),
    )


def source_controls(target, ends, train, ids):
    valid = np.isfinite(target).all((1, 2))
    use = train & valid
    require(use.any(), "No valid TRAINlabels in targetspace")
    mean = target[use].mean(0)
    static = np.broadcast_to(mean, (len(ids), 17, 2)).copy()
    clock = static.copy()
    fallback = []
    for end in np.unique(ends[ids]):
        source = use & (ends == end)
        query = ends[ids] == end
        if source.any():
            clock[query] = target[source].mean(0)
        else:
            fallback.append(int(end))
    return dict(source_mean=static, endpoint_clock=clock), fallback


def score(prediction, target, record):
    p, y = np.asarray(prediction), np.asarray(target)
    pv = np.isfinite(p).all((1, 2))
    yv = np.isfinite(y).all((1, 2))
    bbox = np.linalg.norm(
        np.where(yv[:, None, None], y, 0).max(1)
        - np.where(yv[:, None, None], y, 0).min(1),
        axis=1,
    )
    yv &= bbox > 1e-6
    keep = pv & yv
    result = dict(
        total=len(p),
        valid=int(keep.sum()),
        invalid_predictions=int((~pv).sum()),
        invalid_targets=int((~yv).sum()),
    )
    if not keep.any():
        return dict(result, metrics=None, record_pose_r2=None)
    m = pose_metrics(p[keep], y[keep], np.ones((keep.sum(), 17), bool))
    result.update(
        metrics={
            k: m[k]
            for k in (
                "normalized_joint_error",
                "pck_curve",
                "valid_frames",
                "valid_joints",
            )
        },
        coordinate_mse=float(np.square(p[keep] - y[keep]).mean()),
        record_pose_r2=centered_r2(p[keep], y[keep], record[keep])["r2"],
    )
    return result


def balanced_variance(q):
    require(
        q.ndim == 6 and q.shape[-2:] == (17, 2),
        "Expected balanced room/person/action/time/joint/coordinate blocks",
    )
    global_action_time = q.mean((0, 1))
    room_action_time = q.mean(1)
    total = float(np.square(q - global_action_time).sum())
    room = float(q.shape[1] * np.square(room_action_time - global_action_time).sum())
    within_record = float(np.square(q - q.mean(3, keepdims=True)).sum())
    energy = float(np.square(q).sum())
    return dict(
        total_after_action_endpoint_mean_ss=total,
        room_associated_ss=room,
        room_associated_share=room / total if total else None,
        within_room_person_residual_ss=total - room,
        within_record_temporal_ss=within_record,
        within_record_fraction_of_target_energy=within_record / energy
        if energy
        else None,
        target_squared_energy=energy,
    )


def label_geometry(pose):
    require(pose.shape == (240570, 17, 2), "Expected complete balanced source30 labels")
    report = {}
    for stage in CONFIG["stages"]:
        q, valid = coordinates(pose, stage)
        if not valid.all():
            report[stage] = dict(
                status="unavailable", invalid_frames=int((~valid).sum())
            )
            continue
        q = q.reshape(3, 10, 27, 297, 17, 2)
        report[stage] = dict(
            status="ok", valid_frames=len(pose), **balanced_variance(q)
        )
    p = pose.reshape(3, 10, 27, 297, 17, 2)
    torso = np.linalg.norm(
        p[..., [5, 6], :].mean(-2) - p[..., [11, 12], :].mean(-2), axis=-1
    )
    valid = torso > CONFIG["scale_guard"]
    bones = {}
    for name, a, b in BONES:
        length = np.linalg.norm(p[..., a, :] - p[..., b, :], axis=-1)
        ratio = length / np.where(valid, torso, 1)
        if not valid.all():
            bones[name] = dict(
                status="unavailable", invalid_torso_frames=int((~valid).sum())
            )
            continue
        records = ratio.reshape(810, 297)
        bones[name] = dict(
            room_median=[float(np.median(v)) for v in ratio],
            room_q25=[float(np.quantile(v, 0.25)) for v in ratio],
            room_q75=[float(np.quantile(v, 0.75)) for v in ratio],
            person_median=np.median(ratio.reshape(30, -1), axis=1).tolist(),
            median_within_record_cv=float(
                np.median(records.std(1) / np.maximum(records.mean(1), 1e-8))
            ),
        )
    return dict(
        stages=report,
        bone_to_projected_torso_ratios=bones,
        invalid_torso_frames=int((~valid).sum()),
    )


def stage_predictions(raw_prediction, stage):
    return coordinates(raw_prediction, stage)[0]


def canonical_gate(m):
    rf = m["RF"]
    return bool(
        rf["valid"] == rf["total"]
        and rf["record_pose_r2"] is not None
        and rf["record_pose_r2"] > 0
        and m["RF_shuffle"]["metrics"]["normalized_joint_error"]
        > rf["metrics"]["normalized_joint_error"]
        and all(
            c["valid"] == rf["total"]
            and rf["metrics"]["normalized_joint_error"]
            < c["metrics"]["normalized_joint_error"]
            and all(
                rf["metrics"]["pck_curve"][t] >= c["metrics"]["pck_curve"][t]
                for t in ("0.05", "0.2")
            )
            for c in (m["source_mean"], m["endpoint_clock"])
        )
    )


def run(base, cache, experiment, out):
    if out.exists():
        raise FileExistsError(out)
    started = time.monotonic()
    raw_receipt = json.loads((base / "receipt.json").read_text())
    validate_records(raw_receipt["records"])
    receipt = json.loads((cache / "receipt.json").read_text())
    frozen = json.loads((experiment / "ALL-FOLDS-FROZEN-SELECTIONS.json").read_text())
    original = json.loads((experiment / "results.json").read_text())
    proof = json.loads((experiment / "verification.json").read_text())
    require(
        proof["models_verified"] == 12 and proof["source_input_rehash_passed"],
        "Verified frozen raw-spectrum predictions required",
    )
    with np.load(cache / "metadata.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    paths = [
        base / "receipt.json",
        base / "pose.npy",
        cache / "receipt.json",
        cache / "metadata.npz",
        cache / "natural.npy",
        experiment / "results.json",
        experiment / "verification.json",
        experiment / "ALL-FOLDS-FROZEN-SELECTIONS.json",
    ]
    chosen = {
        room: frozen[room]["natural_selected"]["pose_full"]
        for room in ("E01", "E02", "E03")
    }
    for room, arm in chosen.items():
        require(
            arm in ("natural_linear", "natural_rff"),
            "Frozen family must be natural CURRENTpose",
        )
        paths.extend(
            experiment / (room + "-" + arm + suffix)
            for suffix in ("-map.npz", "-model.npz", "-predictions.npz")
        )
    bindings = {str(p): digest(p) for p in paths}
    require(
        bindings[str(base / "pose.npy")] == raw_receipt["arrays"]["pose"],
        "Source teacher bytes differ",
    )
    for name in ("metadata.npz", "natural.npy"):
        require(
            bindings[str(cache / name)] == receipt["arrays_sha256"][name],
            "Frozen spectral inputs differ",
        )
    sources = {
        str(p.relative_to(ROOT)): digest(p)
        for p in (
            Path(__file__),
            ROOT / "scripts/domain_packet_transfer_probe.py",
            ROOT / "scripts/domain_covariance_source_probe.py",
            ROOT / "scripts/domain_background_probe.py",
            ROOT / "ghostmesh/metrics.py",
        )
    }
    out.mkdir(parents=True)
    save(
        out / "registration.json",
        dict(
            config=CONFIG,
            source_sha256=sources,
            inputs_sha256=bindings,
            frozen_RF_family=chosen,
        ),
    )
    for name in sources:
        dest = out / "executed-source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    labels = np.array(
        np.load(base / "pose.npy", mmap_mode="r", allow_pickle=False), dtype=np.float64
    )
    save(out / "label-geometry.json", label_geometry(labels))
    feature = np.load(cache / "natural.npy", mmap_mode="r", allow_pickle=False)
    targets = {
        stage: coordinates(data["pose_full"], stage)[0] for stage in CONFIG["stages"]
    }
    alignments = {}
    for room, arm in chosen.items():
        train, val, held = fold_masks(data["room"], data["person"], room)
        ids = np.flatnonzero(val)
        with np.load(
            experiment / (room + "-" + arm + "-map.npz"), allow_pickle=False
        ) as z:
            mapping = {k: z[k] for k in z.files}
        with np.load(
            experiment / (room + "-" + arm + "-model.npz"), allow_pickle=False
        ) as z:
            model = {k: z[k] for k in z.files}
        raw = predict(transform(feature[ids], mapping), model)["pose_full"]
        canonical = stage_predictions(raw, "unit_body")
        controls, _ = source_controls(targets["unit_body"], data["end"], train, ids)
        alignments[room] = {
            name: align_source_validation(p, targets["unit_body"][ids])
            for name, p in dict(RF=canonical, **controls).items()
        }
        np.savez_compressed(
            out / (room + "-sourceval-RF.npz"),
            indices=ids,
            raw_prediction=raw,
            unit_body_prediction=canonical,
        )
    # All alignment/mirror choices are frozen before any held-query scores.
    save(out / "ALL-SOURCEVAL-ALIGNMENTS.json", alignments)
    results = {}
    for room, arm in chosen.items():
        train, val, held = fold_masks(data["room"], data["person"], room)
        ids = np.flatnonzero(held)
        with np.load(
            experiment / (room + "-" + arm + "-predictions.npz"), allow_pickle=False
        ) as z:
            require(np.array_equal(z["indices"], ids), "Frozen query rows differ")
            raw = z["pose_full"]
        perm = np.arange(len(ids))
        rng = np.random.default_rng(90210)
        for record in np.unique(data["record"][ids]):
            pick = np.flatnonzero(data["record"][ids] == record)
            perm[pick] = rng.permutation(pick)
        results[room] = {}
        for stage in CONFIG["stages"]:
            controls, fallback = source_controls(
                targets[stage], data["end"], train, ids
            )
            predictions = dict(RF=stage_predictions(raw, stage), **controls)
            modes = {stage: predictions}
            if stage == "unit_body":
                for kind in ("proper", "selected"):
                    modes["unit_body_sourceval_" + kind] = {
                        name: np.einsum(
                            "...j,jk->...k",
                            p,
                            np.array(alignments[room][name][kind]["matrix"]),
                        )
                        for name, p in predictions.items()
                    }
            for mode, items in modes.items():
                rows = {
                    name: score(p, targets[stage][ids], data["record"][ids])
                    for name, p in items.items()
                }
                rows["RF_shuffle"] = score(
                    items["RF"][perm], targets[stage][ids], data["record"][ids]
                )
                for field in ("person", "action"):
                    rows["per_" + field] = {
                        str(key): {
                            name: score(
                                p[pick],
                                targets[stage][ids][pick],
                                data["record"][ids][pick],
                            )
                            for name, p in items.items()
                        }
                        for key in np.unique(data[field][ids])
                        for pick in [data[field][ids] == key]
                    }
                results[room][mode] = dict(
                    methods=rows, endpoint_fallback_ends=fallback
                )
                np.savez_compressed(
                    out / (room + "-" + mode + "-predictions.npz"), indices=ids, **items
                )
        expected = original["selected_natural"][room]["pose_full"]
        observed = results[room]["native"]["methods"]["RF"]["metrics"]
        require(
            abs(expected["normalized_joint_error"] - observed["normalized_joint_error"])
            < 1e-12
            and expected["pck_curve"] == observed["pck_curve"],
            "Original image-plane metrics changed",
        )
    gates = {
        room: canonical_gate(results[room]["unit_body_sourceval_selected"]["methods"])
        for room in chosen
    }
    for path, expected in bindings.items():
        require(digest(Path(path)) == expected, "Evidence changed")
    for path, expected in sources.items():
        require(digest(ROOT / path) == expected, "Source changed")
    save(
        out / "results.json",
        dict(
            config=CONFIG,
            results=results,
            canonical_diagnostic_gates=gates,
            all_three_pass=all(gates.values()),
            original_image_plane_acceptance_satisfied=False,
            seconds=time.monotonic() - started,
        ),
    )
    save(
        out / "artifact-manifest.json",
        {
            str(p.relative_to(out)): digest(p)
            for p in sorted(out.rglob("*"))
            if p.is_file()
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=ROOT / "work/domain-source30-rf-v1"
    )
    parser.add_argument(
        "--cache", type=Path, default=ROOT / "work/domain-packet-full-cache-v1"
    )
    parser.add_argument(
        "--experiment", type=Path, default=ROOT / "work/domain-packet-transfer-v1"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.base, args.cache, args.experiment, args.out)
