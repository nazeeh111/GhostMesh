"""Source-only fine-tuning with invariant annotation-geometry distillation.

Matched four-arm feasibility pilot, fixed final checkpoints. No E02/E04 numeric
data and no RGB enter training or validation. Checkpoints support same-device
sealed recovery; completed experiments and inference outputs are preserved.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.geometric_distillation import GeometricStudent, distillation_objective  # noqa:E402
from scripts import geometric_checkpoint as recovery  # noqa:E402
from scripts import domain_conditioned_dynamics_probe as p  # noqa:E402
from scripts import domain_masked_motion_diagnostic as temporal  # noqa:E402
from scripts import domain_shape_probe as shape  # noqa:E402
from scripts import verify_masked_results as metrics  # noqa:E402
from scripts.domain_masked_local_probe import load_backbone  # noqa:E402
from scripts.domain_masked_pretrain_probe import HistoryReader  # noqa:E402

ARMS = (
    "pose_only",
    "pose_geometry",
    "pose_geometry_relational",
    "pose_geometry_permuted",
)


def validate_config(config):
    for key in (
        "seed",
        "steps",
        "batch_pairs",
        "checkpoint_every",
        "context",
        "lag",
        "permutation_seed",
    ):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError("Nonnegative integer configuration required: " + key)
    if (
        config["schema_version"] != 1
        or config["arms"] != list(ARMS)
        or config["primary"] != ARMS[2]
        or config["context"] != 8
        or config["lag"] != 8
        or not config["steps"]
        or not config["checkpoint_every"]
        or config["steps"] % config["checkpoint_every"]
        or config["batch_pairs"] < 2
        or config["batch_pairs"] % 2
        or config["batch_pairs"] > 16
    ):
        raise ValueError("Unexpected source distillation recipe")
    for key in (
        "encoder_lr",
        "head_lr",
        "gradient_clip",
        "pose_scale_floor",
        "pose_delta_scale_floor",
        "geometry_scale_floor",
        "geometry_delta_scale_floor",
        "teacher_temperature",
        "student_temperature",
    ):
        if (
            isinstance(config[key], bool)
            or not np.isfinite(config[key])
            or config[key] <= 0
        ):
            raise ValueError("Positive finite configuration required: " + key)
    for key in ("weight_decay", "geometry_weight", "relation_weight"):
        if (
            isinstance(config[key], bool)
            or not np.isfinite(config[key])
            or config[key] < 0
        ):
            raise ValueError("Nonnegative finite configuration required: " + key)
    return config


def source_data(study, shape_data, expected):
    if p.digest(shape_data) != expected:
        raise ValueError("Explicit source geometry target hash differs")
    original, train_all, val_all = shape.metadata(study)
    ids = np.flatnonzero(train_all | val_all)
    with np.load(shape_data, allow_pickle=False) as z:
        if set(z.files) != {
            "global_ids",
            "target",
            "train",
            "validation",
            "room",
            "person",
            "record",
            "action",
            "end",
        }:
            raise ValueError("Source geometry dataset fields differ")
        np.testing.assert_array_equal(z["global_ids"], ids)
        for key in ("room", "person", "record", "action", "end"):
            np.testing.assert_array_equal(z[key], original[key][ids])
        for key, mask in (("train", train_all), ("validation", val_all)):
            if z[key].dtype != bool:
                raise ValueError("Geometry source roles must be Boolean")
            np.testing.assert_array_equal(z[key], mask[ids])
        geometry = p.real_finite(
            z["target"], "source geometry", (len(ids), 136)
        ).astype(np.float32)
    pose = p.real_finite(
        p.stored_npz_array(study / "targets-and-metadata.npz", "pose")[ids],
        "source body poses",
        (len(ids), 17, 2),
    ).astype(np.float32)
    data = {key: value[ids] for key, value in original.items()}
    data.update(
        global_ids=ids,
        train=train_all[ids],
        validation=val_all[ids],
        pose=pose,
        geometry=geometry,
    )
    if (data["train"].sum(), data["validation"].sum()) != (14124, 3546):
        raise ValueError("Source person/endpoint cohort differs")
    return data, original, train_all, val_all


def source_pairs(data):
    ids = np.flatnonzero(data["train"])
    if (
        not np.isin(data["room"][ids], ["E01", "E03"]).all()
        or np.isin(data["person"][ids], ["S04", "S09", "S24", "S29"]).any()
    ):
        raise ValueError("Source TRAIN pairs required")
    current, previous = p.pair_indices(data["record"][ids], data["end"][ids])
    return ids[current], ids[previous]


def make_schedule(data, config):
    current, previous = source_pairs(data)
    rng = np.random.default_rng(config["seed"])
    by_room = {
        room: sorted(set(data["person"][current][data["room"][current] == room]))
        for room in ("E01", "E03")
    }
    if any(len(people) != 8 for people in by_room.values()):
        raise ValueError("Exactly eight TRAIN people per source room required")
    by_person = {
        person: np.flatnonzero(data["person"][current] == person)
        for people in by_room.values()
        for person in people
    }
    batches = []
    for _ in range(config["steps"]):
        chosen = []
        for room in ("E01", "E03"):
            for person in rng.choice(
                by_room[room], config["batch_pairs"] // 2, replace=False
            ):
                chosen.append(int(rng.choice(by_person[str(person)])))
        batches.append(np.stack([current[chosen], previous[chosen]], axis=-1))
    return np.stack(batches)


def source_permutation(data, seed):
    result = np.full(len(data["train"]), -1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    for record in sorted(set(data["record"][data["train"]])):
        ids = np.flatnonzero(data["train"] & (data["record"] == record))
        result[ids] = rng.permutation(ids)
    return result


class TrainingLabels:
    def __init__(self, data, permutation):
        permutation = np.asarray(permutation)
        ids = np.flatnonzero(data["train"])
        if (
            permutation.shape != data["train"].shape
            or permutation.dtype.kind not in "iu"
            or not len(ids)
            or (permutation[ids] < 0).any()
            or (permutation[ids] >= len(permutation)).any()
            or not np.array_equal(np.sort(permutation[ids]), ids)
            or not np.array_equal(data["record"][permutation[ids]], data["record"][ids])
            or np.any(permutation[~data["train"]] != -1)
        ):
            raise ValueError(
                "Geometry permutation must bijectively preserve each TRAIN record"
            )
        self.data, self.permutation = data, permutation

    def __call__(self, rows, arm):
        rows = np.asarray(rows)
        if (
            rows.ndim != 1
            or rows.dtype.kind not in "iu"
            or not len(rows)
            or rows.min() < 0
            or rows.max() >= len(self.data["train"])
            or not self.data["train"][rows].all()
        ):
            raise ValueError("Optimizer label access outside explicit source TRAIN")
        if arm not in ARMS:
            raise ValueError("Unknown source distillation arm")
        geometry_ids = self.permutation[rows] if arm == ARMS[3] else rows
        if (
            (geometry_ids < 0).any()
            or not self.data["train"][geometry_ids].all()
            or not np.array_equal(
                self.data["record"][geometry_ids], self.data["record"][rows]
            )
        ):
            raise ValueError("Geometry permutation escaped TRAIN record")
        return self.data["pose"][rows], self.data["geometry"][geometry_ids]


def statistics(data, permutation, config):
    TrainingLabels(data, permutation)  # Validate label scope before target reads.
    ids = np.flatnonzero(data["train"])
    current, previous = source_pairs(data)
    pose = p.real_finite(data["pose"][ids], "TRAIN pose")
    geometry = p.real_finite(data["geometry"][ids], "TRAIN geometry")

    def rms(value, floor):
        return np.maximum(np.sqrt(np.square(value).mean(0)), floor).astype(np.float32)

    return dict(
        pose_mean=pose.mean(0).astype(np.float32),
        pose_scale=np.maximum(pose.std(0), config["pose_scale_floor"]).astype(
            np.float32
        ),
        geometry_mean=geometry.mean(0).astype(np.float32),
        geometry_scale=np.maximum(
            geometry.std(0), config["geometry_scale_floor"]
        ).astype(np.float32),
        pose_delta_scale=rms(
            data["pose"][current].astype(float) - data["pose"][previous],
            config["pose_delta_scale_floor"],
        ),
        geometry_delta_scale=rms(
            data["geometry"][current].astype(float) - data["geometry"][previous],
            config["geometry_delta_scale_floor"],
        ),
        permuted_geometry_delta_scale=rms(
            data["geometry"][permutation[current]].astype(float)
            - data["geometry"][permutation[previous]],
            config["geometry_delta_scale_floor"],
        ),
    )


def make_model(source, stats, device):
    return GeometricStudent(
        load_backbone(source / "masked_temporal-final.pt"),
        *[
            torch.from_numpy(stats[key])
            for key in ("pose_mean", "pose_scale", "geometry_mean", "geometry_scale")
        ],
    ).to(device)


def make_optimizer(model, config):
    return torch.optim.AdamW(
        [
            dict(params=list(model.encoder.parameters()), lr=config["encoder_lr"]),
            dict(
                params=[
                    *model.pose_head.parameters(),
                    *model.geometry_head.parameters(),
                    *model.relation_head.parameters(),
                ],
                lr=config["head_lr"],
            ),
        ],
        weight_decay=config["weight_decay"],
    )


def train_step(
    model, optimizer, histories, pose_target, geometry_target, rooms, stats, config, arm
):
    result = model(histories)
    loss = distillation_objective(
        result,
        pose_target,
        geometry_target,
        stats["pose_delta_scale"],
        stats["permuted_geometry_delta_scale"]
        if arm == ARMS[3]
        else stats["geometry_delta_scale"],
        model.pose_scale,
        model.geometry_mean,
        model.geometry_scale,
        rooms,
        arm,
        geometry_weight=config["geometry_weight"],
        relation_weight=config["relation_weight"],
        teacher_temperature=config["teacher_temperature"],
        student_temperature=config["student_temperature"],
    )
    if not bool(torch.isfinite(loss)):
        raise ValueError("Nonfinite loss before optimizer update")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), config["gradient_clip"], error_if_nonfinite=True
    )
    optimizer.step()
    return loss.detach()


def evaluation(model, reader, data, device):
    ids = np.flatnonzero(data["validation"])
    poses = []
    geometries = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(ids), 16):
            rows = ids[start : start + 16]
            prediction = model(
                torch.from_numpy(reader(data["global_ids"][rows])).to(device)
            )
            poses.append(prediction["pose"].cpu().numpy())
            geometries.append(prediction["geometry"].cpu().numpy())
    pose = np.concatenate(poses)
    geometry = np.concatenate(geometries)
    report = measure(pose, geometry, data, ids)
    return pose, geometry, report


def measure(pose, geometry, data, ids):
    def group(rows):
        return dict(
            pose=metrics.metric(
                pose[rows], data["pose"][ids[rows]], data["record"][ids[rows]], "body"
            ),
            temporal=temporal.motion_scores(
                pose[rows],
                data["pose"][ids[rows]],
                data["record"][ids[rows]],
                data["end"][ids[rows]],
            ),
            geometry=shape.score(
                geometry[rows],
                data["geometry"][ids[rows]],
                data["record"][ids[rows]],
                data["end"][ids[rows]],
            ),
        )

    result = dict(pooled=group(np.arange(len(ids))))
    for field in ("room", "person", "action"):
        labels = data[field][ids]
        result["per_" + field] = {
            str(name): group(np.flatnonzero(labels == name))
            for name in sorted(set(labels))
        }
    perm = metrics.permutation(data["record"][ids], 90210)
    result["shuffled_pose"] = metrics.metric(
        pose[perm], data["pose"][ids], data["record"][ids], "body"
    )
    return result


def decide_gate(reports, controls):
    primary = reports[ARMS[2]]
    po = primary["pooled"]["pose"]
    motion = primary["pooled"]["temporal"]
    baselines = [
        r["pooled"]["pose"] for arm, r in reports.items() if arm != ARMS[2]
    ] + list(controls.values())

    def above(value, threshold=0):
        return value is not None and np.isfinite(value) and value > threshold

    return dict(
        body_motion_useful=above(po["record_centered_articulation_r2"], 0.05),
        delta_useful=above(motion["lag8_zero_reference_delta_r2"], 0.05),
        both_rooms_positive=set(primary["per_room"]) == {"E01", "E03"}
        and all(
            above(row["pose"]["record_centered_articulation_r2"])
            for row in primary["per_room"].values()
        ),
        three_of_four_people_positive=set(primary["per_person"])
        == {"S04", "S09", "S24", "S29"}
        and sum(
            above(row["pose"]["record_centered_articulation_r2"])
            for row in primary["per_person"].values()
        )
        >= 3,
        nje_beats_controls=all(
            po["normalized_joint_error"] < row["normalized_joint_error"]
            for row in baselines
        ),
        pck_not_worse=all(
            po["pck_curve"][k] >= row["pck_curve"][k]
            for row in baselines
            for k in ("0.05", "0.2")
        ),
        shuffle_worse=po["normalized_joint_error"]
        < primary["shuffled_pose"]["normalized_joint_error"],
    )


def run(
    source,
    raw,
    shape_data,
    expected_shape,
    config_path,
    plan_path,
    out,
    device,
    resume=False,
):
    config = validate_config(json.loads(config_path.read_text()))
    if not resume and (out.exists() or out.is_symlink()):
        raise FileExistsError(out)
    if any(
        out.resolve().is_relative_to(path.resolve())
        for path in (source, raw, shape_data.parent)
    ):
        raise ValueError("Output must be outside frozen inputs")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested native MPS unavailable")
    data, original, train_all, val_all = source_data(source, shape_data, expected_shape)
    producer = json.loads((source / "registration.json").read_text())
    raw_receipt = json.loads((raw / "receipt.json").read_text())
    if p.digest(raw / "receipt.json") != producer["raw_receipt_sha256"]:
        raise ValueError("Raw receipt identity differs")
    p.verify_hashes(raw, {"rf.npy": raw_receipt["arrays"]["rf"]})
    paths = [
        Path(__file__),
        ROOT / "ghostmesh/geometric_distillation.py",
        ROOT / "scripts/geometric_checkpoint.py",
        ROOT / "tests/test_geometric_distillation.py",
        ROOT / "scripts/domain_shape_probe.py",
        ROOT / "ghostmesh/pose_shape.py",
        ROOT / "scripts/domain_masked_local_probe.py",
        ROOT / "scripts/domain_masked_pretrain_probe.py",
        ROOT / "ghostmesh/masked_csi.py",
        ROOT / "scripts/domain_masked_motion_diagnostic.py",
        ROOT / "scripts/domain_conditioned_dynamics_probe.py",
        ROOT / "scripts/verify_masked_results.py",
        ROOT / "scripts/verify_masked_calibration_probe.py",
        ROOT / "scripts/verify_masked_fullpose_probe.py",
        ROOT / "scripts/verify_masked_pretrain_probe.py",
        ROOT / "scripts/research_numeric_readback.py",
    ]
    registration = dict(
        kind="source-only-canonical-geometry-distillation",
        config=config,
        config_sha256=p.digest(config_path),
        plan_sha256=p.digest(plan_path),
        source_ready_sha256=p.EXPECTED_READY,
        shape_data_sha256=expected_shape,
        raw_receipt_sha256=p.digest(raw / "receipt.json"),
        init_encoder_sha256=p.digest(source / "masked_temporal-final.pt"),
        source_sha256={str(path.relative_to(ROOT)): p.digest(path) for path in paths},
        device=device,
        torch=str(torch.__version__),
        numpy=np.__version__,
        E02_numeric_values_read=False,
        annotation_limit="Geometry labels are extra source supervision vs pose-only. No query annotation or room/person/action metadata enters model.forward",
    )
    out.mkdir(parents=True, exist_ok=resume)
    if resume:
        if json.loads((out / "registration.json").read_text()) != registration:
            raise ValueError("Recovery source/data/config/device identity differs")
        if (out / "results.json").exists():
            raise FileExistsError("Completed experiment preserved")
    else:
        p.save_json(out / "registration.json", registration)
        shutil.copyfile(config_path, out / "config.json")
        shutil.copyfile(plan_path, out / "prospective-plan.json")
        for path in paths:
            destination = out / "executed-source" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    lock = out / "writer.lock"
    if lock.exists():
        pid = json.loads(lock.read_text())["pid"]
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            if not resume:
                raise ValueError("Stale writer requires explicit recovery")
            shutil.copyfile(lock, out / f"prior-writer-{pid}.json")
            lock.unlink()
        else:
            raise ValueError("A writer with the recorded PID is still present")
    with lock.open("x") as stream:
        json.dump(dict(pid=os.getpid()), stream)
    started = time.monotonic()
    torch.set_num_threads(1)
    try:
        schedule = make_schedule(data, config)
        perm = source_permutation(data, config["permutation_seed"])
        stats = statistics(data, perm, config)
        if resume:
            np.testing.assert_array_equal(
                np.load(out / "schedule.npy", allow_pickle=False), schedule
            )
            np.testing.assert_array_equal(
                np.load(out / "geometry-permutation.npy", allow_pickle=False), perm
            )
            with np.load(out / "TRAIN-statistics.npz", allow_pickle=False) as z:
                for key, value in stats.items():
                    np.testing.assert_array_equal(z[key], value)
        else:
            np.save(out / "schedule.npy", schedule, allow_pickle=False)
            np.save(out / "geometry-permutation.npy", perm, allow_pickle=False)
            p.save_npz(out / "TRAIN-statistics.npz", **stats)
            p.save_npz(out / "source-data.npz", **data)
        rf = np.load(raw / "rf.npy", mmap_mode="r", allow_pickle=False)
        train_reader = HistoryReader(rf, original, raw_receipt["records"], train_all)
        val_reader = HistoryReader(rf, original, raw_receipt["records"], val_all)
        labels = TrainingLabels(data, perm)
        torch.manual_seed(config["seed"])
        initial = make_model(source, stats, "cpu")
        if resume:
            saved = torch.load(
                out / "initial-model.pt", map_location="cpu", weights_only=True
            )
            if set(saved) != set(initial.state_dict()) or any(
                not torch.equal(saved[key], value)
                for key, value in initial.state_dict().items()
            ):
                raise ValueError("Initial state no longer reproduces")
        else:
            torch.save(initial.state_dict(), out / "initial-model.pt")
        identity = (
            p.digest(out / "registration.json")
            + ":"
            + p.digest(out / "schedule.npy")
            + ":"
            + p.digest(out / "TRAIN-statistics.npz")
            + ":"
            + p.digest(out / "geometry-permutation.npy")
            + ":"
            + p.digest(out / "initial-model.pt")
        )
        tensor_stats = {
            key: torch.from_numpy(value).to(device) for key, value in stats.items()
        }
        training = []
        for arm in ARMS:
            arm_dir = out / arm
            arm_dir.mkdir(exist_ok=resume)
            torch.manual_seed(config["seed"])
            model = make_model(source, stats, device)
            model.load_state_dict(initial.state_dict(), strict=True)
            optimizer = make_optimizer(model, config)
            step = (
                recovery.load(arm_dir, model, optimizer, identity, arm, device)
                if resume
                else 0
            )
            if step > config["steps"]:
                raise ValueError("Recovery exceeds frozen optimizer budget")
            for index in range(step, config["steps"]):
                pairs = schedule[index]
                rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
                y, g = labels(rows, arm)
                x = torch.from_numpy(train_reader(data["global_ids"][rows])).to(device)
                rooms = torch.from_numpy(
                    (data["room"][rows] == "E03").astype(np.int64)
                ).to(device)
                model.train()
                loss = train_step(
                    model,
                    optimizer,
                    x,
                    torch.from_numpy(y).to(device),
                    torch.from_numpy(g).to(device),
                    rooms,
                    tensor_stats,
                    config,
                    arm,
                )
                if (index + 1) % config["checkpoint_every"] == 0:
                    recovery.save(
                        arm_dir, model, optimizer, index + 1, identity, arm, device
                    )
                    p.save_json(
                        arm_dir / f"loss-{index + 1:04}.json",
                        dict(step=index + 1, training_loss=float(loss.cpu())),
                    )
                    print(
                        json.dumps(
                            dict(
                                arm=arm,
                                step=index + 1,
                                loss=float(loss.cpu()),
                                run_seconds=time.monotonic() - started,
                            )
                        ),
                        flush=True,
                    )
            complete = dict(
                step=config["steps"],
                identity=identity,
                final_checkpoint_sha256=p.digest(
                    arm_dir / f"step-{config['steps']:04}.pt"
                ),
            )
            if (arm_dir / "complete.json").exists():
                if json.loads((arm_dir / "complete.json").read_text()) != complete:
                    raise ValueError("Completed-arm identity differs")
            else:
                p.save_json(arm_dir / "complete.json", complete)
            training.append(arm)
            del model, optimizer
        # All four fixed-final models exist before the first validation score.
        attempt = 0
        while (out / f"evaluation-attempt-{attempt:02}").exists():
            attempt += 1
        evaluation_dir = out / f"evaluation-attempt-{attempt:02}"
        evaluation_dir.mkdir()
        reports = {}
        for arm in training:
            model = make_model(source, stats, device)
            optimizer = make_optimizer(model, config)
            if (
                recovery.load(out / arm, model, optimizer, identity, arm, device)
                != config["steps"]
            ):
                raise ValueError("Final source checkpoint missing")
            pose, geometry, report = evaluation(model, val_reader, data, device)
            np.save(evaluation_dir / f"{arm}-VALpose.npy", pose, allow_pickle=False)
            np.save(
                evaluation_dir / f"{arm}-VALgeometry.npy", geometry, allow_pickle=False
            )
            reports[arm] = report
            p.save_json(evaluation_dir / f"{arm}-source-results.json", report)
            print(
                json.dumps(dict(arm=arm, source_validation=report["pooled"])),
                flush=True,
            )
            del model, optimizer
        ids = np.flatnonzero(data["validation"])
        target = data["pose"]
        tr = data["train"]
        clock = {
            int(e): target[tr & (data["end"] == e)].mean(0)
            for e in set(data["end"][tr])
        }
        controls = {
            name: metrics.metric(prediction, target[ids], data["record"][ids], "body")
            for name, prediction in {
                "source_mean": np.broadcast_to(target[tr].mean(0), target[ids].shape),
                "source_clock": np.stack([clock[int(e)] for e in data["end"][ids]]),
            }.items()
        }
        gate = decide_gate(reports, controls)
        p.verify_hashes(ROOT, registration["source_sha256"])
        if (
            p.digest(config_path) != registration["config_sha256"]
            or p.digest(shape_data) != expected_shape
        ):
            raise ValueError("Source configuration/data changed while fitting")
        p.verify_hashes(raw, {"rf.npy": raw_receipt["arrays"]["rf"]})
        if (
            p.digest(source / "masked_temporal-final.pt")
            != registration["init_encoder_sha256"]
        ):
            raise ValueError("Warm-start encoder changed on disk")
        p.save_json(
            out / "results.json",
            dict(
                registration=registration,
                arms=reports,
                no_rf_controls=controls,
                gate=gate,
                passes=all(gate.values()),
                invocation_seconds=time.monotonic() - started,
                resumed=resume,
                evaluation_directory=evaluation_dir.name,
                scope="SOURCE-onlyhip-relative2Dfeasibility, no newly scoredE02/E04/3D or calibrated-product claim",
            ),
        )
        p.save_json(
            out / "artifact-manifest.json",
            {
                str(path.relative_to(out)): p.digest(path)
                for path in out.rglob("*")
                if path.is_file() and path != lock
            },
        )
    finally:
        if lock.exists() and json.loads(lock.read_text())["pid"] == os.getpid():
            lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "raw", "shape-data", "expected-shape-sha256", "plan", "out"):
        parser.add_argument(
            "--" + name,
            type=str if name == "expected-shape-sha256" else Path,
            required=True,
        )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/geometric-distillation-source-v1.json",
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(
        args.source,
        args.raw,
        args.shape_data,
        args.expected_shape_sha256,
        args.config,
        args.plan,
        args.out,
        args.device,
        args.resume,
    )
