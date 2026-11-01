"""Dense two-source-room prototype learning with cross-room pose similarity.

No held-room annotation reader or query scorer is implemented. All fixed
training budgets and source predictions finish before source-validation scores.
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
from ghostmesh.cross_room_pose import cross_room_alignment  # noqa:E402
from ghostmesh.episodic_calibration import fit_residual, apply_residual  # noqa:E402
from scripts import domain_pose_alignment_probe as base  # noqa:E402
from scripts import dense_room_data as dataset  # noqa:E402
from scripts import geometric_checkpoint as recovery  # noqa:E402
from scripts.domain_amplitude_probe import select  # noqa:E402

p = base.p
ARMS = ("ordinary", "cross_room", "teacher_permuted")


def validate_config(cfg):
    p.require(
        cfg["schema_version"] == 1
        and cfg["source_rooms"] == ["E01", "E02"]
        and cfg["held_room"] == "E03"
        and cfg["source_validation_people"] == list(dataset.VAL_PEOPLE)
        and cfg["arms"] == list(ARMS)
        and cfg["primary"] == "cross_room",
        "Frozen two-source-room recipe required",
    )
    p.require(
        (cfg["expected_source_train"], cfg["expected_source_validation"])
        == (108956, 3435)
        and (cfg["TRAIN_endpoint_stride"], cfg["VAL_endpoint_stride"]) == (1, 8),
        "DenseTRAIN/stridedVAL roles differ",
    )
    p.require(
        type(cfg["seed"]) is int
        and cfg["seed"] >= 0
        and type(cfg["steps"]) is int
        and 0 < cfg["steps"] < 10000
        and type(cfg["checkpoint_every"]) is int
        and cfg["checkpoint_every"] > 0
        and cfg["steps"] % cfg["checkpoint_every"] == 0,
        "Finite recoverable budget required",
    )
    for key in [
        "learning_rate",
        "alignment_weight",
        "ridge",
        "pose_scale_floor",
        "delta_scale_floor",
    ]:
        p.require(
            not isinstance(cfg[key], bool) and np.isfinite(cfg[key]) and cfg[key] > 0,
            "Positive finite parameter required",
        )
    p.require(
        not isinstance(cfg["weight_decay"], bool)
        and np.isfinite(cfg["weight_decay"])
        and cfg["weight_decay"] >= 0,
        "Nonnegative decay required",
    )
    p.require(
        cfg["source_reference_people"] == {"E01": ["S01", "S02"], "E02": ["S11", "S12"]}
        and cfg["reference_actions"] == ["A03", "A07", "A12"]
        and cfg["references_per_record"] == 20
        and cfg["source_reference_rows_per_room"] == 120
        and cfg["source_reference_rows_total"] == 240
        and cfg["reference_count_convention"] == 24,
        "Matched120perroomreference contract required",
    )
    return cfg


def measure(pred, data, ids):
    result = p.measure(pred, data, ids)
    result["per_room"] = {
        str(room): p.measure(
            pred[data["room"][ids] == room], data, ids[data["room"][ids] == room]
        )["pooled"]
        for room in sorted(set(data["room"][ids]))
    }
    return result


def gate(primary, controls):
    po = primary["pooled"]

    def positive(value):
        return value["motion_r2"] is not None and value["motion_r2"] > 0

    return dict(
        body_motion_useful=po["motion_r2"] is not None and po["motion_r2"] > 0.05,
        delta_useful=po["delta_r2"] is not None and po["delta_r2"] > 0.05,
        both_source_rooms_positive=set(primary["per_room"]) == {"E01", "E02"}
        and all(positive(v) for v in primary["per_room"].values()),
        three_of_four_people_positive=set(primary["per_person"])
        == set(dataset.VAL_PEOPLE)
        and sum(positive(v) for v in primary["per_person"].values()) >= 3,
        nje_beats_controls=all(po["nje"] < v["pooled"]["nje"] for v in controls),
        pck_not_worse=all(
            po[k] >= v["pooled"][k] for v in controls for k in ["pck05", "pck20"]
        ),
        shuffle_worse=po["nje"] < primary["shuffled"]["nje"],
    )


def train_step(model, opt, x, target, rooms, stats, arm, cfg):
    p.require(
        arm in ARMS and len(x) == len(target) == len(rooms) == 32,
        "Known32history batch required",
    )
    result = model(x)
    scale = torch.as_tensor(stats["pose_scale"], device=x.device)
    delta = torch.as_tensor(stats["pose_delta_scale"], device=x.device)
    current = torch.arange(0, 32, 2, device=x.device)
    past = current + 1
    pose = (
        0.5 * ((result["pose"] - target) / scale).square().mean()
        + 0.5
        * (
            (
                (result["pose"][current] - result["pose"][past])
                - (target[current] - target[past])
            )
            / delta
        )
        .square()
        .mean()
    )
    permutation = (
        torch.cat(
            (
                torch.arange(2, 16, device=x.device),
                torch.arange(2, device=x.device),
                torch.arange(18, 32, device=x.device),
                torch.arange(16, 18, device=x.device),
            )
        )
        if arm == "teacher_permuted"
        else None
    )
    alignment = cross_room_alignment(
        result["code"], target, rooms, scale, permutation=permutation
    )
    loss = pose + (cfg["alignment_weight"] * alignment if arm != "ordinary" else 0)
    p.require(bool(torch.isfinite(loss)), "Nonfiniteobjective beforeupdate")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    opt.step()
    return dict(
        loss=float(loss.detach().cpu()),
        pose_loss=float(pose.detach().cpu()),
        alignment=float(alignment.detach().cpu()),
    )


def calibrate(encoded, target, ref_room, query_room, cfg):
    p.require(
        set(ref_room) == set(query_room) == {"E01", "E02"}
        and target.shape == (240, 17, 2)
        and len(encoded["reference_code"]) == len(ref_room) == 240
        and len(encoded["query_code"]) == len(query_room),
        "Explicit two-source-room reference/query roles required",
    )
    output = np.empty_like(encoded["query_pose"])
    states = {}
    for room in ["E01", "E02"]:
        r = ref_room == room
        q = query_room == room
        p.require(r.sum() == 120 and q.any(), "Roomreference/query scope differs")
        state = fit_residual(
            torch.from_numpy(encoded["reference_code"][r]),
            torch.from_numpy(target[r]).flatten(1),
            torch.from_numpy(encoded["reference_pose"][r]).flatten(1),
            ridge=cfg["ridge"],
            reference_count=cfg["reference_count_convention"],
        )
        pred = (
            apply_residual(
                state,
                torch.from_numpy(encoded["query_code"][q]),
                torch.from_numpy(encoded["query_pose"][q]).flatten(1),
            )
            .reshape(-1, 17, 2)
            .numpy()
        )
        pred -= pred[:, [11, 12]].mean(1, keepdims=True)
        output[q] = pred
        states[room] = dict(
            code_mean=state.code_mean.detach().numpy(),
            correction_mean=state.correction_mean.detach().numpy(),
            weights=state.weights.detach().numpy(),
        )
    return output, states


def run(cache, raw, config, out, device, resume=False):
    cfg = validate_config(p.read(config))
    p.require(resume or not out.exists(), "Existingstudy preserved")
    p.require(
        not any(out.resolve().is_relative_to(v.resolve()) for v in [cache, raw]),
        "Outputinsideinputs",
    )
    p.require(
        device == "cpu" or torch.backends.mps.is_available(), "NativeMPSunavailable"
    )
    data, feature, receipt = dataset.load_source(cache, raw)
    paths = [
        Path(__file__),
        ROOT / "scripts/dense_room_data.py",
        ROOT / "ghostmesh/cross_room_pose.py",
        ROOT / "ghostmesh/pose_alignment.py",
        ROOT / "scripts/domain_pose_alignment_probe.py",
        ROOT / "scripts/domain_episodic_probe.py",
        ROOT / "scripts/domain_amplitude_probe.py",
        ROOT / "ghostmesh/episodic_calibration.py",
        ROOT / "scripts/geometric_checkpoint.py",
        ROOT / "scripts/verify_geometric_distillation.py",
    ]
    registration = dict(
        kind="dense-E01E02-cross-room-pose-source",
        config=cfg,
        config_sha256=p.digest(config),
        cache_receipt_sha256=p.digest(cache / "receipt.json"),
        raw_receipt_sha256=p.digest(raw / "receipt.json"),
        source_sha256={str(q.relative_to(ROOT)): p.digest(q) for q in paths},
        device=device,
        torch=str(torch.__version__),
        numpy=np.__version__,
        held_annotation_values_decoded=False,
        output_delay_frames=4,
    )
    out.mkdir(parents=True, exist_ok=resume)
    if resume:
        p.require(
            p.read(out / "registration.json") == registration
            and not (out / "source-results.json").exists(),
            "Recoveryidentitydiffers orcomplete",
        )
    else:
        p.save(out / "registration.json", registration)
        shutil.copyfile(config, out / "config.json")
        for q in paths:
            dest = out / "executed-source" / q.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(q, dest)
    lock = out / "writer.lock"
    if lock.exists():
        pid = p.read(lock)["pid"]
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            p.require(resume, "Explicit recoveryrequired")
            shutil.copyfile(lock, out / f"prior-writer-{pid}.json")
            lock.unlink()
        else:
            raise ValueError("Existinglivewriter")
    p.save(lock, dict(pid=os.getpid()))
    started = time.monotonic()
    torch.set_num_threads(1)
    try:
        train = base.Reader(feature, data["train"])
        val = base.Reader(feature, data["validation"])
        schedule = dataset.make_schedule(data, cfg)
        stats = dataset.statistics(data, feature, cfg)
        if resume:
            np.testing.assert_array_equal(
                np.load(out / "schedule.npy", allow_pickle=False), schedule
            )
            for k, v in stats.items():
                np.testing.assert_array_equal(
                    p.arrays(out / "TRAIN-statistics.npz")[k], v
                )
        else:
            np.save(out / "schedule.npy", schedule, allow_pickle=False)
            np.savez(out / "TRAIN-statistics.npz", **stats)
            np.savez(out / "source-data.npz", **data)
        torch.manual_seed(cfg["seed"])
        initial = base.make_model(stats, "cpu")
        if resume:
            saved = torch.load(
                out / "initial-model.pt", map_location="cpu", weights_only=True
            )
            p.require(
                set(saved) == set(initial.state_dict())
                and all(
                    torch.equal(v, saved[k]) for k, v in initial.state_dict().items()
                ),
                "Initialmodeldiffers",
            )
        else:
            torch.save(initial.state_dict(), out / "initial-model.pt")
        identity = ":".join(
            p.digest(out / name)
            for name in [
                "registration.json",
                "schedule.npy",
                "TRAIN-statistics.npz",
                "initial-model.pt",
            ]
        )
        for arm in ARMS:
            folder = out / arm
            folder.mkdir(exist_ok=resume)
            net = base.make_model(stats, device)
            net.load_state_dict(initial.state_dict(), strict=True)
            opt = base.optimizer(net, cfg)
            step = (
                recovery.load(folder, net, opt, identity, arm, device) if resume else 0
            )
            p.require(step <= cfg["steps"], "Budgetexceeded")
            if not resume:
                recovery.save(folder, net, opt, 0, identity, arm, device)
            for index in range(step, cfg["steps"]):
                rows = schedule[index]
                target = p.training_labels(data, rows)
                rooms = torch.from_numpy(
                    (data["room"][rows] == "E02").astype(np.int64)
                ).to(device)
                result = train_step(
                    net,
                    opt,
                    torch.from_numpy(train(rows)).to(device),
                    torch.from_numpy(target).to(device),
                    rooms,
                    stats,
                    arm,
                    cfg,
                )
                if (index + 1) % cfg["checkpoint_every"] == 0:
                    recovery.save(folder, net, opt, index + 1, identity, arm, device)
                    print(
                        json.dumps(
                            dict(
                                arm=arm,
                                step=index + 1,
                                elapsed=time.monotonic() - started,
                                **result,
                            )
                        ),
                        flush=True,
                    )
            complete = dict(
                step=cfg["steps"],
                identity=identity,
                checkpoint_sha256=p.digest(folder / f"step-{cfg['steps']:04}.pt"),
            )
            if (folder / "complete.json").exists():
                p.require(
                    p.read(folder / "complete.json") == complete,
                    "Completedidentitydiffers",
                )
            else:
                p.save(folder / "complete.json", complete)
            del net, opt
        # Fixed trainingbudgets complete before sourceVAL selection.
        attempt = 0
        while (out / f"source-selection-{attempt:02}").exists():
            attempt += 1
        evaluation = out / f"source-selection-{attempt:02}"
        evaluation.mkdir()
        refs = np.concatenate(
            [dataset.references(data, room) for room in ["E01", "E02"]]
        )
        ids = np.flatnonzero(data["validation"])
        target = p.training_labels(data, refs)
        ref_room = data["room"][refs]
        query_room = data["room"][ids]
        np.save(evaluation / "reference-rows.npy", refs, allow_pickle=False)
        for arm in ARMS:
            for step in range(0, cfg["steps"] + 1, cfg["checkpoint_every"]):
                checkpoint = out / arm / f"step-{step:04}.pt"
                seal = p.read(checkpoint.with_name(checkpoint.name + ".json"))
                p.require(
                    seal
                    == dict(
                        kind="source-geometric-recovery",
                        step=step,
                        identity=identity,
                        arm=arm,
                        device=device,
                        checkpoint_sha256=p.digest(checkpoint),
                    ),
                    "Candidatesealdiffers",
                )
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                p.require(
                    payload["step"] == step and payload["identity"] == identity,
                    "Candidateidentitydiffers",
                )
                net = base.make_model(stats, device)
                net.load_state_dict(payload["model"], strict=True)
                rp, rc = base.encode(net, train, refs, device)
                qp, qc = base.encode(net, val, ids, device)
                encoded = dict(
                    reference_pose=rp, reference_code=rc, query_pose=qp, query_code=qc
                )
                adapted, states = calibrate(encoded, target, ref_room, query_room, cfg)
                stem = f"{arm}-{step:04}"
                np.savez(evaluation / (stem + "-encoded.npz"), **encoded)
                np.save(evaluation / (stem + "-VALzero.npy"), qp, allow_pickle=False)
                np.save(
                    evaluation / (stem + "-VALcal.npy"), adapted, allow_pickle=False
                )
                for room, state in states.items():
                    np.savez(
                        evaluation / (stem + "-" + room + "-reference-state.npz"),
                        **state,
                    )
                del net
        raw_predictions = {
            kind: np.empty((len(ids), 17, 2), dtype=float) for kind in ["linear", "rbf"]
        }
        ref_mean = np.empty_like(raw_predictions["linear"])
        CV = {}
        for room in ["E01", "E02"]:
            r = ref_room == room
            q = query_room == room
            rx = (
                (train(refs[r]).astype(float) - stats["input_mean"])
                / stats["input_scale"]
            ).reshape(120, -1)
            qx = (
                (val(ids[q]).astype(float) - stats["input_mean"]) / stats["input_scale"]
            ).reshape(q.sum(), -1)
            pred, states, grids = p.raw_controls(
                rx, target[r].astype(float), data["person"][refs[r]], qx
            )
            CV[room] = grids
            ref_mean[q] = target[r].mean(0)
            for kind in ["linear", "rbf"]:
                raw_predictions[kind][q] = pred[kind]
                np.savez(evaluation / f"raw-{kind}-{room}-state.npz", **states[kind])
        for kind, pred in raw_predictions.items():
            np.save(evaluation / f"raw-{kind}-VAL.npy", pred, allow_pickle=False)
        np.save(evaluation / "reference-mean-VAL.npy", ref_mean, allow_pickle=False)
        p.save(evaluation / "raw-reference-cv.json", CV)
        manifest = {
            str(q.relative_to(evaluation)): p.digest(q)
            for q in evaluation.iterdir()
            if q.is_file()
        }
        p.save(
            evaluation / "predictions-ready.json",
            dict(
                source_validation_scored=False,
                held_metrics_computed=False,
                artifacts_sha256=manifest,
            ),
        )
        candidates = {}
        selected = {}
        zero = {}
        cal = {}
        for arm in ARMS:
            candidates[arm] = []
            for step in range(0, cfg["steps"] + 1, cfg["checkpoint_every"]):
                stem = f"{arm}-{step:04}"
                a = np.load(evaluation / (stem + "-VALzero.npy"), allow_pickle=False)
                b = np.load(evaluation / (stem + "-VALcal.npy"), allow_pickle=False)
                candidates[arm].append(
                    dict(
                        step=step,
                        zero=measure(a, data, ids),
                        calibrated=measure(b, data, ids),
                    )
                )
            a, b = (
                select(candidates[arm], "zero"),
                select(candidates[arm], "calibrated"),
            )
            selected[arm] = dict(zero_step=a["step"], calibrated_step=b["step"])
            zero[arm] = a["zero"]
            cal[arm] = b["calibrated"]
        tr = np.flatnonzero(data["train"])
        zero["source_mean"] = measure(
            np.broadcast_to(data["pose"][tr].mean(0), data["pose"][ids].shape),
            data,
            ids,
        )
        clock = {
            int(e): data["pose"][data["train"] & (data["end"] == e)].mean(0)
            for e in set(data["end"][tr])
        }
        zero["source_clock"] = measure(
            np.stack([clock[int(e)] for e in data["end"][ids]]), data, ids
        )
        room_mean = {
            room: data["pose"][data["train"] & (data["room"] == room)].mean(0)
            for room in ["E01", "E02"]
        }
        zero["oracle_room_mean"] = measure(
            np.stack([room_mean[str(room)] for room in query_room]), data, ids
        )
        room_clock = {
            (room, int(e)): data["pose"][
                data["train"] & (data["room"] == room) & (data["end"] == e)
            ].mean(0)
            for room in ["E01", "E02"]
            for e in set(data["end"][ids[query_room == room]])
        }
        zero["oracle_room_clock"] = measure(
            np.stack(
                [
                    room_clock[(str(room), int(e))]
                    for room, e in zip(query_room, data["end"][ids])
                ]
            ),
            data,
            ids,
        )
        cal["reference_mean"] = measure(ref_mean, data, ids)
        for kind, pred in raw_predictions.items():
            cal["raw_" + kind] = measure(pred, data, ids)
        same = next(
            v["zero"]
            for v in candidates["cross_room"]
            if v["step"] == selected["cross_room"]["calibrated_step"]
        )
        gates = dict(
            zero=gate(
                zero["cross_room"], [v for k, v in zero.items() if k != "cross_room"]
            ),
            calibrated=gate(
                cal["cross_room"],
                [v for k, v in cal.items() if k != "cross_room"]
                + [zero["cross_room"], same],
            ),
        )
        gates["calibrated"]["beats_matched_own_zero"] = (
            cal["cross_room"]["pooled"]["nje"] < zero["cross_room"]["pooled"]["nje"]
        )
        gates["calibrated"]["beats_same_checkpoint_zero"] = (
            cal["cross_room"]["pooled"]["nje"] < same["pooled"]["nje"]
        )
        p.bind(ROOT, registration["source_sha256"])
        p.bind(cache, receipt["arrays_sha256"])
        p.bind(evaluation, manifest)
        report = dict(
            registration=registration,
            candidates=candidates,
            selected=selected,
            zero=zero,
            calibrated=cal,
            calibrated_checkpoint_zero=same,
            gates=gates,
            passes={k: all(v.values()) for k, v in gates.items()},
            source_train_rows=len(tr),
            source_query_rows=len(ids),
            source_reference_rows_per_room=120,
            source_reference_rows_total=240,
            evaluation_directory=evaluation.name,
            predictions_ready_sha256=p.digest(evaluation / "predictions-ready.json"),
            held_metrics_computed=False,
            seconds=time.monotonic() - started,
            limits=cfg["limits"],
        )
        p.save(out / "source-results.json", report)
        p.save(
            out / "source-ready.json",
            dict(
                held_metrics_computed=False,
                artifacts_sha256={
                    str(q.relative_to(out)): p.digest(q)
                    for q in out.rglob("*")
                    if q.is_file() and q != lock
                },
            ),
        )
        print(
            json.dumps(
                dict(
                    passes=report["passes"],
                    selected=selected,
                    seconds=report["seconds"],
                )
            ),
            flush=True,
        )
    finally:
        if lock.exists() and p.read(lock)["pid"] == os.getpid():
            lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["cache", "raw", "config", "out"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.cache, args.raw, args.config, args.out, args.device, args.resume)
