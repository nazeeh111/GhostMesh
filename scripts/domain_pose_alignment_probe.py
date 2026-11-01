"""E01-only pose-prototype and cross-person similarity source experiment.

All training budgets precede source-validation checkpoint selection. Neither
held-room annotation decoding nor held-room scoring is implemented here.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import time
import json

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.pose_alignment import PoseAlignedStudent, cross_person_alignment  # noqa:E402
from ghostmesh.episodic_calibration import fit_residual, apply_residual  # noqa:E402
from scripts import domain_episodic_probe as p  # noqa:E402
from scripts import geometric_checkpoint as recovery  # noqa:E402
from scripts.domain_amplitude_probe import select  # noqa:E402

ARMS = ("ordinary", "cross_person", "teacher_permuted")


def validate_config(cfg):
    p.require(
        cfg["schema_version"] == 1
        and cfg["source_room"] == "E01"
        and cfg["held_room"] == "E03"
        and cfg["source_validation_people"] == ["S04", "S09"]
        and cfg["arms"] == list(ARMS)
        and cfg["primary"] == "cross_person",
        "Frozen E01-only alignment recipe required",
    )
    for key in ["seed", "steps", "checkpoint_every"]:
        p.require(
            type(cfg[key]) is int and cfg[key] >= 0,
            "Nonnegative integer config required",
        )
    p.require(
        0 < cfg["steps"] < 10000
        and cfg["checkpoint_every"] > 0
        and cfg["steps"] % cfg["checkpoint_every"] == 0,
        "Recoverable finite training budget required",
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
            "Positive finite configuration required",
        )
    p.require(
        not isinstance(cfg["weight_decay"], bool)
        and np.isfinite(cfg["weight_decay"])
        and cfg["weight_decay"] >= 0,
        "Nonnegative finite weight decay required",
    )
    p.require(
        cfg["reference_count_convention"] == 24
        and cfg["source_reference_people"] == ["S01", "S02"]
        and cfg["reference_actions"] == ["A03", "A07", "A12"]
        and cfg["references_per_record"] == 20,
        "Shared120reference contract required",
    )
    return cfg


def load_source(cache, raw):
    receipt = p.read(cache / "receipt.json")
    registration = p.read(cache / "registration.json")
    p.require(
        receipt["kind"] == registration["kind"] == "E01-label-free-delayed-moments"
        and receipt["registration_sha256"] == p.digest(cache / "registration.json")
        and receipt["logical_rows"] == receipt["allocated_rows"] == 8549
        and receipt["annotation_arrays_decoded"] is False,
        "Exactly allocated label-free source cache required",
    )
    p.bind(cache, receipt["arrays_sha256"])
    p.bind(cache / "executed-source", registration["source_sha256"])
    p.require(
        registration["raw_receipt_sha256"] == p.digest(raw / "receipt.json")
        and registration["annotation_arrays_decoded"] is False
        and registration["context_frames"] == 32
        and registration["output_delay_frames"] == 4
        and registration["physical_clock_calibrated"] is False,
        "RF timing/provenance boundary differs",
    )
    data = p.arrays(cache / "metadata.npz")
    n = receipt["logical_rows"]
    p.require(
        set(data) == {"room", "person", "action", "record", "end", "raw_index"}
        and all(len(v) == n for v in data.values())
        and set(data["room"]) == {"E01"},
        "Held room or malformed metadata entered source",
    )
    data["validation"] = np.isin(data["person"], ["S04", "S09"])
    data["train"] = ~data["validation"]
    p.require(
        (int(data["train"].sum()), int(data["validation"].sum())) == (6830, 1719)
        and set(data["person"][data["train"]])
        == {"S01", "S02", "S03", "S05", "S06", "S07", "S08", "S10"},
        "Source role counts differ",
    )
    rr = p.read(raw / "receipt.json")
    p.validate_raw_indices(data, rr["records"])
    p.require(
        data["raw_index"].max() < 80190
        and (data["end"] <= 287).all()
        and rr["config"]["reference_canvas"] == [640, 640],
        "Raw target joins/plane differ",
    )
    for name in ["pose", "valid"]:
        p.require(
            p.digest(raw / f"{name}.npy") == rr["arrays"][name],
            "Source annotation bytes differ",
        )
    target = np.asarray(
        np.load(raw / "pose.npy", mmap_mode="r", allow_pickle=False)[data["raw_index"]],
        dtype=float,
    )
    good = np.load(raw / "valid.npy", mmap_mode="r", allow_pickle=False)[
        data["raw_index"]
    ]
    p.require(
        good.all() and target.shape == (n, 17, 2) and np.isfinite(target).all(),
        "Complete finite source body targets required",
    )
    data["pose"] = (target - target[:, [11, 12]].mean(1, keepdims=True)).astype(
        np.float32
    )
    feature = np.load(cache / "features.npy", mmap_mode="r", allow_pickle=False)
    p.require(
        feature.shape == (n, 32, 12, 114) and feature.dtype == np.float32,
        "Delayed RF history schema differs",
    )
    return data, feature, receipt


class Reader:
    def __init__(self, feature, allowed):
        self.feature = feature
        self.allowed = np.asarray(allowed).copy()

    def __call__(self, rows):
        rows = p.checked_rows(rows, self.allowed)
        x = np.array(self.feature[rows], copy=True)
        p.require(np.isfinite(x).all(), "Nonfinite RF history")
        return x


def prototypes(target, scale, seed):
    y = np.asarray(target, dtype=float).reshape(len(target), 34)
    s = np.asarray(scale, dtype=float).reshape(34)
    z = y / s
    rng = np.random.default_rng(seed)
    centers = [z[int(rng.integers(len(z)))]]
    nearest = np.square(z - centers[0]).sum(1)
    for _ in range(127):
        p.require(
            nearest.sum() > 1e-10, "Insufficient distinct TRAIN poses for128prototypes"
        )
        center = z[int(rng.choice(len(z), p=nearest / nearest.sum()))]
        centers.append(center)
        nearest = np.minimum(nearest, np.square(z - center).sum(1))
    centers = np.stack(centers)
    assign = None
    for _ in range(50):
        distance = np.maximum(
            np.square(z).sum(1)[:, None]
            + np.square(centers).sum(1)[None]
            - 2 * p.product(z, centers.T),
            0,
        )
        new = distance.argmin(1)
        if assign is not None and np.array_equal(new, assign):
            break
        assign = new
        for k in range(128):
            rows = np.flatnonzero(assign == k)
            centers[k] = (
                z[rows].mean(0) if len(rows) else z[int(distance.min(1).argmax())]
            )
    counts = np.bincount(assign, minlength=128)
    p.require((counts > 0).all(), "Empty prototype requires a new prospective recipe")
    value = np.stack([y[assign == k].mean(0) for k in range(128)]).reshape(128, 17, 2)
    prior = counts / counts.sum()
    np.testing.assert_allclose(
        np.einsum("k,kjd->jd", prior, value),
        target.mean(0, dtype=float),
        atol=1e-10,
        rtol=1e-10,
    )
    return value.astype(np.float32), prior.astype(np.float32), assign


def statistics(data, reader, cfg):
    ids = np.flatnonzero(data["train"])
    total = 0
    mean = np.zeros((12, 114))
    m2 = mean.copy()
    for first in range(0, len(ids), 64):
        x = reader(ids[first : first + 64]).astype(float).reshape(-1, 12, 114)
        n = len(x)
        mu = x.mean(0)
        delta = mu - mean
        new = total + n
        m2 += np.square(x - mu).sum(0) + delta**2 * (total * n / new)
        mean += delta * (n / new)
        total = new
    target = p.training_labels(data, ids).astype(float)
    pairs = p.pair_rows(data["record"][ids], data["end"][ids])
    delta = target[pairs[:, 0]] - target[pairs[:, 1]]
    scale = np.maximum(target.std(0), cfg["pose_scale_floor"])
    proto, prior, assignment = prototypes(target, scale, cfg["seed"])
    return dict(
        input_mean=mean.astype(np.float32),
        input_scale=np.maximum(np.sqrt(m2 / total), 0.05).astype(np.float32),
        pose_mean=target.mean(0).astype(np.float32),
        pose_scale=scale.astype(np.float32),
        pose_delta_scale=np.maximum(
            np.sqrt(np.square(delta).mean(0)), cfg["delta_scale_floor"]
        ).astype(np.float32),
        prototypes=proto,
        prior=prior,
        TRAIN_prototype_assignment=assignment,
    )


def make_schedule(data, cfg):
    ids = np.flatnonzero(data["train"])
    pairs = ids[p.pair_rows(data["record"][ids], data["end"][ids])]
    people = sorted(set(data["person"][ids]))
    p.require(len(people) == 8, "Eight TRAINpeople required")
    rng = np.random.default_rng(cfg["seed"])
    s = []
    for _ in range(cfg["steps"]):
        groups = []
        for person in people:
            pair = pairs[
                int(rng.choice(np.flatnonzero(data["person"][pairs[:, 0]] == person)))
            ]
            extra = int(rng.choice(ids[data["person"][ids] == person]))
            groups.extend([int(pair[0]), int(pair[1]), extra])
        s.append(groups)
    return np.asarray(s, dtype=np.int64)


def make_model(stats, device):
    return PoseAlignedStudent(
        *[
            torch.from_numpy(stats[k])
            for k in ["input_mean", "input_scale", "prototypes", "prior"]
        ]
    ).to(device)


def optimizer(model, cfg):
    return torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )


def train_step(model, opt, x, target, people, stats, arm, cfg):
    p.require(
        arm in ARMS and len(x) == len(target) == len(people) == 24,
        "Known24history training batch required",
    )
    result = model(x)
    scale = torch.as_tensor(stats["pose_scale"], device=x.device)
    delta = torch.as_tensor(stats["pose_delta_scale"], device=x.device)
    current = torch.arange(0, 24, 3, device=x.device)
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
    perm = (
        torch.arange(24, device=x.device).reshape(8, 3)[:, [1, 2, 0]].flatten()
        if arm == "teacher_permuted"
        else None
    )
    alignment = cross_person_alignment(
        result["code"], target, people, scale, permutation=perm
    )
    loss = pose + (cfg["alignment_weight"] * alignment if arm != "ordinary" else 0)
    p.require(bool(torch.isfinite(loss)), "Nonfinite objective before update")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    opt.step()
    return dict(
        loss=float(loss.detach().cpu()),
        pose_loss=float(pose.detach().cpu()),
        alignment=float(alignment.detach().cpu()),
    )


def encode(model, reader, rows, device):
    values = []
    codes = []
    model.eval()
    with torch.inference_mode():
        for first in range(0, len(rows), 16):
            r = model(torch.from_numpy(reader(rows[first : first + 16])).to(device))
            values.append(r["pose"].cpu().numpy())
            codes.append(r["code"].cpu().numpy())
    return np.concatenate(values), np.concatenate(codes)


def run(cache, raw, config, out, device, resume=False):
    cfg = validate_config(p.read(config))
    p.require(resume or not out.exists(), "Existing study preserved")
    p.require(
        not any(out.resolve().is_relative_to(v.resolve()) for v in [cache, raw]),
        "Output inside immutable inputs",
    )
    p.require(
        device == "cpu" or torch.backends.mps.is_available(), "Native MPS unavailable"
    )
    data, feature, receipt = load_source(cache, raw)
    paths = [
        Path(__file__),
        ROOT / "ghostmesh/pose_alignment.py",
        ROOT / "scripts/build_delayed_moment_cache.py",
        ROOT / "scripts/domain_episodic_probe.py",
        ROOT / "scripts/domain_amplitude_probe.py",
        ROOT / "ghostmesh/episodic_calibration.py",
        ROOT / "scripts/geometric_checkpoint.py",
        ROOT / "scripts/verify_geometric_distillation.py",
    ]
    registration = dict(
        kind="E01-cross-person-pose-prototype-source",
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
            "Recovery identity differs or study complete",
        )
    else:
        p.save(out / "registration.json", registration)
        shutil.copyfile(config, out / "config.json")
        for q in paths:
            dst = out / "executed-source" / q.relative_to(ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(q, dst)
    lock = out / "writer.lock"
    if lock.exists():
        pid = p.read(lock)["pid"]
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            p.require(resume, "Explicit recovery required")
            shutil.copyfile(lock, out / f"prior-writer-{pid}.json")
            lock.unlink()
        else:
            raise ValueError("Existing live source writer")
    p.save(lock, dict(pid=os.getpid()))
    started = time.monotonic()
    torch.set_num_threads(1)
    try:
        train = Reader(feature, data["train"])
        validation = Reader(feature, data["validation"])
        schedule = make_schedule(data, cfg)
        stats = statistics(data, train, cfg)
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
        initial = make_model(stats, "cpu")
        if resume:
            saved = torch.load(
                out / "initial-model.pt", map_location="cpu", weights_only=True
            )
            p.require(
                all(torch.equal(v, saved[k]) for k, v in initial.state_dict().items()),
                "Initial model differs",
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
        people = torch.arange(8, device=device).repeat_interleave(3)
        for arm in ARMS:
            folder = out / arm
            folder.mkdir(exist_ok=resume)
            net = make_model(stats, device)
            net.load_state_dict(initial.state_dict(), strict=True)
            opt = optimizer(net, cfg)
            step = (
                recovery.load(folder, net, opt, identity, arm, device) if resume else 0
            )
            p.require(step <= cfg["steps"], "Optimizer budget exceeded")
            if not resume:
                recovery.save(folder, net, opt, 0, identity, arm, device)
            for index in range(step, cfg["steps"]):
                rows = schedule[index]
                y = p.training_labels(data, rows)
                result = train_step(
                    net,
                    opt,
                    torch.from_numpy(train(rows)).to(device),
                    torch.from_numpy(y).to(device),
                    people,
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
                    "Completed identity differs",
                )
            else:
                p.save(folder / "complete.json", complete)
            del net, opt
        # All equal budgets are now complete; source validation has not been scored.
        attempt = 0
        while (out / f"source-selection-{attempt:02}").exists():
            attempt += 1
        evaluation = out / f"source-selection-{attempt:02}"
        evaluation.mkdir()
        refs = p.references(data, cfg)
        ids = np.flatnonzero(data["validation"])
        target = p.training_labels(data, refs)
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
                    "Candidate seal differs",
                )
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                p.require(
                    payload["identity"] == identity and payload["step"] == step,
                    "Candidate identity differs",
                )
                net = make_model(stats, device)
                net.load_state_dict(payload["model"], strict=True)
                rp, rc = encode(net, train, refs, device)
                qp, qc = encode(net, validation, ids, device)
                state = fit_residual(
                    torch.from_numpy(rc),
                    torch.from_numpy(target).flatten(1),
                    torch.from_numpy(rp).flatten(1),
                    ridge=cfg["ridge"],
                    reference_count=cfg["reference_count_convention"],
                )
                adapted = (
                    apply_residual(
                        state, torch.from_numpy(qc), torch.from_numpy(qp).flatten(1)
                    )
                    .reshape(-1, 17, 2)
                    .numpy()
                )
                adapted -= adapted[:, [11, 12]].mean(1, keepdims=True)
                stem = f"{arm}-{step:04}"
                np.savez(
                    evaluation / (stem + "-encoded.npz"),
                    reference_pose=rp,
                    reference_code=rc,
                    query_pose=qp,
                    query_code=qc,
                )
                np.savez(
                    evaluation / (stem + "-reference-state.npz"),
                    code_mean=state.code_mean.detach().numpy(),
                    correction_mean=state.correction_mean.detach().numpy(),
                    weights=state.weights.detach().numpy(),
                )
                np.save(evaluation / (stem + "-VALzero.npy"), qp, allow_pickle=False)
                np.save(
                    evaluation / (stem + "-VALcal.npy"), adapted, allow_pickle=False
                )
                del net
        rx = (
            (train(refs).astype(float) - stats["input_mean"]) / stats["input_scale"]
        ).reshape(120, -1)
        qx = (
            (validation(ids).astype(float) - stats["input_mean"]) / stats["input_scale"]
        ).reshape(len(ids), -1)
        raw_predictions, raw_states, raw_grids = p.raw_controls(
            rx, target.astype(float), data["person"][refs], qx
        )
        for kind, pred in raw_predictions.items():
            np.save(evaluation / f"raw-{kind}-VAL.npy", pred, allow_pickle=False)
            np.savez(evaluation / f"raw-{kind}-state.npz", **raw_states[kind])
        p.save(evaluation / "raw-reference-cv.json", raw_grids)
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
                        zero=p.measure(a, data, ids),
                        calibrated=p.measure(b, data, ids),
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
        zero["source_mean"] = p.measure(
            np.broadcast_to(data["pose"][tr].mean(0), data["pose"][ids].shape),
            data,
            ids,
        )
        clock = {
            int(e): data["pose"][data["train"] & (data["end"] == e)].mean(0)
            for e in set(data["end"][tr])
        }
        zero["source_clock"] = p.measure(
            np.stack([clock[int(e)] for e in data["end"][ids]]), data, ids
        )
        cal["reference_mean"] = p.measure(
            np.broadcast_to(target.mean(0), data["pose"][ids].shape), data, ids
        )
        for kind, pred in raw_predictions.items():
            cal["raw_" + kind] = p.measure(pred, data, ids)
        cal_step = selected["cross_person"]["calibrated_step"]
        same_checkpoint_zero = next(
            row["zero"] for row in candidates["cross_person"] if row["step"] == cal_step
        )
        gates = dict(
            zero=p.gate(
                zero["cross_person"],
                [v for k, v in zero.items() if k != "cross_person"],
            ),
            calibrated=p.gate(
                cal["cross_person"],
                [v for k, v in cal.items() if k != "cross_person"]
                + [zero["cross_person"], same_checkpoint_zero],
            ),
        )
        gates["calibrated"]["beats_matched_own_zero"] = (
            cal["cross_person"]["pooled"]["nje"] < zero["cross_person"]["pooled"]["nje"]
        )
        gates["calibrated"]["beats_same_checkpoint_zero"] = (
            cal["cross_person"]["pooled"]["nje"] < same_checkpoint_zero["pooled"]["nje"]
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
            calibrated_checkpoint_zero=same_checkpoint_zero,
            gates=gates,
            passes={k: all(v.values()) for k, v in gates.items()},
            source_train_rows=len(tr),
            source_query_rows=len(ids),
            source_reference_rows=120,
            evaluation_directory=evaluation.name,
            held_metrics_computed=False,
            predictions_ready_sha256=p.digest(evaluation / "predictions-ready.json"),
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
