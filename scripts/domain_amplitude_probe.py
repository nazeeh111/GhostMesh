"""Fresh E01-only full-amplitude/source-person-invariance pilot.

Source validation selects checkpoints after all equal training budgets finish.
No held-room annotation decoding or scoring is implemented by this command.
Both source zero-shot and labeled-reference proxy modes are reported separately.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy import __version__ as SCIPY_VERSION
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.amplitude_pose import AmplitudeStudent, normalize_amplitude  # noqa:E402
from ghostmesh.geometric_distillation import paired_objective  # noqa:E402
from ghostmesh.episodic_calibration import apply_residual, fit_residual  # noqa:E402
from scripts import domain_episodic_probe as common  # noqa:E402
from scripts import geometric_checkpoint as recovery  # noqa:E402

ARMS = ("ordinary", "gain_augmented", "gain_adversarial")


def validate_config(cfg):
    common.require(
        cfg["schema_version"] == 1
        and cfg["arms"] == list(ARMS)
        and cfg["primary"] == "gain_adversarial",
        "Unknown amplitude recipe",
    )
    common.require(
        cfg["source_room"] == "E01"
        and cfg["held_room"] == "E03"
        and cfg["source_validation_people"] == ["S04", "S09"],
        "New source/held recipe required",
    )
    for key in ("seed", "gain_seed", "steps", "checkpoint_every"):
        common.require(
            type(cfg[key]) is int and cfg[key] >= 0,
            "Nonnegative integer config required",
        )
    common.require(
        0 < cfg["steps"] < 10000
        and cfg["checkpoint_every"] > 0
        and cfg["steps"] % cfg["checkpoint_every"] == 0,
        "Finite recoverable budget required",
    )
    common.require(
        type(cfg["adversarial_warmup_steps"]) is int
        and 0 <= cfg["adversarial_warmup_steps"] < cfg["steps"],
        "Valid adversarial warmup required",
    )
    for key in (
        "learning_rate",
        "gain_max_db",
        "nuisance_weight",
        "ridge",
        "pose_scale_floor",
        "delta_scale_floor",
    ):
        common.require(
            not isinstance(cfg[key], bool) and np.isfinite(cfg[key]) and cfg[key] > 0,
            "Positive finite config required: " + key,
        )
    common.require(
        not isinstance(cfg["weight_decay"], bool)
        and np.isfinite(cfg["weight_decay"])
        and cfg["weight_decay"] >= 0,
        "Nonnegative decay required",
    )
    return cfg


class RawReader:
    def __init__(self, rf, raw_indices, allowed):
        self.rf = rf
        self.indices = np.asarray(raw_indices).copy()
        self.allowed = np.asarray(allowed).copy()
        common.require(
            self.allowed.dtype == bool
            and self.indices.shape == self.allowed.shape
            and self.indices.dtype.kind in "iu",
            "Typed raw RF role required",
        )

    def __call__(self, rows, *, gains=None):
        rows = common.checked_rows(rows, self.allowed)
        frames = self.indices[rows, None] - np.arange(7, -1, -1)[None]
        common.require(
            frames.min() >= 0 and frames.max() < len(self.rf),
            "Raw causal history exceeds cache",
        )
        return normalize_amplitude(np.asarray(self.rf[frames, :3]), gain_db=gains)


def make_schedule(data, cfg):
    ids = np.flatnonzero(data["train"])
    pairs = common.pair_rows(data["record"][ids], data["end"][ids])
    pairs = ids[pairs]
    people = sorted(set(data["person"][ids]))
    common.require(
        len(people) == 8 and set(data["room"][ids]) == {"E01"},
        "Eight E01TRAIN people required",
    )
    choices = {
        person: np.flatnonzero(data["person"][pairs[:, 0]] == person)
        for person in people
    }
    rng = np.random.default_rng(cfg["seed"])
    schedule = np.stack(
        [
            pairs[[int(rng.choice(choices[person])) for person in people]]
            for _ in range(cfg["steps"])
        ]
    )
    gains = np.random.default_rng(cfg["gain_seed"]).uniform(
        -cfg["gain_max_db"], cfg["gain_max_db"], (cfg["steps"], 8, 3)
    )
    return schedule, gains


def pose_statistics(data, cfg):
    ids = np.flatnonzero(data["train"])
    target = common.training_labels(data, ids).astype(float)
    pairs = common.pair_rows(data["record"][ids], data["end"][ids])
    delta = target[pairs[:, 0]] - target[pairs[:, 1]]
    return dict(
        pose_mean=target.mean(0).astype(np.float32),
        pose_scale=np.maximum(target.std(0), cfg["pose_scale_floor"]).astype(
            np.float32
        ),
        pose_delta_scale=np.maximum(
            np.sqrt(np.square(delta).mean(0)), cfg["delta_scale_floor"]
        ).astype(np.float32),
    )


def make_model(stats, device):
    return AmplitudeStudent(
        torch.from_numpy(stats["pose_mean"]),
        torch.from_numpy(stats["pose_scale"]),
        people=8,
    ).to(device)


def optimizer(model, cfg):
    return torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )


def train_step(model, opt, x, target, person, delta, arm, index, cfg):
    common.require(arm in ARMS, "Unknown amplitude arm")
    progress = max(
        0.0,
        (index + 1 - cfg["adversarial_warmup_steps"])
        / (cfg["steps"] - cfg["adversarial_warmup_steps"]),
    )
    reversal = (
        2 / (1 + math.exp(-10 * progress)) - 1 if arm == "gain_adversarial" else 0.0
    )
    output = model(x, reversal=reversal)
    pose = paired_objective(output["pose"], target, model.body.pose_scale, delta)
    nuisance = F.cross_entropy(output["nuisance_logits"], person)
    loss = pose + cfg["nuisance_weight"] * nuisance
    common.require(bool(torch.isfinite(loss)), "Nonfinite objective before update")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    opt.step()
    return dict(
        loss=float(loss.detach().cpu()),
        pose_loss=float(pose.detach().cpu()),
        nuisance_ce=float(nuisance.detach().cpu()),
        nuisance_training_accuracy=float(
            (output["nuisance_logits"].argmax(1) == person)
            .float()
            .mean()
            .detach()
            .cpu()
        ),
        reversal=reversal,
    )


def select(candidates, mode):
    usable = [
        row
        for row in candidates
        if row[mode]["pooled"]["motion_r2"] is not None
        and np.isfinite(row[mode]["pooled"]["motion_r2"])
    ]
    common.require(bool(usable), "No finite source validation candidate")
    return min(
        usable,
        key=lambda row: (
            -row[mode]["pooled"]["motion_r2"],
            row[mode]["pooled"]["nje"],
            row["step"],
        ),
    )


def run(cache, raw, config_path, out, device, resume=False):
    cfg = validate_config(common.read(config_path))
    common.require(resume or not out.exists(), "Existing amplitude study preserved")
    common.require(
        not any(out.resolve().is_relative_to(path.resolve()) for path in (cache, raw)),
        "Output inside immutable inputs",
    )
    common.require(
        device == "cpu" or torch.backends.mps.is_available(), "Native MPS unavailable"
    )
    data, _, cache_receipt, _ = common.load_source(cache, raw, cfg)
    raw_receipt = common.read(raw / "receipt.json")
    common.require(
        common.digest(raw / "rf.npy") == raw_receipt["arrays"]["rf"],
        "Raw RF hash differs",
    )
    rf = np.load(raw / "rf.npy", mmap_mode="r", allow_pickle=False)
    paths = [
        Path(__file__),
        ROOT / "ghostmesh/amplitude_pose.py",
        ROOT / "ghostmesh/geometric_distillation.py",
        ROOT / "ghostmesh/episodic_calibration.py",
        ROOT / "scripts/domain_episodic_probe.py",
        ROOT / "scripts/geometric_checkpoint.py",
        ROOT / "scripts/build_moment_cache.py",
        ROOT / "scripts/verify_geometric_distillation.py",
        ROOT / "tests/test_amplitude_pose.py",
        ROOT / "tests/test_amplitude_probe.py",
    ]
    registration = dict(
        kind="E01-only-full-amplitude-person-invariance",
        config=cfg,
        config_sha256=common.digest(config_path),
        cache_receipt_sha256=common.digest(cache / "receipt.json"),
        raw_receipt_sha256=common.digest(raw / "receipt.json"),
        source_sha256={str(p.relative_to(ROOT)): common.digest(p) for p in paths},
        device=device,
        torch=str(torch.__version__),
        numpy=np.__version__,
        scipy=SCIPY_VERSION,
        held_annotation_values_decoded=False,
        scope="SOURCE E01only; noE03/E02/E04 score, nofullimage/3D or successfulcalibrationclaim",
    )
    out.mkdir(parents=True, exist_ok=resume)
    if resume:
        common.require(
            common.read(out / "registration.json") == registration
            and not (out / "source-results.json").exists(),
            "Recovery identity differs or study completed",
        )
    else:
        common.save(out / "registration.json", registration)
        shutil.copyfile(config_path, out / "config.json")
        for path in paths:
            dest = out / "executed-source" / path.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
    lock = out / "writer.lock"
    if lock.exists():
        pid = common.read(lock)["pid"]
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            common.require(resume, "Explicit recovery required")
            shutil.copyfile(lock, out / f"prior-writer-{pid}.json")
            lock.unlink()
        else:
            raise ValueError("Existing amplitude writer")
    common.save(lock, dict(pid=os.getpid()))
    started = time.monotonic()
    torch.set_num_threads(1)
    try:
        tr_reader = RawReader(rf, data["raw_index"], data["train"])
        vr = RawReader(rf, data["raw_index"], data["validation"])
        schedule, gains = make_schedule(data, cfg)
        stats = pose_statistics(data, cfg)
        if resume:
            np.testing.assert_array_equal(
                np.load(out / "schedule.npy", allow_pickle=False), schedule
            )
            np.testing.assert_array_equal(
                np.load(out / "gains-db.npy", allow_pickle=False), gains
            )
            for key, value in stats.items():
                np.testing.assert_array_equal(
                    common.arrays(out / "TRAIN-statistics.npz")[key], value
                )
        else:
            np.save(out / "schedule.npy", schedule, allow_pickle=False)
            np.save(out / "gains-db.npy", gains, allow_pickle=False)
            np.savez(out / "TRAIN-statistics.npz", **stats)
            np.savez(out / "source-data.npz", **data)
        torch.manual_seed(cfg["seed"])
        initial = make_model(stats, "cpu")
        if resume:
            saved = torch.load(
                out / "initial-model.pt", map_location="cpu", weights_only=True
            )
            common.require(
                set(saved) == set(initial.state_dict())
                and all(
                    torch.equal(saved[k], v) for k, v in initial.state_dict().items()
                ),
                "Initial state differs",
            )
        else:
            torch.save(initial.state_dict(), out / "initial-model.pt")
        identity = ":".join(
            common.digest(out / name)
            for name in (
                "registration.json",
                "schedule.npy",
                "gains-db.npy",
                "TRAIN-statistics.npz",
                "initial-model.pt",
            )
        )
        classes = {
            person: i
            for i, person in enumerate(sorted(set(data["person"][data["train"]])))
        }
        delta = torch.from_numpy(stats["pose_delta_scale"]).to(device)
        for arm in ARMS:
            folder = out / arm
            folder.mkdir(exist_ok=resume)
            net = make_model(stats, device)
            net.load_state_dict(initial.state_dict(), strict=True)
            opt = optimizer(net, cfg)
            step = (
                recovery.load(folder, net, opt, identity, arm, device) if resume else 0
            )
            common.require(step <= cfg["steps"], "Optimizer budget exceeded")
            if not resume:
                recovery.save(folder, net, opt, 0, identity, arm, device)
            for index in range(step, cfg["steps"]):
                pair = schedule[index]
                rows = np.concatenate((pair[:, 0], pair[:, 1]))
                target = common.training_labels(data, rows)
                gain = (
                    None
                    if arm == "ordinary"
                    else np.concatenate((gains[index], gains[index]))
                )
                x = tr_reader(rows, gains=gain)
                person = np.array(
                    [classes[str(p)] for p in data["person"][rows]], dtype=np.int64
                )
                net.train()
                progress = train_step(
                    net,
                    opt,
                    torch.from_numpy(x).to(device),
                    torch.from_numpy(target).to(device),
                    torch.from_numpy(person).to(device),
                    delta,
                    arm,
                    index,
                    cfg,
                )
                if (index + 1) % cfg["checkpoint_every"] == 0:
                    recovery.save(folder, net, opt, index + 1, identity, arm, device)
                    progress.update(
                        arm=arm, step=index + 1, seconds=time.monotonic() - started
                    )
                    common.save(folder / f"progress-{index + 1:04}.json", progress)
                    print(json.dumps(progress), flush=True)
            complete = dict(
                step=cfg["steps"],
                identity=identity,
                checkpoint_sha256=common.digest(folder / f"step-{cfg['steps']:04}.pt"),
            )
            if (folder / "complete.json").exists():
                common.require(
                    common.read(folder / "complete.json") == complete,
                    "Completed state differs",
                )
            else:
                common.save(folder / "complete.json", complete)
            del net, opt
        # All three equal training budgets are complete before source selection.
        attempt = 0
        while (out / f"source-selection-{attempt:02}").exists():
            attempt += 1
        evaluation = out / f"source-selection-{attempt:02}"
        evaluation.mkdir()
        refs = common.references(data, cfg)
        ids = np.flatnonzero(data["validation"])
        target = common.training_labels(data, refs)
        np.save(evaluation / "reference-rows.npy", refs, allow_pickle=False)
        candidates = {}
        selected = {}
        zero = {}
        cal = {}
        for arm in ARMS:
            candidates[arm] = []
            for step in range(0, cfg["steps"] + 1, cfg["checkpoint_every"]):
                checkpoint = out / arm / f"step-{step:04}.pt"
                seal = common.read(checkpoint.with_name(checkpoint.name + ".json"))
                common.require(
                    seal
                    == dict(
                        kind="source-geometric-recovery",
                        step=step,
                        identity=identity,
                        arm=arm,
                        device=device,
                        checkpoint_sha256=common.digest(checkpoint),
                    ),
                    "Candidate checkpoint seal differs",
                )
                payload = torch.load(
                    checkpoint,
                    map_location="cpu",
                    weights_only=True,
                )
                common.require(
                    payload["identity"] == identity and payload["step"] == step,
                    "Selected candidate identity differs",
                )
                net = make_model(stats, device)
                net.load_state_dict(payload["model"], strict=True)
                rp, rc = common.encode(net, tr_reader, refs, device)
                qp, qc = common.encode(net, vr, ids, device)
                state = fit_residual(
                    torch.from_numpy(rc),
                    torch.from_numpy(target).flatten(1),
                    torch.from_numpy(rp).flatten(1),
                    ridge=cfg["ridge"],
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
                candidates[arm].append(
                    dict(
                        step=step,
                        zero=common.measure(qp, data, ids),
                        calibrated=common.measure(adapted, data, ids),
                    )
                )
                del net
            a, b = (
                select(candidates[arm], "zero"),
                select(candidates[arm], "calibrated"),
            )
            selected[arm] = dict(zero_step=a["step"], calibrated_step=b["step"])
            zero[arm] = a["zero"]
            cal[arm] = b["calibrated"]
            print(
                json.dumps(
                    dict(
                        arm=arm,
                        selection=selected[arm],
                        zero=zero[arm]["pooled"],
                        calibrated=cal[arm]["pooled"],
                    )
                ),
                flush=True,
            )
        # The raw baselines have no global pose labels. Same120refs; folds split
        # complete reference people. Explicit double arithmetic includes norms.
        rx = tr_reader(refs).reshape(120, -1).astype(float)
        qx = vr(ids).reshape(len(ids), -1).astype(float)
        raw_pred, raw_state, raw_cv = common.raw_controls(
            rx, target.astype(float), data["person"][refs], qx
        )
        for kind, pred in raw_pred.items():
            cal["raw_" + kind] = common.measure(pred, data, ids)
            np.save(evaluation / f"raw-{kind}-VAL.npy", pred, allow_pickle=False)
            np.savez(evaluation / f"raw-{kind}-state.npz", **raw_state[kind])
        common.save(evaluation / "raw-reference-cv.json", raw_cv)
        prior = np.broadcast_to(
            data["pose"][data["train"]].mean(0), data["pose"][ids].shape
        )
        clock = {
            int(e): data["pose"][data["train"] & (data["end"] == e)].mean(0)
            for e in set(data["end"][data["train"]])
        }
        zero["source_mean"] = common.measure(prior, data, ids)
        zero["source_clock"] = common.measure(
            np.stack([clock[int(e)] for e in data["end"][ids]]), data, ids
        )
        cal["reference_mean"] = common.measure(
            np.broadcast_to(target.mean(0), data["pose"][ids].shape), data, ids
        )
        gates = dict(
            zero=common.gate(
                zero["gain_adversarial"],
                [row for name, row in zero.items() if name != "gain_adversarial"],
            ),
            calibrated=common.gate(
                cal["gain_adversarial"],
                [row for name, row in cal.items() if name != "gain_adversarial"],
            ),
        )
        common.bind(ROOT, registration["source_sha256"])
        common.bind(cache, cache_receipt["arrays_sha256"])
        common.require(
            common.digest(raw / "rf.npy") == raw_receipt["arrays"]["rf"]
            and common.digest(config_path) == registration["config_sha256"],
            "Raw/config changed during fitting",
        )
        report = dict(
            registration=registration,
            candidates=candidates,
            selected=selected,
            zero=zero,
            calibrated=cal,
            gates=gates,
            passes={name: all(gate.values()) for name, gate in gates.items()},
            source_reference_rows=120,
            source_query_rows=len(ids),
            evaluation_directory=evaluation.name,
            invocation_seconds=time.monotonic() - started,
            resumed=resume,
            limits="SOURCEE01modelselection only, repeatedly useddevelopmentpeople. Calibrationinrepresentedroom, notunseenroom. NoE03annotations decoded/queryscored. Same-source8classpersonlabels training-only. No physicalroom-simulator, identity-recognition, fullimage,3D orconfidence claim.",
        )
        common.save(out / "source-results.json", report)
        common.save(
            out / "source-ready.json",
            dict(
                kind=registration["kind"],
                held_metrics_computed=False,
                source_validation_computed=True,
                artifacts_sha256={
                    str(p.relative_to(out)): common.digest(p)
                    for p in out.rglob("*")
                    if p.is_file() and p != lock
                },
            ),
        )
        print(
            json.dumps(
                dict(
                    source_passes=report["passes"], seconds=report["invocation_seconds"]
                )
            ),
            flush=True,
        )
    finally:
        if lock.exists() and common.read(lock)["pid"] == os.getpid():
            lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "raw", "config", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.cache, args.raw, args.config, args.out, args.device, args.resume)
