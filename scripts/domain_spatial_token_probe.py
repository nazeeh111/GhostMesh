"""Source-only spatial-token pose potential: fit, then report source VAL only.

Extracts no E02 feature/pose values. Fixed frozen encoders, fixed2000-step heads,
four matched arms. This feasibility study does not authorize a new room score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.spatial_pose import SpatialPose, pose_objective  # noqa:E402
from scripts import domain_conditioned_dynamics_probe as p  # noqa:E402
from scripts.domain_masked_local_probe import load_backbone  # noqa:E402
from scripts.domain_masked_motion_diagnostic import motion_scores  # noqa:E402
from scripts.domain_masked_pretrain_probe import HistoryReader  # noqa:E402

ARMS = {
    "masked_spatial_motion": ("masked_temporal", False, "motion"),
    "masked_global_motion": ("masked_temporal", True, "motion"),
    "random_spatial_motion": ("random_frozen", False, "motion"),
    "masked_spatial_absolute": ("masked_temporal", False, "absolute"),
}
CONFIG = dict(
    seed=2027,
    steps=2000,
    batch_pairs=32,
    lr=0.001,
    weight_decay=0.01,
    gradient_clip=1.0,
    token_input="Frozen pre-global-pooling encoder output95x48; arithmetic mean across8causalframes. TRAIN-only48channel mean/SDfloor.001 over histories and patches",
    head="17learned joint queries,48D4head crossattention over95tokens, shared48->64->2GELU readout; predicted ownhip midpoint removed. Frozen encoder never updated",
    objective="PRIMARY .25 standardized absolute endpoint loss+.75 standardized lag8pose-difference loss. SourceTRAIN poseSDfloor.01 and deltaRMSfloor.001; no query GT centering or initialpose",
    controls="Masked global-only head/same parameters; random encoder spatial head; masked spatial absolute-only objective. Every arm uses same head initialization,2000x32sourcepairs and2endpoint labeloccurrences",
    schedule="Seed2027 uniformsource room then32pairswithoutreplacement, exactsame-record t-8; no gaps bridged",
    checkpoint="Fixed finalstep, no validation checkpoint selection or learning-rate tuning",
    split="E01/E03sourceTRAIN16people/VAL4people, every27action. No numericE02values, noE04/RGB",
    gate="Primary pooled sourceVAL articulationR2>.05 and lag8deltaR2>.05, eachroom articulationR2>0, primary articulationR2 and NJE beat all3learned controls and staticmean. Preset before fitting; no newE02score unless this premise passes",
    limitations="2Dhip-relative SOURCEfeasibility only, not fullimage or zero-shot/calibrated transfer. Pooling nominal10Hz histories smooths to0.35seconds earlier center while labels remain atendpoint. No calibrated confidence or deployment latency claim",
)


def role_data(study):
    p.source_contract(study)
    with np.load(study / "targets-and-metadata.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in ("room", "person", "record", "action", "end")}
    train, validation = p.source_roles(data)
    with np.load(study / "roles.npz", allow_pickle=False) as z:
        for name, mask in (("train", train), ("validation", validation)):
            if z[name].dtype != bool:
                raise ValueError("Stored source roles must be Boolean")
            np.testing.assert_array_equal(z[name], mask)
    ids = np.flatnonzero(train | validation)
    target = p.stored_npz_array(study / "targets-and-metadata.npz", "pose")
    if target.shape != (len(train), 17, 2) or target.dtype.kind not in "iuf":
        raise ValueError("Invalid lazy pose schema")
    # Explicit source-only value access before any target or feature statistics.
    y = p.real_finite(target[ids], "source-only poses").astype(np.float32)
    return (
        {k: v[ids] for k, v in data.items()},
        y,
        ids,
        train[ids],
        validation[ids],
        data,
        train | validation,
    )


def role_pairs(metadata, role):
    if role.dtype != bool or role.shape != metadata["end"].shape or not role.any():
        raise ValueError("Explicit nonempty Boolean source role required")
    ids = np.flatnonzero(role)
    current, past = p.pair_indices(metadata["record"][ids], metadata["end"][ids])
    return ids[current], ids[past]


def schedule(metadata, train):
    current, past = role_pairs(metadata, train)
    rng = np.random.default_rng(CONFIG["seed"])
    output = []
    for _ in range(CONFIG["steps"]):
        room = str(rng.choice(["E01", "E03"]))
        candidates = np.flatnonzero(metadata["room"][current] == room)
        if len(candidates) < CONFIG["batch_pairs"]:
            raise ValueError("Insufficient source room pairs")
        chosen = rng.choice(candidates, CONFIG["batch_pairs"], replace=False)
        output.append(np.stack([current[chosen], past[chosen]], axis=-1))
    return np.stack(output)


def training_statistics(tokens, target, metadata, train):
    if train.dtype != bool or train.shape != (len(target),):
        raise ValueError("Explicit source TRAIN role required")
    ids = np.flatnonzero(train)
    if (
        not len(ids)
        or not np.isin(metadata["room"][ids], ["E01", "E03"]).all()
        or np.isin(metadata["person"][ids], ["S04", "S09", "S24", "S29"]).any()
    ):
        raise ValueError("Only source TRAIN statistics permitted")
    current, past = role_pairs(metadata, train)
    x = p.real_finite(tokens[ids], "TRAIN tokens", (len(ids), 95, 48))
    y = p.real_finite(target[ids], "TRAIN labels", (len(ids), 17, 2))
    delta = p.real_finite(target[current], "TRAIN current pose") - p.real_finite(
        target[past], "TRAIN past pose"
    )
    return dict(
        input_mean=x.mean((0, 1)).astype(np.float32),
        input_scale=np.maximum(x.std((0, 1)), 0.001).astype(np.float32),
        target_mean=y.mean(0).astype(np.float32),
        target_scale=np.maximum(y.std(0), 0.01).astype(np.float32),
        delta_scale=np.maximum(np.sqrt(np.square(delta).mean(0)), 0.001).astype(
            np.float32
        ),
    )


def evaluate(model, tokens, target, metadata, role, stats):
    ids = np.flatnonzero(role)
    prediction = []
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(ids), 64):
            x = (tokens[ids[start : start + 64]] - stats["input_mean"]) / stats[
                "input_scale"
            ]
            prediction.append(
                model(torch.from_numpy(np.array(x)).to(device)).cpu().numpy()
            )
    prediction = np.concatenate(prediction)
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite source inference")
    result = dict(
        pooled=motion_scores(
            prediction, target[ids], metadata["record"][ids], metadata["end"][ids]
        )
    )
    for field in ("room", "person", "action"):
        labels = metadata[field][ids]
        result["per_" + field] = {
            str(label): motion_scores(
                prediction[labels == label],
                target[ids][labels == label],
                metadata["record"][ids][labels == label],
                metadata["end"][ids][labels == label],
            )
            for label in sorted(set(labels))
        }
    return result, prediction


def run(study, raw, out, device):
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested native MPS is unavailable")
    started = time.monotonic()
    torch.set_num_threads(1)
    metadata, target, ids, train, validation, all_metadata, allowed = role_data(study)
    receipt = json.loads((raw / "receipt.json").read_text())
    producer = json.loads((study / "registration.json").read_text())
    if p.digest(raw / "receipt.json") != producer["raw_receipt_sha256"]:
        raise ValueError("Raw source cache identity differs")
    p.verify_hashes(
        raw, {name + ".npy": value for name, value in receipt["arrays"].items()}
    )
    rf = np.load(raw / "rf.npy", mmap_mode="r", allow_pickle=False)
    reader = HistoryReader(rf, all_metadata, receipt["records"], allowed)
    plan = schedule(metadata, train)
    sources = [
        Path(__file__),
        ROOT / "ghostmesh/spatial_pose.py",
        ROOT / "tests/test_spatial_pose_probe.py",
        ROOT / "scripts/domain_conditioned_dynamics_probe.py",
        ROOT / "scripts/domain_masked_local_probe.py",
        ROOT / "scripts/domain_masked_pretrain_probe.py",
        ROOT / "scripts/domain_masked_motion_diagnostic.py",
        ROOT / "ghostmesh/masked_csi.py",
        ROOT / "scripts/domain_background_probe.py",
        ROOT / "scripts/domain_covariance_full_cache.py",
    ]
    registration = dict(
        config=CONFIG,
        device=device,
        source_ready_sha256=p.EXPECTED_READY,
        raw_receipt_sha256=p.digest(raw / "receipt.json"),
        source_sha256={str(x.relative_to(ROOT)): p.digest(x) for x in sources},
        encoder_sha256={
            arm: p.digest(study / (arm + "-final.pt"))
            for arm in ("masked_temporal", "random_frozen")
        },
        held_room_numeric_values_read=False,
        torch=torch.__version__,
        numpy=np.__version__,
    )
    out.mkdir(parents=True)
    p.save_json(out / "registration.json", registration)
    p.save_npz(
        out / "source-data.npz",
        target=target,
        train=train,
        validation=validation,
        global_ids=ids,
        **metadata,
    )
    np.save(out / "schedule.npy", plan, allow_pickle=False)
    for source in sources:
        destination = out / "executed-source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    caches = {}
    for encoder in ("masked_temporal", "random_frozen"):
        model = load_backbone(study / (encoder + "-final.pt")).to(device).eval()
        destination = np.lib.format.open_memmap(
            out / (encoder + "-tokens.npy"),
            mode="w+",
            dtype=np.float32,
            shape=(len(ids), 95, 48),
        )
        with torch.inference_mode():
            for start in range(0, len(ids), 16):
                # HistoryReader already applies the frozen finite-only frame
                # normalization. Applying it twice would change the inputs.
                frames = torch.from_numpy(reader(ids[start : start + 16])).to(device)
                tokens = (
                    model.encode_frames(frames.flatten(0, 1))
                    .reshape(-1, 8, 95, 48)
                    .mean(1)
                )
                value = tokens.cpu().numpy()
                if not np.isfinite(value).all():
                    raise ValueError("Nonfinite frozen tokens")
                destination[start : start + len(value)] = value
                if start % 4096 == 0:
                    print(
                        json.dumps(
                            dict(
                                encoder=encoder,
                                histories=start + len(value),
                                total=len(ids),
                            )
                        ),
                        flush=True,
                    )
        destination.flush()
        caches[encoder] = destination
        print(
            json.dumps(
                dict(
                    encoder=encoder,
                    tokens_complete=True,
                    seconds=time.monotonic() - started,
                )
            ),
            flush=True,
        )
        del model
    reports = {}
    for arm, (encoder, global_pool, objective) in ARMS.items():
        tokens = caches[encoder]
        stats = training_statistics(tokens, target, metadata, train)
        p.save_npz(out / (arm + "-TRAIN-statistics.npz"), **stats)
        torch.manual_seed(CONFIG["seed"])
        model = SpatialPose(
            torch.from_numpy(stats["target_mean"]),
            torch.from_numpy(stats["target_scale"]),
            global_pool=global_pool,
        ).to(device)
        torch.save(
            {k: v.detach().cpu() for k, v in model.state_dict().items()},
            out / (arm + "-initial.pt"),
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
        )
        scales = [
            torch.from_numpy(stats[key]).to(device)
            for key in ("target_scale", "delta_scale")
        ]
        losses = []
        for step, pairs in enumerate(plan):
            model.train()
            rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
            x = (tokens[rows] - stats["input_mean"]) / stats["input_scale"]
            current, past = model(torch.from_numpy(np.array(x)).to(device)).chunk(2)
            y, z = torch.from_numpy(target[rows]).to(device).chunk(2)
            loss = pose_objective(current, past, y, z, *scales, objective)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), CONFIG["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            if step % 100 == 0 or step + 1 == CONFIG["steps"]:
                losses.append(dict(step=step + 1, loss=float(loss.detach().cpu())))
            if (step + 1) % 500 == 0:
                print(
                    json.dumps(
                        dict(
                            arm=arm,
                            step=step + 1,
                            training_loss=float(loss.detach().cpu()),
                        )
                    ),
                    flush=True,
                )
        torch.save(
            {k: v.detach().cpu() for k, v in model.state_dict().items()},
            out / (arm + "-final.pt"),
        )
        validation_result, predicted = evaluate(
            model, tokens, target, metadata, validation, stats
        )
        np.save(out / (arm + "-VAL-prediction.npy"), predicted, allow_pickle=False)
        reports[arm] = dict(validation=validation_result, losses=losses)
        p.save_json(out / (arm + "-source-results.json"), reports[arm])
        print(
            json.dumps(dict(arm=arm, validation=validation_result["pooled"])),
            flush=True,
        )
    static = np.broadcast_to(target[train].mean(0), (validation.sum(), 17, 2))
    static_score = motion_scores(
        static,
        target[validation],
        metadata["record"][validation],
        metadata["end"][validation],
    )
    primary = reports["masked_spatial_motion"]["validation"]
    comparisons = [static_score] + [
        report["validation"]["pooled"]
        for arm, report in reports.items()
        if arm != "masked_spatial_motion"
    ]
    pooled = primary["pooled"]
    checks = dict(
        useful_articulation=pooled["record_centered_articulation_r2"] > 0.05,
        useful_delta=pooled["lag8_zero_reference_delta_r2"] > 0.05,
        each_room_positive=all(
            x["record_centered_articulation_r2"] > 0
            for x in primary["per_room"].values()
        ),
        better_control_articulation=all(
            pooled["record_centered_articulation_r2"]
            > x["record_centered_articulation_r2"]
            for x in comparisons
        ),
        better_control_nje=all(
            pooled["hip_relative_nje"] < x["hip_relative_nje"] for x in comparisons
        ),
    )
    p.verify_hashes(ROOT, registration["source_sha256"])
    p.verify_hashes(
        study,
        {arm + "-final.pt": sha for arm, sha in registration["encoder_sha256"].items()},
    )
    p.save_json(
        out / "results.json",
        dict(
            registration=registration,
            arms=reports,
            static_mean=static_score,
            gate=checks,
            passes=all(checks.values()),
            seconds=time.monotonic() - started,
        ),
    )
    p.save_json(
        out / "artifact-manifest.json",
        {str(x.relative_to(out)): p.digest(x) for x in out.rglob("*") if x.is_file()},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "raw", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    run(args.study, args.raw, args.out, args.device)


if __name__ == "__main__":
    main()
