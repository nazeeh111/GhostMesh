"""Paired absolute/increment supervision on source rooms, with E02 held out.

This is a motion-observability experiment, not a validated pose model. A learned
potential's difference has no information about the unknown initial skeleton.
All tuning/selection uses E01/E03 validation people. No E04 data is accepted.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.calibrated_pose import CalibratedPose, pack_pose  # noqa:E402
from scripts.domain_meta_probe import digest, save, source_stats, targets, windows  # noqa:E402

CONFIG = dict(
    seed=2027,
    steps=1000,
    batch_pairs=128,
    window=8,
    lag=8,
    stride=8,
    validate_every=100,
    lr=0.001,
    weight_decay=0.01,
    gradient_clip=1.0,
    arms=["absolute", "increment"],
    held_room="E02",
    source_validation_people=["S04", "S09", "S24", "S29"],
    selection="Max equal-room source-validation relative record-centered increment R2; first maximum",
    targets="Packed root-relative34+root2. SAME source-pose mean and source-train increment SD floor.001 in both arms",
    absolute_loss="MSE at both endpoints, 256 labeled endpoints per step",
    increment_loss="MSE of F(current)-F(previous) against lag8 pose difference; 256 endpoints per step",
    schedule="Uniform source room, uniform128 canonical pairs without replacement; identical both arms",
    scope="One seed E02 source DEVELOPMENT screen; motion only, no reserved E04, no target fitting",
    caveat="Increment objective cannot identify the DC/initial pose. Neither target recording means nor previous target poses enter predictions",
)
DTYPE = [
    ("record", "i4"),
    ("end", "i4"),
    ("room", "U3"),
    ("person", "U3"),
    ("action", "U3"),
]


def index_pairs(records):
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate record")
    rows = []
    for i, r in enumerate(records):
        room, person, action = r["id"].split("/")
        if (
            room not in ("E01", "E02", "E03")
            or r["frames"] != 297
            or r["offset"] != i * 297
        ):
            raise ValueError("Only contiguous source30 records accepted")
        for end in range(15, 297, 8):
            rows.append((i, end, room, person, action))
    return np.array(rows, dtype=DTYPE)


def roles(index):
    val_people = np.isin(index["person"], CONFIG["source_validation_people"])
    held = index["room"] == "E02"
    return ~held & ~val_people, ~held & val_people, held


def pair_ids(index):
    return np.column_stack((index["record"], index["end"])).astype(np.int64)


def previous(pairs):
    pairs = np.asarray(pairs)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or np.any(pairs[:, 1] < 15):
        raise ValueError("Lag8 requires both8-frame windows inside the record")
    out = pairs.copy()
    out[:, 1] -= 8
    return out


def schedule(index):
    train, _, _ = roles(index)
    rng = np.random.default_rng(CONFIG["seed"])
    for _ in range(CONFIG["steps"]):
        room = str(rng.choice(["E01", "E03"]))
        candidates = np.flatnonzero(train & (index["room"] == room))
        yield rng.choice(candidates, CONFIG["batch_pairs"], replace=False)


def packed(pose):
    return pack_pose(torch.as_tensor(np.array(pose))).numpy()


def initial_stats(frame, pose, records, index):
    fm, fs, ym, _ = source_stats(frame, pose, records)
    train, _, _ = roles(index)
    pairs = pair_ids(index[train])
    delta = packed(targets(pose, records, pairs)) - packed(
        targets(pose, records, previous(pairs))
    )
    # Zero is the known no-motion origin; do not subtract a fitted delta mean.
    return fm, fs, ym, np.maximum(delta.std(0), 0.001)


def training_loss(prediction, standardized_target, arm):
    if prediction.shape != standardized_target.shape or len(prediction) % 2:
        raise ValueError("Expected matching current and previous halves")
    if arm == "absolute":
        return F.mse_loss(prediction, standardized_target)
    if arm != "increment":
        raise ValueError("Unknown objective")
    p, q = prediction.chunk(2)
    y, z = standardized_target.chunk(2)
    return F.mse_loss(p - q, y - z)


def component_scores(prediction, truth, record):
    """Raw zero-reference R2 plus explicitly descriptive per-record centered R2."""
    result = {}
    for name, sl in (
        ("relative", slice(0, 34)),
        ("root", slice(34, 36)),
        ("packed", slice(0, 36)),
    ):
        p, y = np.asarray(prediction[:, sl], float), np.asarray(truth[:, sl], float)
        denominator = np.square(y).sum()
        centered_p, centered_y = p.copy(), y.copy()
        for key in np.unique(record):
            ix = record == key
            centered_p[ix] -= p[ix].mean(0)
            centered_y[ix] -= y[ix].mean(0)
        centered_denominator = np.square(centered_y).sum()
        norms = np.linalg.norm(p, axis=1) * np.linalg.norm(y, axis=1)
        nonzero = norms > 1e-12
        result[name] = dict(
            zero_reference_r2=float(1 - np.square(p - y).sum() / denominator)
            if denominator
            else None,
            record_centered_r2=float(
                1 - np.square(centered_p - centered_y).sum() / centered_denominator
            )
            if centered_denominator
            else None,
            prediction_to_target_energy=float(np.square(p).sum() / denominator)
            if denominator
            else None,
            mean_cosine=float(((p * y).sum(1)[nonzero] / norms[nonzero]).mean())
            if nonzero.any()
            else 0.0,
            nonzero_cosine_pairs=int(nonzero.sum()),
            target_rms=float(np.sqrt(np.square(y).mean())),
        )
    return result


def score_with_controls(prediction, truth, record):
    rng = np.random.default_rng(90210)
    shuffled, shifted = prediction.copy(), prediction.copy()
    for key in np.unique(record):
        ix = np.flatnonzero(record == key)
        shuffled[ix] = prediction[rng.permutation(ix)]
        shifted[ix] = np.roll(prediction[ix], len(ix) // 2, axis=0)
    return dict(
        aligned=component_scores(prediction, truth, record),
        within_record_shuffle=component_scores(shuffled, truth, record),
        half_record_circular_shift=component_scores(shifted, truth, record),
        zero_increment=component_scores(np.zeros_like(prediction), truth, record),
    )


@torch.no_grad()
def predict_pair(model, frame, records, pairs, device):
    model.eval()
    current, past = [], []
    for start in range(0, len(pairs), 128):
        now = pairs[start : start + 128]
        both = np.concatenate((now, previous(now)))
        x = torch.from_numpy(windows(frame, records, both)).to(device)
        # Keep packed coordinates, including the hip-root factorization.
        packed_pred = model(x) * model.target_scale + model.target_mean
        a, b = packed_pred.cpu().numpy().copy().reshape(2, len(now), 36)
        current.append(a)
        past.append(b)
    return np.concatenate(current), np.concatenate(past)


def validation(model, frame, pose, records, index, device):
    _, val, _ = roles(index)
    details = {}
    for room in ("E01", "E03"):
        selected = val & (index["room"] == room)
        pairs = pair_ids(index[selected])
        a, b = predict_pair(model, frame, records, pairs, device)
        y = packed(targets(pose, records, pairs)) - packed(
            targets(pose, records, previous(pairs))
        )
        details[room] = component_scores(a - b, y, pairs[:, 0])
    return float(
        np.mean([d["relative"]["record_centered_r2"] for d in details.values()])
    ), details


def run(cache, base, out, device):
    if out.exists():
        raise FileExistsError(out)
    started = time.monotonic()
    torch.set_num_threads(1)
    receipt = json.loads((cache / "receipt.json").read_text())
    raw = json.loads((base / "receipt.json").read_text())
    records = receipt["records"]
    if (
        len(records) != 810
        or records != raw["records"]
        or receipt["base_receipt_sha256"] != digest(base / "receipt.json")
    ):
        raise ValueError("Source30 receipt mismatch")
    index = index_pairs(records)
    bindings = {
        str(p): digest(p) for p in (cache / "receipt.json", base / "receipt.json")
    }
    for p, expected in (
        (cache / "frame.npy", receipt["arrays"]["frame"]),
        (base / "pose.npy", raw["arrays"]["pose"]),
        (base / "valid.npy", raw["arrays"]["valid"]),
    ):
        bindings[str(p)] = digest(p)
        if bindings[str(p)] != expected:
            raise ValueError("Input hash mismatch")
    frame, pose, valid = [
        np.load(p, mmap_mode="r", allow_pickle=False)
        for p in (cache / "frame.npy", base / "pose.npy", base / "valid.npy")
    ]
    if (
        frame.shape != (240570, 12, 114)
        or pose.shape != (240570, 17, 2)
        or not valid.all()
    ):
        raise ValueError("Source arrays invalid")
    out.mkdir(parents=True)
    sources = [
        Path(__file__),
        ROOT / "scripts/domain_meta_probe.py",
        ROOT / "ghostmesh/calibrated_pose.py",
    ]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    for p in sources:
        dest = out / "executed-source" / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    batches = np.stack(list(schedule(index)))
    np.save(out / "schedule.npy", batches, allow_pickle=False)
    torch.manual_seed(CONFIG["seed"])
    initial = CalibratedPose()
    initial.fit_source_stats(*initial_stats(frame, pose, records, index))
    torch.save(initial.state_dict(), out / "initial.pt")
    masks = roles(index)
    registration = dict(
        config=CONFIG,
        sources_sha256=hashes,
        inputs_sha256=bindings,
        init_sha256=digest(out / "initial.pt"),
        schedule_sha256=digest(out / "schedule.npy"),
        counts=dict(
            zip(
                ("train_pairs", "validation_pairs", "held_pairs"),
                [int(m.sum()) for m in masks],
            )
        ),
        unique_training_pairs=int(len(np.unique(batches))),
        processed_input_windows_per_arm=int(batches.size * 2),
        parameters=sum(p.numel() for p in initial.parameters()),
        software=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            numpy=np.__version__,
            device=device,
        ),
    )
    save(out / "registration.json", registration)
    try:
        for arm in CONFIG["arms"]:
            model = copy.deepcopy(initial).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
            )
            history, best = [], -float("inf")
            for step, ids in enumerate(batches, 1):
                pairs = pair_ids(index[ids])
                both = np.concatenate((pairs, previous(pairs)))
                x = torch.from_numpy(windows(frame, records, both)).to(device)
                y = model.standardize_targets(
                    torch.from_numpy(targets(pose, records, both)).to(device)
                )
                model.train()
                optimizer.zero_grad()
                value = training_loss(model(x), y, arm)
                if not torch.isfinite(value):
                    raise ValueError("Nonfinite loss")
                value.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), CONFIG["gradient_clip"]
                )
                optimizer.step()
                if step % CONFIG["validate_every"] == 0:
                    score, detail = validation(
                        model, frame, pose, records, index, device
                    )
                    row = dict(
                        step=step,
                        loss=float(value.detach().cpu()),
                        selection_r2=score,
                        rooms=detail,
                    )
                    history.append(row)
                    if score > best:
                        best = score
                        torch.save(model.state_dict(), out / (arm + "-best.pt"))
                    save(out / (arm + "-history.json"), history)
                    print(
                        json.dumps(
                            dict(
                                arm=arm,
                                step=step,
                                loss=row["loss"],
                                validation_r2=score,
                            )
                        ),
                        flush=True,
                    )
            torch.save(model.state_dict(), out / (arm + "-last.pt"))
        # Both arms finish and select from source validation BEFORE held-room scoring.
        results, parity = {}, {}
        for arm in CONFIG["arms"]:
            model = copy.deepcopy(initial).to(device)
            model.load_state_dict(
                torch.load(
                    out / (arm + "-best.pt"), map_location=device, weights_only=True
                )
            )
            results[arm] = {}
            for split, mask in zip(("source_validation", "held_room"), masks[1:]):
                subset = index[mask]
                pairs = pair_ids(subset)
                a, b = predict_pair(model, frame, records, pairs, device)
                y = packed(targets(pose, records, pairs)) - packed(
                    targets(pose, records, previous(pairs))
                )
                np.savez_compressed(
                    out / (arm + "-" + split + ".npz"),
                    pairs=pairs,
                    current=a,
                    past=b,
                    target_delta=y,
                )
                results[arm][split] = score_with_controls(a - b, y, pairs[:, 0])
                reloaded = copy.deepcopy(initial).to(device)
                reloaded.load_state_dict(
                    torch.load(
                        out / (arm + "-best.pt"), map_location=device, weights_only=True
                    )
                )
                replay_a, replay_b = predict_pair(
                    reloaded, frame, records, pairs, device
                )
                diff = max(
                    float(abs(a - replay_a).max()), float(abs(b - replay_b).max())
                )
                if diff > 1e-6:
                    raise ValueError("Saved prediction replay differs")
                parity[arm + "/" + split] = diff
            history = json.loads((out / (arm + "-history.json")).read_text())
            results[arm]["selected_step"] = max(
                history, key=lambda h: h["selection_r2"]
            )["step"]
        for p, expected in {
            **bindings,
            **{str(ROOT / k): v for k, v in hashes.items()},
        }.items():
            if digest(p) != expected:
                raise ValueError("Inputs/source changed during experiment")
        save(
            out / "verification.json",
            dict(prediction_replay_max_abs=parity, input_rehash_passed=True),
        )
        save(
            out / "results.json",
            dict(
                registration=registration,
                results=results,
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
    except BaseException as exc:
        save(
            out / "failure.json",
            dict(
                error=str(exc),
                type=type(exc).__name__,
                seconds=time.monotonic() - started,
            ),
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path, default=ROOT / "work/domain-source30-observation-v1"
    )
    parser.add_argument(
        "--base", type=Path, default=ROOT / "work/domain-source30-rf-v1"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    run(args.cache, args.base, args.out, args.device)
