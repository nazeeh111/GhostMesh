"""Fixed E02 source-development test of learned 120-label room adaptation.

Same encoder, solver, initialization, episodes and optimizer budget in both arms.
Ridge meta-learning prior art: https://arxiv.org/abs/1805.08136. No E04 access.
"""

from __future__ import annotations

import argparse
import copy
from hashlib import sha256
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
from scipy.linalg.blas import dgemm
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.calibrated_pose import CalibratedPose, pack_pose, ridge_adapt  # noqa:E402
from ghostmesh.metrics import pose_metrics  # noqa:E402

ACTIONS = ("A01", "A02", "A03", "A06", "A12", "A19")
ENDS = tuple(range(31, 293, 29))
CONFIG = dict(
    seed=2027,
    steps=1000,
    query=128,
    support=120,
    window=8,
    stride=8,
    validate_every=100,
    ridge=0.1,
    lr=0.001,
    weight_decay=0.01,
    raw_ridge_energy="Flattened source-standardized features divided by sqrt(10944/32); n*.1 ridge and fitted intercept",
    feature_stats="All unique source-training frames; channel mean/std over frames/subcarriers, floor .01",
    target_stats="All unique source-training frame labels, packed coordinate mean/std floor .01",
    held_room="E02",
    arms=["erm", "meta"],
    stats_scale_floor=0.01,
    support_actions=ACTIONS,
    support_ends=ENDS,
    target_support_people=["S11", "S12"],
    target_query_people=[f"S{i:02}" for i in range(13, 21)],
    source_validation_people=["S04", "S09", "S24", "S29"],
    source_validation_reference_people={"E01": ["S01", "S02"], "E03": ["S21", "S22"]},
    selection="Mean source-validation NJE over zero-shot and adapted modes, equal room weight; first minimum",
    meta_loss=".5 query-prior MSE + .5 adapted-query MSE; support labels enter only through inner solve",
    erm_loss="MSE over all120 support and128 query labels",
    episodes="Uniform source room; random2 train support people; query uniformly from different train people",
    scope="Source30 E01-E03 DEVELOPMENT only; labeled room calibration, different query people; no E04",
    limitation="Single seed/fold screen; compound learned encoder/adaptation recipe, no novelty or personal-calibration claim",
)


def digest(path):
    h = sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def index_records(records):
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate records")
    result = []
    for i, r in enumerate(records):
        room, person, action = r["id"].split("/")
        if (
            room not in ("E01", "E02", "E03")
            or r["frames"] != 297
            or r["offset"] != i * 297
        ):
            raise ValueError("Forbidden room or noncontiguous297-frame record")
        for end in range(7, 297, 8):
            result.append((i, end, room, person, action))
    return np.array(
        result,
        dtype=[
            ("record", "i4"),
            ("end", "i4"),
            ("room", "U3"),
            ("person", "U3"),
            ("action", "U3"),
        ],
    )


def references(records, room, people):
    rows = [
        (i, end)
        for i, r in enumerate(records)
        if r["id"].split("/")[0] == room
        and r["id"].split("/")[1] in people
        and r["id"].split("/")[2] in ACTIONS
        for end in ENDS
    ]
    if len(rows) != 120 or len(set(people)) != 2:
        raise ValueError("Expected2 people x6 actions x10 reference endpoints")
    return np.asarray(rows, dtype=np.int64)


def episodes(records, index, steps=1000):
    rng = np.random.default_rng(CONFIG["seed"])
    val = {f"S{x:02}" for x in (4, 9, 14, 19, 24, 29)}
    for _ in range(steps):
        room = str(rng.choice(["E01", "E03"]))
        people = sorted(set(index["person"][index["room"] == room]) - val)
        support_people = rng.choice(people, 2, replace=False).tolist()
        candidates = np.flatnonzero(
            (index["room"] == room)
            & ~np.isin(index["person"], sorted(val | set(support_people)))
        )
        q = rng.choice(candidates, CONFIG["query"], replace=False)
        yield references(records, room, support_people), q


def windows(frame, records, pairs):
    starts = np.array([records[int(i)]["offset"] for i in pairs[:, 0]])
    ix = starts[:, None] + pairs[:, 1, None] + np.arange(-7, 1)
    return np.array(frame[ix], dtype=np.float32)


def targets(pose, records, pairs):
    return np.array(
        pose[[records[int(i)]["offset"] + int(e) for i, e in pairs]], dtype=np.float32
    )


def source_stats(frame, pose, records):
    selected = [
        r
        for r in records
        if r["id"][:3] != "E02" and int(r["id"][5:7]) % 10 not in (4, 9)
    ]
    total = sum(r["frames"] for r in selected)
    sm, sq = np.zeros((12, 1)), np.zeros((12, 1))
    ys = []
    for r in selected:
        sl = slice(r["offset"], r["offset"] + r["frames"])
        x = np.asarray(frame[sl], dtype=np.float64)
        sm += x.sum((0, 2))[:, None]
        sq += (x * x).sum((0, 2))[:, None]
        ys.append(pack_pose(torch.from_numpy(np.array(pose[sl]))).numpy())
    mean = sm / (total * 114)
    scale = np.sqrt(np.maximum(sq / (total * 114) - mean * mean, 0))
    y = np.concatenate(ys)
    return mean, np.maximum(scale, 0.01), y.mean(0), np.maximum(y.std(0), 0.01)


def loss(model, x, y, arm, ns=120):
    z = model.encode(x)
    base = model.prior(z)
    if arm == "erm":
        return F.mse_loss(base, y)
    if arm != "meta":
        raise ValueError("Unknown arm")
    adapted = ridge_adapt(z[:ns], y[:ns], z[ns:], model.prior, CONFIG["ridge"])
    return 0.5 * F.mse_loss(base[ns:], y[ns:]) + 0.5 * F.mse_loss(adapted, y[ns:])


def short_score(p, y):
    m = pose_metrics(p, y, np.ones(y.shape[:2], bool))
    return {
        k: m[k]
        for k in ("normalized_joint_error", "pck_curve", "valid_frames", "valid_joints")
    }


def motion_score(p, y, record):
    result = short_score(p, y)
    rng = np.random.default_rng(90210)
    perm = np.arange(len(p))
    pr, yr = (
        p[:, [11, 12]].mean(1, keepdims=True),
        y[:, [11, 12]].mean(1, keepdims=True),
    )
    for name, pred, truth in (
        ("absolute", p, y),
        ("root", pr, yr),
        ("relative", p - pr, y - yr),
    ):
        numerator = denominator = 0.0
        for key in np.unique(record):
            ix = np.flatnonzero(record == key)
            pc, yc = pred[ix] - pred[ix].mean(0), truth[ix] - truth[ix].mean(0)
            numerator += float(((pc - yc) ** 2).sum())
            denominator += float((yc**2).sum())
            if name == "absolute":
                perm[ix] = rng.permutation(ix)
        result[name + "_motion_r2"] = (
            1 - numerator / denominator if denominator else None
        )
    result["within_record_shuffle_nje"] = short_score(p[perm], y)[
        "normalized_joint_error"
    ]
    return result


def stratified_score(prediction, truth, record, action):
    return {
        name: motion_score(prediction[pick], truth[pick], record[pick])
        for name, pick in [
            ("all", np.ones(len(truth), bool)),
            ("represented", np.isin(action, ACTIONS)),
            ("other_actions", ~np.isin(action, ACTIONS)),
        ]
    }


@torch.no_grad()
def predict(model, frame, pose, records, query, device, support=None):
    model.eval()
    if support is not None:
        sx = torch.from_numpy(windows(frame, records, support)).to(device)
        sz = model.encode(sx)
        sy = model.standardize_targets(
            torch.from_numpy(targets(pose, records, support)).to(device)
        )
    out = []
    for start in range(0, len(query), 256):
        qz = model.encode(
            torch.from_numpy(windows(frame, records, query[start : start + 256])).to(
                device
            )
        )
        yp = (
            model.prior(qz)
            if support is None
            else ridge_adapt(sz, sy, qz, model.prior, CONFIG["ridge"])
        )
        out.append(model.decode_targets(yp).cpu().numpy())
    return np.concatenate(out)


def validation(model, frame, pose, records, index, device):
    rows = []
    for room, people in [("E01", ["S01", "S02"]), ("E03", ["S21", "S22"])]:
        selected = (index["room"] == room) & np.isin(
            index["person"], ["S04", "S09", "S24", "S29"]
        )
        q = np.column_stack((index["record"][selected], index["end"][selected]))
        y = targets(pose, records, q)
        s = references(records, room, people)
        rows.append(
            {
                mode: short_score(
                    predict(model, frame, pose, records, q, device, refs), y
                )
                for mode, refs in [("zero", None), ("adapted", s)]
            }
        )
    score = np.mean(
        [r[m]["normalized_joint_error"] for r in rows for m in ("zero", "adapted")]
    )
    return float(score), rows


def benchmark(device):
    torch.set_num_threads(1)
    torch.manual_seed(2027)
    model = CalibratedPose().to(device)
    x = torch.randn(248, 8, 12, 114, device=device)
    y = torch.randn(248, 36, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    times = []
    for _ in range(4):
        started = time.monotonic()
        opt.zero_grad()
        v = loss(model, x, y, "meta")
        v.backward()
        opt.step()
        if device == "mps":
            torch.mps.synchronize()
        times.append(time.monotonic() - started)
    result = dict(
        device=device,
        seconds=times,
        loss=float(v.detach().cpu()),
        parameters=sum(p.numel() for p in model.parameters()),
    )
    print(json.dumps(result), flush=True)
    return result


def run(cache, base, out, device):
    if out.exists():
        raise FileExistsError(out)
    started = time.monotonic()
    torch.set_num_threads(1)
    receipt = json.loads((cache / "receipt.json").read_text())
    original = json.loads((base / "receipt.json").read_text())
    records = receipt["records"]
    if (
        records != original["records"]
        or len(records) != 810
        or receipt["base_receipt_sha256"] != digest(base / "receipt.json")
    ):
        raise ValueError("Source30 receipt or record mismatch")
    index = index_records(records)
    bindings = {
        str(cache / "receipt.json"): digest(cache / "receipt.json"),
        str(base / "receipt.json"): digest(base / "receipt.json"),
    }
    for path, expected in [
        (cache / "frame.npy", receipt["arrays"]["frame"]),
        (base / "pose.npy", original["arrays"]["pose"]),
        (base / "valid.npy", original["arrays"]["valid"]),
    ]:
        bindings[str(path)] = digest(path)
        if bindings[str(path)] != expected:
            raise ValueError("Input hash mismatch: " + str(path))
    frame, pose, valid = [
        np.load(p, mmap_mode="r", allow_pickle=False)
        for p in (cache / "frame.npy", base / "pose.npy", base / "valid.npy")
    ]
    if (
        frame.shape != (240570, 12, 114)
        or pose.shape != (240570, 17, 2)
        or not valid.all()
    ):
        raise ValueError("Unexpected feature/target shape or invalid labels")
    out.mkdir(parents=True)
    sources = [
        Path(__file__),
        ROOT / "ghostmesh/calibrated_pose.py",
        ROOT / "ghostmesh/metrics.py",
    ]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    for p in sources:
        dest = out / "executed-source" / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    schedule = list(episodes(records, index))
    np.savez(
        out / "episodes.npz",
        support=np.stack([s for s, _ in schedule]),
        query=np.stack([q for _, q in schedule]),
    )
    torch.manual_seed(CONFIG["seed"])
    initial = CalibratedPose()
    initial.fit_source_stats(*source_stats(frame, pose, records))
    torch.save(initial.state_dict(), out / "initial.pt")
    registration = dict(
        config=CONFIG,
        source_sha256=hashes,
        inputs_sha256=bindings,
        init_sha256=digest(out / "initial.pt"),
        episodes_sha256=digest(out / "episodes.npz"),
        software=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            numpy=np.__version__,
            device=device,
        ),
        counts=dict(
            train_people=16,
            sourceval_people=4,
            held_people=10,
            train_windows=15984,
            sourceval_windows=3996,
            held_windows=9990,
            optimizer_steps_per_arm=1000,
            input_examples_per_step=248,
            reference_labels_per_step=120,
            query_labels_per_step=128,
            direct_loss_labels_erm=248000,
            direct_loss_labels_meta=128000,
            inner_fit_labels_meta=120000,
        ),
    )
    support_pairs = np.concatenate([s for s, _ in schedule])
    query_pairs = np.concatenate(
        [np.column_stack((index["record"][q], index["end"][q])) for _, q in schedule]
    )
    registration["counts"].update(
        unique_support_labels=int(len(np.unique(support_pairs, axis=0))),
        unique_query_labels=int(len(np.unique(query_pairs, axis=0))),
        unique_combined_labels=int(
            len(np.unique(np.concatenate((support_pairs, query_pairs)), axis=0))
        ),
        source_stats_unique_frames=128304,
        sourceval_reference_labels=240,
        target_reference_labels=120,
    )
    save(out / "registration.json", registration)
    final = {}
    try:
        for arm in CONFIG["arms"]:
            model = copy.deepcopy(initial).to(device)
            opt = torch.optim.AdamW(
                model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
            )
            history, best = [], float("inf")
            for step, (s, qi) in enumerate(schedule, 1):
                q = np.column_stack((index["record"][qi], index["end"][qi]))
                pairs = np.concatenate((s, q))
                x = torch.from_numpy(windows(frame, records, pairs)).to(device)
                y = model.standardize_targets(
                    torch.from_numpy(targets(pose, records, pairs)).to(device)
                )
                model.train()
                opt.zero_grad()
                value = loss(model, x, y, arm)
                if not torch.isfinite(value):
                    raise ValueError("Nonfinite training loss")
                value.backward()
                opt.step()
                if step % 100 == 0:
                    score, detail = validation(
                        model, frame, pose, records, index, device
                    )
                    row = dict(
                        step=step,
                        loss=float(value.detach().cpu()),
                        selection_nje=score,
                        rooms=detail,
                    )
                    history.append(row)
                    if score < best:
                        best = score
                        torch.save(model.state_dict(), out / (arm + "-best.pt"))
                    save(out / (arm + "-history.json"), history)
                    print(json.dumps(dict(arm=arm, **row)), flush=True)
            torch.save(model.state_dict(), out / (arm + "-last.pt"))
        for arm in CONFIG["arms"]:
            model = copy.deepcopy(initial).to(device)
            model.load_state_dict(
                torch.load(
                    out / (arm + "-best.pt"), map_location=device, weights_only=True
                )
            )
            selected = index["room"] == "E02"
            q = np.column_stack((index["record"][selected], index["end"][selected]))
            y = targets(pose, records, q)
            common = ~np.isin(index["person"][selected], ["S11", "S12"])
            support = references(records, "E02", ["S11", "S12"])
            zero = predict(model, frame, pose, records, q, device)
            adapted = predict(model, frame, pose, records, q[common], device, support)
            np.savez(
                out / (arm + "-predictions.npz"),
                pairs=q,
                target=y,
                zero=zero,
                common=common,
                adapted=adapted,
            )
            seq = index["record"][selected]
            history = json.loads((out / (arm + "-history.json")).read_text())
            final[arm] = dict(
                best_step=min(history, key=lambda r: r["selection_nje"])["step"],
                zero_all=motion_score(zero, y, seq),
                zero_common=motion_score(zero[common], y[common], seq[common]),
                adapted={},
            )
            final[arm]["adapted"] = stratified_score(
                adapted, y[common], seq[common], index["action"][selected][common]
            )
            final[arm]["zero_common_strata"] = stratified_score(
                zero[common], y[common], seq[common], index["action"][selected][common]
            )
            valpick = (index["room"] != "E02") & np.isin(
                index["person"], ["S04", "S09", "S24", "S29"]
            )
            vq = np.column_stack((index["record"][valpick], index["end"][valpick]))
            np.savez(
                out / (arm + "-sourceval-predictions.npz"),
                pairs=vq,
                target=targets(pose, records, vq),
                zero=predict(model, frame, pose, records, vq, device),
                adapted=np.concatenate(
                    [
                        predict(
                            model,
                            frame,
                            pose,
                            records,
                            vq[index["room"][valpick] == room],
                            device,
                            references(records, room, people),
                        )
                        for room, people in [
                            ("E01", ["S01", "S02"]),
                            ("E03", ["S21", "S22"]),
                        ]
                    ]
                ),
            )
        # Both nonneural controls receive exactly the same120 reference poses.
        qcommon, yc = q[common], y[common]
        sy = targets(pose, records, support)
        target_mean = np.broadcast_to(sy.mean(0), yc.shape).copy()
        source_mean = initial.decode_targets(torch.zeros(len(yc), 36)).numpy()

        def raw_features(pairs):
            x = (
                windows(frame, records, pairs) - initial.feature_mean.numpy()
            ) / initial.feature_scale.numpy()
            return x.reshape(len(x), -1).astype(np.float64) / np.sqrt(8 * 12 * 114 / 32)

        sx, qx = raw_features(support), raw_features(qcommon)
        xm, ym = sx.mean(0), sy.reshape(120, 34).mean(0)
        xc = sx - xm
        gram = dgemm(1.0, xc, xc.T) + 120 * CONFIG["ridge"] * np.eye(120)
        dual = np.linalg.solve(gram, sy.reshape(120, 34) - ym)
        weight = dgemm(1.0, xc.T, dual)
        raw = (dgemm(1.0, qx - xm, weight) + ym).reshape(-1, 17, 2)
        np.savez(
            out / "controls.npz",
            pairs=qcommon,
            source_mean=source_mean,
            target_mean=target_mean,
            raw_ridge=raw,
            raw_xmean=xm,
            raw_ymean=ym,
            raw_weight=weight,
        )
        final["controls"] = {
            name: stratified_score(
                p, yc, seq[common], index["action"][selected][common]
            )
            for name, p in [
                ("source_mean", source_mean),
                ("target_mean", target_mean),
                ("raw_ridge", raw),
            ]
        }
        parity = {}
        for arm in CONFIG["arms"]:
            loaded = copy.deepcopy(initial).to(device)
            loaded.load_state_dict(
                torch.load(
                    out / (arm + "-best.pt"), map_location=device, weights_only=True
                )
            )
            stored = np.load(out / (arm + "-predictions.npz"), allow_pickle=False)
            replay_zero = predict(loaded, frame, pose, records, stored["pairs"], device)
            replay_adapt = predict(
                loaded,
                frame,
                pose,
                records,
                stored["pairs"][stored["common"]],
                device,
                support,
            )
            differences = dict(
                zero_max_abs=float(np.max(np.abs(replay_zero - stored["zero"]))),
                adapted_max_abs=float(np.max(np.abs(replay_adapt - stored["adapted"]))),
            )
            if max(differences.values()) > 1e-6:
                raise ValueError("Saved-model prediction replay differs")
            parity[arm] = differences
        save(
            out / "verification.json",
            dict(
                saved_model_prediction_parity=parity,
                tolerance=1e-6,
                source_input_rehash="Required below before completed results",
            ),
        )
        for path, expected in {
            **bindings,
            **{str(ROOT / k): v for k, v in hashes.items()},
        }.items():
            if digest(path) != expected:
                raise ValueError("Frozen input/source changed")
        save(
            out / "results.json",
            dict(
                registration=registration,
                results=final,
                seconds=time.monotonic() - started,
            ),
        )
        save(
            out / "final_manifest.json",
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
                type=type(exc).__name__,
                error=str(exc),
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
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if args.benchmark:
        benchmark(args.device)
    elif args.out:
        run(args.cache, args.base, args.out, args.device)
    else:
        parser.error("--out is required for data fitting")
