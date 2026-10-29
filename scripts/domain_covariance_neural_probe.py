"""Matched absolute/motion learning on physically invariant covariance histories.

Fixed E02 source-development fold, all27actions. Same compact two-head network,
initialization, input/label schedule and budget. No E04/calibration/query-label fit.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.covariance_pose import CovariancePose  # noqa:E402
from ghostmesh.metrics import pose_metrics  # noqa:E402
from scripts.domain_covariance_cached_probe import load_inputs  # noqa:E402
from scripts.domain_covariance_source_probe import fold_masks  # noqa:E402
from scripts.domain_background_probe import digest  # noqa:E402
from scripts.report_covariance_source_probe import measure  # noqa:E402

CONFIG = dict(
    seed=2027,
    steps=1000,
    batch=128,
    validate_every=100,
    lr=0.001,
    weight_decay=0.01,
    gradient_clip=1.0,
    arms=["absolute", "motion"],
    held_room="E02",
    input="All2736 covarianceinvariant features, ordered8timesx19binsx6lagsx3centeredlogeigenvalues; reshapeN8x18channelsx19bins, no preliminary projection",
    context="Nominal10Hz, maximum causal40rawframe/3.9s span, same frozen validity/endpoints as full27RFF controls",
    labels="Both current17x2relativepose and past17x2relativepose(t-8)=providedcurrent-provideddelta, from same recording. Past is auxiliary retrodiction from current RF history, not a past-state input",
    feature_statistics="AllsourceTRAINwindow/time/frequency occurrences,18channelmean/SD floor.01. Overlapping raw histories are intentionally weighted repeatedly",
    pose_statistics="Concatenated current+past TRAINlabel occurrences (duplicates intentionally weighted),34coordinate mean/SD floor.01",
    delta_statistics="TRAINlag8 displacement SD floor.001, ZERO origin, no fitted delta-mean subtraction",
    absolute_objective=".5 current standardizedpose MSE+.5 past standardizedpose MSE",
    motion_objective=".5 current standardizedpose MSE+.5 decoded(current-past) standardizedbydeltaSD vsGTdelta MSE",
    selection="Minimum pooled sourcevalidation CURRENTarticulation NJE every100steps; same criterion botharms; firstminimum; keepbestANDlast",
    source_validation_people=["S04", "S09", "S24", "S29"],
    schedule="UniformsourceE01/E03 room,128trainingcurrentwindows withoutreplacement perstep; identical botharms,256endpointlabels perstep",
    scope="SOURCE DEVELOPMENT full27actions/all30sourcepeople E01-E03, fixedE02 held fold/seed2027; no E04 or targetcalibration",
    limits="Hiprootrelativearticulation only, not absolutehipposition/3D. Invariantgeometry assumes fixedreceivercongruence and cannot identify every staticpose. Onefold/seeddevelopment, no new finalefficacy claim",
)


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def observations(feature, ids):
    x = np.array(feature[ids], dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 2736:
        raise ValueError("Expected full2736feature columns")
    return (
        x.reshape(-1, 8, 19, 6, 3)
        .transpose(0, 1, 3, 4, 2)
        .reshape(-1, 8, 18, 19)
        .copy()
    )


def source_stats(feature, meta, train):
    ids = np.flatnonzero(train)
    total, squares = np.zeros(18), np.zeros(18)
    for start in range(0, len(ids), 128):
        x = observations(feature, ids[start : start + 128]).astype(np.float64)
        total += x.sum((0, 1, 3))
        squares += np.square(x).sum((0, 1, 3))
    count = len(ids) * 8 * 19
    fm = total / count
    fs = np.maximum(np.sqrt(np.maximum(squares / count - fm * fm, 0)), 0.01)
    y = meta["pose"][train].reshape(-1, 34)
    delta = meta["delta"][train].reshape(-1, 34)
    both = np.concatenate((y, y - delta))
    return (
        fm[:, None],
        fs[:, None],
        both.mean(0),
        np.maximum(both.std(0), 0.01),
        np.maximum(delta.std(0), 0.001),
    )


def schedule(meta, train):
    rng = np.random.default_rng(CONFIG["seed"])
    for _ in range(CONFIG["steps"]):
        room = str(rng.choice(["E01", "E03"]))
        yield rng.choice(
            np.flatnonzero(train & (meta["room"] == room)),
            CONFIG["batch"],
            replace=False,
        )


def label_coverage(meta, train, batches):
    indices = np.flatnonzero(train)

    def label_ids(ids):
        return {
            (str(meta["record"][i]), int(meta["end"][i] + lag))
            for i in ids
            for lag in (0, -8)
        }

    # Exact union of RF frames actually touched by outer8 contexts and the0/6lags;
    # the conservative validity guard spans40frames but does not use every frame.
    offsets = {-outer - lag for outer in range(8) for lag in (0, 1, 2, 4, 8, 16, 32)}
    frames = {}
    for i in indices:
        key = str(meta["record"][i])
        mask = frames.setdefault(key, np.zeros(297, bool))
        mask[[int(meta["end"][i]) + offset for offset in offsets]] = True
    used = np.unique(batches)
    return dict(
        train_current_rows=len(indices),
        train_pose_stat_occurrences=2 * len(indices),
        train_pose_stat_unique_endpoints=len(label_ids(indices)),
        source_feature_stat_unique_RFframes=sum(int(v.sum()) for v in frames.values()),
        scheduled_unique_current_rows=len(used),
        scheduled_unique_endpoint_labels=len(label_ids(used)),
        processed_RFwindows_per_arm=int(batches.size),
        processed_endpoint_labels_per_arm=int(2 * batches.size),
    )


def summary(prediction, delta, meta, ids):
    y, dy, record = meta["pose"][ids], meta["delta"][ids], meta["record"][ids]
    result = measure(prediction, delta, y, dy, record)
    for grouping in ("person", "action"):
        values = meta[grouping][ids]
        result["per_" + grouping] = {
            str(key): measure(
                prediction[pick], delta[pick], y[pick], dy[pick], record[pick]
            )
            for key in np.unique(values)
            for pick in [values == key]
        }
    return result


def shuffle_indices(record):
    rng = np.random.default_rng(90210)
    perm = np.arange(len(record))
    for key in np.unique(record):
        ix = np.flatnonzero(record == key)
        perm[ix] = rng.permutation(ix)
    return perm


@torch.no_grad()
def predict(model, feature, ids, device):
    model.eval()
    current, past = [], []
    for start in range(0, len(ids), 256):
        x = torch.from_numpy(observations(feature, ids[start : start + 256])).to(device)
        p = model(x)
        current.append(model.decode(p["current"]).cpu().numpy())
        past.append(model.decode(p["past"]).cpu().numpy())
    return np.concatenate(current), np.concatenate(past)


@torch.no_grad()
def validation(model, feature, meta, ids, device):
    current, past = predict(model, feature, ids, device)
    metrics = pose_metrics(current, meta["pose"][ids], np.ones(current.shape[:2], bool))
    return float(metrics["normalized_joint_error"]), dict(
        normalized_joint_error=metrics["normalized_joint_error"],
        pck_curve=metrics["pck_curve"],
        valid_frames=metrics["valid_frames"],
        motion=measure(
            current,
            current - past,
            meta["pose"][ids],
            meta["delta"][ids],
            meta["record"][ids],
        ),
    )


def benchmark(device):
    torch.set_num_threads(1)
    torch.manual_seed(CONFIG["seed"])
    model = CovariancePose().to(device)
    x = torch.randn(128, 8, 18, 19, device=device)
    y = torch.randn(128, 17, 2, device=device) * 0.05
    past = y + torch.randn_like(y) * 0.01
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    times = []
    for _ in range(5):
        started = time.monotonic()
        optimizer.zero_grad()
        loss = model.loss(model(x), y, past, "motion")
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite benchmark loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not all(
            p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
        ):
            raise ValueError("Nonfinite benchmark gradient")
        optimizer.step()
        if device == "mps":
            torch.mps.synchronize()
        times.append(time.monotonic() - started)
    result = dict(
        device=device,
        parameters=sum(p.numel() for p in model.parameters()),
        seconds=times,
        finite_loss=float(loss.detach().cpu()),
        scope="Synthetic engineering only, not accuracy/real-data latency",
    )
    print(json.dumps(result), flush=True)
    return result


def baseline_bindings(baseline):
    names = ["results.json", "registration.json", "targets-and-metadata.npz"]
    names += [
        "E02-" + arm + "-model-prediction.npz"
        for arm in ("untreated_covariance", "log_shape_tangent", "congruence_dynamics")
    ]
    return {name: digest(baseline / name) for name in names}


def run(cache, baseline, out, device):
    if out.exists():
        raise FileExistsError(out)
    started = time.monotonic()
    torch.set_num_threads(1)
    receipt, meta, features = load_inputs(cache)
    feature = features["congruence_dynamics"]
    train, val, held = fold_masks(meta["room"], meta["person"], "E02")
    if tuple(len(set(meta["person"][x])) for x in (train, val, held)) != (16, 4, 10):
        raise ValueError("Unexpectedpersonroles")
    bbindings = baseline_bindings(baseline)
    baseline_meta = np.load(baseline / "targets-and-metadata.npz", allow_pickle=False)
    for key in meta:
        np.testing.assert_array_equal(meta[key], baseline_meta[key])
    source_paths = [
        Path(__file__),
        ROOT / "ghostmesh/covariance_pose.py",
        ROOT / "scripts/domain_covariance_cached_probe.py",
        ROOT / "scripts/domain_covariance_full_cache.py",
        ROOT / "scripts/domain_covariance_source_probe.py",
        ROOT / "scripts/domain_background_probe.py",
        ROOT / "scripts/report_covariance_source_probe.py",
        ROOT / "ghostmesh/metrics.py",
    ]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    bindings = {
        name: digest(cache / name)
        for name in [
            "receipt.json",
            "artifact_manifest.json",
            *receipt["arrays_sha256"],
        ]
    }
    out.mkdir(parents=True)
    for source in source_paths:
        destination = out / "executed-source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    batches = np.stack(list(schedule(meta, train)))
    np.save(out / "schedule.npy", batches, allow_pickle=False)
    torch.manual_seed(CONFIG["seed"])
    initial = CovariancePose()
    initial.fit_source_stats(*source_stats(feature, meta, train))
    torch.save(initial.state_dict(), out / "initial.pt")
    coverage = label_coverage(meta, train, batches)
    registration = dict(
        config=CONFIG,
        source_sha256=hashes,
        cache_sha256=bindings,
        baseline_sha256=bbindings,
        initialization_sha256=digest(out / "initial.pt"),
        schedule_sha256=digest(out / "schedule.npy"),
        counts={
            **coverage,
            "train": int(train.sum()),
            "validation": int(val.sum()),
            "held": int(held.sum()),
        },
        parameters=sum(p.numel() for p in initial.parameters()),
        software=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            numpy=np.__version__,
            device=device,
        ),
        baseline_scope="RFFcontrols use samefull27validrows/personsplit/history but distinct fixedbasis/ridge fitting budgets; onlytwo neuralarms are matchedobjectivecontrols",
    )
    save(out / "registration.json", registration)
    try:
        val_ids = np.flatnonzero(val)
        for arm in CONFIG["arms"]:
            model = copy.deepcopy(initial).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
            )
            history = []
            best = float("inf")
            for step, ids in enumerate(batches, 1):
                x = torch.from_numpy(observations(feature, ids)).to(device)
                y = torch.as_tensor(
                    np.array(meta["pose"][ids]), dtype=torch.float32, device=device
                )
                dy = torch.as_tensor(
                    np.array(meta["delta"][ids]), dtype=torch.float32, device=device
                )
                model.train()
                optimizer.zero_grad()
                value = model.loss(model(x), y, y - dy, arm)
                if not torch.isfinite(value):
                    raise ValueError("Nonfinite training loss")
                value.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), CONFIG["gradient_clip"]
                )
                optimizer.step()
                if step % CONFIG["validate_every"] == 0:
                    score, detail = validation(model, feature, meta, val_ids, device)
                    row = dict(
                        step=step,
                        loss=float(value.detach().cpu()),
                        selection_nje=score,
                        source_validation=detail,
                    )
                    history.append(row)
                    if score < best:
                        best = score
                        torch.save(model.state_dict(), out / (arm + "-best.pt"))
                    save(out / (arm + "-history.json"), history)
                    print(
                        json.dumps(
                            dict(
                                arm=arm,
                                step=step,
                                loss=row["loss"],
                                validation_nje=score,
                            )
                        ),
                        flush=True,
                    )
            torch.save(model.state_dict(), out / (arm + "-last.pt"))
        # Bothfits and source-only checkpoint choices FINISH before any heldscore.
        held_ids = np.flatnonzero(held)
        permutation = shuffle_indices(meta["record"][held_ids])
        np.save(out / "held-shuffle.npy", permutation, allow_pickle=False)
        results = {}
        parity = {}
        for arm in CONFIG["arms"]:
            model = copy.deepcopy(initial).to(device)
            model.load_state_dict(
                torch.load(
                    out / (arm + "-best.pt"), map_location=device, weights_only=True
                )
            )
            results[arm] = {}
            for name, ids in [("source_validation", val_ids), ("held_room", held_ids)]:
                p, q = predict(model, feature, ids, device)
                delta = p - q
                np.savez_compressed(
                    out / (arm + "-" + name + ".npz"),
                    indices=ids,
                    current=p,
                    past=q,
                    delta=delta,
                )
                result = summary(p, delta, meta, ids)
                if name == "held_room":
                    result["within_record_shuffle"] = summary(
                        p[permutation], delta[permutation], meta, ids
                    )
                results[arm][name] = result
                reloaded = copy.deepcopy(initial).to(device)
                reloaded.load_state_dict(
                    torch.load(
                        out / (arm + "-best.pt"), map_location=device, weights_only=True
                    )
                )
                rp, rq = predict(reloaded, feature, ids, device)
                difference = max(float(abs(p - rp).max()), float(abs(q - rq).max()))
                if difference > 1e-6:
                    raise ValueError("Savedcheckpoint prediction replay differs")
                parity[arm + "/" + name] = difference
            history = json.loads((out / (arm + "-history.json")).read_text())
            results[arm]["selected_step"] = min(
                history, key=lambda h: h["selection_nje"]
            )["step"]
        controls = {}
        prior = np.broadcast_to(
            meta["pose"][train].mean(0), meta["pose"][held].shape
        ).copy()
        controls["source_static_prior"] = summary(
            prior, np.zeros_like(prior), meta, held_ids
        )
        controls["source_static_prior"]["within_record_shuffle"] = controls[
            "source_static_prior"
        ].copy()
        np.savez_compressed(
            out / "static-predictions.npz",
            indices=held_ids,
            pose=prior,
            delta=np.zeros_like(prior),
        )
        for arm in ("untreated_covariance", "log_shape_tangent", "congruence_dynamics"):
            with np.load(
                baseline / ("E02-" + arm + "-model-prediction.npz"), allow_pickle=False
            ) as z:
                np.testing.assert_array_equal(z["held_indices"], held_ids)
                p, d = z["pose"], z["delta"]
            controls[arm] = summary(p, d, meta, held_ids)
            controls[arm]["within_record_shuffle"] = summary(
                p[permutation], d[permutation], meta, held_ids
            )
        for name, expected in hashes.items():
            if digest(ROOT / name) != expected:
                raise ValueError("Sourcechanged")
        for name, expected in bindings.items():
            if digest(cache / name) != expected:
                raise ValueError("Cachechanged")
        for name, expected in bbindings.items():
            if digest(baseline / name) != expected:
                raise ValueError("Baselinechanged")
        save(
            out / "verification.json",
            dict(
                savedcheckpoint_prediction_max_abs=parity,
                input_and_source_rehash_passed=True,
            ),
        )
        save(
            out / "results.json",
            dict(
                registration=registration,
                results=results,
                controls=controls,
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
        "--cache", type=Path, default=ROOT / "work/domain-covariance-full-cache-v1"
    )
    parser.add_argument(
        "--baseline", type=Path, default=ROOT / "work/domain-covariance-full27-v1"
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if args.benchmark:
        benchmark(args.device)
    elif args.out:
        run(args.cache, args.baseline, args.out, args.device)
    else:
        parser.error("--out required for fitting")
