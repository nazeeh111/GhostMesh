"""Independently check source selection, reference equations and CPU inference.

This does not train, score a new room or accept query annotations as inputs to a
predictor. Independent equations below check the producer's saved source scores.
CPU inference shares the actual model class; normalization is reimplemented.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import solve
from scipy.linalg.blas import dgemm
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import domain_amplitude_probe as producer  # noqa:E402
from scripts.verify_geometric_distillation import arrays, bind, digest, read, require  # noqa:E402


def metric(prediction, target, records, ends):
    raw = np.asarray(prediction)
    truth = np.asarray(target)
    require(
        raw.dtype.kind in "iuf" and truth.dtype.kind in "iuf",
        "Real pose arrays required",
    )
    x, y = raw.astype(float), truth.astype(float)
    require(
        x.shape == y.shape
        and x.shape[1:] == (17, 2)
        and len(x) > 0
        and np.isfinite(x).all()
        and np.isfinite(y).all(),
        "Finite matching pose arrays required",
    )
    require(len(records) == len(ends) == len(x), "Metric metadata rows differ")
    centered_x = np.empty_like(x)
    centered_y = np.empty_like(y)
    current = []
    previous = []
    for name in sorted(set(records)):
        rows = np.flatnonzero(records == name)
        a, b = x[rows] - x[rows][0], y[rows] - y[rows][0]
        centered_x[rows] = a - a.mean(0)
        centered_y[rows] = b - b.mean(0)
        lookup = {int(ends[row]): int(row) for row in rows}
        require(len(lookup) == len(rows), "Duplicate endpoint")
        for row in rows:
            if int(ends[row]) - 8 in lookup:
                current.append(row)
                previous.append(lookup[int(ends[row]) - 8])
    energy = np.square(centered_y).sum()
    dy = y[current] - y[previous]
    dp = x[current] - x[previous]
    de = np.square(dy).sum()
    diagonal = np.linalg.norm(y.max(1) - y.min(1), axis=1)
    require((diagonal > 1e-6).all(), "Degenerate target body")
    error = np.linalg.norm(x - y, axis=2) / diagonal[:, None]
    return dict(
        motion_r2=float(1 - np.square(centered_x - centered_y).sum() / energy)
        if energy > 0
        else None,
        delta_r2=float(1 - np.square(dp - dy).sum() / de) if de > 0 else None,
        nje=float(error.mean()),
        pck05=float((error <= 0.05).mean()),
        pck20=float((error <= 0.2).mean()),
    )


def compare(actual, expected):
    require(set(actual) == set(expected), "Metric fields differ")
    maximum = 0.0
    for key, value in actual.items():
        if value is None or expected[key] is None:
            require(value is None and expected[key] is None, "Undefined metric differs")
        else:
            difference = abs(value - expected[key])
            require(
                np.isfinite(difference) and difference < 2e-12,
                "Metric disagreement: " + key,
            )
            maximum = max(maximum, difference)
    return maximum


def product(a, b):
    require(np.ndim(a) == np.ndim(b) == 2, "Explicit2Dmatrix product required")
    return dgemm(
        1.0,
        np.asarray(a, dtype=float, order="F"),
        np.asarray(b, dtype=float, order="F"),
    )


def reference_solution(code, target, base, ridge):
    code = np.asarray(code, dtype=float)
    target = np.asarray(target, dtype=float)
    base = np.asarray(base, dtype=float)
    require(
        code.ndim == 2
        and target.ndim == base.ndim == 2
        and target.shape == base.shape
        and len(code) == len(target),
        "Reference matrices must be explicitly flattened",
    )
    z = code / np.maximum(np.linalg.norm(code, axis=1, keepdims=True), 1e-6)
    mean = z.mean(0)
    residual = target - base
    bias = residual.mean(0)
    x = z - mean
    system = product(x.T, x) + (ridge * len(x) / 24) * np.eye(x.shape[1])
    right = product(x.T, residual - bias)
    return mean, bias, solve(system, right, assume_a="gen", check_finite=True)


def independent_normalize(raw):
    x = np.asarray(raw, dtype=float).copy()
    require(
        x.ndim == 5 and x.shape[1:] == (8, 3, 114, 10),
        "Amplitude history schema required",
    )
    frames = x.reshape(-1, 3, 114, 10)
    for frame in frames:
        require(np.isfinite(frame).mean() >= 0.95, "Invalid RF frame")
        for column in range(10):
            values = frame[:, :, column]
            observed = np.isfinite(values)
            require(observed.any(), "Missing packet column")
            values[~observed] = values[observed].mean()
        low, high = float(frame.min()), float(frame.max())
        require(high - low >= 1e-6, "Constant amplitude frame")
        frame -= low
        frame /= high - low
    return x.astype(np.float32)


def evaluate(pred, data, ids):
    def group(rows):
        return metric(
            pred[rows],
            data["pose"][ids[rows]],
            data["record"][ids[rows]],
            data["end"][ids[rows]],
        )

    result = dict(pooled=group(np.arange(len(ids))))
    for field in ("person", "action"):
        result["per_" + field] = {
            str(name): group(np.flatnonzero(data[field][ids] == name))
            for name in sorted(set(data[field][ids]))
        }
    shuffled = np.arange(len(ids))
    rng = np.random.default_rng(90210)
    for name in sorted(set(data["record"][ids])):
        rows = np.flatnonzero(data["record"][ids] == name)
        shuffled[rows] = rng.permutation(rows)
    result["shuffled"] = metric(
        pred[shuffled], data["pose"][ids], data["record"][ids], data["end"][ids]
    )
    return result


def compare_report(actual, expected):
    maximum = compare(actual["pooled"], expected["pooled"])
    for field in ("person", "action"):
        require(
            set(actual["per_" + field]) == set(expected["per_" + field]),
            "Subgroup membership differs",
        )
        for name, row in actual["per_" + field].items():
            maximum = max(maximum, compare(row, expected["per_" + field][name]))
    return max(maximum, compare(actual["shuffled"], expected["shuffled"]))


def gate(primary, controls):
    po = primary["pooled"]
    return dict(
        body_motion_useful=po["motion_r2"] is not None and po["motion_r2"] > 0.05,
        delta_useful=po["delta_r2"] is not None and po["delta_r2"] > 0.05,
        both_people_positive=set(primary["per_person"]) == {"S04", "S09"}
        and all(
            row["motion_r2"] is not None and row["motion_r2"] > 0
            for row in primary["per_person"].values()
        ),
        nje_beats_controls=all(po["nje"] < row["pooled"]["nje"] for row in controls),
        pck_not_worse=all(
            po[key] >= row["pooled"][key]
            for row in controls
            for key in ("pck05", "pck20")
        ),
        shuffle_worse=po["nje"] < primary["shuffled"]["nje"],
    )


def verify(study, cache, raw, out):
    require(not out.exists(), "Existing audit preserved")
    require(
        not any(
            out.resolve().is_relative_to(path.resolve()) for path in (study, cache, raw)
        ),
        "Audit output inside immutable inputs",
    )
    ready = read(study / "source-ready.json")
    require(ready["held_metrics_computed"] is False, "Held metrics already computed")
    bind(study, ready["artifacts_sha256"])
    registration = read(study / "registration.json")
    bind(ROOT, registration["source_sha256"])
    cfg = registration["config"]
    data = arrays(study / "source-data.npz")
    stats = arrays(study / "TRAIN-statistics.npz")
    train, val = data["train"], data["validation"]
    require(
        train.dtype == val.dtype == bool
        and (train.sum(), val.sum()) == (7050, 1773)
        and set(data["room"]) == {"E01"}
        and not set(data["person"][train]) & set(data["person"][val]),
        "Source role leak",
    )
    require(data["raw_index"].max() < 80190, "Held annotation index entered source")
    absolute = np.asarray(
        np.load(raw / "pose.npy", mmap_mode="r", allow_pickle=False)[data["raw_index"]],
        dtype=float,
    )
    body = absolute - absolute[:, [11, 12]].mean(1, keepdims=True)
    target_error = float(np.max(np.abs(body - data["pose"])))
    require(target_error < 3e-8, "Raw annotation/body join differs")
    rf = np.load(raw / "rf.npy", mmap_mode="r", allow_pickle=False)
    report = read(study / "source-results.json")
    evaluation = study / report["evaluation_directory"]
    refs = np.load(evaluation / "reference-rows.npy", allow_pickle=False)
    ids = np.flatnonzero(val)
    require(
        len(refs) == len(set(refs)) == 120 and train[refs].all(),
        "Reference role/repetition differs",
    )
    initial = torch.load(
        study / "initial-model.pt", map_location="cpu", weights_only=True
    )
    measured = {}
    maximum_metric = 0.0
    maximum_coefficient = 0.0
    maximum_calibration = 0.0
    seals = 0
    for arm in producer.ARMS:
        measured[arm] = []
        complete = read(study / arm / "complete.json")
        for step in range(0, cfg["steps"] + 1, cfg["checkpoint_every"]):
            checkpoint = study / arm / f"step-{step:04}.pt"
            seal = read(checkpoint.with_name(checkpoint.name + ".json"))
            require(
                seal
                == dict(
                    kind="source-geometric-recovery",
                    step=step,
                    identity=complete["identity"],
                    arm=arm,
                    device=registration["device"],
                    checkpoint_sha256=digest(checkpoint),
                ),
                "Candidate seal differs",
            )
            seals += 1
            if step == 0:
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                require(
                    set(saved["model"]) == set(initial)
                    and all(
                        torch.equal(saved["model"][key], value)
                        for key, value in initial.items()
                    ),
                    "Arms have different initial models",
                )
            stem = f"{arm}-{step:04}"
            encoded = arrays(evaluation / (stem + "-encoded.npz"))
            state = arrays(evaluation / (stem + "-reference-state.npz"))
            mean, bias, w = reference_solution(
                encoded["reference_code"],
                data["pose"][refs].reshape(120, 34),
                encoded["reference_pose"].reshape(120, 34),
                cfg["ridge"],
            )
            weight_error = float(
                np.linalg.norm(w - state["weights"]) / max(np.linalg.norm(w), 1e-12)
            )
            require(weight_error < 1e-9, "Reference equation differs")
            maximum_coefficient = max(maximum_coefficient, weight_error)
            np.testing.assert_allclose(mean, state["code_mean"], atol=2e-12, rtol=2e-12)
            np.testing.assert_allclose(
                bias, state["correction_mean"], atol=2e-12, rtol=2e-12
            )
            q = encoded["query_code"].astype(float)
            q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-6)
            cal = encoded["query_pose"] + (product(q - mean, w) + bias).astype(
                np.float32
            ).reshape(-1, 17, 2)
            cal -= cal[:, [11, 12]].mean(1, keepdims=True)
            zero = np.load(evaluation / (stem + "-VALzero.npy"), allow_pickle=False)
            saved_cal = np.load(evaluation / (stem + "-VALcal.npy"), allow_pickle=False)
            require(
                np.array_equal(zero, encoded["query_pose"]),
                "Zero predictions differ from encoder receipt",
            )
            error = float(np.max(np.abs(cal - saved_cal)))
            require(error < 2e-7, "Calibrated predictions differ")
            maximum_calibration = max(maximum_calibration, error)
            row = dict(
                step=step,
                zero=evaluate(zero, data, ids),
                calibrated=evaluate(saved_cal, data, ids),
            )
            measured[arm].append(row)
            original = next(
                row for row in report["candidates"][arm] if row["step"] == step
            )
            for mode in ("zero", "calibrated"):
                maximum_metric = max(
                    maximum_metric, compare_report(row[mode], original[mode])
                )
    selected = {}
    zero_reports = {}
    cal_reports = {}
    cpu = {}
    torch.set_num_threads(1)
    for arm in producer.ARMS:
        selected[arm] = {}
        for mode, key in (("zero", "zero_step"), ("calibrated", "calibrated_step")):
            # Scores are independently checked above. Use the recorded values
            # for exact tie-breaking at machine precision, per the frozen rule.
            valid = [
                row
                for row in report["candidates"][arm]
                if row[mode]["pooled"]["motion_r2"] is not None
            ]
            chosen = min(
                valid,
                key=lambda row: (
                    -row[mode]["pooled"]["motion_r2"],
                    row[mode]["pooled"]["nje"],
                    row["step"],
                ),
            )
            selected[arm][key] = chosen["step"]
            require(
                report["selected"][arm][key] == chosen["step"],
                "Source checkpoint selector differs",
            )
        zero_reports[arm] = next(
            row["zero"]
            for row in measured[arm]
            if row["step"] == selected[arm]["zero_step"]
        )
        cal_reports[arm] = next(
            row["calibrated"]
            for row in measured[arm]
            if row["step"] == selected[arm]["calibrated_step"]
        )
        cpu[arm] = {}
        for step in sorted(set(selected[arm].values())):
            saved = torch.load(
                study / arm / f"step-{step:04}.pt",
                map_location="cpu",
                weights_only=True,
            )
            net = producer.make_model(stats, "cpu")
            net.load_state_dict(saved["model"], strict=True)
            net.eval()
            poses = []
            codes = []
            with torch.inference_mode():
                for start in range(0, len(ids), 16):
                    rows = ids[start : start + 16]
                    frames = data["raw_index"][rows, None] - np.arange(7, -1, -1)[None]
                    x = independent_normalize(np.asarray(rf[frames, :3]))
                    pred = net(torch.from_numpy(x))
                    poses.append(pred["pose"].numpy())
                    codes.append(pred["code"].numpy())
            encoded = arrays(evaluation / f"{arm}-{step:04}-encoded.npz")
            pe = float(np.max(np.abs(np.concatenate(poses) - encoded["query_pose"])))
            ce = float(np.max(np.abs(np.concatenate(codes) - encoded["query_code"])))
            require(
                pe < 4e-6 and ce < 3e-5, "CPU model/independent normalization differs"
            )
            cpu[arm][str(step)] = dict(pose_max_abs=pe, code_max_abs=ce)
    tr = data["pose"][train]
    zero_reports["source_mean"] = evaluate(
        np.broadcast_to(tr.mean(0), data["pose"][ids].shape), data, ids
    )
    clock = {
        int(end): data["pose"][train & (data["end"] == end)].mean(0)
        for end in set(data["end"][train])
    }
    zero_reports["source_clock"] = evaluate(
        np.stack([clock[int(end)] for end in data["end"][ids]]), data, ids
    )
    cal_reports["reference_mean"] = evaluate(
        np.broadcast_to(data["pose"][refs].mean(0), data["pose"][ids].shape), data, ids
    )
    for kind in ("linear", "rbf"):
        pred = np.load(evaluation / f"raw-{kind}-VAL.npy", allow_pickle=False)
        cal_reports["raw_" + kind] = evaluate(pred, data, ids)
    for mode, rows in (("zero", zero_reports), ("calibrated", cal_reports)):
        for name, row in rows.items():
            maximum_metric = max(
                maximum_metric, compare_report(row, report[mode][name])
            )
    gates = dict(
        zero=gate(
            zero_reports["gain_adversarial"],
            [row for name, row in zero_reports.items() if name != "gain_adversarial"],
        ),
        calibrated=gate(
            cal_reports["gain_adversarial"],
            [row for name, row in cal_reports.items() if name != "gain_adversarial"],
        ),
    )
    require(
        gates == report["gates"]
        and {name: all(value.values()) for name, value in gates.items()}
        == report["passes"],
        "Independent gate decisions differ",
    )
    bind(study, ready["artifacts_sha256"])
    bind(ROOT, registration["source_sha256"])
    proof = dict(
        status="qualified",
        source_ready_sha256=digest(study / "source-ready.json"),
        auditor_sha256=digest(Path(__file__)),
        source_target_frame_joins=len(train),
        max_raw_body_target_error=target_error,
        candidate_checkpoint_seals=seals,
        independent_reference_equations=seals,
        selected=selected,
        independent_gates=gates,
        cpu_selected_readback=cpu,
        max_metric_disagreement=maximum_metric,
        max_reference_weight_relative=maximum_coefficient,
        max_calibration_prediction_error=maximum_calibration,
        limits="Shared actual model class for CPU inference; independent normalization, reference equations, metric calculations and source selection/gates. Saved original encodings bind reference fits. Raw-baseline saved predictions are metric-checked; their CV/equations are not independently refitted here. SOURCE calibration is represented-room only, no new E03/E04 queries or reliable-sensing claim.",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump(proof, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "cache", "raw", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    result = verify(args.study, args.cache, args.raw, args.out)
    print(
        json.dumps(
            dict(
                status=result["status"],
                seconds=time.monotonic() - started,
                max_metric_disagreement=result["max_metric_disagreement"],
            )
        )
    )
