"""Independent source audit for delayed cross-person pose-prototype learning.

Shares the actual model class and source metadata loader. RF moment equations,
prototype reductions, reference/raw kernel equations, metrics, checkpoint
selection and both efficacy gates are independently checked. No room is scored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import solve
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import domain_pose_alignment_probe as producer  # noqa:E402
from scripts.verify_amplitude_source import (  # noqa:E402
    compare_report,
    evaluate,
    gate,
    product,
    reference_solution,
)
from scripts.verify_geometric_distillation import arrays, bind, digest, read, require  # noqa:E402


def independent_moments(raw):
    x = np.asarray(raw, dtype=float)
    require(
        x.ndim == 5 and x.shape[1:] == (32, 6, 114, 10) and len(x) > 0,
        "Delayed typed RF histories required",
    )
    a, ph = x[:, :, :3], x[:, :, 3:]
    finite = np.isfinite(a) & np.isfinite(ph)
    good = finite[:, :, 1:] & finite[:, :, :1]
    require((good.mean((1, 2, 3, 4)) >= 0.95).all(), "Independent RFquality failed")
    a = np.where(finite, a, 0)
    ph = np.where(finite, ph, 0)
    circular = np.exp(1j * (ph[:, :, 1:] - ph[:, :, :1]))
    q = np.concatenate(
        (
            np.where(good, np.clip((a[:, :, 1:] - a[:, :, :1]) / 20, -2, 2), 0),
            np.where(good, circular.real, 0),
            np.where(good, circular.imag, 0),
        ),
        axis=2,
    )
    return np.concatenate((q.mean(-1), q.std(-1)), axis=2).astype(np.float32)


def raw_kernel(x, q, kind):
    d = x.shape[1]
    dot = product(q, x.T) / d
    if kind == "linear":
        return dot, None
    own = product(x, x.T) / d
    square = np.square(x).mean(1)
    distance = np.maximum(square[:, None] + square[None] - 2 * own, 0)
    width = float(np.median(distance[np.triu_indices(len(x), 1)]))
    require(width > 1e-12, "Raw reference bandwidth degenerate")
    cross = np.maximum(np.square(q).mean(1)[:, None] + square[None] - 2 * dot, 0)
    return np.exp(-cross / (2 * width)), width


def raw_fit(x, y, q, kind, alpha):
    k, width = raw_kernel(x, x, kind)
    cross, _ = raw_kernel(x, q, kind)
    center = y.mean(0)
    weights = solve(
        k + alpha * np.eye(len(x)),
        y.reshape(len(y), 34) - center.reshape(1, 34),
        assume_a="gen",
        check_finite=True,
    )
    return (product(cross, weights) + center.reshape(1, 34)).reshape(-1, 17, 2), dict(
        weights=weights, center=center, width=width
    )


def verify(study, cache, raw, out):
    require(
        not out.exists()
        and not any(
            out.resolve().is_relative_to(v.resolve()) for v in [study, cache, raw]
        ),
        "Existing or nested audit preserved",
    )
    started = time.monotonic()
    audit_dependencies = {
        str(path.relative_to(ROOT)): digest(path)
        for path in (
            Path(__file__),
            ROOT / "scripts/verify_amplitude_source.py",
            ROOT / "scripts/verify_geometric_distillation.py",
        )
    }
    torch.set_num_threads(1)
    ready = read(study / "source-ready.json")
    require(ready["held_metrics_computed"] is False, "Held scores present")
    bind(study, ready["artifacts_sha256"])
    registration = read(study / "registration.json")
    bind(ROOT, registration["source_sha256"])
    cfg = producer.validate_config(registration["config"])
    raw_receipt = read(raw / "receipt.json")
    require(
        digest(raw / "rf.npy") == raw_receipt["arrays"]["rf"],
        "Original RF bytes differ",
    )
    require(
        registration["held_annotation_values_decoded"] is False
        and registration["output_delay_frames"] == 4,
        "Source/timing boundary differs",
    )
    require(
        registration["cache_receipt_sha256"] == digest(cache / "receipt.json")
        and registration["raw_receipt_sha256"] == digest(raw / "receipt.json"),
        "Source cache/raw identities differ",
    )
    fresh, feature, cache_receipt = producer.load_source(cache, raw)
    data = arrays(study / "source-data.npz")
    stats = arrays(study / "TRAIN-statistics.npz")
    require(
        set(data) == set(fresh)
        and all(np.array_equal(v, fresh[k]) for k, v in data.items()),
        "Saved source data differs",
    )
    train, val = data["train"], data["validation"]
    ids = np.flatnonzero(val)
    tr = np.flatnonzero(train)
    require((len(tr), len(ids)) == (6830, 1719), "Source row counts differ")
    absolute = np.asarray(
        np.load(raw / "pose.npy", mmap_mode="r", allow_pickle=False)[data["raw_index"]],
        dtype=float,
    )
    body = absolute - absolute[:, [11, 12]].mean(1, keepdims=True)
    target_error = float(np.max(np.abs(body - data["pose"])))
    require(target_error < 3e-8, "Raw target/body joins differ")
    # Independent reductions verify actual TRAIN-only prototype means and prior.
    assignment = stats["TRAIN_prototype_assignment"]
    require(
        assignment.shape == (len(tr),)
        and assignment.dtype.kind in "iu"
        and assignment.min() == 0
        and assignment.max() == 127,
        "TRAIN prototype membership differs",
    )
    counts = np.bincount(assignment, minlength=128)
    require((counts > 0).all(), "Empty TRAIN prototype")
    prototypes = np.stack(
        [data["pose"][tr[assignment == k]].mean(0, dtype=float) for k in range(128)]
    ).astype(np.float32)
    np.testing.assert_array_equal(prototypes, stats["prototypes"])
    np.testing.assert_array_equal(
        (counts / counts.sum()).astype(np.float32), stats["prior"]
    )
    target = data["pose"][tr].astype(float)
    pairs = producer.p.pair_rows(data["record"][tr], data["end"][tr])
    delta = target[pairs[:, 0]] - target[pairs[:, 1]]
    for key, value in [
        ("pose_mean", target.mean(0)),
        ("pose_scale", np.maximum(target.std(0), cfg["pose_scale_floor"])),
        (
            "pose_delta_scale",
            np.maximum(np.sqrt(np.square(delta).mean(0)), cfg["delta_scale_floor"]),
        ),
    ]:
        np.testing.assert_array_equal(value.astype(np.float32), stats[key])
    # Sum/square reduction with different chunking checks producer Welford stats.
    total = 0
    ss = np.zeros((12, 114))
    squares = ss.copy()
    for start in range(0, len(tr), 53):
        x = np.asarray(feature[tr[start : start + 53]], dtype=float).reshape(
            -1, 12, 114
        )
        ss += x.sum(0)
        squares += np.square(x).sum(0)
        total += len(x)
    mean = ss / total
    scale = np.maximum(np.sqrt(np.maximum(squares / total - mean**2, 0)), 0.05)
    np.testing.assert_allclose(mean, stats["input_mean"], atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(scale, stats["input_scale"], atol=1e-7, rtol=1e-6)
    schedule = np.load(study / "schedule.npy", allow_pickle=False)
    require(
        schedule.shape == (cfg["steps"], 24) and train[schedule].all(),
        "Optimizer role/budget differs",
    )
    groups = schedule.reshape(cfg["steps"], 8, 3)
    require(
        np.array_equal(data["record"][groups[:, :, 0]], data["record"][groups[:, :, 1]])
        and (data["end"][groups[:, :, 0]] - data["end"][groups[:, :, 1]] == 8).all(),
        "Pair clock differs",
    )
    for rows in groups:
        require(
            len(set(data["person"][rows[:, 0]])) == 8
            and np.array_equal(
                data["person"][rows],
                np.repeat(data["person"][rows[:, 0, None]], 3, axis=1),
            ),
            "Training teacher-person grouping differs",
        )
    report = read(study / "source-results.json")
    evaluation = study / report["evaluation_directory"]
    prediction_ready = read(evaluation / "predictions-ready.json")
    require(
        prediction_ready["source_validation_scored"] is False
        and prediction_ready["held_metrics_computed"] is False
        and report["predictions_ready_sha256"]
        == digest(evaluation / "predictions-ready.json"),
        "Prediction-before-score seal differs",
    )
    bind(evaluation, prediction_ready["artifacts_sha256"])
    refs = np.load(evaluation / "reference-rows.npy", allow_pickle=False)
    require(
        np.array_equal(refs, producer.p.references(data, cfg))
        and len(set(refs)) == 120
        and train[refs].all(),
        "Shared reference rows differ",
    )
    # Decode only E01 raw histories. Keep independent VAL moments for CPU readback.
    rf = np.load(raw / "rf.npy", mmap_mode="r", allow_pickle=False)
    checked_rows = np.concatenate((refs, ids))
    independent = np.empty((len(checked_rows), 32, 12, 114), np.float32)
    RFerror = 0.0
    for first in range(0, len(checked_rows), 8):
        rows = checked_rows[first : first + 8]
        frames = data["raw_index"][rows, None] + 4 - np.arange(31, -1, -1)[None]
        value = independent_moments(np.asarray(rf[frames]))
        RFerror = max(RFerror, float(np.max(np.abs(value - feature[rows]))))
        independent[first : first + len(rows)] = value
    require(RFerror < 2e-6, "Independent delayed RF moments differ")
    validation_features = independent[len(refs) :]
    initial = torch.load(
        study / "initial-model.pt", map_location="cpu", weights_only=True
    )
    measured = {}
    selected = {}
    zero_reports = {}
    cal_reports = {}
    cpu = {}
    max_metric = 0.0
    max_coefficient = 0.0
    max_cal = 0.0
    seals = 0
    for arm in producer.ARMS:
        complete = read(study / arm / "complete.json")
        require(complete["step"] == cfg["steps"], "Incomplete training budget")
        measured[arm] = []
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
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                require(
                    set(payload["model"]) == set(initial)
                    and all(
                        torch.equal(v, payload["model"][k]) for k, v in initial.items()
                    ),
                    "Different arm initialization",
                )
            stem = f"{arm}-{step:04}"
            encoded = arrays(evaluation / (stem + "-encoded.npz"))
            state = arrays(evaluation / (stem + "-reference-state.npz"))
            mu, bias, weight = reference_solution(
                encoded["reference_code"],
                data["pose"][refs].reshape(120, 34),
                encoded["reference_pose"].reshape(120, 34),
                cfg["ridge"],
            )
            error = float(
                np.linalg.norm(weight - state["weights"])
                / max(np.linalg.norm(weight), 1e-12)
            )
            require(error < 1e-9, "Independent reference equation differs")
            max_coefficient = max(max_coefficient, error)
            np.testing.assert_allclose(mu, state["code_mean"], atol=2e-12, rtol=2e-12)
            np.testing.assert_allclose(
                bias, state["correction_mean"], atol=2e-12, rtol=2e-12
            )
            q = encoded["query_code"].astype(float)
            q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-6)
            adapted = encoded["query_pose"] + (product(q - mu, weight) + bias).astype(
                np.float32
            ).reshape(-1, 17, 2)
            adapted -= adapted[:, [11, 12]].mean(1, keepdims=True)
            a = np.load(evaluation / (stem + "-VALzero.npy"), allow_pickle=False)
            b = np.load(evaluation / (stem + "-VALcal.npy"), allow_pickle=False)
            require(
                np.array_equal(a, encoded["query_pose"]), "Saved zero output differs"
            )
            error = float(np.max(np.abs(adapted - b)))
            require(error < 2e-7, "Calibrated output differs")
            max_cal = max(max_cal, error)
            row = dict(
                step=step,
                zero=evaluate(a, data, ids),
                calibrated=evaluate(b, data, ids),
            )
            measured[arm].append(row)
            original = next(v for v in report["candidates"][arm] if v["step"] == step)
            for mode in ["zero", "calibrated"]:
                max_metric = max(max_metric, compare_report(row[mode], original[mode]))
        selected[arm] = {}
        for mode, key in [("zero", "zero_step"), ("calibrated", "calibrated_step")]:
            valid = [
                v
                for v in report["candidates"][arm]
                if v[mode]["pooled"]["motion_r2"] is not None
            ]
            choice = min(
                valid,
                key=lambda v: (
                    -v[mode]["pooled"]["motion_r2"],
                    v[mode]["pooled"]["nje"],
                    v["step"],
                ),
            )
            selected[arm][key] = choice["step"]
            require(
                report["selected"][arm][key] == choice["step"],
                "Source checkpoint selector differs",
            )
        zero_reports[arm] = next(
            v["zero"] for v in measured[arm] if v["step"] == selected[arm]["zero_step"]
        )
        cal_reports[arm] = next(
            v["calibrated"]
            for v in measured[arm]
            if v["step"] == selected[arm]["calibrated_step"]
        )
        cpu[arm] = {}
        for step in sorted(set(selected[arm].values())):
            payload = torch.load(
                study / arm / f"step-{step:04}.pt",
                map_location="cpu",
                weights_only=True,
            )
            net = producer.make_model(stats, "cpu")
            net.load_state_dict(payload["model"], strict=True)
            net.eval()
            poses = []
            codes = []
            with torch.inference_mode():
                for first in range(0, len(ids), 16):
                    result = net(
                        torch.from_numpy(validation_features[first : first + 16])
                    )
                    poses.append(result["pose"].numpy())
                    codes.append(result["code"].numpy())
            encoded = arrays(evaluation / f"{arm}-{step:04}-encoded.npz")
            pe = float(np.max(np.abs(np.concatenate(poses) - encoded["query_pose"])))
            ce = float(np.max(np.abs(np.concatenate(codes) - encoded["query_code"])))
            require(pe < 4e-6 and ce < 4e-5, "CPU/independent RF readback differs")
            cpu[arm][str(step)] = dict(pose_max_abs=pe, code_max_abs=ce)
    zero_reports["source_mean"] = evaluate(
        np.broadcast_to(data["pose"][tr].mean(0), data["pose"][ids].shape), data, ids
    )
    clock = {
        int(e): data["pose"][train & (data["end"] == e)].mean(0)
        for e in set(data["end"][tr])
    }
    zero_reports["source_clock"] = evaluate(
        np.stack([clock[int(e)] for e in data["end"][ids]]), data, ids
    )
    cal_reports["reference_mean"] = evaluate(
        np.broadcast_to(data["pose"][refs].mean(0), data["pose"][ids].shape), data, ids
    )
    # Independently refit both full raw kernels, every reference-only CV choice.
    rx = (
        (np.asarray(feature[refs], dtype=float) - stats["input_mean"])
        / stats["input_scale"]
    ).reshape(120, -1)
    qx = (
        (np.asarray(feature[ids], dtype=float) - stats["input_mean"])
        / stats["input_scale"]
    ).reshape(len(ids), -1)
    target = data["pose"][refs].astype(float)
    people = data["person"][refs]
    raw_proof = {}
    CV = read(evaluation / "raw-reference-cv.json")
    for kind in ["linear", "rbf"]:
        grid = []
        for alpha in [1e-5, 0.001, 0.1, 10.0]:
            errors = []
            for person in ["S01", "S02"]:
                fit = people != person
                v = ~fit
                pred, _ = raw_fit(rx[fit], target[fit], rx[v], kind, alpha)
                diagonal = np.linalg.norm(target[v].max(1) - target[v].min(1), axis=1)
                errors.append(
                    float(
                        (
                            np.linalg.norm(pred - target[v], axis=2) / diagonal[:, None]
                        ).mean()
                    )
                )
            grid.append(
                dict(alpha=alpha, reference_person_cv_nje=float(np.mean(errors)))
            )
        for actual, expected in zip(grid, CV[kind]):
            require(
                actual["alpha"] == expected["alpha"]
                and abs(
                    actual["reference_person_cv_nje"]
                    - expected["reference_person_cv_nje"]
                )
                < 2e-12,
                "Independent raw CV differs",
            )
        choice = min(
            grid, key=lambda row: (row["reference_person_cv_nje"], -row["alpha"])
        )
        pred, state = raw_fit(rx, target, qx, kind, choice["alpha"])
        stored = arrays(evaluation / f"raw-{kind}-state.npz")
        require(float(stored["alpha"]) == choice["alpha"], "Raw selector differs")
        coefficient = float(
            np.linalg.norm(state["weights"] - stored["weights"])
            / max(np.linalg.norm(state["weights"]), 1e-12)
        )
        error = float(
            np.max(
                np.abs(
                    pred
                    - np.load(evaluation / f"raw-{kind}-VAL.npy", allow_pickle=False)
                )
            )
        )
        require(
            coefficient < 1e-7 and error < 1e-8,
            "Independent raw equation/prediction differs",
        )
        cal_reports["raw_" + kind] = evaluate(pred, data, ids)
        raw_proof[kind] = dict(
            weight_relative=coefficient,
            prediction_max_abs=error,
            full_reference_cv_reproduced=True,
        )
    for mode, reports in [("zero", zero_reports), ("calibrated", cal_reports)]:
        for arm, value in reports.items():
            max_metric = max(max_metric, compare_report(value, report[mode][arm]))
    same = next(
        v["zero"]
        for v in measured["cross_person"]
        if v["step"] == selected["cross_person"]["calibrated_step"]
    )
    max_metric = max(
        max_metric, compare_report(same, report["calibrated_checkpoint_zero"])
    )
    gates = dict(
        zero=gate(
            zero_reports["cross_person"],
            [v for k, v in zero_reports.items() if k != "cross_person"],
        ),
        calibrated=gate(
            cal_reports["cross_person"],
            [v for k, v in cal_reports.items() if k != "cross_person"]
            + [zero_reports["cross_person"], same],
        ),
    )
    gates["calibrated"]["beats_matched_own_zero"] = (
        cal_reports["cross_person"]["pooled"]["nje"]
        < zero_reports["cross_person"]["pooled"]["nje"]
    )
    gates["calibrated"]["beats_same_checkpoint_zero"] = (
        cal_reports["cross_person"]["pooled"]["nje"] < same["pooled"]["nje"]
    )
    require(
        gates == report["gates"]
        and {k: all(v.values()) for k, v in gates.items()} == report["passes"],
        "Independent gates differ",
    )
    bind(study, ready["artifacts_sha256"])
    bind(ROOT, registration["source_sha256"])
    bind(cache, cache_receipt["arrays_sha256"])
    bind(ROOT, audit_dependencies)
    proof = dict(
        status="qualified",
        source_ready_sha256=digest(study / "source-ready.json"),
        auditor_sha256=digest(Path(__file__)),
        auditor_dependencies_sha256=audit_dependencies,
        source_target_frame_joins=len(train),
        source_VAL_and_reference_RFhistories_independently_recomputed=len(checked_rows),
        max_raw_body_target_error=target_error,
        max_independent_RFhistory_error=RFerror,
        TRAIN_prototype_reductions_verified=True,
        TRAIN_input_statistics_independently_reduced=True,
        candidate_checkpoint_seals=seals,
        independent_reference_equations=seals,
        selected=selected,
        cpu_selected_readback=cpu,
        raw_reference_equations=raw_proof,
        max_metric_disagreement=max_metric,
        max_reference_weight_relative=max_coefficient,
        max_calibration_prediction_error=max_cal,
        independent_gates=gates,
        seconds=time.monotonic() - started,
        limits="Shared actual model class and guarded source loader; independently recomputed RF moments, prototype/normalization reductions, reference/rawkernel equations/allrawCV, metrics/selections/gates. Savednativecodesbindoriginalfit. ReusedE01developmentonly,400msnominalbuffer; noE03/E04 queryvalues, physicalclock calibration,3D or reliable-sensingclaim.",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump(proof, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["study", "cache", "raw", "out"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.study, args.cache, args.raw, args.out)
    print(
        json.dumps(
            dict(
                status=result["status"],
                seconds=result["seconds"],
                max_metric_disagreement=result["max_metric_disagreement"],
            )
        ),
        flush=True,
    )
