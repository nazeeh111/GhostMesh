"""Independent source metrics, CPU replay and NumPy reference-fit equations."""

import json
import argparse
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import solve
from scipy.linalg.blas import dgemm
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import domain_episodic_probe as p  # noqa: E402


def run(study, cache, raw, out):
    p.require(
        torch.backends.mps.is_available(),
        "Exact native reference-code audit requires the original MPS backend",
    )
    OUT = out
    OUT.mkdir(exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(1)
    ready = p.read(study / "source-ready.json")
    p.require(ready["held_metrics_computed"] is False, "Held data scored")
    p.bind(study, ready["artifacts_sha256"])
    registration = p.read(study / "registration.json")
    p.bind(ROOT, registration["source_sha256"])
    cfg = p.validate_config(p.read(study / "config.json"))
    data = p.arrays(study / "source-data.npz")
    stats = p.arrays(study / "TRAIN-statistics.npz")
    fresh, feature, cache_receipt, _ = p.load_source(cache, raw, cfg)
    p.require(
        p.digest(cache / "receipt.json") == registration["cache_receipt_sha256"],
        "Cache receipt differs",
    )
    p.require(
        p.digest(raw / "receipt.json") == registration["raw_receipt_sha256"],
        "Raw receipt differs",
    )
    p.require(set(data) == set(fresh), "Saved source schema differs")
    for key in data:
        p.require(
            np.array_equal(data[key], fresh[key]), "Saved source rows differ: " + key
        )
    train, val = data["train"], data["validation"]
    p.require(
        (train.sum(), val.sum()) == (7050, 1773)
        and set(data["room"]) == {"E01"}
        and not set(data["person"][train]) & set(data["person"][val]),
        "Source role leak",
    )
    p.require(data["raw_index"].max() < 80190, "Decoded held annotation index")
    source = np.asarray(
        np.load(raw / "pose.npy", mmap_mode="r", allow_pickle=False)[data["raw_index"]],
        dtype=float,
    )
    body = source - source[:, [11, 12]].mean(1, keepdims=True)
    target_error = float(np.max(np.abs(body - data["pose"])))
    p.require(target_error < 3e-8, "Raw/source body target join differs")
    episodes = np.load(study / "episodes.npy", allow_pickle=False)
    perms = np.load(study / "support-permutations.npy", allow_pickle=False)
    p.require(
        episodes.shape == (2000, 48) and train[episodes].all(), "Optimizer role leak"
    )
    for rows, perm in zip(episodes, perms):
        p.require(
            not set(data["person"][rows[:24]]) & set(data["person"][rows[24:]]),
            "Episode person overlap",
        )
        p.require(
            np.array_equal(data["record"][rows[perm]], data["record"][rows[:24]]),
            "Inner labels crossed records",
        )
    refs = np.load(
        study
        / p.read(study / "source-results.json")["evaluation_directory"]
        / "reference-rows.npy",
        allow_pickle=False,
    )
    p.require(
        len(refs) == len(set(refs)) == 120
        and train[refs].all()
        and set(data["person"][refs]) == {"S01", "S02"},
        "Source reference scope differs",
    )
    report = p.read(study / "source-results.json")
    evaluation = study / report["evaluation_directory"]
    ids = np.flatnonzero(val)
    reader = p.SourceReader(feature, data["cache_row"], train)
    vr = p.SourceReader(feature, data["cache_row"], val)

    def product(a, b):
        p.require(np.ndim(a) == np.ndim(b) == 2, "Explicit2Dmatrixproduct required")
        return dgemm(
            1.0,
            np.asarray(a, dtype=float, order="F"),
            np.asarray(b, dtype=float, order="F"),
        )

    def metric(pred, target, records, ends):
        pred, target = np.asarray(pred, dtype=float), np.asarray(target, dtype=float)
        pc = np.empty_like(pred)
        yc = np.empty_like(target)
        current = []
        past = []
        for name in sorted(set(records)):
            rows = np.flatnonzero(records == name)
            pc[rows] = pred[rows] - pred[rows].mean(0)
            yc[rows] = target[rows] - target[rows].mean(0)
            lookup = {int(ends[row]): int(row) for row in rows}
            for row in rows:
                if int(ends[row]) - 8 in lookup:
                    current.append(row)
                    past.append(lookup[int(ends[row]) - 8])
        energy = np.square(yc).sum()
        dy = target[current] - target[past]
        dp = pred[current] - pred[past]
        de = np.square(dy).sum()
        diagonal = np.linalg.norm(target.max(1) - target.min(1), axis=1)
        err = np.linalg.norm(pred - target, axis=2) / diagonal[:, None]
        return dict(
            motion_r2=float(1 - np.square(pc - yc).sum() / energy)
            if energy > 0
            else None,
            delta_r2=float(1 - np.square(dp - dy).sum() / de) if de > 0 else None,
            nje=float(err.mean()),
            pck05=float((err <= 0.05).mean()),
            pck20=float((err <= 0.2).mean()),
        )

    def compare(a, b):
        error = 0.0
        for key, value in a.items():
            if value is None or b[key] is None:
                p.require(value is None and b[key] is None, "Undefined metric differs")
            else:
                difference = abs(value - b[key])
                p.require(difference < 2e-12, "Independent metric differs: " + key)
                error = max(error, difference)
        return error

    metric_error = 0.0
    results = {}
    seals = 0
    for arm in p.ARMS:
        complete = p.read(study / arm / "complete.json")
        for step in (500, 1000, 1500, 2000):
            path = study / arm / f"step-{step:04}.pt"
            seal = p.read(path.with_name(path.name + ".json"))
            p.require(
                seal
                == dict(
                    kind="source-geometric-recovery",
                    step=step,
                    identity=complete["identity"],
                    arm=arm,
                    device="mps",
                    checkpoint_sha256=p.digest(path),
                ),
                "Recovery seal differs",
            )
            seals += 1
        saved = torch.load(
            study / arm / "step-2000.pt", map_location="cpu", weights_only=True
        )
        p.require(
            saved["step"] == 2000 and saved["identity"] == complete["identity"],
            "Final checkpoint identity differs",
        )
        net = p.make_model(stats, "cpu")
        net.load_state_dict(saved["model"], strict=True)
        cpu_pose, cpu_code = p.encode(net, vr, ids, "cpu")
        stored = np.load(evaluation / f"{arm}-VALzero.npy", allow_pickle=False)
        cpu_error = float(np.max(np.abs(cpu_pose - stored)))
        p.require(cpu_error < 4e-6, "CPU zero-shot readback differs")
        net.to("mps")
        rp, rc = p.encode(net, reader, refs, "mps")
        qp, qc = p.encode(net, vr, ids, "mps")
        p.require(np.max(np.abs(qp - stored)) < 1e-7, "Native zero-shot replay differs")
        s = rc.astype(float)
        s /= np.maximum(np.linalg.norm(s, axis=1, keepdims=True), 1e-6)
        q = qc.astype(float)
        q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-6)
        mean = s.mean(0)
        x = s - mean
        residual = data["pose"][refs].reshape(120, 34).astype(float) - rp.reshape(
            120, 34
        ).astype(float)
        bias = residual.mean(0)
        system = product(x.T, x) + (0.1 * len(refs) / 24) * np.eye(128)
        right = product(x.T, residual - bias)
        weights = solve(system, right, assume_a="gen", check_finite=True)
        state = p.arrays(evaluation / f"{arm}-reference-map.npz")
        coefficient = float(
            np.linalg.norm(weights - state["weights"])
            / max(np.linalg.norm(weights), 1e-12)
        )
        p.require(coefficient < 1e-9, "Independent reference solve differs")
        np.testing.assert_allclose(state["code_mean"], mean, atol=2e-12, rtol=2e-12)
        np.testing.assert_allclose(
            state["correction_mean"], bias, atol=2e-12, rtol=2e-12
        )
        correction = (product(q - mean, weights) + bias).astype(np.float32)
        adapted = qp + correction.reshape(-1, 17, 2)
        adapted -= adapted[:, [11, 12]].mean(1, keepdims=True)
        cal = np.load(evaluation / f"{arm}-VALcal.npy", allow_pickle=False)
        adaptation_error = float(np.max(np.abs(adapted - cal)))
        p.require(adaptation_error < 2e-7, "Calibrated prediction differs")
        for mode, pred in (("zero", stored), ("calibrated", cal)):
            groups = [("pooled", np.arange(len(ids)), report[mode][arm]["pooled"])]
            for field in ("person", "action"):
                for name in sorted(set(data[field][ids])):
                    groups.append(
                        (
                            field + "/" + str(name),
                            np.flatnonzero(data[field][ids] == name),
                            report[mode][arm]["per_" + field][str(name)],
                        )
                    )
            for name, rows, expected in groups:
                metric_error = max(
                    metric_error,
                    compare(
                        metric(
                            pred[rows],
                            data["pose"][ids[rows]],
                            data["record"][ids[rows]],
                            data["end"][ids[rows]],
                        ),
                        expected,
                    ),
                )
        results[arm] = dict(
            cpu_zero_max_abs=cpu_error,
            independent_reference_weight_relative=coefficient,
            independent_calibration_pose_max_abs=adaptation_error,
        )
        print(json.dumps(dict(arm=arm, **results[arm])), flush=True)
    # Independently reconstruct raw reference kernels and equations; validation
    # outcomes are never used to select their regularization.
    rx = ((reader(refs) - stats["input_mean"]) / stats["input_scale"]).reshape(120, -1)
    qx = ((vr(ids) - stats["input_mean"]) / stats["input_scale"]).reshape(len(ids), -1)
    target = data["pose"][refs].astype(float)
    center = target.mean(0)
    D = rx.shape[1]
    raw_proofs = {}
    for kind in ("linear", "rbf"):
        state = p.arrays(evaluation / f"raw-{kind}-state.npz")
        k = product(rx, rx.T) / D
        kq = product(qx, rx.T) / D
        if kind == "rbf":
            square = np.square(rx).mean(1)
            dist = np.maximum(square[:, None] + square[None, :] - 2 * k, 0)
            width = float(np.median(dist[np.triu_indices(120, 1)]))
            cross = np.maximum(
                np.square(qx).mean(1)[:, None] + square[None, :] - 2 * kq, 0
            )
            k, kq = np.exp(-dist / (2 * width)), np.exp(-cross / (2 * width))
            p.require(
                abs(width - float(state["width"])) < 1e-10, "Raw bandwidth differs"
            )
        # Producer stores a float32 reference mean before the double solve.
        center = data["pose"][refs].mean(0).astype(float)
        system = k + float(state["alpha"]) * np.eye(120)
        rhs = (
            data["pose"][refs].reshape(120, -1)
            - data["pose"][refs].mean(0).reshape(1, -1)
        ).astype(float)
        weights = solve(system, rhs, assume_a="gen", check_finite=True)
        error = float(
            np.linalg.norm(weights - state["weights"])
            / max(np.linalg.norm(weights), 1e-12)
        )
        p.require(error < 1e-7, "Raw reference equation differs")
        pred = (product(kq, weights) + center.reshape(1, -1)).reshape(-1, 17, 2)
        stored = np.load(evaluation / f"raw-{kind}-VAL.npy", allow_pickle=False)
        maximum = float(np.max(np.abs(pred - stored)))
        p.require(maximum < 1e-8, "Raw baseline prediction differs")
        metric_error = max(
            metric_error,
            compare(
                metric(
                    stored, data["pose"][ids], data["record"][ids], data["end"][ids]
                ),
                report["calibrated"]["raw_" + kind]["pooled"],
            ),
        )
        raw_proofs[kind] = dict(weight_relative=error, prediction_max_abs=maximum)
    # Rebuild every control, metric group, temporal-shuffle score and gate.
    # Source-only labels define diagnostic controls, never model inputs.
    clock = {
        int(e): data["pose"][train & (data["end"] == e)].mean(0)
        for e in set(data["end"][train])
    }
    predicted = {
        "zero": {
            arm: np.load(evaluation / f"{arm}-VALzero.npy", allow_pickle=False)
            for arm in p.ARMS
        },
        "calibrated": {
            arm: np.load(evaluation / f"{arm}-VALcal.npy", allow_pickle=False)
            for arm in p.ARMS
        },
    }
    predicted["zero"]["source_mean"] = np.broadcast_to(
        data["pose"][train].mean(0), data["pose"][ids].shape
    )
    predicted["zero"]["source_clock"] = np.stack(
        [clock[int(e)] for e in data["end"][ids]]
    )
    predicted["calibrated"]["reference_mean"] = np.broadcast_to(
        data["pose"][refs].mean(0), data["pose"][ids].shape
    )
    for kind in ("linear", "rbf"):
        predicted["calibrated"]["raw_" + kind] = np.load(
            evaluation / f"raw-{kind}-VAL.npy", allow_pickle=False
        )
    permutation = np.arange(len(ids))
    rng = np.random.default_rng(90210)
    for name in sorted(set(data["record"][ids])):
        rows = np.flatnonzero(data["record"][ids] == name)
        permutation[rows] = rng.permutation(rows)
    independent = {}
    metric_groups = 0
    for mode, arms in predicted.items():
        independent[mode] = {}
        for arm, pred in arms.items():
            measured = {
                "pooled": metric(
                    pred, data["pose"][ids], data["record"][ids], data["end"][ids]
                )
            }
            metric_error = max(
                metric_error, compare(measured["pooled"], report[mode][arm]["pooled"])
            )
            metric_groups += 1
            for field in ("person", "action"):
                measured["per_" + field] = {}
                for name in sorted(set(data[field][ids])):
                    rows = np.flatnonzero(data[field][ids] == name)
                    value = metric(
                        pred[rows],
                        data["pose"][ids[rows]],
                        data["record"][ids[rows]],
                        data["end"][ids[rows]],
                    )
                    metric_error = max(
                        metric_error,
                        compare(value, report[mode][arm]["per_" + field][str(name)]),
                    )
                    measured["per_" + field][str(name)] = value
                    metric_groups += 1
            measured["shuffled"] = metric(
                pred[permutation],
                data["pose"][ids],
                data["record"][ids],
                data["end"][ids],
            )
            metric_error = max(
                metric_error,
                compare(measured["shuffled"], report[mode][arm]["shuffled"]),
            )
            metric_groups += 1
            independent[mode][arm] = measured
    independent_gates = {}
    for mode, arms in independent.items():
        primary = arms["episodic"]
        po = primary["pooled"]
        controls = [value for name, value in arms.items() if name != "episodic"]
        gate = dict(
            body_motion_useful=po["motion_r2"] is not None and po["motion_r2"] > 0.05,
            delta_useful=po["delta_r2"] is not None and po["delta_r2"] > 0.05,
            both_people_positive=set(primary["per_person"]) == {"S04", "S09"}
            and all(
                value["motion_r2"] is not None and value["motion_r2"] > 0
                for value in primary["per_person"].values()
            ),
            nje_beats_controls=all(
                po["nje"] < value["pooled"]["nje"] for value in controls
            ),
            pck_not_worse=all(
                po[key] >= value["pooled"][key]
                for value in controls
                for key in ("pck05", "pck20")
            ),
            shuffle_worse=po["nje"] < primary["shuffled"]["nje"],
        )
        p.require(
            gate == report["gates"][mode]
            and all(gate.values()) == report["passes"][mode],
            "Independent source gate differs",
        )
        independent_gates[mode] = gate
    p.bind(study, ready["artifacts_sha256"])
    p.bind(ROOT, registration["source_sha256"])
    p.bind(cache, cache_receipt["arrays_sha256"])
    proof = dict(
        status="qualified",
        source_ready_sha256=p.digest(study / "source-ready.json"),
        auditor_sha256=p.digest(Path(__file__)),
        source_target_frame_joins=8823,
        max_raw_target_error=target_error,
        source_training_rows=7050,
        source_validation_rows=1773,
        checkpoint_seals=seals,
        independent_metric_max_error=metric_error,
        arms=results,
        raw_reference_equations=raw_proofs,
        independent_metric_groups=metric_groups,
        independent_gates=independent_gates,
        seconds=time.monotonic() - started,
        limits="Shared producer model/RF reader for replay; independent NumPy/SciPy solver and metric equations. Native encodings reproduce original reference-fit inputs; CPU separately checks all zero-shot poses. SOURCEcalibration is represented E01room, not unseenroom; no E03annotations orqueryscores. Gate decisions remainfailed and unmodified. Reference-CV selectors checked by existing unit tests; this audit reconstructs selected raw equations, not all CVfoldfits.",
    )
    (OUT / "proof.json").write_text(json.dumps(proof, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(dict(status=proof["status"], seconds=proof["seconds"])), flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "cache", "raw", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.study, args.cache, args.raw, args.out)
