"""Causal articulation observer and separately labeled reference calibration.

Frozen three-action covariance decoders supply increments. Unknown initial pose
is anchored to a source/reference mean, never to a query annotation. This is a
source-development experiment, not a validated full-pose product or E04 test.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.domain_background_probe import digest, multiply, centered_r2  # noqa:E402
from ghostmesh.metrics import pose_metrics  # noqa:E402

CONFIG = dict(
    scope="SOURCE DEVELOPMENT E01/E02/E03; A03/A07/A12 only; articulation only, no absolute hip-root position or E04",
    reference_pairs=[[int(t), int(t + 8)] for t in range(39, 256, 24)],
    reference_people={
        "E01": ["S01", "S02"],
        "E02": ["S11", "S12"],
        "E03": ["S21", "S22"],
    },
    reference_budget="2people x3actions x10pairs x2poses=120poses;60lag8increments; same reference set for all calibrated controls",
    validation_people=["S04", "S09", "S14", "S19", "S24", "S29"],
    leak_grid=[0.0, 0.5, 0.8, 0.95, 1.0],
    gain_grid=[0.0, 0.25, 0.5, 1.0, 2.0],
    anchor_grid=[0.0, 0.5, 1.0],
    bias_grid=[0.0, 0.5, 1.0],
    recurrence="state=anchor+rho*(previous_state-anchor)+gain*(predicted_delta-bias). New record or missing8frame step resets to anchor. First frame has unknown state, not a hidden reference pose.",
    calibration="Anchor interpolates source mean and120reference mean. Bias is mean prediction-minus-true increment over60reference pairs, with source-validation-selected shrinkage.",
    selection="Minimum pooled source-validation articulation NJE; query people4/9 disjoint from reference TRAINpeople1/2. All parameters frozen before held-room query scoring. First grid tie.",
    limitations="Prior-anchored articulation; no GT previous pose, query mean, action identity or future CSI supplied to recurrence. Development rotations reuse source rooms. Not final or new-room efficacy proof.",
    reference_selection_limit="Source validation reference people are in the frozen decoder's training set; target reference people are unseen by that decoder. Source calibration residuals can therefore be optimistic.",
    shuffle_seed=90210,
)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def track(delta, record, ends, anchor, rho, gain, bias):
    """Label-free streaming recurrence, with explicit discontinuity reset."""
    if not 0 <= rho <= 1 or gain < 0:
        raise ValueError("Invalid observer parameters")
    delta = np.asarray(delta, float)
    if (
        delta.shape[1:] != (17, 2)
        or len(delta) != len(record)
        or len(delta) != len(ends)
    ):
        raise ValueError(
            "Aligned 17joint increments, boundaries and endpoints required"
        )
    result = np.empty_like(delta)
    for name in np.unique(record):
        indices = np.flatnonzero(record == name)
        if np.any(np.diff(ends[indices]) <= 0):
            raise ValueError("Record endpoints must strictly increase")
        state = np.array(anchor, float)
        previous = None
        for ix in indices:
            if previous is None or ends[ix] - previous != 8:
                state = np.array(anchor, float)
            else:
                state = anchor + rho * (state - anchor) + gain * (delta[ix] - bias)
            # Preserve the articulation factorization exactly.
            state = state - state[[11, 12]].mean(0, keepdims=True)
            result[ix] = state
            previous = ends[ix]
    return result


def reference_ids(meta, room):
    chosen = []
    for person in CONFIG["reference_people"][room]:
        for action in ("A03", "A07", "A12"):
            record = f"{room}/{person}/{action}"
            for pair in CONFIG["reference_pairs"]:
                ids = []
                for end in pair:
                    ix = np.flatnonzero(
                        (meta["record"] == record) & (meta["end"] == end)
                    )
                    if len(ix) != 1:
                        raise ValueError(
                            "Registered reference has unavailable RF history: "
                            + record
                            + str(end)
                        )
                    ids.append(int(ix[0]))
                chosen.append(ids)
    result = np.asarray(chosen)
    if result.shape != (60, 2) or len(np.unique(result)) != 120:
        raise ValueError("Reference budget differs")
    return result


def nje(prediction, truth):
    scale = np.maximum(np.linalg.norm(truth.max(1) - truth.min(1), axis=1), 1e-8)
    return float((np.linalg.norm(prediction - truth, axis=2) / scale[:, None]).mean())


def calibrator(meta, predicted_delta, room, source_anchor):
    pairs = reference_ids(meta, room)
    true_delta = meta["pose"][pairs[:, 1]] - meta["pose"][pairs[:, 0]]
    return dict(
        source=source_anchor,
        reference=meta["pose"][pairs.ravel()].mean(0),
        bias=(predicted_delta[pairs[:, 1]] - true_delta).mean(0),
        pairs=pairs,
    )


def zero_calibrator(source_anchor):
    return dict(
        source=source_anchor,
        reference=source_anchor,
        bias=np.zeros((17, 2)),
        pairs=np.empty((0, 2), int),
    )


def apply(meta, delta, pick, calibration, params):
    rho, gain, alpha, beta = params
    anchor = calibration["source"] + alpha * (
        calibration["reference"] - calibration["source"]
    )
    return track(
        delta[pick],
        meta["record"][pick],
        meta["end"][pick],
        anchor,
        rho,
        gain,
        beta * calibration["bias"],
    )


def select(meta, delta, held_room, source_anchor, mode):
    grid = itertools.product(
        CONFIG["leak_grid"],
        CONFIG["gain_grid"],
        [0.0] if mode == "zero" else CONFIG["anchor_grid"],
        [0.0] if mode == "zero" else CONFIG["bias_grid"],
    )
    validation = (meta["room"] != held_room) & np.isin(
        meta["person"], CONFIG["validation_people"]
    )
    ref = {
        room: (
            zero_calibrator(source_anchor)
            if mode == "zero"
            else calibrator(meta, delta, room, source_anchor)
        )
        for room in ("E01", "E02", "E03")
        if room != held_room
    }
    rows = []
    for params in grid:
        p = []
        y = []
        for room, c in ref.items():
            pick = validation & (meta["room"] == room)
            p.append(apply(meta, delta, pick, c, params))
            y.append(meta["pose"][pick])
        rows.append(
            dict(
                parameters=list(params),
                validation_nje=nje(np.concatenate(p), np.concatenate(y)),
            )
        )
    chosen = min(rows, key=lambda r: r["validation_nje"])
    # Strong matched static control selects its own anchor interpolation.
    static = []
    for alpha in [0.0] if mode == "zero" else CONFIG["anchor_grid"]:
        p = []
        y = []
        for room, c in ref.items():
            pick = validation & (meta["room"] == room)
            anchor = c["source"] + alpha * (c["reference"] - c["source"])
            p.append(np.broadcast_to(anchor, meta["pose"][pick].shape))
            y.append(meta["pose"][pick])
        static.append(
            dict(alpha=alpha, validation_nje=nje(np.concatenate(p), np.concatenate(y)))
        )
    return chosen, rows, min(static, key=lambda r: r["validation_nje"]), static


def track_scores(prediction, truth, truth_delta, record, action, ends):
    """Pose scores use every state; increment scores require a previous state."""

    def aggregate(p, y, dy, groups, times):
        metric = pose_metrics(p, y, np.ones(y.shape[:2], bool))
        valid = np.zeros(len(p), bool)
        valid[1:] = (groups[1:] == groups[:-1]) & (np.diff(times) == 8)
        derived = np.zeros_like(p)
        derived[1:] = np.diff(p, axis=0)
        target_delta = dy[valid]
        denominator = float(np.square(target_delta).sum())
        return dict(
            relative_nje=metric["normalized_joint_error"],
            relative_pck05=metric["pck_curve"]["0.05"],
            relative_pck2=metric["pck_curve"]["0.2"],
            pose_motion=centered_r2(p, y, groups),
            delta_motion=centered_r2(derived[valid], target_delta, groups[valid])
            if valid.any()
            else dict(r2=None, per_record=[]),
            delta_zero_r2=1
            - float(np.square(derived[valid] - target_delta).sum()) / denominator
            if denominator > 0
            else None,
            derived_delta_valid_windows=int(valid.sum()),
        )

    result = aggregate(prediction, truth, truth_delta, record, ends)
    result["per_action"] = {
        name: aggregate(
            prediction[ix], truth[ix], truth_delta[ix], record[ix], ends[ix]
        )
        for name in np.unique(action)
        for ix in [action == name]
    }
    return result


def run(experiment, out):
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads((experiment / "artifact_manifest.json").read_text())
    source_results = json.loads((experiment / "results.json").read_text())
    if source_results["registration"]["config"]["selected_actions"] != [
        "A03",
        "A07",
        "A12",
    ]:
        raise ValueError("Only declared3action development source accepted")
    with np.load(experiment / "targets-and-metadata.npz", allow_pickle=False) as data:
        meta = {k: data[k] for k in data.files}
    if set(meta["room"]) != {"E01", "E02", "E03"}:
        raise ValueError("Forbidden room")
    names = ["targets-and-metadata.npz", "results.json", "registration.json"]
    names += [
        room + suffix
        for room in ("E01", "E02", "E03")
        for suffix in (
            "-congruence_dynamics-design.npy",
            "-congruence_dynamics-model-prediction.npz",
        )
    ]
    inputs = {name: digest(experiment / name) for name in names}
    for name, observed in inputs.items():
        if name in manifest and manifest[name] != observed:
            raise ValueError("Source decoder artifact changed")
    out.mkdir(parents=True)
    sources = {
        str(p.relative_to(ROOT)): digest(p)
        for p in (
            Path(__file__),
            ROOT / "scripts/domain_covariance_source_probe.py",
            ROOT / "scripts/domain_background_probe.py",
            ROOT / "ghostmesh/metrics.py",
        )
    }
    for name in sources:
        dest = out / "executed-source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    save(
        out / "registration.json",
        dict(
            config=CONFIG,
            inputs_sha256=inputs,
            source_sha256=sources,
            source_experiment=str(experiment),
        ),
    )
    frozen = {}
    for room in ("E01", "E02", "E03"):
        with np.load(
            experiment / (room + "-congruence_dynamics-model-prediction.npz"),
            allow_pickle=False,
        ) as model:
            z = np.load(
                experiment / (room + "-congruence_dynamics-design.npy"),
                allow_pickle=False,
            )
            delta = (
                multiply(z - model["x_mean"], model["delta_weights"])
                + model["y_mean"][34:]
            ).reshape(-1, 17, 2)
            np.testing.assert_array_equal(delta[model["held_indices"]], model["delta"])
        training = (meta["room"] != room) & ~np.isin(
            meta["person"], CONFIG["validation_people"]
        )
        anchor = meta["pose"][training].mean(0)
        frozen[room] = dict(
            delta=delta,
            source_anchor=anchor,
            selection={
                mode: select(meta, delta, room, anchor, mode)
                for mode in ("zero", "calibrated")
            },
        )
        save(
            out / (room + "-selection.json"),
            {
                mode: dict(
                    chosen=value[0],
                    grid=value[1],
                    static_chosen=value[2],
                    static_grid=value[3],
                )
                for mode, value in frozen[room]["selection"].items()
            },
        )
    # Freeze every fold's choices before any query NJE or motion score.
    results = {}
    for room, bundle in frozen.items():
        delta, anchor = bundle["delta"], bundle["source_anchor"]
        common = (meta["room"] == room) & ~np.isin(
            meta["person"], CONFIG["reference_people"][room]
        )
        results[room] = {}
        for mode, selection in bundle["selection"].items():
            calibration = (
                zero_calibrator(anchor)
                if mode == "zero"
                else calibrator(meta, delta, room, anchor)
            )
            params = selection[0]["parameters"]
            pred = apply(meta, delta, common, calibration, params)
            alpha = selection[2]["alpha"]
            static = np.broadcast_to(
                anchor + alpha * (calibration["reference"] - anchor), pred.shape
            ).copy()
            sequence, action = meta["record"][common], meta["action"][common]
            consecutive = np.zeros(len(pred), bool)
            ids = np.flatnonzero(common)
            for i in range(1, len(pred)):
                if (
                    sequence[i] == sequence[i - 1]
                    and meta["end"][ids[i]] - meta["end"][ids[i - 1]] == 8
                ):
                    consecutive[i] = True
            score = track_scores(
                pred,
                meta["pose"][common],
                meta["delta"][common],
                sequence,
                action,
                meta["end"][common],
            )
            baseline = track_scores(
                static,
                meta["pose"][common],
                meta["delta"][common],
                sequence,
                action,
                meta["end"][common],
            )
            shuffled_delta = delta.copy()
            rng = np.random.default_rng(CONFIG["shuffle_seed"])
            for name in np.unique(sequence):
                group = ids[sequence == name]
                shuffled_delta[group] = delta[rng.permutation(group)]
            shuffled_pred = apply(meta, shuffled_delta, common, calibration, params)
            shuffled_score = track_scores(
                shuffled_pred,
                meta["pose"][common],
                meta["delta"][common],
                sequence,
                action,
                meta["end"][common],
            )
            np.savez_compressed(
                out / (room + "-" + mode + ".npz"),
                indices=np.flatnonzero(common),
                prediction=pred,
                static=static,
                shuffled=shuffled_pred,
                derived_delta_valid=consecutive,
                reference_pairs=calibration["pairs"],
            )
            results[room][mode] = dict(
                parameters=params,
                static_alpha=alpha,
                query_people=sorted(set(meta["person"][common])),
                query_windows=int(common.sum()),
                observer=score,
                static=baseline,
                input_delta_shuffle_articulation_nje=shuffled_score["relative_nje"],
                input_delta_shuffle_pose_motion=shuffled_score["pose_motion"],
            )
    for name, expected in inputs.items():
        if digest(experiment / name) != expected:
            raise ValueError("Input changed")
    for name, expected in sources.items():
        if digest(ROOT / name) != expected:
            raise ValueError("Source changed")
    save(out / "results.json", dict(config=CONFIG, results=results))
    save(
        out / "artifact-manifest.json",
        {
            str(p.relative_to(out)): digest(p)
            for p in sorted(out.rglob("*"))
            if p.is_file()
        },
    )
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.experiment, args.out)
