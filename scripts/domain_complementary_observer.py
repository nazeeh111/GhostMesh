"""Fuse frozen RF configuration and signed increments without a query-pose anchor.

This bounded SOURCE-development experiment predicts hip-root-relative articulation
only. It uses fixed A03/A07/A12 records and no E04 data. The two RF heads are frozen;
reference correction and temporal fusion are selected on source validation people.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.domain_background_probe import digest, multiply  # noqa:E402
from scripts.domain_motion_observer import (  # noqa:E402
    CONFIG as ANCHOR_CONFIG,
    reference_ids,
    track_scores,
)

CONFIG = dict(
    schema_version=1,
    scope="SOURCE DEVELOPMENT only E01/E02/E03, fixed A03/A07/A12; articulation only, no absolute hip-root position, E04 or 3D",
    rho_grid=[0.0, 0.5, 0.9],
    gain_grid=[0.0, 0.5, 1.0],
    offset_grid=[0.0, 1.0],
    bias_grid=[0.0, 1.0],
    view_grid=[0.0, 1.0],
    recurrence="state=rho*(previous+gain*corrected_delta)+(1-rho)*corrected_RFpose; first/gap resets to current corrected_RFpose. No ground-truth initial state",
    parameters="[rho,gain,pose_offset_strength,delta_bias_strength,view_strength]",
    view="Shared 2x2 map on centered reference increments, Ridge towards identity: lambda=.5*(predicted_energy+true_energy), minimum1e-12. Joint/room-specific nonlinear mapping and 3D geometry are not fitted",
    pose_correction="(RFpose-reference_RFpose_mean)@M+reference_RFpose_mean+alpha*(reference_GTmean-reference_RFpose_mean)",
    delta_correction="RFdelta@M-beta*mean(reference_RFdelta@M-reference_GTdelta)",
    reference_people=ANCHOR_CONFIG["reference_people"],
    reference_pairs=ANCHOR_CONFIG["reference_pairs"],
    reference_budget="120unique labeled poses:2people x3actions x10lag8pairs x2poses;60reference increments. Same labels for all calibrated controls",
    validation_people=ANCHOR_CONFIG["validation_people"],
    reference_selection_limit="Source-validation reference people are training people of the frozen decoder. Held-room reference people are unseen by that decoder; residual calibration selection can be optimistic",
    selection="Minimum pooled source-validation articulation NJE; refs TRAINpeople1/2, different query people4/9. Independent selectors for corrected_RFpose, no_motion_filter, static_anchor. First grid tie. Freeze all3fold choices before held-query scoring",
    controls=[
        "frozen_RFpose",
        "independently_selected_RFpose_correction",
        "source_or_independently_selected_reference_static_anchor",
        "same_filter_gain_zero",
        "independently_selected_no_motion_filter",
        "within_record_delta_input_shuffle_RFpose_order_fixed",
    ],
    shuffle_seed=90210,
    limitations="Development rotations reuse source rooms, represented3actions/one seed. Static configuration comes from RFpose head, not motion integration alone; complete absolute pose, new-action or calibrated product efficacy are not established",
)


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def project(pose):
    pose = np.asarray(pose, dtype=float)
    return pose - pose[..., [11, 12], :].mean(-2, keepdims=True)


def view_map(predicted_delta, true_delta):
    """Identity-regularized shared planar regression using reference labels only."""
    x, y = np.asarray(predicted_delta, float), np.asarray(true_delta, float)
    if (
        x.shape != y.shape
        or x.ndim != 3
        or x.shape[1:] != (17, 2)
        or not len(x)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("Matching finite reference increments[N,17,2] required")
    xc = (x - x.mean(0)).reshape(-1, 2)
    yc = (y - y.mean(0)).reshape(-1, 2)
    precision = max(0.5 * float(np.square(xc).sum() + np.square(yc).sum()), 1e-12)
    system = multiply(xc.T, xc) + precision * np.eye(2)
    right = multiply(xc.T, yc) + precision * np.eye(2)
    matrix = np.linalg.solve(system, right)
    residual = float(
        np.linalg.norm(multiply(system, matrix) - right)
        / max(np.linalg.norm(right), 1e-12)
    )
    if not np.isfinite(matrix).all() or not np.isfinite(residual) or residual > 1e-10:
        raise ValueError("Invalid reference view solve")
    return matrix, precision, residual


def zero_calibration(source_anchor):
    zero = np.zeros((17, 2))
    return dict(
        source_anchor=source_anchor,
        pose_reference_mean=zero,
        target_reference_mean=zero,
        delta_reference_mean=zero,
        true_delta_mean=zero,
        view=np.eye(2),
        view_precision=0.0,
        view_residual=0.0,
        reference_pairs=np.empty((0, 2), int),
    )


def calibration(meta, pose_prediction, delta_prediction, room, source_anchor):
    ids = reference_ids(meta, room)
    truth = meta["pose"][ids.ravel()]
    true_delta = meta["pose"][ids[:, 1]] - meta["pose"][ids[:, 0]]
    predicted_delta = delta_prediction[ids[:, 1]]
    matrix, precision, residual = view_map(predicted_delta, true_delta)
    return dict(
        source_anchor=source_anchor,
        pose_reference_mean=pose_prediction[ids.ravel()].mean(0),
        target_reference_mean=truth.mean(0),
        delta_reference_mean=predicted_delta.mean(0),
        true_delta_mean=true_delta.mean(0),
        view=matrix,
        view_precision=precision,
        view_residual=residual,
        reference_pairs=ids,
    )


def corrected(pose_prediction, delta_prediction, state, alpha, beta, view):
    """Query labels are deliberately absent from this prediction interface."""
    matrix = np.eye(2) + view * (state["view"] - np.eye(2))
    pivot = state["pose_reference_mean"]
    pose = np.einsum("...j,jk->...k", pose_prediction - pivot, matrix) + pivot
    pose += alpha * (state["target_reference_mean"] - pivot)
    delta = np.einsum("...j,jk->...k", delta_prediction, matrix)
    bias = (
        np.einsum("...j,jk->...k", state["delta_reference_mean"], matrix)
        - state["true_delta_mean"]
    )
    delta -= beta * bias
    return project(pose), project(delta)


def track(pose, delta, record, ends, rho, gain):
    pose, delta = np.asarray(pose, float), np.asarray(delta, float)
    record, ends = np.asarray(record), np.asarray(ends)
    if (
        pose.shape != delta.shape
        or pose.ndim != 3
        or pose.shape[1:] != (17, 2)
        or len(pose) != len(record)
        or len(record) != len(ends)
    ):
        raise ValueError(
            "Aligned RF configuration/increments[N,17,2] and boundaries required"
        )
    if (
        not np.isfinite(pose).all()
        or not np.isfinite(delta).all()
        or not np.isfinite(rho)
        or not np.isfinite(gain)
        or not 0 <= rho <= 1
        or gain < 0
    ):
        raise ValueError("Finite inputs, rho[0,1], nonnegative gain required")
    output = np.empty_like(pose)
    for name in np.unique(record):
        ids = np.flatnonzero(record == name)
        if np.any(np.diff(ends[ids]) <= 0):
            raise ValueError("Record endpoints must strictly increase")
        previous = None
        state = None
        for ix in ids:
            if previous is None or ends[ix] - previous != 8:
                state = pose[ix].copy()
            else:
                state = rho * (state + gain * delta[ix]) + (1 - rho) * pose[ix]
            state = project(state)
            output[ix] = state
            previous = ends[ix]
    return output


def apply(meta, pose, delta, pick, state, params):
    rho, gain, alpha, beta, view = params
    p, d = corrected(pose[pick], delta[pick], state, alpha, beta, view)
    return track(p, d, meta["record"][pick], meta["end"][pick], rho, gain)


def nje(prediction, truth):
    # Same valid-box convention as the frozen reported metrics.
    scale = np.linalg.norm(truth.max(1) - truth.min(1), axis=1)
    use = np.isfinite(scale) & (scale > 1e-6)
    if not use.any():
        raise ValueError("No valid reference boxes")
    return float(
        (np.linalg.norm(prediction[use] - truth[use], axis=2) / scale[use, None]).mean()
    )


def candidates(mode, control):
    calibrated = mode == "calibrated"
    if mode not in ("zero", "calibrated") or control not in (
        "full",
        "pose",
        "no_motion",
        "static",
    ):
        raise ValueError("Unknown calibration mode/control")
    if control == "static":
        return [
            (0.0, 0.0, alpha, 0.0, 0.0)
            for alpha in CONFIG["offset_grid"]
            if calibrated or alpha == 0
        ]
    return list(
        itertools.product(
            CONFIG["rho_grid"] if control in ("full", "no_motion") else [0.0],
            CONFIG["gain_grid"] if control == "full" else [0.0],
            CONFIG["offset_grid"] if calibrated else [0.0],
            CONFIG["bias_grid"] if calibrated and control == "full" else [0.0],
            CONFIG["view_grid"] if calibrated else [0.0],
        )
    )


def select(meta, pose, delta, held_room, source_anchor, mode):
    validation = (meta["room"] != held_room) & np.isin(
        meta["person"], CONFIG["validation_people"]
    )
    source_rooms = sorted(set(meta["room"][validation]))
    states = {
        room: zero_calibration(source_anchor)
        if mode == "zero"
        else calibration(meta, pose, delta, room, source_anchor)
        for room in source_rooms
    }
    selections = {}
    for control in ("full", "pose", "no_motion", "static"):
        rows = []
        for params in candidates(mode, control):
            predictions, truth = [], []
            for room, state in states.items():
                pick = validation & (meta["room"] == room)
                if control == "static":
                    anchor = source_anchor + params[2] * (
                        state["target_reference_mean"] - source_anchor
                    )
                    prediction = np.broadcast_to(anchor, meta["pose"][pick].shape)
                else:
                    prediction = apply(meta, pose, delta, pick, state, params)
                predictions.append(prediction)
                truth.append(meta["pose"][pick])
            rows.append(
                dict(
                    parameters=list(params),
                    validation_nje=nje(
                        np.concatenate(predictions), np.concatenate(truth)
                    ),
                )
            )
        selections[control] = dict(
            chosen=min(rows, key=lambda r: r["validation_nje"]), grid=rows
        )
    return selections


def shuffled(delta, record, indices):
    output = delta.copy()
    rng = np.random.default_rng(CONFIG["shuffle_seed"])
    for name in np.unique(record[indices]):
        ids = indices[record[indices] == name]
        output[ids] = delta[rng.permutation(ids)]
    return output


def inputs(experiment):
    result = json.loads((experiment / "results.json").read_text())
    verification = json.loads((experiment / "verification.json").read_text())
    if (
        result["registration"]["config"]["selected_actions"] != ["A03", "A07", "A12"]
        or verification["models_verified"] != 9
        or verification["raw_records_reconstructed"] != 90
    ):
        raise ValueError("Verified90record source-development experiment required")
    manifest = json.loads((experiment / "artifact_manifest.json").read_text())
    names = [
        "targets-and-metadata.npz",
        "results.json",
        "registration.json",
        "verification.json",
    ]
    names += [
        room + suffix
        for room in ("E01", "E02", "E03")
        for suffix in (
            "-congruence_dynamics-design.npy",
            "-congruence_dynamics-model-prediction.npz",
        )
    ]
    bindings = {name: digest(experiment / name) for name in names}
    for name, observed in bindings.items():
        if name != "verification.json" and (
            name not in manifest or manifest[name] != observed
        ):
            raise ValueError("Frozen source artifact mismatch: " + name)
    with np.load(experiment / "targets-and-metadata.npz", allow_pickle=False) as values:
        meta = {key: values[key] for key in values.files}
    if (
        set(meta["room"]) != {"E01", "E02", "E03"}
        or set(meta["action"]) != {"A03", "A07", "A12"}
        or set(meta["person"]) != {f"S{i:02}" for i in range(1, 31)}
    ):
        raise ValueError("Forbidden/missing source room, person or action")
    heads = {}
    for room in ("E01", "E02", "E03"):
        with np.load(
            experiment / (room + "-congruence_dynamics-model-prediction.npz"),
            allow_pickle=False,
        ) as model:
            z = np.load(
                experiment / (room + "-congruence_dynamics-design.npy"),
                allow_pickle=False,
            )
            p = (
                multiply(z - model["x_mean"], model["pose_weights"])
                + model["y_mean"][:34]
            ).reshape(-1, 17, 2)
            d = (
                multiply(z - model["x_mean"], model["delta_weights"])
                + model["y_mean"][34:]
            ).reshape(-1, 17, 2)
            np.testing.assert_allclose(
                p[model["held_indices"]], model["pose"], rtol=0, atol=1e-9
            )
            np.testing.assert_allclose(
                d[model["held_indices"]], model["delta"], rtol=0, atol=1e-9
            )
        heads[room] = (p, d)
    return meta, heads, bindings


def array_state(state):
    return {key: np.asarray(value) for key, value in state.items()}


def run(experiment, out):
    if out.exists():
        raise FileExistsError(out)
    started = time.monotonic()
    meta, heads, bindings = inputs(experiment)
    sources = [
        Path(__file__),
        ROOT / "scripts/domain_motion_observer.py",
        ROOT / "scripts/domain_background_probe.py",
        ROOT / "ghostmesh/metrics.py",
    ]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in sources}
    out.mkdir(parents=True)
    for source in sources:
        destination = out / "executed-source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    save(
        out / "registration.json",
        dict(
            config=CONFIG,
            source_sha256=hashes,
            inputs_sha256=bindings,
            source_experiment=str(experiment),
            software=dict(python=platform.python_version(), numpy=np.__version__),
        ),
    )
    frozen = {}
    for room, (pose, delta) in heads.items():
        train = (meta["room"] != room) & ~np.isin(
            meta["person"], CONFIG["validation_people"]
        )
        anchor = meta["pose"][train].mean(0)
        selections = {
            mode: select(meta, pose, delta, room, anchor, mode)
            for mode in ("zero", "calibrated")
        }
        frozen[room] = dict(anchor=anchor, selections=selections)
        save(out / (room + "-selection.json"), selections)
    # No held-query metrics until ALL folds' source-only choices are on disk.
    results, parity = {}, {}
    for room, (pose, delta) in heads.items():
        results[room] = {}
        anchor = frozen[room]["anchor"]
        common = (meta["room"] == room) & ~np.isin(
            meta["person"], CONFIG["reference_people"][room]
        )
        for mode in ("zero", "calibrated"):
            state = (
                zero_calibration(anchor)
                if mode == "zero"
                else calibration(meta, pose, delta, room, anchor)
            )
            np.savez(
                out / (room + "-" + mode + "-calibration.npz"), **array_state(state)
            )
            selection = frozen[room]["selections"][mode]
            params = selection["full"]["chosen"]["parameters"]
            scopes = [("common8", common)]
            if mode == "zero":
                scopes.append(("all10", meta["room"] == room))
            results[room][mode] = {}
            for scope, pick in scopes:
                indices = np.flatnonzero(pick)
                full = apply(meta, pose, delta, pick, state, params)
                no_delta = list(params)
                no_delta[1] = 0.0
                static_alpha = selection["static"]["chosen"]["parameters"][2]
                predictions = dict(
                    complementary=full,
                    frozen_RFpose=pose[pick],
                    corrected_RFpose=apply(
                        meta,
                        pose,
                        delta,
                        pick,
                        state,
                        selection["pose"]["chosen"]["parameters"],
                    ),
                    same_filter_gain_zero=apply(
                        meta, pose, delta, pick, state, no_delta
                    ),
                    selected_no_motion_filter=apply(
                        meta,
                        pose,
                        delta,
                        pick,
                        state,
                        selection["no_motion"]["chosen"]["parameters"],
                    ),
                    static_anchor=np.broadcast_to(
                        anchor
                        + static_alpha * (state["target_reference_mean"] - anchor),
                        full.shape,
                    ).copy(),
                    delta_input_shuffle=apply(
                        meta,
                        pose,
                        shuffled(delta, meta["record"], indices),
                        pick,
                        state,
                        params,
                    ),
                )
                metrics = {
                    name: track_scores(
                        value,
                        meta["pose"][pick],
                        meta["delta"][pick],
                        meta["record"][pick],
                        meta["action"][pick],
                        meta["end"][pick],
                    )
                    for name, value in predictions.items()
                }
                np.savez_compressed(
                    out / (room + "-" + mode + "-" + scope + ".npz"),
                    indices=indices,
                    **predictions,
                )
                with np.load(
                    out / (room + "-" + mode + "-calibration.npz"), allow_pickle=False
                ) as z:
                    loaded = {key: z[key] for key in z.files}
                # Re-open the frozen RF head arrays, rather than retaining a fitted
                # in-memory object, and reconstruct all controls with saved choices.
                with np.load(
                    experiment / (room + "-congruence_dynamics-model-prediction.npz"),
                    allow_pickle=False,
                ) as model:
                    z = np.load(
                        experiment / (room + "-congruence_dynamics-design.npy"),
                        allow_pickle=False,
                    )
                    rp = (
                        multiply(z - model["x_mean"], model["pose_weights"])
                        + model["y_mean"][:34]
                    ).reshape(-1, 17, 2)
                    rd = (
                        multiply(z - model["x_mean"], model["delta_weights"])
                        + model["y_mean"][34:]
                    ).reshape(-1, 17, 2)
                chosen = json.loads((out / (room + "-selection.json")).read_text())[
                    mode
                ]
                replay_params = chosen["full"]["chosen"]["parameters"]
                replay_no_delta = list(replay_params)
                replay_no_delta[1] = 0.0
                replay_alpha = chosen["static"]["chosen"]["parameters"][2]
                replay = dict(
                    complementary=apply(meta, rp, rd, pick, loaded, replay_params),
                    frozen_RFpose=rp[pick],
                    corrected_RFpose=apply(
                        meta,
                        rp,
                        rd,
                        pick,
                        loaded,
                        chosen["pose"]["chosen"]["parameters"],
                    ),
                    same_filter_gain_zero=apply(
                        meta, rp, rd, pick, loaded, replay_no_delta
                    ),
                    selected_no_motion_filter=apply(
                        meta,
                        rp,
                        rd,
                        pick,
                        loaded,
                        chosen["no_motion"]["chosen"]["parameters"],
                    ),
                    static_anchor=np.broadcast_to(
                        loaded["source_anchor"]
                        + replay_alpha
                        * (loaded["target_reference_mean"] - loaded["source_anchor"]),
                        full.shape,
                    ),
                    delta_input_shuffle=apply(
                        meta,
                        rp,
                        shuffled(rd, meta["record"], indices),
                        pick,
                        loaded,
                        replay_params,
                    ),
                )
                with np.load(
                    out / (room + "-" + mode + "-" + scope + ".npz"), allow_pickle=False
                ) as stored:
                    differences = {
                        name: float(abs(value - stored[name]).max())
                        for name, value in replay.items()
                    }
                if max(differences.values()) > 1e-9:
                    raise ValueError("Independent saved-state readback differs")
                parity[room + "/" + mode + "/" + scope] = differences
                results[room][mode][scope] = dict(
                    parameters=params,
                    query_people=sorted(set(meta["person"][pick])),
                    query_windows=int(pick.sum()),
                    methods=metrics,
                )
    for name, expected in hashes.items():
        if digest(ROOT / name) != expected:
            raise ValueError("Source changed")
    for name, expected in bindings.items():
        if digest(experiment / name) != expected:
            raise ValueError("Input changed")
    save(
        out / "results.json",
        dict(config=CONFIG, results=results, seconds=time.monotonic() - started),
    )
    save(
        out / "verification.json",
        dict(
            saved_state_prediction_max_abs=parity, source_and_input_rehash_passed=True
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
    print(
        json.dumps(dict(seconds=time.monotonic() - started, completed=list(results))),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.experiment, args.out)
