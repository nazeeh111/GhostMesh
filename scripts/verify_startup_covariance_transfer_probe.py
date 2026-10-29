"""Independently verify the completed SOURCE startup covariance experiment.

No model refit or RF reconstruction is performed. Reference features use SciPy
full generalized eigenproblems, independently of the fitter's Cholesky whitening.
Ridge candidates use direct positive-definite solves rather than spectral solves.
All consumed arrays are SOURCE E01-E03; private poses/features are not exported.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import eigvalsh, solve
from scipy.linalg.blas import dgemm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.research_numeric_readback import numeric_difference  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def mm(a, b):
    return dgemm(
        alpha=1, a=np.asarray(a, dtype="float64"), b=np.asarray(b, dtype="float64")
    )


def npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def decode(untreated):
    coords = (
        np.asarray(untreated).reshape(-1, 8, 19, 7, 9)[:, :, :, 0].astype("float64")
    )
    c = np.zeros((*coords.shape[:-1], 3, 3), complex)
    c[..., (0, 1, 2), (0, 1, 2)] = coords[..., :3]
    off = (coords[..., 3:6] + 1j * coords[..., 6:9]) / np.sqrt(2)
    c[..., (0, 0, 1), (1, 2, 2)] = off
    c[..., (1, 2, 2), (0, 0, 1)] = off.conj()
    return c, coords


def independent_features(current, coords, startup):
    """startup is an own-record row index; no labels or query statistics."""
    ev = np.linalg.eigvalsh(current)
    require(
        np.all(ev > 0) and np.max(ev[..., -1] / ev[..., 0]) <= 1e10,
        "Non-SPD/ill-conditioned current covariance",
    )
    shapes = current / np.exp(np.log(ev).mean(-1))[..., None, None]
    anchor = shapes[startup].mean(0)
    anchor = (anchor + anchor.conj().swapaxes(-1, -2)) / 2
    anchor_ev = np.linalg.eigvalsh(anchor)
    b = anchor / np.exp(np.log(anchor_ev).mean(-1))[..., None, None]
    reference = (shapes + b) / 2
    logs = np.log(
        eigvalsh(shapes + 1e-4 * reference, b + 1e-4 * reference, driver="gv")
    )
    logs -= logs.mean(-1, keepdims=True)
    mean_volume = np.log(ev[startup]).mean(-1).mean(0)
    volume = np.log(ev).mean(-1) - mean_volume
    n = len(current)
    feature = dict(
        startup_shape=logs.reshape(n, 456),
        startup_shape_volume=np.concatenate((logs, volume[..., None]), axis=-1).reshape(
            n, 608
        ),
        raw_current=coords.reshape(n, 1368),
        startup_frozen_raw=np.broadcast_to(coords[startup].reshape(1368), (n, 1368)),
    )
    return feature, anchor, mean_volume


def roles(data, room):
    validation = (data["room"] != room) & np.isin(
        data["person"], ["S04", "S09", "S14", "S19", "S24", "S29"]
    )
    train = (data["room"] != room) & ~validation
    held = data["room"] == room
    require(
        tuple(len(set(data["person"][x])) for x in (train, validation, held))
        == (16, 4, 10),
        "SOURCE people roles differ",
    )
    return train, validation, held


def metric(pred, target, record):
    require(
        pred.shape == target.shape
        and np.isfinite(pred).all()
        and np.isfinite(target).all(),
        "Nonfinite/misaligned predictions or targets",
    )
    diagonal = np.linalg.norm(target.max(1) - target.min(1), axis=1)
    require(np.all(diagonal > 1e-6), "Degenerate teacher skeleton")
    error = np.linalg.norm(pred - target, axis=2) / diagonal[:, None]
    sse = energy = 0.0
    for name in np.unique(record):
        p, y = pred[record == name], target[record == name]
        p, y = p - p.mean(0), y - y.mean(0)
        sse += float(np.square(p - y).sum())
        energy += float(np.square(y).sum())
    return dict(
        nje=float(error.mean()),
        pck={f"{x:g}": float((error <= x).mean()) for x in (0.05, 0.1, 0.2)},
        r2=1 - sse / energy if energy > 0 else None,
        per_joint_nje=error.mean(0),
        per_joint_pck=(error <= 0.2).mean(0),
        valid_frames=len(target),
        valid_joints=len(target) * 17,
    )


def compare_score(actual, expected):
    values = [
        numeric_difference(
            expected["metrics"]["normalized_joint_error"],
            actual["nje"],
            name="metric NJE",
            atol=1e-10,
        )
    ]
    values.extend(
        numeric_difference(
            expected["metrics"]["pck_curve"][key],
            actual["pck"][key],
            name="metric PCK " + key,
            atol=1e-10,
        )
        for key in actual["pck"]
    )
    if actual["r2"] is None:
        require(
            expected["record_articulation_r2"] is None,
            "Missing variation denominator differs",
        )
    else:
        values.append(
            numeric_difference(
                expected["record_articulation_r2"],
                actual["r2"],
                name="metric R2",
                atol=1e-10,
            )
        )
    for j, row in enumerate(expected["metrics"]["per_joint"]):
        values.extend(
            [
                numeric_difference(
                    row["nje"],
                    actual["per_joint_nje"][j],
                    name="per-joint NJE",
                    atol=1e-10,
                ),
                numeric_difference(
                    row["pck"],
                    actual["per_joint_pck"][j],
                    name="per-joint PCK",
                    atol=1e-10,
                ),
            ]
        )
    require(
        actual["valid_frames"] == expected["metrics"]["valid_frames"]
        and actual["valid_joints"] == expected["metrics"]["valid_joints"],
        "Score coverage differs",
    )
    return max(values)


def check_map(feature, train, state):
    source = np.asarray(feature[train], dtype="float64")
    mean = source.mean(0)
    sd = np.maximum(source.std(0), 0.001)
    numeric_difference(
        state["feature_mean"], mean, name="feature mean", atol=1e-8, rtol=1e-8
    )
    numeric_difference(
        state["feature_scale"], sd, name="feature scale", atol=1e-8, rtol=1e-8
    )
    ids = np.random.default_rng(2027).choice(np.flatnonzero(train), 128, replace=False)
    np.testing.assert_array_equal(ids, state["bandwidth_sample_indices"])
    sample = (np.asarray(feature[ids], dtype="float64") - mean) / sd
    d = np.maximum(
        np.square(sample).sum(1)[:, None]
        + np.square(sample).sum(1)[None]
        - 2 * mm(sample, sample.T),
        0,
    )
    width = float(np.median(d[np.triu_indices(128, 1)]))
    numeric_difference(
        state["bandwidth_squared"], width, name="bandwidth", atol=1e-8, rtol=1e-8
    )
    rng = np.random.default_rng(2027)
    np.testing.assert_array_equal(
        state["weight"],
        rng.normal(size=(feature.shape[1], 512)) / np.sqrt(state["bandwidth_squared"]),
    )
    np.testing.assert_array_equal(state["offset"], rng.uniform(0, 2 * np.pi, 512))
    phi = np.empty((len(feature), 512))
    for start in range(0, len(feature), 128):
        x = (
            np.asarray(feature[start : start + 128], dtype="float64")
            - state["feature_mean"]
        ) / state["feature_scale"]
        phi[start : start + len(x)] = np.sqrt(2 / 512) * np.cos(
            mm(x, state["weight"]) + state["offset"]
        )
    rm = phi[train].mean(0)
    rs = np.maximum(phi[train].std(0), 0.001)
    numeric_difference(state["rff_mean"], rm, name="RFF mean", atol=1e-12)
    numeric_difference(state["rff_scale"], rs, name="RFF scale", atol=1e-12)
    diagnostics = dict(
        feature_mean_difference=float(np.max(abs(mean - state["feature_mean"]))),
        feature_sd_difference=float(np.max(abs(sd - state["feature_scale"]))),
        bandwidth_difference=abs(width - float(state["bandwidth_squared"])),
        rff_mean_difference=float(np.max(abs(rm - state["rff_mean"]))),
        rff_sd_difference=float(np.max(abs(rs - state["rff_scale"]))),
    )
    return (phi - state["rff_mean"]) / state["rff_scale"], diagnostics


def check_readout(z, target, train, val, room, state):
    xm = z[train].mean(0)
    ym = target[train].reshape(train.sum(), 34).mean(0)
    numeric_difference(state["x_mean"], xm, name="readout x_mean", atol=1e-12)
    numeric_difference(state["y_mean"], ym, name="readout y_mean", atol=0.0)
    x = z[train] - xm
    y = target[train].reshape(train.sum(), 34) - ym
    system, right = mm(x.T, x), mm(x.T, y)
    grid = []
    maxres = maxdirect = 0.0
    for j, alpha in enumerate((0.001, 0.01, 0.1, 1.0, 10.0)):
        matrix = system + train.sum() * alpha * np.eye(512)
        w = state["weights"][j]
        residual = float(
            np.linalg.norm(mm(matrix, w) - right) / max(np.linalg.norm(right), 1e-12)
        )
        direct = solve(matrix, right, assume_a="pos", check_finite=True)
        dv = mm(z[val] - xm, direct) + ym
        prediction = (mm(z[val] - xm, w) + ym).reshape(-1, 17, 2)
        nje = float(
            np.mean(
                [
                    metric(
                        prediction[room[val] == key],
                        target[val][room[val] == key],
                        np.repeat("validation", np.count_nonzero(room[val] == key)),
                    )["nje"]
                    for key in np.unique(room[val])
                ]
            )
        )
        require(np.isfinite(residual), "Nonfinite candidate equation residual")
        maxres = max(maxres, residual)
        maxdirect = max(
            maxdirect,
            numeric_difference(
                prediction.reshape(-1, 34),
                dv,
                name="candidate validation prediction",
                atol=1e-7,
            ),
        )
        grid.append(nje)
    selected = min(range(5), key=grid.__getitem__)
    require(selected == state["selected"], "Source-validation alpha selection differs")
    require(
        maxres < 1e-7 and maxdirect < 1e-7, "Candidate equation/direct solve mismatch"
    )
    return grid, maxres, maxdirect


def verify(experiment, cache, old, raw, local, output):
    if output.exists():
        raise FileExistsError(output)
    reg = read(experiment / "registration.json")
    results = read(experiment / "results.json")
    choice = read(experiment / "ALL-FOLDS-FROZEN-SELECTIONS.json")
    receipt = read(cache / "receipt.json")
    require(
        digest(cache / "receipt.json") == reg["feature_cache_receipt_sha256"]
        and digest(cache / "targets-and-metadata.npz") == reg["metadata_sha256"],
        "Cache metadata binding differs",
    )
    for name, h in reg["source_sha256"].items():
        require(
            digest(ROOT / name) == h
            and digest(experiment / "executed-source" / name) == h,
            "Fitted source changed: " + name,
        )
    require(
        digest(local / "results.json") == reg["local_results_sha256"]
        and digest(local / "verification.json") == reg["local_verification_sha256"]
        and digest(local / "artifact-manifest-final.json")
        == reg["local_manifest_sha256"],
        "Local receipt changed",
    )
    for key, directory in (("raw_comparator", raw), ("temporal_comparator", old)):
        for name, h in reg["comparator_artifact_sha256"][key].items():
            require(digest(directory / name) == h, "Comparator identity differs")
    require(
        digest(raw / "verification.json")
        == reg["comparator_artifact_sha256"]["raw_verification_sha256"]
        and digest(old / "verification.json")
        == reg["comparator_artifact_sha256"]["temporal_verification_sha256"],
        "Comparator proof changed",
    )
    untreated_name = "untreated_covariance-features.npy"
    require(
        digest(cache / untreated_name)
        == reg["feature_cache_arrays_sha256"][untreated_name],
        "Actual allocated untreated feature bytes differ",
    )
    data = npz(cache / "targets-and-metadata.npz")
    n = len(data["pose"])
    require(
        n == 26556
        and receipt["logical_rows"] == n
        and receipt["nominal_rows"] == 26730,
        "Logical prefix differs",
    )
    records = [
        f"E{room:02}/S{person:02}/A{action:02}"
        for room in range(1, 4)
        for person in range((room - 1) * 10 + 1, room * 10 + 1)
        for action in range(1, 28)
    ]
    require(
        [r["id"] for r in receipt["records"]] == records
        and set(data["record"]) == set(records),
        "Only full SOURCE records permitted",
    )
    quality = read(cache / "feature-quality.json")
    cursor = 0
    for row in quality:
        ends = np.asarray(row["retained_endpoints"])
        stop = cursor + len(ends)
        require(
            np.all(data["record"][cursor:stop] == row["record"])
            and np.array_equal(data["end"][cursor:stop], ends),
            "Quality-retained row prefix differs",
        )
        cursor = stop
    require(cursor == n, "Unexplained metadata tail")
    allocated = np.load(cache / untreated_name, mmap_mode="r", allow_pickle=False)
    require(allocated.shape == (26730, 9576), "Unexpected untreated allocation")
    untreated = allocated[:n]
    feature_receipt = read(experiment / "feature-receipt.json")
    feature = {
        name[:-13]: np.load(experiment / name, mmap_mode="r", allow_pickle=False)
        for name in feature_receipt["feature_sha256"]
    }
    for name, h in feature_receipt["feature_sha256"].items():
        require(digest(experiment / name) == h, "Fitted feature cache changed")
    bank = npz(experiment / "startup-references.npz")
    ident = read(experiment / "startup-identities.json")
    np.testing.assert_array_equal(bank["record"], records)
    maxanchor = maxvolume = maxrawanchor = maxrawanchorrelative = 0.0
    maxfeature = {k: 0.0 for k in feature}
    for ri, name in enumerate(records):
        ids = np.flatnonzero(data["record"] == name)
        first = int(np.argmin(data["end"][ids]))
        end = int(data["end"][ids[first]])
        current, coords = decode(untreated[ids])
        rebuilt, anchor, volume = independent_features(current, coords, first)
        maxanchor = max(
            maxanchor,
            numeric_difference(
                bank["anchors"][ri], anchor[None], name="startup anchors", atol=1e-12
            ),
        )
        maxvolume = max(
            maxvolume,
            numeric_difference(
                bank["mean_log_volume"][ri],
                volume,
                name="startup mean_log_volume",
                atol=1e-12,
            ),
        )
        raw_anchor = coords[first].reshape(-1)
        saved_raw_anchor = bank["raw_current"][ri]
        numeric_difference(
            saved_raw_anchor,
            raw_anchor,
            name="startup raw anchor",
            rtol=4e-15,
            atol=1e-10,
        )
        raw_difference = abs(raw_anchor - saved_raw_anchor)
        maxrawanchor = max(maxrawanchor, float(np.max(raw_difference)))
        maxrawanchorrelative = max(
            maxrawanchorrelative,
            float(np.max(raw_difference / np.maximum(abs(saved_raw_anchor), 1e-10))),
        )
        # Inverting/re-encoding sqrt(2)-scaled complex coordinates incurs
        # float64 roundoff. This is independent of the float32 feature check.
        np.testing.assert_allclose(raw_anchor, saved_raw_anchor, rtol=4e-15, atol=1e-10)
        require(
            bank["end"][ri] == end
            and ident["records"][name]["end"] == end
            and np.all(data["end"][ids] >= end),
            "Startup causality differs",
        )
        require(
            ident["records"][name]["current_states"] == list(range(end - 7, end + 1)),
            "Startup state support differs",
        )
        for arm, value in rebuilt.items():
            expected = feature[arm][ids]
            maxfeature[arm] = max(
                maxfeature[arm],
                numeric_difference(
                    expected,
                    value.astype("float32"),
                    name="saved feature " + arm,
                    atol=1e-6,
                    rtol=1e-6,
                ),
            )
            np.testing.assert_allclose(
                expected, value.astype("float32"), atol=1e-6, rtol=1e-6
            )
        if ri % 90 == 0:
            print(json.dumps(dict(verified_record=ri + 1, total=810)), flush=True)
    require(
        maxanchor < 1e-12 and maxvolume < 1e-12, "Startup anchor reconstruction differs"
    )
    counts = {
        str(k): int(v) for k, v in zip(*np.unique(bank["end"], return_counts=True))
    }
    require(
        counts
        == ident["nominal_endpoint_distribution"]
        == {"39": 806, "47": 1, "71": 3},
        "Startup distribution differs",
    )
    newmodels = rawequations = 0
    maxres = maxdirect = maxpred = maxmetric = 0.0
    mapchecks = {}
    gates = []
    score_count = 0
    original_roles = npz(raw / "cohort-indices.npz")
    rawreport = read(raw / "results.json")
    rawrows = {(r["room"], r["method"]): r for r in rawreport["results"]}
    for room in ("E01", "E02", "E03"):
        train, val, held = roles(data, room)
        ids = np.flatnonzero(held)
        for key, mask in (("train", train), ("validation", val)):
            np.testing.assert_array_equal(
                original_roles[room + "_" + key], np.flatnonzero(mask)
            )
        predictions = {}
        for arm in (
            "startup_shape",
            "startup_shape_volume",
            "startup_frozen_raw",
            "raw_current",
        ):
            frozen = arm == "raw_current"
            folder = raw if frozen else experiment
            suffix = "raw_current_control" if frozen else arm
            mapping = npz(
                folder / (room + "-" + suffix + ("-rff.npz" if frozen else "-map.npz"))
            )
            state = npz(folder / (room + "-" + suffix + "-model.npz"))
            z, diag = check_map(feature[arm], train, mapping)
            mapchecks[room + "/" + arm] = diag
            grid, residual, direct = check_readout(
                z, data["pose"], train, val, data["room"], state
            )
            require(np.isfinite(residual), "Nonfinite candidate equation residual")
            maxres = max(maxres, residual)
            maxdirect = max(maxdirect, direct)
            if frozen:
                rawequations += 5
                original_grid = rawrows[room, "raw_current_control"][
                    "source_validation_grid"
                ]
            else:
                newmodels += 1
                original_grid = choice[room][arm]["grid"]
            numeric_difference(
                [row["source_validation_nje"] for row in original_grid],
                grid,
                name="source-validation grid",
                atol=1e-11,
            )
            require(
                choice[room][arm]["alpha"]
                == (0.001, 0.01, 0.1, 1.0, 10.0)[int(state["selected"])],
                "Frozen alpha receipt differs",
            )
            p = (
                mm(z[held] - state["x_mean"], state["weights"][int(state["selected"])])
                + state["y_mean"]
            ).reshape(-1, 17, 2)
            saved = npz(experiment / (room + "-" + arm + "-predictions.npz"))
            np.testing.assert_array_equal(saved["indices"], ids)
            maxpred = max(
                maxpred,
                numeric_difference(
                    saved["prediction"], p, name="saved prediction", atol=1e-10
                ),
            )
            predictions[arm] = p
            print(json.dumps(dict(verified_model=room + "/" + arm)), flush=True)
        mean = data["pose"][train].mean(0)
        predictions["source_body_mean"] = np.broadcast_to(
            mean, (len(ids), 17, 2)
        ).copy()
        clock = predictions["source_body_mean"].copy()
        for end in np.unique(data["end"][ids]):
            source = train & (data["end"] == end)
            require(source.any(), "Clock TRAIN endpoint unexpectedly absent")
            clock[data["end"][ids] == end] = data["pose"][source].mean(0)
        predictions["source_endpoint_clock"] = clock
        temporal = npz(old / (room + "-congruence_dynamics-model-prediction.npz"))
        np.testing.assert_array_equal(temporal["held_indices"], ids)
        predictions["frozen_temporal"] = temporal["pose"]
        rng = np.random.default_rng(90210)
        perm = np.arange(len(ids))
        record = data["record"][ids]
        for name in np.unique(record):
            ix = np.flatnonzero(record == name)
            perm[ix] = rng.permutation(ix)
        np.testing.assert_array_equal(
            perm, np.load(experiment / (room + "-permutation.npy"))
        )
        checked = {}
        for method, p in predictions.items():
            saved = npz(experiment / (room + "-" + method + "-predictions.npz"))
            np.testing.assert_array_equal(saved["indices"], ids)
            maxpred = max(
                maxpred,
                numeric_difference(
                    saved["prediction"], p, name="saved prediction", atol=1e-10
                ),
            )
            expected = results["results"][room][method]
            actual = metric(p, data["pose"][ids], record)
            shuffled = metric(p[perm], data["pose"][ids], record)
            maxmetric = max(
                maxmetric,
                compare_score(actual, expected["aligned"]),
                compare_score(shuffled, expected["shuffled"]),
            )
            score_count += 2
            for field in ("person", "action"):
                values = data[field][ids]
                for key in np.unique(values):
                    ix = values == key
                    maxmetric = max(
                        maxmetric,
                        compare_score(
                            metric(p[ix], data["pose"][ids][ix], record[ix]),
                            expected["per_" + field][str(key)],
                        ),
                    )
                    score_count += 1
            checked[method] = dict(aligned=actual, shuffled=shuffled)
        p = checked["startup_shape"]["aligned"]
        nje = all(
            p["nje"] < checked[k]["aligned"]["nje"]
            for k in (
                "source_body_mean",
                "source_endpoint_clock",
                "raw_current",
                "startup_frozen_raw",
            )
        )
        pck = all(
            p["pck"][t] >= checked[k]["aligned"]["pck"][t]
            for k in ("source_body_mean", "source_endpoint_clock")
            for t in ("0.05", "0.2")
        )
        motion = p["r2"] is not None and p["r2"] > 0
        shuffle = p["nje"] < checked["startup_shape"]["shuffled"]["nje"]
        decision = dict(
            room=room,
            beats_all_four_controls_nje=bool(nje),
            pck_not_worse_than_mean_clock=bool(pck),
            positive_motion_r2=bool(motion),
            shuffle_worse_nje=bool(shuffle),
            passed=bool(nje and pck and motion and shuffle),
        )
        require(
            decision
            == next(x for x in results["gate"]["per_room"] if x["room"] == room),
            "Per-room gate differs",
        )
        gates.append(decision)
    require(
        maxpred < 1e-10 and maxmetric < 1e-10,
        "Selected prediction/metric readback differs",
    )
    require(
        results["gate"]["passed"] == all(g["passed"] for g in gates),
        "Overall gate differs",
    )
    proof = dict(
        scope="All810 SOURCE own-record startups and26556 fullquality rows; no E04/RGB/GPU/rawRFbuild/newfit",
        startup_anchors_verified=810,
        feature_rows_verified=26556,
        feature_arms_verified=4,
        startup_nominal_end_distribution=counts,
        maximum_anchor_difference=maxanchor,
        maximum_volume_mean_difference=maxvolume,
        maximum_raw_current_anchor_absolute_difference=maxrawanchor,
        maximum_raw_current_anchor_relative_difference=maxrawanchorrelative,
        maximum_feature_difference=maxfeature,
        new_source_models_verified=newmodels,
        new_candidate_equations_verified=newmodels * 5,
        frozen_raw_candidate_equations_verified=rawequations,
        source_maps_verified=len(mapchecks),
        map_readbacks=mapchecks,
        maximum_normal_equation_relative_residual=maxres,
        maximum_direct_solver_validation_prediction_difference=maxdirect,
        maximum_selected_prediction_difference=maxpred,
        maximum_score_difference=maxmetric,
        scores_and_subgroups_verified=score_count,
        all_three_source_selection_and_gates_verified=True,
        gate=results["gate"],
        local_result_sha256=reg["local_results_sha256"],
        local_verification_sha256=reg["local_verification_sha256"],
        source_sha256=reg["source_sha256"],
        verifier_source_sha256=digest(Path(__file__)),
        numeric_readback_source_sha256=digest(
            ROOT / "scripts/research_numeric_readback.py"
        ),
        byte_manifest_scope="Byte identities and numerical receipts do not certify scientific efficacy/licensing or solve unknown initial pose",
    )
    with output.open("x") as stream:
        stream.write(json.dumps(proof, indent=2, allow_nan=False) + "\n")
    print(json.dumps(proof, indent=2), flush=True)
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument(
        "--cache", type=Path, default=ROOT / "work/domain-covariance-full-cache-v1"
    )
    parser.add_argument(
        "--old", type=Path, default=ROOT / "work/domain-covariance-full27-v1"
    )
    parser.add_argument(
        "--raw", type=Path, default=ROOT / "work/domain-reference-covariance-v1"
    )
    parser.add_argument(
        "--local", type=Path, default=ROOT / "work/domain-startup-positive-v1"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    verify(
        args.experiment,
        args.cache,
        args.old,
        args.raw,
        args.local,
        args.output or args.experiment / "verification.json",
    )
