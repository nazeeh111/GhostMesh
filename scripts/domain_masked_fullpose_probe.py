"""Prospective full-image 2D root heads for already frozen CSI body encoders.

This source-only E02 development experiment never updates an encoder. Fitting
ends at a sealed ready receipt; held metrics require an explicit second command.
The provided 2D teacher plane is not camera calibration or a 3D coordinate frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import shutil
import sys

import numpy as np
from scipy.linalg import eigh
from scipy import __version__ as SCIPY_VERSION

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.metrics import pose_metrics  # noqa: E402
from ghostmesh.reference_covariance import current_covariances  # noqa: E402
from ghostmesh.covariance_dynamics import hermitian_coordinates  # noqa: E402
from scripts.domain_background_probe import centered_r2, digest, multiply  # noqa: E402
from scripts.domain_covariance_cached_probe import load_inputs  # noqa: E402
from scripts.domain_covariance_full_cache import verify_hashes  # noqa: E402
from scripts.domain_covariance_source_probe import fold_masks  # noqa: E402

ENCODERS = (
    "random_frozen",
    "masked_only",
    "masked_temporal",
    "masked_shuffled",
    "supervised",
)
ARMS = ENCODERS + ("raw_current",)
ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0)
BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
CONFIG = dict(
    room="E02",
    primary="masked_temporal",
    alphas=ALPHAS,
    betas=BETAS,
    coordinates="Unclipped supplied full-image COCO17 2D teacher coordinates divided by640; no camera calibration or3D reconstruction",
    model="Frozen hip-relative body prediction plus a source-only two-coordinate hip-midpoint readout; body hip midpoint must already be zero, never silently recentered",
    design="Existing frozen source-training RFF512 maps; no encoder or input-map refit",
    root_equation="(Xc.T Xc+n*alpha*I)W=Xc.T(root-root_TRAIN_mean); prediction=root_TRAIN_mean+beta*(Z-x_TRAIN_mean)W",
    selection="Equal-source-room VAL full-image NJE of frozen body plus root; ties prefer lower beta, then larger alpha. Every arm's choice frozen before held metrics",
    controls="Six matched body/design arms with identical root budgets; source full-pose mean and source endpoint-clock mean. Original temporal body-only gate remains separate",
    gate="Primary full-image NJE lower and PCK.05/.2 no worse than source mean/clock/random/supervised/raw_current; full and body centered-motion R2 positive; aligned NJE better than within-record joint-prediction shuffle",
    limits="One previously exposed E02 source-development fold and one seed. A beta0 root is a static source prior and establishes no hip-position motion reconstruction. A full-image score does not replace the separate articulation gate or reserved five-seed evaluation",
)


def save_json(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(payload)


def save_array(path, value):
    with Path(path).open("xb") as stream:
        np.save(stream, value, allow_pickle=False)


def save_npz(path, **value):
    with Path(path).open("xb") as stream:
        np.savez(stream, **value)


def real_numeric(value):
    """Reject complex/object/string data before any conversion can lose values."""
    x = np.asarray(value)
    if x.dtype.kind not in "iuf":
        raise ValueError("Expected real-valued numeric array")
    return x


def finite(value, shape=None):
    x = real_numeric(value)
    with np.errstate(over="ignore", invalid="ignore"):
        x = np.asarray(x, dtype=np.float64)
    if (
        (shape is not None and x.shape != shape)
        or not x.size
        or not np.isfinite(x).all()
    ):
        raise ValueError("Expected finite compatible numerical array")
    return x


def roles_valid(data, train, validation, held):
    n = len(data["record"])
    if np.asarray(data["end"]).dtype.kind not in "iu":
        raise ValueError("Canonical endpoints must use real integer indices")
    masks = (train, validation, held)
    if any(x.shape != (n,) or x.dtype != bool or not x.any() for x in masks):
        raise ValueError("Nonempty Boolean source roles required")
    if not np.all(sum(x.astype(int) for x in masks) == 1):
        raise ValueError("Source roles must partition without overlap")
    for key in ("person", "record"):
        groups = [set(data[key][x]) for x in masks]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
            raise ValueError("Source role people or recordings overlap")


def join_full_targets(raw_pose, raw_valid, records, data):
    """Exact frame join is separate from source-only statistical fitting."""
    offsets = {r["id"]: (int(r["offset"]), int(r["frames"])) for r in records}
    if len(offsets) != len(records) or any(str(x).startswith("E04/") for x in offsets):
        raise ValueError("Duplicate or reserved-room raw records")
    if np.asarray(data["end"]).dtype.kind not in "iu":
        raise ValueError("Canonical endpoints must use real integer indices")
    if np.asarray(raw_valid).dtype != bool:
        raise ValueError("Raw target validity must use the Boolean schema")
    indices = []
    for name, end in zip(data["record"], data["end"]):
        if name not in offsets or not 0 <= int(end) < offsets[name][1]:
            raise ValueError("Canonical endpoint is outside its raw recording")
        indices.append(offsets[name][0] + int(end))
    indices = np.asarray(indices, dtype=np.int64)
    target = finite(raw_pose[indices], (len(indices), 17, 2))
    if not np.asarray(raw_valid[indices], bool).all():
        raise ValueError("Full-image target has invalid teacher joints")
    body = target - target[:, [11, 12]].mean(1, keepdims=True)
    if not np.allclose(body, data["pose"], atol=1e-12, rtol=0):
        raise ValueError("Full-image join disagrees with frozen hip-relative targets")
    return target, indices


def apply_rff(feature, state):
    """Transform only; every statistic comes from the sealed source map."""
    n, width = feature.shape
    mean = finite(state["feature_mean"], (width,))
    scale = finite(state["feature_scale"], (width,))
    weight = finite(state["weight"])
    if weight.ndim != 2 or weight.shape[0] != width:
        raise ValueError("Frozen RFF projection shape differs")
    d = weight.shape[1]
    offset = finite(state["offset"], (d,))
    rm, rs = finite(state["rff_mean"], (d,)), finite(state["rff_scale"], (d,))
    if np.any(scale <= 0) or np.any(rs <= 0):
        raise ValueError("Frozen RFF scales must be positive")
    z = np.empty((n, d))
    for start in range(0, n, 128):
        x = (finite(feature[start : start + 128]) - mean) / scale
        phi = np.sqrt(2 / d) * np.cos(multiply(x, weight) + offset)
        z[start : start + len(x)] = (phi - rm) / rs
    return finite(z)


def equal_room_nje(prediction, target, rooms):
    prediction, target = finite(prediction), finite(target)
    values = []
    for room in sorted(set(rooms)):
        mask = rooms == room
        metrics = pose_metrics(
            prediction[mask], target[mask], np.ones((mask.sum(), 17), bool)
        )
        value = metrics["normalized_joint_error"]
        if value is None or not np.isfinite(value) or metrics["excluded_frames"]:
            raise ValueError("Invalid source-validation pose metric")
        values.append(value)
    return float(np.mean(values))


def predict_root(z, state):
    z = finite(z)
    xm = finite(state["x_mean"], (z.shape[1],))
    mean = finite(state["root_mean"], (2,))
    weights = finite(state["weights"], (len(ALPHAS), z.shape[1], 2))
    selected_array = real_numeric(state["selected"])
    if selected_array.shape != () or selected_array.dtype.kind not in "iu":
        raise ValueError("Frozen root selection must be an integer scalar")
    selected = int(selected_array)
    beta = float(finite(state["beta"], ()))
    if selected not in range(len(ALPHAS)) or beta not in BETAS:
        raise ValueError("Frozen root selection differs")
    return mean + beta * multiply(z - xm, weights[selected])


def fit_root(z, body, target, train, validation, rooms):
    n, d = z.shape
    z, body = finite(z), finite(body, (n, 17, 2))
    if (
        train.dtype != bool
        or validation.dtype != bool
        or train.shape != (n,)
        or validation.shape != (n,)
        or not train.any()
        or not validation.any()
        or np.any(train & validation)
    ):
        raise ValueError("Disjoint nonempty source training and validation required")
    # Only source TRAIN/VAL targets enter arithmetic. Held annotations may be
    # poisoned or absent at this actual statistical-fit boundary.
    target = real_numeric(target)
    if target.shape != body.shape or not np.isfinite(target[train | validation]).all():
        raise ValueError("Finite source target coordinates required")
    root = finite(target[train])[:, [11, 12]].mean(1)
    xm, ym = z[train].mean(0), root.mean(0)
    x, y = z[train] - xm, root - ym
    covariance, right = multiply(x.T, x), multiply(x.T, y)
    eigen, vectors = eigh(covariance, check_finite=True)
    projected = multiply(vectors.T, right)
    weights, grid = [], []
    for alpha in ALPHAS:
        weight = multiply(vectors, projected / (eigen[:, None] + train.sum() * alpha))
        residual = (
            multiply(covariance + train.sum() * alpha * np.eye(d), weight) - right
        )
        relative = float(np.linalg.norm(residual) / max(np.linalg.norm(right), 1e-12))
        if not np.isfinite(relative) or relative > 1e-6:
            raise ValueError("Root normal equation failed")
        weights.append(weight)
        variation = multiply(z[validation] - xm, weight)
        for beta in BETAS:
            prediction = body[validation] + (ym + beta * variation)[:, None]
            grid.append(
                dict(
                    alpha=alpha,
                    beta=beta,
                    source_validation_nje=equal_room_nje(
                        prediction, target[validation], rooms[validation]
                    ),
                    normal_equation_relative_residual=relative,
                )
            )
    choice = min(
        grid, key=lambda x: (x["source_validation_nje"], x["beta"], -x["alpha"])
    )
    state = dict(
        x_mean=xm,
        root_mean=ym,
        weights=np.stack(weights),
        selected=np.asarray(ALPHAS.index(choice["alpha"])),
        beta=np.asarray(choice["beta"]),
    )
    return state, dict(choice=choice, grid=grid)


def fit_arrays(designs, bodies, target, data, train, validation, held):
    """Actual source-only fit boundary used by the disk-backed experiment."""
    roles_valid(data, train, validation, held)
    target = real_numeric(target)
    n = len(train)
    if not designs or set(designs) != set(bodies):
        raise ValueError("Matched frozen design/body arms required")
    states, choices, predictions = {}, {}, {}
    for arm in designs:
        body = finite(bodies[arm], (n, 17, 2))
        midpoint = float(np.max(np.abs(body[:, [11, 12]].mean(1))))
        if midpoint > 1e-6:
            raise ValueError(
                "Frozen body hip midpoint is not zero; no silent recentering"
            )
        states[arm], choices[arm] = fit_root(
            designs[arm], body, target, train, validation, data["room"]
        )
        choices[arm]["maximum_body_hip_midpoint"] = midpoint
        predictions[arm] = body + predict_root(designs[arm], states[arm])[:, None]
    source_mean = target[train].mean(0)
    predictions["source_mean"] = np.broadcast_to(source_mean, (n, 17, 2)).copy()
    clock = np.empty((n, 17, 2))
    for end in np.unique(data["end"]):
        source = train & (data["end"] == end)
        if not source.any():
            raise ValueError("Endpoint clock lacks a source TRAIN endpoint")
        clock[data["end"] == end] = target[source].mean(0)
    predictions["source_endpoint_clock"] = clock
    return states, choices, predictions


def source_contract(source, cache, raw, expected):
    if digest(source / "source-ready.json") != expected:
        raise ValueError("Source-ready identity differs")
    ready = json.loads((source / "source-ready.json").read_text())
    required = {
        "registration.json",
        "roles.npz",
        "targets-and-metadata.npz",
        "frozen-source-selections.json",
        "raw_current-rff.npz",
        "raw_current-readout.npz",
    }
    required.update(
        arm + suffix
        for arm in ENCODERS
        for suffix in ("-embedding.npy", "-final.pt", "-rff.npz", "-readout.npz")
    )
    if (
        ready["kind"] != "masked-csi-source-only-ready"
        or ready["held_metrics_computed"] is not False
        or not required <= set(ready["artifacts_sha256"])
    ):
        raise ValueError("Expected complete source-only encoder/map/body seal")
    verify_hashes(source, ready["artifacts_sha256"])
    reg = json.loads((source / "registration.json").read_text())
    verify_hashes(ROOT, reg["source_sha256"])
    if (
        reg["config"]["held_room"] != "E02"
        or reg["cache_receipt_sha256"] != digest(cache / "receipt.json")
        or reg["raw_receipt_sha256"] != digest(raw / "receipt.json")
        or digest(source / "targets-and-metadata.npz")
        != digest(cache / "targets-and-metadata.npz")
    ):
        raise ValueError("Source/raw/cache identities differ")
    return reg


def inputs(source, cache, raw, expected):
    source_contract(source, cache, raw, expected)
    receipt, data, cached = load_inputs(cache)
    raw_receipt = json.loads((raw / "receipt.json").read_text())
    if receipt["raw_receipt_sha256"] != digest(raw / "receipt.json"):
        raise ValueError("Raw/covariance target lineage differs")
    verify_hashes(
        raw, {name + ".npy": raw_receipt["arrays"][name] for name in ("pose", "valid")}
    )
    raw_pose = np.load(raw / "pose.npy", mmap_mode="r", allow_pickle=False)
    raw_valid = np.load(raw / "valid.npy", mmap_mode="r", allow_pickle=False)
    target, indices = join_full_targets(
        raw_pose, raw_valid, raw_receipt["records"], data
    )
    train, validation, held = fold_masks(data["room"], data["person"], "E02")
    roles_valid(data, train, validation, held)
    with np.load(source / "roles.npz", allow_pickle=False) as saved:
        for name, mask in (
            ("train", train),
            ("validation", validation),
            ("held", held),
        ):
            np.testing.assert_array_equal(saved[name], mask)
    return data, target, indices, train, validation, held, cached


def fit(source, cache, raw, out, expected):
    if out.exists():
        raise FileExistsError(out)
    if (source / "results.json").exists() or (
        source / "query-permutation.npy"
    ).exists():
        raise ValueError("Freeze full-image source choices before held-body scoring")
    data, target, indices, train, validation, held, cached = inputs(
        source, cache, raw, expected
    )
    designs, bodies = {}, {}
    for arm in ARMS:
        if arm == "raw_current":
            feature = hermitian_coordinates(
                current_covariances(cached["untreated_covariance"])
            ).reshape(len(train), -1)
        else:
            feature = np.load(
                source / (arm + "-embedding.npy"), mmap_mode="r", allow_pickle=False
            )
        with np.load(source / (arm + "-rff.npz"), allow_pickle=False) as mapping:
            designs[arm] = apply_rff(feature, mapping)
        if designs[arm].shape != (len(train), 512):
            raise ValueError("Expected the original frozen RFF512 source design")
        with np.load(source / (arm + "-readout.npz"), allow_pickle=False) as state:
            bodies[arm] = state["prediction"]
    # No target query metric occurs in fit_arrays or this artifact-writing phase.
    states, choices, predictions = fit_arrays(
        designs, bodies, target, data, train, validation, held
    )
    out.mkdir(parents=True)
    sources = [
        Path(__file__),
        ROOT / "tests/test_masked_fullpose_probe.py",
        ROOT / "ghostmesh/metrics.py",
        ROOT / "scripts/domain_background_probe.py",
        ROOT / "scripts/domain_covariance_cached_probe.py",
        ROOT / "scripts/domain_covariance_full_cache.py",
        ROOT / "scripts/domain_covariance_source_probe.py",
        ROOT / "ghostmesh/reference_covariance.py",
        ROOT / "ghostmesh/covariance_dynamics.py",
    ]
    reg = dict(
        config=CONFIG,
        runtime=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            scipy=SCIPY_VERSION,
            device="CPU source-only root readout; encoder artifacts remain unchanged",
        ),
        source_ready_sha256=expected,
        raw_receipt_sha256=digest(raw / "receipt.json"),
        cache_receipt_sha256=digest(cache / "receipt.json"),
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
        counts={
            name: int(mask.sum())
            for name, mask in (
                ("train", train),
                ("validation", validation),
                ("held", held),
            )
        },
        held_metrics_computed=False,
    )
    save_json(out / "registration.json", reg)
    for path in sources:
        destination = out / "executed-source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    save_npz(out / "roles.npz", train=train, validation=validation, held=held)
    save_npz(
        out / "canonical-metadata.npz",
        **{key: data[key] for key in ("record", "person", "action", "room", "end")},
    )
    save_array(out / "full-targets.npy", target)
    save_array(out / "raw-frame-indices.npy", indices)
    save_json(out / "frozen-fullpose-selections.json", choices)
    for arm, state in states.items():
        save_npz(out / (arm + "-root-state.npz"), **state)
    for arm, prediction in predictions.items():
        save_array(out / (arm + "-full-prediction.npy"), prediction)
    save_json(
        out / "fullpose-source-ready.json",
        dict(
            kind="masked-csi-fullimage-source-ready",
            held_metrics_computed=False,
            source_ready_sha256=expected,
            raw_receipt_sha256=reg["raw_receipt_sha256"],
            cache_receipt_sha256=reg["cache_receipt_sha256"],
            artifacts_sha256={
                str(p.relative_to(out)): digest(p)
                for p in sorted(out.rglob("*"))
                if p.is_file()
            },
        ),
    )
    return out


def metric(prediction, target, records):
    prediction, target = finite(prediction), finite(target)
    result = pose_metrics(prediction, target, np.ones(target.shape[:2], bool))
    result.pop("per_frame_nje")
    result["record_centered_full_image_r2"] = centered_r2(prediction, target, records)[
        "r2"
    ]
    p_root, y_root = (
        prediction[:, [11, 12]].mean(1, keepdims=True),
        target[:, [11, 12]].mean(1, keepdims=True),
    )
    result["record_centered_body_r2"] = centered_r2(
        prediction - p_root, target - y_root, records
    )["r2"]
    result["record_centered_root_r2"] = centered_r2(p_root, y_root, records)["r2"]
    return result


def scores(prediction, target, data, ids, permutation):
    records = data["record"][ids]
    result = dict(
        aligned=metric(prediction, target[ids], records),
        shuffled=metric(prediction[permutation], target[ids], records),
    )
    for key in ("person", "action"):
        groups = data[key][ids]
        result["per_" + key] = {
            str(name): metric(
                prediction[groups == name],
                target[ids][groups == name],
                records[groups == name],
            )
            for name in sorted(set(groups))
        }
    return result


def frozen_contract(out, source, cache, raw, expected):
    if digest(out / "fullpose-source-ready.json") != expected:
        raise ValueError("Full-image source-ready identity differs")
    ready = json.loads((out / "fullpose-source-ready.json").read_text())
    required = {
        "registration.json",
        "roles.npz",
        "canonical-metadata.npz",
        "full-targets.npy",
        "raw-frame-indices.npy",
        "frozen-fullpose-selections.json",
    }
    required.update(arm + "-root-state.npz" for arm in ARMS)
    required.update(
        arm + "-full-prediction.npy"
        for arm in (*ARMS, "source_mean", "source_endpoint_clock")
    )
    if (
        ready["kind"] != "masked-csi-fullimage-source-ready"
        or ready["held_metrics_computed"] is not False
        or not required <= set(ready["artifacts_sha256"])
    ):
        raise ValueError("Incomplete full-image source-ready seal")
    verify_hashes(out, ready["artifacts_sha256"])
    reg = json.loads((out / "registration.json").read_text())
    verify_hashes(ROOT, reg["source_sha256"])
    source_contract(source, cache, raw, reg["source_ready_sha256"])
    if (
        ready["source_ready_sha256"] != reg["source_ready_sha256"]
        or ready["cache_receipt_sha256"] != digest(cache / "receipt.json")
        or ready["raw_receipt_sha256"] != digest(raw / "receipt.json")
    ):
        raise ValueError("Frozen full-image lineage differs")
    return reg


def score_frozen(out, source, cache, raw, expected):
    if (out / "results.json").exists() or (out / "query-permutation.npy").exists():
        raise FileExistsError("Full-image scoring already started")
    reg = frozen_contract(out, source, cache, raw, expected)
    with np.load(out / "canonical-metadata.npz", allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    with np.load(out / "roles.npz", allow_pickle=False) as saved:
        train, validation, held = (
            saved[name] for name in ("train", "validation", "held")
        )
    roles_valid(data, train, validation, held)
    target = finite(
        np.load(out / "full-targets.npy", allow_pickle=False), (len(train), 17, 2)
    )
    ids = np.flatnonzero(held)
    permutation = np.arange(len(ids))
    rng = np.random.default_rng(90210)
    for name in sorted(set(data["record"][ids])):
        block = np.flatnonzero(data["record"][ids] == name)
        permutation[block] = rng.permutation(block)
    save_array(out / "query-permutation.npy", permutation)
    results = {}
    for arm in (*ARMS, "source_mean", "source_endpoint_clock"):
        prediction = finite(
            np.load(out / (arm + "-full-prediction.npy"), allow_pickle=False),
            target.shape,
        )
        results[arm] = scores(prediction[ids], target, data, ids, permutation)
    p = results["masked_temporal"]["aligned"]
    controls = [
        results[name]["aligned"]
        for name in (
            "source_mean",
            "source_endpoint_clock",
            "random_frozen",
            "supervised",
            "raw_current",
        )
    ]
    gate = dict(
        nje_beats_matched_controls=all(
            p["normalized_joint_error"] < c["normalized_joint_error"] for c in controls
        ),
        pck_not_worse=all(
            p["pck_curve"][k] >= c["pck_curve"][k]
            for c in controls
            for k in ("0.05", "0.2")
        ),
        positive_full_motion=p["record_centered_full_image_r2"] is not None
        and p["record_centered_full_image_r2"] > 0,
        positive_body_motion=p["record_centered_body_r2"] is not None
        and p["record_centered_body_r2"] > 0,
        shuffle_worse=p["normalized_joint_error"]
        < results["masked_temporal"]["shuffled"]["normalized_joint_error"],
    )
    save_json(
        out / "results.json",
        dict(
            registration=reg,
            fullpose_source_ready_sha256=expected,
            methods=results,
            gate=gate,
            passes=all(gate.values()),
            scope="Full-image2D E02development; root/body motion are separately reported; no3D/final confirmation",
        ),
    )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-ready-sha256")
    parser.add_argument("--score-frozen-sha256")
    args = parser.parse_args()
    if args.score_frozen_sha256:
        score_frozen(
            args.out, args.source, args.cache, args.raw, args.score_frozen_sha256
        )
    elif args.source_ready_sha256:
        fit(args.source, args.cache, args.raw, args.out, args.source_ready_sha256)
    else:
        parser.error("Explicit --source-ready-sha256 or --score-frozen-sha256 required")


if __name__ == "__main__":
    main()
