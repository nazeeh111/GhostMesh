"""Fixed-room CSI reference experiment; enrollment people and queries differ.

This is a calibrated source-development mode. Target reference joints are used
only by the explicit mean controls, never by the primary physics descriptors.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.reference_covariance import (  # noqa: E402
    current_covariances,
    fit_reference,
    reference_features,
)
from ghostmesh.metrics import pose_metrics  # noqa: E402
from scripts.domain_covariance_cached_probe import load_inputs  # noqa: E402
from scripts.domain_covariance_full_cache import verify_hashes  # noqa: E402
from scripts.domain_covariance_source_probe import (  # noqa: E402
    digest,
    feature_stats,
    fold_masks,
    multiply,
    rff_transform,
    save,
)
from scripts.domain_background_probe import centered_r2  # noqa: E402

REF_ACTIONS = ("A03", "A07", "A12")
REF_ENDS = tuple(end for first in range(39, 256, 24) for end in (first, first + 8))
CONFIG = dict(
    seed=2027,
    reference_actions=REF_ACTIONS,
    reference_endpoints=REF_ENDS,
    reference_budget="2 people x3 fixed actions x10 pairs x2 endpoints=120 consented poses;120 endpoint covariances, not960 context matrices",
    anchors="Normalize EACH reference covariance by determinant^(1/3) BEFORE averaging; global120 or3 known protocol groups of40",
    volume="(logdet query C - mean120 reference logdet C)/3; cancels fixed G but NOT independent positive frame gains; one global volume channel/bin",
    arms=[
        "global_shape",
        "global_shape_volume",
        "semantic3_shape",
        "semantic3_shape_volume",
        "raw_current_control",
    ],
    dimensions=dict(
        global_shape=456,
        global_shape_volume=608,
        semantic3_shape=1368,
        semantic3_shape_volume=1520,
        raw_current_control=1368,
    ),
    primary="semantic3_shape",
    gate="Primarysemantic3_shape NJE lower than allfive declared controls including matchedrawcurrent in eachheldroom; PCK05/PCK20 no worse than BOTH source-validation-selected meanmixture and rawcurrent in eachroom; positive record-centeredrelativeposeR2 and betteraligned than fixedwithinrecordshuffled NJE eachroom. Development only, no finalefficacyclaim",
    rff_features=512,
    projection_dimensions=None,
    context_frames=8,
    conservative_availability_context=40,
    alphas=[0.001, 0.01, 0.1, 1.0, 10.0],
    source_ref_mixture_weights=[0.0, 0.25, 0.5, 0.75, 1.0],
    source_selection="Equal-room source-validation current-articulation NJE; first alpha/mixture tie wins; no held query labels",
    normalization="Source-training column mean/SD then RFF column mean/SD; both floors.001. Median128-source-training-sample RBF bandwidth. Same RFF512 and n*alpha regularizer for each arm",
    inference="Physics descriptors consume current CSI and frozen room-reference CSI only; no query actionID, joints, query mean or initial pose",
    scope="Calibrated SOURCE DEVELOPMENT E01-E03/all27actions. Target reference people1/2; queryother8. Root-relative2D articulation only; no E04/globalposition/3D/initialconfiguration guarantee",
    limits="One or several generalized spectra remain noninjective. Shared per-bin receiver mixing is a hypothesis, not proof about actual rooms. Shape is frame-gain invariant; volume is not",
)


def reference_people(room):
    if room not in ("E01", "E02", "E03"):
        raise ValueError("Only development rooms E01-E03 are allowed")
    first = (int(room[-2:]) - 1) * 10 + 1
    return (f"S{first:02}", f"S{first + 1:02}")


def reference_indices(data, room):
    selected = (
        (data["room"] == room)
        & np.isin(data["person"], reference_people(room))
        & np.isin(data["action"], REF_ACTIONS)
        & np.isin(data["end"], REF_ENDS)
    )
    indices = np.flatnonzero(selected)
    if (
        len(indices) != 120
        or len(set(zip(data["record"][indices], data["end"][indices]))) != 120
    ):
        raise ValueError("Required120 unique room reference endpoints are unavailable")
    if any(
        np.count_nonzero(data["action"][indices] == action) != 40
        for action in REF_ACTIONS
    ):
        raise ValueError("Each protocol action must supply40 reference endpoints")
    return indices


def build_room_references(untreated, data):
    """This function deliberately never reads data['pose'] or data['delta']."""
    result = {}
    for room in ("E01", "E02", "E03"):
        indices = reference_indices(data, room)
        endpoint = current_covariances(untreated[indices])[:, -1]
        result[room] = dict(
            indices=indices,
            global_reference=fit_reference(endpoint),
            semantic_reference=fit_reference(
                endpoint, data["action"][indices], group_order=REF_ACTIONS
            ),
        )
    return result


def feature_block(current, room_reference):
    global_volume = reference_features(
        current, room_reference["global_reference"], include_volume=True
    )
    semantic_shape = reference_features(current, room_reference["semantic_reference"])
    from ghostmesh.covariance_dynamics import hermitian_coordinates

    return dict(
        global_shape=global_volume[..., :3].reshape(len(current), -1),
        global_shape_volume=global_volume.reshape(len(current), -1),
        semantic3_shape=semantic_shape.reshape(len(current), -1),
        semantic3_shape_volume=np.concatenate(
            (semantic_shape, global_volume[..., -1:]), axis=-1
        ).reshape(len(current), -1),
        raw_current_control=hermitian_coordinates(current).reshape(len(current), -1),
    )


def equal_room_nje(prediction, target, rooms):
    return float(
        np.mean(
            [
                pose_metrics(
                    prediction[rooms == room],
                    target[rooms == room],
                    np.ones(target[rooms == room].shape[:2], bool),
                )["normalized_joint_error"]
                for room in sorted(set(rooms))
            ]
        )
    )


def fit_readout(z, target, train, validation, rooms):
    if target.shape != (len(z), 17, 2) or np.any(train & validation):
        raise ValueError("Expected joint targets and disjoint source roles")
    xm, ym = z[train].mean(0), target[train].reshape(train.sum(), 34).mean(0)
    x, y = z[train] - xm, target[train].reshape(train.sum(), 34) - ym
    matrix, right = multiply(x.T, x), multiply(x.T, y)
    eigen, vectors = eigh(matrix, check_finite=True)
    projected = multiply(vectors.T, right)
    weights, grid = [], []
    for alpha in CONFIG["alphas"]:
        weight = multiply(vectors, projected / (eigen[:, None] + train.sum() * alpha))
        residual = (
            multiply(matrix + train.sum() * alpha * np.eye(z.shape[1]), weight) - right
        )
        relative = float(np.linalg.norm(residual) / max(np.linalg.norm(right), 1e-12))
        if not np.isfinite(relative) or relative > 1e-6:
            raise ValueError("Source normal equation failed")
        prediction = (multiply(z[validation] - xm, weight) + ym).reshape(-1, 17, 2)
        grid.append(
            dict(
                alpha=alpha,
                source_validation_nje=equal_room_nje(
                    prediction, target[validation], rooms[validation]
                ),
                normal_equation_relative_residual=relative,
            )
        )
        weights.append(weight)
    chosen = min(range(len(grid)), key=lambda i: grid[i]["source_validation_nje"])
    return dict(
        x_mean=xm, y_mean=ym, weights=np.stack(weights), selected=np.asarray(chosen)
    ), grid


def predict(z, state):
    return (
        multiply(z - state["x_mean"], state["weights"][int(state["selected"])])
        + state["y_mean"]
    ).reshape(-1, 17, 2)


def means(data, train, validation):
    source = data["pose"][train].mean(0)
    reference = {
        room: data["pose"][reference_indices(data, room)].mean(0)
        for room in ("E01", "E02", "E03")
    }
    candidates = []
    for beta in CONFIG["source_ref_mixture_weights"]:
        prediction = np.stack(
            [
                (1 - beta) * source + beta * reference[room]
                for room in data["room"][validation]
            ]
        )
        candidates.append(
            dict(
                reference_weight=beta,
                source_validation_nje=equal_room_nje(
                    prediction, data["pose"][validation], data["room"][validation]
                ),
            )
        )
    selected = min(
        range(len(candidates)), key=lambda i: candidates[i]["source_validation_nje"]
    )
    return source, reference, CONFIG["source_ref_mixture_weights"][selected], candidates


def articulation_score(prediction, target, record):
    metrics = pose_metrics(prediction, target, np.ones(target.shape[:2], bool))
    metrics.pop("per_frame_nje")
    metrics["record_centered_articulation_r2"] = centered_r2(
        prediction, target, record
    )["r2"]
    return metrics


def query_indices(data, room):
    indices = np.flatnonzero(
        (data["room"] == room) & ~np.isin(data["person"], reference_people(room))
    )
    if len(set(data["person"][indices])) != 8:
        raise ValueError("Expected eight query people different from reference people")
    return indices


def preflight(untreated, data):
    minimum, maximum = float("inf"), 0.0
    for start in range(0, len(untreated), 512):
        current = current_covariances(untreated[start : start + 512])
        eigen = np.linalg.eigvalsh(current)
        if np.any(eigen[..., 0] <= 0):
            raise ValueError(
                "Decoded float32 covariance is not SPD; no repair or silent row removal"
            )
        ratio = eigen[..., -1] / eigen[..., 0]
        if np.any(ratio > 1e10):
            raise ValueError("Decoded covariance exceeds existing conditioning guard")
        minimum = min(minimum, float(eigen[..., 0].min()))
        maximum = max(maximum, float(ratio.max()))
    return dict(
        rows=len(untreated),
        minimum_eigenvalue=minimum,
        maximum_condition=maximum,
        additional_rejected_rows=0,
        reference_counts={
            room: len(reference_indices(data, room)) for room in ("E01", "E02", "E03")
        },
    )


def frozen_gate(results):
    decisions = []
    control_names = (
        "source_mean",
        "reference120_mean",
        "source_validation_mean_mixture",
        "frozen_temporal_zero_shot",
        "raw_current_control",
    )
    for room in ("E01", "E02", "E03"):
        methods = {r["method"]: r for r in results if r["room"] == room}
        primary = methods[CONFIG["primary"]]
        aligned = primary["aligned"]
        nje = all(
            aligned["normalized_joint_error"]
            < methods[name]["aligned"]["normalized_joint_error"]
            for name in control_names
        )
        pck = all(
            aligned["pck_curve"][threshold]
            >= methods[name]["aligned"]["pck_curve"][threshold]
            for threshold in ("0.05", "0.2")
            for name in ("source_validation_mean_mixture", "raw_current_control")
        )
        motion = (
            aligned["record_centered_articulation_r2"] is not None
            and aligned["record_centered_articulation_r2"] > 0
            and aligned["normalized_joint_error"]
            < primary["shuffled"]["normalized_joint_error"]
        )
        decisions.append(
            dict(
                room=room,
                beats_all_five_controls_nje=nje,
                pck_not_worse_than_mixture_and_raw=pck,
                positive_aligned_articulation_motion=motion,
                passed=bool(nje and pck and motion),
            )
        )
    return dict(passed=all(r["passed"] for r in decisions), per_room=decisions)


def scores(prediction, data, indices, permutation):
    target, record = data["pose"][indices], data["record"][indices]
    result = dict(
        aligned=articulation_score(prediction, target, record),
        shuffled=articulation_score(prediction[permutation], target, record),
    )
    for field in ("person", "action"):
        values = data[field][indices]
        result["per_" + field] = {
            key: articulation_score(prediction[mask], target[mask], record[mask])
            for key in sorted(set(values))
            for mask in [values == key]
        }
    return result


def run(cache, old_probe, out):
    if out.exists():
        raise FileExistsError(out)
    started = time.monotonic()
    receipt, data, old_features = load_inputs(cache)
    proof = json.loads((old_probe / "verification.json").read_text())
    old_reg = json.loads((old_probe / "registration.json").read_text())
    if proof["total_records_feature_verified"] != 810 or old_reg[
        "feature_cache_receipt_sha256"
    ] != digest(cache / "receipt.json"):
        raise ValueError("Prior full-cache proof does not match inputs")
    paths = [
        Path(__file__),
        ROOT / "ghostmesh/reference_covariance.py",
        ROOT / "ghostmesh/covariance_dynamics.py",
        ROOT / "scripts/domain_covariance_cached_probe.py",
        ROOT / "scripts/domain_covariance_full_cache.py",
        ROOT / "scripts/domain_covariance_source_probe.py",
        ROOT / "scripts/domain_background_probe.py",
        ROOT / "ghostmesh/metrics.py",
    ]
    sources = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    verify_hashes(
        old_probe, json.loads((old_probe / "artifact_manifest.json").read_text())
    )
    old_models = {
        room + "-congruence_dynamics-model-prediction.npz": digest(
            old_probe / (room + "-congruence_dynamics-model-prediction.npz")
        )
        for room in ("E01", "E02", "E03")
    }
    out.mkdir(parents=True)
    registration = dict(
        config=CONFIG,
        source_sha256=sources,
        feature_cache_receipt_sha256=digest(cache / "receipt.json"),
        full_proof_sha256=digest(old_probe / "verification.json"),
        frozen_zero_shot_model_sha256=old_models,
        lineage="Fixed-reference mechanism chosen after failed temporal/state-observer development; not an untouched final test",
    )
    save(out / "registration.json", registration)
    for path in paths:
        destination = out / "executed-source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    reference_ids = {}
    roles, indices_archive = {}, {}
    for room in ("E01", "E02", "E03"):
        ix = reference_indices(data, room)
        reference_ids[room] = [
            dict(record=str(data["record"][i]), end=int(data["end"][i])) for i in ix
        ]
        train, validation, _ = fold_masks(data["room"], data["person"], room)
        query = query_indices(data, room)
        for name, indices in (
            ("train", np.flatnonzero(train)),
            ("validation", np.flatnonzero(validation)),
            ("reference", ix),
            ("query", query),
        ):
            indices_archive[room + "_" + name] = indices
        roles[room] = {
            name: dict(
                rows=len(indices),
                people=sorted(set(data["person"][indices])),
                indices_sha256=sha256(indices.tobytes()).hexdigest(),
            )
            for name, indices in (
                ("train", np.flatnonzero(train)),
                ("validation", np.flatnonzero(validation)),
                ("reference", ix),
                ("query", query),
            )
        }
    save(out / "reference-identities.json", reference_ids)
    save(out / "cohort-roles.json", roles)
    np.savez(out / "cohort-indices.npz", **indices_archive)
    evidence = out / "input-evidence"
    evidence.mkdir()
    for source, destination in (
        (cache / "receipt.json", evidence / "feature-cache-receipt.json"),
        (
            cache / "input-evidence/raw-receipt.json",
            evidence / "raw-source-receipt.json",
        ),
        (old_probe / "verification.json", evidence / "prior-full-verification.json"),
    ):
        shutil.copyfile(source, destination)
    # Freeze data-independent random bases BEFORE any real CSI anchor is fitted.
    for arm in CONFIG["arms"]:
        rng = np.random.default_rng(CONFIG["seed"])
        basis = rng.normal(size=(CONFIG["dimensions"][arm], 512))
        offset = rng.uniform(0, 2 * np.pi, 512)
        np.savez(
            out / ("initial-base-" + arm + ".npz"), unscaled_weight=basis, offset=offset
        )
    before = time.monotonic()
    quality = preflight(old_features["untreated_covariance"], data)
    quality["seconds"] = time.monotonic() - before
    save(out / "preflight.json", quality)
    save(
        out / "before-calibration.json",
        {
            str(p.relative_to(out)): digest(p)
            for p in sorted(out.rglob("*"))
            if p.is_file()
        },
    )
    before = time.monotonic()
    references = build_room_references(old_features["untreated_covariance"], data)
    save(
        out / "enrollment-burden.json",
        dict(
            reference_generation_seconds=time.monotonic() - before,
            reference_poses_per_room=120,
            recordings_per_room=6,
            reference_endpoints_per_record=20,
            nominal_frame_hz=10,
            nominal_last_reference_seconds_per_record=26.3,
            nominal_cumulative_recorded_prefix_seconds=158.4,
            limit="Recorded-prefix duration is derived from nominal frame indices, excludes human pauses/setup and is not measured live enrollment time. No30second enrollment promise.",
        ),
    )
    for room, ref in references.items():
        for kind in ("global_reference", "semantic_reference"):
            model = ref[kind]
            np.savez(
                out / (room + "-" + kind + ".npz"),
                anchors=model.anchors,
                mean_log_volume=model.mean_log_volume,
                group_names=np.asarray(model.group_names),
                group_counts=np.asarray(model.group_counts),
            )
    n = len(data["pose"])
    features = {
        arm: np.lib.format.open_memmap(
            out / (arm + "-features.npy"),
            mode="w+",
            dtype="float32",
            shape=(n, CONFIG["dimensions"][arm]),
        )
        for arm in CONFIG["arms"]
    }
    for start in range(0, n, 128):
        end = min(n, start + 128)
        current = current_covariances(old_features["untreated_covariance"][start:end])
        rooms = data["room"][start:end]
        for room in sorted(set(rooms)):
            mask = rooms == room
            values = feature_block(current[mask], references[room])
            for arm in features:
                features[arm][start + np.flatnonzero(mask)] = values[arm]
        if start % 2048 == 0:
            print(
                json.dumps(
                    dict(feature_rows=end, total=n, seconds=time.monotonic() - started)
                ),
                flush=True,
            )
    for value in features.values():
        value.flush()
    results = []
    for room in ("E01", "E02", "E03"):
        train, validation, held = fold_masks(data["room"], data["person"], room)
        indices = query_indices(data, room)
        common = np.zeros(n, bool)
        common[indices] = True
        if (
            set(data["person"][indices]) & set(reference_people(room))
            or len(set(data["person"][indices])) != 8
        ):
            raise ValueError("Calibration/query people overlap or missing")
        permutation = np.arange(len(indices))
        rng = np.random.default_rng(90210)
        for record in sorted(set(data["record"][indices])):
            pick = np.flatnonzero(data["record"][indices] == record)
            permutation[pick] = rng.permutation(pick)
        np.save(out / (room + "-permutation.npy"), permutation, allow_pickle=False)
        source, reference, beta, grid = means(data, train, validation)
        controls = dict(
            source_mean=np.broadcast_to(source, (len(indices), 17, 2)),
            reference120_mean=np.broadcast_to(reference[room], (len(indices), 17, 2)),
            source_validation_mean_mixture=np.broadcast_to(
                (1 - beta) * source + beta * reference[room], (len(indices), 17, 2)
            ),
        )
        with np.load(
            old_probe / (room + "-congruence_dynamics-model-prediction.npz"),
            allow_pickle=False,
        ) as old:
            keep = ~np.isin(data["person"][old["held_indices"]], reference_people(room))
            if not np.array_equal(old["held_indices"][keep], indices):
                raise ValueError("Frozen zero-shot comparator rows differ")
            controls["frozen_temporal_zero_shot"] = old["pose"][keep]
        for name, p in controls.items():
            np.savez(
                out / (room + "-" + name + "-predictions.npz"),
                prediction=p,
                indices=indices,
            )
            results.append(
                dict(
                    room=room,
                    method=name,
                    calibration_gt_used=name
                    in ("reference120_mean", "source_validation_mean_mixture"),
                    mixture_weight=beta
                    if name == "source_validation_mean_mixture"
                    else None,
                    source_validation_grid=grid
                    if name == "source_validation_mean_mixture"
                    else None,
                    **scores(p, data, indices, permutation),
                )
            )
        for arm in CONFIG["arms"]:
            z = rff_transform(
                features[arm], train, out / (room + "-" + arm + "-rff.npz")
            )
            state, grid = fit_readout(z, data["pose"], train, validation, data["room"])
            p = predict(z[common], state)
            np.savez(
                out / (room + "-" + arm + "-model.npz"),
                **state,
                prediction=p,
                indices=indices,
            )
            row = dict(
                room=room,
                method=arm,
                target_reference_gt_used=False,
                room_reference_csi_used=arm != "raw_current_control",
                source_validation_grid=grid,
                selected_alpha=CONFIG["alphas"][int(state["selected"])],
                **scores(p, data, indices, permutation),
            )
            results.append(row)
            print(
                json.dumps(
                    dict(
                        room=room,
                        method=arm,
                        alpha=row["selected_alpha"],
                        nje=row["aligned"]["normalized_joint_error"],
                        relative_r2=row["aligned"]["record_centered_articulation_r2"],
                    )
                ),
                flush=True,
            )
            save(out / "progress.json", dict(results=results))
    verify_hashes(cache, receipt["arrays_sha256"])
    verify_hashes(old_probe, old_models)
    if any(digest(ROOT / name) != expected for name, expected in sources.items()):
        raise ValueError("Frozen source changed")
    save(
        out / "results.json",
        dict(
            registration=registration,
            results=results,
            frozen_continuation_gate=frozen_gate(results),
            seconds=time.monotonic() - started,
        ),
    )
    save(
        out / "artifact_manifest.json",
        {
            str(p.relative_to(out)): digest(p)
            for p in sorted(out.rglob("*"))
            if p.is_file()
        },
    )


def reconstruct_rff(feature, train, path, initial_base):
    with np.load(path, allow_pickle=False) as saved:
        mapping = {key: saved[key] for key in saved.files}
    mean, scale = feature_stats(feature, train)
    if not np.array_equal(mean, mapping["feature_mean"]) or not np.array_equal(
        scale, mapping["feature_scale"]
    ):
        raise ValueError("Feature statistics differ from source-training calculation")
    rng = np.random.default_rng(CONFIG["seed"])
    sample_indices = rng.choice(np.flatnonzero(train), 128, replace=False)
    sample = (np.asarray(feature[sample_indices], dtype="float64") - mean) / scale
    distance = np.maximum(
        (sample * sample).sum(1)[:, None]
        + (sample * sample).sum(1)[None]
        - 2 * multiply(sample, sample.T),
        0,
    )
    width = float(np.median(distance[np.triu_indices(128, 1)]))
    if width != float(mapping["bandwidth_squared"]) or not np.array_equal(
        sample_indices, mapping["bandwidth_sample_indices"]
    ):
        raise ValueError("Bandwidth differs from source-training calculation")
    with np.load(initial_base, allow_pickle=False) as initial:
        if not np.array_equal(
            initial["unscaled_weight"] / np.sqrt(width), mapping["weight"]
        ) or not np.array_equal(initial["offset"], mapping["offset"]):
            raise ValueError("Random map differs from pre-calibration initialization")
    raw = np.empty((len(feature), 512))
    for start in range(0, len(feature), 128):
        x = (np.asarray(feature[start : start + 128], dtype="float64") - mean) / scale
        raw[start : start + len(x)] = np.sqrt(2 / 512) * np.cos(
            multiply(x, mapping["weight"]) + mapping["offset"]
        )
    if not np.array_equal(
        raw[train].mean(0), mapping["rff_mean"]
    ) or not np.array_equal(np.maximum(raw[train].std(0), 0.001), mapping["rff_scale"]):
        raise ValueError("RFF standardization differs from source-training calculation")
    return (raw - mapping["rff_mean"]) / mapping["rff_scale"]


def verify(cache, old_probe, out):
    if (out / "verification.json").exists():
        raise FileExistsError("Verification exists; preserve it")
    result = json.loads((out / "results.json").read_text())
    registration = result["registration"]
    verify_hashes(out, json.loads((out / "artifact_manifest.json").read_text()))
    verify_hashes(out, json.loads((out / "before-calibration.json").read_text()))
    for name, expected in registration["source_sha256"].items():
        if (
            digest(ROOT / name) != expected
            or digest(out / "executed-source" / name) != expected
        ):
            raise ValueError("Frozen executed source differs")
    receipt, data, old_features = load_inputs(cache)
    if digest(cache / "receipt.json") != registration["feature_cache_receipt_sha256"]:
        raise ValueError("Input cache receipt differs")
    verify_hashes(old_probe, registration["frozen_zero_shot_model_sha256"])
    references = build_room_references(old_features["untreated_covariance"], data)
    saved_ids = json.loads((out / "reference-identities.json").read_text())
    for room, ref in references.items():
        actual_ids = [
            dict(record=str(data["record"][i]), end=int(data["end"][i]))
            for i in ref["indices"]
        ]
        if actual_ids != saved_ids[room]:
            raise ValueError("Reference endpoint identities differ")
        for kind in ("global_reference", "semantic_reference"):
            model = ref[kind]
            with np.load(
                out / (room + "-" + kind + ".npz"), allow_pickle=False
            ) as saved:
                if (
                    not np.array_equal(model.anchors, saved["anchors"])
                    or not np.array_equal(
                        model.mean_log_volume, saved["mean_log_volume"]
                    )
                    or not np.array_equal(model.group_names, saved["group_names"])
                    or not np.array_equal(model.group_counts, saved["group_counts"])
                ):
                    raise ValueError("CSI-only fixed reference reconstruction differs")
    n = len(data["pose"])
    features = {
        arm: np.load(out / (arm + "-features.npy"), mmap_mode="r", allow_pickle=False)
        for arm in CONFIG["arms"]
    }
    for start in range(0, n, 128):
        end = min(n, start + 128)
        current = current_covariances(old_features["untreated_covariance"][start:end])
        rooms = data["room"][start:end]
        for room in sorted(set(rooms)):
            mask = rooms == room
            values = feature_block(current[mask], references[room])
            indices = start + np.flatnonzero(mask)
            for arm in CONFIG["arms"]:
                if not np.array_equal(
                    values[arm].astype("float32"), features[arm][indices]
                ):
                    raise ValueError(
                        "Current-state reference features differ from saved cache"
                    )
        if start % 4096 == 0:
            print(
                json.dumps(dict(reference_feature_rows_verified=end, total=n)),
                flush=True,
            )
    maximum = residual = 0.0
    models = 0
    for room in ("E01", "E02", "E03"):
        train, validation, _ = fold_masks(data["room"], data["person"], room)
        indices = query_indices(data, room)
        permutation = np.load(out / (room + "-permutation.npy"), allow_pickle=False)
        expected = np.arange(len(indices))
        rng = np.random.default_rng(90210)
        for record in sorted(set(data["record"][indices])):
            pick = np.flatnonzero(data["record"][indices] == record)
            expected[pick] = rng.permutation(pick)
        if not np.array_equal(permutation, expected):
            raise ValueError("Within-record shuffle differs")
        source, reference, beta, grid = means(data, train, validation)
        controls = dict(
            source_mean=np.broadcast_to(source, (len(indices), 17, 2)),
            reference120_mean=np.broadcast_to(reference[room], (len(indices), 17, 2)),
            source_validation_mean_mixture=np.broadcast_to(
                (1 - beta) * source + beta * reference[room], (len(indices), 17, 2)
            ),
        )
        with np.load(
            old_probe / (room + "-congruence_dynamics-model-prediction.npz"),
            allow_pickle=False,
        ) as old:
            keep = ~np.isin(data["person"][old["held_indices"]], reference_people(room))
            controls["frozen_temporal_zero_shot"] = old["pose"][keep]
            if not np.array_equal(old["held_indices"][keep], indices):
                raise ValueError("Old temporal baseline query identities differ")
        for name, p in controls.items():
            row = next(
                r
                for r in result["results"]
                if r["room"] == room and r["method"] == name
            )
            with np.load(
                out / (room + "-" + name + "-predictions.npz"), allow_pickle=False
            ) as saved:
                maximum = max(maximum, float(np.max(abs(p - saved["prediction"]))))
                if not np.array_equal(saved["indices"], indices):
                    raise ValueError("Control query identities differ")
            observed = scores(p, data, indices, permutation)
            if any(observed[key] != row[key] for key in observed):
                raise ValueError("Control metrics/strata/shuffle differ")
            if name == "source_validation_mean_mixture" and (
                grid != row["source_validation_grid"] or beta != row["mixture_weight"]
            ):
                raise ValueError(
                    "Mixture selection differs from source-validation choice"
                )
        for arm in CONFIG["arms"]:
            z = reconstruct_rff(
                features[arm],
                train,
                out / (room + "-" + arm + "-rff.npz"),
                out / ("initial-base-" + arm + ".npz"),
            )
            state, grid = fit_readout(z, data["pose"], train, validation, data["room"])
            row = next(
                r for r in result["results"] if r["room"] == room and r["method"] == arm
            )
            with np.load(
                out / (room + "-" + arm + "-model.npz"), allow_pickle=False
            ) as saved:
                for key in state:
                    maximum = max(maximum, float(np.max(abs(state[key] - saved[key]))))
                prediction = predict(z[indices], saved)
                maximum = max(
                    maximum, float(np.max(abs(prediction - saved["prediction"])))
                )
                if not np.array_equal(saved["indices"], indices):
                    raise ValueError("Physics query identities differ")
            observed = scores(prediction, data, indices, permutation)
            if (
                any(observed[key] != row[key] for key in observed)
                or grid != row["source_validation_grid"]
            ):
                raise ValueError("Physics metrics/strata/shuffle/selection differ")
            residual = max(
                residual, max(c["normal_equation_relative_residual"] for c in grid)
            )
            models += 1
    if (
        maximum != 0
        or residual > 1e-6
        or models != 15
        or frozen_gate(result["results"]) != result["frozen_continuation_gate"]
    ):
        raise ValueError("Reference experiment readback failed")
    save(
        out / "verification.json",
        dict(
            models_verified=models,
            normal_equations_verified=75,
            reference_room_sets_verified=3,
            reference_endpoint_ids_per_room=120,
            feature_windows_verified=n,
            maximum_saved_model_and_prediction_difference=maximum,
            maximum_normal_equation_relative_residual=residual,
            source_only_statistics_bandwidth_selection_and_baseline_mixture_verified=True,
            all_current_reference_features_and_person_disjoint_queries_verified=True,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--old-probe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(args.cache, args.old_probe, args.out)
    else:
        run(args.cache, args.old_probe, args.out)
