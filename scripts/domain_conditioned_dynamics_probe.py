"""SOURCE-only state-conditioned inverse for Wi-Fi embedding displacements.

No E02 numeric feature/pose values are read. This is a feasibility test of an
odd bilinear displacement map, not absolute pose or a validated room-transfer
model. Every statistic is fitted on source TRAIN pairs only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy import __version__ as SCIPY_VERSION
from scipy.linalg import eigh
from scipy.linalg.blas import dgemm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.domain_background_probe import digest  # noqa:E402
from scripts.domain_covariance_full_cache import verify_hashes  # noqa:E402
from scripts.domain_masked_motion_diagnostic import source_roles, stored_npz_array  # noqa:E402

EXPECTED_READY = "303922cdcb0e8007b25b0488c5cc14515487afb635cf60bc9b8b0b3f31786461"
ARMS = ("masked_temporal", "random_frozen", "supervised")
RANKS = (0, 2, 4, 8)
ALPHAS = (1e-5, 0.001, 0.01, 0.1, 1.0, 10.0)
REQUIRED_TRAINING_SOURCES = frozenset(
    {
        "scripts/domain_masked_pretrain_probe.py",
        "ghostmesh/masked_csi.py",
        "scripts/domain_covariance_cached_probe.py",
        "scripts/domain_covariance_source_probe.py",
        "scripts/domain_covariance_full_cache.py",
        "scripts/domain_reference_covariance_probe.py",
        "scripts/domain_background_probe.py",
        "ghostmesh/reference_covariance.py",
        "ghostmesh/covariance_dynamics.py",
        "ghostmesh/covariance_accelerated.py",
        "ghostmesh/causal_channel.py",
        "ghostmesh/metrics.py",
    }
)
CONFIG = dict(
    source_ready_sha256=EXPECTED_READY,
    arms=ARMS,
    ranks=RANKS,
    alphas=ALPHAS,
    lag_frames=8,
    pairs="Exact same-record retained predecessor t-8 only. Missing canonical ends are skipped, never bridged or filled",
    input="Current frame phi is last64 of each frozen eight-frame512D embedding; no global pose-head/map weights used",
    delta="phi_t-phi_past; TRAIN RMS floor.001, NO mean subtraction so zero RF change yields zero and swapping pair time reverses prediction",
    context="Symmetric midpoint(phi_t+phi_past)/2; TRAIN mean/SD floor.001 and TRAIN PCA64/sign-fixed basis; TRAIN PCscoreRMS floor.001",
    design="delta_scaled/sqrt64 concatenated with (context_PC/PCscale outer delta_scaled)/sqrt(64*r); no intercept. Rank0 is unconditional displacement",
    negative_control="Same-capacity context-shuffled designs use the SAME TRAIN map and fixed independent TRAIN/VAL pair permutations, preserving each role's context distribution. Offline diagnostic, not causal deployment",
    ridge="(Z.T Z+n*alpha*I)W=Z.T deltaHIP17x2; float64 checked SciPy BLAS products, no NumPy matmul or pseudoinverse",
    numerical_policy="Report any nonfinite/failed eigensolve, nonpositive regularized spectrum, condition overflow, solve failure or relative normal-equation residual>1e-6 as unavailable with NO validation score. Never synthesize coefficients or replace invalid scores. Input schema failures abort",
    selection="Primary natural-context rank/alpha maximizes equal-source-room VAL deltaR2. Ties prefer lower rank (rank0 first), then stronger alpha. Shuffled comparison independently selects alpha within that same rank",
    feasibility_gate="Selected PRIMARY pooled sourceVAL deltaR2>.05; each E01/E03 VALroom>0; at least3of4VALpeople>0; if conditionalrank>0, natural equal-room deltaR2 beats matched-rank shuffled bestalpha",
    targets="Unclipped hip-root-relative provided2D displacement only; no complete pose, hip location, initial pose, camera calibration or3D claim",
    exposure="E01/E03 sixteen TRAIN/four disjoint VAL people only; no numeric E02 values. E02 has historical development exposure in other studies and is not newly evaluated here",
    observer_limit="This unconstrained bilinear inverse need not be an integrable pose-gradient field; path/integration drift is possible. A causal complementary pose observer requires separate validation and cannot receive a ground-truth query initial skeleton",
)


def save_json(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(payload)


def save_npz(path, **values):
    with Path(path).open("xb") as stream:
        np.savez(stream, **values)


def real_finite(value, name, shape=None):
    raw = np.asarray(value)
    if (
        raw.dtype.kind not in "iuf"
        or not raw.size
        or (shape is not None and raw.shape != shape)
        or not np.isfinite(raw).all()
    ):
        raise ValueError("Expected real finite compatible array: " + name)
    converted = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(converted).all():
        raise ValueError("Nonfinite float64 conversion: " + name)
    return converted


def mm(a, b):
    a, b = real_finite(a, "matrix left"), real_finite(b, "matrix right")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Matrix product dimensions differ")
    return real_finite(
        dgemm(1.0, np.asarray(a, order="F"), np.asarray(b, order="F")), "matrix result"
    )


def pair_indices(records, ends):
    records, ends = np.asarray(records), np.asarray(ends)
    if records.ndim != 1 or ends.shape != records.shape or ends.dtype.kind not in "iu":
        raise ValueError("Record labels and integer endpoints must align")
    lookup = {
        (str(record), int(end)): i for i, (record, end) in enumerate(zip(records, ends))
    }
    if len(lookup) != len(ends):
        raise ValueError("Duplicate record/endpoint pair")
    current, previous = [], []
    for i, (record, end) in enumerate(zip(records, ends)):
        past = lookup.get((str(record), int(end) - 8))
        if past is not None:
            current.append(i)
            previous.append(past)
    if not current:
        raise ValueError("No exact causal eight-frame pairs remain")
    return np.asarray(current, dtype=np.int64), np.asarray(previous, dtype=np.int64)


def extract_role_pairs(feature, target, metadata, role, name):
    """Index only one explicitly permitted source role, never an E02 value."""
    role = np.asarray(role)
    if (
        name not in ("train", "validation")
        or role.dtype != bool
        or role.shape != np.asarray(metadata["room"]).shape
        or not role.any()
    ):
        raise ValueError("Explicit source role required")
    ids = np.flatnonzero(role)
    rooms, people = metadata["room"][ids], metadata["person"][ids]
    validation_people = np.isin(people, ["S04", "S09", "S24", "S29"])
    if not np.isin(rooms, ["E01", "E03"]).all() or (
        validation_people.any() if name == "train" else not validation_people.all()
    ):
        raise ValueError("Prohibited source role/person or E02 numeric read")
    meta = {
        key: np.asarray(metadata[key])[ids]
        for key in ("room", "person", "record", "action", "end")
    }
    for room, person, record, action in zip(
        meta["room"], meta["person"], meta["record"], meta["action"]
    ):
        first = 1 if room == "E01" else 21
        if person not in {f"S{i:02}" for i in range(first, first + 10)} or str(
            record
        ).split("/") != [str(room), str(person), str(action)]:
            raise ValueError("Source record/person/room/action metadata disagree")
    if (
        meta["end"].dtype.kind not in "iu"
        or np.any(meta["end"] < 39)
        or np.any(meta["end"] > 295)
        or np.any((meta["end"] - 39) % 8)
    ):
        raise ValueError("Source pair endpoints must be canonical integer indices")
    current, previous = pair_indices(meta["record"], meta["end"])
    phi = real_finite(feature[ids, -64:], "source current embeddings", (len(ids), 64))
    pose = real_finite(target[ids], "source hip-relative poses", (len(ids), 17, 2))
    delta, context = pair_values(phi[current], phi[previous])
    target_delta = (pose[current] - pose[previous]).reshape(len(current), 34)
    return dict(
        delta=delta,
        context=context,
        target=target_delta,
        metadata={key: value[current] for key, value in meta.items()},
        current_global=ids[current],
        previous_global=ids[previous],
    )


def pair_values(current, previous):
    current = real_finite(current, "current frame embeddings")
    previous = real_finite(previous, "past frame embeddings", current.shape)
    if current.ndim != 2 or current.shape[1] != 64:
        raise ValueError("Expected current/past embedding matrices with64 columns")
    return current - previous, (current + previous) / 2


def fit_map(delta, context):
    delta = real_finite(delta, "TRAIN delta")
    context = real_finite(context, "TRAIN context", delta.shape)
    if delta.ndim != 2 or delta.shape[1] != 64 or len(delta) < 2:
        raise ValueError("At least two TRAIN pairs and64 embedding dimensions required")
    delta_rms = np.maximum(np.sqrt(np.square(delta).mean(0)), 0.001)
    mean, scale = context.mean(0), np.maximum(context.std(0), 0.001)
    normalized = (context - mean) / scale
    covariance = mm(normalized.T, normalized) / len(context)
    eigen, basis = eigh(covariance, check_finite=True)
    if eigen.min() < -1e-8 * max(float(np.max(np.abs(eigen))), 1.0):
        raise ValueError("Context covariance is not numerically positive semidefinite")
    order = np.arange(63, -1, -1)
    eigen, basis = eigen[order], basis[:, order]
    signs = np.where(
        basis[np.argmax(np.abs(basis), axis=0), np.arange(64)] < 0, -1.0, 1.0
    )
    basis = basis * signs
    projected = mm(normalized, basis)
    pc_scale = np.maximum(np.sqrt(np.square(projected).mean(0)), 0.001)
    return dict(
        delta_rms=real_finite(delta_rms, "delta RMS"),
        context_mean=mean,
        context_scale=scale,
        context_basis=basis,
        context_pc_scale=pc_scale,
        context_eigenvalues=eigen,
    )


def transform(delta, context, state, rank):
    if rank not in RANKS:
        raise ValueError("Undeclared context rank")
    delta = real_finite(delta, "delta")
    context = real_finite(context, "context", delta.shape)
    if delta.ndim != 2 or delta.shape[1] != 64:
        raise ValueError("Expected64-dimensional embedding deltas")
    rms = real_finite(state["delta_rms"], "TRAIN RMS", (64,))
    scale = real_finite(state["context_scale"], "TRAIN context scale", (64,))
    pc_scale = real_finite(state["context_pc_scale"], "TRAIN PC scale", (64,))
    if np.any(rms < 0.001) or np.any(scale < 0.001) or np.any(pc_scale < 0.001):
        raise ValueError("Frozen scale violates its floor")
    normalized_delta = delta / rms
    result = normalized_delta / np.sqrt(64)
    if rank:
        mean = real_finite(state["context_mean"], "TRAIN context mean", (64,))
        basis = real_finite(state["context_basis"], "TRAIN context basis", (64, 64))
        pc = mm((context - mean) / scale, basis[:, :rank]) / pc_scale[:rank]
        interaction = (pc[:, :, None] * normalized_delta[:, None, :]).reshape(
            len(delta), rank * 64
        )
        result = np.concatenate((result, interaction / np.sqrt(64 * rank)), axis=1)
    return real_finite(result, "conditional design", (len(delta), 64 * (1 + rank)))


def fit_ridge(design, target):
    """No intercept and no target mean. Invalid numeric candidates lack scores."""
    x, y = real_finite(design, "TRAIN design"), real_finite(target, "TRAIN target")
    if x.ndim != 2 or y.shape != (len(x), 34):
        raise ValueError("TRAIN design and34-coordinate delta targets must align")
    rows, weights = [], {}
    try:
        covariance, right = mm(x.T, x), mm(x.T, y)
        eigen, basis = eigh(covariance, check_finite=True)
        projected = mm(basis.T, right)
    except (ValueError, np.linalg.LinAlgError) as error:
        return [
            dict(
                alpha=alpha,
                status="unavailable",
                reason="TRAIN system failure: " + str(error),
            )
            for alpha in ALPHAS
        ], weights
    for alpha in ALPHAS:
        try:
            regularized = eigen + len(x) * alpha
            if not np.isfinite(regularized).all() or regularized.min() <= 0:
                raise ValueError("Nonpositive/nonfinite regularized spectrum")
            condition = float(regularized.max() / regularized.min())
            if not np.isfinite(condition):
                raise ValueError("Nonfinite regularized condition number")
            weight = mm(basis, projected / regularized[:, None])
            residual = (
                mm(covariance + len(x) * alpha * np.eye(x.shape[1]), weight) - right
            )
            relative = float(
                np.linalg.norm(residual) / max(np.linalg.norm(right), 1e-12)
            )
            if not np.isfinite(relative) or relative > 1e-6:
                raise ValueError("Normal-equation relative residual exceeds1e-6")
            weights[alpha] = weight
            rows.append(
                dict(
                    alpha=alpha,
                    status="available",
                    normal_equation_relative_residual=relative,
                    regularized_condition_number=condition,
                    gram_eigenvalue_min=float(eigen.min()),
                    gram_eigenvalue_max=float(eigen.max()),
                )
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            rows.append(dict(alpha=alpha, status="unavailable", reason=str(error)))
    return rows, weights


def delta_score(prediction, target):
    prediction = real_finite(prediction, "predicted displacement")
    target = real_finite(target, "target displacement", prediction.shape)
    if prediction.ndim != 2 or prediction.shape[1] != 34:
        raise ValueError("Expected34 displacement coordinates")
    error = prediction - target
    sse, energy = float(np.square(error).sum()), float(np.square(target).sum())
    if not np.isfinite([sse, energy]).all():
        raise ValueError("Nonfinite displacement score energy")
    return dict(
        delta_r2=1 - sse / energy if energy > 0 else None,
        rmse=float(np.sqrt(np.square(error).mean())),
        mae=float(np.abs(error).mean()),
        target_squared_energy=energy,
        samples=len(target),
    )


def scores(prediction, target, metadata):
    result = dict(pooled=delta_score(prediction, target))
    for field in ("room", "person", "action"):
        labels = np.asarray(metadata[field])
        if labels.shape != (len(target),):
            raise ValueError("Displacement score metadata differs")
        result["per_" + field] = {
            str(name): delta_score(prediction[labels == name], target[labels == name])
            for name in sorted(set(labels))
        }
    r2 = [value["delta_r2"] for value in result["per_room"].values()]
    result["equal_room_delta_r2"] = (
        float(np.mean(r2)) if all(x is not None for x in r2) else None
    )
    return result


def source_permutation(n, role):
    if n < 2 or role not in ("train", "validation"):
        raise ValueError("At least two pairs and declared permutation role required")
    return np.random.default_rng(411 if role == "train" else 709).permutation(n)


def fit_source_models(feature, target, metadata, train, validation):
    """Actual fit boundary: no held values are numerically indexed."""
    if train.shape != validation.shape or np.any(train & validation):
        raise ValueError("Disjoint source TRAIN/VAL roles required")
    tr = extract_role_pairs(feature, target, metadata, train, "train")
    va = extract_role_pairs(feature, target, metadata, validation, "validation")
    if set(tr["metadata"]["person"]) & set(va["metadata"]["person"]) or set(
        tr["metadata"]["record"]
    ) & set(va["metadata"]["record"]):
        raise ValueError("Source people or recordings overlap")
    mapping = fit_map(tr["delta"], tr["context"])
    permutations = {
        name: source_permutation(len(data["target"]), name)
        for name, data in (("train", tr), ("validation", va))
    }
    models, reports = {}, {}
    zero = {
        name: scores(np.zeros_like(data["target"]), data["target"], data["metadata"])
        for name, data in (("train", tr), ("validation", va))
    }
    for rank in RANKS:
        for mode in ("natural",) if rank == 0 else ("natural", "shuffled"):
            designs = {}
            for name, data in (("train", tr), ("validation", va)):
                context = (
                    data["context"]
                    if mode == "natural"
                    else data["context"][permutations[name]]
                )
                designs[name] = transform(data["delta"], context, mapping, rank)
            rows, weights = fit_ridge(designs["train"], tr["target"])
            tag = f"rank{rank}-{mode}"
            reports[tag] = []
            for row in rows:
                record = dict(row, rank=rank, context=mode)
                if row["status"] == "available":
                    weight = weights[row["alpha"]]
                    train_prediction, validation_prediction = (
                        mm(designs["train"], weight),
                        mm(designs["validation"], weight),
                    )
                    record["train"] = scores(
                        train_prediction, tr["target"], tr["metadata"]
                    )
                    record["validation"] = scores(
                        validation_prediction, va["target"], va["metadata"]
                    )
                    models[tag, row["alpha"]] = dict(
                        weight=weight, validation_prediction=validation_prediction
                    )
                reports[tag].append(record)
    return dict(
        mapping=mapping,
        models=models,
        reports=reports,
        zero=zero,
        train_pairs=tr,
        validation_pairs=va,
        permutations=permutations,
    )


def selected_and_gate(reports):
    natural = [
        row
        for rows in reports.values()
        for row in rows
        if row["context"] == "natural"
        and row["status"] == "available"
        and row["validation"]["equal_room_delta_r2"] is not None
    ]
    if not natural:
        return dict(
            choice=None,
            matched_shuffled_choice=None,
            checks=dict(available=False),
            passes=False,
        )
    choice = min(
        natural,
        key=lambda row: (
            -row["validation"]["equal_room_delta_r2"],
            row["rank"],
            -row["alpha"],
        ),
    )
    validation = choice["validation"]
    matched = None
    if choice["rank"]:
        candidates = [
            row
            for row in reports[f"rank{choice['rank']}-shuffled"]
            if row["status"] == "available"
            and row["validation"]["equal_room_delta_r2"] is not None
        ]
        if candidates:
            matched = min(
                candidates,
                key=lambda row: (
                    -row["validation"]["equal_room_delta_r2"],
                    -row["alpha"],
                ),
            )
    room_r2 = [
        validation["per_room"].get(room, {}).get("delta_r2") for room in ("E01", "E03")
    ]
    people = validation["per_person"]
    positive_people = sum(
        value["delta_r2"] is not None and value["delta_r2"] > 0
        for value in people.values()
    )
    checks = dict(
        pooled_delta_r2=validation["pooled"]["delta_r2"] is not None
        and validation["pooled"]["delta_r2"] > 0.05,
        each_source_room_positive=all(x is not None and x > 0 for x in room_r2),
        three_of_four_people_positive=len(people) == 4 and positive_people >= 3,
        conditional_context_better_than_matched_shuffle=choice["rank"] == 0
        or (
            matched is not None
            and validation["equal_room_delta_r2"]
            > matched["validation"]["equal_room_delta_r2"]
        ),
    )

    def compact(row):
        if row is None:
            return None
        return dict(
            rank=row["rank"],
            alpha=row["alpha"],
            context=row["context"],
            equal_room_delta_r2=row["validation"]["equal_room_delta_r2"],
        )

    return dict(
        choice=compact(choice),
        matched_shuffled_choice=compact(matched),
        checks=checks,
        positive_validation_people=positive_people,
        passes=all(checks.values()),
    )


def source_contract(study):
    if digest(study / "source-ready.json") != EXPECTED_READY:
        raise ValueError("Unexpected source-ready version")
    ready = json.loads((study / "source-ready.json").read_text())
    required = {"registration.json", "roles.npz", "targets-and-metadata.npz"} | {
        arm + "-embedding.npy" for arm in ARMS
    }
    if (
        ready["kind"] != "masked-csi-source-only-ready"
        or ready["held_metrics_computed"] is not False
        or not required <= set(ready["artifacts_sha256"])
    ):
        raise ValueError("Incomplete source-ready input contract")
    verify_hashes(study, ready["artifacts_sha256"])
    registration = json.loads((study / "registration.json").read_text())
    manifest = registration.get("source_sha256")
    if not isinstance(manifest, dict) or not REQUIRED_TRAINING_SOURCES <= set(manifest):
        raise ValueError("Required producer/runtime source provenance omitted")
    if any(
        ready["artifacts_sha256"].get("executed-source/" + name) != expected
        for name, expected in manifest.items()
    ):
        raise ValueError("Producer snapshots are absent or unbound")
    verify_hashes(ROOT, manifest)
    verify_hashes(study / "executed-source", manifest)
    if (
        registration["config"]["steps"] != 4000
        or registration["config"]["held_room"] != "E02"
    ):
        raise ValueError("Unexpected frozen encoder recipe")
    return registration


def run(study, out):
    if out.exists():
        raise FileExistsError(out)
    started = time.monotonic()
    source_contract(study)
    with np.load(study / "targets-and-metadata.npz", allow_pickle=False) as stored:
        metadata = {
            key: stored[key] for key in ("room", "person", "record", "action", "end")
        }
    train, validation = source_roles(metadata)
    with np.load(study / "roles.npz", allow_pickle=False) as roles:
        if (
            roles["train"].dtype != bool
            or roles["validation"].dtype != bool
            or roles["train"].shape != train.shape
            or roles["validation"].shape != validation.shape
        ):
            raise ValueError("Stored source roles must use the exact Boolean shape")
        np.testing.assert_array_equal(train, roles["train"])
        np.testing.assert_array_equal(validation, roles["validation"])
    target = stored_npz_array(study / "targets-and-metadata.npz", "pose")
    if target.shape != (len(train), 17, 2) or target.dtype.kind not in "fiu":
        raise ValueError("Unexpected lazy real hip-relative target schema")
    sources = [
        Path(__file__),
        ROOT / "tests/test_conditioned_dynamics_probe.py",
        ROOT / "scripts/domain_masked_motion_diagnostic.py",
        ROOT / "scripts/domain_covariance_full_cache.py",
        ROOT / "scripts/domain_background_probe.py",
    ]
    out.mkdir(parents=True)
    registration = dict(
        kind="source-only-conditioned-Jacobian-displacement",
        config=CONFIG,
        source_sha256={str(path.relative_to(ROOT)): digest(path) for path in sources},
        numpy=np.__version__,
        scipy=SCIPY_VERSION,
        held_room_numeric_values_read=False,
    )
    save_json(out / "registration.json", registration)
    for path in sources:
        destination = out / "executed-source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    results = {}
    for arm in ARMS:
        feature = np.load(
            study / (arm + "-embedding.npy"), mmap_mode="r", allow_pickle=False
        )
        if feature.shape != (len(train), 512) or feature.dtype.kind not in "fiu":
            raise ValueError("Unexpected frozen real embedding schema")
        fitted = fit_source_models(feature, target, metadata, train, validation)
        save_npz(out / (arm + "-TRAIN-map.npz"), **fitted["mapping"])
        for name in ("train", "validation"):
            pair = fitted[name + "_pairs"]
            save_npz(
                out / (arm + "-" + name + "-pairs.npz"),
                current_global=pair["current_global"],
                previous_global=pair["previous_global"],
                delta=pair["delta"],
                context=pair["context"],
                target=pair["target"],
                context_permutation=fitted["permutations"][name],
                **pair["metadata"],
            )
        for (tag, alpha), model in fitted["models"].items():
            save_npz(out / f"{arm}-{tag}-alpha{ALPHAS.index(alpha)}-model.npz", **model)
        result = dict(
            all_configs=fitted["reports"],
            zero_displacement=fitted["zero"],
            selection=selected_and_gate(fitted["reports"]),
            train_pair_count=len(fitted["train_pairs"]["target"]),
            validation_pair_count=len(fitted["validation_pairs"]["target"]),
            permutation_fixed_points={
                role: int(np.sum(perm == np.arange(len(perm))))
                for role, perm in fitted["permutations"].items()
            },
        )
        results[arm] = result
        save_json(out / (arm + "-source-results.json"), result)
        print(
            json.dumps(dict(arm=arm, selection=result["selection"]), allow_nan=False),
            flush=True,
        )
    save_json(
        out / "results.json",
        dict(
            registration=registration,
            arms=results,
            primary_feasibility=results["masked_temporal"]["selection"],
            seconds=time.monotonic() - started,
            scope="Source-only displacement feasibility, not completepose/3D/newE02evaluation",
        ),
    )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.study, args.out)


if __name__ == "__main__":
    main()
