"""Independent CPU audit of sealed full-image source-only root readouts.

No encoder, root head or optimizer is fitted here. General LU solves qualify
saved coefficients, and only SOURCE TRAIN/VAL teachers enter predictive error
calculations. Held annotations are used solely for exact raw-frame joins.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import verify_masked_pretrain_probe as source_audit  # noqa: E402

ENCODERS = source_audit.ARMS
ARMS = ENCODERS + ("raw_current",)
ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0)
BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
VALIDATION_PEOPLE = {"S04", "S09", "S14", "S19", "S24", "S29"}
ATOL = RTOL = 2e-8
# Independent required runtime sets. A self-consistent hash loop over an
# unrelated file does not establish which producer or kernel actually ran.
# Test files may be additionally bound by the producers; they are not a
# substitute for any runtime entry below.
REQUIRED_ROOT_SOURCES = frozenset(
    {
        "scripts/domain_masked_fullpose_probe.py",
        "ghostmesh/metrics.py",
        "scripts/domain_background_probe.py",
        "scripts/domain_covariance_cached_probe.py",
        "scripts/domain_covariance_full_cache.py",
        "scripts/domain_covariance_source_probe.py",
        "ghostmesh/reference_covariance.py",
        "ghostmesh/covariance_dynamics.py",
    }
)
REQUIRED_ENCODER_SOURCES = frozenset(
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


def require(condition, message):
    if not condition:
        raise ValueError(message)


def real(value, name, shape=None):
    a = np.asarray(value)
    require(
        a.dtype.kind in "iuf" and a.size > 0 and np.isfinite(a).all(),
        "Expected finite real numeric artifact: " + name,
    )
    require(shape is None or a.shape == shape, "Shape mismatch: " + name)
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.asarray(a, dtype=np.float64)
    require(np.isfinite(converted).all(), "Unrepresentable real artifact: " + name)
    return converted


def close(observed, expected, name, atol=ATOL, rtol=RTOL):
    a, b = real(observed, name + "/saved"), real(expected, name + "/independent")
    require(a.shape == b.shape, "Shape mismatch: " + name)
    require(
        np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False),
        "Numeric readback mismatch: " + name,
    )
    return float(np.max(np.abs(a - b)))


def read(path):
    return json.loads(Path(path).read_text())


def npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def exact(observed, expected, name):
    a, b = np.asarray(observed), np.asarray(expected)
    require(a.shape == b.shape and np.array_equal(a, b), "Exact join mismatch: " + name)


def validate_roles(data, observed):
    n = len(data["end"])
    require(set(data["room"]) == {"E01", "E02", "E03"}, "Only SOURCE rooms allowed")
    require(
        np.asarray(data["end"]).dtype.kind in "iu"
        and np.isin(data["end"], np.arange(39, 297, 8)).all(),
        "Noncanonical endpoints",
    )
    held = data["room"] == "E02"
    validation = ~held & np.isin(data["person"], sorted(VALIDATION_PEOPLE))
    masks = dict(train=~(held | validation), validation=validation, held=held)
    require(set(observed) == set(masks), "Role fields differ")
    for name, mask in masks.items():
        require(
            np.asarray(observed[name]).dtype == bool
            and observed[name].shape == (n,)
            and mask.any(),
            "Malformed Boolean role: " + name,
        )
        exact(observed[name], mask, "role/" + name)
    require(
        tuple(len(set(data["person"][masks[k]])) for k in masks) == (16, 4, 10),
        "SOURCE person coverage differs",
    )
    for field in ("person", "record"):
        groups = [set(data[field][mask]) for mask in masks.values()]
        require(
            not any(groups[i] & groups[j] for i in range(3) for j in range(i)),
            "SOURCE role identities overlap",
        )
    return masks


def raw_target_join(raw_pose, raw_valid, records, data):
    """All-row annotation equality is an index audit, never a held error score."""
    require(np.asarray(raw_pose).dtype.kind in "iuf", "Raw poses must be real")
    require(np.asarray(raw_valid).dtype == bool, "Raw validity must be Boolean")
    names = [row["id"] for row in records]
    require(len(set(names)) == len(names), "Duplicate raw recording")
    require(
        all(name.split("/")[0] in {"E01", "E02", "E03"} for name in names),
        "Reserved or unknown raw room",
    )
    lookup = {row["id"]: (row["offset"], row["frames"]) for row in records}
    require(np.asarray(data["end"]).dtype.kind in "iu", "Endpoint type differs")
    require(
        len(data["record"]) == len(data["end"])
        and len(set(zip(data["record"].tolist(), data["end"].tolist())))
        == len(data["end"]),
        "Duplicate or misaligned canonical row",
    )
    indices = []
    for name, end in zip(data["record"], data["end"]):
        require(name in lookup, "Unknown raw record")
        offset, count = lookup[name]
        require(0 <= int(end) < count, "Endpoint escapes raw recording")
        indices.append(offset + int(end))
    ids = np.asarray(indices, dtype=np.int64)
    require(np.asarray(raw_valid[ids]).all(), "Invalid full-image teacher joint")
    target = real(raw_pose[ids], "raw full-image teachers", (len(ids), 17, 2))
    body = target - target[:, [11, 12]].mean(1, keepdims=True)
    close(data["pose"], body, "raw/cache hip-relative join", 0, 0)
    return target, ids


def source_nje(prediction, target, rooms):
    p, y = (
        real(prediction, "validation predictions"),
        real(target, "validation teachers"),
    )
    require(
        p.shape == y.shape and y.ndim == 3 and y.shape[1:] == (17, 2),
        "VAL pose shape differs",
    )
    require(
        len(rooms) == len(y) and set(rooms) == {"E01", "E03"},
        "Only both SOURCE validation rooms may be scored",
    )
    diagonal = np.linalg.norm(y.max(1) - y.min(1), axis=1)
    require((diagonal > 1e-6).all(), "Degenerate source-validation skeleton")
    errors = np.linalg.norm(p - y, axis=2) / diagonal[:, None]
    return float(np.mean([errors[rooms == room].mean() for room in ("E01", "E03")]))


def audit_root(z, body, target, roles, rooms, state, selection):
    """General LU candidate equations and SOURCE-only alpha/beta selection."""
    z = real(z, "root design")
    n, width = z.shape
    body = real(body, "frozen body", (n, 17, 2))
    midpoint = float(np.max(np.abs(body[:, [11, 12]].mean(1))))
    require(midpoint <= 1e-6, "Frozen body hip midpoint is not zero")
    train, validation = roles["train"], roles["validation"]
    require(
        train.dtype == validation.dtype == bool
        and train.shape == validation.shape == (n,)
        and train.any()
        and validation.any()
        and not (train & validation).any(),
        "Invalid SOURCE fitting roles",
    )
    require(
        np.asarray(rooms).shape == (n,)
        and set(rooms[train]) == {"E01", "E03"}
        and set(rooms[validation]) == {"E01", "E03"},
        "Only both SOURCE rooms may supply TRAIN/VAL root labels",
    )
    require(np.asarray(target).shape == (n, 17, 2), "Full teacher shape differs")
    # Deliberately do not validate or score held teachers at this boundary.
    source_target = real(target[train], "TRAIN teachers")
    val_target = real(target[validation], "VAL teachers")
    root = source_target[:, [11, 12]].mean(1)
    xm, ym = z[train].mean(0), root.mean(0)
    weights = real(state["weights"], "root coefficients", (len(ALPHAS), width, 2))
    close(state["x_mean"], xm, "root TRAIN design mean")
    close(state["root_mean"], ym, "root TRAIN target mean", 0, 0)
    close(
        selection["maximum_body_hip_midpoint"], midpoint, "reported body midpoint", 0, 0
    )
    grid = selection["grid"]
    require(len(grid) == len(ALPHAS) * len(BETAS), "Root candidate grid length differs")
    x, y = z[train] - xm, root - ym
    gram, right = source_audit.mm(x.T, x), source_audit.mm(x.T, y)
    candidates, residuals, coefficient_errors, metric_errors = [], [], [], []
    for ai, alpha in enumerate(ALPHAS):
        system = gram + train.sum() * alpha * np.eye(width)
        direct = solve(system, right, assume_a="gen", check_finite=True)
        residual = float(
            np.linalg.norm(source_audit.mm(system, weights[ai]) - right)
            / max(np.linalg.norm(right), 1e-12)
        )
        require(np.isfinite(residual) and residual < 1e-8, "Root normal equation fails")
        residuals.append(residual)
        coefficient_errors.append(
            close(weights[ai], direct, "general LU root coefficients")
        )
        variation = source_audit.mm(z[validation] - xm, direct)
        for bi, beta in enumerate(BETAS):
            expected = grid[ai * len(BETAS) + bi]
            close(expected["alpha"], alpha, "grid alpha", 0, 0)
            close(expected["beta"], beta, "grid beta", 0, 0)
            predicted = body[validation] + (ym + beta * variation)[:, None]
            nje = source_nje(predicted, val_target, rooms[validation])
            metric_errors.append(
                close(
                    expected["source_validation_nje"],
                    nje,
                    "SOURCE VAL full-image NJE",
                    2e-9,
                    2e-8,
                )
            )
            reported_residual = real(
                expected["normal_equation_relative_residual"],
                "reported root residual",
                (),
            )
            require(
                0 <= float(reported_residual) < 1e-6, "Reported root equation fails"
            )
            candidates.append(dict(alpha=alpha, beta=beta, source_validation_nje=nje))
    chosen = min(
        candidates,
        key=lambda row: (row["source_validation_nje"], row["beta"], -row["alpha"]),
    )
    saved_choice = selection["choice"]
    selected_grid = grid[
        ALPHAS.index(chosen["alpha"]) * len(BETAS) + BETAS.index(chosen["beta"])
    ]
    require(set(saved_choice) == set(selected_grid), "Selected grid fields differ")
    for field in selected_grid:
        close(
            saved_choice[field],
            selected_grid[field],
            "selected grid row/" + field,
            0,
            0,
        )
    require(
        saved_choice["alpha"] == chosen["alpha"]
        and saved_choice["beta"] == chosen["beta"],
        "SOURCE alpha/beta selection differs",
    )
    close(
        saved_choice["source_validation_nje"],
        chosen["source_validation_nje"],
        "selected SOURCE VAL NJE",
        2e-9,
        2e-8,
    )
    selected = np.asarray(state["selected"])
    require(
        selected.shape == ()
        and selected.dtype.kind in "iu"
        and int(selected) == ALPHAS.index(chosen["alpha"]),
        "Root selected alpha index differs",
    )
    beta = float(real(state["beta"], "root beta", ()))
    require(beta == chosen["beta"], "Root selected beta differs")
    root_prediction = ym + beta * source_audit.mm(z - xm, weights[int(selected)])
    return body + root_prediction[:, None], dict(
        selected_alpha=chosen["alpha"],
        selected_beta=beta,
        source_validation_nje=chosen["source_validation_nje"],
        maximum_normal_equation_relative_residual=max(residuals),
        maximum_general_lu_coefficient_difference=max(coefficient_errors),
        maximum_source_validation_nje_difference=max(metric_errors),
        candidate_equations_verified=len(ALPHAS),
        source_validation_candidates_verified=len(candidates),
    )


def audit_numeric(experiment, source, cached, data, target, roles, *, dimensions=512):
    choices = read(experiment / "frozen-fullpose-selections.json")
    require(set(choices) == set(ARMS), "Root choice arms differ")
    n = len(data["end"])
    results = {}
    for arm in ARMS:
        if arm == "raw_current":
            require(
                cached.ndim == 2 and cached.shape[1] == 9576 and len(cached) >= n,
                "Covariance allocation differs",
            )
            require(
                np.asarray(cached).dtype.kind in "iuf",
                "Covariance coordinates must be real",
            )
            feature = np.asarray(
                cached[:n].reshape(n, 8, 19, 7, 9)[:, :, :, 0], dtype=np.float64
            ).reshape(n, 1368)
        else:
            feature = np.load(
                source / (arm + "-embedding.npy"), mmap_mode="r", allow_pickle=False
            )
            real(feature, "saved embedding", (n, 512))
        mapping = npz(source / (arm + "-rff.npz"))
        z, errors = source_audit.independent_rff(
            feature, roles["train"], mapping, dimensions=dimensions
        )
        body = npz(source / (arm + "-readout.npz"))["prediction"]
        predicted, proof = audit_root(
            z,
            body,
            target,
            roles,
            data["room"],
            npz(experiment / (arm + "-root-state.npz")),
            choices[arm],
        )
        saved = np.load(experiment / (arm + "-full-prediction.npy"), allow_pickle=False)
        proof["maximum_full_prediction_difference"] = close(
            saved, predicted, "saved full-image prediction/" + arm
        )
        proof["map_readbacks"] = errors
        results[arm] = proof
    audit_priors(experiment, target, data, roles)
    return results


def audit_priors(experiment, target, data, roles):
    n = len(data["end"])
    mean = real(target[roles["train"]], "static TRAIN teachers").mean(0)
    expected_mean = np.broadcast_to(mean, (n, 17, 2))
    close(
        np.load(experiment / "source_mean-full-prediction.npy", allow_pickle=False),
        expected_mean,
        "source full-pose mean",
        0,
        0,
    )
    clock = np.empty((n, 17, 2))
    for end in np.unique(data["end"]):
        train = roles["train"] & (data["end"] == end)
        require(train.any(), "Endpoint clock lacks TRAIN support")
        clock[data["end"] == end] = real(target[train], "clock TRAIN teachers").mean(0)
    close(
        np.load(
            experiment / "source_endpoint_clock-full-prediction.npy", allow_pickle=False
        ),
        clock,
        "source endpoint-clock prior",
        0,
        0,
    )


def load_contract(experiment, source, raw, cache, expected_fullpose, expected_source):
    # Both external identities are checked before opening any model/array artifact.
    require(
        source_audit.digest(experiment / "fullpose-source-ready.json")
        == expected_fullpose,
        "Full-image ready SHA differs",
    )
    require(
        source_audit.digest(source / "source-ready.json") == expected_source,
        "Encoder ready SHA differs",
    )
    ready, source_ready = (
        read(experiment / "fullpose-source-ready.json"),
        read(source / "source-ready.json"),
    )
    full_required = {
        "registration.json",
        "roles.npz",
        "canonical-metadata.npz",
        "full-targets.npy",
        "raw-frame-indices.npy",
        "frozen-fullpose-selections.json",
    }
    full_required.update(arm + "-root-state.npz" for arm in ARMS)
    full_required.update(
        arm + "-full-prediction.npy"
        for arm in (*ARMS, "source_mean", "source_endpoint_clock")
    )
    source_required = {
        "registration.json",
        "roles.npz",
        "targets-and-metadata.npz",
        "initial.pt",
        "schedule.npy",
        "preoptimization-seal.json",
        "frozen-source-selections.json",
        "raw_current-rff.npz",
        "raw_current-readout.npz",
    }
    source_required.update(
        arm + suffix
        for arm in ENCODERS
        for suffix in (
            "-embedding.npy",
            "-final.pt",
            "-rff.npz",
            "-readout.npz",
            "-training.json",
            "-complete.json",
        )
    )
    source_required.update(arm + "-optimizer.pt" for arm in ENCODERS[1:])
    source_required.update(
        f"{arm}-step-{step:05}{suffix}"
        for arm in ENCODERS[1:]
        for step in (1000, 2000, 3000, 4000)
        for suffix in (".pt", ".json")
    )
    for receipt, directory, kind, required, field in (
        (
            ready,
            experiment,
            "masked-csi-fullimage-source-ready",
            full_required,
            "held_metrics_computed",
        ),
        (
            source_ready,
            source,
            "masked-csi-source-only-ready",
            source_required,
            "held_metrics_computed",
        ),
    ):
        require(
            receipt["kind"] == kind and receipt[field] is False, "Wrong ready stage"
        )
        require(
            required <= set(receipt["artifacts_sha256"]),
            "Ready seal omits required artifacts",
        )
        source_audit.verify_hashes(directory, receipt["artifacts_sha256"])
    reg, source_reg = (
        read(experiment / "registration.json"),
        read(source / "registration.json"),
    )
    require(reg["held_metrics_computed"] is False, "Root registration already scored")
    require(
        ready["source_ready_sha256"] == reg["source_ready_sha256"] == expected_source,
        "Root/encoder ready lineage differs",
    )
    for registration, directory, required_sources, label in (
        (reg, experiment, REQUIRED_ROOT_SOURCES, "root"),
        (source_reg, source, REQUIRED_ENCODER_SOURCES, "encoder"),
    ):
        manifest = registration.get("source_sha256")
        require(
            isinstance(manifest, dict) and required_sources <= set(manifest),
            "Missing required " + label + " producer/runtime source provenance",
        )
        source_audit.verify_hashes(ROOT, registration["source_sha256"])
        source_audit.verify_hashes(
            directory / "executed-source", registration["source_sha256"]
        )
        for key, path in (
            ("raw_receipt_sha256", raw / "receipt.json"),
            ("cache_receipt_sha256", cache / "receipt.json"),
        ):
            require(
                registration[key] == source_audit.digest(path),
                "Raw/cache receipt lineage differs",
            )
    for key in ("raw_receipt_sha256", "cache_receipt_sha256"):
        require(ready[key] == reg[key], "Root ready receipt identity differs")
    require(
        reg["config"]["room"] == "E02"
        and tuple(reg["config"]["alphas"]) == ALPHAS
        and tuple(reg["config"]["betas"]) == BETAS,
        "Root fitting recipe differs",
    )
    require(
        source_reg["config"]["held_room"] == "E02"
        and source_reg["config"]["steps"] == 4000
        and source_reg["config"]["seed"] == 2027,
        "Encoder source protocol differs",
    )
    rr, cr = read(raw / "receipt.json"), read(cache / "receipt.json")
    require(
        cr["kind"] == "full-source-covariance-feature-cache"
        and cr["fits_performed"] == 0
        and cr["raw_receipt_sha256"] == reg["raw_receipt_sha256"]
        and cr["raw_arrays_sha256"] == rr["arrays"],
        "Raw/cache provenance differs",
    )
    source_audit.verify_hashes(
        raw, {name + ".npy": value for name, value in rr["arrays"].items()}
    )
    source_audit.verify_hashes(cache, cr["arrays_sha256"])
    source_audit.verify_hashes(cache, read(cache / "artifact_manifest.json"))
    expected_records = [
        f"E{room:02}/S{person:02}/A{action:02}"
        for room in range(1, 4)
        for person in range((room - 1) * 10 + 1, room * 10 + 1)
        for action in range(1, 28)
    ]
    require(
        rr["records"] == cr["records"]
        and [r["id"] for r in rr["records"]] == expected_records
        and all(
            r["frames"] == 297 and r["offset"] == i * 297
            for i, r in enumerate(rr["records"])
        ),
        "Only canonical SOURCE raw records permitted",
    )
    require(
        source_audit.digest(source / "targets-and-metadata.npz")
        == cr["arrays_sha256"]["targets-and-metadata.npz"],
        "Encoder target cache identity differs",
    )
    data = npz(source / "targets-and-metadata.npz")
    n = len(data["end"])
    require(
        n == cr["logical_rows"] == 26556 and cr["nominal_rows"] == 26730,
        "Canonical logical row prefix differs",
    )
    metadata = npz(experiment / "canonical-metadata.npz")
    require(
        set(metadata) == {"record", "person", "action", "room", "end"},
        "Full metadata fields differ",
    )
    for field in metadata:
        exact(metadata[field], data[field], "canonical metadata/" + field)
    for record, room, person, action in zip(
        data["record"], data["room"], data["person"], data["action"]
    ):
        require(
            record == f"{room}/{person}/{action}", "Canonical identity fields disagree"
        )
    roles = validate_roles(data, npz(experiment / "roles.npz"))
    source_roles = npz(source / "roles.npz")
    for name, mask in roles.items():
        exact(source_roles[name], mask, "encoder role/" + name)
        require(
            reg["counts"][name] == source_reg["counts"][name] == int(mask.sum()),
            "Role count receipt differs",
        )
    raw_pose = np.load(raw / "pose.npy", mmap_mode="r", allow_pickle=False)
    raw_valid = np.load(raw / "valid.npy", mmap_mode="r", allow_pickle=False)
    require(
        raw_pose.shape == (240570, 17, 2) and raw_valid.shape == (240570, 17),
        "Raw target allocation differs",
    )
    target, ids = raw_target_join(raw_pose, raw_valid, rr["records"], data)
    close(
        np.load(experiment / "full-targets.npy", allow_pickle=False),
        target,
        "sealed full-target join",
        0,
        0,
    )
    saved_ids = np.load(experiment / "raw-frame-indices.npy", allow_pickle=False)
    require(saved_ids.dtype.kind in "iu", "Raw frame index type differs")
    exact(saved_ids, ids, "raw frame indices")
    cached = np.load(
        cache / "untreated_covariance-features.npy", mmap_mode="r", allow_pickle=False
    )
    require(cached.shape == (26730, 9576), "Covariance cache allocation differs")
    return data, target, roles, cached, ready, source_ready


def verify(
    experiment,
    source,
    raw,
    cache,
    output,
    *,
    expected_fullpose_sha256,
    expected_source_sha256,
):
    experiment, source, raw, cache, output = map(
        Path, (experiment, source, raw, cache, output)
    )
    require(
        not output.exists() and not output.is_symlink(), "Audit output already exists"
    )
    auditor_sources = {
        str(Path(__file__).relative_to(ROOT)): source_audit.digest(Path(__file__)),
        "tests/test_masked_fullpose_verifier.py": source_audit.digest(
            ROOT / "tests/test_masked_fullpose_verifier.py"
        ),
        "scripts/verify_masked_pretrain_probe.py": source_audit.digest(
            ROOT / "scripts/verify_masked_pretrain_probe.py"
        ),
        "scripts/research_numeric_readback.py": source_audit.digest(
            ROOT / "scripts/research_numeric_readback.py"
        ),
    }
    data, target, roles, cached, ready, source_ready = load_contract(
        experiment, source, raw, cache, expected_fullpose_sha256, expected_source_sha256
    )
    proofs = audit_numeric(experiment, source, cached, data, target, roles)
    source_audit.verify_hashes(experiment, ready["artifacts_sha256"])
    source_audit.verify_hashes(source, source_ready["artifacts_sha256"])
    source_audit.verify_hashes(ROOT, auditor_sources)
    require(
        source_audit.digest(experiment / "fullpose-source-ready.json")
        == expected_fullpose_sha256
        and source_audit.digest(source / "source-ready.json") == expected_source_sha256,
        "Ready receipt changed during audit",
    )
    report = dict(
        kind="independent-masked-fullimage-root-audit",
        fullpose_source_ready_sha256=expected_fullpose_sha256,
        source_ready_sha256=expected_source_sha256,
        held_metrics_computed=False,
        model_training_performed=False,
        device="CPU",
        canonical_target_rows_verified=len(target),
        maximum_target_join_difference=0.0,
        source_counts={k: int(v.sum()) for k, v in roles.items()},
        root_models_verified=len(proofs),
        root_candidate_equations_verified=sum(
            row["candidate_equations_verified"] for row in proofs.values()
        ),
        source_validation_alpha_beta_candidates_verified=sum(
            row["source_validation_candidates_verified"] for row in proofs.values()
        ),
        full_prediction_rows_verified_per_arm=len(target),
        static_priors_verified=2,
        readouts=proofs,
        auditor_sources_sha256=auditor_sources,
        tolerances=dict(
            map_absolute=2e-8,
            map_relative=2e-8,
            root_absolute=ATOL,
            root_relative=RTOL,
            normal_equation_relative=1e-8,
            validation_nje_absolute=2e-9,
            validation_nje_relative=2e-8,
            target_joins_and_static_priors="exact",
        ),
        scope="E01-E03 development only. General LU qualifies saved root coefficients; SOURCE TRAIN/VAL full-image NJE qualifies choices. No held predictive errors, encoder training, optimizer calls, E04, camera calibration or3D. Body/encoder readouts are hash-bound inputs; their numerical qualification belongs to the separate source auditor.",
    )
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("experiment", "source", "raw", "cache", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--fullpose-source-ready-sha256", required=True)
    parser.add_argument("--source-ready-sha256", required=True)
    args = parser.parse_args()
    report = verify(
        args.experiment,
        args.source,
        args.raw,
        args.cache,
        args.output,
        expected_fullpose_sha256=args.fullpose_source_ready_sha256,
        expected_source_sha256=args.source_ready_sha256,
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
