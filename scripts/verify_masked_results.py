"""Independent, read-only post-score qualification of four frozen E02 protocols.

This is explicitly a scored-results audit. It never fits, selects, rewrites a
study, invokes a producer's predictions/metrics, or uses query hips as inputs.
An externally pinned pre-evaluation ledger is mandatory before artifact reads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import verify_masked_calibration_probe as contracts  # noqa: E402
from scripts import verify_masked_fullpose_probe as root_contract  # noqa: E402

ENCODERS = contracts.ENCODERS
METHODS = contracts.METHODS
PRIMARY = "masked_temporal/residual_krr"
PROTOCOLS = {
    "zero_body": (
        "domain-masked-pretrain-e02-v1",
        "source-ready.json",
        "masked-csi-source-only-ready",
        "held_metrics_computed",
    ),
    "zero_full": (
        "domain-masked-fullpose-e02-v1",
        "fullpose-source-ready.json",
        "masked-csi-fullimage-source-ready",
        "held_metrics_computed",
    ),
    "calibrated_body": (
        "domain-masked-calibration-e02-v1",
        "calibration-ready.json",
        "masked-csi-reference-calibration-ready",
        "query_metrics_computed",
    ),
    "calibrated_full": (
        "domain-masked-fullimage-calibration-e02-v1",
        "fullimage-calibration-ready.json",
        "masked-csi-fullimage-calibration-ready",
        "query_metrics_computed",
    ),
}
AUDITS = {
    "source": "work/domain-masked-source-verification-v1.json",
    "root": "work/domain-masked-fullpose-verification-v1.json",
    "body": "work/domain-masked-calibration-verification-v1.json",
    "full": "work/domain-masked-fullimage-calibration-verification-v1.json",
}
ATOL, RTOL = 2e-9, 2e-8
METRIC_ATOL, METRIC_RTOL = 1e-11, 1e-10


def require(condition, message):
    if not condition:
        raise ValueError(message)


def real(value, name="numeric artifact", shape=None):
    a = np.asarray(value)
    require(
        a.dtype.kind in "iuf" and a.size > 0 and np.isfinite(a).all(),
        "Expected finite real " + name,
    )
    require(shape is None or a.shape == shape, "Shape mismatch: " + name)
    with np.errstate(over="ignore", invalid="ignore"):
        a = a.astype(np.float64, copy=False)
    require(np.isfinite(a).all(), "Unrepresentable numeric artifact: " + name)
    return a


def close(a, b, name, atol=ATOL, rtol=RTOL):
    a, b = real(a, name), real(b, name)
    require(
        a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False),
        "Numeric readback mismatch: " + name,
    )
    return float(np.max(np.abs(a - b)))


def exact(a, b, name):
    a, b = np.asarray(a), np.asarray(b)
    require(
        a.shape == b.shape and np.array_equal(a, b), "Exact identity mismatch: " + name
    )


def permutation(records, seed):
    records = np.asarray(records)
    require(records.ndim == 1 and len(records) > 0, "Record vector required")
    result = np.arange(len(records))
    rng = np.random.default_rng(seed)
    for record in sorted(set(records)):
        ids = np.flatnonzero(records == record)
        result[ids] = rng.permutation(ids)
    return result


def motion_r2(prediction, target, records):
    """Center after removing the first frame, avoiding large static cancellation.

    None means target motion is at/below 64 float64 ulps of coordinate scale.
    No energy is rounded or fabricated; pooled nondegenerate motion keeps every
    recording's prediction error, including records with static teachers.
    """
    p, y = real(prediction), real(target)
    records = np.asarray(records)
    require(p.shape == y.shape and len(records) == len(y), "Motion rows differ")
    error = energy = numerical_floor = 0.0
    for record in sorted(set(records)):
        a, b = p[records == record], y[records == record]
        a, b = a - a[:1], b - b[:1]
        a, b = a - a.mean(0), b - b.mean(0)
        error += float(np.square(a - b).sum())
        energy += float(np.square(b).sum())
        scale = max(1.0, float(np.max(np.abs(y[records == record]))))
        numerical_floor += b.size * (64 * np.finfo(np.float64).eps * scale) ** 2
    require(
        np.isfinite([error, energy, numerical_floor]).all(), "Nonfinite motion energy"
    )
    return None if energy <= numerical_floor else float(1.0 - error / energy)


def metric(prediction, target, records, mode):
    p, y = real(prediction), real(target)
    require(
        mode in {"body", "full"}
        and p.shape == y.shape
        and y.ndim == 3
        and y.shape[1:] == (17, 2)
        and len(records) == len(y),
        "Metric pose/record schema differs",
    )
    diagonal = np.linalg.norm(y.max(1) - y.min(1), axis=1)
    require(
        np.isfinite(diagonal).all() and (diagonal > 1e-6).all(),
        "Degenerate teacher bounding box",
    )
    errors = real(
        np.linalg.norm(p - y, axis=2) / diagonal[:, None], "normalized errors"
    )
    n = len(y)
    result = dict(
        pck=float((errors <= 0.2).mean()),
        pck_threshold=0.2,
        pck_curve={
            f"{threshold:g}": float((errors <= threshold).mean())
            for threshold in (0.05, 0.1, 0.2)
        },
        normalized_joint_error=float(errors.mean()),
        normalizer="per-frame valid ground-truth 2D bounding-box diagonal",
        coordinate_units="normalized image coordinates",
        valid_joints=n * 17,
        valid_frames=n,
        excluded_frames=0,
        per_joint=[
            dict(
                joint=j,
                count=n,
                pck=float((errors[:, j] <= 0.2).mean()),
                nje=float(errors[:, j].mean()),
            )
            for j in range(17)
        ],
    )
    if mode == "body":
        result["record_centered_articulation_r2"] = motion_r2(p, y, records)
    else:
        pr, yr = (
            p[:, [11, 12]].mean(1, keepdims=True),
            y[:, [11, 12]].mean(1, keepdims=True),
        )
        result.update(
            record_centered_full_image_r2=motion_r2(p, y, records),
            record_centered_body_r2=motion_r2(p - pr, y - yr, records),
            record_centered_root_r2=motion_r2(pr, yr, records),
        )
    return result


def scores(prediction, target, data, ids, perm, mode):
    require(
        np.asarray(ids).dtype.kind in "iu" and len(set(ids)) == len(ids),
        "Invalid query indices",
    )
    require(np.asarray(perm).dtype.kind in "iu", "Permutation must use integer indices")
    exact(np.sort(perm), np.arange(len(ids)), "permutation coverage")
    records = data["record"][ids]
    exact(records[perm], records, "within-record permutation")
    result = dict(
        aligned=metric(prediction, target[ids], records, mode),
        shuffled=metric(prediction[perm], target[ids], records, mode),
    )
    for field in ("person", "action"):
        groups = data[field][ids]
        result["per_" + field] = {
            str(key): metric(
                prediction[groups == key],
                target[ids][groups == key],
                records[groups == key],
                mode,
            )
            for key in sorted(set(groups))
        }
    return result


def positive(value):
    return value is not None and float(real(value, "motion scalar", ())) > 0


def gate(protocol, methods, beta=None):
    p = methods["masked_temporal" if protocol.startswith("zero_") else PRIMARY][
        "aligned"
    ]
    if protocol == "zero_body":
        controls = [
            methods[name]["aligned"]
            for name in (
                "source_mean",
                "random_frozen",
                "supervised",
                "raw_current",
                "frozen_temporal",
            )
        ]
        return dict(
            nje_better_all_controls=all(
                p["normalized_joint_error"] < c["normalized_joint_error"]
                for c in controls
            ),
            pck_not_worse_controls=all(
                p["pck_curve"][k] >= c["pck_curve"][k]
                for c in controls
                for k in ("0.05", "0.2")
            ),
            positive_motion_r2=positive(p["record_centered_articulation_r2"]),
            shuffle_worse=p["normalized_joint_error"]
            < methods["masked_temporal"]["shuffled"]["normalized_joint_error"],
        )
    if protocol == "zero_full":
        controls = [
            methods[name]["aligned"]
            for name in (
                "source_mean",
                "source_endpoint_clock",
                "random_frozen",
                "supervised",
                "raw_current",
            )
        ]
        return dict(
            nje_beats_matched_controls=all(
                p["normalized_joint_error"] < c["normalized_joint_error"]
                for c in controls
            ),
            pck_not_worse=all(
                p["pck_curve"][k] >= c["pck_curve"][k]
                for c in controls
                for k in ("0.05", "0.2")
            ),
            positive_full_motion=positive(p["record_centered_full_image_r2"]),
            positive_body_motion=positive(p["record_centered_body_r2"]),
            shuffle_worse=p["normalized_joint_error"]
            < methods["masked_temporal"]["shuffled"]["normalized_joint_error"],
        )
    require(
        protocol in {"calibrated_body", "calibrated_full"} and beta is not None,
        "Unknown gate protocol",
    )
    controls = [value["aligned"] for name, value in methods.items() if name != PRIMARY]
    common = dict(beta_positive=float(real(beta, "selected calibration beta", ())) > 0)
    nje = all(
        p["normalized_joint_error"] < c["normalized_joint_error"] for c in controls
    )
    pck = all(
        p["pck_curve"][k] >= c["pck_curve"][k]
        for c in controls
        for k in ("0.05", "0.2")
    )
    shuffle = (
        p["normalized_joint_error"]
        < methods[PRIMARY]["shuffled"]["normalized_joint_error"]
    )
    if protocol == "calibrated_body":
        return dict(
            common,
            nje_strictly_better_all_controls=nje,
            pck_not_worse_all_controls=pck,
            motion_positive=positive(p["record_centered_articulation_r2"]),
            query_shuffle_worse=shuffle,
        )
    return dict(
        common,
        full_nje_strictly_better_all_controls=nje,
        full_pck_not_worse_all_controls=pck,
        full_pose_motion_positive=positive(p["record_centered_full_image_r2"]),
        body_motion_positive=positive(p["record_centered_body_r2"]),
        query_shuffle_worse=shuffle,
    )


def differences(observed, expected, path="", findings=None):
    """Keep scientific disagreements; reject malformed/nonfinite report fields."""
    findings = [] if findings is None else findings
    if isinstance(expected, dict):
        require(
            isinstance(observed, dict) and set(observed) == set(expected),
            "Report fields differ: " + path,
        )
        for key in expected:
            differences(observed[key], expected[key], path + "/" + str(key), findings)
    elif isinstance(expected, list):
        require(
            isinstance(observed, list) and len(observed) == len(expected),
            "Report sequence differs: " + path,
        )
        for i, value in enumerate(expected):
            differences(observed[i], value, path + "/" + str(i), findings)
    elif expected is None:
        if observed is not None:
            real(observed, "reported motion", ())
            findings.append(
                dict(
                    path=path,
                    stored=observed,
                    independent=None,
                    reason="Target motion is numerically degenerate after stable centering",
                )
            )
    elif isinstance(expected, bool):
        require(isinstance(observed, bool), "Expected Boolean report field: " + path)
        if observed != expected:
            findings.append(
                dict(
                    path=path,
                    stored=observed,
                    independent=expected,
                    reason="Formal Boolean decision differs",
                )
            )
    elif isinstance(expected, str):
        require(observed == expected, "Report definition differs: " + path)
    elif isinstance(expected, int):
        require(type(observed) is int, "Expected integer report count: " + path)
        if observed != expected:
            findings.append(
                dict(
                    path=path,
                    stored=observed,
                    independent=expected,
                    reason="Integer metric coverage differs",
                )
            )
    elif observed is None:
        findings.append(
            dict(
                path=path,
                stored=None,
                independent=expected,
                reason="Stored result omits defined motion",
            )
        )
    else:
        a, b = (
            real(observed, "reported scalar", ()),
            real(expected, "independent scalar", ()),
        )
        if not np.isclose(a, b, atol=METRIC_ATOL, rtol=METRIC_RTOL):
            findings.append(
                dict(
                    path=path,
                    stored=float(a),
                    independent=float(b),
                    reason="Independent numerical metric differs",
                )
            )
    return findings


def predict_calibrated(state, query_feature, base):
    """Own direct cdist RBF/constant/zero math, with no query annotation input."""
    contracts.state_schema(state)
    x = real(query_feature, "query feature")
    require(x.ndim == 2, "Query feature matrix required")
    base = real(base, "frozen query f0", (len(x), 34))
    method, kind = str(state["method"]), str(state["correction_kind"])
    require(
        str(state["base_kind"])
        == ("source_mean" if method == "absolute_krr" else "frozen_arm"),
        "Calibration base semantics differ",
    )
    if str(state["base_kind"]) == "source_mean":
        base = np.broadcast_to(
            real(state["source_mean"], "absolute source mean", (34,)), base.shape
        )
    beta = float(real(state["beta"], "calibration beta", ()))
    require((kind == "zero") == (beta == 0), "Zero correction/beta semantics differ")
    if kind == "zero":
        return base.copy()
    if kind == "constant":
        require(method == "constant_residual", "Constant correction method differs")
        return real(
            base + beta * real(state["correction"], "constant correction", (34,))
        )
    width = float(real(state["bandwidth_squared"], "kernel bandwidth", ()))
    reference = real(state["reference_design"], "reference design")
    require(
        reference.ndim == 2 and reference.shape[1] == x.shape[1],
        "Query/reference widths differ",
    )
    mean, scale = (
        real(state["input_mean"], shape=(x.shape[1],)),
        real(state["input_scale"], shape=(x.shape[1],)),
    )
    dual = real(state["dual"], "calibration dual", (len(reference), 34))
    result = np.empty_like(base)
    for start in range(0, len(x), 128):
        stop = start + 128
        distance = real(
            cdist((x[start:stop] - mean) / scale, reference, metric="sqeuclidean"),
            "query/reference distances",
        )
        with np.errstate(under="ignore"):
            kernel = np.exp(-distance / (2 * width))
        result[start:stop] = base[start:stop] + beta * np.einsum(
            "ij,jk->ik", kernel, dual, optimize=False
        )
    return real(result, "reconstructed calibration predictions")


def expected_methods(protocol):
    arms = ENCODERS + (
        ("raw_current", "frozen_temporal")
        if protocol.endswith("body")
        else ("raw_current",)
    )
    if protocol.startswith("zero_"):
        return (
            set(arms)
            | {"source_mean"}
            | ({"source_endpoint_clock"} if protocol.endswith("full") else set())
        )
    return {
        arm + "/" + method for arm in arms for method in ("zero_shot", *METHODS)
    } | {
        "source_mean",
        "reference_mean",
        "reference_mean_mixture",
        "source_endpoint_clock",
    }


def ledger_contract(ledger, expected_sha, root):
    require(
        isinstance(expected_sha, str)
        and len(expected_sha) == 64
        and contracts.digest(ledger) == expected_sha,
        "Evaluation freeze ledger SHA differs",
    )
    frozen = contracts.read(ledger)
    require(
        frozen["stage"] == "frozen_before_first_E02_query_metrics"
        and frozen["held_query_metrics_already_computed"] is False
        and frozen["seed"] == 2027
        and frozen["development_room"] == "E02"
        and frozen["reference_labels"] == 120,
        "Wrong evaluation freeze stage",
    )
    require(
        frozen["primary_zero"] == "masked_temporal"
        and frozen["primary_calibrated"] == PRIMARY,
        "Frozen primary differs",
    )
    require(
        frozen["reference_people"] == ["S11", "S12"]
        and frozen["calibrated_query_people"] == [f"S{i:02}" for i in range(13, 21)]
        and frozen["zero_query_people"] == [f"S{i:02}" for i in range(11, 21)],
        "Frozen people differ",
    )
    required = {
        "work/" + directory + "/" + filename
        for directory, filename, _, _ in PROTOCOLS.values()
    } | set(AUDITS.values())
    require(
        set(frozen["frozen_sha256"]) == required,
        "Freeze ledger omits or adds protocol/audit identities",
    )
    contracts.hashes(root, frozen["frozen_sha256"])
    return frozen


def audit_receipt(root, name, proof, source_sha, full_sha, calibrated_sha=None):
    require(
        proof["model_training_performed"] is False
        and proof["source_ready_sha256"] == source_sha,
        "Upstream audit source/stage differs",
    )
    kinds = {
        "source": "independent-masked-csi-source-only-audit",
        "root": "independent-masked-fullimage-root-audit",
        "body": "independent-masked-calibration-fitted-state-audit",
        "full": "independent-masked-calibration-fitted-state-audit",
    }
    require(proof["kind"] == kinds[name], "Upstream audit kind differs")
    if name in {"source", "root"}:
        require(
            proof["held_metrics_computed"] is False,
            "Upstream audit was not source-only",
        )
        keys = {
            "scripts/verify_masked_pretrain_probe.py",
            "scripts/research_numeric_readback.py",
        }
        if name == "root":
            keys.add("scripts/verify_masked_fullpose_probe.py")
            require(
                proof["fullpose_source_ready_sha256"] == full_sha,
                "Root audit ready differs",
            )
        require(
            keys <= set(proof["auditor_sources_sha256"]),
            "Upstream audit omits runtime fingerprints",
        )
        contracts.hashes(root, proof["auditor_sources_sha256"])
    else:
        require(
            proof["query_metrics_computed"] is False and proof["mode"] == name,
            "Reference audit stage differs",
        )
        require(
            proof["calibration_ready_sha256"] == calibrated_sha,
            "Reference audit ready differs",
        )
        require(
            proof["auditor_sha256"]
            == contracts.digest(root / "scripts/verify_masked_calibration_probe.py"),
            "Reference auditor code differs",
        )


def scored_contract(root, ledger, expected_sha, raw, cache):
    """Explicit post-scored contract; prospective auditors remain unchanged."""
    frozen = ledger_contract(ledger, expected_sha, root)
    directories, ready, registrations = {}, {}, {}
    for protocol, (directory, filename, kind, flag) in PROTOCOLS.items():
        path = root / "work" / directory
        require(
            (path / "results.json").is_file()
            and (path / "query-permutation.npy").is_file(),
            "Post-score audit requires all four completed scored protocols",
        )
        required = {"registration.json"}
        if protocol.startswith("calibrated_"):
            arms = ENCODERS + (
                ("raw_current", "frozen_temporal")
                if protocol.endswith("body")
                else ("raw_current",)
            )
            required |= {
                "roles-and-priors.npz",
                "frozen-reference-choices.json",
                "reference_mean_mixture.npz",
            }
            required |= {
                arm + "--" + method + ".npz" for arm in arms for method in METHODS
            }
        elif protocol == "zero_body":
            required |= {
                "roles.npz",
                "targets-and-metadata.npz",
                "frozen-source-selections.json",
                "frozen-temporal-control.npz",
            }
            required |= {
                arm + suffix
                for arm in (*ENCODERS, "raw_current")
                for suffix in ("-rff.npz", "-readout.npz")
            }
        else:
            required |= {
                "roles.npz",
                "canonical-metadata.npz",
                "full-targets.npy",
                "raw-frame-indices.npy",
                "frozen-fullpose-selections.json",
            }
            required |= {arm + "-root-state.npz" for arm in (*ENCODERS, "raw_current")}
            required |= {
                arm + "-full-prediction.npy" for arm in expected_methods(protocol)
            }
        ready[protocol], registrations[protocol] = contracts.sealed(
            path, filename, kind, required, flag
        )
        contracts.hashes(root, registrations[protocol]["source_sha256"])
        directories[protocol] = path
    source_sha = frozen["frozen_sha256"][
        "work/" + PROTOCOLS["zero_body"][0] + "/source-ready.json"
    ]
    full_sha = frozen["frozen_sha256"][
        "work/" + PROTOCOLS["zero_full"][0] + "/fullpose-source-ready.json"
    ]
    for protocol in ("calibrated_body", "calibrated_full"):
        registration = registrations[protocol]
        require(
            registration["source_ready_sha256"] == source_sha
            and registration["cache_receipt_sha256"]
            == contracts.digest(cache / "receipt.json")
            and registration["query_metrics_computed"] is False
            and registration["reference_rows"] == 120
            and registration["query_rows"] == 7111
            and registration["reference_people"] == frozen["reference_people"]
            and registration["query_people"] == frozen["calibrated_query_people"],
            "Calibrated source/cache/cohort lineage differs",
        )
        if protocol == "calibrated_full":
            require(
                registration["fullpose_source_ready_sha256"] == full_sha
                and registration["raw_receipt_sha256"]
                == contracts.digest(raw / "receipt.json")
                and registration["config"]["target_mode"] == "full_image_2d",
                "Full calibration target lineage differs",
            )
    require(
        contracts.digest(directories["zero_body"] / "frozen-temporal-control.npz")
        == registrations["zero_body"]["frozen_temporal_model_sha256"],
        "Frozen temporal baseline identity differs",
    )
    for name, path in AUDITS.items():
        proof = contracts.read(root / path)
        calibrated_sha = (
            None
            if name in {"source", "root"}
            else frozen["frozen_sha256"][
                "work/"
                + PROTOCOLS["calibrated_" + name][0]
                + "/"
                + PROTOCOLS["calibrated_" + name][1]
            ]
        )
        audit_receipt(root, name, proof, source_sha, full_sha, calibrated_sha)
    data, target, roles, _, _, _ = root_contract.load_contract(
        directories["zero_full"],
        directories["zero_body"],
        raw,
        cache,
        full_sha,
        source_sha,
    )
    require(
        len(target) == 26556 and roles["held"].sum() == 8886,
        "Zero-shot cohort counts differ",
    )
    return frozen, directories, ready, registrations, data, target, roles


def frozen_features(source, cache, arm, ids, n):
    if arm in ENCODERS:
        a = np.load(
            source / (arm + "-embedding.npy"), mmap_mode="r", allow_pickle=False
        )
        require(a.shape == (n, 512), "Encoder embedding allocation differs")
        return real(a[ids], "query embedding")
    filename, width = (
        ("untreated_covariance-features.npy", 9576)
        if arm == "raw_current"
        else ("congruence_dynamics-features.npy", 2736)
    )
    a = np.load(cache / filename, mmap_mode="r", allow_pickle=False)
    require(a.shape == (26730, width), "Cached feature allocation differs")
    x = real(a[ids], "query covariance features")
    return (
        x.reshape(-1, 8, 19, 7, 9)[:, :, :, 0].reshape(len(ids), 1368)
        if arm == "raw_current"
        else x
    )


def source_predictions(source, full_source, cache, data, target, roles):
    ids = np.flatnonzero(roles["held"])
    n = len(target)
    body, full, features, errors = {}, {}, {}, {}
    for arm in (*ENCODERS, "raw_current"):
        features[arm] = frozen_features(source, cache, arm, ids, n)
        z = contracts.frozen_design(
            features[arm], contracts.npz(source / (arm + "-rff.npz"))
        )
        state = contracts.npz(source / (arm + "-readout.npz"))
        weights = real(state["weights"], shape=(5, 512, 34))
        index = np.asarray(state["selected"])
        require(
            index.shape == () and index.dtype.kind in "iu" and int(index) in range(5),
            "Source alpha differs",
        )
        body[arm] = real(
            np.einsum(
                "ij,jk->ik",
                z - real(state["x_mean"], shape=(512,)),
                weights[int(index)],
                optimize=False,
            )
            + real(state["y_mean"], shape=(34,))
        ).reshape(-1, 17, 2)
        errors["body/" + arm] = close(
            state["prediction"][ids], body[arm], "query source readout/" + arm
        )
        rs = contracts.npz(full_source / (arm + "-root-state.npz"))
        ri = np.asarray(rs["selected"])
        require(
            ri.shape == () and ri.dtype.kind in "iu" and int(ri) in range(5),
            "Root alpha differs",
        )
        beta = float(real(rs["beta"], shape=()))
        require(beta in contracts.BETAS, "Root beta differs")
        root = real(rs["root_mean"], shape=(2,)) + beta * np.einsum(
            "ij,jk->ik",
            z - real(rs["x_mean"], shape=(512,)),
            real(rs["weights"], shape=(5, 512, 2))[int(ri)],
            optimize=False,
        )
        full[arm] = body[arm] + root[:, None]
        saved = np.load(
            full_source / (arm + "-full-prediction.npy"),
            mmap_mode="r",
            allow_pickle=False,
        )
        errors["full/" + arm] = close(
            saved[ids], full[arm], "query source full prediction/" + arm
        )
    temporal = contracts.npz(source / "frozen-temporal-control.npz")
    exact(temporal["held_indices"], ids, "temporal query IDs")
    body["frozen_temporal"] = real(temporal["pose"], shape=(len(ids), 17, 2))
    features["frozen_temporal"] = frozen_features(
        source, cache, "frozen_temporal", ids, n
    )
    body["source_mean"] = np.broadcast_to(
        real(data["pose"][roles["train"]]).mean(0), (len(ids), 17, 2)
    )
    full["source_mean"] = np.broadcast_to(
        target[roles["train"]].mean(0), (len(ids), 17, 2)
    )
    clock = {
        int(end): target[roles["train"] & (data["end"] == end)].mean(0)
        for end in np.unique(data["end"][roles["train"]])
    }
    require(set(data["end"][ids]) <= set(clock), "Source clock lacks query endpoints")
    full["source_endpoint_clock"] = np.stack(
        [clock[int(end)] for end in data["end"][ids]]
    )
    for name in ("source_mean", "source_endpoint_clock"):
        saved = np.load(
            full_source / (name + "-full-prediction.npy"),
            mmap_mode="r",
            allow_pickle=False,
        )
        errors["full/" + name] = close(
            saved[ids], full[name], "query source prior/" + name, 0, 0
        )
    return ids, body, full, features, errors


def calibrated_predictions(
    experiment, mode, base, features, all_ids, data, target, roles
):
    priors = contracts.npz(experiment / "roles-and-priors.npz")
    reference, query = priors["reference"], priors["query"]
    contracts.validate_reference_query(data, reference, query)
    require(len(reference) == 120 and len(query) == 7111, "Calibrated cohort differs")
    positions = np.searchsorted(all_ids, query)
    exact(all_ids[positions], query, "calibrated/zero query join")
    source_mean = target[roles["train"]].mean(0)
    close(
        priors["source_mean"], source_mean.reshape(34), "calibration source mean", 0, 0
    )
    close(
        priors["reference_mean"],
        target[reference].mean(0),
        "calibration reference mean",
        0,
        0,
    )
    ends = np.unique(data["end"][roles["train"]])
    exact(priors["clock_ends"], ends, "calibration clock endpoints")
    clock = np.stack(
        [target[roles["train"] & (data["end"] == end)].mean(0) for end in ends]
    )
    close(priors["clock_poses"], clock, "calibration clock priors", 0, 0)
    choices = contracts.read(experiment / "frozen-reference-choices.json")
    arms = ENCODERS + (
        ("raw_current", "frozen_temporal") if mode == "body" else ("raw_current",)
    )
    require(
        set(choices)
        == {arm + "/" + method for arm in arms for method in METHODS}
        | {"reference_mean_mixture"},
        "Calibrated choice set differs",
    )
    predictions, errors = {}, {}
    for arm in arms:
        f0, feature = real(base[arm][positions]), real(features[arm][positions])
        predictions[arm + "/zero_shot"] = f0
        for method in METHODS:
            tag = arm + "/" + method
            state = contracts.npz(experiment / (tag.replace("/", "--") + ".npz"))
            require(str(state["method"]) == method, "Frozen method label differs")
            close(
                state["beta"], choices[tag]["selected"]["beta"], "selected beta", 0, 0
            )
            close(
                state["alpha"],
                choices[tag]["selected"]["alpha"],
                "selected alpha",
                0,
                0,
            )
            if method == "absolute_krr":
                close(
                    state["source_mean"],
                    source_mean.reshape(34),
                    "absolute source mean",
                    0,
                    0,
                )
            if str(state["correction_kind"]) == "kernel":
                real(
                    state["reference_design"],
                    "120-row reference design",
                    (len(reference), feature.shape[1]),
                )
            predictions[tag] = predict_calibrated(
                state, feature, f0.reshape(len(query), 34)
            ).reshape(-1, 17, 2)
    predictions["source_mean"] = np.broadcast_to(source_mean, (len(query), 17, 2))
    predictions["reference_mean"] = np.broadcast_to(
        target[reference].mean(0), (len(query), 17, 2)
    )
    state = contracts.npz(experiment / "reference_mean_mixture.npz")
    close(
        state["beta"],
        choices["reference_mean_mixture"]["selected"]["beta"],
        "mean mixture beta",
        0,
        0,
    )
    predictions["reference_mean_mixture"] = predict_calibrated(
        state,
        np.ones((len(query), 1)),
        predictions["source_mean"].reshape(len(query), 34),
    ).reshape(-1, 17, 2)
    lookup = dict(zip(ends.tolist(), clock))
    require(
        set(data["end"][query]) <= set(lookup), "Frozen clock misses query endpoint"
    )
    predictions["source_endpoint_clock"] = np.stack(
        [lookup[int(end)] for end in data["end"][query]]
    )
    errors = query_prediction_readback(experiment, predictions)
    return query, predictions, choices[PRIMARY]["selected"]["beta"], errors


def query_prediction_readback(experiment, predictions):
    errors = {}
    for name, predicted in predictions.items():
        saved = np.load(
            experiment / (name.replace("/", "--") + "-query-prediction.npy"),
            allow_pickle=False,
        )
        errors[name] = close(
            saved, predicted, "saved calibrated query prediction/" + name
        )
    return errors


def verify(root, raw, cache, ledger, output, *, expected_ledger_sha256):
    root, raw, cache, ledger, output = map(Path, (root, raw, cache, ledger, output))
    require(
        not output.exists() and not output.is_symlink(), "Proof output already exists"
    )
    require(
        not any(
            output.resolve().is_relative_to((root / "work" / spec[0]).resolve())
            for spec in PROTOCOLS.values()
        ),
        "Proof must be outside frozen study directories",
    )
    require(
        not any(
            output.resolve().is_relative_to(path.resolve()) for path in (raw, cache)
        ),
        "Proof must be outside frozen data caches",
    )
    own_sources = {
        "scripts/verify_masked_results.py": contracts.digest(Path(__file__)),
        "tests/test_masked_results_verifier.py": contracts.digest(
            ROOT / "tests/test_masked_results_verifier.py"
        ),
        "scripts/verify_masked_calibration_probe.py": contracts.digest(
            ROOT / "scripts/verify_masked_calibration_probe.py"
        ),
        "scripts/verify_masked_fullpose_probe.py": contracts.digest(
            ROOT / "scripts/verify_masked_fullpose_probe.py"
        ),
        "scripts/verify_masked_pretrain_probe.py": contracts.digest(
            ROOT / "scripts/verify_masked_pretrain_probe.py"
        ),
        "scripts/research_numeric_readback.py": contracts.digest(
            ROOT / "scripts/research_numeric_readback.py"
        ),
    }
    frozen, paths, ready, registrations, data, full_target, roles = scored_contract(
        root, ledger, expected_ledger_sha256, raw, cache
    )
    body_target = real(data["pose"])
    captured = {}
    for protocol, path in paths.items():
        names = ["results.json", "query-permutation.npy"]
        if protocol.startswith("calibrated_"):
            names += [
                name.replace("/", "--") + "-query-prediction.npy"
                for name in expected_methods(protocol)
            ]
        captured[protocol] = {name: contracts.digest(path / name) for name in names}
    ids, body, full, features, zero_errors = source_predictions(
        paths["zero_body"], paths["zero_full"], cache, data, full_target, roles
    )
    payloads = {"zero_body": (ids, body, None), "zero_full": (ids, full, None)}
    prediction_errors = {"zero_body_and_full": zero_errors}
    for protocol, mode, base, target in (
        ("calibrated_body", "body", body, body_target),
        ("calibrated_full", "full", full, full_target),
    ):
        query, predicted, beta, errors = calibrated_predictions(
            paths[protocol], mode, base, features, ids, data, target, roles
        )
        payloads[protocol] = query, predicted, beta
        prediction_errors[protocol] = errors
    reports, findings = {}, []
    for protocol, (query, predictions, beta) in payloads.items():
        require(
            set(predictions) == expected_methods(protocol),
            "Prediction method set differs",
        )
        mode = "body" if protocol.endswith("body") else "full"
        target = body_target if mode == "body" else full_target
        perm = np.load(paths[protocol] / "query-permutation.npy", allow_pickle=False)
        expected_perm = permutation(
            data["record"][query], 90210 if protocol.startswith("zero_") else 2027
        )
        require(perm.dtype.kind in "iu", "Permutation dtype differs")
        exact(perm, expected_perm, "frozen query permutation/" + protocol)
        stored = contracts.read(paths[protocol] / "results.json")
        require(
            stored["registration"] == registrations[protocol],
            "Scored registration differs",
        )
        require(set(stored["methods"]) == set(predictions), "Scored method set differs")
        results = {
            name: scores(predicted, target, data, query, perm, mode)
            for name, predicted in predictions.items()
        }
        decisions = gate(protocol, results, beta)
        differences(stored["methods"], results, protocol + "/methods", findings)
        differences(stored["gate"], decisions, protocol + "/gate", findings)
        differences(
            stored["passes"], all(decisions.values()), protocol + "/passes", findings
        )
        ready_field = {
            "zero_body": "source_ready_sha256",
            "zero_full": "fullpose_source_ready_sha256",
            "calibrated_body": "calibration_ready_sha256",
            "calibrated_full": "fullimage_calibration_ready_sha256",
        }[protocol]
        require(
            stored[ready_field]
            == contracts.digest(paths[protocol] / PROTOCOLS[protocol][1]),
            "Scored ready identity differs",
        )
        if protocol.startswith("calibrated_"):
            require(
                stored["frozen_choices_sha256"]
                == contracts.digest(paths[protocol] / "frozen-reference-choices.json"),
                "Scored choice identity differs",
            )
            reversals = {
                field: {
                    key: [
                        name
                        for name, method in results.items()
                        if name != PRIMARY
                        and value["normalized_joint_error"]
                        >= method["per_" + field][key]["normalized_joint_error"]
                    ]
                    for key, value in results[PRIMARY]["per_" + field].items()
                }
                for field in ("person", "action")
            }
            if stored["reversals"] != reversals:
                findings.append(
                    dict(
                        path=protocol + "/reversals",
                        stored=stored["reversals"],
                        independent=reversals,
                        reason="Subgroup NJE reversal membership differs",
                    )
                )
        elif protocol == "zero_body":
            require(
                stored["frozen_source_selection_sha256"]
                == contracts.digest(paths[protocol] / "frozen-source-selections.json"),
                "Scored source selection identity differs",
            )
        reports[protocol] = dict(
            query_rows=len(query),
            people=sorted(set(data["person"][query])),
            actions=sorted(set(data["action"][query])),
            prediction_sets=len(predictions),
            methods=results,
            independent_gate=decisions,
            independent_passes=all(decisions.values()),
        )
    ledger_contract(ledger, expected_ledger_sha256, root)
    for protocol in paths:
        contracts.hashes(paths[protocol], ready[protocol]["artifacts_sha256"])
        contracts.hashes(paths[protocol], captured[protocol])
    contracts.hashes(ROOT, own_sources)
    proof = dict(
        kind="independent-masked-E02-post-scored-qualification",
        status="qualified"
        if not findings
        else "qualified_predictions_with_metric_findings",
        evaluation_freeze_sha256=expected_ledger_sha256,
        upstream_frozen_sha256=frozen["frozen_sha256"],
        audited_output_sha256=captured,
        auditor_sources_sha256=own_sources,
        model_training_performed=False,
        study_artifacts_modified=False,
        prediction_sets_verified=sum(
            value["prediction_sets"] for value in reports.values()
        ),
        prediction_errors=prediction_errors,
        metric_findings=findings,
        protocols=reports,
        gate_passes={
            name: value["independent_passes"] for name, value in reports.items()
        },
        motion_definition="Recordwise subtract first frame, then temporal mean. Target energy <= N*(64*float64eps*max(1,coordinate scale))^2 is undefined/None; no rounding creates motion.",
        temporal_baseline_scope="The original temporal control is a hash-bound, index-joined saved baseline qualified upstream. Its underlying old encoder is not rerun here; its calibrated corrections are independently reconstructed.",
        tolerances=dict(
            prediction_atol=ATOL,
            prediction_rtol=RTOL,
            metric_atol=METRIC_ATOL,
            metric_rtol=METRIC_RTOL,
        ),
        scope="Post-E02 source-development numerical qualification only. Reconstructs frozen f0/root/reference corrections; directly recomputes held metrics after the pinned evaluation freeze. No optimizer/refit, query-hip placement, original-report rewrite, E04/RGB/GPU or efficacy promotion. Failed gates do not meet the sensing goal.",
    )
    serialized = json.dumps(proof, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--raw", type=Path, default=ROOT / "work/domain-source30-rf-v1")
    parser.add_argument(
        "--cache", type=Path, default=ROOT / "work/domain-covariance-full-cache-v1"
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=ROOT / "work/domain-masked-e02-evaluation-freeze-v1.json",
    )
    parser.add_argument("--ledger-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    proof = verify(
        args.root,
        args.raw,
        args.cache,
        args.ledger,
        args.output,
        expected_ledger_sha256=args.ledger_sha256,
    )
    print(
        json.dumps(
            {
                "status": proof["status"],
                "prediction_sets_verified": proof["prediction_sets_verified"],
                "metric_findings": len(proof["metric_findings"]),
                "gate_passes": proof["gate_passes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
