"""Prospective reference-person-held calibration of frozen CSI encoders.

Fitting consumes only the 120 declared reference labels. Query annotations are
read by the separate score phase, after every arm's choices and coefficients
are sealed. These hip-relative E02 development experiments are not full-pose,
3D, strict zero-shot, or demonstrated calibrated product results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.covariance_dynamics import hermitian_coordinates  # noqa: E402
from ghostmesh.reference_covariance import current_covariances  # noqa: E402
from ghostmesh.reference_kernel import ReferenceRBFKernel  # noqa: E402
from scripts.domain_background_probe import digest  # noqa: E402
from scripts.domain_covariance_cached_probe import load_inputs  # noqa: E402
from scripts.domain_covariance_full_cache import verify_hashes  # noqa: E402
from scripts.domain_covariance_source_probe import fold_masks  # noqa: E402
from scripts.domain_reference_covariance_probe import (  # noqa: E402
    query_indices,
    reference_indices,
    scores,
)

ENCODERS = (
    "random_frozen",
    "masked_only",
    "masked_temporal",
    "masked_shuffled",
    "supervised",
)
ARMS = ENCODERS + ("raw_current", "frozen_temporal")
ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0)
BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
PRIMARY = "masked_temporal/residual_krr"
CONFIG = dict(
    room="E02",
    primary=PRIMARY,
    alphas=ALPHAS,
    betas=BETAS,
    reference="S11/S12, A03/A07/A12, the same20 endpoint windows per recording as the fixed-reference study;120 labels total",
    selection="Reference-person-held CV: fit60 S11 refs, score60 S12 refs, then reverse. Equal two-person mean NJE. Every normalization and bandwidth uses ONLY the fitting60 rows. Ties prefer lower beta, then larger alpha.",
    final="After selection, fit all120 reference rows; no query statistics, labels, action ID or person ID enters the predictor",
    kernel="Reference-only SD floor.001 and median all unordered distinct-reference squared distances; exp(-d2/(2*median)); (K+n*alpha*I)dual=Yref-f0ref; no residual intercept",
    coordinates="Unclipped hip-root-relative supplied2D articulation; no absolute hip position or3D claim",
    controls="Same120 labels and CV grids for all seven arms. Zero-shot, constant residual, absolute KRR around source mean, within-reference-record permuted-label residual; source/ref means, source endpoint clock and independently CV-selected no-RF mean mixture.",
    limitations="Only two reference people and three reference actions support calibration selection. Queries are eight different people/all27actions. This is one previously exposed source-development room/seed, not final confirmation. Full-frame input normalization in the encoder includes masked-frame mean/SD statistics.",
    degenerate="An undefined reference bandwidth or all-near-constant features permits beta0 only; no bandwidth or informative kernel is fabricated.",
    gate="Primary beta>0; NJE strictly beats every other matched eight-person method; PCK.05/.2 no worse than all controls; positive record-centered articulation R2 and worse within-record query shuffle. All person/action reversals retained.",
)


def save_json(path, value):
    payload = json.dumps(value, allow_nan=False, indent=2) + "\n"
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(payload)


def save_npz(path, **values):
    with Path(path).open("xb") as stream:
        np.savez(stream, **values)


def save_array(path, value):
    with Path(path).open("xb") as stream:
        np.save(stream, value, allow_pickle=False)


def matrix(value, *, rows=None, width=None):
    if np.iscomplexobj(value):
        raise ValueError("Expected real-valued matrix")
    x = np.asarray(value, dtype=np.float64)
    if (
        x.ndim != 2
        or not x.shape[1]
        or (rows is not None and len(x) != rows)
        or (width is not None and x.shape[1] != width)
        or not np.isfinite(x).all()
    ):
        raise ValueError("Expected finite compatible matrix")
    return x


def nje(prediction, target):
    target = matrix(target, width=34).reshape(-1, 17, 2)
    prediction = matrix(prediction, rows=len(target), width=34).reshape(-1, 17, 2)
    diagonal = np.linalg.norm(target.max(1) - target.min(1), axis=1)
    if not len(target) or np.any(diagonal <= 1e-6):
        raise ValueError("Nonempty nondegenerate reference poses required")
    return float(
        (np.linalg.norm(prediction - target, axis=2) / diagonal[:, None]).mean()
    )


def permutation(records):
    records = np.asarray(records)
    rng = np.random.default_rng(2027)
    result = np.arange(len(records))
    for name in sorted(set(records)):
        ids = np.flatnonzero(records == name)
        result[ids] = rng.permutation(ids)
    return result


def partitions(people):
    people = np.asarray(people)
    if people.ndim != 1:
        raise ValueError("Exactly two reference people required")
    names = sorted(set(people))
    if len(names) != 2:
        raise ValueError("Exactly two reference people required")
    return [(people == name, people != name) for name in names]


def kernel_or_none(x):
    try:
        return ReferenceRBFKernel.from_reference(x, scale_floor=0.001)
    except ValueError as error:
        if "All reference columns are near constant" in str(
            error
        ) or "median bandwidth_squared" in str(error):
            return None
        raise


def fit_reference_model(
    x, labels, base, people, records, *, method="residual_krr", source_mean=None
):
    """No query values or targets are accepted by this calibration optimizer."""
    x = matrix(x)
    labels, base = (
        matrix(labels, rows=len(x), width=34),
        matrix(base, rows=len(x), width=34),
    )
    people, records = np.asarray(people), np.asarray(records)
    if people.shape != (len(x),) or records.shape != people.shape:
        raise ValueError("Reference identities do not align")
    if method not in (
        "residual_krr",
        "absolute_krr",
        "permuted_residual_krr",
        "constant_residual",
    ):
        raise ValueError("Unknown calibration method")
    if method == "permuted_residual_krr":
        labels = labels[permutation(records)]
    if method == "absolute_krr":
        source_mean = matrix(np.asarray(source_mean).reshape(1, -1), width=34)[0]
        base = np.broadcast_to(source_mean, base.shape)
    folds, diagnostics = [], []
    for train, validation in partitions(people):
        kernel = None if method == "constant_residual" else kernel_or_none(x[train])
        folds.append((train, validation, kernel))
        diagnostics.append(
            dict(
                fitting_people=sorted(set(people[train])),
                validation_people=sorted(set(people[validation])),
                fitting_rows=int(train.sum()),
                validation_rows=int(validation.sum()),
                input_mean=x[train].mean(0).tolist() if kernel is not None else None,
                input_scale=kernel.input_scale.tolist() if kernel is not None else None,
                bandwidth_squared=kernel.bandwidth_squared
                if kernel is not None
                else None,
            )
        )
    grid = []
    alphas = (0.0,) if method == "constant_residual" else ALPHAS
    for alpha in alphas:
        corrections, residuals = [], []
        for train, validation, kernel in folds:
            if method == "constant_residual":
                correction = np.broadcast_to(
                    (labels[train] - base[train]).mean(0), base[validation].shape
                )
                residuals.append(0.0)
            elif kernel is None:
                correction = None
                residuals.append(None)
            else:
                dual, residual = kernel.solve(labels[train] - base[train], alpha)
                correction = kernel.predict(x[validation], dual)
                residuals.append(residual)
            corrections.append(correction)
        for beta in BETAS:
            valid = beta == 0 or all(c is not None for c in corrections)
            values = (
                [
                    nje(base[v] if beta == 0 else base[v] + beta * c, labels[v])
                    for (_, v, _), c in zip(folds, corrections)
                ]
                if valid
                else None
            )
            grid.append(
                dict(
                    alpha=float(alpha),
                    beta=beta,
                    available=valid,
                    reference_person_cv_nje=float(np.mean(values)) if valid else None,
                    per_person_nje=values,
                    normal_equation_residuals=residuals,
                )
            )
    selected = min(
        (row for row in grid if row["available"]),
        key=lambda row: (row["reference_person_cv_nje"], row["beta"], -row["alpha"]),
    )
    state = dict(
        method=np.asarray(method),
        beta=np.asarray(selected["beta"]),
        alpha=np.asarray(selected["alpha"]),
        base_kind=np.asarray(
            "source_mean" if method == "absolute_krr" else "frozen_arm"
        ),
    )
    if method == "absolute_krr":
        state["source_mean"] = source_mean.copy()
    if selected["beta"] == 0:
        state["correction_kind"] = np.asarray("zero")
    elif method == "constant_residual":
        state.update(
            correction_kind=np.asarray("constant"), correction=(labels - base).mean(0)
        )
    else:
        kernel = kernel_or_none(x)
        if kernel is None:
            raise ValueError(
                "Selected informative CV kernel but full reference kernel is undefined"
            )
        dual, residual = kernel.solve(labels - base, selected["alpha"])
        state.update(kernel.state_dict())
        state.update(
            correction_kind=np.asarray("kernel"),
            dual=dual,
            final_equation_residual=np.asarray(residual),
        )
    return state, dict(selected=selected, grid=grid, folds=diagnostics)


def predict_reference_model(state, query, base):
    query = matrix(query)
    base = matrix(base, rows=len(query), width=34)
    if str(state["base_kind"]) == "source_mean":
        source_mean = matrix(np.asarray(state["source_mean"])[None], rows=1, width=34)[
            0
        ]
        base = np.broadcast_to(source_mean, base.shape)
    beta, kind = float(state["beta"]), str(state["correction_kind"])
    if beta not in BETAS:
        raise ValueError("Invalid frozen calibration beta")
    if kind == "zero":
        return base.copy()
    if kind == "constant":
        correction = matrix(np.asarray(state["correction"])[None], rows=1, width=34)[0]
        return matrix(base + beta * correction, rows=len(query), width=34)
    if kind != "kernel":
        raise ValueError("Unknown fitted correction")
    kernel = ReferenceRBFKernel.from_state(state)
    result = np.empty_like(base)
    for start in range(0, len(query), 128):
        stop = start + 128
        result[start:stop] = base[start:stop] + beta * kernel.predict(
            query[start:stop], state["dual"]
        )
    return matrix(result, rows=len(query), width=34)


def validate_roles(data, reference, query):
    n = len(data["person"])
    for ids in (reference, query):
        if (
            ids.ndim != 1
            or ids.dtype.kind not in "iu"
            or not len(ids)
            or len(set(ids)) != len(ids)
            or ids.min() < 0
            or ids.max() >= n
        ):
            raise ValueError("Invalid reference/query indices")
    if (
        set(reference) & set(query)
        or set(data["person"][reference]) & set(data["person"][query])
        or set(data["record"][reference]) & set(data["record"][query])
    ):
        raise ValueError("Reference/query people or recordings overlap")
    partitions(data["person"][reference])


def fit_arrays(features, base, data, reference, query, source_mean):
    """Actual fit boundary: only reference RF features and labels are indexed."""
    validate_roles(data, reference, query)
    labels = np.asarray(data["pose"][reference]).reshape(len(reference), 34)
    state, selection = {}, {}
    for arm, feature in features.items():
        x = feature[reference]
        f0 = np.asarray(base[arm][reference]).reshape(len(reference), 34)
        for method in (
            "residual_krr",
            "absolute_krr",
            "permuted_residual_krr",
            "constant_residual",
        ):
            tag = arm + "/" + method
            state[tag], selection[tag] = fit_reference_model(
                x,
                labels,
                f0,
                data["person"][reference],
                data["record"][reference],
                method=method,
                source_mean=source_mean,
            )
    # This is the same person-held selector with a constant source-mean f0 and
    # reference-mean residual. No CSI or query data enters its choice.
    mean_base = np.broadcast_to(source_mean, labels.shape)
    state["reference_mean_mixture"], selection["reference_mean_mixture"] = (
        fit_reference_model(
            np.ones((len(reference), 1)),
            labels,
            mean_base,
            data["person"][reference],
            data["record"][reference],
            method="constant_residual",
        )
    )
    return state, selection


def source_contract(source, cache, expected_digest):
    if digest(source / "source-ready.json") != expected_digest:
        raise ValueError("Source-ready identity differs")
    ready = json.loads((source / "source-ready.json").read_text())
    if (
        ready["kind"] != "masked-csi-source-only-ready"
        or ready["held_metrics_computed"] is not False
    ):
        raise ValueError("Expected sealed source-only ready state")
    required = {
        "registration.json",
        "roles.npz",
        "targets-and-metadata.npz",
        "frozen-source-selections.json",
        "frozen-temporal-control.npz",
        "raw_current-readout.npz",
        "raw_current-rff.npz",
    }
    required.update(
        arm + suffix
        for arm in ENCODERS
        for suffix in ("-embedding.npy", "-final.pt", "-rff.npz", "-readout.npz")
    )
    if not required <= set(ready["artifacts_sha256"]):
        raise ValueError(
            "Source-only seal lacks required encoder/map/readout artifacts"
        )
    verify_hashes(source, ready["artifacts_sha256"])
    registration = json.loads((source / "registration.json").read_text())
    if (
        registration["config"]["held_room"] != "E02"
        or registration["cache_receipt_sha256"] != digest(cache / "receipt.json")
        or digest(source / "targets-and-metadata.npz")
        != digest(cache / "targets-and-metadata.npz")
    ):
        raise ValueError("Source/cache protocol or metadata differs")
    return registration


class Features:
    def __init__(self, raw, *, current=False):
        self.raw, self.current = raw, current

    def __getitem__(self, ids):
        # Validate stored real coordinates BEFORE a cast or covariance decoder
        # can discard an imaginary component and conceal an invalid artifact.
        x = matrix(self.raw[ids])
        if self.current:
            x = hermitian_coordinates(current_covariances(x)).reshape(len(ids), 1368)
        return matrix(x)


def inputs(source, cache, expected_digest):
    # Validate the complete source-only seal BEFORE loading target-reference GT.
    source_contract(source, cache, expected_digest)
    receipt, data, cached = load_inputs(cache)
    train, validation, held = fold_masks(data["room"], data["person"], "E02")
    with np.load(source / "roles.npz", allow_pickle=False) as roles:
        for name, mask in (
            ("train", train),
            ("validation", validation),
            ("held", held),
        ):
            np.testing.assert_array_equal(roles[name], mask)
    reference, query = reference_indices(data, "E02"), query_indices(data, "E02")
    validate_roles(data, reference, query)
    features, bases = {}, {}
    for arm in ENCODERS:
        features[arm] = Features(
            np.load(
                source / (arm + "-embedding.npy"), mmap_mode="r", allow_pickle=False
            )
        )
        if features[arm].raw.shape != (len(data["pose"]), 512):
            raise ValueError("Frozen embedding shape differs")
    features["raw_current"] = Features(cached["untreated_covariance"], current=True)
    features["frozen_temporal"] = Features(cached["congruence_dynamics"])
    for arm in (*ENCODERS, "raw_current"):
        with np.load(source / (arm + "-readout.npz"), allow_pickle=False) as state:
            bases[arm] = state["prediction"]
        if bases[arm].shape != data["pose"].shape or not np.isfinite(bases[arm]).all():
            raise ValueError("Frozen arm predictions are invalid or misaligned")
    with np.load(source / "frozen-temporal-control.npz", allow_pickle=False) as state:
        np.testing.assert_array_equal(state["held_indices"], np.flatnonzero(held))
        matrix(
            state["pose"].reshape(len(state["held_indices"]), -1),
            rows=int(held.sum()),
            width=34,
        )
        bases["frozen_temporal"] = np.zeros_like(data["pose"], dtype=np.float64)
        bases["frozen_temporal"][held] = state["pose"]
    return receipt, data, train, reference, query, features, bases


def fit(source, cache, out, expected_digest):
    if out.exists():
        raise FileExistsError(out)
    if (source / "results.json").exists() or (
        source / "query-permutation.npy"
    ).exists():
        raise ValueError("Freeze calibration choices before new zero-shot query scores")
    _, data, train, reference, query, features, bases = inputs(
        source, cache, expected_digest
    )
    source_mean = data["pose"][train].reshape(train.sum(), 34).mean(0)
    out.mkdir(parents=True)
    sources = [
        Path(__file__),
        ROOT / "tests/test_masked_calibration_probe.py",
        ROOT / "ghostmesh/reference_kernel.py",
        ROOT / "scripts/domain_reference_covariance_probe.py",
        ROOT / "scripts/domain_covariance_cached_probe.py",
        ROOT / "scripts/domain_covariance_full_cache.py",
        ROOT / "ghostmesh/reference_covariance.py",
        ROOT / "ghostmesh/covariance_dynamics.py",
        ROOT / "ghostmesh/metrics.py",
    ]
    registration = dict(
        config=CONFIG,
        source_ready_sha256=expected_digest,
        cache_receipt_sha256=digest(cache / "receipt.json"),
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
        reference_rows=len(reference),
        query_rows=len(query),
        reference_people=sorted(set(data["person"][reference])),
        query_people=sorted(set(data["person"][query])),
        query_metrics_computed=False,
    )
    save_json(out / "registration.json", registration)
    for path in sources:
        target = out / "executed-source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    states, selection = fit_arrays(features, bases, data, reference, query, source_mean)
    for name, state in states.items():
        save_npz(out / (name.replace("/", "--") + ".npz"), **state)
    ends = np.unique(data["end"][train])
    clock = np.stack(
        [data["pose"][train & (data["end"] == end)].mean(0) for end in ends]
    )
    save_npz(
        out / "roles-and-priors.npz",
        reference=reference,
        query=query,
        source_mean=source_mean,
        reference_mean=data["pose"][reference].mean(0),
        clock_ends=ends,
        clock_poses=clock,
    )
    save_json(out / "frozen-reference-choices.json", selection)
    files = {
        str(p.relative_to(out)): digest(p)
        for p in sorted(out.rglob("*"))
        if p.is_file()
    }
    save_json(
        out / "calibration-ready.json",
        dict(
            kind="masked-csi-reference-calibration-ready",
            query_metrics_computed=False,
            artifacts_sha256=files,
        ),
    )
    return out


def score(source, cache, out):
    if (out / "results.json").exists() or (out / "query-permutation.npy").exists():
        raise FileExistsError("Calibration query scoring already started")
    ready = json.loads((out / "calibration-ready.json").read_text())
    if (
        ready["kind"] != "masked-csi-reference-calibration-ready"
        or ready["query_metrics_computed"] is not False
    ):
        raise ValueError("Expected sealed calibration-ready state")
    required = {
        "registration.json",
        "roles-and-priors.npz",
        "frozen-reference-choices.json",
        "reference_mean_mixture.npz",
    }
    required.update(
        arm + "--" + method + ".npz"
        for arm in ARMS
        for method in (
            "residual_krr",
            "absolute_krr",
            "permuted_residual_krr",
            "constant_residual",
        )
    )
    if not required <= set(ready["artifacts_sha256"]):
        raise ValueError("Calibration seal omits required fitted states or choices")
    verify_hashes(out, ready["artifacts_sha256"])
    registration = json.loads((out / "registration.json").read_text())
    verify_hashes(ROOT, registration["source_sha256"])
    _, data, _, reference, query, features, bases = inputs(
        source, cache, registration["source_ready_sha256"]
    )
    with np.load(out / "roles-and-priors.npz", allow_pickle=False) as state:
        priors = {k: state[k] for k in state.files}
    np.testing.assert_array_equal(reference, priors["reference"])
    np.testing.assert_array_equal(query, priors["query"])
    selection = json.loads((out / "frozen-reference-choices.json").read_text())
    predictions = {}
    for arm in ARMS:
        predictions[arm + "/zero_shot"] = bases[arm][query]
        x = features[arm][query]
        for method in (
            "residual_krr",
            "absolute_krr",
            "permuted_residual_krr",
            "constant_residual",
        ):
            tag = arm + "/" + method
            with np.load(
                out / (tag.replace("/", "--") + ".npz"), allow_pickle=False
            ) as state:
                predictions[tag] = predict_reference_model(
                    state, x, bases[arm][query].reshape(len(query), 34)
                ).reshape(-1, 17, 2)
    predictions["source_mean"] = np.broadcast_to(
        priors["source_mean"].reshape(17, 2), (len(query), 17, 2)
    )
    predictions["reference_mean"] = np.broadcast_to(
        priors["reference_mean"], (len(query), 17, 2)
    )
    with np.load(out / "reference_mean_mixture.npz", allow_pickle=False) as state:
        predictions["reference_mean_mixture"] = predict_reference_model(
            state,
            np.ones((len(query), 1)),
            predictions["source_mean"].reshape(len(query), 34),
        ).reshape(-1, 17, 2)
    clock = {
        int(end): pose for end, pose in zip(priors["clock_ends"], priors["clock_poses"])
    }
    if not set(data["end"][query]) <= set(clock):
        raise ValueError("Query endpoint missing from frozen source clock")
    predictions["source_endpoint_clock"] = np.stack(
        [clock[int(end)] for end in data["end"][query]]
    )
    perm = permutation(data["record"][query])
    save_array(out / "query-permutation.npy", perm)
    for name, prediction in predictions.items():
        save_array(
            out / (name.replace("/", "--") + "-query-prediction.npy"),
            prediction,
        )
    results = {
        name: scores(prediction, data, query, perm)
        for name, prediction in predictions.items()
    }
    primary = results[PRIMARY]["aligned"]
    others = [value["aligned"] for name, value in results.items() if name != PRIMARY]
    gate = dict(
        beta_positive=selection[PRIMARY]["selected"]["beta"] > 0,
        nje_strictly_better_all_controls=all(
            primary["normalized_joint_error"] < value["normalized_joint_error"]
            for value in others
        ),
        pck_not_worse_all_controls=all(
            primary["pck_curve"][threshold] >= value["pck_curve"][threshold]
            for value in others
            for threshold in ("0.05", "0.2")
        ),
        motion_positive=primary["record_centered_articulation_r2"] is not None
        and primary["record_centered_articulation_r2"] > 0,
        query_shuffle_worse=primary["normalized_joint_error"]
        < results[PRIMARY]["shuffled"]["normalized_joint_error"],
    )
    reversals = {
        field: {
            person: [
                name
                for name, method in results.items()
                if name != PRIMARY
                and value["normalized_joint_error"]
                >= method["per_" + field][person]["normalized_joint_error"]
            ]
            for person, value in results[PRIMARY]["per_" + field].items()
        }
        for field in ("person", "action")
    }
    save_json(
        out / "results.json",
        dict(
            registration=registration,
            calibration_ready_sha256=digest(out / "calibration-ready.json"),
            frozen_choices_sha256=digest(out / "frozen-reference-choices.json"),
            methods=results,
            reversals=reversals,
            gate=gate,
            passes=all(gate.values()),
        ),
    )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fit", "score"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-ready-sha256")
    args = parser.parse_args()
    if args.mode == "fit":
        if not args.source_ready_sha256:
            parser.error("fit requires the reviewed --source-ready-sha256")
        fit(args.source, args.cache, args.out, args.source_ready_sha256)
    else:
        score(args.source, args.cache, args.out)


if __name__ == "__main__":
    main()
