"""Independent qualification of BODY/FULL calibration-ready states.

No calibration fitter, kernel class, encoder, optimizer or query scoring helper
is imported. General LU solves qualify saved reference coefficients, using
reference-person-held CV only. Query annotations never enter a metric here.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve

ROOT = Path(__file__).resolve().parents[1]
ENCODERS = (
    "random_frozen",
    "masked_only",
    "masked_temporal",
    "masked_shuffled",
    "supervised",
)
METHODS = ("residual_krr", "absolute_krr", "permuted_residual_krr", "constant_residual")
ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0)
BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
ATOL, RTOL = 2e-9, 2e-8
COMMON_PRODUCTION = {
    "ghostmesh/metrics.py",
    "ghostmesh/covariance_dynamics.py",
    "ghostmesh/reference_covariance.py",
    "scripts/domain_covariance_cached_probe.py",
    "scripts/domain_covariance_full_cache.py",
}
BODY_PRODUCTION = COMMON_PRODUCTION | {
    "scripts/domain_masked_calibration_probe.py",
    "ghostmesh/reference_kernel.py",
    "scripts/domain_reference_covariance_probe.py",
}
FULL_SOURCE_PRODUCTION = COMMON_PRODUCTION | {
    "scripts/domain_masked_fullpose_probe.py",
    "scripts/domain_background_probe.py",
    "scripts/domain_covariance_source_probe.py",
}
PRODUCTION_BY_KIND = {
    "masked-csi-reference-calibration-ready": BODY_PRODUCTION,
    "masked-csi-fullimage-calibration-ready": BODY_PRODUCTION
    | {
        "scripts/domain_masked_fullimage_calibration_probe.py",
        "scripts/domain_masked_fullpose_probe.py",
        "scripts/domain_background_probe.py",
    },
    "masked-csi-fullimage-source-ready": FULL_SOURCE_PRODUCTION,
    "masked-csi-source-only-ready": FULL_SOURCE_PRODUCTION
    - {"scripts/domain_masked_fullpose_probe.py"}
    | {
        "scripts/domain_masked_pretrain_probe.py",
        "ghostmesh/masked_csi.py",
        "scripts/domain_reference_covariance_probe.py",
        "ghostmesh/covariance_accelerated.py",
        "ghostmesh/causal_channel.py",
    },
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def real(value, name="numeric artifact", shape=None):
    a = np.asarray(value)
    require(
        a.dtype.kind in "biuf" and a.size > 0 and np.isfinite(a).all(),
        "Expected finite REAL " + name,
    )
    require(shape is None or a.shape == shape, "Unexpected shape: " + name)
    return a


def source_real(value, name="source numeric artifact", shape=None):
    a = real(value, name, shape)
    require(a.dtype.kind in "iuf", "Expected non-Boolean real source array: " + name)
    return a


def close(observed, expected, name, atol=ATOL, rtol=RTOL):
    a, b = real(observed, name), real(expected, name)
    require(
        a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=rtol),
        "Numerical mismatch: " + name,
    )
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def compare(observed, expected, name):
    if isinstance(expected, (str, np.str_)):
        require(
            isinstance(observed, (str, np.str_)) and str(observed) == str(expected),
            "Field mismatch: " + name,
        )
    elif expected is None or isinstance(expected, (bool, np.bool_)):
        require(
            (
                observed is None
                if expected is None
                else isinstance(observed, (bool, np.bool_))
                and bool(observed) == bool(expected)
            ),
            "Field mismatch: " + name,
        )
    elif isinstance(expected, dict):
        require(
            isinstance(observed, dict) and set(observed) == set(expected),
            "Field keys differ: " + name,
        )
        for key in expected:
            compare(observed[key], expected[key], name + "/" + str(key))
    elif isinstance(expected, (list, tuple)):
        require(
            isinstance(observed, (list, tuple)) and len(observed) == len(expected),
            "Sequence differs: " + name,
        )
        for i, value in enumerate(expected):
            compare(observed[i], value, name + "/" + str(i))
    else:
        close(observed, expected, name)


def digest(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def hashes(directory, manifest):
    require(isinstance(manifest, dict) and bool(manifest), "Empty hash manifest")
    for name, expected in manifest.items():
        p = Path(name)
        require(not p.is_absolute() and ".." not in p.parts, "Unsafe manifest path")
        require(
            digest(directory / p) == expected, "Hash mismatch: " + str(directory / p)
        )


def npz(path):
    with np.load(path, allow_pickle=False) as saved:
        result = {key: saved[key] for key in saved.files}
    for key, value in result.items():
        if value.dtype.kind not in "US":
            real(value, str(path) + "/" + key)
    return result


def mm(a, b):
    # Explicit einsum is independent of the fitter's BLAS/Cholesky path and
    # avoids observed spurious Accelerate matmul floating-point warnings.
    a, b = real(a), real(b)
    return real(np.einsum("ij,jk->ik", a, b, optimize=False), "matrix product")


def distances(a, b):
    a, b = real(a), real(b)
    require(
        a.ndim == b.ndim == 2 and a.shape[1] == b.shape[1], "Distance dimensions differ"
    )
    return real(np.array([np.square(row - b).sum(1) for row in a]), "squared distances")


def reference_kernel(x):
    x = real(x, "reference feature").astype(np.float64, copy=False)
    require(x.ndim == 2 and len(x) >= 2, "Reference matrix required")
    mean, sd = x.mean(0), x.std(0)
    real(mean, "reference mean")
    real(sd, "reference SD")
    floored = sd <= 0.001
    if floored.all():
        return None
    scale = np.maximum(sd, 0.001)
    design = (x - mean) / scale
    width = float(np.median(distances(design, design)[np.triu_indices(len(x), 1)]))
    if width <= 0:
        return None
    real(width, "reference bandwidth")
    return dict(
        kind=np.asarray("reference_residual_rbf_v1"),
        input_mean=mean,
        input_scale=scale,
        reference_design=design,
        bandwidth_squared=np.asarray(width),
        scale_floor=np.asarray(0.001),
        floored_columns=floored,
    )


def cross_kernel(x, kernel):
    x = real(x, "kernel evaluation feature")
    design = (x - kernel["input_mean"]) / kernel["input_scale"]
    with np.errstate(under="ignore"):
        return np.exp(
            -distances(design, kernel["reference_design"])
            / (2 * float(kernel["bandwidth_squared"]))
        )


def lu_solution(kernel, residual, alpha):
    residual = real(residual, "reference residual")
    with np.errstate(under="ignore"):
        k = np.exp(
            -distances(kernel["reference_design"], kernel["reference_design"])
            / (2 * float(kernel["bandwidth_squared"]))
        )
    system = k + len(k) * alpha * np.eye(len(k))
    dual = real(solve(system, residual, assume_a="gen", check_finite=True), "LU dual")
    error = mm(system, dual) - residual
    relative = (
        float(np.linalg.norm(error) / max(np.linalg.norm(residual), 1e-300))
        if np.any(residual)
        else float(np.linalg.norm(error))
    )
    require(np.isfinite(relative) and relative < 1e-8, "Independent LU equation failed")
    return dual, relative, system


def reference_nje(prediction, target):
    prediction, target = (
        real(prediction, "reference prediction"),
        real(target, "reference target"),
    )
    prediction, target = (
        prediction.astype(np.float64, copy=False),
        target.astype(np.float64, copy=False),
    )
    require(
        target.ndim == 2 and target.shape[1] == 34 and prediction.shape == target.shape,
        "Reference pose dimensions differ",
    )
    y, p = target.reshape(-1, 17, 2), prediction.reshape(-1, 17, 2)
    diagonal = np.sqrt(np.square(y.max(1) - y.min(1)).sum(1))
    require((diagonal > 1e-6).all(), "Degenerate reference pose")
    return float((np.sqrt(np.square(p - y).sum(2)) / diagonal[:, None]).mean())


def person_folds(people):
    people = np.asarray(people)
    require(
        people.shape == (120,) and set(people) == {"S11", "S12"},
        "Exactly120 S11/S12 references required",
    )
    require(
        all(np.count_nonzero(people == name) == 60 for name in ("S11", "S12")),
        "CV must split60/60 by reference person",
    )
    return [(people == name, people != name) for name in ("S11", "S12")]


def reference_permutation(records):
    records = np.asarray(records)
    require(records.shape == (120,), "Reference permutation length differs")
    rng, perm = np.random.default_rng(2027), np.arange(120)
    for name in sorted(set(records)):
        ids = np.flatnonzero(records == name)
        perm[ids] = rng.permutation(ids)
    require(
        np.array_equal(records[perm], records)
        and np.array_equal(np.sort(perm), np.arange(120)),
        "Reference permutation crosses records",
    )
    return perm


def reconstruct_reference(x, labels, base, people, records, method, source_mean):
    """Independent reference-only CV and state reconstruction, not a disk fitter."""
    x, labels, base = (
        real(x, "reference features"),
        real(labels, "reference labels", (120, 34)),
        real(base, "reference f0", (120, 34)),
    )
    x, labels, base = (
        value.astype(np.float64, copy=False) for value in (x, labels, base)
    )
    require(
        x.ndim == 2 and len(x) == 120 and method in METHODS,
        "Reference method or feature shape differs",
    )
    require(np.asarray(records).shape == (120,), "Reference records differ")
    people, records = np.asarray(people), np.asarray(records)
    labels = (
        labels[reference_permutation(records)]
        if method == "permuted_residual_krr"
        else labels
    )
    source_mean = real(source_mean, "source GT mean", (34,))
    source_mean = source_mean.astype(np.float64, copy=False)
    if method == "absolute_krr":
        base = np.broadcast_to(source_mean, base.shape)
    diagnostics, folds = [], []
    for train, validation in person_folds(people):
        kernel = None if method == "constant_residual" else reference_kernel(x[train])
        folds.append((train, validation, kernel))
        diagnostics.append(
            dict(
                fitting_people=sorted(set(people[train])),
                validation_people=sorted(set(people[validation])),
                fitting_rows=int(train.sum()),
                validation_rows=int(validation.sum()),
                input_mean=kernel["input_mean"].tolist()
                if kernel is not None
                else None,
                input_scale=kernel["input_scale"].tolist()
                if kernel is not None
                else None,
                bandwidth_squared=float(kernel["bandwidth_squared"])
                if kernel is not None
                else None,
            )
        )
    grid = []
    for alpha in (0.0,) if method == "constant_residual" else ALPHAS:
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
                dual, residual, _ = lu_solution(
                    kernel, labels[train] - base[train], alpha
                )
                correction = mm(cross_kernel(x[validation], kernel), dual)
                residuals.append(residual)
            corrections.append(correction)
        for beta in BETAS:
            available = beta == 0 or all(c is not None for c in corrections)
            values = (
                [
                    reference_nje(
                        base[v] if beta == 0 else base[v] + beta * c, labels[v]
                    )
                    for (_, v, _), c in zip(folds, corrections)
                ]
                if available
                else None
            )
            grid.append(
                dict(
                    alpha=float(alpha),
                    beta=beta,
                    available=available,
                    reference_person_cv_nje=float(np.mean(values))
                    if available
                    else None,
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
        kernel = reference_kernel(x)
        require(kernel is not None, "Undefined final reference kernel")
        dual, residual, _ = lu_solution(kernel, labels - base, selected["alpha"])
        state.update(
            kernel,
            correction_kind=np.asarray("kernel"),
            dual=dual,
            final_equation_residual=np.asarray(residual),
        )
    return state, dict(selected=selected, grid=grid, folds=diagnostics), labels, base


def scalar_text(value, name):
    a = np.asarray(value)
    require(a.shape == () and a.dtype.kind in "US", "Expected text scalar: " + name)
    return str(a)


def state_schema(state):
    """Enforce the prediction parser's real, scalar and Boolean contracts."""
    method = scalar_text(state["method"], "method")
    base_kind = scalar_text(state["base_kind"], "base_kind")
    correction_kind = scalar_text(state["correction_kind"], "correction_kind")
    require(
        method in METHODS
        and base_kind in {"frozen_arm", "source_mean"}
        and correction_kind in {"zero", "constant", "kernel"},
        "Unknown calibration state kind",
    )
    beta = float(real(state["beta"], "beta scalar", ()))
    alpha = float(real(state["alpha"], "alpha scalar", ()))
    require(beta in BETAS, "Invalid exact calibration beta")
    require(
        alpha in ((0.0,) if method == "constant_residual" else ALPHAS),
        "Invalid exact calibration alpha",
    )
    if base_kind == "source_mean":
        real(state["source_mean"], "source mean vector", (34,))
    if correction_kind == "constant":
        real(state["correction"], "constant correction vector", (34,))
    if correction_kind != "kernel":
        return
    require(
        scalar_text(state["kind"], "kind") == "reference_residual_rbf_v1",
        "Invalid reference kernel kind",
    )
    design = real(state["reference_design"], "reference design")
    require(
        design.ndim == 2 and len(design) >= 2 and design.shape[1] > 0,
        "Invalid reference design shape",
    )
    mean = real(state["input_mean"], "reference input mean", (design.shape[1],))
    scale = real(state["input_scale"], "reference input scale", mean.shape)
    floored = np.asarray(state["floored_columns"])
    require(
        floored.dtype == np.dtype(bool) and floored.shape == mean.shape,
        "floored_columns must be an exact Boolean vector",
    )
    floor = float(real(state["scale_floor"], "scale floor scalar", ()))
    width = float(real(state["bandwidth_squared"], "bandwidth scalar", ()))
    require(floor > 0 and width > 0, "Kernel floor and bandwidth must be positive")
    require(
        not floored.all()
        and (scale >= floor).all()
        and (scale[floored] == floor).all()
        and (scale[~floored] > floor).all(),
        "Invalid exact reference floor/scale state",
    )
    real(state["dual"], "reference dual", (len(design), 34))
    real(state["final_equation_residual"], "final residual scalar", ())


def audit_reference_state(
    x, labels, base, people, records, method, source_mean, saved, choices
):
    expected, report, effective_labels, effective_base = reconstruct_reference(
        x, labels, base, people, records, method, source_mean
    )
    for row in choices["grid"]:
        for residual in row["normal_equation_residuals"]:
            if residual is not None:
                require(
                    0 <= float(real(residual, "CV equation residual", ())) < 1e-8,
                    "CV equation residual diagnostic failed",
                )
    compare(choices, report, "reference CV")
    require(
        set(saved) == set(expected),
        "Saved state fields differ; no unregistered intercept permitted",
    )
    state_schema(saved)
    maximum = 0.0
    for key, value in expected.items():
        if key == "floored_columns":
            require(
                np.array_equal(saved[key], value),
                "floored_columns Boolean values differ",
            )
        elif np.asarray(value).dtype.kind in "US":
            require(
                scalar_text(saved[key], key) == str(value), "Saved method/kind differs"
            )
        else:
            maximum = max(maximum, close(saved[key], value, "state/" + key))
    residual, far_error = None, 0.0
    if str(expected["correction_kind"]) == "kernel":
        _, _, system = lu_solution(
            expected, effective_labels - effective_base, float(expected["alpha"])
        )
        delta = mm(system, saved["dual"]) - (effective_labels - effective_base)
        residual = float(
            np.linalg.norm(delta)
            / max(np.linalg.norm(effective_labels - effective_base), 1e-300)
        )
        require(
            np.isfinite(residual) and residual < 1e-8,
            "Saved reference dual equation failed",
        )
        require(
            0 <= float(real(saved["final_equation_residual"])) < 1e-8,
            "Saved reference residual diagnostic failed",
        )
        far = expected["input_mean"][None] + expected["input_scale"][None] * 1e8
        correction = mm(cross_kernel(far, saved), saved["dual"])
        far_error = close(
            correction, np.zeros_like(correction), "far-reference correction", 0, 0
        )
    return dict(
        alpha=float(expected["alpha"]),
        beta=float(expected["beta"]),
        correction_kind=str(expected["correction_kind"]),
        maximum_state_error=maximum,
        normal_equation_residual=residual,
        far_reference_correction_error=far_error,
        cv_candidates=len(report["grid"]),
    )


def validate_reference_query(data, reference, query):
    n = len(data["record"])
    for name, ids in (("reference", reference), ("query", query)):
        require(
            ids.ndim == 1
            and ids.dtype.kind in "iu"
            and len(ids) > 0
            and len(np.unique(ids)) == len(ids)
            and (ids >= 0).all()
            and (ids < n).all(),
            "Invalid " + name + " indices",
        )
    require(not (set(reference) & set(query)), "Reference/query rows overlap")
    for field in ("person", "record"):
        require(
            not (set(data[field][reference]) & set(data[field][query])),
            "Reference/query " + field + " overlap",
        )
    person_folds(data["person"][reference])
    require(
        set(data["person"][query]) == {f"S{i:02}" for i in range(13, 21)},
        "Query people differ",
    )
    ends = [end for first in range(39, 256, 24) for end in (first, first + 8)]
    expected = np.flatnonzero(
        (data["room"] == "E02")
        & np.isin(data["person"], ["S11", "S12"])
        & np.isin(data["action"], ["A03", "A07", "A12"])
        & np.isin(data["end"], ends)
    )
    close(reference, expected, "canonical reference IDs", 0, 0)
    close(
        query,
        np.flatnonzero(
            (data["room"] == "E02") & ~np.isin(data["person"], ["S11", "S12"])
        ),
        "canonical query IDs",
        0,
        0,
    )
    require(
        len(set(zip(data["record"][reference], data["end"][reference]))) == 120,
        "Reference endpoint duplicates",
    )
    require(
        all(
            np.count_nonzero(data["record"][reference] == record) == 20
            for record in set(data["record"][reference])
        ),
        "Reference records need20 endpoints each",
    )


def sealed(directory, filename, kind, required, flag):
    ready = read(directory / filename)
    require(ready["kind"] == kind and ready[flag] is False, "Wrong ready stage")
    require(
        required <= set(ready["artifacts_sha256"]),
        "Ready seal omits required artifacts",
    )
    hashes(directory, ready["artifacts_sha256"])
    registration = read(directory / "registration.json")
    production = PRODUCTION_BY_KIND.get(kind)
    require(
        production is not None and production <= set(registration["source_sha256"]),
        "Source provenance omits required production dependencies",
    )
    require(
        {"executed-source/" + name for name in production}
        <= set(ready["artifacts_sha256"]),
        "Ready seal omits executed production dependencies",
    )
    hashes(directory / "executed-source", registration["source_sha256"])
    return ready, registration


def feature_rows(source, cache, arm, ids, n):
    if arm in ENCODERS:
        array = np.load(
            source / (arm + "-embedding.npy"), mmap_mode="r", allow_pickle=False
        )
        require(array.shape == (n, 512), "Embedding shape differs")
        return real(array[ids], "reference embedding").astype(np.float64)
    filename, width = (
        ("untreated_covariance-features.npy", 9576)
        if arm == "raw_current"
        else ("congruence_dynamics-features.npy", 2736)
    )
    array = np.load(cache / filename, mmap_mode="r", allow_pickle=False)
    require(array.shape == (26730, width), "Cached allocation differs")
    x = real(array[ids], "reference covariance coordinates").astype(np.float64)
    return (
        x.reshape(-1, 8, 19, 7, 9)[:, :, :, 0].reshape(len(ids), 1368)
        if arm == "raw_current"
        else x
    )


def frozen_design(feature, mapping):
    x = source_real(feature, "frozen feature")
    require(x.ndim == 2, "Frozen feature matrix required")
    weight = source_real(mapping["weight"], "frozen RFF weight")
    require(
        weight.ndim == 2 and weight.shape[0] == x.shape[1],
        "Frozen RFF weight shape differs",
    )
    d = weight.shape[1]
    require(d == 512, "Frozen source map must be RFF512")
    mean, scale = (
        source_real(mapping["feature_mean"], shape=(x.shape[1],)),
        source_real(mapping["feature_scale"], shape=(x.shape[1],)),
    )
    require((scale > 0).all(), "Nonpositive frozen feature scale")
    offset = source_real(mapping["offset"], shape=(d,))
    rmean, rscale = (
        source_real(mapping["rff_mean"], shape=(d,)),
        source_real(mapping["rff_scale"], shape=(d,)),
    )
    require((rscale > 0).all(), "Nonpositive frozen RFF scale")
    return (
        np.sqrt(2 / d) * np.cos(mm((x - mean) / scale, mapping["weight"]) + offset)
        - rmean
    ) / rscale


def source_base_reference(source, full_source, arm, feature, reference, held, mode):
    if arm == "frozen_temporal":
        old = npz(source / "frozen-temporal-control.npz")
        close(
            old["held_indices"], np.flatnonzero(held), "frozen temporal held IDs", 0, 0
        )
        positions = np.searchsorted(old["held_indices"], reference)
        return real(
            old["pose"][positions], "temporal reference f0", (120, 17, 2)
        ).reshape(120, 34), 0.0
    z = frozen_design(feature, npz(source / (arm + "-rff.npz")))
    state = npz(source / (arm + "-readout.npz"))
    source_real(state["x_mean"], "body design mean", (512,))
    source_real(state["y_mean"], "body target mean", (34,))
    source_real(state["weights"], "body five-alpha weights", (5, 512, 34))
    source_real(state["prediction"], "body predictions", (len(held), 17, 2))
    selected = real(state["selected"], "source selected", ())
    require(
        selected.dtype.kind in "iu" and int(selected) in range(5),
        "Source alpha selection differs",
    )
    body = (
        mm(z - state["x_mean"], state["weights"][int(selected)]) + state["y_mean"]
    ).reshape(120, 17, 2)
    error = close(state["prediction"][reference], body, "frozen reference body f0")
    if mode == "body":
        return body.reshape(120, 34), error
    root = npz(full_source / (arm + "-root-state.npz"))
    source_real(root["x_mean"], "root design mean", (512,))
    source_real(root["root_mean"], "root target mean", (2,))
    source_real(root["weights"], "root five-alpha weights", (5, 512, 2))
    close(root["x_mean"], state["x_mean"], "root/source design mean")
    index = real(root["selected"], "root selected", ())
    require(
        index.dtype.kind in "iu"
        and int(index) in range(5)
        and float(source_real(root["beta"], shape=())) in BETAS,
        "Root selection differs",
    )
    hip = root["root_mean"] + float(root["beta"]) * mm(
        z - root["x_mean"], root["weights"][int(index)]
    )
    full = body + hip[:, None]
    saved = np.load(
        full_source / (arm + "-full-prediction.npy"), mmap_mode="r", allow_pickle=False
    )
    error = max(error, close(saved[reference], full, "frozen full-image reference f0"))
    return full.reshape(120, 34), error


def verify(
    mode, source, cache, raw, full_source, experiment, output, expected_ready_sha256
):
    require(not output.exists(), "Output already exists")
    require(mode in ("body", "full"), "Unknown audit mode")
    require(
        not (experiment / "results.json").exists()
        and not (experiment / "query-permutation.npy").exists(),
        "Query scoring already started; fitted-state audit is prospective only",
    )
    own_hash = digest(Path(__file__))
    arms = ENCODERS + (
        ("raw_current", "frozen_temporal") if mode == "body" else ("raw_current",)
    )
    states = {arm + "/" + method for arm in arms for method in METHODS} | {
        "reference_mean_mixture"
    }
    required = {
        "registration.json",
        "roles-and-priors.npz",
        "frozen-reference-choices.json",
    } | {tag.replace("/", "--") + ".npz" for tag in states}
    filename = (
        "calibration-ready.json"
        if mode == "body"
        else "fullimage-calibration-ready.json"
    )
    require(
        isinstance(expected_ready_sha256, str)
        and len(expected_ready_sha256) == 64
        and digest(experiment / filename) == expected_ready_sha256,
        "Reviewed calibration-ready hash differs",
    )
    kind = (
        "masked-csi-reference-calibration-ready"
        if mode == "body"
        else "masked-csi-fullimage-calibration-ready"
    )
    ready, registration = sealed(
        experiment, filename, kind, required, "query_metrics_computed"
    )
    require(
        registration["query_metrics_computed"] is False
        and registration["config"]["room"] == "E02",
        "Calibration registration differs",
    )
    compare(registration["config"]["alphas"], list(ALPHAS), "registered alphas")
    compare(registration["config"]["betas"], list(BETAS), "registered betas")
    source_required = {
        "registration.json",
        "roles.npz",
        "targets-and-metadata.npz",
        "frozen-source-selections.json",
        "frozen-temporal-control.npz",
        "raw_current-readout.npz",
        "raw_current-rff.npz",
    }
    source_required.update(
        arm + suffix
        for arm in ENCODERS
        for suffix in ("-embedding.npy", "-final.pt", "-rff.npz", "-readout.npz")
    )
    source_ready, sr = sealed(
        source,
        "source-ready.json",
        "masked-csi-source-only-ready",
        source_required,
        "held_metrics_computed",
    )
    require(
        digest(source / "source-ready.json") == registration["source_ready_sha256"]
        and sr["config"]["held_room"] == "E02",
        "Source-ready identity differs",
    )
    cr, rr = read(cache / "receipt.json"), read(raw / "receipt.json")
    require(
        digest(cache / "receipt.json")
        == registration["cache_receipt_sha256"]
        == sr["cache_receipt_sha256"],
        "Cache receipt differs",
    )
    require(
        digest(raw / "receipt.json")
        == sr["raw_receipt_sha256"]
        == cr["raw_receipt_sha256"],
        "Raw receipt differs",
    )
    hashes(cache, cr["arrays_sha256"])
    hashes(raw, {name + ".npy": rr["arrays"][name] for name in ("pose", "valid")})
    require(
        digest(source / "targets-and-metadata.npz")
        == cr["arrays_sha256"]["targets-and-metadata.npz"],
        "Source target metadata differs",
    )
    data, roles = npz(source / "targets-and-metadata.npz"), npz(source / "roles.npz")
    n = len(data["end"])
    require(
        set(data["room"]) == {"E01", "E02", "E03"} and n == cr["logical_rows"],
        "Canonical source rows differ",
    )
    train = (data["room"] != "E02") & ~np.isin(
        data["person"], ["S04", "S09", "S14", "S19", "S24", "S29"]
    )
    held = data["room"] == "E02"
    validation = ~(train | held)
    for name, mask in (("train", train), ("validation", validation), ("held", held)):
        close(roles[name], mask, "source role/" + name, 0, 0)
    record_offsets = {
        row["id"]: (row["offset"], row["frames"]) for row in rr["records"]
    }
    expected_records = [
        f"E{room:02}/S{person:02}/A{action:02}"
        for room in range(1, 4)
        for person in range((room - 1) * 10 + 1, room * 10 + 1)
        for action in range(1, 28)
    ]
    require(
        [row["id"] for row in rr["records"]] == expected_records
        and all(
            row["frames"] == 297 and row["offset"] == i * 297
            for i, row in enumerate(rr["records"])
        ),
        "Raw records differ",
    )
    require(data["end"].dtype.kind in "iu", "Canonical endpoints must be integers")
    for record, end in zip(data["record"], data["end"]):
        require(
            record in record_offsets and 0 <= end < record_offsets[record][1],
            "Endpoint outside own record",
        )
    require(
        all(
            str(record) == f"{room}/{person}/{action}"
            for record, room, person, action in zip(
                data["record"], data["room"], data["person"], data["action"]
            )
        ),
        "Canonical record identity fields disagree",
    )
    raw_indices = np.array(
        [
            record_offsets[record][0] + end
            for record, end in zip(data["record"], data["end"])
        ],
        dtype=np.int64,
    )
    raw_pose = np.load(raw / "pose.npy", mmap_mode="r", allow_pickle=False)
    target = real(raw_pose[raw_indices], "full-image target", (n, 17, 2)).astype(
        np.float64
    )
    valid = np.load(raw / "valid.npy", mmap_mode="r", allow_pickle=False)
    require(valid.dtype == bool and valid[raw_indices].all(), "Invalid target joints")
    close(
        data["pose"],
        target - target[:, [11, 12]].mean(1, keepdims=True),
        "raw/body join",
        1e-12,
        0,
    )
    full_ready = None
    if mode == "full":
        require(full_source is not None, "Full mode requires --full-source")
        full_required = {
            "registration.json",
            "roles.npz",
            "canonical-metadata.npz",
            "full-targets.npy",
            "raw-frame-indices.npy",
            "frozen-fullpose-selections.json",
        }
        full_required.update(arm + "-root-state.npz" for arm in arms)
        full_required.update(
            arm + "-full-prediction.npy"
            for arm in (*arms, "source_mean", "source_endpoint_clock")
        )
        full_ready, fr = sealed(
            full_source,
            "fullpose-source-ready.json",
            "masked-csi-fullimage-source-ready",
            full_required,
            "held_metrics_computed",
        )
        require(
            digest(full_source / "fullpose-source-ready.json")
            == registration["fullpose_source_ready_sha256"],
            "Full source-ready identity differs",
        )
        for key, expected in (
            ("source_ready_sha256", digest(source / "source-ready.json")),
            ("raw_receipt_sha256", digest(raw / "receipt.json")),
            ("cache_receipt_sha256", digest(cache / "receipt.json")),
        ):
            require(
                fr[key] == full_ready[key] == expected,
                "Full source lineage differs: " + key,
            )
        require(
            registration["config"]["target_mode"] == "full_image_2d"
            and registration["raw_receipt_sha256"] == digest(raw / "receipt.json"),
            "Full calibration target lineage differs",
        )
        metadata = npz(full_source / "canonical-metadata.npz")
        for key in metadata:
            require(np.array_equal(metadata[key], data[key]), "Full metadata differs")
        full_roles = npz(full_source / "roles.npz")
        for key in roles:
            close(full_roles[key], roles[key], "full source roles", 0, 0)
        close(
            np.load(full_source / "raw-frame-indices.npy", allow_pickle=False),
            raw_indices,
            "full raw frame IDs",
            0,
            0,
        )
        close(
            np.load(full_source / "full-targets.npy", allow_pickle=False),
            target,
            "full target join",
            0,
            0,
        )
    else:
        target = real(data["pose"], "body target")
    priors = npz(experiment / "roles-and-priors.npz")
    reference, query = priors["reference"], priors["query"]
    validate_reference_query(data, reference, query)
    require(
        len(reference) == registration["reference_rows"] == 120
        and len(query) == registration["query_rows"] == 7111,
        "Role counts differ",
    )
    require(
        registration["reference_people"] == ["S11", "S12"]
        and registration["query_people"] == sorted(set(data["person"][query])),
        "Registered people differ",
    )
    source_mean = target[train].reshape(train.sum(), 34).mean(0)
    close(priors["source_mean"], source_mean, "source GT mean", 0, 0)
    close(
        priors["reference_mean"], target[reference].mean(0), "reference GT mean", 0, 0
    )
    ends = np.unique(data["end"][train])
    clock = np.stack([target[train & (data["end"] == end)].mean(0) for end in ends])
    close(priors["clock_ends"], ends, "source clock endpoints", 0, 0)
    close(priors["clock_poses"], clock, "source clock poses", 0, 0)
    choices = read(experiment / "frozen-reference-choices.json")
    require(set(choices) == states, "Calibration choice methods differ")
    results, f0_errors = {}, {}
    labels = target[reference].reshape(120, 34)
    people, records = data["person"][reference], data["record"][reference]
    for arm in arms:
        if mode == "full":
            root_state = npz(full_source / (arm + "-root-state.npz"))
            close(
                root_state["root_mean"],
                target[train][:, [11, 12]].mean(1).mean(0),
                "source TRAIN root mean",
                0,
                0,
            )
        x = feature_rows(source, cache, arm, reference, n)
        base, f0_errors[arm] = source_base_reference(
            source, full_source, arm, x, reference, held, mode
        )
        for method in METHODS:
            tag = arm + "/" + method
            results[tag] = audit_reference_state(
                x,
                labels,
                base,
                people,
                records,
                method,
                source_mean,
                npz(experiment / (tag.replace("/", "--") + ".npz")),
                choices[tag],
            )
    tag = "reference_mean_mixture"
    results[tag] = audit_reference_state(
        np.ones((120, 1)),
        labels,
        np.broadcast_to(source_mean, labels.shape),
        people,
        records,
        "constant_residual",
        source_mean,
        npz(experiment / (tag + ".npz")),
        choices[tag],
    )
    hashes(experiment, ready["artifacts_sha256"])
    hashes(source, source_ready["artifacts_sha256"])
    if full_ready is not None:
        hashes(full_source, full_ready["artifacts_sha256"])
    require(
        digest(Path(__file__)) == own_hash, "Auditor source changed during execution"
    )
    report = dict(
        kind="independent-masked-calibration-fitted-state-audit",
        mode=mode,
        auditor_sha256=own_hash,
        calibration_ready_sha256=digest(experiment / filename),
        source_ready_sha256=digest(source / "source-ready.json"),
        query_metrics_computed=False,
        model_training_performed=False,
        reference_rows=120,
        query_rows=len(query),
        fitted_states=len(results),
        states=results,
        maximum_reference_f0_errors=f0_errors,
        scope="Independent reference-only CV/normalizers/kernels/LU equations/coefficients and frozen priors; no query GT scoring or kernel-model refit artifact written",
        limits="Source f0 reference predictions are reconstructed from frozen coefficients; this audit does not duplicate full source-root training/validation qualification",
        atol=ATOL,
        rtol=RTOL,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("body", "full"), required=True)
    for name in ("source", "cache", "raw", "experiment", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--full-source", type=Path)
    parser.add_argument("--expected-ready-sha256", required=True)
    args = parser.parse_args()
    verify(
        args.mode,
        args.source,
        args.cache,
        args.raw,
        args.full_source,
        args.experiment,
        args.output,
        args.expected_ready_sha256,
    )


if __name__ == "__main__":
    main()
