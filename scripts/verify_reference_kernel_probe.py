"""Audit saved residual-KRR artifacts without changing any fitted state.

Reconstruct source-only maps, audit saved duals with independent dense solves,
and replay saved predictions, nested choices and metrics. Boundary hashes verify
the saved registration record; hashes alone cannot establish execution chronology.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import solve
from scipy.linalg.blas import dgemm, dnrm2
from scipy.spatial.distance import cdist, pdist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import domain_reference_kernel_probe as study  # noqa: E402
from scripts.domain_covariance_full_cache import verify_hashes  # noqa: E402


def read_json(path):
    return json.loads(path.read_text())


def load_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same_array(actual, expected, message):
    require(np.array_equal(actual, expected), message)


def same_tree(actual, expected, path="result"):
    """Compare complete JSON trees, allowing only roundoff in numeric metrics."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), path)
        for key in expected:
            same_tree(actual[key], expected[key], path + "/" + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), path)
        for index, value in enumerate(expected):
            same_tree(actual[index], value, path + "/" + str(index))
    elif isinstance(expected, float):
        require(
            isinstance(actual, (float, int))
            and np.isfinite(actual)
            and np.isclose(actual, expected, rtol=1e-10, atol=1e-12),
            path,
        )
    else:
        require(actual == expected, path)


def audit_duals(
    reference_kernel, targets, base_reference, source_mean, permutation, state
):
    """Audit SAVED coefficients, not coefficients reconstructed by the fitter."""
    n = len(reference_kernel)
    labels = np.asarray(targets, dtype=np.float64).reshape(n, 34)
    base = np.asarray(base_reference, dtype=np.float64).reshape(n, 34)
    rhs = {
        "residual_krr": labels - base,
        "absolute_krr": labels - np.asarray(source_mean).reshape(34),
        "permuted_residual_krr": labels[permutation] - base,
    }
    maximum_residual = maximum_difference = 0.0
    residuals = {}
    for method in study.KERNEL_METHODS:
        coefficients = np.asarray(state[method])
        require(
            coefficients.shape == (len(study.CONFIG["alpha"]), n, 34)
            and np.isfinite(coefficients).all(),
            "Invalid saved dual shape/values: " + method,
        )
        residuals[method] = []
        for alpha, dual in zip(study.CONFIG["alpha"], coefficients):
            system = reference_kernel + n * alpha * np.eye(n)
            residual = dgemm(1.0, system, dual) - rhs[method]
            denominator = dnrm2(rhs[method].ravel())
            relative = (
                float(dnrm2(residual.ravel()) / denominator)
                if denominator
                else float(dnrm2(residual.ravel()))
            )
            # General LU solve is independent of the fitter's Cholesky solve.
            direct = solve(system, rhs[method], assume_a="gen", check_finite=True)
            difference = float(
                dnrm2((direct - dual).ravel()) / max(dnrm2(direct.ravel()), 1e-300)
            )
            require(
                np.isfinite(relative + difference)
                and relative <= 1e-8
                and difference <= 1e-8,
                "Saved dual fails independent normal equation/direct solve: " + method,
            )
            maximum_residual = max(maximum_residual, relative)
            maximum_difference = max(maximum_difference, difference)
            residuals[method].append(relative)
    same_array(
        state["constant_residual"], (labels - base).mean(0), "Constant residual differs"
    )
    same_array(
        state["mean_mixture"],
        labels.mean(0) - np.asarray(source_mean).reshape(34),
        "Mean mixture differs",
    )
    return maximum_residual, maximum_difference, residuals


def reference_state(raw, indices, saved):
    """Rebuild mean, SD, all-pair median and RBF directly from reference rows."""
    x = np.asarray(raw[indices], dtype=np.float64)
    mean, std = x.mean(0), x.std(0)
    floor = 0.001
    require(not np.all(std <= floor), "All reference columns are near constant")
    scale = np.maximum(std, floor)
    design = (x - mean) / scale
    same_array(saved["input_mean"], mean, "Reference-only feature means differ")
    same_array(saved["input_scale"], scale, "Reference-only feature scales differ")
    same_array(
        saved["floored_columns"], std <= floor, "Reference floor disclosure differs"
    )
    same_array(saved["reference_design"], design, "Normalized references differ")
    require(float(saved["scale_floor"]) == floor, "Reference scale floor differs")
    width = float(np.median(pdist(design, metric="sqeuclidean")))
    require(np.isfinite(width) and width > 0, "Invalid reference bandwidth")
    require(
        float(saved["bandwidth_squared"]) == width, "All-pair reference median differs"
    )
    study.ReferenceRBFKernel.from_state(saved)
    return dict(mean=mean, scale=scale, design=design, width=width), np.exp(
        (-0.5 * cdist(design, design, "sqeuclidean")) / width
    )


def cross_kernel(raw, indices, reference):
    query = (
        np.asarray(raw[indices], dtype=np.float64) - reference["mean"]
    ) / reference["scale"]
    return np.exp(
        (-0.5 * cdist(query, reference["design"], "sqeuclidean")) / reference["width"]
    )


def query_permutation(data, indices):
    permutation = np.arange(len(indices))
    rng = np.random.default_rng(90210)
    for record in sorted(set(data["record"][indices])):
        local = np.flatnonzero(data["record"][indices] == record)
        permutation[local] = rng.permutation(local)
    return permutation


def verify(cache, anchor, temporal, out):
    report = out / "verification.json"
    if report.exists():
        raise FileExistsError(report)
    manifest = read_json(out / "artifact_manifest.json")
    verify_hashes(out, manifest)
    prefit = read_json(out / "before-fitting.json")
    verify_hashes(out, prefit)
    registration = read_json(out / "registration.json")
    result = read_json(out / "results.json")
    same_tree(registration, result["registration"], "registration")
    same_tree(registration["config"], study.CONFIG, "executed config")
    verify_hashes(ROOT, registration["source_sha256"])
    verify_hashes(out / "executed-source", registration["source_sha256"])
    required_prefit = {
        "registration.json",
        "reference-identities.json",
        "before-fit-reference-permutations.npz",
        "cohort-roles.json",
        "cohort-indices.npz",
        "before-fit-rff-base.npz",
        "enrollment-burden.json",
        *("executed-source/" + key for key in registration["source_sha256"]),
    }
    require(required_prefit <= set(prefit), "Pre-fit boundary omits required artifacts")
    require(
        required_prefit
        | {
            "before-fitting.json",
            "nested-source-cases.json",
            "outer-choices.json",
            "before-outer-query-scoring.json",
            "results.json",
            "progress.json",
        }
        <= set(manifest),
        "Manifest omits required artifacts",
    )
    receipt, data, _ = study.load_inputs(cache)
    require(set(data["room"]) == set(study.ROOMS), "Only E01-E03 are permitted")
    require(
        study.digest(cache / "receipt.json")
        == registration["feature_cache_receipt_sha256"],
        "Cache receipt differs",
    )
    require(
        study.digest(anchor / "verification.json")
        == registration["verified_anchor_proof_sha256"],
        "Anchor proof differs",
    )
    require(
        read_json(anchor / "verification.json")["models_verified"] == 15,
        "Anchor is not verified",
    )
    require(
        read_json(anchor / "registration.json")["feature_cache_receipt_sha256"]
        == registration["feature_cache_receipt_sha256"],
        "Anchor cache binding differs",
    )
    verify_hashes(anchor, read_json(anchor / "artifact_manifest.json"))
    verify_hashes(anchor, registration["frozen_outer_prior_sha256"])
    verify_hashes(temporal, registration["frozen_temporal_control_sha256"])
    raw_path = anchor / "raw_current_control-features.npy"
    require(
        study.digest(raw_path) == registration["raw_current_features_sha256"],
        "Current features differ",
    )
    raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
    require(
        raw.shape == (receipt["logical_rows"], 1368) and raw.dtype == np.float32,
        "Current feature layout differs",
    )
    choices = read_json(out / "outer-choices.json")
    boundary = read_json(out / "before-outer-query-scoring.json")
    same_tree(
        boundary,
        dict(
            choices_sha256=study.digest(out / "outer-choices.json"),
            source_cases_sha256=study.digest(out / "nested-source-cases.json"),
            all_three_choices_frozen=True,
        ),
        "Outer scoring boundary",
    )
    same_tree(choices, result["choices"], "Final choices")
    same_tree(
        read_json(out / "enrollment-burden.json"),
        dict(
            reference_pose_labels_per_room=120,
            endpoint_windows=120,
            current_covariance_states_per_window=8,
            maximum_distinct_current_covariance_states=960,
            recordings=6,
            nominal_cumulative_recorded_prefix_seconds=158.4,
            limit="Same120labels/sixrecordings as anchor route, but uses up to960 currentCSIcovariance frames instead of120endpointmatrices. Nominalprefix excludes pauses/setup, not live capture time.",
        ),
        "Enrollment disclosure",
    )
    rng = np.random.default_rng(study.CONFIG["seed"])
    initialization = load_npz(out / "before-fit-rff-base.npz")
    same_array(
        initialization["unscaled_weight"],
        rng.normal(size=(1368, 512)),
        "RFF initialization differs",
    )
    same_array(
        initialization["offset"], rng.uniform(0, 2 * np.pi, 512), "RFF offsets differ"
    )
    roles = load_npz(out / "cohort-indices.npz")
    role_summary = read_json(out / "cohort-roles.json")
    identities = read_json(out / "reference-identities.json")
    permutations = load_npz(out / "before-fit-reference-permutations.npz")
    references, statistics, matrices, priors = {}, {}, {}, {}
    maximum_residual = maximum_direct = maximum_prediction = 0.0
    equations = 0
    for room in study.ROOMS:
        refs = references[room] = study.reference_indices(data, room)
        train, validation = study.inner_roles(data, room)
        query = study.query_indices(data, room)
        for name, indices in (
            ("inner_train", np.flatnonzero(train)),
            ("validation", np.flatnonzero(validation)),
            ("reference", refs),
            ("outer_query", query),
        ):
            same_array(roles[room + "_" + name], indices, "Cohort roles differ")
        same_tree(
            role_summary[room],
            dict(
                inner_train_people=sorted(set(data["person"][train])),
                inner_train_rows=int(train.sum()),
                inner_own_validation_people=sorted(set(data["person"][validation])),
                inner_own_validation_rows=int(validation.sum()),
                reference_people=sorted(set(data["person"][refs])),
                reference_rows=120,
                outer_query_people=sorted(set(data["person"][query])),
                outer_query_rows=len(query),
            ),
            "Cohort summary",
        )
        same_tree(
            identities[room],
            [
                dict(record=str(data["record"][i]), end=int(data["end"][i]))
                for i in refs
            ],
            "Reference identity",
        )
        same_array(
            permutations[room],
            study.label_permutation(data, refs),
            "Reference permutation differs",
        )
        kernel = load_npz(out / (room + "-reference-kernel.npz"))
        same_array(kernel["reference_indices"], refs, "Kernel reference IDs differ")
        same_array(
            kernel["reference_label_permutation"],
            permutations[room],
            "Kernel permutation differs",
        )
        same_array(
            kernel["room_indices"],
            np.flatnonzero(data["room"] == room),
            "Kernel room IDs differ",
        )
        statistics[room], matrices[room] = reference_state(raw, refs, kernel)
        expected = study.train_inner(raw, data, room, out=None)
        mapping = load_npz(out / (room + "-inner-map.npz"))
        prior = load_npz(out / (room + "-inner-prior.npz"))
        for key, value in expected["mapping"].items():
            same_array(mapping[key], value, "Source-only inner map differs: " + key)
        for key, value in expected["state"].items():
            same_array(prior[key], value, "Source-only inner prior differs: " + key)
        prediction = study.predict(study.apply_map(raw, mapping), prior)
        same_array(prior["prediction"], prediction, "Saved inner prior replay differs")
        priors[room] = dict(prediction=prediction, mean=prior["y_mean"].reshape(17, 2))
        print("Verified source-only inner map/prior " + room, flush=True)
    cases = []
    for source in study.ROOMS:
        for target in study.ROOMS:
            if source == target:
                continue
            path = source + "-to-" + target + "-adaptation.npz"
            require(path in manifest, "Manifest omits adaptation")
            state = load_npz(out / path)
            refs = references[target]
            query = np.flatnonzero((data["room"] == target) & study.val_people(data))
            same_array(state["reference_indices"], refs, "Inner reference IDs differ")
            same_array(state["query_indices"], query, "Inner validation IDs differ")
            residual, difference, residuals = audit_duals(
                matrices[target],
                data["pose"][refs],
                priors[source]["prediction"][refs],
                priors[source]["mean"],
                permutations[target],
                state,
            )
            maximum_residual, maximum_direct = (
                max(maximum_residual, residual),
                max(maximum_direct, difference),
            )
            equations += 15
            cross = cross_kernel(raw, query, statistics[target])
            curves = {
                method: [
                    dict(
                        **choice,
                        source_validation_nje=study.nje(
                            study.candidate_prediction(
                                method,
                                choice,
                                state,
                                cross,
                                priors[source]["prediction"][query],
                                priors[source]["mean"],
                            ),
                            data["pose"][query],
                        ),
                    )
                    for choice in study.candidate_grid(method)
                ]
                for method in study.METHODS
            }
            cases.append(
                dict(
                    source=source,
                    target=target,
                    reference_indices=refs.tolist(),
                    validation_query_indices=query.tolist(),
                    reference_people=sorted(set(data["person"][refs])),
                    validation_people=sorted(set(data["person"][query])),
                    curves=curves,
                    normal_equation_relative_residuals=residuals,
                )
            )
    same_tree(read_json(out / "nested-source-cases.json"), cases, "Nested cases")
    for room in study.ROOMS:
        same_tree(
            choices[room],
            study.select_outer(cases, room),
            "Other-two-room selection " + room,
        )
    rows = []
    parity = {}
    methods = (
        *study.METHODS,
        "source_mean",
        "reference_mean",
        "source_prior_zero_shot",
        "frozen_temporal_zero_shot",
    )
    require(len(result["results"]) == 27, "Expected all 27 method/room results")
    for room in study.ROOMS:
        mapping = load_npz(anchor / (room + "-raw_current_control-rff.npz"))
        prior = load_npz(anchor / (room + "-raw_current_control-model.npz"))
        base = study.predict(study.apply_map(raw, mapping), prior)
        query, refs = study.query_indices(data, room), references[room]
        same_array(prior["indices"], query, "Frozen prior IDs differ")
        parity[room] = float(np.max(np.abs(base[query] - prior["prediction"])))
        require(parity[room] <= 1e-12, "Frozen f0 replay differs")
        mean = prior["y_mean"].reshape(17, 2)
        state = load_npz(out / (room + "-outer-adaptation.npz"))
        same_array(state["reference_indices"], refs, "Outer reference IDs differ")
        same_array(state["query_indices"], query, "Outer query IDs differ")
        residual, difference, residuals = audit_duals(
            matrices[room],
            data["pose"][refs],
            base[refs],
            mean,
            permutations[room],
            state,
        )
        maximum_residual, maximum_direct = (
            max(maximum_residual, residual),
            max(maximum_direct, difference),
        )
        equations += 15
        same_tree(
            state["normal_equation_residuals"].tolist(),
            [residuals[m] for m in study.KERNEL_METHODS],
            "Outer equation residuals",
        )
        permutation = query_permutation(data, query)
        same_array(
            np.load(out / (room + "-query-permutation.npy"), allow_pickle=False),
            permutation,
            "Query shuffle differs",
        )
        cross = cross_kernel(raw, query, statistics[room])
        predictions = {
            method: study.candidate_prediction(
                method,
                choices[room][method]["selected"],
                state,
                cross,
                base[query],
                mean,
            )
            for method in study.METHODS
        }
        predictions.update(
            source_mean=np.broadcast_to(mean, (len(query), 17, 2)),
            reference_mean=np.broadcast_to(
                data["pose"][refs].mean(0), (len(query), 17, 2)
            ),
            source_prior_zero_shot=base[query],
        )
        old = load_npz(temporal / (room + "-congruence_dynamics-model-prediction.npz"))
        keep = np.isin(old["held_indices"], query)
        same_array(old["held_indices"][keep], query, "Temporal control IDs differ")
        predictions["frozen_temporal_zero_shot"] = old["pose"][keep]
        for method in methods:
            name = room + "-" + method + "-prediction.npz"
            require(name in manifest, "Manifest omits prediction")
            saved = load_npz(out / name)
            same_array(saved["query_indices"], query, "Prediction query IDs differ")
            prediction = predictions[method]
            same_array(
                saved["prediction"], prediction, "Saved prediction differs: " + name
            )
            maximum_prediction = max(
                maximum_prediction,
                float(np.max(np.abs(saved["prediction"] - prediction))),
            )
            rows.append(
                dict(
                    room=room,
                    method=method,
                    mode="supervised_room_calibration"
                    if method in study.METHODS or method == "reference_mean"
                    else "zero_shot_control",
                    choice=choices[room][method]["selected"]
                    if method in study.METHODS
                    else None,
                    **study.scores(prediction, data, query, permutation),
                )
            )
        print("Verified all nine predictions/metrics " + room, flush=True)
    same_tree(result["results"], rows, "Metrics/strata/shuffles")
    same_tree(read_json(out / "progress.json"), dict(results=rows), "Progress results")
    same_tree(
        result["frozen_prior_prediction_replay_maximum"], parity, "Frozen prior parity"
    )
    same_tree(
        result["frozen_continuation_gate"],
        study.gate(rows, choices),
        "Continuation gate",
    )
    require(equations == 135, "Expected all 135 saved dual systems")
    verify_hashes(out, manifest)
    study.save(
        report,
        dict(
            models_verified=27,
            saved_dual_normal_equations_verified=equations,
            source_only_inner_maps_and_priors_verified=3,
            nested_direction_curves_verified=6,
            outer_other_two_room_choices_verified=3,
            reference_endpoint_ids_per_room=120,
            feature_rows_verified=len(raw),
            maximum_saved_prediction_difference=maximum_prediction,
            maximum_normal_equation_relative_residual=maximum_residual,
            maximum_independent_dual_relative_difference=maximum_direct,
            all_metrics_strata_query_shuffles_and_gate_verified=True,
            prefit_and_before_scoring_boundary_hashes_verified=True,
            chronology_limit="Saved boundary contents and hashes verified; hashes alone do not establish execution chronology.",
        ),
    )
    return read_json(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--anchor-probe", type=Path, required=True)
    parser.add_argument("--temporal-probe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verify(args.cache, args.anchor_probe, args.temporal_probe, args.out)
