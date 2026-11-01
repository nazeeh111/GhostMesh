"""Nested source-room validation of supervised 120-label residual calibration.

Outer-fold choices use only its OTHER TWO rooms. Reference-only kernel statistics
and labeled enrollment are calibrated mode, never zero-shot inference. The same
120 labels describe eight-frame endpoint windows, up to960 covariance states.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.reference_kernel import ReferenceRBFKernel  # noqa: E402
from scripts.domain_covariance_cached_probe import load_inputs  # noqa: E402
from scripts.domain_covariance_full_cache import verify_hashes  # noqa: E402
from scripts.domain_covariance_source_probe import digest, multiply, rff_transform, save  # noqa: E402
from scripts.domain_reference_covariance_probe import (  # noqa: E402
    fit_readout,
    predict,
    query_indices,
    reference_indices,
    scores,
)

ROOMS = ("E01", "E02", "E03")
KERNEL_METHODS = ("residual_krr", "absolute_krr", "permuted_residual_krr")
SHIFT_METHODS = ("constant_residual", "mean_mixture")
METHODS = KERNEL_METHODS + SHIFT_METHODS
ALL_METHODS = METHODS + (
    "source_mean",
    "reference_mean",
    "source_prior_zero_shot",
    "frozen_temporal_zero_shot",
)
CONFIG = dict(
    seed=2027,
    primary="residual_krr",
    feature="RAW-CURRENT1368, eight current covariance states, no RF rebuild or reference-anchor descriptor",
    alpha=[0.001, 0.01, 0.1, 1.0, 10.0],
    beta=[0.0, 0.25, 0.5, 0.75, 1.0],
    base_features=512,
    base_selection="One inner room's8TRAIN people excluding4/9; source-only RFF stats/bandwidth/readout; basealpha minimizes NJE on itsOWNroom4/9 only",
    nested_selection="For each outerholdR, use ONLYother2rooms A->B andB->A. Pseudo-targetBfirst2people reference120; queryONLYBpeople4/9; equalmean of2directions selects sharedalpha/beta withinthatouterfold",
    feature_stats="Kernel mean/SD(.001floor)/medianALLunorderedpair squared distances fittedONLY120reference endpointWINDOWS, up to960currentCSIcovarianceframes",
    kernel="exp(-squared_distance/(2*reference_median_squared_distance)); no query mean/scale/action/labels",
    residual="(K+120*alpha*I)dual=Yref-f0(Xref); no targetcentering/intercept; f=f0+beta*Kq,ref*dual; farreferencefallbackf0",
    permutation="Seed2027, independently permute GT labels WITHINeachof6reference RECORDS; preserveeachrecord's20pose-labeldistribution; indices fixed beforeallfits",
    controls="Source mean,reference mean,independentlynestedselectedmeanmixture, frozenRAWf0, independentlyselectedconstantresidual,absoluteKRRaroundsourceMean,independentlyselectedwithinrecordreference-labelpermutedKRR,frozenstrict-temporalzero-shot",
    coordinate_convention="UNCLIPPED hip-root-relative supplied2Dannotation coordinates; no correctionclipping, anatomyprojection or GTroot placement",
    gate="ResidualNJE strictlylower thanALL8controls eachroom; PCK05/.2 no worse than meanmixtureANDf0 eachroom; centeredarticulationR2>0 andalignedNJE<withinrecordqueryshuffle eachroom; beta>0",
    limitation="Innerbase8sourcepeople vsfrozenfinalbase16; sourceDEVELOPMENT one seed/all27actions, no E04,fullabsolute-pose/3D/initial-pose/calibrated-productclaim",
)


def val_people(data):
    return np.isin(data["person"], [f"S{i:02}" for i in (4, 9, 14, 19, 24, 29)])


def inner_roles(data, room):
    own = data["room"] == room
    validation = own & val_people(data)
    train = own & ~validation
    if set(data["person"][train]) & set(data["person"][validation]):
        raise ValueError("Source people overlap")
    return train, validation


def label_permutation(data, reference):
    perm = np.arange(len(reference))
    rng = np.random.default_rng(CONFIG["seed"])
    for record in sorted(set(data["record"][reference])):
        local = np.flatnonzero(data["record"][reference] == record)
        perm[local] = rng.permutation(local)
    return perm


def apply_map(raw, mapping):
    z = np.empty((len(raw), 512))
    for start in range(0, len(raw), 128):
        x = (
            np.asarray(raw[start : start + 128], dtype="float64")
            - mapping["feature_mean"]
        ) / mapping["feature_scale"]
        z[start : start + len(x)] = (
            np.sqrt(2 / 512)
            * np.cos(multiply(x, mapping["weight"]) + mapping["offset"])
            - mapping["rff_mean"]
        ) / mapping["rff_scale"]
    return z


def train_inner(raw, data, room, out=None):
    train, validation = inner_roles(data, room)
    if out is not None:
        z = rff_transform(raw, train, out / (room + "-inner-map.npz"))
        with np.load(out / (room + "-inner-map.npz"), allow_pickle=False) as saved:
            mapping = {k: saved[k] for k in saved.files}
    else:
        # Pure source-only equivalent of the frozen map builder, without writes.
        from scripts.domain_covariance_source_probe import feature_stats

        mean, scale = feature_stats(raw, train)
        rng = np.random.default_rng(CONFIG["seed"])
        ix = rng.choice(np.flatnonzero(train), 128, replace=False)
        small = (np.asarray(raw[ix], dtype="float64") - mean) / scale
        dist = np.maximum(
            (small * small).sum(1)[:, None]
            + (small * small).sum(1)[None]
            - 2 * multiply(small, small.T),
            0,
        )
        width = float(np.median(dist[np.triu_indices(128, 1)]))
        if width <= 0 or not np.isfinite(width):
            raise ValueError("Degenerate source-only RBF bandwidth")
        rng = np.random.default_rng(CONFIG["seed"])
        w = rng.normal(size=(raw.shape[1], 512)) / np.sqrt(width)
        b = rng.uniform(0, 2 * np.pi, 512)
        base = dict(
            feature_mean=mean,
            feature_scale=scale,
            weight=w,
            offset=b,
            bandwidth_squared=np.asarray(width),
            bandwidth_sample_indices=ix,
        )
        phi = np.empty((len(raw), 512))
        for start in range(0, len(raw), 128):
            x = (np.asarray(raw[start : start + 128], dtype="float64") - mean) / scale
            phi[start : start + len(x)] = np.sqrt(2 / 512) * np.cos(multiply(x, w) + b)
        mapping = dict(
            base,
            rff_mean=phi[train].mean(0),
            rff_scale=np.maximum(phi[train].std(0), 0.001),
        )
        z = (phi - mapping["rff_mean"]) / mapping["rff_scale"]
    state, grid = fit_readout(z, data["pose"], train, validation, data["room"])
    prediction = predict(z, state)
    if out is not None:
        np.savez(out / (room + "-inner-prior.npz"), **state, prediction=prediction)
    return dict(
        mapping=mapping,
        state=state,
        prediction=prediction,
        source_mean=state["y_mean"].reshape(17, 2),
        base_grid=grid,
        train_indices=np.flatnonzero(train),
        validation_indices=np.flatnonzero(validation),
    )


def nje(prediction, target):
    if np.asarray(prediction).shape != target.shape:
        raise ValueError("Source-validation predictions must match every query target")
    error = np.linalg.norm(np.asarray(prediction).reshape(-1, 17, 2) - target, axis=-1)
    diagonal = np.linalg.norm(target.max(1) - target.min(1), axis=-1)
    usable = np.isfinite(target).all((1, 2)) & (diagonal > 1e-6)
    if not usable.any() or not np.isfinite(error[usable]).all():
        raise ValueError("Invalid source-validation pose error")
    return float((error[usable] / diagonal[usable, None]).mean())


def adaptations(kernel, reference_y, base_reference, source_mean, permutation):
    reference_y = reference_y.reshape(-1, 34)
    base_reference = base_reference.reshape(-1, 34)
    source_mean = source_mean.reshape(34)
    rhs = dict(
        residual_krr=reference_y - base_reference,
        absolute_krr=reference_y - source_mean,
        permuted_residual_krr=reference_y[permutation] - base_reference,
    )
    state = {}
    residuals = {}
    for method in KERNEL_METHODS:
        solutions = [kernel.solve(rhs[method], alpha) for alpha in CONFIG["alpha"]]
        state[method] = np.stack([solution[0] for solution in solutions])
        residuals[method] = [solution[1] for solution in solutions]
    state["constant_residual"] = (reference_y - base_reference).mean(0)
    state["mean_mixture"] = reference_y.mean(0) - source_mean
    return state, residuals


def candidate_grid(method):
    return [
        dict(alpha=alpha, beta=beta)
        for alpha in (CONFIG["alpha"] if method in KERNEL_METHODS else [None])
        for beta in CONFIG["beta"]
    ]


def candidate_prediction(method, choice, state, cross, base_query, source_mean):
    prior = (
        np.broadcast_to(np.asarray(source_mean).reshape(1, 34), (len(base_query), 34))
        if method in ("absolute_krr", "mean_mixture")
        else np.asarray(base_query).reshape(-1, 34)
    )
    if method in KERNEL_METHODS:
        index = CONFIG["alpha"].index(choice["alpha"])
        adjustment = multiply(cross, state[method][index])
    else:
        adjustment = state[method]
    result = (prior + choice["beta"] * adjustment).reshape(-1, 17, 2)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite unprojected calibrated prediction")
    return result


def direction_case(
    source,
    target,
    raw,
    data,
    priors,
    kernels,
    reference,
    crosses,
    row_positions,
    permutations,
    out=None,
):
    if source == target:
        raise ValueError("Pseudo-target room must be absent from inner base")
    prior = priors[source]
    refs = reference[target]
    query = np.flatnonzero((data["room"] == target) & val_people(data))
    if set(data["person"][refs]) & set(data["person"][query]):
        raise ValueError("Calibration/query people overlap")
    state, residuals = adaptations(
        kernels[target],
        data["pose"][refs],
        prior["prediction"][refs],
        prior["source_mean"],
        permutations[target],
    )
    cross = crosses[target][row_positions[target][query]]
    curves = {}
    for method in METHODS:
        curves[method] = [
            dict(
                **choice,
                source_validation_nje=nje(
                    candidate_prediction(
                        method,
                        choice,
                        state,
                        cross,
                        prior["prediction"][query],
                        prior["source_mean"],
                    ),
                    data["pose"][query],
                ),
            )
            for choice in candidate_grid(method)
        ]
    if out is not None:
        np.savez(
            out / (source + "-to-" + target + "-adaptation.npz"),
            **state,
            query_indices=query,
            reference_indices=refs,
        )
    return dict(
        source=source,
        target=target,
        reference_indices=refs.tolist(),
        validation_query_indices=query.tolist(),
        reference_people=sorted(set(data["person"][refs])),
        validation_people=sorted(set(data["person"][query])),
        curves=curves,
        normal_equation_relative_residuals=residuals,
    )


def select_outer(cases, held_room):
    others = set(ROOMS) - {held_room}
    selected = [case for case in cases if {case["source"], case["target"]} == others]
    if (
        len(selected) != 2
        or {case["source"] for case in selected} != others
        or {case["target"] for case in selected} != others
    ):
        raise ValueError(
            "Exactly both directions of the OTHER TWO source rooms required"
        )
    result = {}
    for method in METHODS:
        grid = candidate_grid(method)
        values = []
        for i, choice in enumerate(grid):
            if any(
                {k: case["curves"][method][i][k] for k in ("alpha", "beta")} != choice
                for case in selected
            ):
                raise ValueError("Nested grids differ")
            values.append(
                dict(
                    **choice,
                    source_validation_nje=float(
                        np.mean(
                            [
                                case["curves"][method][i]["source_validation_nje"]
                                for case in selected
                            ]
                        )
                    ),
                )
            )
        chosen = min(
            range(len(values)), key=lambda i: values[i]["source_validation_nje"]
        )
        result[method] = dict(
            selected=values[chosen],
            grid=values,
            source_directions=[
                case["source"] + "->" + case["target"] for case in selected
            ],
        )
    return result


def prepare(raw, data, out=None):
    reference = {room: reference_indices(data, room) for room in ROOMS}
    permutations = {room: label_permutation(data, reference[room]) for room in ROOMS}
    kernels = {
        room: ReferenceRBFKernel.from_reference(raw[reference[room]]) for room in ROOMS
    }
    room_indices = {room: np.flatnonzero(data["room"] == room) for room in ROOMS}
    row_positions = {room: np.full(len(raw), -1, dtype=np.int64) for room in ROOMS}
    crosses = {}
    for room in ROOMS:
        row_positions[room][room_indices[room]] = np.arange(len(room_indices[room]))
        crosses[room] = kernels[room].cross_kernel(raw[room_indices[room]])
        if out is not None:
            np.savez(
                out / (room + "-reference-kernel.npz"),
                **kernels[room].state_dict(),
                reference_indices=reference[room],
                reference_label_permutation=permutations[room],
                room_indices=room_indices[room],
            )
    priors = {room: train_inner(raw, data, room, out) for room in ROOMS}
    cases = [
        direction_case(
            source,
            target,
            raw,
            data,
            priors,
            kernels,
            reference,
            crosses,
            row_positions,
            permutations,
            out,
        )
        for source in ROOMS
        for target in ROOMS
        if source != target
    ]
    choices = {room: select_outer(cases, room) for room in ROOMS}
    return dict(
        reference=reference,
        permutations=permutations,
        kernels=kernels,
        room_indices=room_indices,
        row_positions=row_positions,
        crosses=crosses,
        priors=priors,
        cases=cases,
        choices=choices,
    )


def gate(results, choices):
    decisions = []
    for room in ROOMS:
        rows = {r["method"]: r for r in results if r["room"] == room}
        if set(rows) != set(ALL_METHODS) or sum(
            r["room"] == room for r in results
        ) != len(ALL_METHODS):
            raise ValueError("Gate requires every method exactly once in every room")
        primary = rows[CONFIG["primary"]]
        aligned = primary["aligned"]
        controls = [row for name, row in rows.items() if name != CONFIG["primary"]]
        beat = all(
            aligned["normalized_joint_error"] < row["aligned"]["normalized_joint_error"]
            for row in controls
        )
        pck = all(
            aligned["pck_curve"][threshold]
            >= rows[name]["aligned"]["pck_curve"][threshold]
            for threshold in ("0.05", "0.2")
            for name in ("mean_mixture", "source_prior_zero_shot")
        )
        motion = (
            aligned["record_centered_articulation_r2"] is not None
            and aligned["record_centered_articulation_r2"] > 0
            and aligned["normalized_joint_error"]
            < primary["shuffled"]["normalized_joint_error"]
        )
        positive_beta = choices[room][CONFIG["primary"]]["selected"]["beta"] > 0
        decisions.append(
            dict(
                room=room,
                beats_all_eight_controls_nje=beat,
                pck_not_worse_than_static_and_prior=pck,
                positive_aligned_articulation_motion=motion,
                positive_beta=positive_beta,
                passed=bool(beat and pck and motion and positive_beta),
            )
        )
    return dict(passed=all(r["passed"] for r in decisions), per_room=decisions)


def run(cache, anchor, temporal, out):
    if out.exists():
        raise FileExistsError(out)
    started = time.monotonic()
    receipt, data, _ = load_inputs(cache)
    proof = json.loads((anchor / "verification.json").read_text())
    anchor_registration = json.loads((anchor / "registration.json").read_text())
    if proof["models_verified"] != 15 or anchor_registration[
        "feature_cache_receipt_sha256"
    ] != digest(cache / "receipt.json"):
        raise ValueError("Verified anchor/current-state source does not match")
    verify_hashes(anchor, json.loads((anchor / "artifact_manifest.json").read_text()))
    raw_path = anchor / "raw_current_control-features.npy"
    raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
    if raw.shape != (receipt["logical_rows"], 1368) or raw.dtype != np.dtype("float32"):
        raise ValueError("Unexpected current-state feature cache")
    for start in range(0, len(raw), 256):
        if not np.isfinite(raw[start : start + 256]).all():
            raise ValueError("Nonfinite current covariance features")
    paths = [
        Path(__file__),
        ROOT / "ghostmesh/reference_kernel.py",
        ROOT / "scripts/domain_reference_covariance_probe.py",
        ROOT / "scripts/domain_covariance_cached_probe.py",
        ROOT / "scripts/domain_covariance_source_probe.py",
        ROOT / "scripts/domain_covariance_full_cache.py",
        ROOT / "ghostmesh/metrics.py",
    ]
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    prior_files = {
        room + "-raw_current_control-" + suffix: digest(
            anchor / (room + "-raw_current_control-" + suffix)
        )
        for room in ROOMS
        for suffix in ("rff.npz", "model.npz")
    }
    old_temporal = {
        room + "-congruence_dynamics-model-prediction.npz": digest(
            temporal / (room + "-congruence_dynamics-model-prediction.npz")
        )
        for room in ROOMS
    }
    out.mkdir(parents=True)
    registration = dict(
        config=CONFIG,
        source_sha256=source_hashes,
        feature_cache_receipt_sha256=digest(cache / "receipt.json"),
        raw_current_features_sha256=digest(raw_path),
        verified_anchor_proof_sha256=digest(anchor / "verification.json"),
        frozen_outer_prior_sha256=prior_files,
        frozen_temporal_control_sha256=old_temporal,
        lineage="Chosen after failed CSI-only reference adaptation; SOURCE DEVELOPMENT only. All source-only nested choices frozen before any final outer query score",
    )
    save(out / "registration.json", registration)
    for p in paths:
        dest = out / "executed-source" / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    # Freeze reference identity, exact roles, permutations and random initialization.
    references = {room: reference_indices(data, room) for room in ROOMS}
    save(
        out / "reference-identities.json",
        {
            room: [
                dict(record=str(data["record"][i]), end=int(data["end"][i])) for i in ix
            ]
            for room, ix in references.items()
        },
    )
    np.savez(
        out / "before-fit-reference-permutations.npz",
        **{room: label_permutation(data, ix) for room, ix in references.items()},
    )
    role_archive = {}
    role_summary = {}
    for room in ROOMS:
        train, validation = inner_roles(data, room)
        query = query_indices(data, room)
        role_summary[room] = dict(
            inner_train_people=sorted(set(data["person"][train])),
            inner_train_rows=int(train.sum()),
            inner_own_validation_people=sorted(set(data["person"][validation])),
            inner_own_validation_rows=int(validation.sum()),
            reference_people=sorted(set(data["person"][references[room]])),
            reference_rows=120,
            outer_query_people=sorted(set(data["person"][query])),
            outer_query_rows=len(query),
        )
        for name, ix in (
            ("inner_train", np.flatnonzero(train)),
            ("validation", np.flatnonzero(validation)),
            ("reference", references[room]),
            ("outer_query", query),
        ):
            role_archive[room + "_" + name] = ix
    save(out / "cohort-roles.json", role_summary)
    np.savez(out / "cohort-indices.npz", **role_archive)
    rng = np.random.default_rng(CONFIG["seed"])
    np.savez(
        out / "before-fit-rff-base.npz",
        unscaled_weight=rng.normal(size=(1368, 512)),
        offset=rng.uniform(0, 2 * np.pi, 512),
    )
    save(
        out / "enrollment-burden.json",
        dict(
            reference_pose_labels_per_room=120,
            endpoint_windows=120,
            current_covariance_states_per_window=8,
            maximum_distinct_current_covariance_states=960,
            recordings=6,
            nominal_cumulative_recorded_prefix_seconds=158.4,
            limit="Same120labels/sixrecordings as anchor route, but uses up to960 currentCSIcovariance frames instead of120endpointmatrices. Nominalprefix excludes pauses/setup, not live capture time.",
        ),
    )
    save(
        out / "before-fitting.json",
        {str(p.relative_to(out)): digest(p) for p in out.rglob("*") if p.is_file()},
    )
    plan = prepare(raw, data, out)
    save(out / "nested-source-cases.json", plan["cases"])
    save(out / "outer-choices.json", plan["choices"])
    save(
        out / "before-outer-query-scoring.json",
        dict(
            choices_sha256=digest(out / "outer-choices.json"),
            source_cases_sha256=digest(out / "nested-source-cases.json"),
            all_three_choices_frozen=True,
        ),
    )
    results = []
    prior_parity = {}
    for room in ROOMS:
        with np.load(
            anchor / (room + "-raw_current_control-rff.npz"), allow_pickle=False
        ) as saved:
            mapping = {k: saved[k] for k in saved.files}
        with np.load(
            anchor / (room + "-raw_current_control-model.npz"), allow_pickle=False
        ) as saved:
            prior_state = {
                key: saved[key] for key in ("x_mean", "y_mean", "weights", "selected")
            }
            stored_prediction = saved["prediction"]
            stored_indices = saved["indices"]
        base = predict(apply_map(raw, mapping), prior_state)
        query = query_indices(data, room)
        if not np.array_equal(query, stored_indices):
            raise ValueError("Frozen prior query rows differ")
        prior_parity[room] = float(np.max(abs(base[query] - stored_prediction)))
        if prior_parity[room] > 1e-12:
            raise ValueError("Frozen source-prior replay differs")
        mean = prior_state["y_mean"].reshape(17, 2)
        refs = plan["reference"][room]
        state, residuals = adaptations(
            plan["kernels"][room],
            data["pose"][refs],
            base[refs],
            mean,
            plan["permutations"][room],
        )
        np.savez(
            out / (room + "-outer-adaptation.npz"),
            **state,
            reference_indices=refs,
            query_indices=query,
            normal_equation_residuals=np.asarray(
                [residuals[m] for m in KERNEL_METHODS]
            ),
        )
        cross = plan["crosses"][room][plan["row_positions"][room][query]]
        permutation = np.arange(len(query))
        rng = np.random.default_rng(90210)
        for record in sorted(set(data["record"][query])):
            local = np.flatnonzero(data["record"][query] == record)
            permutation[local] = rng.permutation(local)
        np.save(
            out / (room + "-query-permutation.npy"), permutation, allow_pickle=False
        )
        predictions = {
            method: candidate_prediction(
                method,
                plan["choices"][room][method]["selected"],
                state,
                cross,
                base[query],
                mean,
            )
            for method in METHODS
        }
        predictions.update(
            source_mean=np.broadcast_to(mean, (len(query), 17, 2)),
            reference_mean=np.broadcast_to(
                data["pose"][refs].mean(0), (len(query), 17, 2)
            ),
            source_prior_zero_shot=base[query],
        )
        with np.load(
            temporal / (room + "-congruence_dynamics-model-prediction.npz"),
            allow_pickle=False,
        ) as saved:
            kept = np.isin(saved["held_indices"], query)
            if not np.array_equal(saved["held_indices"][kept], query):
                raise ValueError("Temporal zero-shot comparator rows differ")
            predictions["frozen_temporal_zero_shot"] = saved["pose"][kept]
        for method, prediction in predictions.items():
            np.savez(
                out / (room + "-" + method + "-prediction.npz"),
                prediction=prediction,
                query_indices=query,
            )
            row = dict(
                room=room,
                method=method,
                mode="supervised_room_calibration"
                if method in METHODS or method == "reference_mean"
                else "zero_shot_control",
                choice=plan["choices"][room][method]["selected"]
                if method in METHODS
                else None,
                **scores(prediction, data, query, permutation),
            )
            results.append(row)
            print(
                json.dumps(
                    dict(
                        room=room,
                        method=method,
                        nje=row["aligned"]["normalized_joint_error"],
                        r2=row["aligned"]["record_centered_articulation_r2"],
                    )
                ),
                flush=True,
            )
        save(out / "progress.json", dict(results=results))
    verify_hashes(ROOT, source_hashes)
    verify_hashes(anchor, prior_files)
    verify_hashes(temporal, old_temporal)
    if digest(raw_path) != registration["raw_current_features_sha256"]:
        raise ValueError("Current-state feature inputs changed")
    save(
        out / "results.json",
        dict(
            registration=registration,
            results=results,
            choices=plan["choices"],
            frozen_continuation_gate=gate(results, plan["choices"]),
            frozen_prior_prediction_replay_maximum=prior_parity,
            seconds=time.monotonic() - started,
        ),
    )
    save(
        out / "artifact_manifest.json",
        {str(p.relative_to(out)): digest(p) for p in out.rglob("*") if p.is_file()},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--anchor-probe", type=Path, required=True)
    parser.add_argument("--temporal-probe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.cache, args.anchor_probe, args.temporal_probe, args.out)
