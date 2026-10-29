"""Motion-only source-development probe of exact window-swap antisymmetry.

The negative labels are algebraic copies of the SAME measured pose pairs.
This does not identify an unknown initial pose or claim a new physical law.
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
from scripts.domain_covariance_cached_probe import load_inputs  # noqa: E402
from scripts.domain_covariance_full_cache import verify_hashes  # noqa: E402
from scripts.domain_covariance_source_probe import fold_masks  # noqa: E402
from scripts.domain_background_probe import centered_r2, digest, multiply  # noqa: E402

CONFIG = dict(
    seed=2027,
    dimensions=456,
    actual_RF_span_frames=16,
    conservative_cache_availability_span_frames=40,
    inference_compute="Odd inference evaluates phi twice; ordinary needs one evaluation without the swap diagnostic. Both training arms process the same2n derived examples",
    rff_dimensions=512,
    arms=["augmented_ordinary", "exact_odd"],
    folds=["E01", "E02", "E03"],
    alphas=[0.001, 0.01, 0.1, 1.0, 10.0],
    floor=0.001,
    pair="Lag8 only: full invariant array[N,8,19,6,3][:,:,:,3,:], flattened456",
    swap="Each sorted log-eigenvalue triple s maps to -reverse(s); swap TWO chronological8framewindows, not the entire recorded time axis",
    stats="Raw column mean/SD on sourceTRAIN [x,Sx] only; SDfloor.001",
    bandwidth="Median squared distance of128 fixed sourceTRAIN augmented samples after raw normalization",
    basis="Same seed2027/W456x512/phases for both arms; both evaluate phi(x) and phi(Sx)",
    ordinary="(phi-augmentedTRAINmean)/augmentedTRAINsd; SDfloor.001; target intercept EXACTZERO",
    odd="(phi(x)-phi(Sx))/2 divided by sourceTRAIN RMS; floor.001; no odd centering or target intercept",
    equation="Both arms process2n TRAIN examples [x,Sx] with [delta,-delta]; (X.TX+2n*alpha*I)W=X.TY. Derived negative targets are not independent measurements or new labels",
    selection="Smallest ORIGINAL chronological source-validation raw displacement MSE; first alpha wins ties",
    gate="Exactodd rawdeltaR2>0 and better than fixedwithinrecordshuffle in all3rooms; oddMSE<=matchedaugordinaryMSE everyroom and strictlybetter at leastone",
    scope="FULL27-ACTION SOURCE DEVELOPMENT E01-E03, one seed; no E04, target calibration, global position, 3D or complete pose claim",
    lineage="Chosen after observing earlier covariance source-development results; frozen before this probe, not a new final hypothesis test",
)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def swap(x):
    x = np.asarray(x)
    if x.shape[-1] != 456:
        raise ValueError("Expected456 lag8 pair coordinates")
    return (-x.reshape(*x.shape[:-1], 8, 19, 3)[..., ::-1]).reshape(x.shape)


def pair_features(full):
    if full.ndim != 2 or full.shape[1] != 2736:
        raise ValueError("Expected full invariant features[N,2736]")
    return (
        np.asarray(full)
        .reshape(-1, 8, 19, 6, 3)[:, :, :, 3, :]
        .reshape(-1, 456)
        .astype("float64")
    )


def fit_map(x, train):
    source, reflected = x[train], swap(x[train])
    mean = 0.5 * (source.mean(0) + reflected.mean(0))
    scale = np.maximum(
        np.sqrt(
            0.5 * (((source - mean) ** 2).mean(0) + ((reflected - mean) ** 2).mean(0))
        ),
        CONFIG["floor"],
    )
    indices = np.flatnonzero(train)
    rng = np.random.default_rng(CONFIG["seed"])
    selected = rng.choice(2 * len(indices), 128, replace=False)
    row_ids, flipped = indices[selected % len(indices)], selected >= len(indices)
    sample = x[row_ids].copy()
    sample[flipped] = swap(sample[flipped])
    sample = (sample - mean) / scale
    distances = np.maximum(
        (sample * sample).sum(1)[:, None]
        + (sample * sample).sum(1)[None]
        - 2 * multiply(sample, sample.T),
        0,
    )
    bandwidth = float(np.median(distances[np.triu_indices(128, 1)]))
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("Degenerate source bandwidth")
    rng = np.random.default_rng(CONFIG["seed"])
    weight = rng.normal(size=(456, 512)) / np.sqrt(bandwidth)
    phase = rng.uniform(0, 2 * np.pi, 512)
    mapping = dict(
        input_mean=mean,
        input_scale=scale,
        weight=weight,
        phase=phase,
        bandwidth=np.asarray(bandwidth),
        bandwidth_rows=row_ids,
        bandwidth_flipped=flipped,
    )
    a, b = phi(x, mapping), phi(swap(x), mapping)
    ordinary_mean = 0.5 * (a[train].mean(0) + b[train].mean(0))
    ordinary_scale = np.maximum(
        np.sqrt(
            0.5
            * (
                ((a[train] - ordinary_mean) ** 2).mean(0)
                + ((b[train] - ordinary_mean) ** 2).mean(0)
            )
        ),
        CONFIG["floor"],
    )
    odd = (a[train] - b[train]) / 2
    mapping.update(
        ordinary_mean=ordinary_mean,
        ordinary_scale=ordinary_scale,
        odd_scale=np.maximum(np.sqrt((odd * odd).mean(0)), CONFIG["floor"]),
    )
    return mapping


def phi(x, mapping):
    result = np.empty((len(x), 512))
    for start in range(0, len(x), 256):
        normalized = (x[start : start + 256] - mapping["input_mean"]) / mapping[
            "input_scale"
        ]
        result[start : start + len(normalized)] = np.sqrt(2 / 512) * np.cos(
            multiply(normalized, mapping["weight"]) + mapping["phase"]
        )
    return result


def designs(x, mapping):
    a, b = phi(x, mapping), phi(swap(x), mapping)
    odd = (a - b) / (2 * mapping["odd_scale"])
    return dict(
        augmented_ordinary=(
            (a - mapping["ordinary_mean"]) / mapping["ordinary_scale"],
            (b - mapping["ordinary_mean"]) / mapping["ordinary_scale"],
        ),
        exact_odd=(odd, -odd),
    )


def fit_ridge(a, b, target, train, validation):
    if target.shape != (len(a), 34) or a.shape != b.shape or np.any(train & validation):
        raise ValueError(
            "Compatible targets/designs and disjoint source roles required"
        )
    matrix = multiply(a[train].T, a[train]) + multiply(b[train].T, b[train])
    right = multiply(a[train].T, target[train]) - multiply(b[train].T, target[train])
    eigen, vectors = eigh(matrix, check_finite=True)
    projected = multiply(vectors.T, right)
    candidates, weights = [], []
    for alpha in CONFIG["alphas"]:
        weight = multiply(
            vectors, projected / (eigen[:, None] + 2 * train.sum() * alpha)
        )
        residual = (
            multiply(matrix + 2 * train.sum() * alpha * np.eye(a.shape[1]), weight)
            - right
        )
        relative = float(np.linalg.norm(residual) / max(np.linalg.norm(right), 1e-12))
        if not np.isfinite(relative) or relative > 1e-6:
            raise ValueError("Normal equation failed")
        error = multiply(a[validation], weight) - target[validation]
        candidates.append(
            dict(
                alpha=alpha,
                source_validation_raw_delta_mse=float((error * error).mean()),
                normal_equation_relative_residual=relative,
            )
        )
        weights.append(weight)
    selected = min(
        range(len(candidates)),
        key=lambda i: candidates[i]["source_validation_raw_delta_mse"],
    )
    return selected, np.stack(weights), candidates


def score(prediction, truth, body, record):
    error = prediction - truth
    denominator = float((truth * truth).sum())
    diagonal = np.linalg.norm(body.max(1) - body.min(1), axis=1)
    if np.any(diagonal <= 1e-8):
        raise ValueError("Degenerate reference body")
    return dict(
        raw_delta_mse=float((error * error).mean()),
        raw_zero_reference_r2=1 - float((error * error).sum()) / denominator
        if denominator
        else None,
        record_centered_r2=centered_r2(prediction, truth, record)["r2"],
        normalized_joint_displacement_error=float(
            (np.linalg.norm(error, axis=-1) / diagonal[:, None]).mean()
        ),
        true_squared_displacement_energy=denominator,
        windows=len(truth),
    )


def report(prediction, data, indices, permutation):
    truth, body, record = (data[key][indices] for key in ("delta", "pose", "record"))
    result = dict(
        aligned=score(prediction, truth, body, record),
        shuffled=score(prediction[permutation], truth, body, record),
        zero=score(np.zeros_like(prediction), truth, body, record),
    )
    for field in ("person", "action"):
        values = data[field][indices]
        result["per_" + field] = {
            key: score(prediction[pick], truth[pick], body[pick], record[pick])
            for key in sorted(set(values))
            for pick in [values == key]
        }
        result["per_" + field + "_shuffled"] = {
            key: score(
                prediction[permutation][pick], truth[pick], body[pick], record[pick]
            )
            for key in sorted(set(values))
            for pick in [values == key]
        }
        result["per_" + field + "_zero"] = {
            key: score(
                np.zeros_like(prediction[pick]), truth[pick], body[pick], record[pick]
            )
            for key in sorted(set(values))
            for pick in [values == key]
        }
    return result


def run(cache, verified_full_probe, out):
    if out.exists():
        raise FileExistsError(out)
    started = time.monotonic()
    proof = json.loads((verified_full_probe / "verification.json").read_text())
    if proof["total_records_feature_verified"] != 810 or proof["models_verified"] != 9:
        raise ValueError("Completed full-source verification required")
    receipt, data, features = load_inputs(cache)
    verified_registration = json.loads(
        (verified_full_probe / "registration.json").read_text()
    )
    if verified_registration["feature_cache_receipt_sha256"] != digest(
        cache / "receipt.json"
    ):
        raise ValueError("Full-source proof belongs to another cache")
    if receipt["logical_rows"] != proof["retained_examples_verified"]:
        raise ValueError("Verified cache coverage differs")
    x = pair_features(features["congruence_dynamics"])
    paths = [
        Path(__file__),
        ROOT / "scripts/domain_covariance_cached_probe.py",
        ROOT / "scripts/domain_covariance_full_cache.py",
        ROOT / "scripts/domain_covariance_source_probe.py",
        ROOT / "scripts/domain_background_probe.py",
    ]
    sources = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    out.mkdir(parents=True)
    registration = dict(
        config=CONFIG,
        source_sha256=sources,
        feature_cache_receipt_sha256=digest(cache / "receipt.json"),
        full_source_verification_sha256=digest(
            verified_full_probe / "verification.json"
        ),
        role_policy="16 source-training/4 source-validation/10 held-room people; all27actions, canonical retained endpoints",
        negative_label_budget="2n processed labels, but only n original measured displacement pairs; swapped negatives derived algebraically",
    )
    save(out / "registration.json", registration)
    for p in paths:
        destination = out / "executed-source" / p.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, destination)
    target = data["delta"].reshape(-1, 34)
    results = []
    for room in CONFIG["folds"]:
        train, validation, held = fold_masks(data["room"], data["person"], room)
        indices = np.flatnonzero(held)
        mapping = fit_map(x, train)
        np.savez(out / (room + "-initial-map.npz"), **mapping)
        save(
            out / (room + "-before-fit.json"),
            dict(
                map_sha256=digest(out / (room + "-initial-map.npz")),
                train_mask_sha256=sha256(train.tobytes()).hexdigest(),
                source_validation_mask_sha256=sha256(validation.tobytes()).hexdigest(),
                original_training_pairs=int(train.sum()),
                augmented_training_examples=int(2 * train.sum()),
                held_windows=int(held.sum()),
            ),
        )
        permutation = np.arange(len(indices))
        rng = np.random.default_rng(90210)
        for key in sorted(set(data["record"][held])):
            pick = np.flatnonzero(data["record"][held] == key)
            permutation[pick] = rng.permutation(pick)
        np.save(out / (room + "-permutation.npy"), permutation, allow_pickle=False)
        for arm, (a, b) in designs(x, mapping).items():
            selected, weights, candidates = fit_ridge(a, b, target, train, validation)
            prediction, swapped = (
                multiply(design[held], weights[selected]).reshape(-1, 17, 2)
                for design in (a, b)
            )
            symmetry_error = float(np.max(abs(prediction + swapped)))
            if arm == "exact_odd" and symmetry_error > 1e-12:
                raise ValueError("Exact antisymmetry failed")
            np.savez(
                out / (room + "-" + arm + "-model.npz"),
                weights=weights,
                selected=np.asarray(selected),
                alpha=np.asarray(CONFIG["alphas"][selected]),
                prediction=prediction,
                swapped_prediction=swapped,
                held_indices=indices,
            )
            row = dict(
                held_room=room,
                arm=arm,
                selected_alpha=CONFIG["alphas"][selected],
                source_validation_grid=candidates,
                swap_prediction_max_abs_sum=symmetry_error,
                **report(prediction, data, indices, permutation),
            )
            results.append(row)
            save(out / "progress.json", dict(results=results))
            print(
                json.dumps(
                    dict(
                        room=room,
                        arm=arm,
                        alpha=row["selected_alpha"],
                        raw_r2=row["aligned"]["raw_zero_reference_r2"],
                        shuffled_r2=row["shuffled"]["raw_zero_reference_r2"],
                        seconds=time.monotonic() - started,
                    )
                ),
                flush=True,
            )
    odd = [r for r in results if r["arm"] == "exact_odd"]
    ordinary = {r["held_room"]: r for r in results if r["arm"] == "augmented_ordinary"}
    gate = bool(
        all(
            r["aligned"]["raw_zero_reference_r2"] > 0
            and r["aligned"]["raw_zero_reference_r2"]
            > r["shuffled"]["raw_zero_reference_r2"]
            and r["aligned"]["raw_delta_mse"]
            <= ordinary[r["held_room"]]["aligned"]["raw_delta_mse"]
            for r in odd
        )
        and any(
            r["aligned"]["raw_delta_mse"]
            < ordinary[r["held_room"]]["aligned"]["raw_delta_mse"]
            for r in odd
        )
    )
    if (
        any(digest(ROOT / p) != h for p, h in sources.items())
        or digest(cache / "receipt.json")
        != registration["feature_cache_receipt_sha256"]
    ):
        raise ValueError("Frozen inputs/source changed")
    verify_hashes(cache, receipt["arrays_sha256"])
    save(
        out / "results.json",
        dict(
            registration=registration,
            results=results,
            frozen_continuation_gate_passed=gate,
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


def verify(cache, out):
    result = json.loads((out / "results.json").read_text())
    registration = result["registration"]
    verify_hashes(out, json.loads((out / "artifact_manifest.json").read_text()))
    for name, expected in registration["source_sha256"].items():
        if (
            digest(ROOT / name) != expected
            or digest(out / "executed-source" / name) != expected
        ):
            raise ValueError("Source snapshot mismatch")
    receipt, data, features = load_inputs(cache)
    if digest(cache / "receipt.json") != registration["feature_cache_receipt_sha256"]:
        raise ValueError("Cache receipt mismatch")
    x, target = (
        pair_features(features["congruence_dynamics"]),
        data["delta"].reshape(-1, 34),
    )
    maximum = residual = 0.0
    for room in CONFIG["folds"]:
        train, validation, held = fold_masks(data["room"], data["person"], room)
        permutation = np.load(out / (room + "-permutation.npy"), allow_pickle=False)
        mapping = fit_map(x, train)
        with np.load(out / (room + "-initial-map.npz"), allow_pickle=False) as saved:
            if not all(np.array_equal(mapping[key], saved[key]) for key in mapping):
                raise ValueError("Source-only map reproduction differs")
        for arm, (a, b) in designs(x, mapping).items():
            selected, weights, candidates = fit_ridge(a, b, target, train, validation)
            with np.load(
                out / (room + "-" + arm + "-model.npz"), allow_pickle=False
            ) as saved:
                maximum = max(
                    maximum,
                    float(np.max(abs(weights - saved["weights"]))),
                    float(
                        np.max(
                            abs(
                                multiply(a[held], weights[selected]).reshape(-1, 17, 2)
                                - saved["prediction"]
                            )
                        )
                    ),
                    float(
                        np.max(
                            abs(
                                multiply(b[held], weights[selected]).reshape(-1, 17, 2)
                                - saved["swapped_prediction"]
                            )
                        )
                    ),
                )
                if int(saved["selected"]) != selected or not np.array_equal(
                    saved["held_indices"], np.flatnonzero(held)
                ):
                    raise ValueError("Saved selection/query rows differ")
            row = next(
                r
                for r in result["results"]
                if r["held_room"] == room and r["arm"] == arm
            )
            prediction = multiply(a[held], weights[selected]).reshape(-1, 17, 2)
            actual_report = report(prediction, data, np.flatnonzero(held), permutation)
            if (
                any(actual_report[key] != row[key] for key in actual_report)
                or candidates != row["source_validation_grid"]
            ):
                raise ValueError(
                    "Metrics, shuffle, zero, strata or validation grid differ"
                )
            residual = max(
                residual,
                max(c["normal_equation_relative_residual"] for c in candidates),
            )
    if maximum != 0 or residual > 1e-6:
        raise ValueError("Saved model readback differs")
    save(
        out / "verification.json",
        dict(
            models_verified=6,
            normal_equations_verified=30,
            maximum_saved_weight_and_prediction_difference=maximum,
            maximum_normal_equation_relative_residual=residual,
            source_only_symmetric_stats_bandwidth_and_validation_choice_recomputed=True,
            feature_cache_receipt_sha256=digest(cache / "receipt.json"),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--full-probe", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(args.cache, args.out)
    elif args.full_probe:
        run(args.cache, args.full_probe, args.out)
    else:
        parser.error("--full-probe is required for fitting")
