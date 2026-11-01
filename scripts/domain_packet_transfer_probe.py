"""Full-source nominal packet-spectrum transfer, with frozen source-only choices.

CPU-only source development; no E04, physical velocity/BVP or calibration claims.
The completed local packet-time experiment is left untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.doppler_features import complex_quotients, spectral_features  # noqa:E402
from ghostmesh.metrics import pose_metrics  # noqa:E402
from scripts.domain_background_probe import digest, multiply, centered_r2  # noqa:E402
from scripts.domain_covariance_cached_probe import load_inputs  # noqa:E402
from scripts.domain_covariance_source_probe import fold_masks, validate_records  # noqa:E402

VARIANTS = ("natural", "within_frame_shuffle", "whole_window_shuffle")
ARMS = {
    "natural_linear": ("natural", "linear"),
    "natural_rff": ("natural", "rff"),
    "within_frame_shuffle": ("within_frame_shuffle", "rff"),
    "whole_window_shuffle": ("whole_window_shuffle", "rff"),
}
SLICES = {
    "pose_full": slice(0, 34),
    "delta_full": slice(34, 68),
    "delta_root": slice(68, 70),
    "delta_relative": slice(70, 104),
}
CONFIG = dict(
    seed=2027,
    alphas=[0.001, 0.01, 0.1, 1.0, 10.0],
    feature_floor=0.001,
    rff_dimensions=512,
    arms=ARMS,
    targets=list(SLICES),
    folds=["E01", "E02", "E03"],
    scope="FULL27-ACTION SOURCE DEVELOPMENT E01-E03/all30people; no E04, target calibration or final efficacy claim",
    input="Fixed380D two-quotient/19frequency-bin/ten-signed-band logpowers, nominal100Hz; exact local Doppler-feature module unchanged",
    timing="Eight causal groupedframes/80nominal samples. Exact individual packet timestamps/rate unverified. Covariance comparator has longer40-frame features; the SAMEconservative40-frame validity mask/endpoints is shared, not identical temporal information",
    mask="Fullcovariance canonical retained rows, then intersect availability of ALLthree spectral variants. Same rows for every spectral arm; covariance comparisons use exactly these held rows, retaining original covariance training data",
    source_roles="16training/4validation/10held people perroom, sourcevalidationperson4/9 in represented rooms",
    feature_stats="TRAINcolumnmean/SD floor.001. RFFbandwidth median distances of128 fixed TRAINsamples; same seed380x512 Gaussianbasis/phases; TRAIN RFFcolumnmean/SDfloor.001",
    equation="Centered source Ridge (X.TX+n*alphaI)W=X.T(Y-sourceMean), rawtargetunits. Currentpose source-validationNJE after common[0,1]clipping; eachDelta source-validation RAWMSE. No targetnormalization/querymean",
    targets_note="Full2Dcurrentpose34, lag8fullDelta34, rootDelta2, hiprelativeDelta34. All same provided measuredendpointlabels; component readouts select alpha independently, so do not combine their outputs as an exact decomposition",
    natural_choice="Each room/target chooses natural exactlinear380 vsRFF512 by minimum SOURCEvalidation criterion; firstlinear tie. Every arm retained. ALLfoldchoices frozen before held scoring",
    gate="Validation-selected NATURALfullDelta rawR2>0 andbetterwithinrecordprediction-shuffle in all3rooms; MSE<=BOTHpacket-permuted RFFcontrols in everyroom andstrictlybetter thaneach in at leastone room. Currentpose efficacy separately lowerNJE/no-worsePCK.05/.2 than source staticprior in everyroom",
    comparisons="Frozenfull27covariance RFF ONLYhiprelativepose/Delta heads on identical commonheldrows. Different features/optimization budgets and any original training-mask differences disclosed. ZeroDelta andsource static/currentmean/meanDelta controls retained",
    endpoint_control="CONTROL ONLY: source TRAIN means of current pose/Delta pooled over all people/actions separately for each canonical frameend; queryend used only by this clock baseline, never RF model; missing sourceend falls back to globalmean",
    limits="Displacement over lag8 nominalframes, not forecast or m/s. NoGTpreviouspose/querymean/actionID as modelinput. Packet-order advantage empirical, never assumed; noBVP/AoA/3D/hardwarecalibration claims",
)


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sources(out):
    paths = [
        Path(__file__),
        ROOT / "ghostmesh/doppler_features.py",
        ROOT / "ghostmesh/metrics.py",
        ROOT / "scripts/domain_background_probe.py",
        ROOT / "scripts/domain_covariance_cached_probe.py",
        ROOT / "scripts/domain_covariance_full_cache.py",
        ROOT / "scripts/domain_covariance_source_probe.py",
    ]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    for source in paths:
        target = out / "executed-source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return hashes


def window_features(raw, ends):
    q, _ = complex_quotients(
        raw, magnitude_units="log_magnitude_db", phase_units="radians"
    )
    result = {name: [] for name in VARIANTS}
    keep = []
    rejected = []
    for i, end in enumerate(ends):
        try:
            row = {
                name: spectral_features(q[int(end) - 7 : int(end) + 1], name)
                for name in VARIANTS
            }
        except ValueError as exc:
            rejected.append(dict(end=int(end), reason=str(exc)))
            continue
        keep.append(i)
        for name in VARIANTS:
            result[name].append(row[name])
    return (
        {
            name: np.asarray(value, dtype=np.float32).reshape(-1, 380)
            for name, value in result.items()
        },
        np.asarray(keep, dtype=np.int64),
        rejected,
    )


def join_labels(pose, ends, cov_pose, cov_delta):
    current = np.asarray(pose[ends], dtype=np.float64)
    past = np.asarray(pose[ends - 8], dtype=np.float64)
    relative = current - current[:, [11, 12]].mean(1, keepdims=True)
    previous_relative = past - past[:, [11, 12]].mean(1, keepdims=True)
    np.testing.assert_array_equal(relative, cov_pose)
    np.testing.assert_array_equal(relative - previous_relative, cov_delta)
    full_delta = current - past
    root_delta = full_delta[:, [11, 12]].mean(1, keepdims=True)
    np.testing.assert_allclose(full_delta - root_delta, cov_delta, rtol=0, atol=1e-15)
    if not np.isfinite(current).all() or not np.isfinite(past).all():
        raise ValueError("Invalid provided endpointlabels")
    return dict(
        pose_full=current,
        pose_relative=relative,
        delta_full=full_delta,
        delta_root=root_delta,
        delta_relative=cov_delta,
    )


def build(base, covariance, full_probe, out):
    if out.exists():
        raise FileExistsError(out)
    started = time.monotonic()
    cov_receipt, cov_data, _ = load_inputs(covariance)
    proof = json.loads((full_probe / "verification.json").read_text())
    reg = json.loads((full_probe / "registration.json").read_text())
    if (
        proof["total_records_feature_verified"] != 810
        or proof["models_verified"] != 9
        or reg["feature_cache_receipt_sha256"] != digest(covariance / "receipt.json")
    ):
        raise ValueError("Independently verified full covariance source required")
    receipt = json.loads((base / "receipt.json").read_text())
    validate_records(receipt["records"])
    if (
        digest(base / "receipt.json") != cov_receipt["raw_receipt_sha256"]
        or receipt["arrays"] != cov_receipt["raw_arrays_sha256"]
    ):
        raise ValueError("Rawsource binding differs")
    for key, expected in receipt["arrays"].items():
        if digest(base / (key + ".npy")) != expected:
            raise ValueError("Rawsourcebytes changed")
    out.mkdir(parents=True)
    hashes = sources(out)
    registration = dict(
        config=CONFIG,
        source_sha256=hashes,
        raw_receipt_sha256=digest(base / "receipt.json"),
        raw_arrays_sha256=receipt["arrays"],
        covariance_receipt_sha256=digest(covariance / "receipt.json"),
        covariance_metadata_sha256=digest(covariance / "targets-and-metadata.npz"),
        full_source_proof_sha256=digest(full_probe / "verification.json"),
    )
    save(out / "registration.json", registration)
    raw = np.load(base / "rf.npy", mmap_mode="r", allow_pickle=False)
    pose = np.load(base / "pose.npy", mmap_mode="r", allow_pickle=False)
    features = {name: [] for name in VARIANTS}
    labels = {
        key: []
        for key in (
            "pose_full",
            "pose_relative",
            "delta_full",
            "delta_root",
            "delta_relative",
        )
    }
    indices = []
    quality = []
    for count, record in enumerate(receipt["records"], 1):
        original = np.flatnonzero(cov_data["record"] == record["id"])
        ends = cov_data["end"][original]
        offset = record["offset"]
        full = np.asarray(pose[offset : offset + 297], dtype=np.float64)
        target = join_labels(
            full, ends, cov_data["pose"][original], cov_data["delta"][original]
        )
        values, keep, rejected = window_features(raw[offset : offset + 297], ends)
        indices.extend(original[keep].tolist())
        for name in VARIANTS:
            features[name].append(values[name])
        for name in labels:
            labels[name].append(target[name][keep])
        quality.append(
            dict(
                record=record["id"],
                canonical_covariance_rows=len(original),
                retained_spectral_rows=len(keep),
                rejected=rejected,
            )
        )
        if count % 30 == 0:
            print(
                json.dumps(
                    dict(records=count, total=810, seconds=time.monotonic() - started)
                ),
                flush=True,
            )
    indices = np.asarray(indices, dtype=np.int64)
    data = {
        key: cov_data[key][indices]
        for key in ("room", "person", "action", "record", "end")
    }
    data.update({key: np.concatenate(value) for key, value in labels.items()})
    data["covariance_index"] = indices
    np.savez_compressed(out / "metadata.npz", **data)
    for name, value in features.items():
        np.save(out / (name + ".npy"), np.concatenate(value), allow_pickle=False)
    save(out / "quality.json", quality)
    coverage = {}
    for room in CONFIG["folds"]:
        masks = fold_masks(data["room"], data["person"], room)
        if any(
            set(data["action"][m]) != {f"A{i:02}" for i in range(1, 28)} for m in masks
        ):
            raise ValueError("A role lost an action")
        coverage[room] = dict(
            zip(("train", "validation", "held"), [int(m.sum()) for m in masks])
        )
    for name, expected in hashes.items():
        if digest(ROOT / name) != expected:
            raise ValueError("Source changed during featurebuild")
    arrays = {
        name: digest(out / name)
        for name in [
            "metadata.npz",
            "quality.json",
            *[name + ".npy" for name in VARIANTS],
        ]
    }
    save(
        out / "receipt.json",
        dict(
            config=CONFIG,
            registration=registration,
            arrays_sha256=arrays,
            logical_rows=len(indices),
            covariance_canonical_rows=len(cov_data["pose"]),
            additional_spectral_drops=len(cov_data["pose"]) - len(indices),
            coverage_per_fold=coverage,
            seconds=time.monotonic() - started,
        ),
    )
    save(
        out / "artifact-manifest.json",
        {
            str(p.relative_to(out)): digest(p)
            for p in sorted(out.rglob("*"))
            if p.is_file()
        },
    )


def load_cache(cache):
    receipt = json.loads((cache / "receipt.json").read_text())
    for name, expected in receipt["arrays_sha256"].items():
        if digest(cache / name) != expected:
            raise ValueError("Spectralcachebytes changed")
    with np.load(cache / "metadata.npz", allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    if set(data["room"]) != {"E01", "E02", "E03"} or set(data["person"]) != {
        f"S{i:02}" for i in range(1, 31)
    }:
        raise ValueError("Forbidden/missing source domain")
    features = {
        name: np.load(cache / (name + ".npy"), mmap_mode="r", allow_pickle=False)
        for name in VARIANTS
    }
    if any(
        x.shape != (receipt["logical_rows"], 380) or not np.isfinite(x).all()
        for x in features.values()
    ):
        raise ValueError("Invalid retained spectral features")
    return receipt, data, features


def fit_map(x, train, family):
    mean = np.asarray(x[train], dtype=np.float64).mean(0)
    scale = np.maximum(np.asarray(x[train], dtype=np.float64).std(0), 0.001)
    normalized = (np.asarray(x, dtype=np.float64) - mean) / scale
    mapping = dict(input_mean=mean, input_scale=scale, kind=np.asarray(family))
    if family == "linear":
        return normalized, mapping
    rng = np.random.default_rng(2027)
    ids = rng.choice(np.flatnonzero(train), 128, replace=False)
    sample = normalized[ids]
    distance = np.maximum(
        np.square(sample).sum(1)[:, None]
        + np.square(sample).sum(1)[None]
        - 2 * multiply(sample, sample.T),
        0,
    )
    width = float(np.median(distance[np.triu_indices(128, 1)]))
    if width <= 0:
        raise ValueError("Degenerate TRAINbandwidth")
    rng = np.random.default_rng(2027)
    weight = rng.normal(size=(x.shape[1], 512)) / np.sqrt(width)
    phase = rng.uniform(0, 2 * np.pi, 512)
    raw = np.sqrt(2 / 512) * np.cos(multiply(normalized, weight) + phase)
    rm, rs = raw[train].mean(0), np.maximum(raw[train].std(0), 0.001)
    mapping.update(
        weight=weight,
        phase=phase,
        bandwidth=np.asarray(width),
        bandwidth_rows=ids,
        rff_mean=rm,
        rff_scale=rs,
    )
    return (raw - rm) / rs, mapping


def transform(x, mapping):
    x = (np.asarray(x, dtype=np.float64) - mapping["input_mean"]) / mapping[
        "input_scale"
    ]
    if str(mapping["kind"]) == "linear":
        return x
    return (
        np.sqrt(2 / 512) * np.cos(multiply(x, mapping["weight"]) + mapping["phase"])
        - mapping["rff_mean"]
    ) / mapping["rff_scale"]


def target_matrix(data):
    return np.concatenate(
        [data[name].reshape(len(data["record"]), -1) for name in SLICES], axis=1
    )


def current_metrics(prediction, target):
    m = pose_metrics(prediction, target, np.ones(target.shape[:2], bool))
    return {
        key: m[key]
        for key in (
            "normalized_joint_error",
            "pck_curve",
            "valid_frames",
            "valid_joints",
        )
    }


def displacement_metrics(prediction, target, body, record):
    denominator = float(np.square(target).sum())
    scale = np.linalg.norm(body.max(1) - body.min(1), axis=1)
    return dict(
        raw_mse=float(np.square(prediction - target).mean()),
        raw_zero_reference_r2=1
        - float(np.square(prediction - target).sum()) / denominator
        if denominator
        else None,
        record_centered_r2=centered_r2(prediction, target, record)["r2"],
        normalized_displacement_error=float(
            (np.linalg.norm(prediction - target, axis=2) / scale[:, None]).mean()
        ),
        true_squared_energy=denominator,
    )


def fit_readouts(z, y, train, val):
    if y.shape != (len(z), 104) or np.any(train & val):
        raise ValueError("Compatible targets/disjoint source roles required")
    xm, ym = z[train].mean(0), y[train].mean(0)
    x = z[train] - xm
    target = y[train] - ym
    system = multiply(x.T, x)
    eigen, vectors = eigh(system, check_finite=True)
    right = multiply(vectors.T, multiply(x.T, target))
    weights = []
    grid = []
    selected = {name: None for name in SLICES}
    for alpha in CONFIG["alphas"]:
        w = multiply(vectors, right / (eigen[:, None] + train.sum() * alpha))
        weights.append(w)
        residual = float(
            np.linalg.norm(
                multiply(system + train.sum() * alpha * np.eye(z.shape[1]), w)
                - multiply(x.T, target)
            )
            / max(np.linalg.norm(multiply(x.T, target)), 1e-12)
        )
        if residual > 1e-6:
            raise ValueError("Ridge normal equation failed")
        prediction = multiply(z[val] - xm, w) + ym
        criteria = {}
        for name, sl in SLICES.items():
            if name == "pose_full":
                criteria[name] = current_metrics(
                    prediction[:, sl].reshape(-1, 17, 2).clip(0, 1),
                    y[val, sl].reshape(-1, 17, 2),
                )["normalized_joint_error"]
            else:
                criteria[name] = float(np.square(prediction[:, sl] - y[val, sl]).mean())
            if selected[name] is None or criteria[name] < selected[name][0]:
                selected[name] = (criteria[name], len(weights) - 1)
        grid.append(
            dict(
                alpha=alpha,
                source_validation_criteria=criteria,
                normal_equation_residual=residual,
            )
        )
    return dict(
        x_mean=xm,
        y_mean=ym,
        weights=np.stack(weights),
        selected_index=np.array([selected[name][1] for name in SLICES]),
        alphas=np.asarray(CONFIG["alphas"]),
    ), grid


def predict(z, model):
    values = {}
    for i, (name, sl) in enumerate(SLICES.items()):
        weight = model["weights"][int(model["selected_index"][i]), :, sl]
        value = multiply(z - model["x_mean"], weight) + model["y_mean"][sl]
        values[name] = value.reshape(-1, 1 if name == "delta_root" else 17, 2)
    values["pose_full"] = values["pose_full"].clip(0, 1)
    return values


def scores(prediction, data, ids):
    result = dict(
        pose_full=current_metrics(prediction["pose_full"], data["pose_full"][ids])
    )
    relative = prediction["pose_full"] - prediction["pose_full"][:, [11, 12]].mean(
        1, keepdims=True
    )
    result["pose_relative"] = current_metrics(relative, data["pose_relative"][ids])
    for name in ("delta_full", "delta_root", "delta_relative"):
        result[name] = displacement_metrics(
            prediction[name],
            data[name][ids],
            data["pose_full"][ids],
            data["record"][ids],
        )
    return result


def stratified(prediction, data, ids):
    result = scores(prediction, data, ids)
    for field in ("person", "action"):
        result["per_" + field] = {
            str(key): scores(
                {k: v[pick] for k, v in prediction.items()}, data, ids[pick]
            )
            for key in np.unique(data[field][ids])
            for pick in [data[field][ids] == key]
        }
    return result


def permutation(record):
    result = np.arange(len(record))
    rng = np.random.default_rng(90210)
    for name in np.unique(record):
        ix = np.flatnonzero(record == name)
        result[ix] = rng.permutation(ix)
    return result


def endpoint_control(data, train, ids):
    prediction = {
        name: np.broadcast_to(data[name][train].mean(0), data[name][ids].shape).copy()
        for name in SLICES
    }
    fallback = []
    for end in np.unique(data["end"][ids]):
        source = train & (data["end"] == end)
        query = data["end"][ids] == end
        if not source.any():
            fallback.append(int(end))
            continue
        for name in SLICES:
            prediction[name][query] = data[name][source].mean(0)
    return prediction, fallback


def fit(cache, baseline, out):
    if out.exists():
        raise FileExistsError(out)
    started = time.monotonic()
    receipt, data, features = load_cache(cache)
    y = target_matrix(data)
    baseline_names = [
        "results.json",
        "verification.json",
        "targets-and-metadata.npz",
    ] + [
        room + "-" + arm + "-model-prediction.npz"
        for room in CONFIG["folds"]
        for arm in ("untreated_covariance", "log_shape_tangent", "congruence_dynamics")
    ]
    baseline_hashes = {name: digest(baseline / name) for name in baseline_names}
    out.mkdir(parents=True)
    hashes = sources(out)
    save(
        out / "registration.json",
        dict(
            config=CONFIG,
            source_sha256=hashes,
            spectral_cache_receipt_sha256=digest(cache / "receipt.json"),
            spectral_cache_arrays_sha256=receipt["arrays_sha256"],
            baseline_sha256=baseline_hashes,
            additional_spectral_drops=receipt["additional_spectral_drops"],
        ),
    )
    frozen = {}
    # Fit/select all12 arm/fold combinations and freeze naturalfamily choices
    # BEFORE any held-query metric is computed.
    for room in CONFIG["folds"]:
        train, val, held = fold_masks(data["room"], data["person"], room)
        arm_choices = {}
        for arm, (variant, family) in ARMS.items():
            z, mapping = fit_map(features[variant], train, family)
            np.savez(out / (room + "-" + arm + "-map.npz"), **mapping)
            save(
                out / (room + "-" + arm + "-before-fit.json"),
                dict(
                    map_sha256=digest(out / (room + "-" + arm + "-map.npz")),
                    train=int(train.sum()),
                    validation=int(val.sum()),
                    held=int(held.sum()),
                ),
            )
            model, grid = fit_readouts(z, y, train, val)
            np.savez(out / (room + "-" + arm + "-model.npz"), **model)
            arm_choices[arm] = {
                name: dict(
                    index=int(model["selected_index"][i]),
                    alpha=float(model["alphas"][int(model["selected_index"][i])]),
                    validation=grid[int(model["selected_index"][i])][
                        "source_validation_criteria"
                    ][name],
                )
                for i, name in enumerate(SLICES)
            }
            save(out / (room + "-" + arm + "-grid.json"), grid)
            print(
                json.dumps(
                    dict(
                        fitted=room + "/" + arm,
                        source_validation=arm_choices[arm],
                        seconds=time.monotonic() - started,
                    )
                ),
                flush=True,
            )
        natural = {
            name: min(
                ("natural_linear", "natural_rff"),
                key=lambda arm: arm_choices[arm][name]["validation"],
            )
            for name in SLICES
        }
        frozen[room] = dict(arms=arm_choices, natural_selected=natural)
    save(out / "ALL-FOLDS-FROZEN-SELECTIONS.json", frozen)
    with np.load(baseline / "targets-and-metadata.npz", allow_pickle=False) as z:
        old = {k: z[k] for k in z.files}
    results = {}
    parity = {}
    comparators = {}
    for room in CONFIG["folds"]:
        train, val, held = fold_masks(data["room"], data["person"], room)
        ids = np.flatnonzero(held)
        perm = permutation(data["record"][ids])
        np.save(out / (room + "-permutation.npy"), perm, allow_pickle=False)
        results[room] = {}
        for arm, (variant, _) in ARMS.items():
            with np.load(
                out / (room + "-" + arm + "-map.npz"), allow_pickle=False
            ) as z:
                mapping = {k: z[k] for k in z.files}
            with np.load(
                out / (room + "-" + arm + "-model.npz"), allow_pickle=False
            ) as z:
                model = {k: z[k] for k in z.files}
            design = transform(features[variant][ids], mapping)
            p = predict(design, model)
            np.savez_compressed(
                out / (room + "-" + arm + "-predictions.npz"), indices=ids, **p
            )
            results[room][arm] = dict(
                aligned=stratified(p, data, ids),
                within_record_prediction_shuffle=stratified(
                    {k: v[perm] for k, v in p.items()}, data, ids
                ),
                selected=frozen[room]["arms"][arm],
            )
            # Reopen all state and sourcefeatures in a distinct readback path.
            with np.load(
                out / (room + "-" + arm + "-map.npz"), allow_pickle=False
            ) as z:
                rm = {k: z[k] for k in z.files}
            with np.load(
                out / (room + "-" + arm + "-model.npz"), allow_pickle=False
            ) as z:
                rw = {k: z[k] for k in z.files}
            fresh = np.load(
                cache / (variant + ".npy"), mmap_mode="r", allow_pickle=False
            )
            rp = predict(transform(fresh[ids], rm), rw)
            differences = {name: float(abs(rp[name] - p[name]).max()) for name in p}
            if max(differences.values()) > 1e-9:
                raise ValueError("Saved-model readback differs")
            parity[room + "/" + arm] = differences
        static = {
            name: np.broadcast_to(
                data[name][train].mean(0), data[name][ids].shape
            ).copy()
            for name in SLICES
        }
        zero = {
            name: static[name] if name == "pose_full" else np.zeros_like(static[name])
            for name in SLICES
        }
        comparators[room] = dict(
            source_mean=stratified(static, data, ids),
            zero_delta=stratified(zero, data, ids),
            covariance={},
        )
        endpoint, fallback = endpoint_control(data, train, ids)
        comparators[room]["endpoint_only"] = stratified(endpoint, data, ids)
        comparators[room]["endpoint_fallback_ends"] = fallback
        original = np.flatnonzero(old["room"] == room)
        lookup = np.full(len(old["pose"]), -1, dtype=int)
        lookup[original] = np.arange(len(original))
        old_ids = lookup[data["covariance_index"][ids]]
        if np.any(old_ids < 0):
            raise ValueError("Covariance comparison row belongs to another room")
        np.testing.assert_array_equal(
            old["record"][data["covariance_index"][ids]], data["record"][ids]
        )
        np.testing.assert_array_equal(
            old["end"][data["covariance_index"][ids]], data["end"][ids]
        )
        for arm in ("untreated_covariance", "log_shape_tangent", "congruence_dynamics"):
            with np.load(
                baseline / (room + "-" + arm + "-model-prediction.npz"),
                allow_pickle=False,
            ) as z:
                np.testing.assert_array_equal(z["held_indices"], original)
                p, d = z["pose"][old_ids], z["delta"][old_ids]
            comparators[room]["covariance"][arm] = dict(
                scope="ONLYhiprelativepose/Delta; no fullpose/root heads; originally trained on fullcovariancemask",
                pose_relative=current_metrics(p, data["pose_relative"][ids]),
                delta_relative=displacement_metrics(
                    d,
                    data["delta_relative"][ids],
                    data["pose_full"][ids],
                    data["record"][ids],
                ),
            )
    selected = {
        room: {
            name: results[room][frozen[room]["natural_selected"][name]]["aligned"][name]
            for name in SLICES
        }
        for room in CONFIG["folds"]
    }

    def rawr2(room, arm):
        return results[room][arm]["aligned"]["delta_full"]["raw_zero_reference_r2"]

    motion_gate = all(
        selected[room]["delta_full"]["raw_zero_reference_r2"] is not None
        and selected[room]["delta_full"]["raw_zero_reference_r2"] > 0
        and selected[room]["delta_full"]["raw_zero_reference_r2"]
        > results[room][frozen[room]["natural_selected"]["delta_full"]][
            "within_record_prediction_shuffle"
        ]["delta_full"]["raw_zero_reference_r2"]
        and all(
            selected[room]["delta_full"]["raw_mse"]
            <= results[room][control]["aligned"]["delta_full"]["raw_mse"]
            for control in ("within_frame_shuffle", "whole_window_shuffle")
        )
        for room in CONFIG["folds"]
    ) and all(
        any(
            selected[room]["delta_full"]["raw_mse"]
            < results[room][control]["aligned"]["delta_full"]["raw_mse"]
            for room in CONFIG["folds"]
        )
        for control in ("within_frame_shuffle", "whole_window_shuffle")
    )
    pose_gate = all(
        selected[room]["pose_full"]["normalized_joint_error"]
        < comparators[room]["source_mean"]["pose_full"]["normalized_joint_error"]
        and all(
            selected[room]["pose_full"]["pck_curve"][threshold]
            >= comparators[room]["source_mean"]["pose_full"]["pck_curve"][threshold]
            for threshold in ("0.05", "0.2")
        )
        for room in CONFIG["folds"]
    )
    for name, expected in hashes.items():
        if digest(ROOT / name) != expected:
            raise ValueError("Executedsource changed")
    for name, expected in baseline_hashes.items():
        if digest(baseline / name) != expected:
            raise ValueError("Baseline changed")
    for name, expected in receipt["arrays_sha256"].items():
        if digest(cache / name) != expected:
            raise ValueError("Featurecache changed")
    save(
        out / "results.json",
        dict(
            config=CONFIG,
            results=results,
            selected_natural=selected,
            choices=frozen,
            comparators=comparators,
            motion_gate_passed=bool(motion_gate),
            separate_pose_gate_passed=bool(pose_gate),
            seconds=time.monotonic() - started,
        ),
    )
    save(
        out / "verification.json",
        dict(
            saved_state_prediction_max_abs=parity,
            source_input_rehash_passed=True,
            models_verified=12,
            target_readouts_verified=48,
        ),
    )
    save(
        out / "artifact-manifest.json",
        {
            str(p.relative_to(out)): digest(p)
            for p in sorted(out.rglob("*"))
            if p.is_file()
        },
    )
    print(
        json.dumps(
            dict(
                motion_gate=motion_gate,
                pose_gate=pose_gate,
                seconds=time.monotonic() - started,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--base", type=Path, default=ROOT / "work/domain-source30-rf-v1")
    b.add_argument(
        "--covariance", type=Path, default=ROOT / "work/domain-covariance-full-cache-v1"
    )
    b.add_argument(
        "--full-probe", type=Path, default=ROOT / "work/domain-covariance-full27-v1"
    )
    b.add_argument("--out", type=Path, required=True)
    f = sub.add_parser("fit")
    f.add_argument("--cache", type=Path, required=True)
    f.add_argument(
        "--baseline", type=Path, default=ROOT / "work/domain-covariance-full27-v1"
    )
    f.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args.base, args.covariance, args.full_probe, args.out)
    else:
        fit(args.cache, args.baseline, args.out)
