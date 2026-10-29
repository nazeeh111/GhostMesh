"""Input-local log-power-fraction normalization under a conditional gain model.

No RF/FFT redo, target normalization, query-dataset statistics or new labels.
Completed raw-power results remain a fixed matched-representation reference.
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
from scripts import domain_packet_transfer_probe as raw  # noqa:E402

ARMS, SLICES, VARIANTS = raw.ARMS, raw.SLICES, raw.VARIANTS
save, digest = raw.save, raw.digest
fold_masks, target_matrix = raw.fold_masks, raw.target_matrix
fit_map, fit_readouts, predict = raw.fit_map, raw.fit_readouts, raw.predict
transform, stratified, permutation = raw.transform, raw.stratified, raw.permutation
current_metrics, displacement_metrics = raw.current_metrics, raw.displacement_metrics
endpoint_control = raw.endpoint_control
CONFIG = dict(
    raw.CONFIG,
    representation="Per-window signed-band log-power fractions: each of 38 link/subcarrier-bin vectors normalizes over its 10 signed bands",
    formula="x-(10/ln10)*logsumexp(x*ln10/10), separately on the last 10-band axis; stable max subtraction; no example pooling",
    conditional_invariance="Any common additive dB shift to all 10 cached band powers per link/bin cancels. Physical interpretation requires window-constant gain common across bands and averaged subcarriers",
    limitations_added="Clipping, numerical pre-log floors, time-varying gain and unequal per-subcarrier gains may break the physical gain assumption. Total dynamic power is discarded and may carry useful motion information. No universal room invariance claim",
    lineage_added="Predeclared after negative raw-power full-source run; raw reference fixed. This is source development, not an untouched final hypothesis test. Same four arms/targets, rows, seeds, grids, gates and source-choice freeze",
)


def log_power_fractions(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim < 1 or x.shape[-1] != 380 or not np.isfinite(x).all():
        raise ValueError("Finite380D rawlog-bandpower features required")
    bands = x.reshape(*x.shape[:-1], 2, 19, 10)
    logpower = bands * (np.log(10.0) / 10.0)
    maximum = logpower.max(-1, keepdims=True)
    normalizer = (
        maximum + np.log(np.exp(logpower - maximum).sum(-1, keepdims=True))
    ) * (10.0 / np.log(10.0))
    return (bands - normalizer).reshape(x.shape)


def sources(out):
    paths = [
        Path(__file__),
        ROOT / "scripts/domain_packet_transfer_probe.py",
        ROOT / "ghostmesh/doppler_features.py",
        ROOT / "ghostmesh/metrics.py",
        ROOT / "scripts/domain_background_probe.py",
        ROOT / "scripts/domain_covariance_cached_probe.py",
        ROOT / "scripts/domain_covariance_full_cache.py",
        ROOT / "scripts/domain_covariance_source_probe.py",
    ]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    for source in paths:
        dest = out / "executed-source" / source.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    return hashes


def normalize_cache(cache, out):
    if out.exists():
        raise FileExistsError(out)
    receipt, data, features = raw.load_cache(cache)
    out.mkdir(parents=True)
    hashes = sources(out)
    save(
        out / "registration.json",
        dict(
            config=CONFIG,
            source_sha256=hashes,
            raw_spectral_receipt_sha256=digest(cache / "receipt.json"),
            raw_arrays_sha256=receipt["arrays_sha256"],
        ),
    )
    for name in ("metadata.npz", "quality.json"):
        shutil.copyfile(cache / name, out / name)
    floor = {}
    for variant, value in features.items():
        normalized = log_power_fractions(value)
        np.save(
            out / (variant + ".npy"), normalized.astype(np.float32), allow_pickle=False
        )
        floor[variant] = dict(
            fraction_raw_bands_at_numericalfloor=float(np.mean(value <= -120.0)),
            fraction_vectors_all10bands_at_floor=float(
                np.mean((value.reshape(-1, 2, 19, 10) <= -120.0).all(-1))
            ),
        )
    arrays = {
        name: digest(out / name)
        for name in ["metadata.npz", "quality.json", *[v + ".npy" for v in VARIANTS]]
    }
    for name, expected in hashes.items():
        if digest(ROOT / name) != expected:
            raise ValueError("Normalization source changed")
    for name, expected in receipt["arrays_sha256"].items():
        if digest(cache / name) != expected:
            raise ValueError("Raw feature cache changed")
    save(
        out / "receipt.json",
        dict(
            receipt,
            config=CONFIG,
            representation="log_power_fractions",
            source_sha256=hashes,
            raw_spectral_receipt_sha256=digest(cache / "receipt.json"),
            raw_arrays_sha256=receipt["arrays_sha256"],
            arrays_sha256=arrays,
            floor_diagnostics=floor,
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
    receipt, data, features = raw.load_cache(cache)
    if receipt.get("representation") != "log_power_fractions":
        raise ValueError("Expected registered logpowerfraction cache")
    return receipt, data, features


def fit(cache, baseline, raw_experiment, out):
    if out.exists():
        raise FileExistsError(out)
    raw_reference_sha256 = digest(raw_experiment / "results.json")
    started = time.monotonic()
    receipt, data, features = load_cache(cache)
    raw_registration = json.loads((raw_experiment / "registration.json").read_text())
    if (
        raw_registration["spectral_cache_receipt_sha256"]
        != receipt["raw_spectral_receipt_sha256"]
        or raw_registration["spectral_cache_arrays_sha256"]
        != receipt["raw_arrays_sha256"]
    ):
        raise ValueError("Fixed raw experiment used a different spectral cohort")
    for name in (
        "seed",
        "alphas",
        "feature_floor",
        "rff_dimensions",
        "arms",
        "targets",
        "folds",
    ):
        expected = json.loads(json.dumps(CONFIG[name]))
        if raw_registration["config"][name] != expected:
            raise ValueError("Fixed raw experiment used a different protocol: " + name)
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
            raw_power_reference_results_sha256=raw_reference_sha256,
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
    if digest(raw_experiment / "results.json") != raw_reference_sha256:
        raise ValueError("Fixed raw-power reference changed")
    raw_reference = json.loads((raw_experiment / "results.json").read_text())
    if raw_reference["config"]["targets"] != CONFIG["targets"]:
        raise ValueError("Raw reference target protocol differs")
    raw_comparison = dict(
        results_sha256=raw_reference_sha256,
        selected_natural=raw_reference["selected_natural"],
        choices=raw_reference["choices"],
        aligned={
            room: {
                arm: {name: row["aligned"][name] for name in (*SLICES, "pose_relative")}
                for arm, row in methods.items()
            }
            for room, methods in raw_reference["results"].items()
        },
    )
    save(
        out / "results.json",
        dict(
            config=CONFIG,
            raw_power_comparison=raw_comparison,
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
    n = sub.add_parser("normalize")
    n.add_argument(
        "--cache", type=Path, default=ROOT / "work/domain-packet-full-cache-v1"
    )
    n.add_argument("--out", type=Path, required=True)
    f = sub.add_parser("fit")
    f.add_argument("--cache", type=Path, required=True)
    f.add_argument(
        "--baseline", type=Path, default=ROOT / "work/domain-covariance-full27-v1"
    )
    f.add_argument(
        "--raw-experiment", type=Path, default=ROOT / "work/domain-packet-transfer-v1"
    )
    f.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "normalize":
        normalize_cache(args.cache, args.out)
    else:
        fit(args.cache, args.baseline, args.raw_experiment, args.out)
