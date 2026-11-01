"""Offline source-only RF/pose trajectory-geometry diagnostic, never a predictor.

Whole-clip self distances are compared with teacher distances only for scoring.
No alignment/model is fitted. Lag-residual and circular-index controls distinguish
geometric correspondence from a shared clock/autocorrelation where possible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.covariance_dynamics import CONDITION_LIMIT, relative_log_spectrum  # noqa:E402
from scripts.domain_covariance_full_cache import record_blocks  # noqa:E402
from scripts.domain_covariance_source_probe import digest, validate_records  # noqa:E402

CONFIG = dict(
    scope="OFFLINE SOURCE DEVELOPMENT diagnostic only; no alignment/prediction/model fit, no E04 or calibrated/live efficacy claim",
    records=90,
    selected_actions=["A03", "A07", "A12"],
    fixed_first_record="E01/S01/A12",
    extraction="Untreated raw cached covariance features[N,8times,19bins,7states,9coordinates]: LAST time,current state0; no fitted normalization",
    covariance_coordinates="3real diagonals, sqrt2*real(01,02,12), sqrt2*imag(01,02,12); inverse uses /sqrt2 and exact Hermitian lower triangle",
    endpoints="Use actual quality-retained canonical39:8:296 endpoints. Never resurrect rejected histories or the allocated tail",
    precision="Cached covariance coordinates are float32, decoded float64/complex128; reject newly nonpositive or condition>1e10 states, no I jitter",
    arms=["det_normalized_paired_spectrum", "raw_covariance_airm"],
    rf_distance="sqrt(mean_over19bins(sum_over3(log generalized eigenvalues squared))); first arm uses existing determinant normalization/paired shrinkage/centered spectrum, second raw SPD eigenvalues without normalization/shrinkage",
    teacher_distance="Euclidean RMS over34 hip-root-relative coordinate differences, from supplied 2D annotations; teacher labels used ONLY for diagnostic scoring",
    correlation="Upper triangle excluding diagonal. Raw Spearman=Pearson of average ranks. Residual=Pearson after independently subtracting each exact endpoint-time-lag group's mean from both global rank vectors",
    clock_control="Report Spearman of actual endpoint lag vs RF distance and vs teacher distance. Time/record/action IDs are score/control metadata, never model features",
    circular_null="Every1..n-1 circular RF-index shift permutes BOTH distance axes against fixed teachers/times; preserves observed-index RF autocorrelation, not exact physical-time autocorrelation when endpoints are missing",
    null_quantile="95th percentile of finite residual correlations, NumPy linear method; all null vectors retained; descriptive, not an independence-based pvalue",
    gate="PER ARM: fixedA12 residual>=.3 AND aligned-minus95th-null>=.1; all90record residuals defined, median>=.1 and at least60positive. Missing/degenerate records prevent continuation; no best-arm-per-record pooling",
    limits="Repeated participants/actions, correlated pair distances, source-only post-motivation protocol. Fixed receiver-mixing assumption unproven across actual rooms. Static pose/orientation/DC unknown; no physical AoA/Doppler/3D claim",
)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reconstruct(coordinates):
    coordinates = np.asarray(coordinates, dtype=np.float64)
    require(
        coordinates.shape[-1] == 9 and np.isfinite(coordinates).all(),
        "Finite9-coordinate Hermitian encoding required",
    )
    result = np.zeros((*coordinates.shape[:-1], 3, 3), dtype=np.complex128)
    result[..., np.arange(3), np.arange(3)] = coordinates[..., :3]
    off = (coordinates[..., 3:6] + 1j * coordinates[..., 6:9]) / np.sqrt(2)
    result[..., (0, 0, 1), (1, 2, 2)] = off
    result[..., (1, 2, 2), (0, 0, 1)] = off.conj()
    return result


def current_covariances(features):
    features = np.asarray(features)
    require(
        features.ndim == 2 and features.shape[1] == 9576,
        "Expected untreated9576feature columns",
    )
    return reconstruct(features.reshape(-1, 8, 19, 7, 9)[:, -1, :, 0, :])


def accepted_states(covariance):
    eigenvalues = np.linalg.eigvalsh(covariance)
    positive = (eigenvalues[..., 0] > 0).all(-1)
    condition = np.full(eigenvalues.shape[:-1], np.inf)
    np.divide(
        eigenvalues[..., -1],
        eigenvalues[..., 0],
        out=condition,
        where=eigenvalues[..., 0] > 0,
    )
    return positive & (condition <= CONDITION_LIMIT).all(-1), condition.max(-1)


def rf_distances(covariance, arm):
    covariance = np.asarray(covariance, dtype=np.complex128)
    require(
        covariance.ndim == 4
        and covariance.shape[1:] == (19, 3, 3)
        and len(covariance) >= 3,
        "At least3 accepted19bin covariance states required",
    )
    accepted, _ = accepted_states(covariance)
    require(accepted.all(), "Non-SPD or ill-conditioned covariance; no identity jitter")
    current, past = covariance[:, None], covariance[None, :]
    if arm == "det_normalized_paired_spectrum":
        spectrum = relative_log_spectrum(current, past)
    elif arm == "raw_covariance_airm":
        inverse = np.linalg.inv(np.linalg.cholesky(past))
        whitened = inverse @ current @ inverse.conj().swapaxes(-1, -2)
        whitened = (whitened + whitened.conj().swapaxes(-1, -2)) / 2
        eigenvalues = np.linalg.eigvalsh(whitened)
        require(
            np.isfinite(eigenvalues).all() and (eigenvalues > 0).all(),
            "Unstable raw AIRM generalized spectrum; no eigenvalue clipping",
        )
        spectrum = np.log(eigenvalues)
    else:
        raise ValueError("Unknown RF distance arm")
    distance = np.sqrt(np.square(spectrum).sum(-1).mean(-1))
    asymmetry = float(np.max(abs(distance - distance.T)))
    require(
        np.isfinite(distance).all()
        and asymmetry <= 1e-6 * max(1.0, float(distance.max())),
        "Pair-distance numerical asymmetry exceeds tolerance",
    )
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance, 0)
    return distance, asymmetry


def pose_distances(pose):
    pose = np.asarray(pose, dtype=np.float64)
    require(
        pose.ndim == 3 and pose.shape[1:] == (17, 2) and np.isfinite(pose).all(),
        "Finite17x2 teacher coordinates required",
    )
    relative = pose - pose[:, [11, 12]].mean(1, keepdims=True)
    return np.sqrt(np.square(relative[:, None] - relative[None, :]).mean((2, 3)))


def correlation(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a - a.mean(), b - b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return (
        float(np.clip(np.dot(a, b) / denominator, -1, 1))
        if denominator > 1e-12
        else None
    )


def residual_ranks(vector, lags):
    values = rankdata(vector, method="average")
    for lag in np.unique(lags):
        mask = lags == lag
        values[mask] -= values[mask].mean()
    return values


def association(rf, teacher, ends):
    require(
        rf.shape == teacher.shape == (len(ends), len(ends)),
        "Distance/time shapes differ",
    )
    require(np.all(np.diff(ends) > 0), "Endpoints must strictly increase")
    i, j = np.triu_indices(len(ends), 1)
    lags = ends[j] - ends[i]
    a, b = rf[i, j], teacher[i, j]
    ar, br = rankdata(a, method="average"), rankdata(b, method="average")
    return dict(
        raw_spearman=correlation(ar, br),
        lag_residual_rank_correlation=correlation(
            residual_ranks(a, lags), residual_ranks(b, lags)
        ),
        rf_time_lag_spearman=correlation(ar, rankdata(lags, method="average")),
        teacher_time_lag_spearman=correlation(br, rankdata(lags, method="average")),
        pairs=len(i),
        lag_groups=len(np.unique(lags)),
    )


def controls(rf, teacher, ends):
    aligned = association(rf, teacher, ends)
    shifts = []
    for shift in range(1, len(ends)):
        order = np.roll(np.arange(len(ends)), shift)
        shifts.append(
            dict(shift=shift, **association(rf[np.ix_(order, order)], teacher, ends))
        )
    values = [
        r["lag_residual_rank_correlation"]
        for r in shifts
        if r["lag_residual_rank_correlation"] is not None
    ]
    q95 = float(np.quantile(values, 0.95, method="linear")) if values else None
    residual = aligned["lag_residual_rank_correlation"]
    return dict(
        aligned=aligned,
        circular_shifts=shifts,
        finite_residual_nulls=len(values),
        residual_null_q95=q95,
        residual_above_null_q95=residual - q95
        if residual is not None and q95 is not None
        else None,
    )


def gate(rows, arm):
    selected = [r["arms"][arm] for r in rows]
    values = [
        r.get("aligned", {}).get("lag_residual_rank_correlation") for r in selected
    ]
    valid = [v for v in values if v is not None]
    fixed = next(
        r["arms"][arm] for r in rows if r["record"] == CONFIG["fixed_first_record"]
    )
    fixed_value = fixed.get("aligned", {}).get("lag_residual_rank_correlation")
    drop = fixed.get("residual_above_null_q95")
    median = float(np.median(valid)) if valid else None
    positive = sum(v > 0 for v in valid)
    return dict(
        passed=bool(
            len(rows) == 90
            and len(valid) == 90
            and fixed_value is not None
            and drop is not None
            and fixed_value >= 0.3
            and drop >= 0.1
            and median >= 0.1
            and positive >= 60
        ),
        records=len(rows),
        defined_residuals=len(valid),
        positive_residuals=positive,
        median_defined_residual=median,
        fixed_A12_residual=fixed_value,
        fixed_A12_residual_above_q95=drop,
    )


def run(cache, proof_root, out):
    if out.exists():
        raise FileExistsError(out)
    started = time.monotonic()
    receipt = json.loads((cache / "receipt.json").read_text())
    proof = json.loads((proof_root / "verification.json").read_text())
    proof_registration = json.loads((proof_root / "registration.json").read_text())
    validate_records(receipt["records"])
    require(
        proof["total_records_feature_verified"] == 810
        and proof["models_verified"] == 9
        and proof["input_and_artifact_hashes_verified"]
        and proof["maximum_saved_prediction_difference"] == 0,
        "Completed810record independent source verification required",
    )
    require(
        proof_registration["feature_cache_receipt_sha256"]
        == digest(cache / "receipt.json"),
        "Proof/cache identity differs",
    )
    bindings = {
        str(cache / name): digest(cache / name)
        for name in (
            "receipt.json",
            "targets-and-metadata.npz",
            "feature-quality.json",
            "untreated_covariance-features.npy",
        )
    }
    for name in (
        "targets-and-metadata.npz",
        "feature-quality.json",
        "untreated_covariance-features.npy",
    ):
        require(
            bindings[str(cache / name)] == receipt["arrays_sha256"][name],
            "Actual cache bytes differ: " + name,
        )
    for name in ("verification.json", "registration.json"):
        bindings[str(proof_root / name)] = digest(proof_root / name)
    with np.load(cache / "targets-and-metadata.npz", allow_pickle=False) as z:
        meta = {key: z[key] for key in z.files}
    quality = json.loads((cache / "feature-quality.json").read_text())
    blocks = record_blocks(meta, quality, [r["id"] for r in receipt["records"]])
    require(
        len(meta["pose"]) == receipt["logical_rows"], "Logical metadata prefix differs"
    )
    feature = np.load(
        cache / "untreated_covariance-features.npy", mmap_mode="r", allow_pickle=False
    )
    require(
        feature.shape == (receipt["nominal_rows"], 9576)
        and feature.dtype == np.dtype("float32"),
        "Unexpected feature allocation",
    )
    records = [
        name for name in blocks if name.split("/")[2] in CONFIG["selected_actions"]
    ]
    require(len(records) == 90, "Expected fixed90records")
    records.remove(CONFIG["fixed_first_record"])
    records.insert(0, CONFIG["fixed_first_record"])
    sources = {
        str(p.relative_to(ROOT)): digest(p)
        for p in (
            Path(__file__),
            ROOT / "ghostmesh/covariance_dynamics.py",
            ROOT / "scripts/domain_covariance_full_cache.py",
            ROOT / "scripts/domain_covariance_source_probe.py",
            ROOT / "scripts/domain_background_probe.py",
        )
    }
    out.mkdir(parents=True)
    save(
        out / "registration.json",
        dict(
            config=CONFIG,
            record_order=records,
            inputs_sha256=bindings,
            source_sha256=sources,
            numpy=np.__version__,
        ),
    )
    for name in sources:
        dest = out / "executed-source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    rows = []
    for name in records:
        block, _ = blocks[name]
        original_indices = np.arange(block.start, block.stop)
        covariance = current_covariances(feature[block])
        accepted, condition = accepted_states(covariance)
        indices = original_indices[accepted]
        ends = meta["end"][indices]
        teacher = pose_distances(meta["pose"][indices])
        saved = dict(
            indices=indices,
            ends=ends,
            source_indices=original_indices,
            rejected_after_float32=original_indices[~accepted],
            condition_per_source_node=condition,
            teacher_distance=teacher,
        )
        row = dict(
            record=name,
            source_nodes=len(original_indices),
            accepted_nodes=len(indices),
            endpoints=ends.tolist(),
            post_float32_rejected_endpoints=meta["end"][
                original_indices[~accepted]
            ].tolist(),
            arms={},
        )
        for arm in CONFIG["arms"]:
            try:
                distance, asymmetry = rf_distances(covariance[accepted], arm)
                row["arms"][arm] = dict(
                    status="ok",
                    maximum_distance_asymmetry=asymmetry,
                    **controls(distance, teacher, ends),
                )
                saved[arm + "_distance"] = distance
            except ValueError as error:
                row["arms"][arm] = dict(status="unavailable", reason=str(error))
        np.savez_compressed(out / (name.replace("/", "-") + ".npz"), **saved)
        rows.append(row)
        save(out / "progress.json", dict(completed=len(rows), results=rows))
        print(
            json.dumps(
                dict(
                    record=name,
                    nodes=len(indices),
                    residual={
                        a: r.get("aligned", {}).get("lag_residual_rank_correlation")
                        for a, r in row["arms"].items()
                    },
                )
            ),
            flush=True,
        )
    for name, expected in sources.items():
        require(digest(ROOT / name) == expected, "Source changed during diagnostic")
    for name, expected in bindings.items():
        require(digest(Path(name)) == expected, "Evidence changed during diagnostic")
    save(
        out / "results.json",
        dict(
            config=CONFIG,
            results=rows,
            continuation_gates={arm: gate(rows, arm) for arm in CONFIG["arms"]},
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path, default=ROOT / "work/domain-covariance-full-cache-v1"
    )
    parser.add_argument(
        "--proof", type=Path, default=ROOT / "work/domain-covariance-full27-v1"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.cache, args.proof, args.out)
