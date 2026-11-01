"""Audit the conditional ten-column packet clock before temporal spectral fits.

Audit mode reads typed source RF only, never pose arrays. About100Hz is a source
report, not independently timestamp-verified sampling of these released arrays.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.doppler_features import (  # noqa:E402
    BAND_EDGES,
    COLUMN_ORDER,
    POWER_FLOOR,
    SAMPLE_RATE,
    WHOLE_ORDER,
    clock_audit,
    complex_quotients,
)
from scripts.domain_background_probe import digest  # noqa:E402
from scripts.domain_covariance_source_probe import validate_records  # noqa:E402

CONFIG = dict(
    scope="SOURCE DEVELOPMENT ONLY RFclockaudit: fixed E01/S01/A12 and90A03/A07/A12records/all30sourcepeople E01-E03; noE04",
    source="https://arxiv.org/html/2305.10345v2#S3.SS1",
    paper_report="About100HzfirmwareCSI, ten-column100ms matrices; ROSclosestframes synchronizedat10Hz. Exactindividualpacket timestamps/order notpresentinthiscache",
    uncertainty="Augmentation/resampling and frameboundarytiming/phasecoherence are not independently established. Continuitycannotprovetrueclock/order",
    nominal_sample_rate=SAMPLE_RATE,
    reference_receiver=0,
    ratio_links=[1, 2],
    magnitude_units="log_magnitude_db",
    phase_units="radians",
    ratio_log10_clip=[-2.0, 2.0],
    column_permutation=COLUMN_ORDER.tolist(),
    whole80_permutation=WHOLE_ORDER.tolist(),
    audit="Noimputation: onlyfinite neighboringcomplexquotientpairs; withinframeadjacency, last→nextfirstboundary, fixed10columnshuffle; complexstep/phase/amplitude/correlation",
    clock_gate="A12 AND>=75percentof90records: medianwithincomplexstep<fixedcolumnshuffledstep AND boundarymedian<10*max(withinmedian,1e-12). Heuristicplausibilityonly, nottimestampverification",
    prospective_spectrum="8causalframes×10columns=80nominalsamples; perlink/subcarriercomplexmeanremove,Hann80,FFT; integrate5frequencybandsperSIGN, dropDC/Nyquist; averagePOWERSover6adjacentsubcarriers19bins; log10powerfloor",
    signed_abs_band_edges_hz=list(BAND_EDGES),
    power_floor=POWER_FLOOR,
    feature_dimensions=380,
    spectral_missing_policy="Allvariants sharefinitecount: at least76of80values per2link/subcarriertrace; causalwindowcomplexmeanfillmissing≤5percent. AuditNEVERfills. Noamplitude-onlyflatguard",
    positive_control=dict(
        record="E01/S01/A12",
        raw_blocks_half_open=dict(
            train=[0, 118], validation=[126, 178], test=[186, 297]
        ),
        pose_maximum_counts=dict(train=111, validation=45, test=104),
        displacement_maximum_counts=dict(train=110, validation=44, test=103),
        displacement="Lag8 providedpose(t)-pose(t-8); BOTHendpointlabels lieinsideownrawblock; noGTpastinput, noquerymeans; noclipping",
        variants=["natural", "within_frame_shuffle", "whole_window_shuffle"],
        kernels=["linear", "rbf"],
        ridge_lambdas=[1e-5, 0.001, 0.1, 10.0],
        normalization="TRAINcolumnmean/SDfloor.05; TRAINmedianRBFdistance only. DeltaTRAINmean/SDfloor.001 for numericalconditioning, rawvalidationDeltaMSEselection",
        pose_selection="Withinfeature/kernel lowestvalidationNJE, naturalbestkernelvalidationNJEonly",
        displacement_selection="Withinfeature/kernel lowestvalidationRAWfulljointDeltaMSE; naturalbestkernel samevalidationonly",
        pose_gate="NaturalrelativeposeR2>=.5, NJE<=.75staticmean, aligned-minus-withinrecordpredictionshuffledrelativeR2>=.3",
        motion_gate="NaturalrawfulljointDeltaR2>=.1 AND alignedminuswithinrecordpredictionshuffledrawDeltaR2>=.2",
        controls="Poseexistingfrozenraw/quotRBFreferences; DeltaPAIREDactualRFraw/quotlinear/RBFcontrols withsame110/44/103maxcounts pluszero/TRAINmeanDelta",
        refinement="Additionalfulljoint/root/hiprelative displacementtestdeclaredBEFOREANYpose/Deltafit; motionfeatures neednotidentifyunknownDC/currentpose",
        no_claim="Localoneperson/room/action only. Packetorderadvantage reported, notassumed. NoBVP/AoA/velocity/3D/room-generalizationclaim",
    ),
)


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def snapshot(out):
    paths = [
        Path(__file__),
        ROOT / "ghostmesh/doppler_features.py",
        ROOT / "scripts/domain_background_probe.py",
        ROOT / "scripts/domain_covariance_source_probe.py",
    ]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    for source in paths:
        destination = out / "executed-source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return hashes


def audit(cache, out):
    if out.exists():
        raise FileExistsError(out)
    started = time.monotonic()
    receipt = json.loads((cache / "receipt.json").read_text())
    validate_records(receipt["records"])
    records = [
        r for r in receipt["records"] if r["id"].split("/")[2] in ("A03", "A07", "A12")
    ]
    if len(records) != 90:
        raise ValueError("Expected90fixedsource records")
    if digest(cache / "rf.npy") != receipt["arrays"]["rf"]:
        raise ValueError("ActualRFcache changed")
    out.mkdir(parents=True)
    sources = snapshot(out)
    registration = dict(
        config=CONFIG,
        source_sha256=sources,
        cache_receipt_sha256=digest(cache / "receipt.json"),
        rf_array_sha256=receipt["arrays"]["rf"],
        poses_read=False,
    )
    save(out / "registration.json", registration)
    raw = np.load(cache / "rf.npy", mmap_mode="r", allow_pickle=False)
    if raw.shape != (240570, 6, 114, 10):
        raise ValueError("UnexpectedtypedsourceRFshape")
    results = []
    for i, row in enumerate(records, 1):
        selected = np.asarray(raw[row["offset"] : row["offset"] + 297])
        q, quality = complex_quotients(
            selected, magnitude_units="log_magnitude_db", phase_units="radians"
        )
        result = dict(
            record=row["id"],
            RFslice_sha256=sha256(np.ascontiguousarray(selected).tobytes()).hexdigest(),
            quality=quality,
            **clock_audit(q),
        )
        results.append(result)
        if i % 10 == 0:
            print(
                json.dumps(
                    dict(audited=i, total=90, seconds=time.monotonic() - started)
                ),
                flush=True,
            )
    a12 = next(r for r in results if r["record"] == "E01/S01/A12")
    passed = sum(r["plausible_nominal_continuity"] for r in results)
    gate = bool(a12["plausible_nominal_continuity"] and passed / 90 >= 0.75)
    for name, expected in sources.items():
        if digest(ROOT / name) != expected:
            raise ValueError("Executedsourcechanged")
    if digest(cache / "rf.npy") != registration["rf_array_sha256"]:
        raise ValueError("RFcachechanged")
    summary = dict(
        passed=gate,
        A12_passed=a12["plausible_nominal_continuity"],
        passing_records=passed,
        total_records=90,
        passing_fraction=passed / 90,
        limitation="RFcontinuityheuristicdoesNOTverifyreleasedpacketclock/order/phasecoherenceorphysicalDoppler",
    )
    save(
        out / "results.json",
        dict(
            registration=registration,
            clock_gate=summary,
            records=results,
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
    print(json.dumps(summary), flush=True)


# The following stage is registered BEFORE opening any supplied pose array.
def window_designs(raw):
    from ghostmesh.doppler_features import spectral_features
    from scripts.domain_same_condition_probe import record_features

    q, _ = complex_quotients(
        raw, magnitude_units="log_magnitude_db", phase_units="radians"
    )
    blocks = CONFIG["positive_control"]["raw_blocks_half_open"]
    variants = CONFIG["positive_control"]["variants"]
    values = {name: [] for name in variants}
    ends = []
    splits = []
    delta_eligible = []
    rejected = []
    packet = record_features(raw)
    extras = {name: [] for name in ("raw_db", "quotient")}
    for split, (start, stop) in blocks.items():
        for end in range(start + 7, stop):
            try:
                current = {
                    variant: spectral_features(q[end - 7 : end + 1], variant)
                    for variant in variants
                }
            except ValueError as exc:
                rejected.append(dict(split=split, end=end, reason=str(exc)))
                continue
            for variant in variants:
                values[variant].append(current[variant])
            for name in extras:
                extras[name].append(packet[name][end - 7 : end + 1].reshape(-1))
            ends.append(end)
            splits.append(split)
            delta_eligible.append(end >= start + 8)
    return (
        {k: np.asarray(v) for k, v in (values | extras).items()},
        np.asarray(ends),
        np.asarray(splits),
        np.asarray(delta_eligible),
        rejected,
    )


def delta_score(prediction, truth, body):
    scale = np.linalg.norm(body.max(1) - body.min(1), axis=1)
    if np.any(scale <= 1e-6):
        raise ValueError("Degenerate current reference body")
    rp = prediction[:, [11, 12]].mean(1, keepdims=True)
    ry = truth[:, [11, 12]].mean(1, keepdims=True)
    result = {}
    for name, p, y in [
        ("full_joint", prediction, truth),
        ("root", rp, ry),
        ("hip_relative", prediction - rp, truth - ry),
    ]:
        denominator = float(np.square(y).sum())
        pc = p - p.mean(0)
        yc = y - y.mean(0)
        centered = float(np.square(yc).sum())
        result[name] = dict(
            raw_zero_reference_r2=1 - float(np.square(p - y).sum()) / denominator
            if denominator
            else None,
            record_centered_r2=1 - float(np.square(pc - yc).sum()) / centered
            if centered
            else None,
            mse=float(np.square(p - y).mean()),
            true_squared_energy=denominator,
            normalized_displacement_error=float(
                (np.linalg.norm(p - y, axis=2) / scale[:, None]).mean()
            ),
        )
    return result


def fit_kernel(x, target, body, split, task, feature, out):
    from scipy.linalg import eigh
    from scripts import domain_same_condition_probe as base

    train, val, test = (split == k for k in ("train", "validation", "test"))
    transformed, normalizer = base.normalize_fit(x, train)
    kernels, _, bandwidth = base.kernel_inputs(transformed, train)
    y = target.reshape(-1, 34)
    mean = y[train].mean(0)
    scale = np.maximum(y[train].std(0), 0.001 if task == "delta" else 0.01)
    standardized = (y - mean) / scale
    names = np.repeat("E01/S01/A12", len(y))
    rows = []
    parity = {}
    for kernel, matrix in kernels.items():
        eigen, vectors = eigh(matrix[train], check_finite=True)
        right = base.multiply(vectors.T, standardized[train])
        candidates = []
        all_duals = []
        best = None
        residual_max = 0.0
        direct_max = 0.0
        for regularization in CONFIG["positive_control"]["ridge_lambdas"]:
            dual = base.multiply(vectors, right / (eigen[:, None] + regularization))
            system = matrix[train] + regularization * np.eye(train.sum())
            residual = float(
                np.linalg.norm(base.multiply(system, dual) - standardized[train])
                / max(np.linalg.norm(standardized[train]), 1e-12)
            )
            direct = np.linalg.solve(system, standardized[train])
            difference = float(
                np.linalg.norm(direct - dual) / max(np.linalg.norm(dual), 1e-12)
            )
            residual_max = max(residual_max, residual)
            direct_max = max(direct_max, difference)
            if residual > 1e-6 or difference > 1e-6:
                raise ValueError("Kernel equation/directsolve mismatch")
            raw = (mean + base.multiply(matrix, dual) * scale).reshape(-1, 17, 2)
            prediction = raw.clip(0, 1) if task == "pose" else raw
            selection = (
                base.score(prediction[val], target[val], names[val])["metrics"][
                    "normalized_joint_error"
                ]
                if task == "pose"
                else float(np.square(prediction[val] - target[val]).mean())
            )
            candidates.append(
                dict(
                    regularization=regularization,
                    validation_criterion=selection,
                    normal_equation_residual=residual,
                    directsolve_relative_difference=difference,
                )
            )
            all_duals.append(dual)
            if best is None or selection < best[0]:
                best = (selection, regularization, dual, prediction, raw)
        _, selected, dual, prediction, raw = best
        tag = task + "-" + feature + "-" + kernel
        np.savez_compressed(
            out / (tag + "-model.npz"),
            **normalizer,
            training_design=transformed[train],
            dual=np.stack(all_duals),
            regularizations=np.asarray(CONFIG["positive_control"]["ridge_lambdas"]),
            selected_regularization=np.asarray(selected),
            target_mean=mean,
            target_scale=scale,
            bandwidth_squared=np.asarray(bandwidth),
        )
        permutation = np.random.default_rng(2027).permutation(test.sum())
        shuffled = prediction[test][permutation]
        scores = {}
        for name, pick in [("train", train), ("validation", val), ("test", test)]:
            scores[name] = (
                base.score(prediction[pick], target[pick], names[pick])
                if task == "pose"
                else delta_score(prediction[pick], target[pick], body[pick])
            )
        control = (
            base.score(shuffled, target[test], names[test])
            if task == "pose"
            else delta_score(shuffled, target[test], body[test])
        )
        np.savez_compressed(
            out / (tag + "-predictions.npz"),
            prediction=prediction,
            raw_prediction=raw,
            target=target,
            shuffle_indices=permutation,
            shuffled_prediction=shuffled,
        )
        with np.load(out / (tag + "-model.npz"), allow_pickle=False) as state:
            source = (x - state["input_mean"]) / state["input_scale"]
            actual = base.kernel_predict_input(
                source,
                state["training_design"],
                kernel,
                float(state["bandwidth_squared"]),
            )
            choose = np.flatnonzero(
                state["regularizations"] == state["selected_regularization"]
            )
            replay = (
                state["target_mean"]
                + base.multiply(actual, state["dual"][int(choose[0])])
                * state["target_scale"]
            ).reshape(-1, 17, 2)
            if task == "pose":
                replay = replay.clip(0, 1)
        difference = float(abs(replay - prediction).max())
        if difference > 1e-9:
            raise ValueError("Savedmodel actualinput prediction replay differs")
        parity[tag] = dict(
            actual_input_prediction_max_abs=difference,
            maximum_normal_equation_residual=residual_max,
            maximum_directsolve_relative_difference=direct_max,
        )
        row = dict(
            task=task,
            feature=feature,
            kernel=kernel,
            regularization=selected,
            validation_selection=best[0],
            source_validation_candidates=candidates,
            counts={
                k: int((split == k).sum()) for k in ("train", "validation", "test")
            },
            scores=scores,
            within_record_prediction_shuffle=control,
        )
        rows.append(row)
        print(
            json.dumps(
                dict(
                    task=task,
                    feature=feature,
                    kernel=kernel,
                    selection=best[0],
                    held=scores["test"],
                )
            ),
            flush=True,
        )
    return rows, parity


def positive(cache, clock, predecessor, out):
    from scripts import domain_same_condition_probe as base

    if out.exists():
        raise FileExistsError(out)
    audit_result = json.loads((clock / "results.json").read_text())
    if not audit_result["clock_gate"]["passed"]:
        raise ValueError("Registered nominalclock plausibility gate failed")
    audit_sources = audit_result["registration"]["source_sha256"]
    if (
        digest(ROOT / "ghostmesh/doppler_features.py")
        != audit_sources["ghostmesh/doppler_features.py"]
    ):
        raise ValueError("Audited quotient/FFT implementation changed")
    receipt = json.loads((cache / "receipt.json").read_text())
    validate_records(receipt["records"])
    if (
        digest(cache / "receipt.json")
        != audit_result["registration"]["cache_receipt_sha256"]
        or digest(cache / "rf.npy") != audit_result["registration"]["rf_array_sha256"]
    ):
        raise ValueError("AuditedRFcache changed")
    record = next(r for r in receipt["records"] if r["id"] == "E01/S01/A12")
    raw = np.load(cache / "rf.npy", mmap_mode="r", allow_pickle=False)[
        record["offset"] : record["offset"] + 297
    ].copy()
    expected = next(r for r in audit_result["records"] if r["record"] == record["id"])[
        "RFslice_sha256"
    ]
    if sha256(raw.tobytes()).hexdigest() != expected:
        raise ValueError("ActualA12RF changed")
    historical_manifest = json.loads(
        (predecessor / "artifact-manifest-final.json").read_text()
    )["sha256"]
    for filename in ("results.json", "registration.json", "verification.json"):
        if digest(predecessor / filename) != historical_manifest[filename]:
            raise ValueError("Frozen historical reference changed")
    if json.loads((predecessor / "registration.json").read_text())[
        "cache_receipt_sha256"
    ] != digest(cache / "receipt.json"):
        raise ValueError("Historical references use another cache")
    design, ends, splits, eligible, rejected = window_designs(raw)
    out.mkdir(parents=True)
    sources = snapshot(out)
    for extra in [
        "scripts/domain_same_condition_probe.py",
        "scripts/domain_structured_probe.py",
        "ghostmesh/rf_features.py",
        "ghostmesh/metrics.py",
    ]:
        sources[extra] = digest(ROOT / extra)
        destination = out / "executed-source" / extra
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / extra, destination)
    registration = dict(
        config=CONFIG,
        source_sha256=sources,
        clock_results_sha256=digest(clock / "results.json"),
        cache_receipt_sha256=digest(cache / "receipt.json"),
        RFslice_sha256=expected,
        pose_array_sha256=receipt["arrays"]["pose"],
        counts_before_pose_read={
            task: {
                name: int(
                    ((splits == name) & (eligible if task == "delta" else True)).sum()
                )
                for name in ("train", "validation", "test")
            }
            for task in ("pose", "delta")
        },
        rejected_RFwindows=rejected,
        GTlabels_opened_only_after_this_registration=True,
        predecessor_results_sha256=digest(predecessor / "results.json"),
    )
    save(out / "registration.json", registration)
    # FIRST pose-array access in this stage. All clock decisions, representation,
    # control variants, targets, grids and gates are already frozen on disk.
    if digest(cache / "pose.npy") != registration["pose_array_sha256"]:
        raise ValueError("Providedpose bytes changed")
    pose = np.load(cache / "pose.npy", mmap_mode="r", allow_pickle=False)[
        record["offset"] : record["offset"] + 297
    ].astype(np.float64)
    historical = json.loads((predecessor / "results.json").read_text())
    refs = [
        r
        for r in historical["results"]
        if r.get("protocol") == "held_time"
        and r.get("feature") in ("raw_db", "quotient")
        and r.get("kernel") == "rbf"
    ]
    rows = []
    parity = {}
    controls = {}
    for task in ("pose", "delta"):
        pick = np.ones(len(ends), bool) if task == "pose" else eligible
        current = ends[pick]
        split = splits[pick]
        body = pose[current]
        target = body if task == "pose" else pose[current] - pose[current - 8]
        np.savez_compressed(
            out / (task + "-indices.npz"),
            end=current,
            split=split,
            target=target,
            current_body=body,
        )
        names = np.repeat(record["id"], len(current))
        train, val, test = (split == k for k in ("train", "validation", "test"))
        mean = np.broadcast_to(target[train].mean(0), target.shape)
        controls[task] = {}
        for name, pred in [("training_mean", mean), ("zero", np.zeros_like(target))]:
            if task == "pose" and name == "zero":
                continue
            controls[task][name] = (
                base.score(pred[test], target[test], names[test])
                if task == "pose"
                else delta_score(pred[test], target[test], body[test])
            )
        features = (
            CONFIG["positive_control"]["variants"]
            if task == "pose"
            else [*CONFIG["positive_control"]["variants"], "raw_db", "quotient"]
        )
        for feature in features:
            x = design[feature][pick]
            np.save(
                out / (task + "-" + feature + "-features.npy"), x, allow_pickle=False
            )
            fitted, checks = fit_kernel(x, target, body, split, task, feature, out)
            rows.extend(fitted)
            parity.update(checks)
    # Independently reload actualRF and regenerate every retained input variant.
    actual_raw = np.load(cache / "rf.npy", mmap_mode="r", allow_pickle=False)[
        record["offset"] : record["offset"] + 297
    ].copy()
    fresh, actual_ends, actual_splits, actual_eligible, actual_rejected = (
        window_designs(actual_raw)
    )
    if (
        not np.array_equal(ends, actual_ends)
        or not np.array_equal(splits, actual_splits)
        or not np.array_equal(eligible, actual_eligible)
        or rejected != actual_rejected
    ):
        raise ValueError("RFquality/index replay differs")
    for task in ("pose", "delta"):
        pick = np.ones(len(ends), bool) if task == "pose" else eligible
        features = (
            CONFIG["positive_control"]["variants"]
            if task == "pose"
            else [*CONFIG["positive_control"]["variants"], "raw_db", "quotient"]
        )
        for feature in features:
            if not np.array_equal(
                np.load(
                    out / (task + "-" + feature + "-features.npy"), allow_pickle=False
                ),
                fresh[feature][pick],
            ):
                raise ValueError("ActualRFfeature readback differs")
    pose_row = min(
        [r for r in rows if r["task"] == "pose" and r["feature"] == "natural"],
        key=lambda r: r["validation_selection"],
    )
    delta_row = min(
        [r for r in rows if r["task"] == "delta" and r["feature"] == "natural"],
        key=lambda r: r["validation_selection"],
    )
    pr2 = pose_row["scores"]["test"]["motion"]["relative"]["within_record_r2"]
    ps = pose_row["within_record_prediction_shuffle"]["motion"]["relative"][
        "within_record_r2"
    ]
    ratio = (
        pose_row["scores"]["test"]["metrics"]["normalized_joint_error"]
        / controls["pose"]["training_mean"]["metrics"]["normalized_joint_error"]
    )
    dr2 = delta_row["scores"]["test"]["full_joint"]["raw_zero_reference_r2"]
    ds = delta_row["within_record_prediction_shuffle"]["full_joint"][
        "raw_zero_reference_r2"
    ]
    gates = dict(
        pose=dict(
            kernel_selected_validation=pose_row["kernel"],
            relative_pose_r2=pr2,
            aligned_minus_shuffled_r2=pr2 - ps,
            NJE_fraction_of_static=ratio,
            passed=bool(pr2 >= 0.5 and pr2 - ps >= 0.3 and ratio <= 0.75),
        ),
        motion=dict(
            kernel_selected_validation=delta_row["kernel"],
            full_joint_raw_delta_r2=dr2,
            aligned_minus_shuffled_r2=dr2 - ds
            if dr2 is not None and ds is not None
            else None,
            passed=bool(
                dr2 is not None and ds is not None and dr2 >= 0.1 and dr2 - ds >= 0.2
            ),
        ),
    )
    for name, expected in sources.items():
        if digest(ROOT / name) != expected:
            raise ValueError("Sourcechanged")
    if (
        sha256(actual_raw.tobytes()).hexdigest() != registration["RFslice_sha256"]
        or digest(cache / "pose.npy") != registration["pose_array_sha256"]
    ):
        raise ValueError("ActualRF/labels changed")
    save(
        out / "results.json",
        dict(
            registration=registration,
            results=rows,
            controls=controls,
            historical_pose_references=refs,
            historical_scope="Original104poseheldwindows; notmatchedifnewRFmissingmaskdropswindows. Delta controls freshlyrecomputed onidenticalnewpairedrows.",
            continuation_gates=gates,
        ),
    )
    save(
        out / "verification.json",
        dict(
            models_verified=len(parity),
            normal_equations_verified=len(parity) * 4,
            all_actual_RFfeatures_exact=True,
            savedmodel_actualinput_checks=parity,
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
    print(json.dumps(gates), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--cache", type=Path, default=ROOT / "work/domain-source30-rf-v1")
    a.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("positive")
    b.add_argument("--cache", type=Path, default=ROOT / "work/domain-source30-rf-v1")
    b.add_argument("--clock", type=Path, required=True)
    b.add_argument(
        "--predecessor", type=Path, default=ROOT / "work/domain-same-condition-v1"
    )
    b.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "audit":
        audit(args.cache, args.out)
    else:
        positive(args.cache, args.clock, args.predecessor, args.out)
