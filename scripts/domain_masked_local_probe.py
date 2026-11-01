"""Prespecified dense personal-calibration diagnostic on E01/S04/A12 only.

S04 was excluded from all encoder optimizer updates. This experiment fits new
local heads and never loads global pose heads or pose RFF maps. It tests one
person/room/action, not room transfer. No held score occurs in the fit phase.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.linalg import eigh
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.masked_csi import MaskedCSI, normalize_frames  # noqa:E402
from ghostmesh.rf_features import quotient_features  # noqa:E402
from scripts.domain_background_probe import digest, multiply  # noqa:E402
from scripts.domain_covariance_full_cache import verify_hashes  # noqa:E402
from scripts.domain_masked_fullpose_probe import (  # noqa:E402
    finite,
    metric,
    real_numeric,
    save_array,
    save_json,
    save_npz,
)

ENCODERS = (
    "random_frozen",
    "masked_only",
    "masked_temporal",
    "masked_shuffled",
    "supervised",
)
ARMS = ENCODERS + ("quotient", "raw_log")
# Production dependencies are a declared schema, never inferred from whatever
# potentially incomplete producer manifest happens to be supplied.
REQUIRED_TRAINING_SOURCES = frozenset(
    {
        "scripts/domain_masked_pretrain_probe.py",
        "ghostmesh/masked_csi.py",
        "scripts/domain_covariance_cached_probe.py",
        "scripts/domain_covariance_source_probe.py",
        "scripts/domain_covariance_full_cache.py",
        "scripts/domain_reference_covariance_probe.py",
        "scripts/domain_background_probe.py",
        "ghostmesh/reference_covariance.py",
        "ghostmesh/covariance_dynamics.py",
        "ghostmesh/covariance_accelerated.py",
        "ghostmesh/causal_channel.py",
        "ghostmesh/metrics.py",
    }
)
BLOCKS = {"train": (0, 118), "validation": (126, 178), "query": (186, 297)}
ALPHAS = (0.00001, 0.001, 0.01, 0.1, 10.0)
CONFIG = dict(
    record="E01/S04/A12",
    person="S04",
    context=8,
    blocks=BLOCKS,
    alphas=ALPHAS,
    kernels=("linear", "rbf"),
    normalization="Per-column LOCAL TRAIN mean/SD only; SD floor.05 in each feature's declared units. No global pose maps or fitted heads",
    raw_controls="Full54720D inter-RX log-ratio/circular-phase histories; full27360D raw log-magnitude histories/20 with same-frame per-receiver finite-median fill only",
    quality="RF-only common mask BEFORE loading local labels: each history <=5percent invalid inter-RX pairs; each constituent frame has finite nonconstant observed magnitude and at least one observed sample per receiver",
    encoder="Five fixed-final4000-step encoder backbones, CPU eval/inference only; frame embeddings64D then causal8frame concat512. Global pose_head and decoder never called",
    equation="Exact dual(K+alpha*I)A=Y-Y_LOCAL_TRAIN_mean; linearK=XXtrain.T/D or exp(-d2/(2*median_ALL_LOCAL_TRAIN_pair_d2)); no D-by-D fit",
    selection="LOCAL earlier validation NJE independently per arm/hip-relative or full-image target; tie linear before RBF, then stronger alpha. All choices sealed before query labels are scored",
    controls="Local TRAIN mean and linear endpoint-time trend; fixed seed90210 within-query prediction shuffle. Strong raw controls always reported",
    gate="Observability only per hip-relative arm: record-centered bodyR2>.5, aligned-minus-shuffled bodyR2>.3, NJE<.75TRAINmean. Never room/person/action generalization",
    burden="111 dense endpoint labels before RF-quality exclusions,118 raw frames span11.8nominalseconds; same person, room and action. Query104windows before exclusions",
    exposure="S04 has historical development exposure and source-VAL labels selected global RFF heads AFTER final encoders. Those heads/maps are not used here; no untouched-source-person claim",
    inference="RBF returns LOCAL TRAIN target mean far from references; no fitted joint confidence or camera input during inference",
)


def supports():
    ends, raw, split = [], [], []
    for name, (start, stop) in BLOCKS.items():
        current = np.arange(start + 7, stop, dtype=np.int64)
        history = current[:, None] - np.arange(7, -1, -1)[None]
        if history.min() < start or history.max() >= stop:
            raise ValueError("Causal support escaped its purged block")
        ends.extend(current)
        raw.extend(history)
        split.extend([name] * len(current))
    return np.asarray(ends), np.asarray(raw), np.asarray(split)


def common_quality(raw, history):
    x = real_numeric(raw)
    if (
        x.shape != (297, 6, 114, 10)
        or history.ndim != 2
        or history.shape[1] != 8
        or history.dtype.kind not in "iu"
        or history.min() < 0
        or history.max() >= 297
        or np.any(np.diff(history, axis=1) != 1)
    ):
        raise ValueError("Expected one297frame typed record and8frame supports")
    observed = np.isfinite(x[:, :3])
    count = observed.sum(axis=(1, 2, 3))
    clean = np.where(observed, x[:, :3], 0).astype(np.float64)
    mean = clean.sum(axis=(1, 2, 3)) / np.maximum(count, 1)
    centered = np.where(observed, clean - mean[:, None, None, None], 0)
    variance = np.square(centered).sum(axis=(1, 2, 3)) / np.maximum(count, 1)
    frame_ok = (count > 0) & (variance >= 1e-12) & observed.any(axis=(2, 3)).all(1)
    complete = np.isfinite(x[:, :3]) & np.isfinite(x[:, 3:])
    pair_ok = complete[:, 1:] & complete[:, :1]
    fractions = pair_ok[history].mean(axis=(1, 2, 3, 4))
    keep = frame_ok[history].all(1) & ((1.0 - fractions) <= 0.05)
    return keep, dict(
        nominal_rows=len(history),
        retained_rows=int(keep.sum()),
        rejected_rows=int((~keep).sum()),
        minimum_pair_coverage=float(fractions.min()),
        rejected_history_rows=np.flatnonzero(~keep).tolist(),
    )


def frozen_encoder_role(source, expected):
    """Verify optimizer exclusion before any local feature/label work."""
    if digest(source / "source-ready.json") != expected:
        raise ValueError("Source-ready identity differs")
    ready = json.loads((source / "source-ready.json").read_text())
    required = {
        "registration.json",
        "roles.npz",
        "targets-and-metadata.npz",
        "schedule.npy",
    }
    required.update(
        arm + suffix
        for arm in ENCODERS
        for suffix in ("-final.pt", "-complete.json", "-training.json")
    )
    if (
        ready["kind"] != "masked-csi-source-only-ready"
        or ready["held_metrics_computed"] is not False
        or not required <= set(ready["artifacts_sha256"])
    ):
        raise ValueError("Fixed final encoder/source-role artifacts are incomplete")
    # Hashing pose artifacts is provenance checking, not loading their values.
    verify_hashes(source, ready["artifacts_sha256"])
    registration = json.loads((source / "registration.json").read_text())
    manifest = registration.get("source_sha256")
    if not isinstance(manifest, dict) or not REQUIRED_TRAINING_SOURCES <= set(manifest):
        raise ValueError("Encoder producer manifest omits required production sources")
    if any(
        ready["artifacts_sha256"].get("executed-source/" + name) != expected_hash
        for name, expected_hash in manifest.items()
    ):
        raise ValueError(
            "Encoder producer snapshots are absent from the sealed artifacts"
        )
    verify_hashes(ROOT, manifest)
    verify_hashes(source / "executed-source", manifest)
    if (
        registration["config"]["steps"] != 4000
        or registration["config"]["held_room"] != "E02"
    ):
        raise ValueError("Expected prospectively fixed4000-step E02 encoders")
    with np.load(source / "targets-and-metadata.npz", allow_pickle=False) as z:
        people = z["person"]
    with np.load(source / "roles.npz", allow_pickle=False) as z:
        train, validation = z["train"], z["validation"]
    schedule = np.load(source / "schedule.npy", allow_pickle=False)
    if schedule.shape != (4000, 16) or registration["config"].get("batch") != 16:
        raise ValueError("Expected the declared4000x16 encoder source schedule")
    check_encoder_exclusion(people, train, validation, schedule)
    for arm in ENCODERS:
        training = json.loads((source / (arm + "-training.json")).read_text())
        if (
            training["steps"] != (0 if arm == "random_frozen" else 4000)
            or training["selection"]
            != "Fixed finalstep, no pose-based checkpointselection"
        ):
            raise ValueError("Encoder checkpoint was not the declared fixed final step")
    return registration


def check_encoder_exclusion(people, train, validation, schedule):
    if (
        train.dtype != bool
        or validation.dtype != bool
        or train.shape != people.shape
        or validation.shape != people.shape
        or np.any(train & validation)
    ):
        raise ValueError("Malformed encoder source roles")
    selected = people == CONFIG["person"]
    if not selected.any() or train[selected].any() or not validation[selected].all():
        raise ValueError("S04 must be exclusively encoder source validation")
    if (
        schedule.dtype.kind not in "iu"
        or schedule.ndim != 2
        or schedule.size == 0
        or schedule.min() < 0
        or schedule.max() >= len(people)
        or not train[schedule].all()
        or np.any(people[schedule] == CONFIG["person"])
    ):
        raise ValueError("Encoder optimizer schedule contains prohibited person/role")


@torch.inference_mode()
def encoder_features(model, raw, history):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    used = np.unique(history)
    x = normalize_frames(real_numeric(raw[used, :3]))
    # Framewise encoding cannot inspect future frames. Unique frames are encoded
    # once; chronological8frame vectors are assembled from their exact supports.
    embedding = np.zeros((len(raw), 64), dtype=np.float32)
    for start in range(0, len(used), 16):
        values = torch.from_numpy(x[start : start + 16])
        tokens = model.encode_frames(values)
        embedding[used[start : start + len(values)]] = (
            model.embedding(tokens.mean(1)).cpu().numpy()
        )
    return finite(embedding[history].reshape(len(history), 512))


def load_backbone(path):
    """Load only encoder parameters, discarding all training/pose task heads."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or not all(isinstance(name, str) for name in saved):
        raise ValueError("Expected a named frozen encoder tensor state")
    excluded = ("pose_head.", "decoder.", "projector.")
    backbone = {
        name: value for name, value in saved.items() if not name.startswith(excluded)
    }
    model = MaskedCSI().to(device="cpu", dtype=torch.float32)
    for head in ("pose_head", "decoder", "projector"):
        delattr(model, head)
    expected = model.state_dict()
    if set(backbone) != set(expected):
        raise ValueError("Frozen backbone tensor names differ from the model schema")
    for name, tensor in backbone.items():
        template = expected[name]
        if (
            not torch.is_tensor(tensor)
            or tensor.device.type != "cpu"
            or tensor.is_complex()
            or tensor.dtype != template.dtype
            or tensor.shape != template.shape
            or tensor.layout != template.layout
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError("Invalid real finite backbone tensor: " + name)
    model.load_state_dict(backbone, strict=True)
    return model


def raw_features(raw, history):
    raw = real_numeric(raw)
    magnitude = raw[:, :3]
    filled = np.empty_like(magnitude, dtype=np.float64)
    for frame in range(len(raw)):
        for receiver in range(3):
            values = magnitude[frame, receiver]
            observed = np.isfinite(values)
            if not observed.any():
                # Invalid frames never enter retained histories. No fabricated
                # data may enter a used row, even if other record frames fail.
                filled[frame, receiver] = 0
            else:
                filled[frame, receiver] = (
                    np.where(observed, values, np.median(values[observed])) / 20
                )
    quotient = np.empty((len(history), 8 * 6 * 114 * 10))
    for row, support in enumerate(history):
        selected = raw[support]
        quotient[row] = quotient_features(
            selected[:, :3], selected[:, 3:], units="log_magnitude_db"
        ).reshape(-1)
    return dict(
        raw_log=finite(filled[history].reshape(len(history), -1)),
        quotient=finite(quotient),
    )


def kernel_inputs(feature, train):
    x = finite(feature)
    mean, scale = x[train].mean(0), np.maximum(x[train].std(0), 0.05)
    x = (x - mean) / scale
    source = x[train]
    cross = multiply(x, source.T)
    distance = np.maximum(
        np.square(x).sum(1)[:, None] + np.square(source).sum(1)[None] - 2 * cross, 0
    )
    width = float(np.median(distance[train][np.triu_indices(train.sum(), 1)]))
    kernels = {"linear": cross / x.shape[1]}
    if np.isfinite(width) and width > 1e-12:
        kernels["rbf"] = np.exp(-distance / (2 * width))
    return kernels, dict(
        input_mean=mean,
        input_scale=scale,
        training_design=source,
        bandwidth_squared=np.asarray(width),
    )


def choose_head(feature, target, train, validation):
    if (
        train.dtype != bool
        or validation.dtype != bool
        or train.shape != validation.shape
        or train.sum() < 2
        or not validation.any()
        or np.any(train & validation)
    ):
        raise ValueError("Disjoint nonempty local TRAIN/VAL roles required")
    target = real_numeric(target)
    if target.shape != (len(train), 17, 2):
        raise ValueError("Expected local17joint targets")
    finite(target[train | validation])
    kernels, normalization = kernel_inputs(feature, train)
    mean = target[train].reshape(train.sum(), 34).mean(0)
    y = target[train].reshape(train.sum(), 34) - mean
    grid, candidates = [], []
    for name, kernel in kernels.items():
        eigen, vectors = eigh(kernel[train], check_finite=True)
        if eigen.min() < -1e-7:
            raise ValueError("Local kernel is not positive semidefinite")
        right = multiply(vectors.T, y)
        for alpha in ALPHAS:
            dual = multiply(vectors, right / (eigen[:, None] + alpha))
            residual = multiply(kernel[train] + alpha * np.eye(train.sum()), dual) - y
            relative = float(np.linalg.norm(residual) / max(np.linalg.norm(y), 1e-12))
            if not np.isfinite(relative) or relative > 1e-6:
                raise ValueError("Local dual equation failed")
            prediction = (multiply(kernel, dual) + mean).reshape(-1, 17, 2)
            validation_nje = metric(
                prediction[validation],
                target[validation],
                np.repeat("local-validation", validation.sum()),
            )["normalized_joint_error"]
            grid.append(
                dict(
                    kernel=name,
                    alpha=alpha,
                    validation_nje=validation_nje,
                    normal_equation_relative_residual=relative,
                )
            )
            candidates.append((name, alpha, dual, prediction))
    selected = min(
        range(len(grid)),
        key=lambda i: (
            grid[i]["validation_nje"],
            grid[i]["kernel"] != "linear",
            -grid[i]["alpha"],
        ),
    )
    name, alpha, dual, prediction = candidates[selected]
    state = dict(
        **normalization,
        target_mean=mean,
        dual=dual,
        alpha=np.asarray(alpha),
        kernel=np.asarray(name),
    )
    return (
        state,
        dict(choice=grid[selected], grid=grid, rbf_available="rbf" in kernels),
        prediction,
    )


def fit_arrays(features, full_target, ends, split):
    """Real local-fit boundary: query targets never enter fit/selection."""
    full_target = real_numeric(full_target)
    train, validation, query = (
        split == name for name in ("train", "validation", "query")
    )
    if (
        train.sum() < 2
        or not validation.any()
        or query.sum() < 2
        or not np.all(train | validation | query)
    ):
        raise ValueError("Every local split must retain RF-quality rows")
    if (
        ends.dtype.kind not in "iu"
        or ends.shape != split.shape
        or full_target.shape != (len(split), 17, 2)
    ):
        raise ValueError("Local endpoint/target shapes differ")
    source_rows = train | validation
    body = np.full_like(full_target, np.nan)
    body[source_rows] = full_target[source_rows] - full_target[source_rows][
        :, [11, 12]
    ].mean(1, keepdims=True)
    states, choices, predictions = {}, {}, {}
    for space, target in (("hip_relative", body), ("full_image", full_target)):
        for arm, feature in features.items():
            tag = space + "/" + arm
            states[tag], choices[tag], predictions[tag] = choose_head(
                feature, target, train, validation
            )
        mean = target[train].mean(0)
        predictions[space + "/source_mean"] = np.broadcast_to(mean, target.shape).copy()
        time = ends.astype(float)
        tm = time[train].mean()
        slope = (
            (time[train] - tm)[:, None]
            * (target[train].reshape(train.sum(), 34) - mean.reshape(34))
        ).sum(0) / np.square(time[train] - tm).sum()
        predictions[space + "/source_time_trend"] = (
            mean.reshape(34) + (time - tm)[:, None] * slope
        ).reshape(-1, 17, 2)
    return states, choices, predictions


def fit(source, raw, out, expected):
    if out.exists():
        raise FileExistsError(out)
    registration = frozen_encoder_role(source, expected)
    receipt = json.loads((raw / "receipt.json").read_text())
    if (
        registration["raw_receipt_sha256"] != digest(raw / "receipt.json")
        or receipt["frames"] != 240570
    ):
        raise ValueError("Expected the original source30 raw cache")
    verify_hashes(
        raw, {name + ".npy": value for name, value in receipt["arrays"].items()}
    )
    row = next(r for r in receipt["records"] if r["id"] == CONFIG["record"])
    if row["frames"] != 297:
        raise ValueError("Prespecified local recording length differs")
    offset = row["offset"]
    selected = slice(offset, offset + 297)
    raw_rf = np.array(
        np.load(raw / "rf.npy", mmap_mode="r", allow_pickle=False)[selected]
    )
    ends, history, split = supports()
    keep, quality = common_quality(raw_rf, history)
    ends, history, split = ends[keep], history[keep], split[keep]
    if any(not np.any(split == name) for name in BLOCKS):
        raise ValueError("RF quality removed an entire local role")
    # The common retained supports are final BEFORE any local annotation load.
    features = raw_features(raw_rf, history)
    torch.set_num_threads(1)
    encoder_bytes = {}
    for arm in ENCODERS:
        model = load_backbone(source / (arm + "-final.pt"))
        before = {name: value.clone() for name, value in model.state_dict().items()}
        features[arm] = encoder_features(model, raw_rf, history)
        if any(
            not torch.equal(before[name], value)
            for name, value in model.state_dict().items()
        ):
            raise ValueError("Frozen encoder changed during local inference")
        encoder_bytes[arm] = digest(source / (arm + "-final.pt"))
    source_role = split != "query"
    pose = np.load(raw / "pose.npy", mmap_mode="r", allow_pickle=False)
    valid = np.load(raw / "valid.npy", mmap_mode="r", allow_pickle=False)
    source_indices = offset + ends[source_role]
    if not valid[source_indices].all():
        raise ValueError("Retained local target has invalid annotation joints")
    target = np.full((len(ends), 17, 2), np.nan)
    target[source_role] = finite(pose[source_indices], (source_role.sum(), 17, 2))
    states, choices, predictions = fit_arrays(features, target, ends, split)
    out.mkdir(parents=True)
    sources = [
        Path(__file__),
        ROOT / "tests/test_masked_local_probe.py",
        ROOT / "ghostmesh/masked_csi.py",
        ROOT / "ghostmesh/rf_features.py",
        ROOT / "scripts/domain_masked_fullpose_probe.py",
        ROOT / "scripts/domain_background_probe.py",
        ROOT / "scripts/domain_covariance_full_cache.py",
        ROOT / "ghostmesh/metrics.py",
    ]
    reg = dict(
        config=CONFIG,
        source_ready_sha256=expected,
        raw_receipt_sha256=digest(raw / "receipt.json"),
        encoder_sha256=encoder_bytes,
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
        common_quality=quality,
        counts={name: int(np.sum(split == name)) for name in BLOCKS},
        raw_record_sha256=sha256(raw_rf.tobytes()).hexdigest(),
        raw_record_offset=int(offset),
        query_labels_loaded=False,
        query_metrics_computed=False,
    )
    save_json(out / "registration.json", reg)
    for path in sources:
        destination = out / "executed-source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    save_npz(
        out / "local-data.npz",
        ends=ends,
        history=history,
        split=split,
        source_target=target,
    )
    save_json(out / "frozen-local-selections.json", choices)
    for arm, feature in features.items():
        save_array(out / (arm + "-features.npy"), feature)
    for name, state in states.items():
        save_npz(out / (name.replace("/", "-") + "-state.npz"), **state)
    for name, prediction in predictions.items():
        save_array(out / (name.replace("/", "-") + "-prediction.npy"), prediction)
    save_json(
        out / "local-source-ready.json",
        dict(
            kind="masked-csi-S04-local-ready",
            query_metrics_computed=False,
            artifacts_sha256={
                str(p.relative_to(out)): digest(p)
                for p in sorted(out.rglob("*"))
                if p.is_file()
            },
        ),
    )
    return out


def score_frozen(out, source, raw, expected):
    if (out / "results.json").exists() or (out / "query-permutation.npy").exists():
        raise FileExistsError("Local query scoring already started")
    if digest(out / "local-source-ready.json") != expected:
        raise ValueError("Local ready identity differs")
    ready = json.loads((out / "local-source-ready.json").read_text())
    required = {"registration.json", "local-data.npz", "frozen-local-selections.json"}
    required.update(
        space + "-" + arm + suffix
        for space in ("hip_relative", "full_image")
        for arm in ARMS
        for suffix in ("-state.npz", "-prediction.npy")
    )
    required.update(
        space + "-" + arm + "-prediction.npy"
        for space in ("hip_relative", "full_image")
        for arm in ("source_mean", "source_time_trend")
    )
    if (
        ready["kind"] != "masked-csi-S04-local-ready"
        or ready["query_metrics_computed"] is not False
        or not required <= set(ready["artifacts_sha256"])
    ):
        raise ValueError("Local source choices are not completely sealed")
    verify_hashes(out, ready["artifacts_sha256"])
    registration = json.loads((out / "registration.json").read_text())
    verify_hashes(ROOT, registration["source_sha256"])
    frozen_encoder_role(source, registration["source_ready_sha256"])
    if digest(raw / "receipt.json") != registration["raw_receipt_sha256"]:
        raise ValueError("Local scoring raw receipt differs")
    with np.load(out / "local-data.npz", allow_pickle=False) as data:
        query = data["split"] == "query"
        query_ends = data["ends"][query]
        if np.isfinite(data["source_target"][query]).any():
            raise ValueError("Local query labels were present before choice sealing")
    raw_receipt = json.loads((raw / "receipt.json").read_text())
    verify_hashes(
        raw, {name + ".npy": raw_receipt["arrays"][name] for name in ("pose", "valid")}
    )
    if registration["query_labels_loaded"] is not False:
        raise ValueError("Expected source-only local calibration receipt")
    query_indices = registration["raw_record_offset"] + query_ends
    pose = np.load(raw / "pose.npy", mmap_mode="r", allow_pickle=False)
    valid = np.load(raw / "valid.npy", mmap_mode="r", allow_pickle=False)
    if not valid[query_indices].all():
        raise ValueError("Local query teacher contains invalid joints")
    full_target = finite(pose[query_indices], (query.sum(), 17, 2))
    bodies = full_target - full_target[:, [11, 12]].mean(1, keepdims=True)
    permutation = np.random.default_rng(90210).permutation(query.sum())
    save_array(out / "query-permutation.npy", permutation)
    results, gate = {}, {}
    records = np.repeat(CONFIG["record"], query.sum())
    for space, target in (("hip_relative", bodies), ("full_image", full_target)):
        for arm in (*ARMS, "source_mean", "source_time_trend"):
            tag = space + "/" + arm
            prediction = finite(
                np.load(
                    out / (tag.replace("/", "-") + "-prediction.npy"),
                    allow_pickle=False,
                )
            )[query]
            results[tag] = dict(
                aligned=metric(prediction, target, records),
                shuffled=metric(prediction[permutation], target, records),
            )
    static = results["hip_relative/source_mean"]["aligned"]["normalized_joint_error"]
    for arm in ARMS:
        aligned, shuffled = (
            results["hip_relative/" + arm]["aligned"],
            results["hip_relative/" + arm]["shuffled"],
        )
        r2, sr2 = (
            aligned["record_centered_body_r2"],
            shuffled["record_centered_body_r2"],
        )
        gate[arm] = dict(
            positive_observability=r2 is not None and r2 > 0.5,
            shuffle_gap=r2 is not None and sr2 is not None and r2 - sr2 > 0.3,
            error_vs_static=aligned["normalized_joint_error"] < 0.75 * static,
        )
    save_json(
        out / "results.json",
        dict(
            registration=registration,
            methods=results,
            local_observability_gate=gate,
            passes={arm: all(checks.values()) for arm, checks in gate.items()},
            scope="Dense personalcalibration only; no room/person/action transfer or original final-test success claim",
        ),
    )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-ready-sha256")
    parser.add_argument("--score-frozen-sha256")
    args = parser.parse_args()
    if args.score_frozen_sha256:
        score_frozen(args.out, args.source, args.raw, args.score_frozen_sha256)
    elif args.source_ready_sha256:
        fit(args.source, args.raw, args.out, args.source_ready_sha256)
    else:
        parser.error(
            "Explicit source-ready fit or local-source-ready scoring SHA required"
        )


if __name__ == "__main__":
    main()
