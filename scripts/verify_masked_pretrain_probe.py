"""Independent source-only audit of a frozen masked-CSI experiment.

This program never trains an encoder, invokes the fitter, or scores held poses.
Direct linear solves qualify already saved ridge coefficients. Encoder replay
uses only the hash-verified executed model source, on CPU, for sixteen histories.
Use --preoptimization-only while training is in progress; the default requires
the complete source-ready seal before reading final learned artifacts.
"""

from __future__ import annotations

import argparse
import ast
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import solve
from scipy.linalg.blas import dgemm
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.research_numeric_readback import numeric_difference  # noqa: E402

ARMS = (
    "random_frozen",
    "masked_only",
    "masked_temporal",
    "masked_shuffled",
    "supervised",
)
ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0)
VALIDATION_PEOPLE = {"S04", "S09", "S14", "S19", "S24", "S29"}
ENCODER_ATOL = 2e-5
ENCODER_RTOL = 2e-5
MAP_ATOL = 2e-8
MAP_RTOL = 2e-8
RIDGE_ATOL = 2e-8
RIDGE_RTOL = 2e-8
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
REQUIRED_CACHE_SOURCES = frozenset(
    {
        "scripts/domain_covariance_full_cache.py",
        "scripts/domain_covariance_source_probe.py",
        "scripts/domain_background_probe.py",
        "ghostmesh/covariance_dynamics.py",
        "ghostmesh/covariance_accelerated.py",
        "ghostmesh/metrics.py",
    }
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def verify_hashes(directory, manifest):
    require(isinstance(manifest, dict) and bool(manifest), "Empty hash manifest")
    for name, expected in manifest.items():
        path = Path(name)
        require(
            not path.is_absolute() and ".." not in path.parts, "Unsafe artifact path"
        )
        require(
            digest(directory / path) == expected,
            "Hash mismatch: " + str(directory / path),
        )


def finite(value, name, shape=None):
    a = np.asarray(value)
    require(not np.iscomplexobj(a), "Complex artifact: " + name)
    require(
        a.dtype.kind in "biuf" and a.size > 0 and np.isfinite(a).all(),
        "Nonfinite/non-numeric artifact: " + name,
    )
    require(shape is None or a.shape == shape, "Unexpected shape: " + name)
    return a


def npz(path):
    with np.load(path, allow_pickle=False) as saved:
        result = {key: saved[key] for key in saved.files}
    for name, value in result.items():
        if value.dtype.kind in "biufc":
            finite(value, str(path) + ":" + name)
    return result


def close(a, b, name, atol=MAP_ATOL, rtol=MAP_RTOL):
    finite(a, name + "/observed")
    finite(b, name + "/expected")
    if np.asarray(a).dtype == bool and np.asarray(b).dtype == bool:
        a, b = np.asarray(a, dtype=np.int8), np.asarray(b, dtype=np.int8)
    return numeric_difference(a, b, name=name, atol=atol, rtol=rtol)


def mm(a, b):
    finite(a, "matrix product left")
    finite(b, "matrix product right")
    result = dgemm(
        1.0, np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    )
    return finite(result, "independent matrix product")


def tensor_tree(value, name):
    if torch.is_tensor(value):
        require(not value.is_complex(), "Complex tensor: " + name)
        require(
            value.device.type == "cpu" and bool(torch.isfinite(value).all()),
            "Nonfinite/device tensor: " + name,
        )
    elif isinstance(value, dict):
        for key, item in value.items():
            tensor_tree(item, name + "/" + str(key))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            tensor_tree(item, name + "/" + str(i))
    elif isinstance(value, float):
        require(np.isfinite(value), "Nonfinite scalar: " + name)
    elif isinstance(value, (np.ndarray, np.generic, complex)):
        finite(value, name)


def load_torch(path):
    value = torch.load(path, map_location="cpu", weights_only=True)
    tensor_tree(value, str(path))
    return value


def equal_state(observed, expected, name):
    require(set(observed) == set(expected), "State keys differ: " + name)
    for key in observed:
        close(observed[key].numpy(), expected[key].numpy(), name + "/" + key, 0, 0)


def equal_tree(observed, expected, name):
    if torch.is_tensor(expected):
        require(torch.is_tensor(observed), "Tensor type differs: " + name)
        close(observed.numpy(), expected.numpy(), name, 0, 0)
    elif isinstance(expected, dict):
        require(
            isinstance(observed, dict) and set(observed) == set(expected),
            "Tree keys differ: " + name,
        )
        for key in expected:
            equal_tree(observed[key], expected[key], name + "/" + str(key))
    elif isinstance(expected, (list, tuple)):
        require(
            type(observed) is type(expected) and len(observed) == len(expected),
            "Tree sequence differs: " + name,
        )
        for index, value in enumerate(expected):
            equal_tree(observed[index], value, name + "/" + str(index))
    else:
        require(observed == expected, "Tree value differs: " + name)


def load_model_source(experiment):
    # The fitter and its statistical helpers are deliberately never imported.
    path = experiment / "executed-source/ghostmesh/masked_csi.py"
    spec = importlib.util.spec_from_file_location("masked_csi_frozen_audit", path)
    require(spec is not None and spec.loader is not None, "Cannot load frozen encoder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_source_stage_contract(path):
    tree = ast.parse(path.read_text())
    run = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    calls = {
        node.func.id
        for node in ast.walk(run)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    require(
        not ({"scores", "score_frozen"} & calls), "Training stage calls held scoring"
    )
    require("seal_source_ready" in calls, "Training stage lacks source-ready seal")


def roles_and_schedule(data, records, registration, roles, schedule, seal):
    n = len(data["end"])
    require(
        set(data) == {"pose", "delta", "room", "person", "record", "action", "end"},
        "Metadata fields differ",
    )
    require(all(len(a) == n for a in data.values()), "Metadata lengths differ")
    require(set(data["room"]) == {"E01", "E02", "E03"}, "Only E01-E03 allowed")
    expected_records = [
        f"E{room:02}/S{person:02}/A{action:02}"
        for room in range(1, 4)
        for person in range((room - 1) * 10 + 1, room * 10 + 1)
        for action in range(1, 28)
    ]
    require(
        [r["id"] for r in records] == expected_records, "Raw record identities differ"
    )
    require(
        all(
            r["frames"] == 297 and r["offset"] == i * 297 for i, r in enumerate(records)
        ),
        "Raw record offsets differ",
    )
    expected = dict(held=data["room"] == "E02")
    expected["validation"] = ~expected["held"] & np.isin(
        data["person"], sorted(VALIDATION_PEOPLE)
    )
    expected["train"] = ~(expected["held"] | expected["validation"])
    require(set(roles) == set(expected), "Role fields differ")
    for name, mask in expected.items():
        require(roles[name].dtype == bool, "Role is not Boolean")
        close(roles[name], mask, "role/" + name, 0, 0)
        receipt = dict(
            people=sorted(set(data["person"][mask])),
            records=sorted(set(data["record"][mask])),
            rows=int(mask.sum()),
        )
        require(
            seal["roles"][name] == receipt
            and registration["counts"][name] == int(mask.sum()),
            "Role receipt differs: " + name,
        )
    for a, b in (("train", "validation"), ("train", "held"), ("validation", "held")):
        require(
            not (set(data["person"][expected[a]]) & set(data["person"][expected[b]])),
            "Person roles overlap",
        )
    require(
        tuple(
            len(set(data["person"][expected[k]]))
            for k in ("train", "validation", "held")
        )
        == (16, 4, 10),
        "Person role coverage differs",
    )
    config = registration["config"]
    require(
        config["held_room"] == "E02"
        and config["steps"] == 4000
        and config["batch"] == 16,
        "Pilot budget/fold differs",
    )
    require(
        schedule.dtype.kind in "iu"
        and schedule.shape == (config["steps"], config["batch"]),
        "Schedule shape/type differs",
    )
    require(
        (schedule >= 0).all()
        and (schedule < n).all()
        and expected["train"][schedule].all(),
        "Optimizer schedule accesses non-TRAIN rows",
    )
    rng = np.random.default_rng(config["seed"])
    groups = [
        np.flatnonzero(expected["train"] & (data["room"] == room))
        for room in ("E01", "E03")
    ]
    regenerated = np.stack(
        [
            rng.choice(groups[int(rng.integers(2))], config["batch"], replace=False)
            for _ in range(config["steps"])
        ]
    )
    close(schedule, regenerated, "schedule", 0, 0)
    offsets_by_record = {r["id"]: r["offset"] for r in records}
    offsets = np.array([offsets_by_record[str(record)] for record in data["record"]])
    require(np.isin(data["end"], np.arange(39, 297, 8)).all(), "Noncanonical endpoint")
    for name, room, person, action in zip(
        data["record"], data["room"], data["person"], data["action"]
    ):
        require(str(name) == f"{room}/{person}/{action}", "Record field join differs")
    unique = np.unique(schedule)
    frames = offsets[unique, None] + data["end"][unique, None] - np.arange(7, -1, -1)
    require(
        ((frames - offsets[unique, None]) >= 0).all()
        and ((frames - offsets[unique, None]) < 297).all(),
        "History crosses record boundary",
    )
    exposure = registration["exposure"]
    for name, value in dict(
        optimizer_histories_per_trained_arm=schedule.size,
        optimizer_frame_occurrences_per_trained_arm=schedule.size * 8,
        optimizer_unique_endpoint_rows=len(unique),
        supervised_optimizer_endpoint_label_occurrences=schedule.size,
        ssl_optimizer_pose_labels=0,
        frozen_probe_training_endpoint_labels=int(expected["train"].sum()),
    ).items():
        require(exposure[name] == value, "Exposure receipt differs: " + name)
    require(
        seal["optimizer_unique_raw_frames"] == len(np.unique(frames)),
        "Raw exposure count differs",
    )
    return expected, offsets


def check_targets(raw, data, offsets):
    pose = np.load(raw / "pose.npy", mmap_mode="r", allow_pickle=False)
    finite(pose, "raw pose", (240570, 17, 2))
    maximum = 0.0
    for start in range(0, len(offsets), 256):
        ids = np.arange(start, min(len(offsets), start + 256))
        current = np.asarray(pose[offsets[ids] + data["end"][ids]], dtype=np.float64)
        past = np.asarray(pose[offsets[ids] + data["end"][ids] - 8], dtype=np.float64)
        current -= current[:, [11, 12]].mean(1, keepdims=True)
        past -= past[:, [11, 12]].mean(1, keepdims=True)
        maximum = max(
            maximum,
            close(data["pose"][ids], current, "target endpoint", 0, 0),
            close(data["delta"][ids], current - past, "target displacement", 0, 0),
        )
    return maximum


def independent_rff(feature, train, saved, *, seed=2027, dimensions=512):
    feature = finite(feature, "feature")
    require(
        feature.ndim == 2
        and train.dtype == bool
        and train.shape == (len(feature),)
        and train.sum() >= 128,
        "Invalid RFF source role",
    )
    source = np.asarray(feature[train], dtype=np.float64)
    mean, scale = source.mean(0), np.maximum(source.std(0), 0.001)
    errors = {
        "feature_mean": close(saved["feature_mean"], mean, "feature mean"),
        "feature_scale": close(saved["feature_scale"], scale, "feature scale"),
    }
    rng = np.random.default_rng(seed)
    ids = rng.choice(np.flatnonzero(train), 128, replace=False)
    close(saved["bandwidth_sample_indices"], ids, "bandwidth source sample", 0, 0)
    sample = (np.asarray(feature[ids], dtype=np.float64) - mean) / scale
    # Explicit pair differences are independent of the fitter's Gram expansion.
    distance = np.array(
        [
            np.square(sample[i] - sample[j]).sum()
            for i in range(128)
            for j in range(i + 1, 128)
        ]
    )
    bandwidth = float(np.median(distance))
    require(np.isfinite(bandwidth) and bandwidth > 0, "Invalid independent bandwidth")
    errors["bandwidth"] = close(saved["bandwidth_squared"], bandwidth, "bandwidth")
    rng = np.random.default_rng(seed)
    weight = rng.normal(size=(feature.shape[1], dimensions)) / np.sqrt(bandwidth)
    offset = rng.uniform(0, 2 * np.pi, dimensions)
    errors["weight"] = close(saved["weight"], weight, "RFF weight")
    errors["offset"] = close(saved["offset"], offset, "RFF offset", 0, 0)
    z = np.empty((len(feature), dimensions))
    for start in range(0, len(feature), 128):
        x = (np.asarray(feature[start : start + 128], dtype=np.float64) - mean) / scale
        z[start : start + len(x)] = np.sqrt(2 / dimensions) * np.cos(
            mm(x, weight) + offset
        )
    rmean, rscale = z[train].mean(0), np.maximum(z[train].std(0), 0.001)
    errors["rff_mean"] = close(saved["rff_mean"], rmean, "RFF mean")
    errors["rff_scale"] = close(saved["rff_scale"], rscale, "RFF scale")
    return (z - rmean) / rscale, errors


def source_validation_nje(prediction, target, rooms):
    """Independent equal-room NJE. Caller supplies SOURCE validation rows only."""
    prediction, target = (
        finite(prediction, "validation predictions"),
        finite(target, "validation target"),
    )
    require(
        prediction.shape == target.shape
        and target.ndim == 3
        and target.shape[1:] == (17, 2),
        "Validation pose shape differs",
    )
    require(
        set(rooms) <= {"E01", "E03"} and len(rooms) == len(target),
        "Held room passed to validation metric",
    )
    scale = np.linalg.norm(target.max(1) - target.min(1), axis=1)
    require((scale > 1e-6).all(), "Degenerate source-validation pose")
    error = np.linalg.norm(prediction - target, axis=2) / scale[:, None]
    return float(np.mean([error[rooms == room].mean() for room in sorted(set(rooms))]))


def independent_readout(
    z,
    source_target,
    validation_target,
    train,
    validation,
    rooms,
    state,
    selection,
    alphas=ALPHAS,
):
    """Qualify saved coefficients; never persist a fit or consume held labels."""
    n, dimensions = z.shape
    finite(z, "RFF design")
    require(not np.any(train & validation), "Source roles overlap")
    finite(source_target, "TRAIN labels", (int(train.sum()), 17, 2))
    finite(validation_target, "VAL labels", (int(validation.sum()), 17, 2))
    finite(state["weights"], "saved ridge coefficients", (len(alphas), dimensions, 34))
    finite(state["prediction"], "saved predictions", (n, 17, 2))
    xm, ym = z[train].mean(0), source_target.reshape(-1, 34).mean(0)
    close(state["x_mean"], xm, "ridge design mean")
    close(state["y_mean"], ym, "ridge target mean")
    x, y = z[train] - xm, source_target.reshape(-1, 34) - ym
    gram, right = mm(x.T, x), mm(x.T, y)
    require(len(selection["grid"]) == len(alphas), "Alpha grid length differs")
    grids, residuals, coefficient_errors, metric_errors = [], [], [], []
    for i, alpha in enumerate(alphas):
        system = gram + len(x) * alpha * np.eye(dimensions)
        direct = solve(system, right, assume_a="pos", check_finite=True)
        saved = state["weights"][i]
        residual = float(
            np.linalg.norm(mm(system, saved) - right)
            / max(np.linalg.norm(right), 1e-12)
        )
        require(np.isfinite(residual) and residual < 1e-8, "Saved ridge equation fails")
        residuals.append(residual)
        coefficient_errors.append(
            close(saved, direct, "direct ridge coefficients", RIDGE_ATOL, RIDGE_RTOL)
        )
        pred = (mm(z[validation] - xm, direct) + ym).reshape(-1, 17, 2)
        nje = source_validation_nje(pred, validation_target, rooms[validation])
        grid = selection["grid"][i]
        require(grid["alpha"] == alpha, "Alpha order differs")
        metric_errors.append(
            close(
                grid["source_validation_nje"], nje, "source-validation NJE", 2e-9, 2e-8
            )
        )
        finite(grid["normal_equation_relative_residual"], "reported ridge residual")
        require(
            0 <= grid["normal_equation_relative_residual"] < 1e-6,
            "Reported normal equation fails",
        )
        grids.append(nje)
    selected = int(np.argmin(grids))
    require(
        np.asarray(state["selected"]).shape == ()
        and np.asarray(state["selected"]).dtype.kind in "iu"
        and int(state["selected"]) == selected == selection["selected"],
        "Source-only alpha selection differs",
    )
    prediction = (mm(z - xm, state["weights"][selected]) + ym).reshape(n, 17, 2)
    prediction_error = close(
        state["prediction"],
        prediction,
        "saved linear predictions",
        RIDGE_ATOL,
        RIDGE_RTOL,
    )
    return dict(
        selected=selected,
        alpha=alphas[selected],
        maximum_normal_equation_residual=max(residuals),
        maximum_direct_coefficient_error=max(coefficient_errors),
        maximum_source_validation_nje_error=max(metric_errors),
        maximum_prediction_error=prediction_error,
    )


def preoptimization(experiment, raw, cache, old_probe):
    registration = read(experiment / "registration.json")
    require(
        REQUIRED_TRAINING_SOURCES <= set(registration["source_sha256"]),
        "Training provenance omits required production sources",
    )
    seal = read(experiment / "preoptimization-seal.json")
    verify_hashes(experiment, seal["files_sha256"])
    verify_hashes(experiment / "executed-source", registration["source_sha256"])
    require(
        digest(raw / "receipt.json") == registration["raw_receipt_sha256"],
        "Raw receipt differs",
    )
    require(
        digest(cache / "receipt.json") == registration["cache_receipt_sha256"],
        "Cache receipt differs",
    )
    rr, cr = read(raw / "receipt.json"), read(cache / "receipt.json")
    require(
        REQUIRED_CACHE_SOURCES <= set(cr["source_sha256"]),
        "Cache provenance omits required production sources",
    )
    verify_hashes(cache / "executed-source", cr["source_sha256"])
    for name, expected_hash in cr["source_sha256"].items():
        if name in registration["source_sha256"]:
            require(
                registration["source_sha256"][name] == expected_hash,
                "Cache/experiment dependency source differs",
            )
    check_source_stage_contract(
        experiment / "executed-source/scripts/domain_masked_pretrain_probe.py"
    )
    require(
        rr["config"]["magnitude_units"] == "log_magnitude_db",
        "Raw magnitude units differ",
    )
    require(
        cr["raw_receipt_sha256"] == registration["raw_receipt_sha256"]
        and cr["raw_arrays_sha256"] == rr["arrays"],
        "Raw/cache lineage differs",
    )
    require(
        cr["kind"] == "full-source-covariance-feature-cache"
        and cr["fits_performed"] == 0
        and rr["records"] == cr["records"],
        "Cache provenance differs",
    )
    verify_hashes(raw, {name + ".npy": value for name, value in rr["arrays"].items()})
    verify_hashes(cache, cr["arrays_sha256"])
    verify_hashes(cache, read(cache / "artifact_manifest.json"))
    require(
        digest(experiment / "targets-and-metadata.npz")
        == cr["arrays_sha256"]["targets-and-metadata.npz"],
        "Copied target metadata differs",
    )
    oldname = "E02-congruence_dynamics-model-prediction.npz"
    oldmanifest = read(old_probe / "artifact_manifest.json")
    require(
        digest(old_probe / oldname)
        == oldmanifest[oldname]
        == registration["frozen_temporal_model_sha256"]
        == digest(experiment / "frozen-temporal-control.npz"),
        "Frozen temporal control differs",
    )
    require(
        oldmanifest["targets-and-metadata.npz"]
        == cr["arrays_sha256"]["targets-and-metadata.npz"],
        "Temporal target index lineage differs",
    )
    data = npz(experiment / "targets-and-metadata.npz")
    roles = npz(experiment / "roles.npz")
    schedule = np.load(experiment / "schedule.npy", allow_pickle=False)
    expected, offsets = roles_and_schedule(
        data, rr["records"], registration, roles, schedule, seal
    )
    require(
        len(data["end"]) == cr["logical_rows"] and cr["nominal_rows"] == 26730,
        "Cache logical/allocated rows differ",
    )
    baseline = npz(experiment / "frozen-temporal-control.npz")
    close(
        baseline["held_indices"],
        np.flatnonzero(expected["held"]),
        "temporal held indices",
        0,
        0,
    )
    finite(
        baseline["pose"], "temporal predictions", (int(expected["held"].sum()), 17, 2)
    )
    target_error = check_targets(raw, data, offsets)
    module = load_model_source(experiment)
    torch.set_num_threads(1)
    torch.manual_seed(registration["config"]["seed"])
    initialized = module.MaskedCSI().state_dict()
    initial = load_torch(experiment / "initial.pt")
    equal_state(initial, initialized, "seeded initial state")
    return (
        registration,
        data,
        expected,
        offsets,
        module,
        dict(
            stage="preoptimization",
            rows=len(offsets),
            counts={k: int(v.sum()) for k, v in expected.items()},
            schedule_sha256=digest(experiment / "schedule.npy"),
            unique_optimizer_endpoints=len(np.unique(schedule)),
            target_join_maximum_error=target_error,
            initial_state_exact=True,
            executed_source_sha256=registration["source_sha256"],
        ),
    )


def checkpoint_audit(experiment, registration, module):
    identity = {
        key: digest(experiment / name)
        for key, name in (
            ("registration", "registration.json"),
            ("schedule", "schedule.npy"),
            ("initial", "initial.pt"),
        )
    }
    steps = registration["config"]["steps"]
    generator = torch.Generator().manual_seed(registration["config"]["mask_seed"])
    mask_states = {}
    for step in range(1, steps + 1):
        # Independent random consumption: mask construction consumes one CPU
        # uniform number for each of128x95 tokens, without other RNG draws.
        torch.rand(registration["config"]["batch"] * 8, 95, generator=generator)
        if step % 1000 == 0 or step == steps:
            mask_states[step] = generator.get_state().clone()
    checked = 0
    for arm in ARMS:
        final = load_torch(experiment / (arm + "-final.pt"))
        require(
            read(experiment / (arm + "-complete.json"))
            == dict(
                identity=identity, model_sha256=digest(experiment / (arm + "-final.pt"))
            ),
            "Completed arm identity differs",
        )
        model = module.MaskedCSI()
        model.load_state_dict(final)
        training = read(experiment / (arm + "-training.json"))
        tensor_tree(training, arm + "/training")
        require(
            training["steps"] == (0 if arm == "random_frozen" else steps),
            "Arm optimizer budget differs",
        )
        if arm == "random_frozen":
            equal_state(
                final, load_torch(experiment / "initial.pt"), "random frozen state"
            )
            require(
                not list(experiment.glob(arm + "-step-*.pt")),
                "Random arm was optimized",
            )
            continue
        paths = sorted(experiment.glob(arm + "-step-*.pt"))
        require(
            [p.name for p in paths]
            == [f"{arm}-step-{step:05}.pt" for step in mask_states],
            "Checkpoint sequence differs",
        )
        final_compared = False
        for expected_step, path in zip(mask_states, paths):
            receipt = read(path.with_suffix(".json"))
            saved = load_torch(path)
            require(
                path.name == f"{arm}-step-{expected_step:05}.pt"
                and type(saved["step"]) is int
                and saved["step"] == expected_step
                and type(receipt["step"]) is int
                and receipt["step"] == expected_step
                and receipt["arm"] == saved["arm"] == arm,
                "Checkpoint filename, payload and receipt step/arm differ",
            )
            step = expected_step
            require(
                receipt
                == dict(
                    checkpoint_sha256=digest(path),
                    identity=identity,
                    arm=arm,
                    step=step,
                ),
                "Checkpoint seal differs",
            )
            require(
                saved["identity"] == identity
                and saved["arm"] == arm
                and saved["device"] == registration["device"],
                "Checkpoint identity differs",
            )
            require(set(saved["model"]) == set(final), "Checkpoint model keys differ")
            require(
                all(saved["model"][key].shape == final[key].shape for key in final),
                "Checkpoint model shapes differ",
            )
            close(
                saved["mask_generator"].numpy(),
                mask_states[step].numpy(),
                "mask generator state",
                0,
                0,
            )
            require(
                saved["cpu_rng"].dtype == torch.uint8 and saved["cpu_rng"].ndim == 1,
                "CPU RNG state differs",
            )
            require(
                (saved["mps_rng"] is not None) == (registration["device"] == "mps"),
                "MPS RNG presence differs",
            )
            if saved["mps_rng"] is not None:
                require(
                    saved["mps_rng"].dtype == torch.uint8
                    and saved["mps_rng"].ndim == 1,
                    "MPS RNG state differs",
                )
            require(saved["optimizer"]["state"], "Missing Adam state")
            groups = saved["optimizer"]["param_groups"]
            require(
                len(groups) == 1
                and groups[0]["lr"] == registration["config"]["learning_rate"]
                and groups[0]["weight_decay"] == registration["config"]["weight_decay"],
                "Adam hyperparameters differ",
            )
            for state in saved["optimizer"]["state"].values():
                require(int(state["step"]) == step, "Adam step differs from schedule")
            if step == steps:
                equal_state(final, saved["model"], "final checkpoint model")
                equal_tree(
                    load_torch(experiment / (arm + "-optimizer.pt")),
                    saved["optimizer"],
                    "final optimizer",
                )
                equal_tree(training["logs"], saved["logs"], "final training logs")
                final_compared = True
            checked += 1
        require(final_compared, "Final checkpoint comparison was not executed: " + arm)
    return dict(checkpoints=checked, shared_mask_generator_states_exact=True)


def replay_embeddings(experiment, raw, data, roles, offsets, module):
    # Five TRAIN, five VAL and six HELD rows, including both source rooms and
    # spread endpoints. No GT poses are passed to this function's model calls.
    ids = np.concatenate(
        [
            np.flatnonzero(roles[name])[
                np.linspace(0, roles[name].sum() - 1, count, dtype=int)
            ]
            for name, count in (("train", 5), ("validation", 5), ("held", 6))
        ]
    )
    rf = np.load(raw / "rf.npy", mmap_mode="r", allow_pickle=False)
    require(not np.iscomplexobj(rf), "Complex artifact: raw RF")
    require(rf.shape == (240570, 6, 114, 10), "Raw RF shape differs")
    frames = offsets[ids, None] + data["end"][ids, None] - np.arange(7, -1, -1)
    x = np.asarray(rf[frames, :3], dtype=np.float64)
    observed = np.isfinite(x)
    count = observed.sum(axis=(-3, -2, -1), keepdims=True)
    require((count > 0).all(), "All-missing replay frame")
    mean = np.where(observed, x, 0).sum(axis=(-3, -2, -1), keepdims=True) / count
    centered = np.where(observed, x - mean, 0)
    scale = np.sqrt(np.square(centered).sum(axis=(-3, -2, -1), keepdims=True) / count)
    require((scale >= 1e-6).all(), "Constant replay frame")
    histories = torch.from_numpy((centered / scale).astype(np.float32))
    errors = {}
    for arm in ARMS:
        model = module.MaskedCSI().eval()
        model.load_state_dict(load_torch(experiment / (arm + "-final.pt")))
        with torch.no_grad():
            replay = model.history(histories).numpy()
        saved = np.load(
            experiment / (arm + "-embedding.npy"), mmap_mode="r", allow_pickle=False
        )
        require(
            saved.shape == (len(offsets), 512) and saved.dtype == np.float32,
            "Embedding allocation differs",
        )
        errors[arm] = close(
            saved[ids],
            replay,
            "CPU/MPS embedding replay/" + arm,
            ENCODER_ATOL,
            ENCODER_RTOL,
        )
    return dict(
        rows=ids.tolist(),
        maximum_errors=errors,
        atol=ENCODER_ATOL,
        rtol=ENCODER_RTOL,
        tolerance_basis="float32 CPU versus MPS attention/reductions; prior synthetic discrepancy1.788e-7; bounds2e-5 absolute+relative, no accuracy scores",
    )


def verify(experiment, raw, cache, old_probe, output, *, preoptimization_only=False):
    require(not output.exists(), "Output already exists; preserve prior audit")
    auditor_sources = {
        str(Path(__file__).relative_to(ROOT)): digest(Path(__file__)),
        "scripts/research_numeric_readback.py": digest(
            ROOT / "scripts/research_numeric_readback.py"
        ),
    }
    if not preoptimization_only:
        require(
            (experiment / "source-ready.json").is_file(),
            "SOURCE_READY required; partial learned artifacts are not audited",
        )
        ready = read(experiment / "source-ready.json")
        require(
            ready["kind"] == "masked-csi-source-only-ready"
            and ready["held_metrics_computed"] is False,
            "Source-ready stage differs",
        )
        required = {
            "registration.json",
            "initial.pt",
            "schedule.npy",
            "roles.npz",
            "targets-and-metadata.npz",
            "preoptimization-seal.json",
            "frozen-source-selections.json",
            "frozen-temporal-control.npz",
            "raw_current-rff.npz",
            "raw_current-readout.npz",
            "supervised-training-statistics.npz",
        }
        required.update(
            arm + suffix
            for arm in ARMS
            for suffix in (
                "-final.pt",
                "-rff.npz",
                "-readout.npz",
                "-embedding.npy",
                "-training.json",
                "-complete.json",
            )
        )
        required.update(arm + "-optimizer.pt" for arm in ARMS[1:])
        required.update(
            f"{arm}-step-{step:05}{suffix}"
            for arm in ARMS[1:]
            for step in (1000, 2000, 3000, 4000)
            for suffix in (".pt", ".json")
        )
        require(
            required <= set(ready["artifacts_sha256"]),
            "Source-ready seal omits critical artifacts",
        )
        verify_hashes(experiment, ready["artifacts_sha256"])
    registration, data, roles, offsets, module, report = preoptimization(
        experiment, raw, cache, old_probe
    )
    report.update(
        kind="independent-masked-csi-source-only-audit",
        held_metrics_computed=False,
        model_training_performed=False,
        scope="SOURCE E01-E03; numeric joins may inspect held annotations but no held error/selection metric is computed",
        auditor_sha256=auditor_sources[str(Path(__file__).relative_to(ROOT))],
        auditor_sources_sha256=auditor_sources,
    )
    if not preoptimization_only:
        report["stage"] = "source-ready"
        report["source_ready_sha256"] = digest(experiment / "source-ready.json")
        report["checkpoints"] = checkpoint_audit(experiment, registration, module)
        report["encoder_replay"] = replay_embeddings(
            experiment, raw, data, roles, offsets, module
        )
        stats = npz(experiment / "supervised-training-statistics.npz")
        source = data["pose"][roles["train"]].reshape(-1, 34)
        close(stats["mean"], source.mean(0), "supervised mean", 0, 0)
        close(stats["scale"], np.maximum(source.std(0), 0.01), "supervised scale", 0, 0)
        selections = read(experiment / "frozen-source-selections.json")
        require(
            set(selections) == set((*ARMS, "raw_current")),
            "Source selection arms differ",
        )
        report["readouts"] = {}
        for arm in (*ARMS, "raw_current"):
            if arm == "raw_current":
                cached = np.load(
                    cache / "untreated_covariance-features.npy",
                    mmap_mode="r",
                    allow_pickle=False,
                )
                require(
                    cached.dtype == np.float32, "Covariance allocation dtype differs"
                )
                require(cached.shape == (26730, 9576), "Covariance allocation differs")
                # The current control is exactly lag-slot0's nine coordinates;
                # decode/reencode affects off-diagonals only at roundoff scale.
                feature = np.asarray(
                    cached[: len(offsets)].reshape(-1, 8, 19, 7, 9)[:, :, :, 0],
                    dtype=np.float64,
                ).reshape(len(offsets), 1368)
            else:
                feature = np.load(
                    experiment / (arm + "-embedding.npy"),
                    mmap_mode="r",
                    allow_pickle=False,
                )
            z, map_errors = independent_rff(
                feature,
                roles["train"],
                npz(experiment / (arm + "-rff.npz")),
                seed=registration["config"]["seed"],
            )
            result = independent_readout(
                z,
                data["pose"][roles["train"]],
                data["pose"][roles["validation"]],
                roles["train"],
                roles["validation"],
                data["room"],
                npz(experiment / (arm + "-readout.npz")),
                selections[arm],
            )
            result["map_errors"] = map_errors
            report["readouts"][arm] = result
        # Recheck the seal after replay so concurrent mutation cannot be accepted.
        verify_hashes(experiment, ready["artifacts_sha256"])
    verify_hashes(ROOT, auditor_sources)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("experiment", "raw", "cache", "old-probe", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--preoptimization-only", action="store_true")
    args = parser.parse_args()
    verify(
        args.experiment,
        args.raw,
        args.cache,
        args.old_probe,
        args.output,
        preoptimization_only=args.preoptimization_only,
    )


if __name__ == "__main__":
    main()
