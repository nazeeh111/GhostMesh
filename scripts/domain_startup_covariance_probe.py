"""Unlabeled per-record startup covariance: bounded source-development probe.

A frozen anchor uses ONLY the eight CURRENT covariance states of the earliest
available forty-frame endpoint in its own recording. This mode requires a known
recording/reset boundary. It is neither the previous strict zero-shot mode nor
room calibration with pose labels. Unknown initial pose remains unobserved.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.covariance_dynamics import hermitian_coordinates  # noqa: E402
from ghostmesh.metrics import pose_metrics  # noqa: E402
from ghostmesh.reference_covariance import (  # noqa: E402
    RoomReference,
    current_covariances,
    fit_reference,
    reference_features,
)
from scripts.domain_background_probe import centered_r2, digest, multiply  # noqa: E402

FULL_CONFIG = dict(
    scope="SOURCE E01-E03 DEVELOPMENT, zero pose labels for startup; per-record causal startup mode, not previous strictzero-shot or room pose-label calibration",
    records=810,
    actions=27,
    expected_logical_rows=26556,
    primary="startup_shape",
    arms=["startup_shape", "startup_shape_volume", "raw_current", "startup_frozen_raw"],
    dimensions=dict(
        startup_shape=456,
        startup_shape_volume=608,
        raw_current=1368,
        startup_frozen_raw=1368,
    ),
    anchor="Earliest quality-retained canonical40-frame endpoint PER RECORD;8CURRENT states ending at that endpoint, normalize EACH by determinant^(1/3) BEFORE averaging. Freeze for all later endpoints. No GT/action/person inputs, future updates or other-record statistics",
    query="Require query end>=startup anchor end and same record; reset at every known record boundary. First score can be the startup endpoint because all its CSI has already arrived",
    availability="Same preverified26556 forty-frame-valid rows; no filled covariance, row repair or additional drop",
    target="Current hip-midpoint-relative17x2 supplied2D joints; translation only removed, no scale/rotation/reflection fit, no fullpose/globalposition/3D claim",
    nominal_startup="First nominal end39 implies3.9s since frame0 at nominal10Hz; later first-valid endpoints reported explicitly, not live measured acquisition latency",
    startup_raw_control="Copy startup8x19x9 unnormalized current-covariance encoding to every later row in same record; no current moving CSI. A learned template/reset control, not a motion input",
    volume="Separate608dimensional ablation includes log-volume relative to startup mean; NOT independently frame-gain invariant. Primary456 excludes it",
    seed=2027,
    permutation_seed=90210,
    rff_dimensions=512,
    stats="TRAIN-only feature mean/SDfloor.001 and RFFmean/SDfloor.001; TRAIN128 median squared distance; ridge n*alpha, five alphas",
    alphas=[0.001, 0.01, 0.1, 1.0, 10.0],
    selection="Equal-room source-validation articulationNJE only, first alpha tie. ALL3fold selections frozen before held scoring; no held target labels for model/stats/reference choice",
    folds="Each SOURCE heldroom:16TRAIN/4sourceVAL/10HELDpeople, all27actions. No role overlap",
    controls=[
        "source_body_mean",
        "source_endpoint_clock",
        "raw_current",
        "startup_frozen_raw",
        "frozen_temporal",
    ],
    gate="PrimaryNJE lower than mean/clock/rawcurrent/startupfrozen in EACH3rooms; PCK.05/.2 no worse than BOTH mean/clock; centered articulationR2 positive and RFshuffleNJE worse in EACHroom. Frozen temporal comparator also reported. Heuristicdevelopment only",
    limitations="Fixed per-bin receiver mixing within a session is unproven on real rooms. One generalized spectrum is noninjective. Unknown startup pose/DC remains; known reset/action-record boundaries required. No per-query labels or means. Dataset action-reset assumption is not automatically a streaming deployment capability",
)
LOCAL_CONFIG = dict(
    record="E01/S01/A12",
    source="Verified work/domain-covariance-positive-v1 untreated covariance features and exact164row targets/supports",
    counts=dict(train=79, validation=13, test=72),
    raw_blocks_half_open=dict(train=[0, 118], validation=[126, 178], test=[186, 297]),
    variants=["continuous", "reset_stress"],
    primary_variant="continuous",
    continuous_anchor="TRAIN startup at end39, current states32..39, forty-frame validity support0..39; frozen across all79/13/72 observations. Shared historicalTRAIN preprocessing state. Observation40frame supports remain purged, but total feature raw supports SHARE the TRAINstartup summary",
    reset_stress="Separate unknown-initial-state stress ablation: reset at39/165/225. Each block owns its earliest startup; observation and anchor supports stay within its block. This variant cannot replace a failed primary continuous gate",
    primary="startup_shape",
    normalization="TRAIN-only mean/SDfloor.05, inherited local exactlinear/RBF kernel; allTRAINpair median squared RBFdistance",
    kernels=["linear", "rbf"],
    lambdas=[1e-5, 0.001, 0.1, 10.0],
    selection="Perkernel lambda then primarykernel selected ONLYvalidation articulationNJE; first ties. No clipping hiprelativecoordinates",
    gate="Selected primaryshape testrecordR2>=.5, alignedminusshuffleR2>=.3, NJE<=.75TRAINmean and NJE<startupfrozen. Volume cannot rescue a failed primary gate",
    constant_design="If TRAINfeatures constant, no validRBFwidth exists: return TRAINtargetmean and explicitly report rankzero/no fitted RFreadout. Do not fabricate bandwidth",
    scope="Previously studied local sourceperson/room/action; purgedheldtime observability control, not unseenroom or untouched finaltest",
)


def save(path, obj):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Startup:
    reference: RoomReference
    end: int
    raw_current: np.ndarray

    def __post_init__(self):
        value = np.array(self.raw_current, dtype="float64", copy=True)
        require(
            value.shape == (1368,) and np.isfinite(value).all(),
            "Invalid startup raw coordinates",
        )
        value.setflags(write=False)
        object.__setattr__(self, "raw_current", value)


def validate_metadata(record, end, n):
    record, end = np.asarray(record), np.asarray(end)
    require(
        record.shape == end.shape == (n,) and n > 0,
        "Expected one record and endpoint per row",
    )
    require(
        np.issubdtype(end.dtype, np.integer) and np.all(end >= 39),
        "Expected canonical forty-frame endpoints",
    )
    require(
        len(set(zip(record.tolist(), end.tolist()))) == n, "Duplicate record endpoint"
    )
    return record, end


def fit_startups(untreated, record, end):
    """No pose/label argument; read exactly one earliest CSI row per recording."""
    record, end = validate_metadata(record, end, len(untreated))
    result = {}
    for name in np.unique(record):
        ids = np.flatnonzero(record == name)
        index = ids[np.argmin(end[ids])]
        current = current_covariances(untreated[index : index + 1])[0]
        result[str(name)] = Startup(
            fit_reference(current),
            int(end[index]),
            hermitian_coordinates(current).reshape(-1),
        )
    return result


def transform_startups(untreated, record, end, startup):
    """Use fixed own-record anchors; no fitting, labels, query mean or future CSI."""
    record, end = validate_metadata(record, end, len(untreated))
    result = {
        name: np.empty((len(untreated), width))
        for name, width in FULL_CONFIG["dimensions"].items()
    }
    for name in np.unique(record):
        require(str(name) in startup, "Missing own-record startup")
        frozen = startup[str(name)]
        ids = np.flatnonzero(record == name)
        require(np.all(end[ids] >= frozen.end), "Query precedes startup availability")
        current = current_covariances(untreated[ids])
        volume = reference_features(current, frozen.reference, include_volume=True)
        result["startup_shape"][ids] = volume[..., :3].reshape(len(ids), 456)
        result["startup_shape_volume"][ids] = volume.reshape(len(ids), 608)
        result["raw_current"][ids] = hermitian_coordinates(current).reshape(
            len(ids), 1368
        )
        result["startup_frozen_raw"][ids] = frozen.raw_current
    require(
        all(np.isfinite(value).all() for value in result.values()),
        "Nonfinite startup features",
    )
    return result


def articulation(target):
    target = np.asarray(target, dtype="float64")
    require(
        target.ndim == 3 and target.shape[1:] == (17, 2) and np.isfinite(target).all(),
        "Expected finite2D17joint teachers",
    )
    return target - target[:, [11, 12]].mean(1, keepdims=True)


def score(prediction, target, record):
    metrics = pose_metrics(prediction, target, np.ones(target.shape[:2], bool))
    metrics.pop("per_frame_nje")
    return dict(
        metrics=metrics,
        record_articulation_r2=centered_r2(prediction, target, record)["r2"],
    )


def local_design(x, train):
    mean = x[train].mean(0)
    scale = np.maximum(x[train].std(0), 0.05)
    z = (x - mean) / scale
    # Frozen-startup TRAIN is one constant template in this one-record control.
    constant = np.max(np.ptp(x[train], axis=0)) == 0
    if constant:
        return (
            {},
            dict(
                mean=mean,
                scale=scale,
                bandwidth_squared=np.asarray(0),
                training_design=z[train],
            ),
            True,
        )
    source = z[train]
    cross = multiply(z, source.T)
    squared = np.maximum(
        np.square(z).sum(1)[:, None] + np.square(source).sum(1)[None] - 2 * cross, 0
    )
    width = float(np.median(squared[train][np.triu_indices(train.sum(), 1)]))
    require(
        np.isfinite(width) and width > 0,
        "Nonconstant design has invalid TRAIN RBF bandwidth",
    )
    return (
        dict(linear=cross / x.shape[1], rbf=np.exp(-squared / (2 * width))),
        dict(
            mean=mean,
            scale=scale,
            bandwidth_squared=np.asarray(width),
            training_design=source,
        ),
        False,
    )


def local_fit(kernel, target, train, validation):
    """Read labels ONLY through TRAIN and validation indexes; return all4 weights."""
    require(
        not np.any(train & validation) and train.any() and validation.any(),
        "Source roles overlap or empty",
    )
    y = target[train].reshape(train.sum(), 34)
    mean = y.mean(0)
    matrix = kernel[train]
    eigen, vector = eigh(matrix, check_finite=True)
    right = multiply(vector.T, y - mean)
    weights, candidates = [], []
    for alpha in LOCAL_CONFIG["lambdas"]:
        weight = multiply(vector, right / (eigen[:, None] + alpha))
        residual = multiply(matrix, weight) + alpha * weight - (y - mean)
        error = float(np.linalg.norm(residual) / max(np.linalg.norm(y - mean), 1e-12))
        require(
            error < 1e-6 and np.isfinite(error),
            "Kernel normal-equation residual too large",
        )
        val_pred = (multiply(kernel[validation], weight) + mean).reshape(-1, 17, 2)
        nje = score(
            val_pred, target[validation], np.repeat("validation", validation.sum())
        )["metrics"]["normalized_joint_error"]
        weights.append(weight)
        candidates.append(
            dict(
                alpha=alpha, validation_nje=nje, normal_equation_relative_residual=error
            )
        )
    selected = min(
        range(len(candidates)), key=lambda i: candidates[i]["validation_nje"]
    )
    return dict(
        weights=np.stack(weights), target_mean=mean, selected=np.asarray(selected)
    ), candidates


def local_gate(primary, static, frozen):
    r2 = primary["test"]["record_articulation_r2"]
    shuffle = primary["shuffle"]["record_articulation_r2"]
    nje = primary["test"]["metrics"]["normalized_joint_error"]
    comparable = r2 is not None and shuffle is not None
    difference = r2 - shuffle if comparable else None
    return dict(
        motion_r2=r2,
        aligned_minus_shuffle_r2=difference,
        nje_fraction_of_mean=nje / static["metrics"]["normalized_joint_error"]
        if static["metrics"]["normalized_joint_error"] > 0
        else None,
        beats_startup_frozen=nje < frozen["metrics"]["normalized_joint_error"],
        passed=bool(
            comparable
            and r2 >= 0.5
            and difference >= 0.3
            and nje <= 0.75 * static["metrics"]["normalized_joint_error"]
            and nje < frozen["metrics"]["normalized_joint_error"]
        ),
    )


def local(predecessor, out):
    if out.exists():
        raise FileExistsError(out)
    started = time.monotonic()
    manifest = json.loads((predecessor / "artifact-manifest-final.json").read_text())
    manifest = manifest.get("sha256", manifest)
    consumed = [
        "untreated_covariance-features.npy",
        "indices.npz",
        "registration.json",
        "verification.json",
    ]
    bindings = {}
    for name in consumed:
        require(
            digest(predecessor / name) == manifest[name],
            "Verified predecessor bytes differ: " + name,
        )
        bindings[name] = manifest[name]
    proof = json.loads((predecessor / "verification.json").read_text())
    require(
        proof["raw_RF_and_target_bindings_verified"]
        and proof["forty_frame_histories_disjoint_between_splits"],
        "Predecessor lacks verified RF/target/split bindings",
    )
    with np.load(predecessor / "indices.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    untreated = np.load(
        predecessor / "untreated_covariance-features.npy", allow_pickle=False
    )
    require(
        untreated.shape == (164, 9576)
        and set(data["records"]) == {LOCAL_CONFIG["record"]},
        "Not the fixed164row A12 local cache",
    )
    masks = {key: data["split"] == key for key in LOCAL_CONFIG["counts"]}
    for name, mask in masks.items():
        require(mask.sum() == LOCAL_CONFIG["counts"][name], "Local count differs")
        start, stop = LOCAL_CONFIG["raw_blocks_half_open"][name]
        require(
            np.array_equal(data["end"][mask], np.arange(start + 39, stop)),
            "Local endpoint set differs",
        )
    require(
        not any(
            set(data["raw_support"][masks[a]].ravel())
            & set(data["raw_support"][masks[b]].ravel())
            for a, b in (
                ("train", "validation"),
                ("train", "test"),
                ("validation", "test"),
            )
        ),
        "Local raw histories overlap",
    )
    source_paths = [
        Path(__file__),
        ROOT / "ghostmesh/reference_covariance.py",
        ROOT / "ghostmesh/covariance_dynamics.py",
        ROOT / "ghostmesh/metrics.py",
        ROOT / "scripts/domain_background_probe.py",
    ]
    sources = {str(path.relative_to(ROOT)): digest(path) for path in source_paths}
    out.mkdir(parents=True)
    save(
        out / "registration.json",
        dict(
            full_recipe=FULL_CONFIG,
            local_recipe=LOCAL_CONFIG,
            source_sha256=sources,
            consumed_predecessor_sha256=bindings,
            predecessor_manifest_sha256=digest(
                predecessor / "artifact-manifest-final.json"
            ),
        ),
    )
    for path in source_paths:
        dest = out / "executed-source" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    target = articulation(data["target"])
    train, validation, test = (masks[k] for k in ("train", "validation", "test"))
    np.savez(out / "targets-and-rows.npz", articulation_target=target, **data)
    static = np.broadcast_to(target[train].mean(0), target.shape)
    states, grids, designs, choices = {}, {}, {}, {}
    identities = {}
    for variant in LOCAL_CONFIG["variants"]:
        episode = data["records"] if variant == "continuous" else data["split"]
        # Deliberately fit the continuous reference only from TRAIN rows; the
        # reset stress reference uses its own block startup before any query.
        anchor_rows = train if variant == "continuous" else np.ones(len(target), bool)
        startup = fit_startups(
            untreated[anchor_rows], episode[anchor_rows], data["end"][anchor_rows]
        )
        features = transform_startups(untreated, episode, data["end"], startup)
        folder = out / variant
        folder.mkdir()
        identities[variant] = {
            name: dict(
                end=item.end,
                current_state_ends=list(range(item.end - 7, item.end + 1)),
                conservative_validity_support=list(range(item.end - 39, item.end + 1)),
                nominal_seconds_since_record0=item.end / 10,
            )
            for name, item in startup.items()
        }
        for name, item in startup.items():
            np.savez(
                folder / (name.replace("/", "-") + "-startup.npz"),
                anchors=item.reference.anchors,
                mean_log_volume=item.reference.mean_log_volume,
                raw_current=item.raw_current,
            )
        for name, values in features.items():
            np.save(folder / (name + "-features.npy"), values, allow_pickle=False)
            kernels, norm, constant = local_design(values, train)
            np.savez(folder / (name + "-normalization.npz"), **norm)
            if constant:
                key = variant + "/" + name + "/constant"
                states[key] = dict(prediction=static.copy(), constant=True)
                grids[key] = dict(
                    constant_design=True,
                    validation_nje=score(
                        static[validation],
                        target[validation],
                        data["records"][validation],
                    )["metrics"]["normalized_joint_error"],
                )
            for kind, kernel in kernels.items():
                state, grid = local_fit(kernel, target, train, validation)
                key = variant + "/" + name + "/" + kind
                states[key], grids[key], designs[key] = state, grid, kernel
                np.savez(folder / (name + "-" + kind + "-model.npz"), **state)
        primary_keys = [
            key
            for key in states
            if key.startswith(variant + "/" + LOCAL_CONFIG["primary"] + "/")
        ]

        def best_index(key):
            if states[key].get("constant", False):
                return grids[key]["validation_nje"]
            return float(grids[key][int(states[key]["selected"])]["validation_nje"])

        choices[variant] = min(primary_keys, key=best_index)
    save(out / "startup-identities.json", identities)
    # Both variants' choices are frozen together; continuous is always primary.
    save(
        out / "FROZEN-LOCAL-SELECTIONS.json",
        dict(
            primary_variant=LOCAL_CONFIG["primary_variant"],
            selected_per_variant=choices,
            candidates=grids,
        ),
    )
    rng = np.random.default_rng(FULL_CONFIG["permutation_seed"])
    perm = rng.permutation(test.sum())
    np.save(out / "test-permutation.npy", perm, allow_pickle=False)
    results = {}
    for key, state in states.items():
        if state.get("constant", False):
            prediction = state["prediction"]
        else:
            prediction = (
                multiply(designs[key], state["weights"][int(state["selected"])])
                + state["target_mean"]
            ).reshape(-1, 17, 2)
        np.savez(
            out / (key.replace("/", "-") + "-predictions.npz"), prediction=prediction
        )
        results[key] = {
            name: score(prediction[mask], target[mask], data["records"][mask])
            for name, mask in masks.items()
        }
        results[key]["shuffle"] = score(
            prediction[test][perm], target[test], data["records"][test]
        )
    static_score = score(static[test], target[test], data["records"][test])
    gates = {
        variant: local_gate(
            results[choices[variant]],
            static_score,
            results[variant + "/startup_frozen_raw/constant"]["test"],
        )
        for variant in choices
    }
    for name, expected in sources.items():
        require(digest(ROOT / name) == expected, "Source changed during local run")
    save(
        out / "results.json",
        dict(
            full_recipe=FULL_CONFIG,
            local_recipe=LOCAL_CONFIG,
            results=results,
            source_body_mean=static_score,
            selected_per_variant=choices,
            local_gates=gates,
            primary_continuation_passed=gates[LOCAL_CONFIG["primary_variant"]][
                "passed"
            ],
            seconds=time.monotonic() - started,
            scope="Local only. No full SOURCE fit or E04 evaluation performed",
        ),
    )
    print(
        json.dumps(
            dict(
                selected=choices, local_gates=gates, seconds=time.monotonic() - started
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predecessor", type=Path, default=ROOT / "work/domain-covariance-positive-v1"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    local(args.predecessor, args.out)
