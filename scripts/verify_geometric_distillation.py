"""Independent metrics and CPU CSI-only replay of a sealed geometry pilot.

Requires the experiment's immutable local artifacts, its original warm-start
study and typed RF cache. Shared model/reader classes are used for inference;
metric equations below do not call the producer's metric or gating functions.
No optimizer, training diagnostic, camera or target-room evaluation occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import domain_geometric_distillation_probe as producer  # noqa: E402
from scripts.domain_masked_pretrain_probe import HistoryReader  # noqa: E402


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bind(root, hashes):
    for name, expected in hashes.items():
        path = root / name
        require(
            path.resolve().is_relative_to(root.resolve()), "Artifact path escaped root"
        )
        require(digest(path) == expected, "Changed artifact: " + name)


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def metadata_only(path):
    # The producer container also has E02 targets. NPZ members are lazy: select
    # only identifiers/indices, rather than decoding any target-room annotations.
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in ("record", "person", "room", "action", "end")}


def scores(prediction, target, record, ends):
    raw_p, raw_y = np.asarray(prediction), np.asarray(target)
    require(
        raw_p.dtype.kind in "iuf" and raw_y.dtype.kind in "iuf",
        "Real metric arrays required",
    )
    p, y = raw_p.astype(float), raw_y.astype(float)
    record, ends = np.asarray(record), np.asarray(ends)
    require(
        p.shape == y.shape
        and p.ndim in (2, 3)
        and len(p) > 0
        and np.isfinite(p).all()
        and np.isfinite(y).all(),
        "Matching finite metric arrays required",
    )
    require(
        (p.shape[1:] == (17, 2) if p.ndim == 3 else p.shape[1:] == (136,)),
        "Pose or geometry metric schema required",
    )
    require(
        record.shape == ends.shape == (len(p),) and ends.dtype.kind in "iu",
        "Record/integer endpoint schema required",
    )
    lookup = {(str(r), int(e)): i for i, (r, e) in enumerate(zip(record, ends))}
    require(len(lookup) == len(p), "Duplicate record endpoints")
    pc, yc = np.empty_like(p), np.empty_like(y)
    for name in set(record):
        rows = record == name
        for value, centered in ((p, pc), (y, yc)):
            relative = value[rows] - value[rows][0]
            centered[rows] = relative - relative.mean(0)
    energy = float(np.square(yc).sum())
    pairs = [
        (i, lookup[(str(r), int(e) - 8)])
        for i, (r, e) in enumerate(zip(record, ends))
        if (str(r), int(e) - 8) in lookup
    ]
    delta_r2 = None
    if pairs:
        current, previous = np.asarray(pairs).T
        dy = y[current] - y[previous]
        dp = p[current] - p[previous]
        de = float(np.square(dy).sum())
        if de > 0:
            delta_r2 = float(1 - np.square(dp - dy).sum() / de)
    result = dict(
        motion_r2=float(1 - np.square(pc - yc).sum() / energy) if energy > 0 else None,
        delta_r2=delta_r2,
    )
    if p.ndim == 3:
        diagonal = np.sqrt(np.square(y.max(1) - y.min(1)).sum(1))
        require((diagonal > 1e-6).all(), "Degenerate target body")
        error = np.sqrt(np.square(p - y).sum(2)) / diagonal[:, None]
        result.update(
            nje=float(error.mean()),
            pck05=float((error <= 0.05).mean()),
            pck20=float((error <= 0.2).mean()),
        )
    else:
        result["mse"] = float(np.square(p - y).mean())
    return result


def close_scores(actual, reported):
    require(set(actual) == set(reported), "Metric fields differ")
    maximum = 0.0
    for key, value in actual.items():
        expected = reported[key]
        if value is None or expected is None:
            require(
                value is None and expected is None, "Undefined metric differs: " + key
            )
        else:
            require(np.isfinite(expected), "Nonfinite reported metric")
            error = abs(value - expected)
            require(error < 2e-12, "Metric disagreement: " + key)
            maximum = max(maximum, error)
    return maximum


def pose_report(row):
    return dict(
        motion_r2=row["pose"]["record_centered_articulation_r2"],
        delta_r2=row["temporal"]["lag8_zero_reference_delta_r2"],
        nje=row["pose"]["normalized_joint_error"],
        pck05=row["pose"]["pck_curve"]["0.05"],
        pck20=row["pose"]["pck_curve"]["0.2"],
    )


def verify(study, source, raw, out):
    require(not out.exists(), "Existing audit preserved")
    require(
        not any(
            out.resolve().is_relative_to(p.resolve()) for p in (study, source, raw)
        ),
        "Audit output must be outside immutable inputs",
    )
    manifest = read(study / "artifact-manifest.json")
    bind(study, manifest)
    registration = read(study / "registration.json")
    bind(ROOT, registration["source_sha256"])
    config = read(study / "config.json")
    require(config == registration["config"], "Config differs")
    data = arrays(study / "source-data.npz")
    stats = arrays(study / "TRAIN-statistics.npz")
    train, val = data["train"], data["validation"]
    require(
        train.dtype == val.dtype == bool
        and (train.sum(), val.sum()) == (14124, 3546)
        and not (train & val).any()
        and (train | val).all(),
        "Source roles differ",
    )
    require(
        set(data["room"]) == {"E01", "E03"}
        and not set(data["person"][train]) & set(data["person"][val]),
        "Source person/room leakage",
    )
    schedule = np.load(study / "schedule.npy", allow_pickle=False)
    require(
        schedule.shape == (2000, 8, 2) and train[schedule].all(),
        "Training schedule differs",
    )
    require(
        np.all(data["record"][schedule[..., 0]] == data["record"][schedule[..., 1]])
        and np.all(data["end"][schedule[..., 0]] - data["end"][schedule[..., 1]] == 8),
        "Training pairs crossed time/record",
    )
    for pairs in schedule:
        require(
            len(set(data["person"][pairs[:, 0]])) == 8
            and sum(data["room"][pairs[:, 0]] == "E01") == 4,
            "Room/person balancing differs",
        )
    permutation = np.load(study / "geometry-permutation.npy", allow_pickle=False)
    require(
        np.array_equal(np.sort(permutation[train]), np.flatnonzero(train))
        and np.array_equal(data["record"][permutation[train]], data["record"][train])
        and np.all(permutation[~train] == -1),
        "Geometry permutation leaked roles/records",
    )
    receipt = read(raw / "receipt.json")
    require(
        digest(raw / "rf.npy") == receipt["arrays"]["rf"]
        and digest(source / "masked_temporal-final.pt")
        == registration["init_encoder_sha256"],
        "RF/warm-start identity differs",
    )
    metadata = metadata_only(source / "targets-and-metadata.npz")
    for key in ("record", "person", "room", "action", "end"):
        require(
            np.array_equal(metadata[key][data["global_ids"]], data[key]),
            "Source row join differs",
        )
    allowed = np.zeros(len(metadata["record"]), bool)
    allowed[data["global_ids"][val]] = True
    reader = HistoryReader(
        np.load(raw / "rf.npy", mmap_mode="r", allow_pickle=False),
        metadata,
        receipt["records"],
        allowed,
    )
    ids = np.flatnonzero(val)
    reports = read(study / "results.json")
    evaluation = study / reports["evaluation_directory"]
    require(
        evaluation.resolve().is_relative_to(study.resolve()), "Evaluation escaped study"
    )
    results = {}
    maximum = 0.0
    groups_checked = 0
    identities = set()
    torch.set_num_threads(1)
    for arm in producer.ARMS:
        complete = read(study / arm / "complete.json")
        identities.add(complete["identity"])
        for step in (500, 1000, 1500, 2000):
            checkpoint = study / arm / f"step-{step:04}.pt"
            seal = read(checkpoint.with_name(checkpoint.name + ".json"))
            require(
                seal
                == dict(
                    kind="source-geometric-recovery",
                    step=step,
                    identity=complete["identity"],
                    arm=arm,
                    device="mps",
                    checkpoint_sha256=digest(checkpoint),
                ),
                "Checkpoint seal differs",
            )
        checkpoint = study / arm / "step-2000.pt"
        require(
            complete["step"] == 2000
            and complete["final_checkpoint_sha256"] == digest(checkpoint),
            "Completed checkpoint differs",
        )
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        require(
            saved["step"] == 2000 and saved["identity"] == complete["identity"],
            "Checkpoint payload differs",
        )
        model = producer.make_model(source, stats, "cpu")
        model.load_state_dict(saved["model"], strict=True)
        model.eval()
        pose = []
        geometry = []
        with torch.inference_mode():
            for start in range(0, len(ids), 16):
                rows = ids[start : start + 16]
                pred = model(torch.from_numpy(reader(data["global_ids"][rows])))
                pose.append(pred["pose"].numpy())
                geometry.append(pred["geometry"].numpy())
        pose = np.concatenate(pose)
        geometry = np.concatenate(geometry)
        stored_pose = np.load(evaluation / f"{arm}-VALpose.npy", allow_pickle=False)
        stored_geometry = np.load(
            evaluation / f"{arm}-VALgeometry.npy", allow_pickle=False
        )
        pose_error = float(np.max(np.abs(pose - stored_pose)))
        geometry_error = float(np.max(np.abs(geometry - stored_geometry)))
        require(
            pose_error < 3e-6 and geometry_error < 1e-5,
            "CPU inference differs from original MPS",
        )
        independent = {}
        groups = [("pooled", np.arange(len(ids)), reports["arms"][arm]["pooled"])]
        for field in ("room", "person", "action"):
            for label in sorted(set(data[field][ids])):
                groups.append(
                    (
                        field + "/" + str(label),
                        np.flatnonzero(data[field][ids] == label),
                        reports["arms"][arm]["per_" + field][str(label)],
                    )
                )
        for name, rows, reported in groups:
            actual = scores(
                stored_pose[rows],
                data["pose"][ids[rows]],
                data["record"][ids[rows]],
                data["end"][ids[rows]],
            )
            maximum = max(maximum, close_scores(actual, pose_report(reported)))
            actual_geometry = scores(
                stored_geometry[rows],
                data["geometry"][ids[rows]],
                data["record"][ids[rows]],
                data["end"][ids[rows]],
            )
            gr = reported["geometry"]
            maximum = max(
                maximum,
                close_scores(
                    actual_geometry,
                    dict(
                        motion_r2=gr["record_centered_shape_r2"],
                        delta_r2=gr["lag8_zero_reference_delta_r2"],
                        mse=gr["mse"],
                    ),
                ),
            )
            independent[name] = actual
            groups_checked += 1
        results[arm] = dict(
            cpu_pose_max_abs=pose_error,
            cpu_geometry_max_abs=geometry_error,
            independent=independent,
        )
    require(
        len(identities) == 1, "Arms have different initial/data/schedule identities"
    )
    bind(study, manifest)
    bind(ROOT, registration["source_sha256"])
    proof = dict(
        status="qualified",
        manifest_sha256=digest(study / "artifact-manifest.json"),
        auditor_sha256=digest(Path(__file__)),
        checkpoint_seals=16,
        independent_pose_and_geometry_groups=groups_checked,
        max_metric_disagreement=maximum,
        source_validation_endpoints_per_arm=len(ids),
        arms=results,
        limits="Shared producer model/RF reader for CPU execution; independent saved prediction metric equations. No refitting, numeric target-room features/annotations, calibration scoring or claim that original efficacy gates passed. Whole raw-file hashing streams bytes without decoding target-room values.",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump(proof, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "source", "raw", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    result = verify(args.study, args.source, args.raw, args.out)
    print(
        json.dumps(
            dict(
                status=result["status"],
                metric_disagreement=result["max_metric_disagreement"],
                seconds=time.monotonic() - started,
            )
        )
    )
