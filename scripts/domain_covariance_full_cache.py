"""Build a reusable full E01-E03 covariance cache without fitting any model.

Only the independently verified A03/A07/A12 cache supplies reused features.
Interrupted, unverified feature arrays are never trusted or modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import domain_covariance_source_probe as study  # noqa: E402

ARMS = tuple(study.CONFIG["arms"])
METADATA_KEYS = {"pose", "delta", "room", "person", "record", "action", "end"}
FROZEN_SOURCES = (
    "scripts/domain_covariance_source_probe.py",
    "ghostmesh/covariance_dynamics.py",
    "ghostmesh/covariance_accelerated.py",
    "ghostmesh/metrics.py",
    "scripts/domain_background_probe.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def verify_hashes(directory, hashes):
    """Verify actual bytes, including allocated feature tails when present."""
    for name, expected in hashes.items():
        relative = Path(name)
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            "Unsafe manifest path",
        )
        path = directory / relative
        require(
            path.is_file() and study.digest(path) == expected,
            "Input hash mismatch: " + str(path),
        )


def logical_features(directory, rows, nominal_rows):
    """Metadata defines the valid prefix; allocated bytes never define records."""
    result = {}
    require(0 < rows <= nominal_rows, "Invalid logical feature row count")
    for arm in ARMS:
        value = np.load(
            directory / (arm + "-features.npy"), mmap_mode="r", allow_pickle=False
        )
        require(
            value.dtype == np.dtype("float32")
            and value.shape == (nominal_rows, study.CONFIG["raw_dimensions"][arm]),
            "Unexpected feature allocation: " + arm,
        )
        result[arm] = value[:rows]
    return result


def record_blocks(data, quality, expected_ids):
    """Recover exact record cursors using saved record/end labels and quality."""
    require(set(data) == METADATA_KEYS, "Unexpected target metadata fields")
    n = len(data["pose"])
    require(all(len(value) == n for value in data.values()), "Metadata lengths differ")
    require(
        data["pose"].shape == data["delta"].shape == (n, 17, 2),
        "Unexpected target dimensions",
    )
    require(
        [row["record"] for row in quality] == list(expected_ids),
        "Quality record ordering differs",
    )
    cursor, blocks = 0, {}
    canonical = set(study.CONFIG["endpoints"])
    for row in quality:
        name = row["record"]
        ends = np.asarray(row["retained_endpoints"], dtype=np.int64)
        require(
            ends.ndim == 1
            and list(ends) == sorted(set(ends))
            and set(ends) <= canonical,
            "Invalid retained endpoint list",
        )
        count = len(ends)
        stop = cursor + count
        room, person, action = name.split("/")
        require(
            stop <= n and np.array_equal(data["end"][cursor:stop], ends),
            "Record endpoint cursor differs",
        )
        for key, expected in (
            ("record", name),
            ("room", room),
            ("person", person),
            ("action", action),
        ):
            require(
                np.array_equal(data[key][cursor:stop], np.repeat(expected, count)),
                "Record metadata differs: " + key,
            )
        blocks[name] = (slice(cursor, stop), row)
        cursor = stop
    require(cursor == n, "Unexplained target rows beyond known records")
    return blocks


def load_verified_limited(directory, raw_receipt, raw_receipt_sha256):
    """Require complete verification provenance before reading reusable blocks."""
    for name in ("verification.json", "artifact_manifest.json", "registration.json"):
        require(
            (directory / name).is_file(),
            "Reuse requires independently verified limited cache: " + name,
        )
    manifest = read_json(directory / "artifact_manifest.json")
    required = {arm + "-features.npy" for arm in ARMS}
    required.update(
        {
            "registration.json",
            "results.json",
            "targets-and-metadata.npz",
            "feature-quality.json",
        }
    )
    required.update("executed-source/" + name for name in FROZEN_SOURCES)
    require(required <= set(manifest), "Limited manifest omits required artifacts")
    verify_hashes(directory, manifest)
    reg = read_json(directory / "registration.json")
    verification = read_json(directory / "verification.json")
    require(
        reg["cache_receipt_sha256"] == raw_receipt_sha256
        and reg["cache_arrays_sha256"] == raw_receipt["arrays"],
        "Limited cache raw-source binding differs",
    )
    config = reg["config"]
    require(
        config["records"] == 90
        and config["cached_shapes"] is True
        and config["selected_actions"] == ["A03", "A07", "A12"],
        "Reuse scope must be verified fixed three-action cache",
    )
    for key in (
        "endpoints",
        "raw_history",
        "invalid_history_policy",
        "covariance_lags",
        "arms",
        "raw_dimensions",
        "maximum_rejected_fraction",
    ):
        require(
            config[key] == json.loads(json.dumps(study.CONFIG[key])),
            "Limited feature contract changed: " + key,
        )
    require(
        set(reg["source_sha256"]) == set(FROZEN_SOURCES), "Limited source list differs"
    )
    for name, expected in reg["source_sha256"].items():
        require(
            study.digest(ROOT / name)
            == expected
            == study.digest(directory / "executed-source" / name),
            "Frozen limited source binding differs: " + name,
        )
    require(
        verification["verification_script_sha256"]
        == study.digest(ROOT / "scripts/verify_covariance_source_probe.py"),
        "Independent verification source changed",
    )
    require(
        verification["raw_records_reconstructed"] == 90
        and verification["feature_arms_verified"] == 3
        and verification["models_verified"] == 9
        and verification["normal_equations_verified"] == 45
        and verification["input_and_artifact_hashes_verified"] is True
        and verification["source_statistics_bandwidth_alpha_selection_verified"]
        is True,
        "Incomplete independent verification",
    )
    require(
        verification["maximum_RFF_design_difference"] == 0
        and 0 <= verification["maximum_saved_prediction_difference"] <= 1e-9
        and 0 <= verification["maximum_normal_equation_relative_residual"] < 1e-6,
        "Independent verification numerical checks failed",
    )
    with np.load(directory / "targets-and-metadata.npz", allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    n = len(data["pose"])
    require(
        n == verification["retained_examples_verified"],
        "Verified logical feature row count differs",
    )
    quality = read_json(directory / "feature-quality.json")
    expected = [r["id"] for r in study.selected_records(raw_receipt["records"], True)]
    blocks = record_blocks(data, quality, expected)
    features = logical_features(directory, n, 90 * len(study.CONFIG["endpoints"]))
    return (
        data,
        features,
        blocks,
        {
            "directory": str(directory.resolve()),
            "verification_sha256": study.digest(directory / "verification.json"),
            "artifact_manifest_sha256": study.digest(
                directory / "artifact_manifest.json"
            ),
            "registration_sha256": study.digest(directory / "registration.json"),
            "verified_records": 90,
            "verified_logical_rows": n,
            "verification": verification,
        },
    )


def coverage(data, records):
    nominal = len(records) * len(study.CONFIG["endpoints"])
    require(
        len(data["pose"]) >= 0.9 * nominal,
        "More than ten percent histories unavailable",
    )
    actions = {f"A{i:02}" for i in range(1, 28)}
    require(
        set(data["person"]) == {f"S{i:02}" for i in range(1, 31)}
        and set(data["action"]) == actions
        and set(data["room"]) == set(study.CONFIG["folds"]),
        "Incomplete full-source person/action/room coverage",
    )
    result = {}
    for held in study.CONFIG["folds"]:
        masks = study.fold_masks(data["room"], data["person"], held)
        counts = {}
        for name, mask in zip(("train", "validation", "held"), masks):
            require(
                set(data["action"][mask]) == actions,
                "Every fold role must retain all 27 actions",
            )
            counts[name] = int(mask.sum())
        result[held] = counts
    return result


def run(cache, limited, out, *, excluded_partial=None):
    require(not out.exists(), "Output already exists; preserve prior bytes")
    started = time.monotonic()
    raw_receipt = read_json(cache / "receipt.json")
    raw_receipt_sha = study.digest(cache / "receipt.json")
    records = study.selected_records(raw_receipt["records"])
    require(
        raw_receipt["frames"] == 240570
        and set(raw_receipt["arrays"]) == {"rf", "pose", "valid"},
        "Unexpected complete-source receipt",
    )
    raw_hashes = {name + ".npy": value for name, value in raw_receipt["arrays"].items()}
    verify_hashes(cache, raw_hashes)
    rf, pose, valid = [
        np.load(cache / (name + ".npy"), mmap_mode="r", allow_pickle=False)
        for name in ("rf", "pose", "valid")
    ]
    require(
        rf.shape == (240570, 6, 114, 10)
        and pose.shape == (240570, 17, 2)
        and valid.all(),
        "Unexpected raw RF or pose validity",
    )
    old_data, old_features, old_blocks, reuse = load_verified_limited(
        limited, raw_receipt, raw_receipt_sha
    )
    source_paths = (
        *FROZEN_SOURCES,
        "scripts/verify_covariance_source_probe.py",
        "scripts/domain_covariance_full_cache.py",
    )
    sources = {name: study.digest(ROOT / name) for name in source_paths}
    nominal = len(records) * len(study.CONFIG["endpoints"])
    registration = {
        "schema_version": 1,
        "kind": "full-source-covariance-feature-cache",
        "scope": study.CONFIG["scope"],
        "fits_performed": 0,
        "feature_config": study.CONFIG,
        "nominal_rows": nominal,
        "raw_receipt_sha256": raw_receipt_sha,
        "raw_arrays_sha256": raw_receipt["arrays"],
        "source_sha256": sources,
        "reuse_input": reuse,
        "software": {"numpy": np.__version__, "python": platform.python_version()},
        "excluded_partial": (
            {
                "directory": str(excluded_partial.resolve()),
                "reason": "Unverified interrupted arrays; no bytes reused or modified",
                "registration_sha256": study.digest(
                    excluded_partial / "registration.json"
                ),
            }
            if excluded_partial is not None
            else None
        ),
    }
    out.mkdir(parents=True)
    study.save(out / "registration.json", registration)
    for name in source_paths:
        destination = out / "executed-source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    evidence = out / "input-evidence"
    evidence.mkdir()
    shutil.copyfile(cache / "receipt.json", evidence / "raw-receipt.json")
    for name in ("verification.json", "artifact_manifest.json", "registration.json"):
        shutil.copyfile(limited / name, evidence / ("limited-" + name))
    features = {
        arm: np.lib.format.open_memmap(
            out / (arm + "-features.npy"),
            mode="w+",
            dtype="float32",
            shape=(nominal, study.CONFIG["raw_dimensions"][arm]),
        )
        for arm in ARMS
    }
    parts = {key: [] for key in METADATA_KEYS}
    quality, cursor, reused_rows = [], 0, 0
    try:
        for index, record in enumerate(records):
            offset, name = record["offset"], record["id"]
            if name in old_blocks:
                block, q = old_blocks[name]
                ends = np.asarray(q["retained_endpoints"], dtype=np.int64)
                values = {arm: old_features[arm][block] for arm in ARMS}
                reused_rows += len(ends)
            else:
                values, ends, condition, rejected = study.available_examples(
                    rf[offset : offset + 297],
                    feature_function=study.cached_shape_features,
                )
                q = {
                    "record": name,
                    "maximum_covariance_condition": condition,
                    "retained_endpoints": ends.tolist(),
                    "rejected_frames": rejected,
                }
            count = len(ends)
            relative = np.asarray(pose[offset : offset + 297], dtype="float64")
            relative = relative - relative[:, [11, 12]].mean(1, keepdims=True)
            targets, deltas = relative[ends], relative[ends] - relative[ends - 8]
            if name in old_blocks:
                require(
                    np.array_equal(old_data["pose"][block], targets)
                    and np.array_equal(old_data["delta"][block], deltas),
                    "Reused raw-pose target join changed",
                )
            for arm in ARMS:
                require(
                    values[arm].shape == (count, study.CONFIG["raw_dimensions"][arm])
                    and np.isfinite(values[arm]).all(),
                    "Invalid feature block",
                )
                features[arm][cursor : cursor + count] = values[arm]
            room, person, action = name.split("/")
            for key, value in (("pose", targets), ("delta", deltas), ("end", ends)):
                parts[key].append(value)
            for key, value in (
                ("record", name),
                ("room", room),
                ("person", person),
                ("action", action),
            ):
                parts[key].append(np.repeat(value, count))
            quality.append(q)
            cursor += count
            if (index + 1) % 10 == 0 or index + 1 == len(records):
                progress = {
                    "feature_records": index + 1,
                    "total_records": len(records),
                    "logical_rows": cursor,
                    "reused_rows": reused_rows,
                    "seconds": time.monotonic() - started,
                }
                study.save(out / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        for value in features.values():
            value.flush()
        data = {key: np.concatenate(values) for key, values in parts.items()}
        record_blocks(data, quality, [r["id"] for r in records])
        fold_coverage = coverage(data, records)
        np.savez(out / "targets-and-metadata.npz", **data)
        study.save(out / "feature-quality.json", quality)
        # Recheck mutable provenance after computation and before issuing a receipt.
        verify_hashes(cache, raw_hashes)
        require(
            study.digest(cache / "receipt.json") == raw_receipt_sha,
            "Raw receipt changed during build",
        )
        verify_hashes(ROOT, sources)
        verify_hashes(limited, read_json(evidence / "limited-artifact_manifest.json"))
        require(
            study.digest(limited / "verification.json") == reuse["verification_sha256"],
            "Limited verification changed during build",
        )
        output_names = [arm + "-features.npy" for arm in ARMS]
        output_names += ["targets-and-metadata.npz", "feature-quality.json"]
        receipt = dict(
            registration,
            logical_rows=cursor,
            allocated_rows=nominal,
            records=records,
            coverage_per_fold=fold_coverage,
            retained_fraction=cursor / nominal,
            dtype="float32",
            logical_prefix_policy="All arrays use only [:logical_rows]; allocated tail is invalid",
            arrays_sha256={name: study.digest(out / name) for name in output_names},
            reusecounts={
                "records": len(old_blocks),
                "rows": reused_rows,
                "computed_records": len(records) - len(old_blocks),
                "computed_rows": cursor - reused_rows,
                "unverified_partial_records": 0,
            },
            seconds=time.monotonic() - started,
        )
        study.save(out / "receipt.json", receipt)
        study.save(
            out / "artifact_manifest.json",
            {
                str(p.relative_to(out)): study.digest(p)
                for p in sorted(out.rglob("*"))
                if p.is_file()
            },
        )
        return receipt
    except BaseException as error:
        for value in features.values():
            value.flush()
        study.save(
            out / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "completed_records": len(quality),
                "logical_rows": cursor,
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--limited", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--excluded-partial", type=Path)
    args = parser.parse_args()
    run(args.cache, args.limited, args.out, excluded_partial=args.excluded_partial)
