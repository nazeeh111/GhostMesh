"""Build source-only RF histories with an explicit four-frame output delay.

No annotations or encoder are loaded. The common forty-four-frame guard keeps
the cohort comparable to the earlier immediate/delayed recurrence experiment.
It is not a claim that the physical sensor clock has been calibrated.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.motion_filter import packet_moments  # noqa:E402
from scripts.build_moment_cache import digest, require, retained_endpoints, save  # noqa:E402
from scripts.verify_geometric_distillation import read  # noqa:E402


def output_endpoints(rejected_frames):
    rejected = np.asarray(rejected_frames)
    require(
        rejected.ndim == 1
        and rejected.dtype.kind in "iu"
        and ((rejected >= 0) & (rejected < 297)).all(),
        "Integer rejected frame indices required",
    )
    valid = np.ones(297, dtype=bool)
    valid[rejected] = False
    return np.asarray(
        [end for end in range(39, 293, 8) if valid[end - 39 : end + 5].all()],
        dtype=np.int64,
    )


def history_indices(ends):
    ends = np.asarray(ends)
    require(
        ends.ndim == 1
        and ends.dtype.kind in "iu"
        and ((ends >= 39) & (ends <= 287) & ((ends - 39) % 8 == 0)).all(),
        "Frozen target endpoint grid required",
    )
    return ends[:, None] + 4 - np.arange(31, -1, -1)[None]


def build(raw, out):
    require(not out.exists(), "Existing delayed cache preserved")
    require(not out.resolve().is_relative_to(raw.resolve()), "Output inside raw inputs")
    receipt = read(raw / "receipt.json")
    records = [r for r in receipt["records"] if r["id"].startswith("E01/")]
    expected = {
        f"E01/S{person:02d}/A{action:02d}"
        for person in range(1, 11)
        for action in range(1, 28)
    }
    require(
        len(records) == 270
        and {r["id"] for r in records} == expected
        and all(r["frames"] == 297 for r in records),
        "Complete E01 source records required",
    )
    require(digest(raw / "rf.npy") == receipt["arrays"]["rf"], "RF bytes differ")
    rf = np.load(raw / "rf.npy", mmap_mode="r", allow_pickle=False)
    require(
        rf.dtype == np.float32
        and rf.shape == (receipt["frames"], 6, 114, 10)
        and receipt["config"]["magnitude_units"] == "log_magnitude_db"
        and receipt["config"]["phase_units"] == "radians",
        "Explicit typed RF units/schema required",
    )
    paths = [
        Path(__file__),
        ROOT / "scripts/build_moment_cache.py",
        ROOT / "ghostmesh/motion_filter.py",
        ROOT / "ghostmesh/rf_features.py",
        ROOT / "ghostmesh/covariance_dynamics.py",
    ]
    registration = dict(
        kind="E01-label-free-delayed-moments",
        raw_receipt_sha256=digest(raw / "receipt.json"),
        raw_rf_sha256=receipt["arrays"]["rf"],
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in paths},
        context_frames=32,
        output_delay_frames=4,
        nominal_output_buffer_ms=400,
        physical_clock_calibrated=False,
        annotation_arrays_decoded=False,
        quality_guard="Target e-39 through e+4 inclusive, same RF-only guard as original; no jitter",
    )
    out.mkdir(parents=True)
    save(out / "registration.json", registration)
    for path in paths:
        target = out / "executed-source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    started = time.monotonic()
    quality = []
    metadata = {
        k: [] for k in ("room", "person", "action", "record", "end", "raw_index")
    }
    for record in records:
        start = record["offset"]
        _, rejected = retained_endpoints(np.asarray(rf[start : start + 297]))
        ends = output_endpoints(rejected)
        quality.append(
            dict(
                record=record["id"],
                rejected_frames=rejected.tolist(),
                ends=ends.tolist(),
            )
        )
        room, person, action = record["id"].split("/")
        for key, value in (
            ("room", room),
            ("person", person),
            ("action", action),
            ("record", record["id"]),
        ):
            metadata[key].extend([value] * len(ends))
        metadata["end"].extend(ends.tolist())
        metadata["raw_index"].extend((start + ends).tolist())
    n = len(metadata["end"])
    require(n > 0, "No accepted RF histories")
    feature = np.lib.format.open_memmap(
        out / "features.npy", mode="w+", dtype=np.float32, shape=(n, 32, 12, 114)
    )
    cursor = 0
    for record, accepted in zip(records, quality):
        ends = np.asarray(accepted["ends"], dtype=np.int64)
        if not len(ends):
            continue
        start = record["offset"]
        moments = packet_moments(np.asarray(rf[start : start + 297]))
        feature[cursor : cursor + len(ends)] = moments[history_indices(ends)]
        cursor += len(ends)
    require(cursor == n, "Allocated/source row count differs")
    feature.flush()
    del feature
    np.savez_compressed(
        out / "metadata.npz", **{k: np.asarray(v) for k, v in metadata.items()}
    )
    save(out / "quality.json", quality)
    for path in paths:
        require(
            digest(path) == registration["source_sha256"][str(path.relative_to(ROOT))],
            "Source changed during cache build",
        )
    result = dict(
        kind=registration["kind"],
        registration_sha256=digest(out / "registration.json"),
        logical_rows=n,
        allocated_rows=n,
        annotation_arrays_decoded=False,
        arrays_sha256={
            name: digest(out / name)
            for name in ("features.npy", "metadata.npz", "quality.json")
        },
        seconds=time.monotonic() - started,
    )
    save(out / "receipt.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.raw, args.out))
