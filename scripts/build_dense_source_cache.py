"""Build exactly allocated E01/E02 RF-only frame cache for dense source histories.

No annotation array is opened. Output delay/quality boundaries and all producer
versions are recorded. Requires typed RF cache from the documented acquisition.
"""

import argparse
from pathlib import Path
import sys
import json
import shutil
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_moment_cache as b  # noqa:E402
from scripts.verify_geometric_distillation import read, bind  # noqa:E402
from ghostmesh.motion_filter import packet_moments  # noqa:E402


def source_records(receipt):
    b.require(
        receipt.get("config", {}).get("magnitude_units") == "log_magnitude_db"
        and receipt.get("config", {}).get("phase_units") == "radians",
        "Explicit typed RF units required",
    )
    rows = [r for r in receipt["records"] if r["id"].split("/")[0] in ["E01", "E02"]]
    expected = {
        f"E{room:02}/S{person:02}/A{action:02}"
        for room, first in [(1, 1), (2, 11)]
        for person in range(first, first + 10)
        for action in range(1, 28)
    }
    b.require(
        len(rows) == 540
        and {r["id"] for r in rows} == expected
        and all(r["frames"] == 297 for r in rows),
        "Exactsource20recordsrequired",
    )
    ranges = sorted((r["offset"], r["offset"] + 297) for r in rows)
    b.require(
        all(type(r["offset"]) is int and r["offset"] >= 0 for r in rows)
        and all(a[1] <= c[0] for a, c in zip(ranges, ranges[1:]))
        and ranges[-1][1] <= receipt["frames"],
        "Source RF ranges overlap or exceed the typed cache",
    )
    return sorted(rows, key=lambda r: r["id"])


def build(raw, out, plan):
    RAW, OUT = Path(raw), Path(out)
    b.require(not OUT.exists() and not OUT.is_symlink(), "Existing cache preserved")
    receipt = read(RAW / "receipt.json")
    rows = source_records(receipt)
    b.require(
        b.digest(RAW / "rf.npy") == receipt["arrays"]["rf"], "OriginalRFbyteschanged"
    )
    rf = np.load(RAW / "rf.npy", mmap_mode="r", allow_pickle=False)
    b.require(
        rf.dtype == np.float32 and rf.shape == (receipt["frames"], 6, 114, 10),
        "TypedRFschemadiffers",
    )
    paths = [
        Path(__file__),
        ROOT / "scripts/build_moment_cache.py",
        ROOT / "ghostmesh/motion_filter.py",
        ROOT / "ghostmesh/rf_features.py",
        ROOT / "ghostmesh/covariance_dynamics.py",
    ]
    registration = dict(
        kind="E01E02-label-free-perframe-moments",
        plan_sha256=b.digest(plan),
        raw_receipt_sha256=b.digest(RAW / "receipt.json"),
        raw_rf_sha256=receipt["arrays"]["rf"],
        rooms=["E01", "E02"],
        frame_count=160380,
        annotation_arrays_decoded=False,
        source_sha256={str(p.relative_to(ROOT)): b.digest(p) for p in paths},
        context_frames=32,
        output_delay_frames=4,
        clock="nominal10Hz,400msoutputbuffer,notcalibratedphysicalclock",
        endpoint_grid="target39..292 inclusive dense, common44RFguard e-39..e+4; TRAIN/VALroles assigned by downstreamloaderbeforelabels",
    )
    OUT.mkdir(parents=True)
    b.save(OUT / "registration.json", registration)
    for p in paths:
        dest = OUT / "executed-source" / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    started = time.monotonic()
    moments = np.lib.format.open_memmap(
        OUT / "moments.npy", mode="w+", dtype=np.float32, shape=(160380, 12, 114)
    )
    meta = {
        k: []
        for k in [
            "room",
            "person",
            "action",
            "record",
            "end",
            "raw_index",
            "cache_center",
        ]
    }
    quality = []
    records = []
    cursor = 0
    for r in rows:
        start = r["offset"]
        x = np.asarray(rf[start : start + 297])
        _, rejected = b.retained_endpoints(x)
        valid = np.ones(297, bool)
        valid[rejected] = False
        ends = np.array(
            [e for e in range(39, 293) if valid[e - 39 : e + 5].all()], dtype=np.int64
        )
        moments[cursor : cursor + 297] = packet_moments(x)
        room, person, action = r["id"].split("/")
        for key, value in [
            ("room", room),
            ("person", person),
            ("action", action),
            ("record", r["id"]),
        ]:
            meta[key].extend([value] * len(ends))
        meta["end"].extend(ends.tolist())
        meta["raw_index"].extend((start + ends).tolist())
        meta["cache_center"].extend((cursor + ends).tolist())
        quality.append(
            dict(
                record=r["id"],
                rejected_frames=rejected.tolist(),
                dense_endpoints=ends.tolist(),
            )
        )
        records.append(
            dict(record=r["id"], raw_start=start, cache_start=cursor, frames=297)
        )
        cursor += 297
        if cursor % (297 * 90) == 0:
            print(
                json.dumps(
                    dict(records=cursor // 297, seconds=time.monotonic() - started)
                ),
                flush=True,
            )
    b.require(cursor == 160380, "Allocationdiffers")
    moments.flush()
    del moments
    np.savez_compressed(
        OUT / "metadata.npz", **{k: np.asarray(v) for k, v in meta.items()}
    )
    b.save(OUT / "quality.json", quality)
    b.save(OUT / "records.json", records)
    bind(ROOT, registration["source_sha256"])
    result = dict(
        kind=registration["kind"],
        registration_sha256=b.digest(OUT / "registration.json"),
        frame_rows=cursor,
        logical_dense_endpoints=len(meta["end"]),
        annotation_arrays_decoded=False,
        arrays_sha256={
            name: b.digest(OUT / name)
            for name in ["moments.npy", "metadata.npz", "quality.json", "records.json"]
        },
        seconds=time.monotonic() - started,
    )
    b.save(OUT / "receipt.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["raw", "out", "plan"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    build(args.raw, args.out, args.plan)
