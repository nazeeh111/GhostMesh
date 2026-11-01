"""Build an exactly allocated, label-free CSI moment cache from typed MM-Fi RF.

No encoder, archived source-ready identity, 2D/3D annotation decoding or images
are needed. The RF-only forty-frame quality guard matches the preceding source
cohort. A new receipt binds actual input/output bytes for a fresh experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.covariance_dynamics import CONDITION_LIMIT, receiver_covariances  # noqa:E402
from ghostmesh.motion_filter import packet_moments  # noqa:E402


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1048576), b""):
            h.update(part)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def retained_endpoints(raw):
    """Reject endpoints if any of the preceding forty RF frames is unacceptable."""
    raw = np.asarray(raw)
    require(raw.shape == (297, 6, 114, 10), "Complete297frame typed RF record required")
    complete = (np.isfinite(raw[:, :3]) & np.isfinite(raw[:, 3:])).all(1)
    valid = (complete.reshape(297, 19, 60).sum(2) >= 57).all(1)
    if valid.any():
        ids = np.flatnonzero(valid)
        covariance = receiver_covariances(
            raw[valid], magnitude_units="log_magnitude_db", phase_units="radians"
        )
        eigen = np.linalg.eigvalsh(covariance)
        ratio = np.full(eigen.shape[:2], np.inf)
        np.divide(eigen[..., -1], eigen[..., 0], out=ratio, where=eigen[..., 0] > 0)
        accepted = (eigen[..., 0] > 0).all(1) & (ratio <= CONDITION_LIMIT).all(1)
        valid[ids[~accepted]] = False
    endpoints = np.asarray(
        [e for e in range(39, 297, 8) if valid[e - 39 : e + 1].all()], dtype=np.int64
    )
    return endpoints, np.flatnonzero(~valid)


def source_records(receipt):
    require(
        receipt["config"]["magnitude_units"] == "log_magnitude_db"
        and receipt["config"]["phase_units"] == "radians",
        "Explicit typed RF units required",
    )
    expected = {
        f"{room}/S{person:02}/A{action:02}"
        for room, start in (("E01", 1), ("E03", 21))
        for person in range(start, start + 10)
        for action in range(1, 28)
    }
    rows = [r for r in receipt["records"] if r["id"].split("/")[0] in ("E01", "E03")]
    require(
        len(rows) == len(expected) and {r["id"] for r in rows} == expected,
        "Complete540E01/E03 records required",
    )
    require(
        all(
            type(r["offset"]) is int and r["offset"] >= 0 and r["frames"] == 297
            for r in rows
        ),
        "Raw frame offsets differ",
    )
    ranges = sorted((r["offset"], r["offset"] + 297) for r in rows)
    require(
        all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:])),
        "Raw record frame ranges overlap",
    )
    return sorted(rows, key=lambda r: r["id"])


def build(raw_dir, out):
    require(not out.exists() and not out.is_symlink(), "Existing cache preserved")
    require(
        not out.resolve().is_relative_to(raw_dir.resolve()),
        "Output must be outside raw inputs",
    )
    receipt = json.loads((raw_dir / "receipt.json").read_text())
    records = source_records(receipt)
    require(
        digest(raw_dir / "rf.npy") == receipt["arrays"]["rf"], "Raw RF hash differs"
    )
    rf = np.load(raw_dir / "rf.npy", mmap_mode="r", allow_pickle=False)
    require(
        rf.dtype == np.float32 and rf.shape == (receipt["frames"], 6, 114, 10),
        "Typed RF array schema differs",
    )
    require(
        all(r["offset"] + 297 <= len(rf) for r in records), "Raw record exceeds array"
    )
    paths = [
        Path(__file__),
        ROOT / "ghostmesh/motion_filter.py",
        ROOT / "ghostmesh/rf_features.py",
        ROOT / "ghostmesh/covariance_dynamics.py",
    ]
    registration = dict(
        kind="label-free-causal-moment-cache",
        rooms=["E01", "E03"],
        raw_receipt_sha256=digest(raw_dir / "receipt.json"),
        raw_rf_sha256=receipt["arrays"]["rf"],
        context=8,
        endpoints=list(range(39, 297, 8)),
        quality_guard="forty consecutive frames; at least57/60 complete3receiver vectors in each6subcarrier bin; SPDcovariancecondition<=1e10; no jitter or filled covariance",
        features="Perframe6quotientchannelmean/populationSD over10packetcolumns,12x114; retain114frequency indices, no calibratedpacketclock",
        annotation_arrays_decoded=False,
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in paths},
        numpy=np.__version__,
    )
    out.mkdir(parents=True)
    save(out / "registration.json", registration)
    shutil.copyfile(raw_dir / "receipt.json", out / "raw-receipt.json")
    for path in paths:
        dest = out / "executed-source" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    started = time.monotonic()
    quality = []
    metadata = []
    try:
        for record in records:
            start = record["offset"]
            ends, rejected = retained_endpoints(rf[start : start + 297])
            quality.append(
                dict(
                    record=record["id"],
                    retained_endpoints=ends.tolist(),
                    rejected_frames=rejected.tolist(),
                )
            )
            room, person, action = record["id"].split("/")
            metadata.extend(
                (room, person, action, record["id"], int(e), int(start + e))
                for e in ends
            )
        require(
            len(metadata) >= 0.9 * len(records) * 33,
            "More than10percent RF-quality exclusions",
        )
        require(
            {r[3] for r in metadata} == {r["id"] for r in records},
            "An entire record failed RF quality",
        )
        meta = {
            key: np.array([r[i] for r in metadata])
            for i, key in enumerate(
                ("room", "person", "action", "record", "end", "raw_index")
            )
        }
        np.savez(out / "metadata.npz", **meta)
        save(out / "quality.json", quality)
        values = np.lib.format.open_memmap(
            out / "features.npy",
            mode="w+",
            dtype="float32",
            shape=(len(metadata), 8, 12, 114),
        )
        cursor = 0
        for record, q in zip(records, quality):
            start = record["offset"]
            frames = packet_moments(np.asarray(rf[start : start + 297]))
            for end in q["retained_endpoints"]:
                values[cursor] = frames[end - 7 : end + 1]
                cursor += 1
        require(cursor == len(values), "Feature cursor differs")
        values.flush()
        for name, expected in registration["source_sha256"].items():
            require(digest(ROOT / name) == expected, "Producer source changed")
        require(
            digest(raw_dir / "receipt.json") == registration["raw_receipt_sha256"],
            "Raw receipt changed",
        )
        save(
            out / "receipt.json",
            dict(
                kind=registration["kind"],
                registration_sha256=digest(out / "registration.json"),
                raw_receipt_sha256=registration["raw_receipt_sha256"],
                logical_rows=cursor,
                allocated_rows=cursor,
                records=len(records),
                annotation_arrays_decoded=False,
                seconds=time.monotonic() - started,
                arrays_sha256={
                    name: digest(out / name)
                    for name in ("features.npy", "metadata.npz", "quality.json")
                },
                scope="Label-free E01/E03 development inputs only; no E02/E04/image values. No model trained or accuracy scored.",
            ),
        )
        return json.loads((out / "receipt.json").read_text())
    except BaseException as error:
        save(
            out / "failure.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                completed_quality_records=len(quality),
                seconds=time.monotonic() - started,
            ),
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.raw, args.out)
    print(json.dumps({k: v for k, v in result.items() if k != "arrays_sha256"}))
