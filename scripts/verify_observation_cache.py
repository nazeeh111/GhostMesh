"""Independently recompute fixed frames from every source recording, never E04."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.observation import observation_features  # noqa:E402
from scripts.domain_raw_cache import digest  # noqa:E402


def verify(source, cache, output):
    if output.exists():
        raise FileExistsError(output)
    base = json.loads((source / "receipt.json").read_text())
    receipt = json.loads((cache / "receipt.json").read_text())
    if digest(source / "receipt.json") != receipt["base_receipt_sha256"]:
        raise ValueError("Base receipt mismatch")
    if base["records"] != receipt["records"] or len(base["records"]) != 810:
        raise ValueError("Source identity mismatch")
    if any(row["id"][:3] not in {"E01", "E02", "E03"} for row in base["records"]):
        raise ValueError("Only E01-E03 source data permitted")
    if digest(cache / "frame.npy") != receipt["arrays"]["frame"]:
        raise ValueError("Observation bytes changed")
    for name, expected in receipt["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Feature-source bytes changed")
    for name, expected in receipt["base_arrays_sha256"].items():
        if (
            digest(source / f"{name}.npy") != expected
            or base["arrays"][name] != expected
        ):
            raise ValueError("Base array bytes changed")
    raw = np.load(source / "rf.npy", allow_pickle=False, mmap_mode="r")
    frame = np.load(cache / "frame.npy", allow_pickle=False, mmap_mode="r")
    worst, exact, count = 0.0, 0, 0
    for row in base["records"]:
        for relative in (0, 148, 296):
            index = row["offset"] + relative
            recalculated = observation_features(
                raw[index : index + 1],
                magnitude_units="log_magnitude_db",
                phase_units="radians",
            )[0]
            error = float(np.abs(frame[index] - recalculated).max())
            if error > 1e-7:
                raise ValueError(
                    f"Feature comparison failed: {row['id']} frame{relative + 1}"
                )
            worst = max(worst, error)
            exact += int(np.array_equal(frame[index], recalculated))
            count += 1
    report = {
        "schema_version": 1,
        "scope": "Three fixed frames in each of810source recordings, no E04",
        "record_count": 810,
        "frames_checked": count,
        "bitexact_frames": exact,
        "maximum_absolute_difference": worst,
        "frame_indices_one_based": [1, 149, 297],
        "source_receipt_sha256": digest(source / "receipt.json"),
        "observation_receipt_sha256": digest(cache / "receipt.json"),
        "frame_array_sha256": receipt["arrays"]["frame"],
        "verification_script_sha256": digest(Path(__file__)),
    }
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verify(args.source, args.cache, args.out)
