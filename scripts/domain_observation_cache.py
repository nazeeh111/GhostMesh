"""Build frame-causal observation summaries for source30 only; no E04 arrays."""

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


def build(source, output):
    if output.exists():
        raise FileExistsError(output)
    receipt = json.loads((source / "receipt.json").read_text())
    config = receipt["config"]
    if (
        config["environments"] != ["E01", "E02", "E03"]
        or receipt["frames"] != 240570
        or len(receipt["records"]) != 810
    ):
        raise ValueError("Expected source-only E01-E03/full30 cache")
    for name, expected in receipt["arrays"].items():
        if digest(source / f"{name}.npy") != expected:
            raise ValueError("Frozen raw source bytes changed")
    raw = np.load(source / "rf.npy", allow_pickle=False, mmap_mode="r")
    output.mkdir(parents=True)
    result = np.lib.format.open_memmap(
        output / "frame.npy",
        mode="w+",
        dtype="float32",
        shape=(receipt["frames"], 12, 114),
    )
    for index, row in enumerate(receipt["records"]):
        if row["id"][:3] not in {"E01", "E02", "E03"} or row["offset"] != index * 297:
            raise ValueError("Unexpected record identity or offset")
        start, length = row["offset"], row["frames"]
        result[start : start + length] = observation_features(
            raw[start : start + length],
            magnitude_units=config["magnitude_units"],
            phase_units=config["phase_units"],
        )
        if (index + 1) % 90 == 0:
            print(json.dumps({"records": index + 1, "total": 810}), flush=True)
    result.flush()
    del result
    report = {
        "schema_version": 1,
        "scope": "Only E01-E03; frame-local statistics, no E04",
        "frames": receipt["frames"],
        "records": receipt["records"],
        "arrays": {"frame": digest(output / "frame.npy")},
        "base_receipt_sha256": digest(source / "receipt.json"),
        "base_arrays_sha256": receipt["arrays"],
        "feature_channels": [
            "magnitude_mean_rx0-2",
            "magnitude_sd_rx0-2",
            "cos_phase_pairs01_02_12",
            "sin_phase_pairs01_02_12",
        ],
        "source_sha256": {
            str(p.relative_to(ROOT)): digest(p)
            for p in [Path(__file__), ROOT / "ghostmesh/observation.py"]
        },
    }
    (output / "receipt.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.out)
