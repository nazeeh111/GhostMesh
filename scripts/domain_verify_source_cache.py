"""Verify source30 cache and exact agreement with independently prepared v1 data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.preprocess import prepare_window  # noqa: E402
from ghostmesh.rf_features import quotient_features  # noqa: E402
from scripts.domain_raw_cache import digest  # noqa: E402


def verify(cache, old_manifest, reference, stream, quotient, output):
    if output.exists():
        raise FileExistsError(output)
    receipt = json.loads((cache / "receipt.json").read_text())
    arrays = {}
    for name, expected in receipt["arrays"].items():
        if digest(cache / f"{name}.npy") != expected:
            raise ValueError(f"Array digest mismatch: {name}")
        arrays[name] = np.load(cache / f"{name}.npy", allow_pickle=False, mmap_mode="r")
    if digest(ROOT / "scripts/domain_full_source_cache.py") != receipt["script_sha256"]:
        raise ValueError("Builder source drift")
    ids = np.load(reference / "sequence.npy", allow_pickle=False)
    ends = np.load(reference / "end.npy", allow_pickle=False)
    lookup = {
        (str(record), int(end)): i for i, (record, end) in enumerate(zip(ids, ends))
    }
    legacy = np.load(stream / "legacy.npy", allow_pickle=False, mmap_mode="r")
    quotients = np.load(quotient / "quotient.npy", allow_pickle=False, mmap_mode="r")
    offsets = {row["id"]: row["offset"] for row in receipt["records"]}
    if len(offsets) != 810 or any(
        key[:3] not in {"E01", "E02", "E03"} for key in offsets
    ):
        raise ValueError("Unexpected source identities")
    manifest = json.loads(old_manifest.read_text())
    checked_records = checked_windows = 0
    for row in manifest["sequences"]:
        if row["environment"] == "E04":
            continue
        if row["environment"] not in {"E01", "E02", "E03"}:
            raise ValueError("Unexpected development room")
        path = old_manifest.parent / row["path"]
        if digest(path) != row["sha256"]:
            raise ValueError("Reference source bytes changed")
        start = offsets[row["id"]]
        sl = slice(start, start + row["frames"])
        with np.load(path, allow_pickle=False) as original:
            for actual, old in [
                (arrays["rf"][sl, :3], original["csi"]),
                (arrays["pose"][sl], original["pose"]),
                (arrays["valid"][sl], original["valid"]),
            ]:
                np.testing.assert_array_equal(actual, old)
        for end in (7, 143, 295):
            index = lookup[row["id"], end]
            raw = arrays["rf"][start + end - 7 : start + end + 1]
            normalized = np.stack([prepare_window(x[None, :3], True)[0] for x in raw])
            np.testing.assert_array_equal(normalized, legacy[index])
            np.testing.assert_array_equal(
                quotient_features(raw[:, :3], raw[:, 3:], units="log_magnitude_db"),
                quotients[index],
            )
            checked_windows += 1
        checked_records += 1
    if checked_records != 405 or checked_windows != 1215:
        raise ValueError("Incomplete independent-reference comparison")
    result = {
        "schema_version": 1,
        "source_receipt_sha256": digest(cache / "receipt.json"),
        "old_manifest_sha256": digest(old_manifest),
        "script_sha256": digest(Path(__file__)),
        "full_cache_array_hashes_verified": receipt["arrays"],
        "source_records": 810,
        "source_frames": receipt["frames"],
        "old_source_records_exact_amplitude_pose_validity": checked_records,
        "sampled_windows_exact_legacy_and_quotient": checked_windows,
        "no_E04_arrays_decoded": True,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "old-manifest", "reference", "stream", "quotient", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    verify(
        args.cache,
        args.old_manifest,
        args.reference,
        args.stream,
        args.quotient,
        args.output,
    )
