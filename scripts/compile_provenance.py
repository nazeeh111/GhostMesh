"""Compile compact source fingerprints from an importer's raw-hash receipts.

This does not reread original arrays. The prepared manifest records SHA-256
computed during data import; this command binds its receipt compilation to the
exact manifest bytes and states that evidence boundary explicitly.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def digest(data):
    return hashlib.sha256(data).hexdigest()


def compile_provenance(manifest_path, output, inspection_path=None):
    manifest_path, output = Path(manifest_path), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to replace provenance: {output}")
    content = manifest_path.read_bytes()
    manifest = json.loads(content)
    if manifest.get("schema_version") != 1 or not manifest.get("sequences"):
        raise ValueError("Expected a nonempty schema_version 1 prepared manifest")
    global_hash = hashlib.sha256()
    records = []
    identifiers = set()
    for row in sorted(manifest["sequences"], key=lambda item: item["id"]):
        identifier = row["id"]
        if (
            not re.fullmatch(r"E\d{2}/S\d{2}/A\d{2}", identifier)
            or identifier in identifiers
        ):
            raise ValueError("Expected unique MM-Fi E##/S##/A## identifiers")
        identifiers.add(identifier)
        hashes = row.get("raw_frame_hashes", [])
        frames = [item["frame"] for item in hashes]
        if (
            not hashes
            or frames != sorted(set(frames))
            or len(hashes) != row.get("frames")
        ):
            raise ValueError(
                f"Missing, duplicate or unsorted raw hash receipts: {identifier}"
            )
        sequence_hash = hashlib.sha256()
        for item in hashes:
            if not isinstance(item["frame"], int) or item["frame"] < 1:
                raise ValueError("Frame IDs must be positive integers")
            for modality, extension, key in (
                ("wifi-csi", "mat", "csi_sha256"),
                ("rgb", "npy", "pose_sha256"),
            ):
                sha = item[key]
                if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
                    raise ValueError("Raw receipt requires a lowercase SHA-256")
                record = f"{identifier}/{modality}/frame{item['frame']:03d}.{extension}\t{sha}\n".encode()
                sequence_hash.update(record)
                global_hash.update(record)
        records.append(
            {
                "id": identifier,
                "frames": len(hashes),
                "split": row["split"],
                "prepared_npz_sha256": row["sha256"],
                "raw_source_paths_and_sha256_digest": sequence_hash.hexdigest(),
            }
        )
    result = {
        "schema_version": 1,
        "dataset": manifest["provenance"],
        "protocol": manifest["protocol"],
        "manifest_sha256": digest(content),
        "evidence_basis": "Compiled from raw_frame_hashes computed by the importer while reading original source files. No second source-byte or prepared-NPZ verification pass is claimed here. Training/audit separately verifies prepared file hashes.",
        "hash_convention": "Sort E/S/A sequence IDs lexically; within each sequence sort numeric frame IDs. For each frame append CSI then RGB records: UTF-8 relative_path<TAB>sha256<LF>. Hash the concatenated records using SHA-256.",
        "sequences": len(records),
        "frames": sum(row["frames"] for row in records),
        "subjects": sorted({row["id"].split("/")[1] for row in records}),
        "environments": sorted({row["id"].split("/")[0] for row in records}),
        "actions": sorted({row["id"].split("/")[2] for row in records}),
        "sequence_counts_by_split": dict(Counter(row["split"] for row in records)),
        "raw_source_paths_and_sha256_digest": global_hash.hexdigest(),
        "sequence_records": records,
    }
    if inspection_path:
        inspected_content = Path(inspection_path).read_bytes()
        inspection = json.loads(inspected_content)
        if set(item["id"] for item in inspection["sequence_records"]) != identifiers:
            raise ValueError("Label inspection covers a different sequence cohort")
        result["label_inspection_sha256"] = digest(inspected_content)
        result["label_observations"] = {
            key: value for key, value in inspection.items() if key != "sequence_records"
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label-inspection")
    args = parser.parse_args()
    result = compile_provenance(args.manifest, args.out, args.label_inspection)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "sequences",
                    "frames",
                    "manifest_sha256",
                    "raw_source_paths_and_sha256_digest",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
