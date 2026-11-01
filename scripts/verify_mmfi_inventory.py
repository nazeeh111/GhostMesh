"""Verify downloaded MM-Fi bytes against official ZIP inventories, without decoding.

Reads only file bytes. This is safe to run on reserved recordings before choosing
models: no MAT/NPY arrays, labels, or model predictions are inspected.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import time
import zlib


def verify_entry(root, entry):
    name = PurePosixPath(entry["name"])
    if name.is_absolute() or ".." in name.parts or len(name.parts) != 5:
        raise ValueError(f"Unsafe or unexpected inventory path: {name}")
    if name.parts[3] not in {"wifi-csi", "rgb"}:
        raise ValueError(f"Unexpected modality: {name}")
    root = Path(root).resolve()
    path = root.joinpath(*name.parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Path escapes dataset: {name}")
    value = path.read_bytes()
    if len(value) != entry["size"]:
        raise ValueError(f"Size mismatch: {name}")
    if zlib.crc32(value) != entry["crc"]:
        raise ValueError(f"CRC mismatch: {name}")
    return sha256(value).digest()


def verify(root, inventories, reserved, output):
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    record_hashes, record_files = {}, {}
    environments = []
    for inventory in sorted(inventories):
        entries = json.loads(inventory.read_text())
        seen = set()
        for entry in sorted(entries, key=lambda item: item["name"]):
            name = PurePosixPath(entry["name"])
            if str(name) in seen:
                raise ValueError(f"Duplicate inventory path: {name}")
            seen.add(str(name))
            digest = verify_entry(root, entry)
            key = "/".join(name.parts[:3])
            record_hashes.setdefault(key, sha256()).update(
                "/".join(name.parts[3:]).encode() + b"\0" + digest
            )
            record_files[key] = record_files.get(key, 0) + 1
        environments.append(
            {
                "inventory": inventory.name,
                "inventory_sha256": sha256(inventory.read_bytes()).hexdigest(),
                "verified_files": len(seen),
            }
        )
        print(json.dumps(environments[-1]), flush=True)
    digests = {key: value.hexdigest() for key, value in record_hashes.items()}
    frozen = json.loads(reserved.read_text())
    for row in frozen["reserved_records"]:
        key = row["id"]
        if digests.get(key) != row["ordered_path_and_file_digest_sha256"]:
            raise ValueError(f"Reserved bytes changed: {key}")
        if record_files[key] != row["files"]:
            raise ValueError(f"Reserved file count changed: {key}")
    expected = {
        f"E{room:02d}/S{person:02d}/A{action:02d}"
        for room in range(1, 5)
        for person in range(10 * (room - 1) + 1, 10 * room + 1)
        for action in range(1, 28)
    }
    if set(digests) != expected or set(record_files.values()) != {594}:
        raise ValueError("Incomplete full40 cohort or unexpected recording size")
    result = {
        "schema_version": 1,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Byte lengths, ZIP CRC32, ordered SHA256 only; no arrays decoded",
        "records": len(digests),
        "paired_frames": sum(record_files.values()) // 2,
        "verified_files": sum(record_files.values()),
        "reserved_records_unchanged": len(frozen["reserved_records"]),
        "reserved_config_sha256": sha256(reserved.read_bytes()).hexdigest(),
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "environments": environments,
        "ordered_record_sha256": digests,
        "elapsed_seconds": time.monotonic() - started,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inventories", type=Path, required=True)
    parser.add_argument("--reserved", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(
        args.root,
        args.inventories.glob("*-inventory.json"),
        args.reserved,
        args.output,
    )
