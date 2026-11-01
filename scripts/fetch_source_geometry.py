"""Acquire only the 540 E01/E03 numeric annotation members from official MM-Fi.

No images or E02/E04 payloads. Access denials are terminal in fetch_mmfi.get.
Raw bytes and evidence have separate exclusive output directories. Data remain
under MM-Fi's CC BY-NC4.0 terms; this is not a semantic joint/3D pose contract.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import sys
import threading
import time
import zlib

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import fetch_mmfi as provider  # noqa: E402


def source_records(receipt):
    ids = [row["id"] for row in receipt["records"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source receipt record")
    selected = {name for name in ids if name.split("/")[0] in {"E01", "E03"}}
    expected = {
        f"{room}/S{person:02}/A{action:02}"
        for room, first in (("E01", 1), ("E03", 21))
        for person in range(first, first + 10)
        for action in range(1, 28)
    }
    if selected != expected:
        raise ValueError("Complete source E01/E03 membership required")
    return selected


def selection(catalog, room, records):
    if room not in {"E01", "E03"}:
        raise ValueError("Source rooms only")
    chosen = [
        row
        for row in catalog
        if row["name"].startswith(room + "/")
        and row["name"].endswith("/ground_truth.npy")
        and row["name"].removesuffix("/ground_truth.npy") in records
    ]
    expected = {
        name + "/ground_truth.npy" for name in records if name.startswith(room + "/")
    }
    if (
        len(chosen) != 270
        or len({row["name"] for row in chosen}) != len(chosen)
        or {row["name"] for row in chosen} != expected
        or any(row["size"] != 60716 for row in chosen)
    ):
        raise ValueError("Numeric source annotation catalog differs")
    return chosen


def verify_member(root, row):
    name = row["name"]
    if not re.fullmatch(r"E0[13]/S\d{2}/A\d{2}/ground_truth\.npy", name):
        raise ValueError("Unsafe or non-source annotation path")
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Annotation symlink escaped root")
    raw = path.read_bytes()
    if len(raw) != row["size"] or zlib.crc32(raw) & 0xFFFFFFFF != row["crc"]:
        raise ValueError("Annotation CRC/size differs")
    points = np.load(path, allow_pickle=False)
    if (
        points.dtype != np.float32
        or points.shape != (297, 17, 3)
        or not np.isfinite(points).all()
    ):
        raise ValueError("Finite float32 297x17x3 annotation required")
    return dict(path=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def acquire(receipt, out, evidence, cache, *, existing_inventory=None):
    records = source_records(json.loads(receipt.read_text()))
    if evidence.exists() or (out.exists() and existing_inventory is None):
        raise FileExistsError("Existing output preserved")
    evidence.mkdir(parents=True)
    started = time.monotonic()
    completed = []
    downloaded = 0
    stop = threading.Event()
    try:
        if existing_inventory is None:
            out.mkdir(parents=True)
            provider.CACHE = cache
            cache.mkdir(parents=True, exist_ok=True)
            entries = [
                row
                for room in ("E01", "E03")
                for row in selection(provider.directory(room), room, records)
            ]
        else:
            entries = json.loads(existing_inventory.read_text())
            for room in ("E01", "E03"):
                selection(entries, room, records)
            if len(entries) != 540:
                raise ValueError("Existing inventory includes extra members")
        save(evidence / "inventory.json", entries)

        def fetch(row):
            if stop.is_set():
                return 0
            room = row["name"].split("/")[0]
            try:
                return provider.extract_group(room, [row], out, resume=False)[1]
            except BaseException:
                stop.set()
                raise

        if existing_inventory is None:
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(fetch, row) for row in entries]
                try:
                    for result in as_completed(futures):
                        downloaded += result.result()
                except BaseException:
                    stop.set()
                    for pending in futures:
                        pending.cancel()
                    raise
        for row in entries:
            completed.append(verify_member(out, row))
        result = dict(
            status="CRC_verified_source_only_skeletons",
            source="Official MM-Fi E01/E03 archives in scripts/fetch_mmfi.py",
            license="CC-BY-NC-4.0",
            rooms=["E01", "E03"],
            records=sorted(completed, key=lambda row: row["path"]),
            downloaded_member_range_bytes=downloaded,
            existing_bytes_only=existing_inventory is not None,
            seconds=time.monotonic() - started,
            source_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
            scope="Numeric source annotations only. No RGB/E02/E04, semantic joint mapping, physical units or 3D reconstruction validation.",
        )
        save(evidence / "acquisition-receipt.json", result)
        return result
    except BaseException as error:
        save(
            evidence / "failure.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                verified_members=len(completed),
                seconds=time.monotonic() - started,
            ),
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-receipt", "out", "evidence", "catalog-cache"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--verify-existing-inventory", type=Path)
    args = parser.parse_args()
    result = acquire(
        args.source_receipt,
        args.out,
        args.evidence,
        args.catalog_cache,
        existing_inventory=args.verify_existing_inventory,
    )
    print(
        json.dumps(
            dict(
                status=result["status"],
                records=len(result["records"]),
                seconds=result["seconds"],
            )
        )
    )
