"""Bounded public ZIP64 range extraction. No credentials or third-party code.
Downloads only ZIP directory metadata and explicitly selected modalities.
"""

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import pathlib
import re
import struct
import tempfile
import time
import urllib.error
import urllib.request
import zlib

FILES = {
    "E01": ("1ExV3AQeHstQ3Z1VBFOC0z1BDtTD5nZ1B", 22301071119),
    "E02": ("1oIPGmsjDlzQsnTDVzIhYRQq-3BHTxQ8o", 20886295287),
    "E03": ("1WjfPToIpi1a0cRYBvIr2yZoq_2jQpPQq", 18864127686),
    "E04": ("1-XTwxO0ymJ1AtI5HsOOjD-XTrIHKPaA1", 20953639005),
}
CACHE = (
    pathlib.Path("/private/tmp")
    if pathlib.Path("/private/tmp").is_dir()
    else pathlib.Path(tempfile.gettempdir())
) / "ghostmesh-mmfi-ranges"
MAX_RANGE_BYTES = 96 * 1024 * 1024
MAX_FILE_BYTES = 16 * 1024 * 1024


def get(env, start, end):
    ident, size = FILES[env]
    if start < 0 or end < start or end >= size or end - start + 1 > MAX_RANGE_BYTES:
        raise ValueError("range budget")
    url = f"https://drive.usercontent.google.com/download?id={ident}&export=download&confirm=t"
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=90) as r:
                expected = f"bytes {start}-{end}/{size}"
                if r.status != 206 or r.headers.get("Content-Range") != expected:
                    raise ValueError(
                        f"range unsupported: requested {expected}; received status={r.status}, Content-Range={r.headers.get('Content-Range')!r}, Content-Type={r.headers.get('Content-Type')!r}"
                    )
                data = r.read(end - start + 2)
                if len(data) != end - start + 1:
                    raise ValueError("wrong range length")
                return data
        except urllib.error.HTTPError as exc:
            # An access denial is terminal, not a reason to try another route.
            if exc.code in (401, 403, 404) or attempt == 3:
                raise
            retry_after = exc.headers.get("Retry-After", "")
            delay = float(retry_after) if retry_after.isdigit() else 1 + attempt
            if delay > 60:
                raise RuntimeError(
                    "Server requested a retry delay above60seconds; retry later."
                ) from exc
            time.sleep(max(1, delay))
        except (TimeoutError, urllib.error.URLError, OSError):
            if attempt == 3:
                raise
            time.sleep(1 + attempt)


def directory(env):
    path = CACHE / (env + "-directory.bin")
    if path.exists():
        raw = path.read_bytes()
    else:
        size = FILES[env][1]
        tail = get(env, size - 65536, size - 1)
        i = tail.rfind(b"PK\x05\x06")
        if i < 0 or i + 22 > len(tail):
            raise ValueError("missing ZIP end directory")
        eocd = struct.unpack("<4s4H2LH", tail[i : i + 22])
        count, cdsize, offset = eocd[4:7]
        if offset == 0xFFFFFFFF or count == 65535:
            j = tail.rfind(b"PK\x06\x07")
            if j < 0 or j + 20 > len(tail):
                raise ValueError("missing ZIP64 locator")
            loc = struct.unpack("<4sLQL", tail[j : j + 20])[2]
            z = (
                tail[loc - (size - 65536) :]
                if size - 65536 <= loc <= size - 56
                else get(env, loc, loc + 55)
            )
            if len(z) < 56 or z[:4] != b"PK\x06\x06":
                raise ValueError("missing ZIP64 directory")
            _, _, _, _, _, _, _, count, cdsize, offset = struct.unpack(
                "<4sQ2H2L4Q", z[:56]
            )
        print(env, "central_directory_bytes", cdsize, "entries", count, flush=True)
        raw = get(env, offset, offset + cdsize - 1)
        path.write_bytes(raw)
    return parse_directory(raw)


def parse_directory(raw):
    entries = []
    p = 0
    while p < len(raw):
        if p + 46 > len(raw):
            raise ValueError("truncated central header")
        v = struct.unpack("<4s6H3L5H2L", raw[p : p + 46])
        if v[0] != b"PK\x01\x02":
            raise ValueError("central signature")
        (
            _,
            made,
            need,
            flags,
            method,
            mt,
            md,
            crc,
            compressed,
            size,
            nl,
            el,
            cl,
            disk,
            internal,
            external,
            off,
        ) = v
        if p + 46 + nl + el + cl > len(raw):
            raise ValueError("truncated central entry")
        if flags & 1:
            raise ValueError("encrypted ZIP unsupported")
        name = raw[p + 46 : p + 46 + nl].decode("utf8")
        extra = raw[p + 46 + nl : p + 46 + nl + el]
        q = 0
        while q + 4 <= len(extra):
            tag, n = struct.unpack("<HH", extra[q : q + 4])
            x = extra[q + 4 : q + 4 + n]
            q += 4 + n
            if len(x) != n:
                raise ValueError("truncated ZIP extra field")
            if tag == 1:
                k = 0
                if size == 0xFFFFFFFF:
                    size = struct.unpack("<Q", x[k : k + 8])[0]
                    k += 8
                if compressed == 0xFFFFFFFF:
                    compressed = struct.unpack("<Q", x[k : k + 8])[0]
                    k += 8
                if off == 0xFFFFFFFF:
                    off = struct.unpack("<Q", x[k : k + 8])[0]
        entries.append(
            dict(
                name=name,
                size=size,
                compressed=compressed,
                offset=off,
                method=method,
                crc=crc,
            )
        )
        p += 46 + nl + el + cl
    return entries


def extract_group(env, group, out, resume=False):
    if not group:
        raise ValueError("empty extraction group")
    for e in group:
        if e["size"] > MAX_FILE_BYTES or e["compressed"] > MAX_FILE_BYTES:
            raise ValueError("archive member exceeds file budget")
    if resume:
        records = []
        for e in group:
            relative = pathlib.PurePosixPath(e["name"])
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or relative.parts[0] != env
            ):
                raise ValueError("unsafe archive path")
            target = out / e["name"]
            if not target.resolve().is_relative_to(out.resolve()):
                raise ValueError("archive target escapes output through symlink")
            if not target.is_file():
                break
            if target.stat().st_size != e["size"]:
                raise ValueError("existing member length mismatch: " + str(target))
            data = target.read_bytes()
            if zlib.crc32(data) & 0xFFFFFFFF != e["crc"]:
                raise ValueError("existing member CRC mismatch: " + str(target))
            records.append(
                {
                    "path": e["name"],
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
        else:
            return records, 0
    start = group[0]["offset"]
    end = max(
        e["offset"] + 30 + len(e["name"].encode()) + 4096 + e["compressed"]
        for e in group
    )
    end = min(end, FILES[env][1] - 1)
    blob = get(env, start, end)
    records = []
    for e in group:
        p = e["offset"] - start
        v = struct.unpack("<4s5H3L2H", blob[p : p + 30])
        nl, el = v[-2:]
        if v[0] != b"PK\x03\x04":
            raise ValueError("local header signature")
        if v[2] & 1 or v[3] != e["method"]:
            raise ValueError("local/central ZIP header mismatch")
        local_name = blob[p + 30 : p + 30 + nl].decode("utf8")
        if local_name != e["name"]:
            raise ValueError("local/central ZIP filename mismatch")
        data = blob[p + 30 + nl + el : p + 30 + nl + el + e["compressed"]]
        if len(data) != e["compressed"]:
            raise ValueError("truncated ZIP payload")
        if e["method"] not in (0, 8):
            raise ValueError("unsupported ZIP compression")
        if e["method"] == 8:
            decoder = zlib.decompressobj(-15)
            data = decoder.decompress(data, e["size"] + 1)
            if not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:
                raise ValueError("invalid or oversized deflate payload")
        if len(data) != e["size"] or zlib.crc32(data) & 0xFFFFFFFF != e["crc"]:
            raise ValueError("checksum mismatch")
        relative = pathlib.PurePosixPath(e["name"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != env:
            raise ValueError("unsafe archive path")
        target = out / e["name"]
        if not target.resolve().is_relative_to(out.resolve()):
            raise ValueError("archive target escapes output through symlink")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != data:
            raise ValueError(
                "refusing to overwrite different existing bytes: " + str(target)
            )
        if not target.exists():
            target.write_bytes(data)
        records.append(
            {
                "path": e["name"],
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return records, len(blob)


def main():
    global CACHE
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=FILES)
    p.add_argument("--subjects", nargs="+")
    p.add_argument("--actions", nargs="+", default=["A02", "A06"])
    p.add_argument("--out", type=pathlib.Path)
    p.add_argument("--inventory", action="store_true")
    p.add_argument("--cache", type=pathlib.Path, default=CACHE)
    p.add_argument("--inventory-dir", type=pathlib.Path)
    p.add_argument("--max-payload-mb", type=float, default=2048)
    p.add_argument(
        "--resume",
        action="store_true",
        help="After access is restored, verify local CRC/length and skip complete groups; does not retry access denials",
    )
    args = p.parse_args()
    CACHE = args.cache
    CACHE.mkdir(parents=True, exist_ok=True)
    if any(not re.fullmatch(r"A(0[1-9]|1[0-9]|2[0-7])", a) for a in args.actions):
        raise ValueError("actions must be A01 through A27")
    if args.subjects and any(
        not re.fullmatch(r"S(0[1-9]|[1-3][0-9]|40)", a) for a in args.subjects
    ):
        raise ValueError("subjects must be S01 through S40")
    entries = directory(args.env)
    selected = [
        e
        for e in entries
        if re.search(r"/(wifi-csi|rgb)/frame\d+\.(mat|npy)$", e["name"])
        and (not args.subjects or e["name"].split("/")[1] in args.subjects)
        and e["name"].split("/")[2] in args.actions
    ]
    print(
        json.dumps(
            {
                "selected_files": len(selected),
                "compressed_bytes": sum(e["compressed"] for e in selected),
                "sample": selected[:3],
            },
            indent=2,
        ),
        flush=True,
    )
    selection_id = (
        "-".join(sorted(args.actions))
        + "-"
        + ("-".join(sorted(args.subjects)) if args.subjects else "all-subjects")
    )
    inventory_dir = args.inventory_dir or CACHE
    inventory_dir.mkdir(parents=True, exist_ok=True)
    inv = inventory_dir / ("data-" + args.env + "-" + selection_id + "-inventory.json")
    inv.write_text(json.dumps(selected, indent=2))
    if args.inventory:
        return
    if not args.out:
        raise ValueError("--out required")
    if not selected:
        raise ValueError("selection has no matching files")
    if sum(e["compressed"] for e in selected) > args.max_payload_mb * 1024 * 1024:
        raise ValueError(
            "selected compressed payload exceeds --max-payload-mb; inspect inventory before increasing"
        )
    selected.sort(key=lambda e: e["offset"])
    groups = []
    for e in selected:
        if (
            not groups
            or e["offset"] - (groups[-1][-1]["offset"] + groups[-1][-1]["compressed"])
            > 65536
            or e["offset"] - groups[-1][0]["offset"] > 16 * 1024 * 1024
        ):
            groups.append([e])
        else:
            groups[-1].append(e)
    print("coalesced_requests", len(groups), flush=True)
    records = []
    total = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for rec, n in pool.map(
            lambda g: extract_group(args.env, g, args.out, resume=args.resume), groups
        ):
            records.extend(rec)
            total += n
            print("extracted", len(records), "download_bytes", total, flush=True)
    provenance = {
        "source": f"https://drive.google.com/file/d/{FILES[args.env][0]}/view",
        "archive_bytes": FILES[args.env][1],
        "license": "CC-BY-NC-4.0",
        "attribution": "Jianfei Yang et al., MM-Fi, NeurIPS 2023",
        "selection": {
            "subjects": args.subjects,
            "actions": args.actions,
            "modalities": ["wifi-csi", "rgb 2D keypoints"],
        },
        "http_download_bytes": total,
        "retrieved_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "toolbox_reference_commit": "c9c1e3a8d5099779a618eae1bb351e253daa52d0",
        "directory_sha256": hashlib.sha256(
            (CACHE / (args.env + "-directory.bin")).read_bytes()
        ).hexdigest(),
        "files": records,
    }
    (args.out / (args.env + "-" + selection_id + "-provenance.json")).write_text(
        json.dumps(provenance, indent=2)
    )


if __name__ == "__main__":
    main()
