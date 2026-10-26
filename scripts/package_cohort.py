#!/usr/bin/env python3
"""Package exact v1 prepared MM-Fi cohort bytes; no data acquisition or ML imports."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from package_study import digest, require_new, write_archive  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "ghostmesh-mmfi-cohort20-v1"
RELEASE = "https://github.com/nazeeh111/GhostMesh/blob/v0.1.0/"
HASHES = {
    "manifest.json": "0a97b3e76b896757e3c281f48e068939f100738ed2338ff1e136afc543ebb768",
    "cross-environment.json": "a02f2ebfd406c16fd46bfc8f45ed61cff7217c1dc5a2979f542b7de8a07fe360",
    "docs/cohort-provenance.json": "d77e05d215899ab6e78e83443f689aca46159144b3d03c4fef9da4affa282736",
}
README = """# GhostMesh prepared MM-Fi cohort20 v1

This exact-byte archive contains 540 prepared sequences, 160380 frames, 20 people,
27 supplied action IDs and four environments. It is not the full 40-person MM-Fi
study. NPZ fields are CSI amplitude, supplied 2D keypoints, validity and nominal
timestamps. There are no RGB images, raw MAT files, 3D arrays or window caches.
Keypoints are supplied model-generated teacher estimates, not manually verified ground truth.
CSI remains [N,3,114,10]; the declared 640-square annotation plane preserves isotropic
geometry but is not recovered camera calibration. Nominal 10 Hz timestamps are
reconstructed from frame IDs, not independently verified packet synchronization.

Credit: Jianfei Yang, He Huang, Yunjiao Zhou, Xinyan Chen, Yuecong Xu, Shenghai Yuan,
Han Zou, Chris Xiaoxuan Lu and Lihua Xie. MM-Fi: Multi-Modal Non-Intrusive 4D Human
Dataset for Versatile Wireless Sensing. NeurIPS 2023 Datasets and Benchmarks.
Source and original license: https://ntu-aiot-lab.github.io/mm-fi
Dataset: https://github.com/ybhbingo/MMFi_dataset
Paper: https://arxiv.org/abs/2305.10345

Prepared data retain Creative Commons Attribution-NonCommercial 4.0 International:
https://creativecommons.org/licenses/by-nc/4.0/
Legal code: https://creativecommons.org/licenses/by-nc/4.0/legalcode.en
Give attribution, retain license/source links, identify modifications and use the
material noncommercially. No additional license restrictions are imposed. No MM-Fi
author endorsement is implied. Original GhostMesh software has the separate MIT
license. Modifications: select the balanced cohort; pair CSI and supplied 2D labels
by frame ID; convert to float32; divide 2D coordinates by 640; reconstruct nominal
10 Hz timing; preserve both registered subject/environment splits. This archive
copies the resulting NPZ/manifests without recomputation or JSON reformatting.

Extract into a NEW folder, preserving any existing project data:

```sh
mkdir ghostmesh-data-extracted
tar -xzf ghostmesh-mmfi-cohort20-v1.tar.gz -C ghostmesh-data-extracted
cd ghostmesh-data-extracted/ghostmesh-mmfi-cohort20-v1
python3 - <<'VERIFY'
import hashlib, json
from pathlib import Path
for name, record in json.loads(Path('bundle_manifest.json').read_text())['members'].items():
    path = Path(name)
    assert not path.is_absolute() and '..' not in path.parts
    assert path.stat().st_size == record['size'], name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], name
print('All listed data bytes verified.')
VERIFY
```

Compare the external archive SHA256 to the release receipt first; internal hashes
establish consistency, not authenticity. The manifest excludes itself to avoid a
circular hash. In a fresh checkout of GhostMesh tag v0.1.0, copy the extracted
`data/mmfi-cohort20` folder only if that destination does not already exist. Do not
overwrite another prepared dataset. Install the locked environment and evaluate
the released validation-selected primary weights independently of provider access:

```sh
uv sync --locked --extra test
uv run ghostmesh evaluate --manifest data/mmfi-cohort20/manifest.json \\
  --checkpoint models/ghostmesh.pt --split test --out results/my-primary-test.json
```

The output path must be new. The weights' embedded manifest/source identity must
match; different source/runtime/hardware can fail identity or change numerical
results. This recomputes predictions and aggregate metrics, rather than merely
re-rendering receipts. For all 12 models, use the corresponding exact checkpoints
from the separate study archive and the matching protocol manifest. This does not
retrain weights, reproduce acquisition, establish unseen-action transfer, or turn
room/person-confounded E04 transfer into a causal motion/localization result.
Full provenance: https://github.com/nazeeh111/GhostMesh/blob/v0.1.0/docs/cohort-provenance.json
Protocol: https://github.com/nazeeh111/GhostMesh/blob/v0.1.0/docs/EXPERIMENTS.md
"""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def package_cohort(out):
    out, prepared = Path(out).absolute(), ROOT / "data/mmfi-cohort20"
    require_new(out)
    require_new(out.with_name(out.name + ".partial"))
    receipt_path = ROOT / "docs/cohort-provenance.json"
    require(
        digest(receipt_path) == HASHES["docs/cohort-provenance.json"],
        "Original provenance receipt changed",
    )
    receipt = json.loads(receipt_path.read_text())
    manifests = []
    entries = {}

    def add(name, source=None, data=None, expected=None):
        content = source.read_bytes() if source is not None else data
        checksum = sha256(content).hexdigest()
        require(
            expected is None or checksum == expected, f"Source hash mismatch: {name}"
        )
        entries[name] = {
            "sha256": checksum,
            "size": len(content),
            **({"source": source} if source is not None else {"data": content}),
        }

    for name, protocol in (
        ("manifest.json", "cross_subject"),
        ("cross-environment.json", "cross_environment"),
    ):
        path = prepared / name
        require(
            digest(path) == HASHES[name], f"Original manifest hash mismatch: {name}"
        )
        manifest = json.loads(path.read_text())
        require(manifest["protocol"] == protocol, "Manifest protocol mismatch")
        manifests.append(manifest["sequences"])
        add("data/mmfi-cohort20/" + name, source=path, expected=HASHES[name])

    def signature(rows):
        return {(r["id"], r["path"], r["sha256"], r["frames"]) for r in rows}

    rows = manifests[0]
    require(
        len(rows) == len(manifests[1]) == 540
        and len(signature(rows)) == 540
        and signature(rows) == signature(manifests[1]),
        "Cohort records differ between manifests",
    )
    require(
        receipt["manifest_sha256"] == HASHES["manifest.json"]
        and {(r["id"], r["frames"], r["sha256"]) for r in rows}
        == {
            (r["id"], r["frames"], r["prepared_npz_sha256"])
            for r in receipt["sequence_records"]
        },
        "Provenance records differ",
    )
    require(
        sum(r["frames"] for r in rows) == receipt["frames"] == 160380
        and receipt["sequences"] == 540,
        "Cohort frame/sequence total mismatch",
    )
    for key, count in (("subject", 20), ("action", 27), ("environment", 4)):
        require(
            {r[key] for r in rows} == set(receipt[key + "s"])
            and len({r[key] for r in rows}) == count,
            f"Unexpected cohort {key} set",
        )
    paths = set()
    for row in rows:
        relative = Path(row["path"])
        source = (prepared / relative).resolve()
        require(
            not relative.is_absolute()
            and ".." not in relative.parts
            and relative.as_posix() == row["path"]
            and source.is_relative_to(prepared.resolve())
            and source.suffix == ".npz"
            and source not in paths
            and source.is_file()
            and source.stat().st_size > 0,
            "Unsafe/duplicate/missing NPZ path",
        )
        paths.add(source)
        with zipfile.ZipFile(source) as archive:
            require(
                sorted(archive.namelist())
                == ["csi.npy", "pose.npy", "timestamps.npy", "valid.npy"],
                "Unexpected NPZ fields",
            )
        add(
            "data/mmfi-cohort20/" + relative.as_posix(),
            source=source,
            expected=row["sha256"],
        )
    for name in (
        "docs/cohort-provenance.json",
        "docs/mmfi-source.json",
        "NOTICE.md",
        "LICENSE",
    ):
        add(name, source=ROOT / name, expected=HASHES.get(name))
    license_text = (ROOT / "data/LICENSE.md").read_text()
    license_text = re.sub(
        r"\]\(\.\./([^)]*)\)", lambda m: "](" + RELEASE + m[1] + ")", license_text
    )
    add("data/LICENSE.md", data=license_text.encode())
    add("README_DATASET.md", data=README.encode())
    manifest = {
        "schema_version": 1,
        "archive_prefix": PREFIX + "/",
        "scope": "exact prepared cohort; no RGB/raw/3D/cache",
        "original_manifest_sha256": HASHES,
        "members": {
            k: {"sha256": v["sha256"], "size": v["size"]}
            for k, v in sorted(entries.items())
        },
        "self_hash_note": "Manifest excludes itself; compare the separately printed archive SHA256 to the release receipt.",
    }
    add(
        "bundle_manifest.json",
        data=(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode(),
    )
    return write_archive(out, entries, prefix=PREFIX, compresslevel=1, max_bytes=2**31)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, required=True, help="New .tar.gz output, less than 2 GiB"
    )
    args = parser.parse_args()
    archive = package_cohort(args.out)
    print(
        json.dumps(
            {
                "archive": str(archive),
                "sha256": digest(archive),
                "size": archive.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
