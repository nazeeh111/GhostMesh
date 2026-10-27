"""Retain packet-axis structure for a source-only CSI encoder diagnostic."""

from hashlib import sha256
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.preprocess import prepare_window  # noqa: E402


def digest(path):
    with Path(path).open("rb") as stream:
        from hashlib import file_digest

        return file_digest(stream, "sha256").hexdigest()


def build(manifest, reference, out):
    if out.exists():
        raise FileExistsError(out)
    receipt = json.loads((reference / "receipt.json").read_text())
    assert receipt["window"] == 8
    for name, expected in receipt["arrays"].items():
        assert digest(reference / name) == expected, name
    sequence = np.load(reference / "sequence.npy", allow_pickle=False)
    ends = np.load(reference / "end.npy", allow_pickle=False)
    assert set(s[:3] for s in sequence) == {"E01", "E02", "E03"}
    manifest_bytes = manifest.read_bytes()
    assert sha256(manifest_bytes).hexdigest() == receipt["manifest_sha256"]
    rows = [
        r
        for r in json.loads(manifest_bytes)["sequences"]
        if r["environment"] in {"E01", "E02", "E03"}
    ]
    assert len(rows) == 405
    out.mkdir(parents=True)
    x = np.lib.format.open_memmap(
        out / "raw.npy",
        mode="w+",
        dtype="float32",
        shape=(len(sequence), 8, 3, 114, 10),
    )
    cursor = 0
    for i, row in enumerate(rows):
        p = manifest.parent / row["path"]
        assert digest(p) == row["sha256"] == receipt["sequences"][row["id"]]
        with np.load(p, allow_pickle=False) as archive:
            raw = archive["csi"]
        for end in range(7, len(raw), 8):
            assert sequence[cursor] == row["id"] and ends[cursor] == end
            x[cursor] = prepare_window(raw[end - 7 : end + 1], normalize=True)
            cursor += 1
        if (i + 1) % 45 == 0:
            print(
                json.dumps({"sequences": i + 1, "total": 405, "endpoints": cursor}),
                flush=True,
            )
    assert cursor == len(sequence)
    x.flush()
    del x
    result = {
        "schema_version": 1,
        "scope": "E01-E03 development exclusively; packet axis preserved, original timing inside blocks unknown",
        "reference_cache_receipt_sha256": digest(reference / "receipt.json"),
        "source_manifest_sha256": sha256(manifest_bytes).hexdigest(),
        "shape": [len(sequence), 8, 3, 114, 10],
        "raw_sha256": digest(out / "raw.npy"),
        "source_sha256": {
            "scripts/domain_raw_cache.py": digest(__file__),
            "ghostmesh/preprocess.py": digest(ROOT / "ghostmesh/preprocess.py"),
        },
    }
    (out / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    build(a.manifest, a.reference, a.out)
