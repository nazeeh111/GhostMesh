"""Import previously unused phase from original source-room MAT files only."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.phase_features import relative_phase  # noqa:E402
from domain_raw_cache import digest  # noqa:E402


def build(manifest, raw_root, reference, out):
    if out.exists():
        raise FileExistsError(out)
    receipt = json.loads((reference / "receipt.json").read_text())
    assert receipt["window"] == 8
    assert digest(manifest) == receipt["manifest_sha256"]
    sequence = np.load(reference / "sequence.npy", allow_pickle=False)
    ends = np.load(reference / "end.npy", allow_pickle=False)
    assert set(s[:3] for s in sequence) == {"E01", "E02", "E03"}
    rows = [
        r
        for r in json.loads(manifest.read_text())["sequences"]
        if r["environment"] in {"E01", "E02", "E03"}
    ]
    assert len(rows) == 405
    out.mkdir(parents=True)
    output = np.lib.format.open_memmap(
        out / "phase.npy",
        mode="w+",
        dtype="float32",
        shape=(len(sequence), 8, 4, 114, 10),
    )
    cursor = 0
    hashes = {}
    quality = []
    started = time.monotonic()
    for i, row in enumerate(rows):
        prepared = manifest.parent / row["path"]
        assert digest(prepared) == row["sha256"]
        with np.load(prepared, allow_pickle=False) as archive:
            amp = archive["csi"]
        p = np.empty(amp.shape, dtype="float32")
        h = sha256()
        for f in range(len(amp)):
            source = raw_root / row["id"] / "wifi-csi" / f"frame{f + 1:03d}.mat"
            b = source.read_bytes()
            h.update(source.name.encode() + b"\0" + sha256(b).digest())
            m = loadmat(source, variable_names=["CSIamp", "CSIphase"])
            assert m["CSIphase"].shape == amp.shape[1:]
            assert np.array_equal(
                np.asarray(m["CSIamp"], dtype="float32"), amp[f], equal_nan=True
            ), (row["id"], f)
            p[f] = m["CSIphase"]
        for end in range(7, len(amp), 8):
            assert sequence[cursor] == row["id"] and ends[cursor] == end
            output[cursor] = relative_phase(
                amp[end - 7 : end + 1], p[end - 7 : end + 1]
            )
            cursor += 1
        quality.append(
            {"id": row["id"], "phase_finite_fraction": float(np.isfinite(p).mean())}
        )
        hashes[row["id"]] = h.hexdigest()
        if (i + 1) % 15 == 0:
            print(
                json.dumps(
                    {
                        "sequences": i + 1,
                        "total": 405,
                        "endpoints": cursor,
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    assert cursor == len(sequence)
    output.flush()
    del output
    result = {
        "schema_version": 1,
        "scope": "Source-development E01-E03 only. All raw amplitudes checked exactly against original prepared arrays. No E04 or reserved data decoded.",
        "shape": [len(sequence), 8, 4, 114, 10],
        "phase_features": "cos(phi_i-phi_0),sin(phi_i-phi_0) for RX1/RX2; no unwrap, no calibrated AoA/ToF claim",
        "phase_sha256": digest(out / "phase.npy"),
        "reference_cache_receipt_sha256": digest(reference / "receipt.json"),
        "raw_sequence_ordered_file_hashes": hashes,
        "phase_quality": quality,
        "sources": {
            str(p.relative_to(ROOT)): digest(p)
            for p in [
                Path(__file__),
                ROOT / "ghostmesh/phase_features.py",
                ROOT / "scripts/domain_raw_cache.py",
            ]
        },
    }
    (out / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    build(a.manifest, a.raw_root, a.reference, a.out)
