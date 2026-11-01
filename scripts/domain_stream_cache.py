"""Pointwise causal RF normalization and dense source-label supervision cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.preprocess import prepare_window  # noqa:E402
from ghostmesh.rf_features import prepare_log_magnitude  # noqa:E402
from domain_raw_cache import digest  # noqa:E402


def build(manifest, reference, out):
    if out.exists():
        raise FileExistsError(out)
    r = json.loads((reference / "receipt.json").read_text())
    assert digest(manifest) == r["manifest_sha256"] and r["window"] == 8
    ids = np.load(reference / "sequence.npy", allow_pickle=False)
    ends = np.load(reference / "end.npy", allow_pickle=False)
    assert set(n[:3] for n in ids) == {"E01", "E02", "E03"}
    rows = [
        v
        for v in json.loads(manifest.read_text())["sequences"]
        if v["environment"] in {"E01", "E02", "E03"}
    ]
    assert len(rows) == 405
    n = len(ids)
    shapes = {
        "legacy": (n, 8, 3, 114, 10),
        "db": (n, 8, 3, 114, 10),
        "root_spectrum": (n, 8, 342),
        "pose_sequence": (n, 8, 17, 2),
        "valid_sequence": (n, 8, 17),
    }
    out.mkdir(parents=True)
    arrays = {
        k: np.lib.format.open_memmap(
            out / (k + ".npy"),
            mode="w+",
            dtype="bool" if k == "valid_sequence" else "float32",
            shape=v,
        )
        for k, v in shapes.items()
    }
    cursor = 0
    started = time.monotonic()
    for i, row in enumerate(rows):
        p = manifest.parent / row["path"]
        assert digest(p) == row["sha256"]
        with np.load(p, allow_pickle=False) as a:
            raw, pose, valid = a["csi"], a["pose"], a["valid"]
        assert valid.all() and np.isfinite(pose).all()
        legacy = np.empty_like(raw)
        db = np.empty_like(raw)
        root = np.empty((len(raw), 342), dtype="float32")
        for frame in range(len(raw)):
            x = raw[frame : frame + 1]
            legacy[frame] = prepare_window(x, True)[0]
            db[frame] = prepare_log_magnitude(x, units="log_magnitude_db")[0]
            filled = np.where(np.isfinite(x), x, np.nan)
            median = np.nanmedian(filled, axis=(2, 3), keepdims=True)
            assert np.isfinite(median).all()
            filled = np.where(np.isfinite(filled), filled, median)
            root[frame] = ((filled.mean(-1) - 40.0) / 20.0).reshape(-1)
        for end in range(7, len(raw), 8):
            assert ids[cursor] == row["id"] and ends[cursor] == end
            sl = slice(end - 7, end + 1)
            for k, v in [
                ("legacy", legacy),
                ("db", db),
                ("root_spectrum", root),
                ("pose_sequence", pose),
                ("valid_sequence", valid),
            ]:
                arrays[k][cursor] = v[sl]
            cursor += 1
        if (i + 1) % 15 == 0:
            print(
                json.dumps(
                    {
                        "sequences": i + 1,
                        "total": 405,
                        "windows": cursor,
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    assert cursor == n
    for array in arrays.values():
        array.flush()
    arrays.clear()
    result = {
        "schema_version": 1,
        "scope": "Only E01-E03; each frame normalized using itself, never later frames. Dense past/current labels for training only. Endpoints remain evaluated at stride8 with7warmupframes.",
        "windows": n,
        "window": 8,
        "frames_supervised_per_window": 8,
        "root_spectrum": "Packet-mean original log magnitude after finite frame-median fill; fixed (dB-40)/20 transform preserves receiver power information. Relative IQ units, not dBm.",
        "reference_receipt_sha256": digest(reference / "receipt.json"),
        "array_sha256": {k: digest(out / (k + ".npy")) for k in shapes},
        "shapes": {k: list(v) for k, v in shapes.items()},
        "source_sha256": {
            str(p.relative_to(ROOT)): digest(p)
            for p in [
                Path(__file__),
                ROOT / "ghostmesh/preprocess.py",
                ROOT / "ghostmesh/rf_features.py",
            ]
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
