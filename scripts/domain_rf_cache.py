"""Audit MM-Fi log units and build source-only dB/complex-quotient caches."""

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
from ghostmesh.rf_features import prepare_log_magnitude, quotient_features  # noqa:E402
from domain_raw_cache import digest  # noqa:E402


def build(manifest, raw_root, reference, out):
    if out.exists():
        raise FileExistsError(out)
    receipt = json.loads((reference / "receipt.json").read_text())
    assert digest(manifest) == receipt["manifest_sha256"]
    sequence = np.load(reference / "sequence.npy", allow_pickle=False)
    ends = np.load(reference / "end.npy", allow_pickle=False)
    assert set(s[:3] for s in sequence) == {"E01", "E02", "E03"}
    rows = [
        r
        for r in json.loads(manifest.read_text())["sequences"]
        if r["environment"] in {"E01", "E02", "E03"}
    ]
    assert len(rows) == 405 and receipt["window"] == 8
    out.mkdir(parents=True)
    outputs = {
        key: np.lib.format.open_memmap(
            out / (key + ".npy"),
            mode="w+",
            dtype="float32",
            shape=(len(sequence), 8, channels, 114, 10),
        )
        for key, channels in [("db", 3), ("quotient", 6)]
    }
    cursor = 0
    audit = []
    start = time.monotonic()
    for i, row in enumerate(rows):
        prepared = manifest.parent / row["path"]
        assert digest(prepared) == row["sha256"]
        with np.load(prepared, allow_pickle=False) as archive:
            amp = archive["csi"]
        phase = np.empty(amp.shape, dtype="float32")
        count = integer = 0
        residual_max = 0.0
        source_digest = sha256()
        for f in range(len(amp)):
            p = raw_root / row["id"] / "wifi-csi" / f"frame{f + 1:03d}.mat"
            b = p.read_bytes()
            source_digest.update(p.name.encode() + b"\0" + sha256(b).digest())
            m = loadmat(p, variable_names=["CSIamp", "CSIphase"])
            a, phi = m["CSIamp"], m["CSIphase"]
            assert a.shape == phi.shape == amp.shape[1:]
            assert np.array_equal(a.astype("float32"), amp[f], equal_nan=True)
            good = np.isfinite(a) & np.isfinite(phi)
            value = 10 ** (a[good] / 20.0) * np.exp(1j * phi[good])
            residual = np.maximum(
                abs(value.real - value.real.round()),
                abs(value.imag - value.imag.round()),
            )
            count += len(residual)
            integer += int((residual < 1e-8).sum())
            residual_max = max(residual_max, float(residual.max()))
            phase[f] = phi
        for end in range(7, len(amp), 8):
            assert sequence[cursor] == row["id"] and ends[cursor] == end
            a, phi = amp[end - 7 : end + 1], phase[end - 7 : end + 1]
            outputs["db"][cursor] = prepare_log_magnitude(a, units="log_magnitude_db")
            outputs["quotient"][cursor] = quotient_features(
                a, phi, units="log_magnitude_db"
            )
            cursor += 1
        audit.append(
            {
                "id": row["id"],
                "finite_complex_samples": count,
                "integer_iq_at1e-8": integer,
                "max_iq_rounding_residual": residual_max,
                "ordered_raw_mat_hash": source_digest.hexdigest(),
            }
        )
        if (i + 1) % 15 == 0:
            print(
                json.dumps(
                    {
                        "sequences": i + 1,
                        "total": 405,
                        "windows": cursor,
                        "seconds": time.monotonic() - start,
                    }
                ),
                flush=True,
            )
    assert cursor == len(sequence)
    for array in outputs.values():
        array.flush()
    outputs.clear()
    result = {
        "schema_version": 1,
        "scope": "E01-E03 development only, no E04 targets decoded",
        "units_inference": "CSIamp behaves as20log10 magnitude of integer-valued original receiver I/Q: tested against original double-precision amplitude+phase for every source MAT sample. This is not calibrated RSSI in dBm.",
        "unit_audit": audit,
        "finite_complex_samples": sum(r["finite_complex_samples"] for r in audit),
        "integer_iq_at1e8": sum(r["integer_iq_at1e-8"] for r in audit),
        "window": 8,
        "windows": cursor,
        "reference_receipt_sha256": digest(reference / "receipt.json"),
        "array_sha256": {
            name: digest(out / (name + ".npy")) for name in ["db", "quotient"]
        },
        "source_sha256": {
            str(p.relative_to(ROOT)): digest(p)
            for p in [
                Path(__file__),
                ROOT / "ghostmesh/rf_features.py",
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
