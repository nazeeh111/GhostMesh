"""Build compact typed amplitude/phase source30 cache; E04 is never decoded."""

from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.domain_raw_cache import digest  # noqa: E402


def source_records(config):
    if config["environments"] != ["E01", "E02", "E03"]:
        raise ValueError("This development builder forbids E04")
    if (
        config["subjects_per_environment"],
        config["actions"],
        config["frames_per_record"],
    ) != (10, 27, 297):
        raise ValueError("Expected frozen complete source30 scope")
    return [
        f"E{room:02d}/S{person:02d}/A{action:02d}"
        for room in range(1, 4)
        for person in range(10 * (room - 1) + 1, 10 * room + 1)
        for action in range(1, 28)
    ]


def build(raw_root, config_path, byte_audit_path, output):
    if output.exists():
        raise FileExistsError(output)
    config = json.loads(config_path.read_text())
    records = source_records(config)
    audit = json.loads(byte_audit_path.read_text())
    if (
        audit["records"],
        audit["paired_frames"],
        audit["reserved_records_unchanged"],
    ) != (1080, 320760, 16):
        raise ValueError("A successful full40 byte audit is required first")
    if (
        config["magnitude_units"] != "log_magnitude_db"
        or config["phase_units"] != "radians"
    ):
        raise ValueError("Explicit typed RF units required")
    canvas = np.asarray(config["reference_canvas"], dtype="float32")
    if canvas.shape != (2,) or not np.all(canvas > 0):
        raise ValueError("Explicit positive label reference plane required")
    n = len(records) * 297
    output.mkdir(parents=True)
    arrays = {
        name: np.lib.format.open_memmap(
            output / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
        )
        for name, dtype, shape in [
            ("rf", "float32", (n, 6, 114, 10)),
            ("pose", "float32", (n, 17, 2)),
            ("valid", "bool", (n, 17)),
        ]
    }
    started = time.monotonic()
    receipt_rows = []
    for index, record in enumerate(records):
        folder = raw_root / record
        expected_names = {f"frame{i:03d}.mat" for i in range(1, 298)}
        if {p.name for p in (folder / "wifi-csi").iterdir()} != expected_names:
            raise ValueError(f"Unexpected CSI frame set: {record}")
        ordered = sha256()
        invalid = missing_amp = missing_phase = unit_points = 0
        unit_residual_max = 0.0
        # The audit hashes relative paths in lexicographic modality order.
        for modality, suffix in [("rgb", ".npy"), ("wifi-csi", ".mat")]:
            for frame in range(1, 298):
                relative = f"{modality}/frame{frame:03d}{suffix}"
                path = folder / relative
                content = path.read_bytes()
                ordered.update(relative.encode() + b"\0" + sha256(content).digest())
                row = index * 297 + frame - 1
                if modality == "rgb":
                    pose = np.asarray(
                        np.load(BytesIO(content), allow_pickle=False), dtype="float32"
                    )
                    if pose.shape != (17, 2):
                        raise ValueError(f"Expected provided2D labels: {path}")
                    pose = pose / canvas
                    valid = np.isfinite(pose).all(-1) & ((pose >= 0) & (pose <= 1)).all(
                        -1
                    )
                    arrays["pose"][row] = np.where(valid[:, None], pose, 0)
                    arrays["valid"][row] = valid
                    invalid += int((~valid).sum())
                else:
                    mat = loadmat(
                        BytesIO(content), variable_names=["CSIamp", "CSIphase"]
                    )
                    amp, phase = mat["CSIamp"], mat["CSIphase"]
                    if amp.shape != (3, 114, 10) or amp.shape != phase.shape:
                        raise ValueError(f"Unexpected RF shape: {path}")
                    missing_amp += int((~np.isfinite(amp)).sum())
                    missing_phase += int((~np.isfinite(phase)).sum())
                    if frame in (1, 149, 297):
                        finite = np.isfinite(amp) & np.isfinite(phase)
                        complex_value = 10 ** (amp[finite] / 20) * np.exp(
                            1j * phase[finite]
                        )
                        residual = np.maximum(
                            abs(complex_value.real - complex_value.real.round()),
                            abs(complex_value.imag - complex_value.imag.round()),
                        )
                        if len(residual) == 0 or not np.all(residual < 1e-8):
                            raise ValueError(f"Log-unit inference check failed: {path}")
                        unit_points += len(residual)
                        unit_residual_max = max(
                            unit_residual_max, float(residual.max())
                        )
                    arrays["rf"][row, :3] = amp
                    arrays["rf"][row, 3:] = phase
        if ordered.hexdigest() != audit["ordered_record_sha256"][record]:
            raise ValueError(f"Source bytes changed since CRC audit: {record}")
        receipt_rows.append(
            {
                "id": record,
                "offset": index * 297,
                "frames": 297,
                "ordered_raw_sha256": ordered.hexdigest(),
                "invalid_joints": invalid,
                "missing_magnitudes": missing_amp,
                "missing_phases": missing_phase,
                "unit_check_frames": [1, 149, 297],
                "unit_check_finite_points": unit_points,
                "unit_check_max_iq_rounding_residual": unit_residual_max,
            }
        )
        if (index + 1) % 27 == 0:
            print(
                json.dumps(
                    {
                        "records": index + 1,
                        "total": len(records),
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    for value in arrays.values():
        value.flush()
    arrays.clear()
    result = {
        "schema_version": 1,
        "scope": config["scope"],
        "config": config,
        "config_sha256": digest(config_path),
        "byte_audit_sha256": digest(byte_audit_path),
        "script_sha256": digest(Path(__file__)),
        "arrays": {
            name: digest(output / f"{name}.npy") for name in ("rf", "pose", "valid")
        },
        "frames": n,
        "records": receipt_rows,
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "receipt.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--byte-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.raw_root, args.config, args.byte_audit, args.out)
