"""Measure compute-only latency on a fixed real SOURCE RF window.

No pose labels, fitting or model selection. Excludes disk I/O, acquisition,
UI, synchronization and conservative availability-mask handling. Faster compute
throughput does not increase the released data's nominal 10 Hz frame rate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.doppler_features import complex_quotients, spectral_features  # noqa:E402
from scripts import domain_packet_transfer_probe as raw  # noqa:E402
from scripts import domain_spectral_shape_probe as shape  # noqa:E402


def benchmark(mode, experiment, source, report):
    if report.exists():
        raise FileExistsError(report)
    receipt = json.loads((source / "receipt.json").read_text())
    raw.validate_records(receipt["records"])
    record = next(r for r in receipt["records"] if r["id"] == "E01/S01/A12")
    rf = np.load(source / "rf.npy", mmap_mode="r", allow_pickle=False)
    window = np.asarray(rf[record["offset"] + 32 : record["offset"] + 40])
    with np.load(experiment / "E01-natural_rff-map.npz", allow_pickle=False) as f:
        mapping = dict(f)
    with np.load(experiment / "E01-natural_rff-model.npz", allow_pickle=False) as f:
        model = dict(f)

    def infer():
        q, _ = complex_quotients(
            window, magnitude_units="log_magnitude_db", phase_units="radians"
        )
        value = spectral_features(q, "natural")
        if mode == "shape":
            value = shape.log_power_fractions(value)
        return raw.predict(raw.transform(value[None], mapping), model)

    for _ in range(10):
        infer()
    latency = []
    for _ in range(200):
        start = time.perf_counter()
        infer()
        latency.append((time.perf_counter() - start) * 1000)
    result = dict(
        mode=mode,
        scope="Post-fit compute-only timing on fixed E01/S01/A12 window ending39; E01 natural RFF512 and all four heads. Includes quotients, FFT380, normalization if shape, map and prediction. Excludes I/O, acquisition, UI, synchronization and conservative mask handling. No pose labels or selection.",
        warmup=10,
        iterations=200,
        median_ms=float(np.median(latency)),
        p95_ms=float(np.percentile(latency, 95)),
        mean_ms=float(np.mean(latency)),
        median_compute_windows_per_second=float(1000 / np.median(latency)),
        platform=platform.platform(),
        input_source_receipt_sha256=raw.digest(source / "receipt.json"),
        model_sha256=raw.digest(experiment / "E01-natural_rff-model.npz"),
        map_sha256=raw.digest(experiment / "E01-natural_rff-map.npz"),
        source_sha256={
            name: raw.digest(ROOT / name)
            for name in (
                "scripts/benchmark_packet_transfer.py",
                "ghostmesh/doppler_features.py",
                "scripts/domain_packet_transfer_probe.py",
                "scripts/domain_spectral_shape_probe.py",
            )
        },
        thread_settings="Set OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 before starting Python to reproduce single-thread settings. Actual timing depends on host load.",
        units="Milliseconds of computation per window. Input nominally 8 grouped 10 Hz frames; exact subpacket timestamps not independently verified. This is not end-to-end application FPS.",
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x") as f:
        f.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("raw", "shape"), required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "work/domain-source30-rf-v1"
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    benchmark(args.mode, args.experiment, args.source, args.report)
