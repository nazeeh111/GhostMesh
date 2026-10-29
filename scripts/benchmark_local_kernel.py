"""Measure batch-one local-kernel inference including signal preprocessing.

Wall-clock latency includes scheduling contention. It is not dataset cadence or
rendering throughput, and is not directly comparable with forward-only timing.
"""

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.data import sha256  # noqa:E402
from ghostmesh.local_kernel import load_model, predict  # noqa:E402


def run(checkpoint, recording, output, repetitions=30):
    if output.exists():
        raise FileExistsError(output)
    if repetitions < 10:
        raise ValueError("At least10 measured calls required")
    model = load_model(checkpoint)
    with np.load(recording, allow_pickle=False) as source:
        if (
            str(source["magnitude_units"]) != "log_magnitude_db"
            or str(source["phase_units"]) != "radians"
        ):
            raise ValueError("Explicit compatible RF units required")
        rf = source["rf"]
    if len(rf) < 8:
        raise ValueError("No complete context")
    ends = np.linspace(7, len(rf) - 1, repetitions).round().astype(int)
    for end in ends[:5]:
        predict(model, rf[end - 7 : end + 1])
    timings, cpu = [], []
    for end in ends:
        began, cpu_began = time.perf_counter_ns(), time.process_time_ns()
        prediction = predict(model, rf[end - 7 : end + 1])
        elapsed, cpu_elapsed = (
            time.perf_counter_ns() - began,
            time.process_time_ns() - cpu_began,
        )
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite prediction")
        timings.append(elapsed / 1e6)
        cpu.append(cpu_elapsed / 1e6)
    result = dict(
        scope="Restricted same-condition model. Batch1 CPU preprocessing+RBF inference, warm checkpoint; excludes model load, acquisition, I/O and rendering.",
        cadence_hz=10.0,
        cadence_basis="Nominal training/sample cadence; not the inference throughput below",
        measured_calls=repetitions,
        warmup_calls=5,
        selected_sample_endpoints=ends.tolist(),
        wall_ms=dict(
            median=float(np.median(timings)),
            p95=float(np.percentile(timings, 95)),
            samples=timings,
        ),
        cpu_ms=dict(median=float(np.median(cpu)), samples=cpu),
        reciprocal_wall_median_fps=1000 / float(np.median(timings)),
        python=platform.python_version(),
        platform=platform.platform(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        source_sha256={
            str(p.relative_to(ROOT)): sha256(p)
            for p in (
                Path(__file__),
                ROOT / "ghostmesh/local_kernel.py",
                ROOT / "ghostmesh/rf_features.py",
            )
        },
        checkpoint_sha256=sha256(checkpoint),
        recording_sha256=sha256(recording),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("source_sha256", "wall_ms", "cpu_ms")
            }
            | {
                "wall_median_ms": result["wall_ms"]["median"],
                "cpu_median_ms": result["cpu_ms"]["median"],
            }
        ),
        flush=True,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "models/local-kernel.npz"
    )
    parser.add_argument(
        "--recording", type=Path, default=ROOT / "examples/local-csi.npz"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=30)
    args = parser.parse_args()
    run(args.checkpoint, args.recording, args.out, args.repetitions)
