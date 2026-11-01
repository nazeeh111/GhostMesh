"""Interleaved comparison with the exact kernel from commit88a9653.

Caches a constant reference norm; model coefficients and predictions are fixed.
No annotations, fit, device purchase, or claimed generalization improvement.
"""

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy.linalg.blas import dgemm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.local_kernel import design, load_model, predict  # noqa:E402
from ghostmesh.data import sha256  # noqa:E402


def uncached(model, rf):
    # Exact preoptimization computation, retained only as a measured control.
    query = (design(rf) - model["input_mean"]) / model["input_scale"]
    training = model["training_design"]
    cross = dgemm(1.0, query, training.T)
    distance = np.maximum(
        np.square(query).sum(1)[:, None] + np.square(training).sum(1)[None] - 2 * cross,
        0,
    )
    kernel = np.exp(-distance / (2 * float(model["bandwidth_squared"])))
    return (
        (model["target_mean"] + dgemm(1.0, kernel, model["dual"]))
        .reshape(-1, 17, 2)
        .clip(0, 1)
    )


def run(model_path, recording_path, out):
    if out.exists():
        raise FileExistsError(out)
    model = load_model(model_path)
    with np.load(recording_path, allow_pickle=False) as data:
        if (
            str(data["magnitude_units"]) != "log_magnitude_db"
            or str(data["phase_units"]) != "radians"
        ):
            raise ValueError("TypedRFrequired")
        rf = data["rf"]
    ends = np.linspace(7, len(rf) - 1, 30).round().astype(int)
    methods = {"uncached_88a9653": uncached, "cached": predict}
    for end in ends[:5]:
        for function in methods.values():
            function(model, rf[end - 7 : end + 1])
    samples = {name: [] for name in methods}
    rng = np.random.default_rng(2027)
    maximum_difference = 0.0
    for end in ends:
        predictions = {}
        for name in rng.permutation(list(methods)):
            started, cpu = time.perf_counter_ns(), time.process_time_ns()
            predictions[name] = methods[name](model, rf[end - 7 : end + 1])
            wall_elapsed, cpu_elapsed = (
                time.perf_counter_ns() - started,
                time.process_time_ns() - cpu,
            )
            samples[name].append(
                dict(end=int(end), wall_ms=wall_elapsed / 1e6, cpu_ms=cpu_elapsed / 1e6)
            )
        maximum_difference = max(
            maximum_difference,
            float(abs(predictions["cached"] - predictions["uncached_88a9653"]).max()),
        )
    if maximum_difference > 1e-9:
        raise ValueError("Optimization changed inference")
    summary = {
        name: dict(
            wall_median_ms=float(np.median([x["wall_ms"] for x in rows])),
            cpu_median_ms=float(np.median([x["cpu_ms"] for x in rows])),
        )
        for name, rows in samples.items()
    }
    for value in summary.values():
        value["reciprocal_wall_median_fps"] = 1000 / value["wall_median_ms"]
    result = dict(
        scope="Batch1 CPU quotient preprocessing+RBF inference, warmcheckpoint. Interleaved paired controls, randomized order seed2027; excludes load,acquisition,I/O,rendering. Runtime contention can affect wall latency.",
        baseline_commit="88a9653744510893476ec934bf12903f8ac7ccd9",
        measured_calls_per_method=30,
        warmup_per_method=5,
        prediction_max_abs=maximum_difference,
        summary=summary,
        samples=samples,
        median_wall_speedup=summary["uncached_88a9653"]["wall_median_ms"]
        / summary["cached"]["wall_median_ms"],
        median_cpu_speedup=summary["uncached_88a9653"]["cpu_median_ms"]
        / summary["cached"]["cpu_median_ms"],
        source_sha256={
            str(p.relative_to(ROOT)): sha256(p)
            for p in (
                Path(__file__),
                ROOT / "ghostmesh/local_kernel.py",
                ROOT / "ghostmesh/rf_features.py",
            )
        },
        model_sha256=sha256(model_path),
        recording_sha256=sha256(recording_path),
        platform=platform.platform(),
        python=platform.python_version(),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("samples", "source_sha256")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/local-kernel.npz")
    parser.add_argument(
        "--recording", type=Path, default=ROOT / "examples/local-csi.npz"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.model, args.recording, args.out)
