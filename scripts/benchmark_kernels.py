#!/usr/bin/env python3
"""Verify and time equivalent temporal kernels on random latent activations.

This measures three causal temporal blocks only. It does not measure GhostMesh
end-to-end speed, CSI acquisition, preprocessing, the frame encoder, a teacher,
rendering, or pose accuracy. No dataset or checkpoint is required.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh import models  # noqa: E402 - allow direct execution from a checkout
from ghostmesh.models import CausalBlock  # noqa: E402


def make_pair(width: int, dtype=torch.float32):
    native = nn.Sequential(
        *(CausalBlock(width, dilation, "native") for dilation in (1, 2, 4))
    ).to(dtype=dtype)
    direct = nn.Sequential(
        *(CausalBlock(width, dilation, "direct") for dilation in (1, 2, 4))
    ).to(dtype=dtype)
    direct.load_state_dict(native.state_dict(), strict=True)
    return native, direct


def verify_parity(widths, batch_size: int, timesteps: int):
    """Check full chains, gradients, native checkpoint reload and prefix causality."""
    cases = []
    for width in widths:
        for dtype in (torch.float32, torch.float64):
            for batch in sorted({1, batch_size}):
                native, direct = make_pair(width, dtype)
                x = torch.randn(
                    batch, width, timesteps, dtype=dtype, requires_grad=True
                )
                y = x.detach().clone().requires_grad_(True)
                expected, actual = native(x), direct(y)
                tolerance = 3e-5 if dtype == torch.float32 else 1e-10
                torch.testing.assert_close(
                    actual, expected, atol=tolerance, rtol=tolerance
                )
                native_gradients = torch.autograd.grad(
                    expected.square().mean(), (x, *native.parameters())
                )
                direct_gradients = torch.autograd.grad(
                    actual.square().mean(), (y, *direct.parameters())
                )
                gradient_difference = 0.0
                for first, second in zip(native_gradients, direct_gradients):
                    torch.testing.assert_close(
                        first, second, atol=tolerance, rtol=tolerance
                    )
                    gradient_difference = max(
                        gradient_difference, float((first - second).abs().max())
                    )
                native_schema = {
                    key: list(value.shape) for key, value in native.state_dict().items()
                }
                direct_schema = {
                    key: list(value.shape) for key, value in direct.state_dict().items()
                }
                if native_schema != direct_schema:
                    raise AssertionError(
                        "Kernel checkpoint keys or parameter shapes changed"
                    )
                payload = BytesIO()
                torch.save(direct.state_dict(), payload)
                payload.seek(0)
                reloaded, _ = make_pair(width, dtype)
                reloaded.load_state_dict(
                    torch.load(payload, weights_only=True), strict=True
                )
                torch.testing.assert_close(
                    reloaded(x.detach()),
                    expected.detach(),
                    atol=tolerance,
                    rtol=tolerance,
                )
                midpoint = timesteps // 2
                changed = y.detach().clone()
                changed[..., midpoint:] += 20
                torch.testing.assert_close(
                    direct(changed)[..., :midpoint],
                    actual.detach()[..., :midpoint],
                    atol=tolerance,
                    rtol=tolerance,
                )
                cases.append(
                    {
                        "width": width,
                        "dtype": str(dtype),
                        "batch_size": batch,
                        "timesteps": timesteps,
                        "max_forward_abs_difference": float(
                            (expected - actual).detach().abs().max()
                        ),
                        "max_gradient_abs_difference": gradient_difference,
                        "tolerance": tolerance,
                        "checkpoint_reload": "passed",
                        "prefix_causality": "passed",
                    }
                )
    return {
        "status": "passed",
        "scope": "three blocks with dilations 1,2,4; shared weights",
        "cases": cases,
    }


def _measure(operation):
    wall, cpu = time.perf_counter_ns(), time.process_time_ns()
    operation()
    return (time.perf_counter_ns() - wall) / 1e6, (time.process_time_ns() - cpu) / 1e6


def time_pair(width, threads, timesteps, batch_size, repeats, warmup):
    torch.set_num_threads(threads)
    native, direct = make_pair(width)
    kernels = {"native": native, "direct": direct}
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=0.001)
        for name, model in kernels.items()
    }
    single = torch.randn(1, width, timesteps)
    batch = torch.randn(batch_size, width, timesteps)

    def forward(name):
        with torch.no_grad():
            kernels[name](single)

    def train_step(name):
        optimizers[name].zero_grad(set_to_none=True)
        kernels[name](batch).square().mean().backward()
        optimizers[name].step()

    for name in kernels:
        for _ in range(warmup):
            forward(name)
            train_step(name)
    samples = {
        name: {
            "forward_wall_ms": [],
            "forward_cpu_ms": [],
            "train_wall_ms": [],
            "train_cpu_ms": [],
        }
        for name in kernels
    }
    for mode, count in (("forward", max(15, repeats * 3)), ("train", repeats)):
        for index in range(count):
            order = ("native", "direct") if index % 2 == 0 else ("direct", "native")
            for name in order:
                operation = (
                    (lambda name=name: forward(name))
                    if mode == "forward"
                    else (lambda name=name: train_step(name))
                )
                wall_ms, cpu_ms = _measure(operation)
                samples[name][f"{mode}_wall_ms"].append(wall_ms)
                samples[name][f"{mode}_cpu_ms"].append(cpu_ms)
    statistics = {
        name: {
            key: {
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
            }
            for key, values in measurements.items()
        }
        for name, measurements in samples.items()
    }
    ratios = {
        key: statistics["native"][key]["median"] / statistics["direct"][key]["median"]
        for key in samples["native"]
    }
    return {
        "width": width,
        "torch_threads": threads,
        "timesteps": timesteps,
        "forward_batch_size": 1,
        "train_batch_size": batch_size,
        "warmup_steps_per_kernel": warmup,
        "statistics_ms": statistics,
        "native_over_direct_median_ratio": ratios,
        "samples_ms": samples,
    }


def run_benchmark(
    output,
    *,
    widths=(32, 64),
    threads=(1, 2),
    timesteps=8,
    batch_size=32,
    repeats=7,
    warmup=3,
    seed=2027,
):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark: {output}")
    if not widths or not threads or any(width < 8 or width % 8 for width in widths):
        raise ValueError("Widths must be positive multiples of eight")
    if (
        any(value < 1 for value in threads)
        or min(batch_size, repeats, warmup) < 1
        or timesteps < 2
    ):
        raise ValueError(
            "Positive threads/batch/repeats/warmup and at least two timesteps required"
        )
    if repeats > 1000:
        raise ValueError("This bounded benchmark accepts at most 1000 training repeats")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    parity = verify_parity(widths, batch_size, timesteps)
    measurements = [
        time_pair(width, count, timesteps, batch_size, repeats, warmup)
        for count in threads
        for width in widths
    ]
    result = {
        "schema_version": 1,
        "purpose": "equivalent_temporal_kernel_benchmark",
        "scope": "three temporal blocks on random latent activations; batch-one forward and optimizer step",
        "excludes": [
            "CSI acquisition",
            "data loading",
            "signal preprocessing",
            "frame encoder",
            "EMA teacher",
            "network/server",
            "rendering",
            "end-to-end FPS",
            "pose accuracy",
        ],
        "test_data_used": False,
        "seed": seed,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "torch_interop_threads": torch.get_num_interop_threads(),
        },
        "source_sha256": {
            "ghostmesh/models.py": sha256(
                Path(models.__file__).read_bytes()
            ).hexdigest(),
            "scripts/benchmark_kernels.py": sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
        "parity": parity,
        "measurements": measurements,
        "interpretation": "Ratios apply only to the measured temporal kernels. Wall latency is affected by competing work; process CPU time sums this process's CPU use. These are not whole-system speedups or accuracy improvements.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--widths", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--timesteps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args(argv)
    result = run_benchmark(
        args.out,
        widths=args.widths,
        threads=args.threads,
        timesteps=args.timesteps,
        batch_size=args.batch_size,
        repeats=args.repeats,
        warmup=args.warmup,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "output": str(args.out),
                "parity": result["parity"]["status"],
                "cases": len(result["measurements"]),
                "scope": result["scope"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
