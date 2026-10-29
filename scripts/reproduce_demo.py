"""Recompute the shipped replay from a recording containing no pose labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.replay import infer_recording  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-kernel",
        action="store_true",
        help="Reproduce the separate same-condition kernel diagnostic",
    )
    parser.add_argument("--recording", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--tolerance", type=float, default=2e-5)
    args = parser.parse_args(argv)
    defaults = (
        (
            "examples/local-csi.npz",
            "models/local-kernel.npz",
            "examples/local-motion-replay.json",
        )
        if args.local_kernel
        else (
            "examples/heldout-csi.npz",
            "models/ghostmesh.pt",
            "ghostmesh/examples/heldout-replay.json",
        )
    )
    for name, default in zip(("recording", "checkpoint", "reference"), defaults):
        if getattr(args, name) is None:
            setattr(args, name, ROOT / default)
    if not 0 < args.tolerance <= 0.001:
        raise ValueError("Choose tolerance in (0,.001]")
    with np.load(args.recording, allow_pickle=False) as recording:
        expected_fields = (
            {"rf", "timestamps", "magnitude_units", "phase_units"}
            if args.local_kernel
            else {"csi", "timestamps"}
        )
        if set(recording.files) != expected_fields:
            raise ValueError(
                "The example must contain only CSI, timestamps and required signal units, with no pose annotations"
            )
    reference = json.loads(args.reference.read_text())

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    if reference["provenance"]["kind"] != "dataset_inference":
        raise ValueError("Expected actual dataset inference reference")
    if reference["provenance"]["checkpoint_sha256"] != digest(args.checkpoint):
        raise ValueError("Reference checkpoint hash mismatch")
    recording_digest = reference["provenance"].get("recording_sha256")
    if recording_digest is not None and recording_digest != digest(args.recording):
        raise ValueError("Reference recording hash mismatch")
    with tempfile.TemporaryDirectory(prefix="ghostmesh-reproduction-") as temporary:
        output = Path(temporary) / "recomputed.json"
        if args.local_kernel:
            from ghostmesh.local_kernel import infer_recording as infer_local

            infer_local(args.checkpoint, args.recording, output)
        else:
            infer_recording(args.recording, args.checkpoint, output)
        computed = json.loads(output.read_text())
    if len(computed["frames"]) != len(reference["frames"]):
        raise ValueError("Replay frame count differs")
    maxima = {"pose": 0.0, "confidence": 0.0}
    compared = 0
    for actual, expected in zip(computed["frames"], reference["frames"]):
        if (
            not np.isfinite([actual["time"], expected["time"]]).all()
            or abs(actual["time"] - expected["time"]) > 1e-8
        ):
            raise ValueError("Replay synchronization differs")
        if (actual["pose"] is None) != (expected["pose"] is None):
            raise ValueError("Failure withholding differs")
        if actual["pose"] is None:
            continue
        for field in maxima:
            a, b = np.asarray(actual[field]), np.asarray(expected[field])
            if (
                not np.isfinite(a).all()
                or not np.isfinite(b).all()
                or a.shape != b.shape
            ):
                raise ValueError(f"Invalid recomputed {field}")
            delta = float(np.max(np.abs(a - b)))
            if delta > args.tolerance:
                raise ValueError(f"{field} differs by {delta}, above {args.tolerance}")
            maxima[field] = max(maxima[field], delta)
        compared += 1
    if not compared:
        raise ValueError("No valid predictions compared")
    print(
        json.dumps(
            {
                "status": "reproduced_from_CSI_only",
                "frames_compared": compared,
                "max_absolute_difference": maxima,
                "tolerance": args.tolerance,
                "mode": "same_condition_kernel_diagnostic"
                if args.local_kernel
                else "released_neural",
                "checkpoint_sha256": digest(args.checkpoint),
                "recording_sha256": digest(args.recording),
                "scope": "Pose/confidence and timestamps; measured timings and unknown-domain metadata may differ. No annotation input was loaded by inference.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
