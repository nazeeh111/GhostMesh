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
        "--recording", type=Path, default=ROOT / "examples/heldout-csi.npz"
    )
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/ghostmesh.pt")
    parser.add_argument(
        "--reference",
        type=Path,
        default=ROOT / "ghostmesh/examples/heldout-replay.json",
    )
    parser.add_argument("--tolerance", type=float, default=2e-5)
    args = parser.parse_args(argv)
    if not 0 < args.tolerance <= 0.001:
        raise ValueError("Choose tolerance in (0,.001]")
    with np.load(args.recording, allow_pickle=False) as recording:
        if set(recording.files) != {"csi", "timestamps"}:
            raise ValueError(
                "The example must contain only CSI and timestamps, with no pose annotations"
            )
    reference = json.loads(args.reference.read_text())

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    if reference["provenance"]["kind"] != "dataset_inference":
        raise ValueError("Expected actual dataset inference reference")
    if reference["provenance"]["checkpoint_sha256"] != digest(args.checkpoint):
        raise ValueError("Reference checkpoint hash mismatch")
    with tempfile.TemporaryDirectory(prefix="ghostmesh-reproduction-") as temporary:
        output = Path(temporary) / "recomputed.json"
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
                "checkpoint_sha256": digest(args.checkpoint),
                "recording_sha256": digest(args.recording),
                "scope": "Pose/confidence and timestamps; measured timings and unknown-domain metadata may differ. No annotation input was loaded by inference.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
