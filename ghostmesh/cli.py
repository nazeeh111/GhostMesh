"""Commands deliberately separate data preparation, training and held-out testing."""

from __future__ import annotations
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="ghostmesh", description="Temporal Wi-Fi CSI to 2D pose research"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser(
        "fixture", help="Generate synthetic software-test fixtures, not efficacy data"
    )
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=17)
    p = sub.add_parser(
        "prepare-mmfi", help="Pair real MM-Fi CSIamp and supplied 2D labels by frame ID"
    )
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--width", required=True, type=int)
    p.add_argument("--height", required=True, type=int)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--labels", choices=["rgb", "infra1", "infra2"], default="rgb")
    p.add_argument(
        "--subjects", nargs="+", help="Explicit subject cohort, e.g. S01 S02"
    )
    p.add_argument(
        "--actions", nargs="+", help="Explicit action selection, e.g. A02 A06"
    )
    p.add_argument(
        "--provenance",
        required=True,
        help="JSON dataset, source, license, consent_basis",
    )
    for name in ("train", "evaluate", "export"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
        if name == "train":
            p.add_argument("--config", required=True)
            p.add_argument(
                "--resume",
                action="store_true",
                help="Resume an identical interrupted run from its last completed epoch",
            )
        else:
            p.add_argument("--checkpoint", required=True)
            p.add_argument("--split", default="test", choices=["validation", "test"])
        if name == "export":
            p.add_argument("--sequence")
    p = sub.add_parser(
        "audit", help="Validate content hashes, schema and split separation"
    )
    p.add_argument("--manifest", required=True)
    p = sub.add_parser(
        "esp-record",
        help="Hardware-only serial signal capture; requires consenting participants",
    )
    p.add_argument("--port", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--consent-note", required=True)
    p = sub.add_parser(
        "esp-convert",
        help="Convert ESP-CSI CSV to amplitude recording; does not infer pose",
    )
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("serve", help="Open the local hardware-free replay application")
    p.add_argument(
        "--replay",
        default=str(
            Path(__file__).resolve().parent / "examples" / "heldout-replay.json"
        ),
    )
    p.add_argument("--port", type=int, default=8765)
    p = sub.add_parser(
        "infer-recording",
        help="Camera/annotation-free inference on a compatible CSI NPZ recording",
    )
    p.add_argument("--recording", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--max-gap", type=float, default=0.25)
    args = parser.parse_args(argv)
    if args.command == "fixture":
        from .prepare import make_fixture

        result = {"manifest": str(make_fixture(args.out, args.seed))}
    elif args.command == "prepare-mmfi":
        from .prepare import prepare_mmfi

        result = {
            "manifest": str(
                prepare_mmfi(
                    args.source,
                    args.out,
                    width=args.width,
                    height=args.height,
                    fps=args.fps,
                    label_modality=args.labels,
                    provenance=json.loads(Path(args.provenance).read_text()),
                    subjects=args.subjects,
                    actions=args.actions,
                )
            )
        }
    elif args.command == "audit":
        from .data import load_manifest, WindowDataset

        m = load_manifest(args.manifest)
        result = {
            "provenance": m["provenance"],
            "protocol": m["protocol"],
            "sequences": len(m["sequences"]),
            "splits": {
                s: len(WindowDataset(args.manifest, s))
                for s in ("train", "validation", "test")
            },
        }
    elif args.command == "esp-record":
        from .esp32 import record_serial

        result = record_serial(
            args.port,
            args.out,
            duration=args.seconds,
            baud=args.baud,
            consent_note=args.consent_note,
        )
    elif args.command == "esp-convert":
        from .esp32 import convert_csv

        result = convert_csv(args.source, args.out)
    elif args.command == "train":
        from .engine import train

        result = train(
            args.manifest, args.config, args.out, args.device, resume=args.resume
        )
    elif args.command == "evaluate":
        from .engine import evaluate

        result = evaluate(
            args.manifest, args.checkpoint, args.split, args.device, args.out
        )
    elif args.command == "export":
        from .replay import export_replay

        result = export_replay(
            args.manifest,
            args.checkpoint,
            args.out,
            sequence=args.sequence,
            split=args.split,
            device=args.device,
        )
    elif args.command == "infer-recording":
        from .replay import infer_recording

        result = infer_recording(
            args.recording,
            args.checkpoint,
            args.out,
            device=args.device,
            max_gap=args.max_gap,
        )
    else:
        from .server import serve

        serve(args.replay, port=args.port)
        return
    print(json.dumps(result, indent=2, allow_nan=False))
