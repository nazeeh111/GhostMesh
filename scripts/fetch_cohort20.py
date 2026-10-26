"""Fetch the exact released 20-person, 27-action MM-Fi cohort, sequentially.

Use --inventory or --dry-run before acquisition. Provider failures stop the
wrapper; it never changes download hosts or bypasses quota/access responses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/mmfi-raw"))
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Verify and reuse identical existing files after provider access returns",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print exact commands without network or filesystem writes",
    )
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args(argv)
    script = Path(__file__).with_name("fetch_mmfi.py")
    commands = []
    for environment, start in enumerate([1, 11, 21, 31], start=1):
        subjects = [f"S{i:02d}" for i in range(start, start + 5)]
        command = [
            sys.executable,
            str(script),
            "--env",
            f"E{environment:02d}",
            "--subjects",
            *subjects,
            "--actions",
            *[f"A{i:02d}" for i in range(1, 28)],
            "--out",
            str(args.out),
            "--inventory-dir",
            str(args.out / "provenance"),
            "--max-payload-mb",
            "2000",
        ]
        if args.cache:
            command.extend(["--cache", str(args.cache)])
        if args.inventory:
            command.append("--inventory")
        if args.resume:
            command.append("--resume")
        commands.append(command)
    if args.dry_run:
        print(
            json.dumps({"subjects": 20, "actions": 27, "commands": commands}, indent=2)
        )
        return
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
