"""Fetch full40 MM-Fi CSI/provided2D labels sequentially, honoring provider access.

Only the existing official-source fetcher performs network requests. A failure
stops this wrapper immediately. No alternate host or quota workaround exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def commands(args):
    result = []
    for room in range(1, 5):
        command = [
            sys.executable,
            str(Path(__file__).with_name("fetch_mmfi.py")),
            "--env",
            f"E{room:02d}",
            "--subjects",
            *[f"S{i:02d}" for i in range(10 * (room - 1) + 1, 10 * room + 1)],
            "--actions",
            *[f"A{i:02d}" for i in range(1, 28)],
            "--out",
            str(args.out),
            "--inventory-dir",
            str(args.inventories),
            "--max-payload-mb",
            "4000",
        ]
        if args.cache:
            command.extend(["--cache", str(args.cache)])
        if args.inventory:
            command.append("--inventory")
        if args.resume:
            command.append("--resume")
        result.append(command)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("work/data/mmfi"))
    parser.add_argument(
        "--inventories", type=Path, default=Path("work/references/full-v2")
    )
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    selected = commands(args)
    if args.dry_run:
        print(
            json.dumps({"subjects": 40, "actions": 27, "commands": selected}, indent=2)
        )
        return
    for command in selected:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
