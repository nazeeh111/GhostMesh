"""Write a byte-identity manifest for a completed experiment directory.

This inventories bytes, not scientific correctness or immutability. It neither
changes artifacts nor certifies data permissions, splits, claims or results.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import stat

SCOPE = (
    "Byte identity snapshot only. This is not scientific certification, an "
    "immutability guarantee, or evidence that experiment claims are correct."
)


def file_digest(path):
    value = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def seal(root, output="artifact-manifest-final.json"):
    root = Path(root)
    filename = Path(output)
    if (
        filename.name != str(output)
        or filename.name in ("", ".", "..")
        or filename.is_absolute()
    ):
        raise ValueError("Output must be a filename inside the experiment root")
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Root must be an existing ordinary directory")
    destination = root / filename
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    hashes = {}
    # Reject symlink files AND directories, including dangling symlinks. Never
    # traverse an artifact pointer outside the selected experiment directory.
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"Symlink is not an artifact byte snapshot: {path}")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError(f"Only regular files can be sealed: {path}")
        hashes[path.relative_to(root).as_posix()] = file_digest(path)
    manifest = {"scope": SCOPE, "sha256": hashes}
    # Exclusive creation also protects against a concurrently created manifest.
    with destination.open("x", encoding="utf-8") as target:
        target.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", default="artifact-manifest-final.json")
    args = parser.parse_args()
    result = seal(args.root, args.output)
    print(
        json.dumps(
            {
                "files": len(result["sha256"]),
                "manifest": str(args.root / args.output),
                "scope": SCOPE,
            }
        )
    )
