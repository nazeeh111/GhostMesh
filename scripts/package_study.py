#!/usr/bin/env python3
"""Package a completed, validated cohort20 study without importing ML libraries.

All input JSON and checkpoint bytes are copied exactly. No model evaluation,
training, data acquisition, or publication is performed.
"""

from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import importlib.util
import io
import json
import os
import re
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "ghostmesh-study-v1"
EXTRAS = (
    "scripts/report_results.py",
    "scripts/package_study.py",
    "scripts/package_cohort.py",
    "scripts/qualitative.py",
    "scripts/run_study.py",
    "scripts/run_signal_controls.py",
    "ghostmesh/controls.py",
    "configs/signal-controls-v1.json",
    "docs/SIGNAL_CONTROLS.md",
    "docs/REPRODUCE.md",
    "results/README.md",
    "results/signal-controls-v1/plan.json",
    "results/signal-controls-v1/results.json",
    "results/signal-controls-v1/results.csv",
    "results/signal-controls-v1/README.md",
    "results/qualitative-v1/cases.json",
    "results/qualitative-v1/qualitative.png",
    "results/qualitative-v1/qualitative.svg",
    "results/qualitative-v1/render_manifest.json",
    "requirements-plots.txt",
    "LICENSE",
    "NOTICE.md",
    "data/LICENSE.md",
    "docs/cohort-provenance.json",
    "docs/RESULTS_FORMAT.md",
    "docs/EXPERIMENTS.md",
    "docs/STUDY_BUNDLE.md",
    "docs/MODEL_CARD.md",
    "docs/DATA_ACCESS.md",
    "docs/REFERENCES.md",
    "docs/VERIFICATION.md",
    "docs/verification-host.json",
    "docs/access-status.json",
    "docs/data-provenance.json",
    "docs/mmfi-source.json",
    "docs/research-development.json",
)
SCOPE = (
    "Exact frozen manifests, selected checkpoints and completed study receipts; "
    "sufficient to validate and re-render stored results with Python and Matplotlib. "
    "Does not include raw/prepared CSI, raw RGB, prepared NPZ arrays, window caches, "
    "latest.pt, optimizer/RNG recovery states, or training dependencies. "
    "Re-rendering is not independent re-evaluation or retraining."
)


def digest(path):
    result = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def require_new(path):
    # lexists also rejects dangling symlinks instead of following them.
    if os.path.lexists(path):
        raise FileExistsError(f"Refusing to overwrite existing path: {path}")


def inside_project(path):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"Study input lies outside project root: {path}")
    return resolved


def load_reporter():
    path = ROOT / "scripts" / "report_results.py"
    checksum = digest(path)
    spec = importlib.util.spec_from_file_location("ghostmesh_study_reporter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if digest(path) != checksum:
        raise ValueError("Reporter changed during import")
    return module, checksum


def instructions(suite_relative, provenance_relative):
    return f"""# GhostMesh frozen cohort20 study bundle

This bundle contains completed dataset-experiment receipts and exact selected
checkpoint/manifests. It preserves 27 actions, 20 people and four environments.
All actions were seen in training; E04 transfer confounds room and person.
Stored scores and paired intervals can be re-rendered without raw CSI or
PyTorch. This does not independently reproduce model predictions or training.

Extract into a new directory and enter the bundle root:

```sh
mkdir ghostmesh-study-extracted
tar -xzf ghostmesh-study-v1.tar.gz -C ghostmesh-study-extracted
cd ghostmesh-study-extracted/{PREFIX}
```

Verify every manifest-listed file before rendering:

```sh
python3 - <<'VERIFY'
from hashlib import sha256
import json
from pathlib import Path
manifest = json.loads(Path('bundle_manifest.json').read_text())
for name, record in manifest['members'].items():
    path = Path(name)
    assert not path.is_absolute() and '..' not in path.parts
    content = path.read_bytes()
    assert len(content) == record['size'], name
    assert sha256(content).hexdigest() == record['sha256'], name
print('All listed study bytes match the bundle manifest.')
VERIFY
```

Create a separate plotting environment; only Matplotlib is required:

```sh
uv venv --python 3.12 .report-venv
uv pip install --python .report-venv/bin/python -r requirements-plots.txt
MPLCONFIGDIR=/tmp/ghostmesh-mpl-cache .report-venv/bin/python scripts/report_results.py \\
  --suite {suite_relative} \\
  --out results/cohort20-v1 \\
  --provenance {provenance_relative}
```

The results directory must be new. The renderer validates frozen source,
manifest, config, checkpoint and report bindings without importing PyTorch or
loading checkpoint contents. Inspect `results/cohort20-v1/results.md`, raw CSV
and JSON tables, SVG/PNG figures and `renderer_manifest.json`.

Paired bootstrap intervals quantify held-out sampling for fixed checkpoints,
not training-seed variability. Absolute PCK bars have zero-based axes and no
borrowed paired-change error bars. See `docs/EXPERIMENTS.md` and
`docs/RESULTS_FORMAT.md` for protocol and metric definitions.

`bundle_manifest.json` lists SHA-256 and byte size for every other archive
member. It excludes itself to avoid a circular hash. Obtain the archive's
SHA-256 from the release receipt to establish its identity; internal hashes
alone are not an authenticity signature. Compression/library versions can
affect archive bytes across platforms even with canonical member metadata.

Scope: {SCOPE}

Original software is MIT licensed. MM-Fi-derived artifacts retain CC BY-NC
4.0 terms. Read `LICENSE`, `NOTICE.md` and `data/LICENSE.md` before reuse.
"""


def write_archive(out, entries, prefix=PREFIX, compresslevel=9, max_bytes=None):
    """Write sorted canonical regular files, retaining any failed .partial file.

    Each entry has source (Path) or data (bytes), sha256 and size. This small
    helper permits a software-only archive layout check without fake studies.
    """
    if not isinstance(prefix, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", prefix
    ):
        raise ValueError("Unsafe archive prefix")
    if not isinstance(compresslevel, int) or not 0 <= compresslevel <= 9:
        raise ValueError("gzip compresslevel must be between 0 and 9")
    out = Path(out).absolute()
    partial = out.with_name(out.name + ".partial")
    require_new(out)
    require_new(partial)
    for name in entries:
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or name != relative.as_posix()
        ):
            raise ValueError(f"Unsafe archive member: {name}")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation refuses concurrent builders and existing failed partials.
    with partial.open("xb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=compresslevel
        ) as compressed:
            with tarfile.open(
                fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT
            ) as archive:
                for name, entry in sorted(entries.items()):
                    content = (
                        entry["source"].read_bytes()
                        if "source" in entry
                        else entry["data"]
                    )
                    if (
                        len(content) != entry["size"]
                        or sha256(content).hexdigest() != entry["sha256"]
                    ):
                        raise ValueError(f"Input changed before archival: {name}")
                    member = tarfile.TarInfo(f"{prefix}/{name}")
                    member.size = len(content)
                    member.mtime = member.uid = member.gid = 0
                    member.mode = 0o644
                    member.uname = member.gname = ""
                    archive.addfile(member, io.BytesIO(content))
        raw.flush()
        os.fsync(raw.fileno())
    # Check sources again so a changed live input cannot silently become a bundle.
    for name, entry in entries.items():
        if "source" in entry and digest(entry["source"]) != entry["sha256"]:
            raise ValueError(f"Input changed during archival: {name}")
    if max_bytes is not None and partial.stat().st_size >= max_bytes:
        raise ValueError(
            f"Archive must be smaller than {max_bytes} bytes; retained .partial"
        )
    # An exclusive hard link publishes the finished file without replacing a race winner.
    os.link(partial, out)
    partial.unlink()
    return out


def add_run_receipts(validated, inputs):
    """Bind exact standalone run receipts without introducing an identity schema."""
    suite = validated["suite"]
    for row in validated["registration"]["variants"]:
        name = row["name"]
        for filename in ("history.json", "config.json", "run_identity.json"):
            path = inside_project(suite / name / filename)
            content = path.read_bytes()
            if filename == "history.json":
                saved = json.loads(content)
                training = validated["training"][name]
                if (
                    not isinstance(saved, dict)
                    or "history" not in training
                    or saved.get("epochs") != training["history"]
                ):
                    raise ValueError(
                        f"{name}: standalone history differs from training report"
                    )
            elif filename == "config.json":
                if json.loads(content) != row["config"]:
                    raise ValueError(
                        f"{name}: standalone config differs from preregistration"
                    )
            # run_identity.json is retained and hashed as a receipt. The suite
            # validator already enforces the operative frozen-run bindings.
            inputs[str(path)] = sha256(content).hexdigest()


def package_study(suite, out, provenance):
    out = Path(out).absolute()
    require_new(out)
    require_new(out.with_name(out.name + ".partial"))
    suite, provenance = inside_project(suite), inside_project(provenance)
    reporter, reporter_hash = load_reporter()
    validated = reporter.load_suite(suite, provenance)
    inputs = dict(validated["inputs_sha256"])
    add_run_receipts(validated, inputs)
    for name in EXTRAS:
        source = inside_project(ROOT / name)
        if not source.is_file():
            raise FileNotFoundError(f"Missing bundle support file: {source}")
        inputs[str(source)] = digest(source)
    if inputs[str(ROOT / "scripts" / "report_results.py")] != reporter_hash:
        raise ValueError("Reporter changed after study validation")
    entries = {}
    for name, expected_hash in inputs.items():
        source = inside_project(name)
        relative = source.relative_to(ROOT).as_posix()
        if not source.is_file() or digest(source) != expected_hash:
            raise ValueError(f"Validated input changed: {relative}")
        entries[relative] = {
            "source": source,
            "sha256": expected_hash,
            "size": source.stat().st_size,
        }
    readme = instructions(
        suite.relative_to(ROOT).as_posix(), provenance.relative_to(ROOT).as_posix()
    ).encode()
    entries["README_STUDY.md"] = {
        "data": readme,
        "sha256": sha256(readme).hexdigest(),
        "size": len(readme),
    }
    manifest = {
        "schema_version": 1,
        "purpose": "dataset_experiment",
        "archive_prefix": PREFIX + "/",
        "suite": suite.relative_to(ROOT).as_posix(),
        "provenance": provenance.relative_to(ROOT).as_posix(),
        "scope": SCOPE,
        "member_metadata": {
            "mtime": 0,
            "mode": "0644",
            "uid": 0,
            "gid": 0,
            "gzip_mtime": 0,
        },
        "members": {
            name: {"sha256": row["sha256"], "size": row["size"]}
            for name, row in sorted(entries.items())
        },
        "self_hash_note": "This manifest excludes itself to avoid a circular hash; use the separate archive SHA-256 for bundle identity.",
    }
    content = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    entries["bundle_manifest.json"] = {
        "data": content,
        "sha256": sha256(content).hexdigest(),
        "size": len(content),
    }
    return write_archive(out, entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="New .tar.gz path; existing outputs and .partial files are refused",
    )
    parser.add_argument(
        "--provenance", type=Path, default=ROOT / "docs" / "cohort-provenance.json"
    )
    args = parser.parse_args()
    try:
        archive = package_study(args.suite, args.out, args.provenance)
    except (ValueError, OSError, KeyError, tarfile.TarError) as error:
        parser.exit(2, f"Study bundle refused: {error}\n")
    print(
        json.dumps(
            {
                "archive": str(archive),
                "sha256": digest(archive),
                "size": archive.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
