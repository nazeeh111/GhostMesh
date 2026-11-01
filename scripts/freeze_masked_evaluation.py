"""Seal the four completed, independently audited fits BEFORE query scoring.

The ledger pins the same fixed E02 development protocol used by the post-score
auditor. It is not a final-test registration and cannot be made retroactively.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import verify_masked_results as v  # noqa:E402


def freeze(root, output):
    root, output = Path(root), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    paths = {}
    registrations = {}
    for name, (directory, filename, kind, flag) in v.PROTOCOLS.items():
        directory = root / "work" / directory
        if (
            (directory / "results.json").exists()
            or (directory / "query-permutation.npy").exists()
            or any(directory.glob("*-query-prediction.npy"))
        ):
            raise ValueError("Query scoring already started: " + name)
        _, registration = v.contracts.sealed(
            directory, filename, kind, {"registration.json"}, flag
        )
        v.contracts.hashes(root, registration["source_sha256"])
        registrations[name] = registration
        paths["work/" + directory.name + "/" + filename] = v.contracts.digest(
            directory / filename
        )
    source_sha = paths["work/domain-masked-pretrain-e02-v1/source-ready.json"]
    full_sha = paths["work/domain-masked-fullpose-e02-v1/fullpose-source-ready.json"]
    if registrations["zero_body"]["config"]["seed"] != 2027:
        raise ValueError("This frozen protocol is seed2027 only")
    for name, filename in v.AUDITS.items():
        proof = v.contracts.read(root / filename)
        calibrated_sha = (
            None
            if name in {"source", "root"}
            else paths[
                "work/"
                + v.PROTOCOLS["calibrated_" + name][0]
                + "/"
                + v.PROTOCOLS["calibrated_" + name][1]
            ]
        )
        v.audit_receipt(root, name, proof, source_sha, full_sha, calibrated_sha)
        paths[filename] = v.contracts.digest(root / filename)
    ledger = dict(
        stage="frozen_before_first_E02_query_metrics",
        time_utc=datetime.now(timezone.utc).isoformat(),
        seed=2027,
        development_room="E02",
        held_query_metrics_already_computed=False,
        reference_people=["S11", "S12"],
        calibrated_query_people=[f"S{i:02}" for i in range(13, 21)],
        zero_query_people=[f"S{i:02}" for i in range(11, 21)],
        reference_labels=120,
        primary_zero="masked_temporal",
        primary_calibrated=v.PRIMARY,
        gates="Unchanged producer NJE/PCK/motion/shuffle gates, separate target spaces and cohorts. Source-development only, no E04 or final confirmation.",
        frozen_sha256=paths,
    )
    # Recheck every pinned input after preparation and before exclusive writing.
    v.contracts.hashes(root, paths)
    for directory, _, _, _ in v.PROTOCOLS.values():
        if any(
            (root / "work" / directory / name).exists()
            for name in ("results.json", "query-permutation.npy")
        ):
            raise ValueError("Scoring started while preparing the ledger")
    payload = json.dumps(ledger, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(payload)
    return v.contracts.digest(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(freeze(args.root, args.output))
