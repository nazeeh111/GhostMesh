"""Source-only causal history pooling for a conditioned displacement decoder.

This follows the unsuccessful single-frame conditioned screen. The only input
change is an unweighted mean of eight already causal frozen frame embeddings.
No new encoder, target-room read, query statistics or complete-pose claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import domain_conditioned_dynamics_probe as p  # noqa: E402

CONFIG = {
    **p.CONFIG,
    "input": "Mean of eight causal64D frame embeddings in each frozen512D history, rather than last frame. Every constituent timestamp is <=endpoint. No query normalization",
    "rationale": "After the original conditional screen failed, a separate source-only fixed-effects diagnostic found greater lag8 movement R2 in full histories than last-frame features. Test fixed pooling before any new room evaluation",
    "temporal_limit": "Pooling spans seven nominal frame intervals and smooths motion. At10Hz its center is0.35seconds before the endpoint. Predict ENDPOINT displacement, never shift labels to improve scores",
    "protocol": "Same original126candidates/threeencoders/sourcepeople/lag8targets/numerics/negativecontrols and gate. A distinct post-diagnostic development experiment, not a preregistered original hypothesis",
}


class PooledHistory:
    """Lazy source-row-only view. Never materialize unrelated E02 rows."""

    def __init__(self, feature):
        self.feature = feature

    def __getitem__(self, key):
        ids, columns = key
        if columns != slice(-64, None):
            raise ValueError("Expected the declared64D view")
        history = p.real_finite(self.feature[ids], "source causal history")
        if history.shape != (len(ids), 512):
            raise ValueError("Eight64D frame embeddings required")
        return p.real_finite(history.reshape(-1, 8, 64).mean(1), "pooled history")


def run(study, out):
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    start = time.monotonic()
    p.source_contract(study)
    with np.load(study / "targets-and-metadata.npz", allow_pickle=False) as stored:
        metadata = {
            key: stored[key] for key in ("room", "person", "record", "action", "end")
        }
    train, validation = p.source_roles(metadata)
    with np.load(study / "roles.npz", allow_pickle=False) as stored:
        for name, mask in (("train", train), ("validation", validation)):
            if stored[name].dtype != bool:
                raise ValueError("Frozen source roles must be Boolean")
            np.testing.assert_array_equal(stored[name], mask)
    target = p.stored_npz_array(study / "targets-and-metadata.npz", "pose")
    if target.shape != (len(train), 17, 2) or target.dtype.kind not in "fiu":
        raise ValueError("Unexpected lazy pose schema")
    sources = [
        Path(__file__),
        ROOT / "tests/test_pooled_dynamics_probe.py",
        ROOT / "scripts/domain_conditioned_dynamics_probe.py",
        ROOT / "scripts/domain_masked_motion_diagnostic.py",
        ROOT / "scripts/domain_covariance_full_cache.py",
        ROOT / "scripts/domain_background_probe.py",
    ]
    registration = dict(
        kind="source-only-pooled-conditioned-displacement",
        config=CONFIG,
        source_sha256={str(x.relative_to(ROOT)): p.digest(x) for x in sources},
        held_room_numeric_values_read=False,
        numpy=np.__version__,
    )
    out.mkdir(parents=True)
    p.save_json(out / "registration.json", registration)
    for source in sources:
        destination = out / "executed-source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    results = {}
    for arm in p.ARMS:
        feature = np.load(
            study / (arm + "-embedding.npy"), mmap_mode="r", allow_pickle=False
        )
        if feature.shape != (len(train), 512) or feature.dtype.kind not in "fiu":
            raise ValueError("Unexpected real frozen history schema")
        fitted = p.fit_source_models(
            PooledHistory(feature), target, metadata, train, validation
        )
        p.save_npz(out / (arm + "-TRAIN-map.npz"), **fitted["mapping"])
        for role in ("train", "validation"):
            pair = fitted[role + "_pairs"]
            p.save_npz(
                out / (arm + "-" + role + "-pairs.npz"),
                current_global=pair["current_global"],
                previous_global=pair["previous_global"],
                delta=pair["delta"],
                context=pair["context"],
                target=pair["target"],
                context_permutation=fitted["permutations"][role],
                **pair["metadata"],
            )
        for (tag, alpha), model in fitted["models"].items():
            p.save_npz(
                out / f"{arm}-{tag}-alpha{p.ALPHAS.index(alpha)}-model.npz", **model
            )
        results[arm] = dict(
            all_configs=fitted["reports"],
            zero_displacement=fitted["zero"],
            selection=p.selected_and_gate(fitted["reports"]),
            train_pair_count=len(fitted["train_pairs"]["target"]),
            validation_pair_count=len(fitted["validation_pairs"]["target"]),
        )
        p.save_json(out / (arm + "-source-results.json"), results[arm])
        print(
            json.dumps(dict(arm=arm, selection=results[arm]["selection"])), flush=True
        )
    p.verify_hashes(ROOT, registration["source_sha256"])
    p.save_json(
        out / "results.json",
        dict(
            registration=registration,
            arms=results,
            primary_feasibility=results["masked_temporal"]["selection"],
            seconds=time.monotonic() - start,
            scope="Source-only endpoint displacement; no newE02metric, completepose or3D claim",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.study, args.out)


if __name__ == "__main__":
    main()
