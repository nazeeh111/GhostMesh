"""Separate full-image calibration adapter for frozen CSI body and root heads.

The reviewed articulation-only calibration remains unchanged. This experiment
uses the same 120 supplied17x2 reference poses, now including image translation.
No query hip, annotation, statistic or camera calibration enters prediction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import domain_masked_calibration_probe as calibration  # noqa:E402
from scripts import domain_masked_fullpose_probe as fullpose  # noqa:E402
from scripts.domain_covariance_full_cache import verify_hashes  # noqa:E402
from scripts.domain_reference_covariance_probe import query_indices, reference_indices  # noqa:E402

ARMS = fullpose.ARMS
PRIMARY = calibration.PRIMARY
METHODS = ("residual_krr", "absolute_krr", "permuted_residual_krr", "constant_residual")
CONFIG = dict(
    calibration.CONFIG,
    target_mode="full_image_2d",
    coordinates="Unclipped supplied full-image17joint2D coordinates divided by640, including predicted hip position. No query hip placement, camera calibration,3D or efficacy claim",
    controls="Six full-image arms: five frozen encoders and raw-current; each same120reference labels/CV grid. Frozen temporal body-only arm is outside this full-image comparison. Full-image source/ref means, source clock and no-RF reference-mean mixture",
    gate="Primary beta>0, lowerfullNJE/PCK.05/.2 notworse versus all33matchedcontrols; positivefullrecordmotion ANDbodyrecordmotion, queryshufflegivesworseNJE. Rootmotionseparatelyreported, notassertedfromstaticroot",
)


def inputs(source, cache, raw, full_source, expected):
    # The complete source-only root/body/encoder contract is checked before any
    # raw pose arrays are decoded by fullpose.inputs below.
    registration = fullpose.frozen_contract(full_source, source, cache, raw, expected)
    data, target, raw_indices, train, validation, held, cached = fullpose.inputs(
        source, cache, raw, registration["source_ready_sha256"]
    )
    with np.load(full_source / "canonical-metadata.npz", allow_pickle=False) as saved:
        for key in ("record", "person", "action", "room", "end"):
            np.testing.assert_array_equal(saved[key], data[key])
    with np.load(full_source / "roles.npz", allow_pickle=False) as saved:
        for key, mask in (("train", train), ("validation", validation), ("held", held)):
            np.testing.assert_array_equal(saved[key], mask)
    np.testing.assert_array_equal(
        np.load(full_source / "raw-frame-indices.npy", allow_pickle=False), raw_indices
    )
    np.testing.assert_array_equal(
        np.load(full_source / "full-targets.npy", allow_pickle=False), target
    )
    full_data = dict(data, pose=target)
    reference, query = reference_indices(data, "E02"), query_indices(data, "E02")
    calibration.validate_roles(data, reference, query)
    features, bases = {}, {}
    for arm in ARMS:
        if arm == "raw_current":
            features[arm] = calibration.Features(
                cached["untreated_covariance"], current=True
            )
        else:
            embedding = np.load(
                source / (arm + "-embedding.npy"), mmap_mode="r", allow_pickle=False
            )
            if embedding.shape != (len(target), 512):
                raise ValueError("Source embedding shape differs")
            features[arm] = calibration.Features(embedding)
        prediction = np.load(
            full_source / (arm + "-full-prediction.npy"),
            mmap_mode="r",
            allow_pickle=False,
        )
        fullpose.finite(prediction, target.shape)
        bases[arm] = prediction
    return full_data, train, reference, query, features, bases


def source_paths():
    return [
        Path(__file__),
        ROOT / "tests/test_masked_fullimage_calibration_probe.py",
        ROOT / "scripts/domain_masked_calibration_probe.py",
        ROOT / "scripts/domain_masked_fullpose_probe.py",
        ROOT / "ghostmesh/reference_kernel.py",
        ROOT / "ghostmesh/metrics.py",
        ROOT / "scripts/domain_reference_covariance_probe.py",
        ROOT / "scripts/domain_background_probe.py",
        ROOT / "scripts/domain_covariance_cached_probe.py",
        ROOT / "scripts/domain_covariance_full_cache.py",
        ROOT / "ghostmesh/reference_covariance.py",
        ROOT / "ghostmesh/covariance_dynamics.py",
    ]


def fit(source, cache, raw, full_source, out, expected):
    if out.exists():
        raise FileExistsError(out)
    if any(
        (directory / filename).exists()
        for directory in (source, full_source)
        for filename in ("results.json", "query-permutation.npy")
    ):
        raise ValueError(
            "Freeze all calibration choices before new zero-shot query scores"
        )
    data, train, reference, query, features, bases = inputs(
        source, cache, raw, full_source, expected
    )
    source_mean = data["pose"][train].reshape(train.sum(), 34).mean(0)
    out.mkdir(parents=True)
    sources = source_paths()
    registration = dict(
        config=CONFIG,
        fullpose_source_ready_sha256=expected,
        source_ready_sha256=calibration.digest(source / "source-ready.json"),
        cache_receipt_sha256=calibration.digest(cache / "receipt.json"),
        raw_receipt_sha256=calibration.digest(raw / "receipt.json"),
        source_sha256={
            str(path.relative_to(ROOT)): calibration.digest(path) for path in sources
        },
        reference_rows=len(reference),
        query_rows=len(query),
        reference_people=sorted(set(data["person"][reference])),
        query_people=sorted(set(data["person"][query])),
        query_metrics_computed=False,
    )
    calibration.save_json(out / "registration.json", registration)
    for path in sources:
        destination = out / "executed-source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    states, choices = calibration.fit_arrays(
        features, bases, data, reference, query, source_mean
    )
    for name, state in states.items():
        calibration.save_npz(out / (name.replace("/", "--") + ".npz"), **state)
    ends = np.unique(data["end"][train])
    clock = np.stack(
        [data["pose"][train & (data["end"] == end)].mean(0) for end in ends]
    )
    calibration.save_npz(
        out / "roles-and-priors.npz",
        reference=reference,
        query=query,
        source_mean=source_mean,
        reference_mean=data["pose"][reference].mean(0),
        clock_ends=ends,
        clock_poses=clock,
    )
    calibration.save_json(out / "frozen-reference-choices.json", choices)
    calibration.save_json(
        out / "fullimage-calibration-ready.json",
        dict(
            kind="masked-csi-fullimage-calibration-ready",
            query_metrics_computed=False,
            artifacts_sha256={
                str(path.relative_to(out)): calibration.digest(path)
                for path in sorted(out.rglob("*"))
                if path.is_file()
            },
        ),
    )
    return out


def fitted_contract(out):
    ready = json.loads((out / "fullimage-calibration-ready.json").read_text())
    required = {
        "registration.json",
        "roles-and-priors.npz",
        "frozen-reference-choices.json",
        "reference_mean_mixture.npz",
    }
    required.update(arm + "--" + method + ".npz" for arm in ARMS for method in METHODS)
    if (
        ready["kind"] != "masked-csi-fullimage-calibration-ready"
        or ready["query_metrics_computed"] is not False
        or not required <= set(ready["artifacts_sha256"])
    ):
        raise ValueError("Incomplete full-image calibration seal")
    verify_hashes(out, ready["artifacts_sha256"])
    registration = json.loads((out / "registration.json").read_text())
    verify_hashes(ROOT, registration["source_sha256"])
    if registration["config"]["target_mode"] != "full_image_2d":
        raise ValueError("Full-image target mode differs")
    return registration


def score(source, cache, raw, full_source, out):
    if (out / "results.json").exists() or (out / "query-permutation.npy").exists():
        raise FileExistsError("Full-image calibration query scoring already started")
    registration = fitted_contract(out)
    if (
        registration["source_ready_sha256"]
        != calibration.digest(source / "source-ready.json")
        or registration["cache_receipt_sha256"]
        != calibration.digest(cache / "receipt.json")
        or registration["raw_receipt_sha256"]
        != calibration.digest(raw / "receipt.json")
    ):
        raise ValueError("Frozen source/full-image calibration lineage differs")
    data, _, reference, query, features, bases = inputs(
        source, cache, raw, full_source, registration["fullpose_source_ready_sha256"]
    )
    with np.load(out / "roles-and-priors.npz", allow_pickle=False) as saved:
        priors = {key: saved[key] for key in saved.files}
    np.testing.assert_array_equal(reference, priors["reference"])
    np.testing.assert_array_equal(query, priors["query"])
    choices = json.loads((out / "frozen-reference-choices.json").read_text())
    predictions = {}
    for arm in ARMS:
        base = np.asarray(bases[arm][query])
        predictions[arm + "/zero_shot"] = base
        x = features[arm][query]
        for method in METHODS:
            tag = arm + "/" + method
            with np.load(
                out / (tag.replace("/", "--") + ".npz"), allow_pickle=False
            ) as state:
                predictions[tag] = calibration.predict_reference_model(
                    state, x, base.reshape(len(query), 34)
                ).reshape(-1, 17, 2)
    predictions["source_mean"] = np.broadcast_to(
        priors["source_mean"].reshape(17, 2), (len(query), 17, 2)
    )
    predictions["reference_mean"] = np.broadcast_to(
        priors["reference_mean"], (len(query), 17, 2)
    )
    with np.load(out / "reference_mean_mixture.npz", allow_pickle=False) as state:
        predictions["reference_mean_mixture"] = calibration.predict_reference_model(
            state,
            np.ones((len(query), 1)),
            predictions["source_mean"].reshape(len(query), 34),
        ).reshape(-1, 17, 2)
    clock = {
        int(end): pose for end, pose in zip(priors["clock_ends"], priors["clock_poses"])
    }
    if not set(data["end"][query]) <= set(clock):
        raise ValueError("Query endpoint absent from frozen source clock")
    predictions["source_endpoint_clock"] = np.stack(
        [clock[int(end)] for end in data["end"][query]]
    )
    perm = calibration.permutation(data["record"][query])
    calibration.save_array(out / "query-permutation.npy", perm)
    for name, prediction in predictions.items():
        calibration.save_array(
            out / (name.replace("/", "--") + "-query-prediction.npy"), prediction
        )
    results = {
        name: fullpose.scores(prediction, data["pose"], data, query, perm)
        for name, prediction in predictions.items()
    }
    primary = results[PRIMARY]["aligned"]
    controls = [value["aligned"] for name, value in results.items() if name != PRIMARY]
    gate = dict(
        beta_positive=choices[PRIMARY]["selected"]["beta"] > 0,
        full_nje_strictly_better_all_controls=all(
            primary["normalized_joint_error"] < value["normalized_joint_error"]
            for value in controls
        ),
        full_pck_not_worse_all_controls=all(
            primary["pck_curve"][threshold] >= value["pck_curve"][threshold]
            for value in controls
            for threshold in ("0.05", "0.2")
        ),
        full_pose_motion_positive=primary["record_centered_full_image_r2"] is not None
        and primary["record_centered_full_image_r2"] > 0,
        body_motion_positive=primary["record_centered_body_r2"] is not None
        and primary["record_centered_body_r2"] > 0,
        query_shuffle_worse=primary["normalized_joint_error"]
        < results[PRIMARY]["shuffled"]["normalized_joint_error"],
    )
    reversals = {
        field: {
            key: [
                name
                for name, method in results.items()
                if name != PRIMARY
                and value["normalized_joint_error"]
                >= method["per_" + field][key]["normalized_joint_error"]
            ]
            for key, value in results[PRIMARY]["per_" + field].items()
        }
        for field in ("person", "action")
    }
    calibration.save_json(
        out / "results.json",
        dict(
            registration=registration,
            fullimage_calibration_ready_sha256=calibration.digest(
                out / "fullimage-calibration-ready.json"
            ),
            frozen_choices_sha256=calibration.digest(
                out / "frozen-reference-choices.json"
            ),
            methods=results,
            reversals=reversals,
            gate=gate,
            passes=all(gate.values()),
        ),
    )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fit", "score"))
    for name in ("source", "cache", "raw", "fullpose-source-ready", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--fullpose-source-ready-sha256")
    args = parser.parse_args()
    if args.mode == "fit":
        if not args.fullpose_source_ready_sha256:
            parser.error("fit requires the reviewed --fullpose-source-ready-sha256")
        fit(
            args.source,
            args.cache,
            args.raw,
            args.fullpose_source_ready,
            args.out,
            args.fullpose_source_ready_sha256,
        )
    else:
        score(args.source, args.cache, args.raw, args.fullpose_source_ready, args.out)


if __name__ == "__main__":
    main()
