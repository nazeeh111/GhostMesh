"""SOURCE-only transfer test of a causal, unlabeled own-record startup reference.

The local implementation is frozen and imported. Every held record supplies its
OWN earliest available startup, never the training session's startup. Startup
reset/unknown-pose failure is a central limitation, not repaired with labels.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.domain_background_probe import digest, multiply  # noqa: E402
from scripts.domain_covariance_cached_probe import load_inputs  # noqa: E402
from scripts.domain_covariance_source_probe import (  # noqa: E402
    CONFIG as SOURCE_CONFIG,
    fold_masks,
    rff_transform,
    validate_records,
)
from scripts.domain_reference_covariance_probe import (  # noqa: E402
    CONFIG as REFERENCE_CONFIG,
    fit_readout,
    predict,
)
from scripts.domain_startup_covariance_probe import (  # noqa: E402
    FULL_CONFIG,
    fit_startups,
    require,
    save,
    score,
    transform_startups,
)

CONFIG = dict(
    **FULL_CONFIG,
    fit_arms=["startup_shape", "startup_shape_volume", "startup_frozen_raw"],
    feature_cache_dtype="float32, matching the frozen raw-current comparator; covariance decode and reference math float64/complex128",
    raw_current="Frozen raw-current RFF512 map/readout from independently verified room-reference experiment, with original16TRAIN/4VAL roles and no calibration-label input. Reapply to ALL10heldpeople. No query refit or8person-only comparison",
    temporal="Frozen full27-action temporal model predictions on EXACT same10person held indices. Original source-validation coordinate-MSE selector retained; no new held selection",
    registration="Frozen localrecipe/helper source, localpositive result+verification identities and sources bound before full feature build. ALL3fold selections before held scoring",
    limits_from_local="Continuous own-history local gate passed, but resetstress failed and rawcurrent/oldtemporal were stronger. Full own-record startups test unknown initial states; no startup transfer success assumed",
)


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def consume_manifest(directory, names):
    manifest = json.loads((directory / "artifact_manifest.json").read_text())
    result = {}
    for name in names:
        require(
            name in manifest and digest(directory / name) == manifest[name],
            "Frozen comparator bytes differ: " + name,
        )
        result[name] = manifest[name]
    return result


def transform_frozen(feature, state):
    """Apply a saved TRAIN-only RFF map, without fitting target statistics."""
    x = (np.asarray(feature, dtype="float64") - state["feature_mean"]) / state[
        "feature_scale"
    ]
    phi = np.sqrt(2 / len(state["offset"])) * np.cos(
        multiply(x, state["weight"]) + state["offset"]
    )
    return (phi - state["rff_mean"]) / state["rff_scale"]


def source_priors(target, endpoint, train, query):
    """Only TRAIN labels contribute. Query endpoint is clock-control metadata."""
    mean = target[train].mean(0)
    p = np.broadcast_to(mean, (len(query), 17, 2)).copy()
    clock = p.copy()
    missing = []
    for end in np.unique(endpoint[query]):
        source = train & (endpoint == end)
        use = endpoint[query] == end
        if source.any():
            clock[use] = target[source].mean(0)
        else:
            missing.append(int(end))
    return dict(source_body_mean=p, source_endpoint_clock=clock), missing


def permutation(record):
    result = np.arange(len(record))
    rng = np.random.default_rng(CONFIG["permutation_seed"])
    for name in np.unique(record):
        ids = np.flatnonzero(record == name)
        result[ids] = rng.permutation(ids)
    return result


def scores(p, target, record, people, action, perm):
    result = dict(
        aligned=score(p, target, record), shuffled=score(p[perm], target, record)
    )
    for name, values in (("person", people), ("action", action)):
        result["per_" + name] = {
            str(key): score(
                p[values == key], target[values == key], record[values == key]
            )
            for key in np.unique(values)
        }
    return result


def frozen_gate(results):
    decisions = []
    for room in ("E01", "E02", "E03"):
        methods = results[room]
        p = methods[CONFIG["primary"]]["aligned"]
        controls = (
            "source_body_mean",
            "source_endpoint_clock",
            "raw_current",
            "startup_frozen_raw",
        )
        nje = all(
            p["metrics"]["normalized_joint_error"]
            < methods[name]["aligned"]["metrics"]["normalized_joint_error"]
            for name in controls
        )
        pck = all(
            p["metrics"]["pck_curve"][threshold]
            >= methods[name]["aligned"]["metrics"]["pck_curve"][threshold]
            for name in ("source_body_mean", "source_endpoint_clock")
            for threshold in ("0.05", "0.2")
        )
        motion = (
            p["record_articulation_r2"] is not None and p["record_articulation_r2"] > 0
        )
        shuffle = (
            p["metrics"]["normalized_joint_error"]
            < methods[CONFIG["primary"]]["shuffled"]["metrics"][
                "normalized_joint_error"
            ]
        )
        decisions.append(
            dict(
                room=room,
                beats_all_four_controls_nje=bool(nje),
                pck_not_worse_than_mean_clock=bool(pck),
                positive_motion_r2=bool(motion),
                shuffle_worse_nje=bool(shuffle),
                passed=bool(nje and pck and motion and shuffle),
            )
        )
    return dict(passed=all(row["passed"] for row in decisions), per_room=decisions)


def load_frozen_comparators(data, old, raw):
    old_proof = json.loads((old / "verification.json").read_text())
    raw_proof = json.loads((raw / "verification.json").read_text())
    require(
        old_proof["total_records_feature_verified"] == 810
        and old_proof["retained_examples_verified"] == 26556,
        "Prior temporal cache proof differs",
    )
    require(
        raw_proof["feature_windows_verified"] == 26556
        and raw_proof[
            "all_current_reference_features_and_person_disjoint_queries_verified"
        ],
        "Raw-current proof differs",
    )
    names = ["cohort-indices.npz", "registration.json", "results.json"]
    for room in ("E01", "E02", "E03"):
        names.extend(
            room + "-raw_current_control-" + suffix
            for suffix in ("model.npz", "rff.npz")
        )
    raw_hashes = consume_manifest(raw, names)
    old_names = [
        room + "-congruence_dynamics-model-prediction.npz"
        for room in ("E01", "E02", "E03")
    ]
    old_hashes = consume_manifest(old, old_names + ["registration.json"])
    original_roles = load_npz(raw / "cohort-indices.npz")
    result, identities = {}, {}
    for room in ("E01", "E02", "E03"):
        train, val, held = fold_masks(data["room"], data["person"], room)
        ids = np.flatnonzero(held)
        require(
            np.array_equal(original_roles[room + "_train"], np.flatnonzero(train))
            and np.array_equal(
                original_roles[room + "_validation"], np.flatnonzero(val)
            ),
            "Frozen raw-current source roles differ",
        )
        temporal = load_npz(old / (room + "-congruence_dynamics-model-prediction.npz"))
        require(
            np.array_equal(temporal["held_indices"], ids), "Frozen temporal rows differ"
        )
        state = load_npz(raw / (room + "-raw_current_control-model.npz"))
        mapping = load_npz(raw / (room + "-raw_current_control-rff.npz"))
        require(
            mapping["feature_mean"].shape == (1368,)
            and mapping["weight"].shape == (1368, 512)
            and state["weights"].shape == (5, 512, 34),
            "Unexpected frozen raw-current dimensions",
        )
        require(
            np.all(np.isin(state["indices"], ids)),
            "Old eight-person rows escape current held cohort",
        )
        require(
            len(set(data["person"][ids])) == 10
            and set(data["action"][ids]) == {f"A{i:02}" for i in range(1, 28)},
            "Full held cohort missing people/actions",
        )
        result[room] = dict(
            mapping=mapping,
            state=state,
            temporal_prediction=temporal["pose"],
            indices=ids,
        )
        identities[room] = dict(
            train=int(train.sum()),
            validation=int(val.sum()),
            held=len(ids),
            held_people=sorted(set(data["person"][ids])),
            all27actions=True,
            raw_control_old_saved_rows=len(state["indices"]),
            raw_control_new_rows=len(ids),
            held_indices_sha256=sha256(ids.tobytes()).hexdigest(),
        )
    return (
        result,
        dict(
            raw_comparator=raw_hashes,
            temporal_comparator=old_hashes,
            raw_verification_sha256=digest(raw / "verification.json"),
            temporal_verification_sha256=digest(old / "verification.json"),
        ),
        identities,
    )


def input_setup(cache, old, raw, local):
    receipt, data, features = load_inputs(cache)
    validate_records(receipt["records"])
    require(
        len(data["pose"]) == CONFIG["expected_logical_rows"]
        and len(set(data["record"])) == 810,
        "Not the full810record/26556row SOURCE cohort",
    )
    require(
        np.max(abs(data["pose"][:, [11, 12]].mean(1))) < 1e-7,
        "Expected existing hip-relative teacher coordinates",
    )
    local_result = json.loads((local / "results.json").read_text())
    local_proof = json.loads((local / "verification.json").read_text())
    local_reg = json.loads((local / "registration.json").read_text())
    require(
        local_result["primary_continuation_passed"]
        and local_proof["sourceval_selections_and_gates_match"],
        "Local primary observability not verified",
    )
    require(
        not local_result["local_gates"]["reset_stress"]["passed"],
        "Preserved reset failure changed",
    )
    for name, expected in local_reg["source_sha256"].items():
        require(digest(ROOT / name) == expected, "Frozen local source changed")
    require(
        CONFIG["seed"] == SOURCE_CONFIG["seed"] == 2027
        and CONFIG["rff_dimensions"] == SOURCE_CONFIG["rff_dimensions"] == 512
        and CONFIG["alphas"] == REFERENCE_CONFIG["alphas"],
        "Imported fitting recipe differs",
    )
    comp, hashes, roles = load_frozen_comparators(data, old, raw)
    old_reg = json.loads((old / "registration.json").read_text())
    require(
        old_reg["feature_cache_receipt_sha256"] == digest(cache / "receipt.json"),
        "Frozen temporal feature cache differs",
    )
    raw_reg = json.loads((raw / "registration.json").read_text())
    require(
        raw_reg["feature_cache_receipt_sha256"] == digest(cache / "receipt.json"),
        "Frozen raw-current feature cache differs",
    )
    return receipt, data, features["untreated_covariance"], comp, hashes, roles


def run(cache, old, raw, local, out, preflight_only=False):
    if out.exists():
        raise FileExistsError(out)
    started = time.monotonic()
    receipt, data, untreated, comparators, control_hashes, roles = input_setup(
        cache, old, raw, local
    )
    paths = [
        Path(__file__),
        ROOT / "scripts/domain_startup_covariance_probe.py",
        ROOT / "scripts/domain_reference_covariance_probe.py",
        ROOT / "scripts/domain_covariance_cached_probe.py",
        ROOT / "scripts/domain_covariance_full_cache.py",
        ROOT / "scripts/domain_covariance_source_probe.py",
        ROOT / "scripts/domain_background_probe.py",
        ROOT / "ghostmesh/reference_covariance.py",
        ROOT / "ghostmesh/covariance_dynamics.py",
        ROOT / "ghostmesh/covariance_accelerated.py",
        ROOT / "ghostmesh/metrics.py",
    ]
    source = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    out.mkdir(parents=True)
    registration = dict(
        config=CONFIG,
        source_sha256=source,
        feature_cache_receipt_sha256=digest(cache / "receipt.json"),
        feature_cache_arrays_sha256=receipt["arrays_sha256"],
        metadata_sha256=digest(cache / "targets-and-metadata.npz"),
        comparator_artifact_sha256=control_hashes,
        local_results_sha256=digest(local / "results.json"),
        local_verification_sha256=digest(local / "verification.json"),
        local_manifest_sha256=digest(local / "artifact-manifest-final.json"),
        lineage="Mechanism motivated after failed cross-room physics/reference experiments and local continuous success/reset failure. SOURCE DEVELOPMENT, not untouched final evaluation",
    )
    save(out / "registration.json", registration)
    for path in paths:
        dest = out / "executed-source" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    save(out / "cohort-control-joins.json", roles)
    anchors = fit_startups(untreated, data["record"], data["end"])
    distribution = {}
    for anchor in anchors.values():
        distribution[str(anchor.end)] = distribution.get(str(anchor.end), 0) + 1
    identities = {
        name: dict(
            end=anchor.end,
            current_states=list(range(anchor.end - 7, anchor.end + 1)),
            validity_start=anchor.end - 39,
            nominal_seconds=anchor.end / 10,
            reset_required=True,
        )
        for name, anchor in anchors.items()
    }
    require(len(identities) == 810, "One own-record startup required for each record")
    save(
        out / "startup-identities.json",
        dict(
            records=identities,
            nominal_endpoint_distribution=distribution,
            scope="Known record resets; frame-index/10 is nominal recorded time, not measured live startup latency",
        ),
    )
    names = sorted(anchors)
    np.savez(
        out / "startup-references.npz",
        record=np.asarray(names),
        end=np.asarray([anchors[name].end for name in names]),
        anchors=np.stack([anchors[name].reference.anchors for name in names]),
        mean_log_volume=np.stack(
            [anchors[name].reference.mean_log_volume for name in names]
        ),
        raw_current=np.stack([anchors[name].raw_current for name in names]),
    )
    if preflight_only:
        save(
            out / "preflight.json",
            dict(
                fits_performed=0,
                full_feature_build=False,
                roles=roles,
                startup_endpoint_distribution=distribution,
                source_scope="E01-E03 only",
                seconds=time.monotonic() - started,
            ),
        )
        return
    n = len(data["pose"])
    features = {
        arm: np.lib.format.open_memmap(
            out / (arm + "-features.npy"), mode="w+", dtype="float32", shape=(n, width)
        )
        for arm, width in CONFIG["dimensions"].items()
    }
    for ri, name in enumerate(np.unique(data["record"])):
        ids = np.flatnonzero(data["record"] == name)
        built = transform_startups(
            untreated[ids], data["record"][ids], data["end"][ids], anchors
        )
        for arm in features:
            features[arm][ids] = built[arm]
        if ri % 90 == 0:
            print(
                json.dumps(
                    dict(
                        feature_records=ri + 1,
                        total_records=810,
                        seconds=time.monotonic() - started,
                    )
                ),
                flush=True,
            )
    for value in features.values():
        value.flush()
    feature_hashes = {
        arm + "-features.npy": digest(out / (arm + "-features.npy")) for arm in features
    }
    save(
        out / "feature-receipt.json",
        dict(
            feature_sha256=feature_hashes,
            logical_rows=n,
            additional_drops=0,
            raw_current_control="float32 unnormalized8currentstateHermitian coordinates",
            no_query_labels_for_feature_fit=True,
        ),
    )
    selections, designs, models = {}, {}, {}
    for room in ("E01", "E02", "E03"):
        train, val, held = fold_masks(data["room"], data["person"], room)
        selections[room] = {}
        for arm in CONFIG["fit_arms"]:
            z = rff_transform(
                features[arm], train, out / (room + "-" + arm + "-map.npz")
            )
            state, grid = fit_readout(z, data["pose"], train, val, data["room"])
            models[room, arm], designs[room, arm] = state, z
            np.savez(out / (room + "-" + arm + "-model.npz"), **state)
            selections[room][arm] = dict(
                alpha=CONFIG["alphas"][int(state["selected"])], grid=grid
            )
        raw_state = comparators[room]["state"]
        selections[room]["raw_current"] = dict(
            frozen=True,
            alpha=CONFIG["alphas"][int(raw_state["selected"])],
            source_roles_exact=True,
        )
    # No held target score is computed until every room/arm's choice is frozen.
    save(out / "ALL-FOLDS-FROZEN-SELECTIONS.json", selections)
    results, raw_readbacks = {}, {}
    for room in ("E01", "E02", "E03"):
        train, val, held = fold_masks(data["room"], data["person"], room)
        ids = np.flatnonzero(held)
        perm = permutation(data["record"][ids])
        np.save(out / (room + "-permutation.npy"), perm, allow_pickle=False)
        predictions, missing = source_priors(data["pose"], data["end"], train, ids)
        predictions["frozen_temporal"] = comparators[room]["temporal_prediction"]
        z = transform_frozen(features["raw_current"][ids], comparators[room]["mapping"])
        predictions["raw_current"] = predict(z, comparators[room]["state"])
        oldids = comparators[room]["state"]["indices"]
        position = np.searchsorted(ids, oldids)
        require(
            np.array_equal(ids[position], oldids),
            "Frozen eight-person raw-current subset rows differ",
        )
        error = float(
            np.max(
                abs(
                    predictions["raw_current"][position]
                    - comparators[room]["state"]["prediction"]
                )
            )
        )
        require(
            error < 1e-9,
            "Frozen raw-current readback differs on original eight-person subset",
        )
        raw_readbacks[room] = dict(
            original_eight_person_prediction_max_difference=error,
            full_ten_person_rows=len(ids),
            original_subset_rows=len(oldids),
        )
        for arm in CONFIG["fit_arms"]:
            predictions[arm] = predict(designs[room, arm][held], models[room, arm])
        results[room] = {}
        for name, p in predictions.items():
            np.savez(
                out / (room + "-" + name + "-predictions.npz"),
                prediction=p,
                indices=ids,
            )
            results[room][name] = scores(
                p,
                data["pose"][ids],
                data["record"][ids],
                data["person"][ids],
                data["action"][ids],
                perm,
            )
        results[room]["clock_fallback_endpoints"] = missing
    gate = frozen_gate(results)
    for name, expected in source.items():
        require(digest(ROOT / name) == expected, "Source changed during fit")
    save(out / "frozen-raw-current-readback.json", raw_readbacks)
    save(
        out / "results.json",
        dict(
            config=CONFIG,
            results=results,
            gate=gate,
            seconds=time.monotonic() - started,
            scope="Full SOURCE development only; local reset failure retained; no E04/final efficacy/fullpose/3D claim",
        ),
    )
    print(json.dumps(dict(gate=gate, seconds=time.monotonic() - started)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path, default=ROOT / "work/domain-covariance-full-cache-v1"
    )
    parser.add_argument(
        "--old", type=Path, default=ROOT / "work/domain-covariance-full27-v1"
    )
    parser.add_argument(
        "--raw", type=Path, default=ROOT / "work/domain-reference-covariance-v1"
    )
    parser.add_argument(
        "--local", type=Path, default=ROOT / "work/domain-startup-positive-v1"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    run(args.cache, args.old, args.raw, args.local, args.out, args.preflight_only)
