"""SOURCE-only predictability of rigid/scale-invariant annotation geometry.

The separate17x3 annotation point order is retained. This is a136-distance
auxiliary-target feasibility screen, not a COCO remap, 3D reconstruction,
complete2Dpose model or newly scored E02 experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import zlib

import numpy as np
from scipy import __version__ as SCIPY_VERSION

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.pose_shape import canonical_distances  # noqa:E402
from scripts import domain_conditioned_dynamics_probe as p  # noqa:E402
from scripts import domain_masked_motion_diagnostic as linear  # noqa:E402
from scripts import verify_masked_results as features  # noqa:E402

ARMS = ("masked_temporal", "random_frozen", "supervised", "raw_current")
CONFIG = dict(
    arms=ARMS,
    alphas=linear.ALPHAS,
    seed=2027,
    target="136ordered pairwise distances in supplied17x3point arrays, divided by per-frameRMSradius after translation removal. Rotation, reflection and isotropic scale invariant. No COCO remap, semanticjointname or physicalunit assumption",
    target_limit="Orientation/chirality unavailable from distances alone. Label-sidegeometry is not an inference input and does not establish 3D sensing",
    feature="Three frozen encoders'full512causalhistory; raw-current512RFFfromprevioussource-onlymap. No pose-readout weights used. No numericE02/E04values",
    fit="TRAIN-only sharedwithin-recordfeatureSDfloor.001; ordinary and record-fixed-effects ridge. Fixed-effects centering uses TRAINrecordsONLY; inference uses TRAINglobalmeans and no queryrecordmeans/GT",
    selection="Perarm/decoder choose max equal-source-room VALrecord-centered shapeR2 over5alphas; strongeralpha tie. PRIMARYmasked_temporal/fixed_effects fixed before fitting",
    controls="Randomencoder, supervisedencoder, raw-current, primaryordinary, source staticmean and endpointclock; within-recordprediction shuffle. All candidates reported",
    gate="PRIMARYpooledshape-motionR2>.05, eachsourceVALroom>0,3of4VALpeople>0, lag8deltaR2>0, bettermotionR2andlowerMSEthanallselectedlearned/noRFcontrols, within-recordshuffleworse. NoE02scoringfromthisscreen",
    prior_exposure="Sourcepersonsplitpreviouslyusedfor2Ddevelopment; newgeometrysource-targetscreenisexploratory, notfinalconfirmation",
)


def metadata(study):
    p.source_contract(study)
    with np.load(study / "targets-and-metadata.npz", allow_pickle=False) as z:
        data = {key: z[key] for key in ("room", "person", "record", "action", "end")}
    train, validation = p.source_roles(data)
    with np.load(study / "roles.npz", allow_pickle=False) as z:
        for key, expected in (("train", train), ("validation", validation)):
            if z[key].dtype != bool:
                raise ValueError("Boolean frozen source roles required")
            np.testing.assert_array_equal(z[key], expected)
    return data, train, validation


def load_targets(data, source, annotations, receipt, inventory):
    """Only explicitly selected E01/E03 annotation members can be decoded."""
    source = np.asarray(source)
    if source.dtype != bool or source.shape != data["room"].shape:
        raise ValueError("Explicit Boolean source mask required")
    ids = np.flatnonzero(source)
    if not len(ids) or not np.isin(data["room"][ids], ["E01", "E03"]).all():
        raise ValueError("Only E01/E03 source annotations permitted")
    wanted = {str(name) + "/ground_truth.npy" for name in data["record"][ids]}
    audit = json.loads(receipt.read_text())
    entries = json.loads(inventory.read_text())
    if (
        audit["status"] != "CRC_verified_source_only_skeletons"
        or audit["rooms"] != ["E01", "E03"]
        or len(audit["records"]) != 540
        or len(entries) != 540
    ):
        raise ValueError("Expected complete source-only annotation acquisition")
    records = {row["path"]: row for row in audit["records"]}
    entries = {row["name"]: row for row in entries}
    if (
        len(records) != 540
        or len(entries) != 540
        or set(records) != wanted
        or set(entries) != wanted
    ):
        raise ValueError("Annotation membership differs from source records")
    # Verify all declared bytes/CRC before any annotation is decoded.
    for name in sorted(wanted):
        path = annotations / name
        if not path.resolve().is_relative_to(annotations.resolve()):
            raise ValueError("Annotation path escaped its declared root")
        raw = path.read_bytes()
        if (
            len(raw) != records[name]["bytes"]
            or len(raw) != entries[name]["size"]
            or p.digest(path) != records[name]["sha256"]
            or zlib.crc32(raw) & 0xFFFFFFFF != entries[name]["crc"]
        ):
            raise ValueError("Annotation CRC/hash differs: " + name)
    result = np.empty((len(ids), 136), dtype=np.float64)
    labels = data["record"][ids]
    for name in sorted(wanted):
        record = name.removesuffix("/ground_truth.npy")
        values = np.load(annotations / name, allow_pickle=False)
        if values.shape != (297, 17, 3) or values.dtype != np.float32:
            raise ValueError("Expected original297x17x3float32 annotation array")
        chosen = np.flatnonzero(labels == record)
        ends = data["end"][ids[chosen]]
        if ends.dtype.kind not in "iu" or (ends < 0).any() or (ends >= 297).any():
            raise ValueError("Annotation endpoints are outside297frames")
        result[chosen] = canonical_distances(values[ends])
    return ids, result


def score(prediction, target, record, ends):
    prediction = p.real_finite(prediction, "predicted shape")
    target = p.real_finite(target, "provided shape", prediction.shape)
    if prediction.ndim != 2 or prediction.shape[1] != 136 or len(record) != len(target):
        raise ValueError("Matching136distance descriptors and records required")
    pc, yc = (
        linear.center_groups(prediction, record),
        linear.center_groups(target, record),
    )
    energy = float(np.square(yc).sum())
    current, past = linear.lag_pairs(record, ends)
    dy, dp = target[current] - target[past], prediction[current] - prediction[past]
    delta_energy = float(np.square(dy).sum())
    return dict(
        mse=float(np.square(prediction - target).mean()),
        record_centered_shape_r2=1 - float(np.square(pc - yc).sum()) / energy
        if energy > 0
        else None,
        lag8_zero_reference_delta_r2=1 - float(np.square(dp - dy).sum()) / delta_energy
        if delta_energy > 0
        else None,
        target_motion_squared_energy=energy,
        target_delta_squared_energy=delta_energy,
        endpoints=len(target),
    )


def breakdown(prediction, target, meta):
    result = dict(pooled=score(prediction, target, meta["record"], meta["end"]))
    for field in ("room", "person", "action"):
        result["per_" + field] = {
            str(name): score(
                prediction[meta[field] == name],
                target[meta[field] == name],
                meta["record"][meta[field] == name],
                meta["end"][meta[field] == name],
            )
            for name in sorted(set(meta[field]))
        }
    r2 = [row["record_centered_shape_r2"] for row in result["per_room"].values()]
    result["equal_room_shape_r2"] = (
        float(np.mean(r2)) if all(x is not None for x in r2) else None
    )
    return result


def fit_source(x, target, meta, train, validation):
    if (
        train.dtype != bool
        or validation.dtype != bool
        or train.shape != validation.shape
        or train.shape != meta["room"].shape
        or np.any(train & validation)
        or not train.any()
        or not validation.any()
    ):
        raise ValueError("Explicit disjoint source TRAIN/VAL required")
    for mask, name in ((train, "train"), (validation, "validation")):
        if not np.isin(meta["room"][mask], ["E01", "E03"]).all():
            raise ValueError("No E02 numeric feature/target access permitted")
        validation_people = np.isin(meta["person"][mask], ["S04", "S09", "S24", "S29"])
        if validation_people.any() if name == "train" else not validation_people.all():
            raise ValueError("Source TRAIN/VAL people differ")
    if set(meta["person"][train]) & set(meta["person"][validation]):
        raise ValueError("Source people overlap")
    states = linear.fit_training(
        p.real_finite(x[np.flatnonzero(train)], "source TRAIN features"),
        p.real_finite(
            target[np.flatnonzero(train)], "source TRAIN shape", (int(train.sum()), 136)
        ),
        meta["record"][train],
    )
    vi = np.flatnonzero(validation)
    vx, vy = (
        p.real_finite(x[vi], "source VALfeatures"),
        p.real_finite(target[vi], "source VALshape", (len(vi), 136)),
    )
    vm = {k: v[vi] for k, v in meta.items()}
    permutation = features.permutation(vm["record"], 90210)
    reports, predictions, selections = {}, {}, {}
    for name, state in states.items():
        candidates = []
        for i, alpha in enumerate(linear.ALPHAS):
            prediction = linear.predict(state, vx, i)
            candidates.append(prediction)
            reports.setdefault(name, []).append(
                dict(
                    alpha=alpha,
                    aligned=breakdown(prediction, vy, vm),
                    shuffled=breakdown(prediction[permutation], vy, vm),
                )
            )
        available = [
            i
            for i, row in enumerate(reports[name])
            if row["aligned"]["equal_room_shape_r2"] is not None
        ]
        if not available:
            raise ValueError("No defined source-validation shape motion")
        chosen = min(
            available,
            key=lambda i: (
                -reports[name][i]["aligned"]["equal_room_shape_r2"],
                -linear.ALPHAS[i],
            ),
        )
        selections[name] = dict(index=chosen, **reports[name][chosen])
        predictions[name] = candidates[chosen]
    return states, reports, selections, predictions


def decide_gate(results, no_rf):
    primary = results["masked_temporal"]["selections"]["fixed_effects"]
    aligned = primary["aligned"]
    pooled = aligned["pooled"]
    controls = [
        selected["aligned"]["pooled"]
        for arm, value in results.items()
        for family, selected in value["selections"].items()
        if (arm, family) != ("masked_temporal", "fixed_effects")
    ] + [value["pooled"] for value in no_rf.values()]

    def positive(value, threshold=0):
        return value is not None and np.isfinite(value) and value > threshold

    def better(a, b):
        return a is not None and b is not None and positive(a - b)

    return dict(
        useful_shape_motion=positive(pooled["record_centered_shape_r2"], 0.05),
        both_rooms_positive=set(aligned["per_room"]) == {"E01", "E03"}
        and all(
            positive(row["record_centered_shape_r2"])
            for row in aligned["per_room"].values()
        ),
        three_of_four_people_positive=set(aligned["per_person"])
        == {"S04", "S09", "S24", "S29"}
        and sum(
            positive(row["record_centered_shape_r2"])
            for row in aligned["per_person"].values()
        )
        >= 3,
        positive_delta=positive(pooled["lag8_zero_reference_delta_r2"]),
        beats_control_motion=all(
            better(pooled["record_centered_shape_r2"], row["record_centered_shape_r2"])
            for row in controls
        ),
        beats_control_mse=all(pooled["mse"] < row["mse"] for row in controls),
        shuffle_worse=better(
            pooled["record_centered_shape_r2"],
            primary["shuffled"]["pooled"]["record_centered_shape_r2"],
        ),
    )


def run(study, cache, annotations, receipt, inventory, out):
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    if any(
        out.resolve().is_relative_to(path.resolve())
        for path in (study, cache, annotations)
    ):
        raise ValueError("Output must be outside the frozen input directories")
    start = time.monotonic()
    data, train_all, val_all = metadata(study)
    sources = [
        Path(__file__),
        ROOT / "ghostmesh/pose_shape.py",
        ROOT / "tests/test_pose_shape.py",
        ROOT / "tests/test_shape_probe.py",
        ROOT / "scripts/domain_conditioned_dynamics_probe.py",
        ROOT / "scripts/domain_masked_motion_diagnostic.py",
        ROOT / "scripts/verify_masked_results.py",
        ROOT / "scripts/verify_masked_calibration_probe.py",
        ROOT / "scripts/verify_masked_fullpose_probe.py",
        ROOT / "scripts/verify_masked_pretrain_probe.py",
        ROOT / "scripts/research_numeric_readback.py",
        ROOT / "scripts/domain_covariance_full_cache.py",
        ROOT / "scripts/domain_background_probe.py",
    ]
    registration = dict(
        config=CONFIG,
        source_ready_sha256=p.EXPECTED_READY,
        acquisition_receipt_sha256=p.digest(receipt),
        inventory_sha256=p.digest(inventory),
        cache_receipt_sha256=p.digest(cache / "receipt.json"),
        source_sha256={str(path.relative_to(ROOT)): p.digest(path) for path in sources},
        numeric_E02_values_read=False,
        numpy=np.__version__,
        scipy=SCIPY_VERSION,
    )
    out.mkdir(parents=True)
    p.save_json(out / "registration.json", registration)
    shutil.copyfile(receipt, out / "annotation-acquisition-receipt.json")
    shutil.copyfile(inventory, out / "annotation-inventory.json")
    for source in sources:
        destination = out / "executed-source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    ids, target = load_targets(
        data, train_all | val_all, annotations, receipt, inventory
    )
    train, validation = train_all[ids], val_all[ids]
    meta = {k: v[ids] for k, v in data.items()}
    p.save_npz(
        out / "source-targets.npz",
        global_ids=ids,
        target=target,
        train=train,
        validation=validation,
        **meta,
    )
    if not (len(ids) == 17670 and train.sum() == 14124 and validation.sum() == 3546):
        raise ValueError("Source endpoint cohort changed")
    if (
        p.digest(cache / "receipt.json")
        != json.loads((study / "registration.json").read_text())["cache_receipt_sha256"]
    ):
        raise ValueError("Raw-current cache identity differs")
    cache_receipt = json.loads((cache / "receipt.json").read_text())
    p.verify_hashes(
        cache,
        {
            "untreated_covariance-features.npy": cache_receipt["arrays_sha256"][
                "untreated_covariance-features.npy"
            ]
        },
    )
    results = {}
    for arm in ARMS:
        x = features.frozen_features(study, cache, arm, ids, len(train_all))
        if arm == "raw_current":
            x = features.contracts.frozen_design(
                x, features.contracts.npz(study / "raw_current-rff.npz")
            )
        states, reports, selections, predictions = fit_source(
            x, target, meta, train, validation
        )
        for name, state in states.items():
            p.save_npz(out / (arm + "-" + name + "-states.npz"), **state)
            np.save(
                out / (arm + "-" + name + "-VALprediction.npy"),
                predictions[name],
                allow_pickle=False,
            )
        results[arm] = dict(all_configs=reports, selections=selections)
        print(
            json.dumps(
                dict(
                    arm=arm,
                    selections={
                        name: dict(
                            alpha=row["alpha"], validation=row["aligned"]["pooled"]
                        )
                        for name, row in selections.items()
                    },
                )
            ),
            flush=True,
        )
    vm = {k: v[validation] for k, v in meta.items()}
    static = np.broadcast_to(target[train].mean(0), target[validation].shape)
    clock = {
        int(e): target[train & (meta["end"] == e)].mean(0)
        for e in set(meta["end"][train])
    }
    no_rf = dict(
        source_mean=breakdown(static, target[validation], vm),
        source_clock=breakdown(
            np.stack([clock[int(e)] for e in meta["end"][validation]]),
            target[validation],
            vm,
        ),
    )
    gate = decide_gate(results, no_rf)
    p.verify_hashes(ROOT, registration["source_sha256"])
    p.save_json(
        out / "results.json",
        dict(
            registration=registration,
            arms=results,
            no_rf_controls=no_rf,
            gate=gate,
            passes=all(gate.values()),
            seconds=time.monotonic() - start,
            scope="Source geometry feasibility only; no2Dpose/3Dreconstruction/unseen-room efficacy claim",
        ),
    )
    p.save_json(
        out / "artifact-manifest.json",
        {
            str(path.relative_to(out)): p.digest(path)
            for path in out.rglob("*")
            if path.is_file()
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "cache", "annotations", "receipt", "inventory", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(
        args.study, args.cache, args.annotations, args.receipt, args.inventory, args.out
    )
