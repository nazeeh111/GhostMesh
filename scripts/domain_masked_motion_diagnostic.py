"""SOURCE-only fixed-effects motion screen of frozen CSI embeddings.

All alphas and three encoders are reported. This is a feasibility diagnostic,
not a selected pose decoder, full-image reconstruction, or held-room evaluation.
No E02 pose values, embedding rows, or raw CSI are consumed by the screen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import shutil
import time
import zipfile

import numpy as np
from scipy.linalg import eigh
from scipy.linalg.blas import dgemm

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_READY = "303922cdcb0e8007b25b0488c5cc14515487afb635cf60bc9b8b0b3f31786461"
ARMS = ("masked_temporal", "random_frozen", "supervised")
ALPHAS = (1e-5, 0.001, 0.01, 0.1, 1.0)
SCALE_FLOOR = 0.001


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def stored_npz_array(path, member):
    """Read-only lazy array view, avoiding decoding unrelated held GT rows.

    np.savez uses uncompressed ZIP_STORED NPY members. Reject compressed/object
    members instead of materializing the entire pose member (including E02).
    """
    with zipfile.ZipFile(path) as archive:
        item = archive.getinfo(member + ".npy")
        if item.compress_type != zipfile.ZIP_STORED or item.flag_bits & 1:
            raise ValueError("Expected unencrypted ZIP_STORED array member")
    with Path(path).open("rb") as handle:
        handle.seek(item.header_offset)
        header = handle.read(30)
        if len(header) != 30 or header[:4] != b"PK\x03\x04":
            raise ValueError("Invalid ZIP local header")
        name_length, extra_length = struct.unpack_from("<HH", header, 26)
        handle.seek(name_length + extra_length, 1)
        version = np.lib.format.read_magic(handle)
        readers = {
            (1, 0): np.lib.format.read_array_header_1_0,
            (2, 0): np.lib.format.read_array_header_2_0,
        }
        if version not in readers:
            raise ValueError("Unsupported NPY member header version")
        shape, fortran, dtype = readers[version](handle)
        if dtype.hasobject:
            raise ValueError("Object arrays are forbidden")
        offset = handle.tell()
    if (
        int(np.prod(shape)) * dtype.itemsize + offset
        > item.header_offset + 30 + name_length + extra_length + item.file_size
    ):
        raise ValueError("Truncated array member")
    return np.memmap(
        path,
        mode="r",
        dtype=dtype,
        shape=shape,
        offset=offset,
        order="F" if fortran else "C",
    )


def source_roles(metadata):
    room, person, record, end = (
        np.asarray(metadata[k]) for k in ("room", "person", "record", "end")
    )
    if len({len(room), len(person), len(record), len(end)}) != 1:
        raise ValueError("Metadata lengths differ")
    if end.dtype.kind not in "iu":
        raise ValueError("Integer causal endpoints required")
    own = np.isin(room, ["E01", "E03"])
    for name, start in [("E01", 1), ("E03", 21)]:
        if not set(person[room == name]) <= set(
            f"S{i:02}" for i in range(start, start + 10)
        ):
            raise ValueError("Person IDs disagree with source room")
    validation = own & np.isin(person, ["S04", "S09", "S24", "S29"])
    train = own & ~validation
    if tuple(int(mask.sum()) for mask in (train, validation)) != (14124, 3546):
        raise ValueError("Frozen source cohort counts differ")
    if tuple(len(set(person[mask])) for mask in (train, validation)) != (16, 4):
        raise ValueError("Required source people missing")
    if set(person[train]) & set(person[validation]) or set(record[train]) & set(
        record[validation]
    ):
        raise ValueError("Training and validation identities overlap")
    for ids in (np.flatnonzero(train), np.flatnonzero(validation)):
        if (
            np.any(end[ids] < 39)
            or np.any(end[ids] > 295)
            or np.any((end[ids] - 39) % 8)
        ):
            raise ValueError("Noncanonical causal endpoint")
        if any(
            str(r).split("/")[:2] != [str(a), str(b)]
            for r, a, b in zip(record[ids], room[ids], person[ids])
        ):
            raise ValueError("Record/room/person metadata disagree")
    return train, validation


def center_groups(values, groups):
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    if (
        values.ndim != 2
        or len(values) != len(groups)
        or not len(values)
        or not np.isfinite(values).all()
    ):
        raise ValueError("Finite nonempty matrix and matching groups required")
    centered = np.empty_like(values)
    for group in np.unique(groups):
        ids = groups == group
        selected = values[ids]
        relative = selected - selected[0]
        centered[ids] = relative - relative.mean(0)
    return centered


def checked_product(left, right):
    """Finite float64 BLAS product; reject genuine failures, no warning filters.

    Actual source Grams were checked against einsum and compensated scalar sums.
    This avoids NumPy matmul's spurious FP status warnings on finite products
    while preserving all statistical equations and checking every result.
    """
    a, b = np.asarray(left), np.asarray(right)
    if (
        a.ndim != 2
        or b.ndim != 2
        or a.shape[1] != b.shape[0]
        or not all(a.shape)
        or not all(b.shape)
        or a.dtype.kind not in "biuf"
        or b.dtype.kind not in "biuf"
    ):
        raise ValueError("Matching nonempty real two-dimensional matrices required")
    a, b = (
        np.asfortranarray(a, dtype=np.float64),
        np.asfortranarray(b, dtype=np.float64),
    )
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite BLAS product input")
    result = dgemm(alpha=1.0, a=a, b=b)
    if result.shape != (a.shape[0], b.shape[1]) or not np.isfinite(result).all():
        raise ValueError("Nonfinite or malformed BLAS product output")
    return result


def fit_training(x, y, records):
    """Only TRAIN observations/labels are accepted, with no validation inputs."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or not np.isfinite(y).all():
        raise ValueError("Finite paired training matrices required")
    within_x, within_y = center_groups(x, records), center_groups(y, records)
    xm, ym = x.mean(0), y.mean(0)
    scale = np.maximum(np.sqrt(np.square(within_x).mean(0)), SCALE_FLOOR)
    models = {}
    for name, design, target in [
        ("ordinary", (x - xm) / scale, y - ym),
        ("fixed_effects", within_x / scale, within_y),
    ]:
        matrix = checked_product(design.T, design)
        right = checked_product(design.T, target)
        eigen, vectors = eigh(matrix, check_finite=True)
        projected = checked_product(vectors.T, right)
        weights, residuals = [], []
        for alpha in ALPHAS:
            weight = checked_product(
                vectors, projected / (eigen[:, None] + len(x) * alpha)
            )
            residual = np.linalg.norm(
                checked_product(matrix + len(x) * alpha * np.eye(x.shape[1]), weight)
                - right
            ) / max(np.linalg.norm(right), 1e-12)
            if not np.isfinite(residual) or residual > 1e-6:
                raise ValueError("Normal equation residual failed")
            weights.append(weight)
            residuals.append(float(residual))
        models[name] = dict(
            x_mean=xm,
            feature_scale=scale,
            y_mean=ym,
            weights=np.stack(weights),
            alphas=np.asarray(ALPHAS),
            normal_equation_relative_residual=np.asarray(residuals),
        )
    return models


def fit_source_rows(feature, target, metadata, train):
    train = np.asarray(train)
    if train.dtype != bool or train.shape != (len(metadata["room"]),):
        raise ValueError("Explicit Boolean source TRAIN mask required")
    ids = np.flatnonzero(train)
    if (
        not len(ids)
        or not np.isin(np.asarray(metadata["room"])[ids], ["E01", "E03"]).all()
    ):
        raise ValueError("Only explicit source TRAIN rows may fit")
    return fit_training(
        feature[ids],
        np.asarray(target[ids]).reshape(len(ids), -1),
        np.asarray(metadata["record"])[ids],
    )


def predict(state, x, alpha_index):
    """No labels, record IDs, query centering, or reference statistics accepted."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(state["x_mean"]) or not np.isfinite(x).all():
        raise ValueError("Finite matching feature matrix required")
    return (
        checked_product(
            (x - state["x_mean"]) / state["feature_scale"],
            state["weights"][alpha_index],
        )
        + state["y_mean"]
    )


def lag_pairs(records, ends):
    records, ends = np.asarray(records), np.asarray(ends)
    if (
        records.ndim != 1
        or ends.ndim != 1
        or ends.dtype.kind not in "iu"
        or len(records) != len(ends)
    ):
        raise ValueError("Matching1D records/integer endpoints required")
    keys = [(str(record), int(end)) for record, end in zip(records, ends)]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate record/endpoint pair")
    lookup = {key: i for i, key in enumerate(keys)}
    pairs = [
        (i, lookup[(record, end - 8)])
        for i, (record, end) in enumerate(keys)
        if (record, end - 8) in lookup
    ]
    if not pairs:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    current, past = np.asarray(pairs, dtype=int).T
    return current, past


def motion_scores(prediction, target, records, ends):
    prediction, target = (
        np.asarray(prediction, dtype=np.float64),
        np.asarray(target, dtype=np.float64),
    )
    if (
        prediction.shape != target.shape
        or target.shape[1:] != (17, 2)
        or not np.isfinite(prediction).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("Finite17joint2D predictions and targets required")
    pc = center_groups(prediction.reshape(len(target), 34), records)
    yc = center_groups(target.reshape(len(target), 34), records)
    energy = float(np.square(yc).sum())
    error = float(np.square(pc - yc).sum())
    current, past = lag_pairs(records, ends)
    dy = target[current] - target[past]
    dp = prediction[current] - prediction[past]
    delta_energy = float(np.square(dy).sum())
    delta_error = float(np.square(dp - dy).sum())
    diagonal = np.linalg.norm(target.max(1) - target.min(1), axis=1)
    usable = diagonal > 1e-6
    nje = (
        float(
            (
                np.linalg.norm(prediction[usable] - target[usable], axis=-1)
                / diagonal[usable, None]
            ).mean()
        )
        if usable.any()
        else None
    )
    return dict(
        hip_relative_nje=nje,
        record_centered_articulation_r2=1 - error / energy if energy > 0 else None,
        record_centered_target_squared_energy=energy,
        lag8_zero_reference_delta_r2=1 - delta_error / delta_energy
        if delta_energy > 0
        else None,
        lag8_target_squared_energy=delta_energy,
        lag8_pairs=len(current),
        windows=len(target),
        records=len(set(records)),
        zero_energy_policy="R2=None, never a fabricated zero/positive score",
    )


def breakdown(prediction, target, metadata):
    output = dict(
        pooled=motion_scores(prediction, target, metadata["record"], metadata["end"])
    )
    for field in ("person", "room"):
        values = metadata[field]
        output["per_" + field] = {
            str(name): motion_scores(
                prediction[values == name],
                target[values == name],
                metadata["record"][values == name],
                metadata["end"][values == name],
            )
            for name in np.unique(values)
        }
    return output


def variance_energy(values, metadata, train_mean, train_scale):
    """Exact weighted nested ANOVA identity, descriptive only for validation."""
    normalized = (np.asarray(values, dtype=np.float64) - train_mean) / train_scale
    if not np.isfinite(normalized).all() or not len(normalized):
        raise ValueError("Finite nonempty normalized values required")
    # Cohort centering is DESCRIPTIVE ONLY. Separating this common shift avoids
    # attributing a shared validation offset to room differences. Centering
    # differences from one observed row makes repeated rows exactly zero.
    relative = normalized - normalized[0]
    relative_mean = relative.mean(0)
    cohort_mean = normalized[0] + relative_mean
    x = relative - relative_mean
    train_origin_total = float(np.square(normalized).sum())
    cohort_shift_energy = float(len(x) * np.square(cohort_mean).sum())
    room, person, record = (metadata[k] for k in ("room", "person", "record"))
    if any(len(set(room[person == key])) != 1 for key in np.unique(person)) or any(
        len(set(person[record == key])) != 1 for key in np.unique(record)
    ):
        raise ValueError("Grouping hierarchy is not nested")
    rm = {key: x[room == key].mean(0) for key in np.unique(room)}
    pm = {key: x[person == key].mean(0) for key in np.unique(person)}
    cm = {key: x[record == key].mean(0) for key in np.unique(record)}
    between_room = sum(
        int((room == key).sum()) * float(np.square(value).sum())
        for key, value in rm.items()
    )
    person_in_room = sum(
        int((person == key).sum())
        * float(np.square(value - rm[room[np.flatnonzero(person == key)[0]]]).sum())
        for key, value in pm.items()
    )
    record_in_person = sum(
        int((record == key).sum())
        * float(np.square(value - pm[person[np.flatnonzero(record == key)[0]]]).sum())
        for key, value in cm.items()
    )
    within = float(np.square(center_groups(x, record)).sum())
    total = float(np.square(x).sum())
    components = dict(
        between_room=between_room,
        person_within_room=person_in_room,
        record_within_person=record_in_person,
        frame_within_record=within,
    )
    residual = abs(sum(components.values()) - total) / max(total, 1e-12)
    shift_identity_residual = abs(
        total + cohort_shift_energy - train_origin_total
    ) / max(train_origin_total, 1e-12)
    if residual > 1e-10 or shift_identity_residual > 1e-10:
        raise ValueError("Nested variance/mean-shift energy identity failed")
    return dict(
        cohort_centered_total_squared_energy=total,
        train_origin_total_squared_energy=train_origin_total,
        common_cohort_mean_shift_squared_energy=cohort_shift_energy,
        common_cohort_mean_shift_fraction_of_train_origin=cohort_shift_energy
        / train_origin_total
        if train_origin_total > 0
        else None,
        centered_fraction_of_train_origin=total / train_origin_total
        if train_origin_total > 0
        else None,
        components_squared_energy=components,
        component_fractions={
            key: value / total if total > 0 else None
            for key, value in components.items()
        },
        cumulative_between_record_fraction=(
            between_room + person_in_room + record_in_person
        )
        / total
        if total > 0
        else None,
        cumulative_between_person_fraction=(between_room + person_in_room) / total
        if total > 0
        else None,
        centered_identity_relative_residual=residual,
        shift_plus_centered_identity_relative_residual=shift_identity_residual,
        scope="Descriptive only: use the cohort's OWNmean forANOVA, separatelyreportmean-shiftfromTRAINorigin; TRAINnormalization remainsfixed. VALmeans never enter predictor/fitting.",
    )


def run(study, out):
    if out.exists():
        raise FileExistsError(out)
    started = time.monotonic()
    if digest(study / "source-ready.json") != EXPECTED_READY:
        raise ValueError("Unexpected source-ready version")
    ready = json.loads((study / "source-ready.json").read_text())
    for relative, expected in ready["artifacts_sha256"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or digest(study / path) != expected:
            raise ValueError("Frozen artifact integrity failed: " + relative)
    registration = json.loads((study / "registration.json").read_text())
    for relative, expected in registration["source_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError("Original frozen source changed")
    with np.load(study / "targets-and-metadata.npz", allow_pickle=False) as stored:
        metadata = {
            key: stored[key] for key in ("room", "person", "record", "action", "end")
        }
    train, val = source_roles(metadata)
    target = stored_npz_array(study / "targets-and-metadata.npz", "pose")
    if target.shape != (len(train), 17, 2):
        raise ValueError("Frozen hip-relative target dimensions differ")
    ids = {"train": np.flatnonzero(train), "validation": np.flatnonzero(val)}
    # Only these source rows are numerically read from the lazy GT/feature views.
    y = {name: np.array(target[indices]) for name, indices in ids.items()}
    meta = {
        name: {key: value[indices] for key, value in metadata.items()}
        for name, indices in ids.items()
    }
    ym = y["train"].reshape(len(y["train"]), 34).mean(0)
    out.mkdir(parents=True)
    save(
        out / "registration.json",
        dict(
            kind="source-only-masked-motion-feasibility",
            source_ready_sha256=EXPECTED_READY,
            alphas=ALPHAS,
            arms=ARMS,
            views=["last64_current_frame", "full512_eight_frame_history"],
            scale_floor=SCALE_FLOOR,
            input_scale="TRAINwithin-recordSDonly; ordinary/fixed-effectssharetheSAMEscale",
            centering="TRAINfixed-effects ownrecordmeans forfitONLY. PredictorusesTRAINglobalX/Ymeans, neverquerymean/GT/recordID. RecordcenteringinR2isevaluationonly.",
            target="All17 HIP-relative2Djoints, NOTfullimagepose/3D",
            selection="NONE: allalpha/encoder/view/decoder resultsreported; exploratorysourcefeasibilityscreen only",
            e02_policy="NoE02posevalues/embeddingrows/rawCSIread; metadataforroleexclusion andwhole-filecryptographicintegrityonly",
            source_sha256={
                str(Path(__file__).relative_to(ROOT)): digest(Path(__file__)),
                "tests/test_masked_motion_diagnostic.py": digest(
                    ROOT / "tests/test_masked_motion_diagnostic.py"
                ),
            },
        ),
    )
    np.savez(
        out / "source-row-roles.npz", train=ids["train"], validation=ids["validation"]
    )
    for path in [Path(__file__), ROOT / "tests/test_masked_motion_diagnostic.py"]:
        destination = out / "executed-source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    save(
        out / "role-proof.json",
        {
            name: dict(
                rows=len(indices),
                people=sorted(set(meta[name]["person"])),
                records=sorted(set(meta[name]["record"])),
                rooms=sorted(set(meta[name]["room"])),
            )
            for name, indices in ids.items()
        },
    )
    baseline = {
        name: breakdown(
            np.broadcast_to(ym.reshape(17, 2), y[name].shape), y[name], meta[name]
        )
        for name in ids
    }
    result = {}
    for arm in ARMS:
        feature = np.load(
            study / (arm + "-embedding.npy"), mmap_mode="r", allow_pickle=False
        )
        if feature.shape != (len(train), 512):
            raise ValueError("Embedding dimensions differ")
        for view, columns in [
            ("last64_current_frame", slice(-64, None)),
            ("full512_eight_frame_history", slice(None)),
        ]:
            x = {
                name: np.array(feature[indices, columns], dtype=np.float64)
                for name, indices in ids.items()
            }
            models = fit_source_rows(feature[:, columns], target, metadata, train)
            common = models["ordinary"]
            variance = {
                name: variance_energy(
                    x[name], meta[name], common["x_mean"], common["feature_scale"]
                )
                for name in ids
            }
            reports = {}
            for decoder, state in models.items():
                tag = arm + "-" + view + "-" + decoder
                np.savez(out / (tag + "-models.npz"), **state)
                rows = []
                for i, alpha in enumerate(ALPHAS):
                    row = dict(
                        alpha=alpha,
                        normal_equation_relative_residual=float(
                            state["normal_equation_relative_residual"][i]
                        ),
                    )
                    for name in ids:
                        pred = predict(state, x[name], i).reshape(-1, 17, 2)
                        row[name] = breakdown(pred, y[name], meta[name])
                    rows.append(row)
                reports[decoder] = rows
            result[arm + "/" + view] = dict(
                normalized_feature_variance=variance, decoders=reports
            )
    save(
        out / "results.json",
        dict(
            scope="SOURCEfeasibilityonly, noE02metrics/modelselection/promotion",
            baseline_train_mean=baseline,
            models=result,
            seconds=time.monotonic() - started,
        ),
    )
    save(
        out / "artifact_manifest.json",
        {
            str(path.relative_to(out)): digest(path)
            for path in sorted(out.rglob("*"))
            if path.is_file()
        },
    )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    run(args.study, args.out)


if __name__ == "__main__":
    main()
