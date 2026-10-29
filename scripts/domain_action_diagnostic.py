"""Source-only CSI action information diagnostic, with whole-person splits.

This is not a pose model, an untouched final evaluation, or a generalization
claim outside these E01-E03 development recordings. No E04 arrays are opened.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import time

import numpy as np
from scipy.linalg import eigh

ROOMS = ("E01", "E02", "E03")
ALPHAS = (0.01, 0.1, 1.0, 10.0)
SEED = 2027


def multiply(first, second):
    """Explicit contraction avoids this host's Accelerate matmul FP warnings."""
    result = np.einsum("ik,kj->ij", first, second, optimize=False)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite contraction")
    return result


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def masks(room, subject, held_room, protocol):
    numbers = np.asarray([int(s[1:]) % 10 for s in subject])
    if protocol == "within_room":
        source = room == held_room
        train = source & np.isin(numbers, (1, 2, 3))
        validation = source & (numbers == 4)
        held = source & (numbers == 5)
    elif protocol == "unseen_room":
        source = room != held_room
        train = source & (numbers != 4)
        validation = source & (numbers == 4)
        held = ~source
    else:
        raise ValueError("Unknown protocol")
    for first, second in ((train, validation), (train, held), (validation, held)):
        if np.any(first & second) or set(subject[first]) & set(subject[second]):
            raise ValueError("People overlap between splits")
    if not all(np.any(value) for value in (train, validation, held)):
        raise ValueError("Empty split")
    return train, validation, held


def record_permutation(labels, subjects, sequences, training, seed=SEED):
    """Permute one action assignment per recording within each training person."""
    result = labels.copy()
    rng = np.random.default_rng(seed)
    for person in sorted(set(subjects[training])):
        records = sorted(set(sequences[training & (subjects == person)]))
        if len(records) != 27:
            raise ValueError("Each source person must have all27 action records")
        values = [labels[np.flatnonzero(sequences == record)[0]] for record in records]
        if len(set(values)) != 27:
            raise ValueError("Expected one record per action/person")
        shuffled = rng.permutation(values)
        for record, value in zip(records, shuffled):
            result[sequences == record] = value
    return result


def standardize(x, training):
    mean = x[training].mean(0)
    scale = np.maximum(x[training].std(0), 0.1)
    return (x - mean) / scale


def rbf_features(x, training):
    """Fixed random Fourier approximation; bandwidth uses source training only."""
    source = x[training]
    ix = np.linspace(0, len(source) - 1, min(128, len(source)), dtype=int)
    small = source[ix]
    distances = np.maximum(
        (small * small).sum(1)[:, None]
        + (small * small).sum(1)[None, :]
        - 2 * multiply(small, small.T),
        0,
    )
    bandwidth_squared = float(np.median(distances[np.triu_indices(len(small), 1)]))
    if not np.isfinite(bandwidth_squared) or bandwidth_squared <= 0:
        raise ValueError("Degenerate training bandwidth")
    rng = np.random.default_rng(SEED)
    projection = rng.normal(size=(x.shape[1], 512)) / np.sqrt(bandwidth_squared)
    offset = rng.uniform(0, 2 * np.pi, size=512)
    return np.sqrt(2 / 512) * np.cos(multiply(x, projection) + offset), bandwidth_squared


def ridge_scores(x, labels, training):
    source = x[training]
    mean = source.mean(0)
    source = source - mean
    y = np.eye(27)[labels[training]]
    ym = y.mean(0)
    covariance = multiply(source.T, source)
    values, vectors = eigh(covariance, check_finite=True)
    right = multiply(vectors.T, multiply(source.T, y - ym))
    for alpha in ALPHAS:
        weights = multiply(vectors, right / (values[:, None] + alpha * len(source)))
        prediction = multiply(x - mean, weights) + ym
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite classifier scores")
        yield alpha, prediction


def accuracy(scores, labels, sequences):
    predicted = scores.argmax(1)
    classes = sorted(set(labels))
    balanced = float(np.mean([np.mean(predicted[labels == c] == c) for c in classes]))
    per_record = []
    for record in sorted(set(sequences)):
        mask = sequences == record
        if len(set(labels[mask])) != 1:
            raise ValueError("A recording changed action label")
        per_record.append(int(scores[mask].mean(0).argmax()) == int(labels[mask][0]))
    return {
        "accuracy": float(np.mean(predicted == labels)),
        "balanced_accuracy": balanced,
        "record_mean_score_accuracy": float(np.mean(per_record)),
        "windows": len(labels),
        "records": len(per_record),
    }


def run(cache, out, feature_set="static"):
    if out.exists():
        raise FileExistsError(out)
    receipt = json.loads((cache / "receipt.json").read_text())
    if feature_set not in {"static", "dynamic"}:
        raise ValueError("Unknown feature set")
    names = ("moments", "room", "subject", "sequence", "end")
    if feature_set == "dynamic":
        names += ("dynamic",)
    hashes = {name: digest(cache / (name + ".npy")) for name in names}
    for name, value in hashes.items():
        if receipt["arrays"][name + ".npy"] != value:
            raise ValueError("Cache binding changed")
    arrays = {name: np.load(cache / (name + ".npy"), allow_pickle=False) for name in names}
    room, subject, sequence = (arrays[name] for name in ("room", "subject", "sequence"))
    if set(room) != set(ROOMS) or len(set(sequence)) != 405:
        raise ValueError("Only405source recordings allowed")
    labels = np.asarray([int(s.split("/")[-1][1:]) - 1 for s in sequence])
    moments = arrays["moments"].astype("float64")
    features = {"last": moments[:, -1], "temporal": moments.reshape(len(moments), -1)}
    if feature_set == "dynamic":
        centered = arrays["moments"] - arrays["moments"].mean(1, keepdims=True)
        if not np.array_equal(arrays["dynamic"], centered):
            raise ValueError("Dynamic cache is not within-window centered moments")
        features = {"dynamic": arrays["dynamic"].astype("float64").reshape(len(moments), -1)}
    out.mkdir(parents=True)
    started = time.monotonic()
    registration = {
        "scope": "Development only E01-E03; no E04 decode, no pose efficacy claim",
        "cache_receipt_sha256": digest(cache / "receipt.json"),
        "cache_arrays_sha256": hashes,
        "script_sha256": digest(Path(__file__)),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "seed": SEED,
        "ridge_grid": ALPHAS,
        "rbf": "512 fixed random Fourier features, median128-source-pair squared distance bandwidth",
        "selection": "Maximum source-validation balanced accuracy; first alpha wins ties",
        "permutation": "Shuffle27whole action assignments independently within each training person; validation/held labels unchanged",
        "split": "within-room train persons1/2/3,val4,test5; unseen-room train non4people fromother2rooms,val4people,testall5heldroompeople",
        "chance": 1 / 27,
        "inputs": "CSI moments only; record/subject/action names select split and targets, never enter classifier",
        "feature_set": feature_set,
        "feature_names": list(features),
        "dynamic_definition": "Same912-dimensional8-frame moments minus each feature's within-window time mean; no target-distribution fitted parameters",
        "limitations": [
            "Source training accuracy can reflect recording identity",
            "Record-level mean scores use the entire recording and are offline summaries, not causal inference",
            "A negative result does not establish lack of information in raw CSI/phase or nonlinear models",
            "All comparisons are exploratory development results from one fixed seed",
        ],
    }
    (out / "registration.json").write_text(json.dumps(registration, indent=2) + "\n")
    (out / "executed-source.py").write_bytes(Path(__file__).read_bytes())
    rows = []
    for protocol in ("within_room", "unseen_room"):
        for held_room in ROOMS:
            train, val, held = masks(room, subject, held_room, protocol)
            permuted = record_permutation(labels, subject, sequence, train)
            for name, values in features.items():
                x = standardize(values, train)
                for kernel in ("linear", "rbf512"):
                    transformed, bandwidth = (x, None) if kernel == "linear" else rbf_features(x, train)
                    for control, training_labels in (("true_labels", labels), ("record_permutation", permuted)):
                        candidates = []
                        best = None
                        for alpha, score in ridge_scores(transformed, training_labels, train):
                            validation = accuracy(score[val], labels[val], sequence[val])
                            candidates.append({"alpha": alpha, "validation": validation})
                            if best is None or validation["balanced_accuracy"] > best[0]:
                                best = validation["balanced_accuracy"], alpha, score
                        _, alpha, score = best
                        result = {
                            "protocol": protocol, "held_room": held_room,
                            "feature": name, "kernel": kernel, "control": control,
                            "alpha": alpha, "bandwidth_squared": bandwidth,
                            "training_assigned_labels": accuracy(score[train], training_labels[train], sequence[train]),
                            "training_real_labels": accuracy(score[train], labels[train], sequence[train]),
                            "validation": accuracy(score[val], labels[val], sequence[val]),
                            "held": accuracy(score[held], labels[held], sequence[held]),
                            "selection_grid": candidates,
                        }
                        rows.append(result)
                        np.savez_compressed(
                            out / f"{protocol}-{held_room}-{name}-{kernel}-{control}.npz",
                            held_indices=np.flatnonzero(held), scores=score[held], labels=labels[held],
                        )
                        print(json.dumps({k: result[k] for k in ("protocol", "held_room", "feature", "kernel", "control", "held")}), flush=True)
    for name, value in hashes.items():
        if digest(cache / (name + ".npy")) != value:
            raise ValueError("Cache mutated during diagnostic")
    if digest(Path(__file__)) != registration["script_sha256"]:
        raise ValueError("Source mutated during diagnostic")
    result = {"registration": registration, "results": rows, "elapsed_seconds": time.monotonic() - started}
    (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--features", choices=("static", "dynamic"), default="static")
    args = parser.parse_args()
    run(args.cache, args.out, args.features)
