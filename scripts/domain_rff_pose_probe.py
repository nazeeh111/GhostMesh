"""Bounded E01-E03 pose regression probe for nonlinear dynamic CSI features.

All hyperparameters are selected on source-validation people. Final E04 arrays
are not opened. Source-only design standardization makes Ridge regularization
comparable between linear and512-dimensional random Fourier features.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.metrics import pose_metrics  # noqa:E402
from scripts.domain_action_diagnostic import ALPHAS, SEED, masks, multiply  # noqa:E402
from scripts.domain_structured_probe import motion_metrics  # noqa:E402

SHRINK = (0.0, 0.25, 0.5, 1.0)
ROOMS = ("E01", "E02", "E03")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fit_design(x, training, kernel):
    """Both levels of normalization and the bandwidth fit training people only."""
    state = {
        "input_mean": x[training].mean(0),
        "input_scale": np.maximum(x[training].std(0), 0.1),
    }
    z = (x - state["input_mean"]) / state["input_scale"]
    if kernel == "rbf512":
        sample = z[training]
        ix = np.linspace(0, len(sample) - 1, min(128, len(sample)), dtype=int)
        small = sample[ix]
        distances = np.maximum(
            (small * small).sum(1)[:, None]
            + (small * small).sum(1)[None, :]
            - 2 * multiply(small, small.T),
            0,
        )
        width = float(np.median(distances[np.triu_indices(len(small), 1)]))
        if not np.isfinite(width) or width <= 0:
            raise ValueError("Degenerate source bandwidth")
        rng = np.random.default_rng(SEED)
        state.update(
            projection=rng.normal(size=(x.shape[1], 512)) / np.sqrt(width),
            phase=rng.uniform(0, 2 * np.pi, size=512),
            bandwidth_squared=np.asarray(width),
        )
        z = np.sqrt(2 / 512) * np.cos(multiply(z, state["projection"]) + state["phase"])
    elif kernel != "linear":
        raise ValueError("Unknown kernel")
    state.update(
        design_mean=z[training].mean(0),
        design_scale=np.maximum(z[training].std(0), 1e-6),
    )
    z = (z - state["design_mean"]) / state["design_scale"]
    if not np.isfinite(z).all():
        raise ValueError("Nonfinite design")
    return z, state


def apply_design(x, state):
    z = (x - state["input_mean"]) / state["input_scale"]
    if "projection" in state:
        z = np.sqrt(2 / 512) * np.cos(multiply(z, state["projection"]) + state["phase"])
    return (z - state["design_mean"]) / state["design_scale"]


def ridge_solutions(design, target, training):
    x = design[training]
    y = target[training].astype("float64").reshape(training.sum(), -1)
    mean = y.mean(0)
    covariance = multiply(x.T, x)
    cross = multiply(x.T, y - mean)
    values, vectors = eigh(covariance, check_finite=True)
    projected = multiply(vectors.T, cross)
    for alpha in ALPHAS:
        weights = multiply(vectors, projected / (values[:, None] + alpha * len(x)))
        residual = multiply(covariance, weights) + alpha * len(x) * weights - cross
        error = float(np.linalg.norm(residual) / max(np.linalg.norm(cross), 1e-12))
        if not np.isfinite(error) or error > 1e-8:
            raise ValueError("Ridge normal-equation residual too large")
        yield alpha, weights, mean, error


def scores(prediction, target, valid):
    result = pose_metrics(prediction, target, valid)
    result.pop("per_frame_nje")
    return result


def shuffled_within_record(prediction, sequence):
    rng = np.random.default_rng(SEED)
    output = prediction.copy()
    for record in sorted(set(sequence)):
        ix = np.flatnonzero(sequence == record)
        output[ix] = prediction[rng.permutation(ix)]
    return output


def analyze(prediction, target, valid, sequence):
    shuffled = shuffled_within_record(prediction, sequence)
    return {
        "metrics": scores(prediction, target, valid),
        "motion": motion_metrics(prediction, target, sequence),
        "within_record_shuffle": {
            "metrics": scores(shuffled, target, valid),
            "motion": motion_metrics(shuffled, target, sequence),
        },
    }


def run(cache, out):
    if out.exists():
        raise FileExistsError(out)
    receipt = json.loads((cache / "receipt.json").read_text())
    names = ("moments", "dynamic", "pose", "valid", "room", "subject", "sequence", "end")
    hashes = {name: digest(cache / (name + ".npy")) for name in names}
    for name, value in hashes.items():
        if receipt["arrays"][name + ".npy"] != value:
            raise ValueError("Cache binding changed")
    a = {name: np.load(cache / (name + ".npy"), allow_pickle=False) for name in names}
    if set(a["room"]) != set(ROOMS) or len(set(a["sequence"])) != 405:
        raise ValueError("Only405E01-E03 source recordings allowed")
    if not a["valid"].all() or not np.isfinite(a["pose"]).all():
        raise ValueError("Complete finite source targets required")
    if not np.array_equal(a["dynamic"], a["moments"] - a["moments"].mean(1, keepdims=True)):
        raise ValueError("Dynamic definition differs")
    features = {
        "last": a["moments"][:, -1].astype("float64"),
        "dynamic8": a["dynamic"].astype("float64").reshape(len(a["pose"]), -1),
    }
    source_paths = [
        Path(__file__), ROOT / "scripts/domain_action_diagnostic.py",
        ROOT / "scripts/domain_structured_probe.py", ROOT / "ghostmesh/metrics.py",
    ]
    source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in source_paths}
    registration = {
        "scope": "Development only E01-E03; no E04 decode or final claim",
        "cache_receipt_sha256": digest(cache / "receipt.json"),
        "cache_arrays_sha256": hashes,
        "sources_sha256": source_hashes,
        "numpy": np.__version__, "platform": platform.platform(),
        "seed": SEED, "threads": 1, "ridge_grid": ALPHAS, "shrink_grid": SHRINK,
        "selection": "Minimum source-validation NJE; alpha grid then shrink grid order resolves ties; zero shrink allowed and disclosed",
        "objective": "Mean squared normalized-image2D target error; source-centered34outputs, no target-dependent weighting",
        "design": "source input SD floor.1; RFF512Gaussian fixed2027, bandwidthmedian128source-pair squared distance; finaldesignsourceSDfloor1e-6 for BOTH linear/RFF",
        "regularization": "(X.T X + alpha*N*I)W=X.T(Y-sourceMean); finaldesignnormalization differs explicitly from previous action classifier",
        "split": "Two source rooms:8trainingpeople,2person4validationpeople; heldroomall5people; no shared persons or records",
        "clipping": "Final absolute2D coordinates clipped[0,1] for metrics, with raw values and clipping fraction saved",
        "features": {"last": "114last-frame packet moments", "dynamic8": "912flattened8-frame moments minus per-feature within-window time mean"},
        "limits": "Originalv1 log-compressed/MAD-window-normalized packet moments, not corrected rawIQ/phase;1developmentseed, not demonstrated generalization",
        "historical_reference": "Original smallNN scores use these same8train/2valpeople and endpoints, but different model/trainingbudget; contextual comparison only",
    }
    out.mkdir(parents=True)
    started = time.monotonic()
    write(out / "registration.json", registration)
    for path in source_paths:
        destination = out / "executed-source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    rows = []
    for room in ROOMS:
        train, val, held = masks(a["room"], a["subject"], room, "unseen_room")
        if (train.sum(), val.sum(), held.sum()) != (7992, 1998, 4995):
            raise ValueError("Matched fold coverage differs")
        np.savez_compressed(
            out / f"{room}-indices.npz", training=np.flatnonzero(train), validation=np.flatnonzero(val), held=np.flatnonzero(held),
        )
        mean = a["pose"][train].mean(0, dtype="float64")
        prior = np.broadcast_to(mean, a["pose"].shape)
        row = {"held_room": room, "feature": "source_mean", "kernel": "none", "alpha": None, "shrink": 0.0}
        for label, mask in (("training", train), ("validation", val), ("held", held)):
            row[label] = analyze(prior[mask], a["pose"][mask], a["valid"][mask], a["sequence"][mask])
        rows.append(row)
        for name, raw in features.items():
            for kernel in ("linear", "rbf512"):
                design, state = fit_design(raw, train, kernel)
                best, candidates, saved_weights = None, [], []
                for alpha, weights, target_mean, residual in ridge_solutions(design, a["pose"], train):
                    saved_weights.append(weights)
                    delta = multiply(design, weights).reshape(-1, 17, 2)
                    for shrink in SHRINK:
                        raw_prediction = target_mean.reshape(17, 2) + shrink * delta
                        prediction = raw_prediction.clip(0, 1)
                        val_scores = scores(prediction[val], a["pose"][val], a["valid"][val])
                        candidate = {"alpha": alpha, "shrink": shrink, "source_validation": val_scores, "normal_equation_relative_residual": residual}
                        candidates.append(candidate)
                        if best is None or val_scores["normalized_joint_error"] < best[0]:
                            best = (val_scores["normalized_joint_error"], alpha, shrink, prediction, raw_prediction, target_mean)
                _, alpha, shrink, prediction, raw_prediction, target_mean = best
                prefix = f"{room}-{name}-{kernel}"
                np.savez_compressed(
                    out / (prefix + "-weights.npz"), **state, weights=np.stack(saved_weights), alphas=np.asarray(ALPHAS),
                    target_mean=target_mean, selected_alpha=np.asarray(alpha), selected_shrink=np.asarray(shrink),
                )
                np.savez_compressed(
                    out / (prefix + "-predictions.npz"),
                    training_indices=np.flatnonzero(train), validation_indices=np.flatnonzero(val), held_indices=np.flatnonzero(held),
                    training=prediction[train], validation=prediction[val], held=prediction[held], held_raw=raw_prediction[held],
                )
                row = {"held_room": room, "feature": name, "kernel": kernel, "alpha": alpha, "shrink": shrink, "candidates": candidates,
                       "held_coordinate_clipping_fraction": float(np.mean((raw_prediction[held] < 0) | (raw_prediction[held] > 1)))}
                for label, mask in (("training", train), ("validation", val), ("held", held)):
                    row[label] = analyze(prediction[mask], a["pose"][mask], a["valid"][mask], a["sequence"][mask])
                rows.append(row)
                print(json.dumps({"room": room, "feature": name, "kernel": kernel, "alpha": alpha, "shrink": shrink,
                                  "held_nje": row['held']['metrics']['normalized_joint_error'],
                                  "held_relative_r2": row['held']['motion']['relative']['within_record_r2']}), flush=True)
    for name, value in hashes.items():
        if digest(cache / (name + ".npy")) != value:
            raise ValueError("Cache mutated during probe")
    for name, value in source_hashes.items():
        if digest(ROOT / name) != value:
            raise ValueError("Source mutated during probe")
    historical = json.loads((ROOT / "work/domain-neural-v2/results.json").read_text())
    write(out / "historical-neural-results.json", historical)
    write(out / "results.json", {"registration": registration, "results": rows, "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.cache, args.out)
