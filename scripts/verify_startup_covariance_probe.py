"""Independently audit saved local startup features, readouts and controls.

Requires the separately prepared consented dataset-derived experiment artifacts.
This verifier does not fit a model, change frozen receipts or access raw RF/E04.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from scipy.linalg import eigvalsh, solve
from scipy.linalg.blas import dgemm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.domain_startup_covariance_probe import FULL_CONFIG, LOCAL_CONFIG  # noqa: E402
from scripts.research_numeric_readback import numeric_difference  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mm(a, b):
    return dgemm(alpha=1, a=np.asarray(a, dtype=float), b=np.asarray(b, dtype=float))


def metrics(prediction, target):
    p, t = (np.asarray(prediction, dtype=float), np.asarray(target, dtype=float))
    if (
        p.shape != t.shape
        or t.ndim != 3
        or t.shape[1:] != (17, 2)
        or (not len(t))
        or (not np.isfinite(p).all())
        or (not np.isfinite(t).all())
    ):
        raise ValueError("Expected finite matching [N,17,2] poses")
    scale = np.linalg.norm(t.max(1) - t.min(1), axis=1)
    pc = p - p.mean(0)
    tc = t - t.mean(0)
    energy = float(np.square(tc).sum())
    if np.any(scale <= 0) or energy <= 0:
        raise ValueError("Undefined normalization or motion R2")
    err = np.linalg.norm(p - t, axis=2) / scale[:, None]
    return {
        "nje": float(err.mean()),
        "pck": {str(a): float((err <= a).mean()) for a in (0.05, 0.1, 0.2)},
        "r2": 1 - float(np.square(pc - tc).sum()) / energy,
    }


def independent_features(current, coordinates, anchor_index):
    c, coords = (np.asarray(current), np.asarray(coordinates))
    if (
        c.ndim != 5
        or c.shape[1:] != (8, 19, 3, 3)
        or coords.shape != (*c.shape[:-2], 9)
        or (not 0 <= anchor_index < len(c))
        or (not np.isfinite(c).all())
        or (not np.isfinite(coords).all())
    ):
        raise ValueError("Invalid covariance episode or anchor")
    np.testing.assert_allclose(c, c.conj().swapaxes(-1, -2), atol=1e-12, rtol=1e-12)
    e = np.linalg.eigvalsh(c)
    if np.any(e <= 0):
        raise ValueError("Strictly positive covariance required")
    shape = c / np.exp(np.log(e).mean(-1))[..., None, None]
    anchor = shape[anchor_index].mean(0)
    anchor = (anchor + anchor.swapaxes(-1, -2).conj()) / 2
    b = anchor / np.exp(np.log(np.linalg.eigvalsh(anchor)).mean(-1))[..., None, None]
    vmean = np.log(e[anchor_index]).mean(-1).mean(0)
    features = {
        "startup_shape": np.empty((len(c), 456)),
        "startup_shape_volume": np.empty((len(c), 608)),
        "raw_current": coords.reshape(len(c), 1368).copy(),
        "startup_frozen_raw": np.broadcast_to(
            coords[anchor_index].reshape(-1), (len(c), 1368)
        ).copy(),
    }
    for i in range(len(c)):
        a = shape[i]
        ref = (a + b) / 2
        aa = a + 0.0001 * ref
        bb = b + 0.0001 * ref
        logs = np.empty((8, 19, 3))
        for timeidx in range(8):
            for frequency in range(19):
                logs[timeidx, frequency] = np.log(
                    eigvalsh(
                        aa[timeidx, frequency], bb[timeidx, frequency], driver="gv"
                    )
                )
        logs -= logs.mean(-1, keepdims=True)
        volume = np.log(e[i]).mean(-1) - vmean
        features["startup_shape"][i] = logs.reshape(-1)
        features["startup_shape_volume"][i] = np.concatenate(
            (logs, volume[..., None]), axis=-1
        ).reshape(-1)
    return (features, anchor, vmean)


NORMALIZER_TOLERANCES = {
    "mean": dict(rtol=2e-14, atol=1e-10),
    "scale": dict(rtol=2e-14, atol=1e-10),
    "training_design": dict(rtol=2e-13, atol=1e-11),
    "bandwidth_squared": dict(rtol=2e-13, atol=1e-9),
}


def check_normalizer(observed, expected, field):
    """Reject malformed or nonfinite saved state before using recomputed state."""
    actual = np.asarray(observed, dtype="float64")
    expected = np.asarray(expected, dtype="float64")
    if (
        actual.shape != expected.shape
        or not np.isfinite(actual).all()
        or not np.isfinite(expected).all()
        or not np.allclose(actual, expected, **NORMALIZER_TOLERANCES[field])
    ):
        raise ValueError("Saved normalization mismatch: " + field)
    return float(np.max(np.abs(actual - expected)))


def verify(base, pre, output):
    base, pre, output = (Path(base), Path(pre), Path(output))
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    reg = read(base / "registration.json")
    if not (reg["full_recipe"] == FULL_CONFIG and reg["local_recipe"] == LOCAL_CONFIG):
        raise ValueError(
            "Local audit mismatch: reg['full_recipe'] == FULL_CONFIG and reg['local_recipe'] == LOCAL_CONFIG"
        )
    for name, h in reg["source_sha256"].items():
        if not (sha(ROOT / name) == h and sha(base / "executed-source" / name) == h):
            raise ValueError(
                "Local audit mismatch: sha(ROOT / name) == h and sha(base / 'executed-source' / name) == h"
            )
    for name, h in reg["consumed_predecessor_sha256"].items():
        if not sha(pre / name) == h:
            raise ValueError("Local audit mismatch: sha(pre / name) == h")
    if (
        not sha(pre / "artifact-manifest-final.json")
        == reg["predecessor_manifest_sha256"]
    ):
        raise ValueError(
            "Local audit mismatch: sha(pre / 'artifact-manifest-final.json') == reg['predecessor_manifest_sha256']"
        )
    with np.load(pre / "indices.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    x = np.load(pre / "untreated_covariance-features.npy", allow_pickle=False)
    coords = x.reshape(-1, 8, 19, 7, 9)[:, :, :, 0]
    c = np.zeros((*coords.shape[:-1], 3, 3), complex)
    c[..., (0, 1, 2), (0, 1, 2)] = coords[..., :3]
    o = (coords[..., 3:6] + 1j * coords[..., 6:9]) / np.sqrt(2)
    c[..., (0, 0, 1), (1, 2, 2)] = o
    c[..., (1, 2, 2), (0, 0, 1)] = o.conj()
    e = np.linalg.eigvalsh(c)
    if not np.all(e > 0):
        raise ValueError("Local audit mismatch: np.all(e > 0)")
    y = data["target"] - data["target"][:, [11, 12]].mean(1, keepdims=True)
    train, val, test = (data["split"] == k for k in ("train", "validation", "test"))
    if not (train.sum(), val.sum(), test.sum()) == (79, 13, 72):
        raise ValueError(
            "Local audit mismatch: (train.sum(), val.sum(), test.sum()) == (79, 13, 72)"
        )
    for a, b in ((train, val), (train, test), (val, test)):
        if not not set(data["raw_support"][a].ravel()) & set(
            data["raw_support"][b].ravel()
        ):
            raise ValueError(
                "Local audit mismatch: not set(data['raw_support'][a].ravel()) & set(data['raw_support'][b].ravel())"
            )
    with np.load(base / "targets-and-rows.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(z["articulation_target"], y)
    identities = read(base / "startup-identities.json")
    freeze = read(base / "FROZEN-LOCAL-SELECTIONS.json")
    results = read(base / "results.json")
    perm = np.random.default_rng(90210).permutation(test.sum())
    np.testing.assert_array_equal(perm, np.load(base / "test-permutation.npy"))
    max_anchor = max_feature = max_normalization = max_residual = (
        max_direct_prediction
    ) = max_saved_prediction = max_metric = 0.0
    models = equations = 0
    independent_scores = {}

    def check_score(observed, expected):
        nonlocal max_metric
        vals = [
            numeric_difference(
                expected["metrics"]["normalized_joint_error"],
                observed["nje"],
                name="metric NJE",
                atol=1e-11,
            ),
            numeric_difference(
                expected["record_articulation_r2"],
                observed["r2"],
                name="metric R2",
                atol=1e-11,
            ),
        ]
        vals.extend(
            numeric_difference(
                expected["metrics"]["pck_curve"][key],
                observed["pck"][key],
                name="metric PCK " + key,
                atol=1e-11,
            )
            for key in observed["pck"]
        )
        max_metric = max(max_metric, *vals)

    for variant in ("continuous", "reset_stress"):
        episode = data["records"] if variant == "continuous" else data["split"]
        start = {
            name: np.flatnonzero(episode == name)[
                np.argmin(data["end"][episode == name])
            ]
            for name in np.unique(episode)
        }
        if variant == "continuous":
            if not (list(start.values()) == [0] and data["split"][0] == "train"):
                raise ValueError(
                    "Local audit mismatch: list(start.values()) == [0] and data['split'][0] == 'train'"
                )
        feat = {
            name: np.empty((164, width))
            for name, width in [
                ("startup_shape", 456),
                ("startup_shape_volume", 608),
                ("raw_current", 1368),
                ("startup_frozen_raw", 1368),
            ]
        }
        for name, j in start.items():
            ids = np.flatnonzero(episode == name)
            anchor_index = int(np.flatnonzero(ids == j)[0])
            episode_features, anchor, vmean = independent_features(
                c[ids], coords[ids], anchor_index
            )
            with np.load(
                base / variant / (str(name).replace("/", "-") + "-startup.npz"),
                allow_pickle=False,
            ) as z:
                max_anchor = max(
                    max_anchor,
                    numeric_difference(
                        z["anchors"], anchor[None], name="startup anchors", atol=1e-12
                    ),
                    numeric_difference(
                        z["mean_log_volume"],
                        vmean,
                        name="startup mean_log_volume",
                        atol=1e-12,
                    ),
                )
                np.testing.assert_array_equal(z["raw_current"], coords[j].reshape(-1))
            if not identities[variant][str(name)]["current_state_ends"] == list(
                range(int(data["end"][j]) - 7, int(data["end"][j]) + 1)
            ):
                raise ValueError(
                    "Local audit mismatch: identities[variant][str(name)]['current_state_ends'] == list(range(int(data['end'][j]) - 7, int(data['end'][j]) + 1))"
                )
            if not identities[variant][str(name)][
                "conservative_validity_support"
            ] == list(range(int(data["end"][j]) - 39, int(data["end"][j]) + 1)):
                raise ValueError(
                    "Local audit mismatch: identities[variant][str(name)]['conservative_validity_support'] == list(range(int(data['end'][j]) - 39, int(data['end'][j]) + 1))"
                )
            if not np.all(data["end"][ids] >= data["end"][j]):
                raise ValueError(
                    "Local audit mismatch: np.all(data['end'][ids] >= data['end'][j])"
                )
            for arm in feat:
                feat[arm][ids] = episode_features[arm]
        for arm, rebuilt in feat.items():
            saved = np.load(base / variant / (arm + "-features.npy"))
            max_feature = max(
                max_feature,
                numeric_difference(
                    saved, rebuilt, name="saved feature " + arm, atol=1e-11
                ),
            )
            mean = saved[train].mean(0)
            sd = np.maximum(saved[train].std(0), 0.05)
            z = (saved - mean) / sd
            with np.load(base / variant / (arm + "-normalization.npz")) as norm:
                max_normalization = max(
                    max_normalization,
                    check_normalizer(norm["mean"], mean, "mean"),
                    check_normalizer(norm["scale"], sd, "scale"),
                    check_normalizer(
                        norm["training_design"], z[train], "training_design"
                    ),
                )
                if arm == "startup_frozen_raw":
                    max_normalization = max(
                        max_normalization,
                        check_normalizer(
                            norm["bandwidth_squared"],
                            np.asarray(0.0),
                            "bandwidth_squared",
                        ),
                    )
                    if not (
                        norm["bandwidth_squared"] == 0
                        and np.max(np.ptp(saved[train], axis=0)) == 0
                    ):
                        raise ValueError(
                            "Local audit mismatch: norm['bandwidth_squared'] == 0 and np.max(np.ptp(saved[train], axis=0)) == 0"
                        )
                    key = variant + "/" + arm + "/constant"
                    prediction = np.broadcast_to(y[train].mean(0), y.shape)
                    with np.load(
                        base / (key.replace("/", "-") + "-predictions.npz")
                    ) as p:
                        numeric_difference(
                            p["prediction"],
                            prediction,
                            name="constant saved prediction",
                            atol=0.0,
                        )
                    for role, mask in (
                        ("train", train),
                        ("validation", val),
                        ("test", test),
                    ):
                        check_score(
                            metrics(prediction[mask], y[mask]),
                            results["results"][key][role],
                        )
                    independent_scores[key] = {
                        "test": metrics(prediction[test], y[test]),
                        "shuffle": metrics(prediction[test][perm], y[test]),
                    }
                    continue
                cross = mm(z, z[train].T)
                distance = np.maximum(
                    np.square(z).sum(1)[:, None]
                    + np.square(z[train]).sum(1)[None]
                    - 2 * cross,
                    0,
                )
                width = float(
                    np.median(distance[train][np.triu_indices(train.sum(), 1)])
                )
                max_normalization = max(
                    max_normalization,
                    check_normalizer(
                        norm["bandwidth_squared"],
                        np.asarray(width),
                        "bandwidth_squared",
                    ),
                )
            ks = {
                "linear": cross / saved.shape[1],
                "rbf": np.exp(-distance / (2 * width)),
            }
            for kind, kernel in ks.items():
                key = variant + "/" + arm + "/" + kind
                with np.load(
                    base / variant / (arm + "-" + kind + "-model.npz")
                ) as zstate:
                    state = {k: zstate[k] for k in zstate.files}
                yy = y[train].reshape(train.sum(), 34)
                ym = yy.mean(0)
                np.testing.assert_array_equal(ym, state["target_mean"])
                candidates = []
                for ai, alpha in enumerate(reg["local_recipe"]["lambdas"]):
                    system = kernel[train] + alpha * np.eye(train.sum())
                    right = yy - ym
                    w = state["weights"][ai]
                    max_residual = max(
                        max_residual,
                        float(
                            np.linalg.norm(mm(system, w) - right)
                            / np.linalg.norm(right)
                        ),
                    )
                    direct = solve(system, right, assume_a="pos", check_finite=True)
                    directp = (mm(kernel, direct) + ym).reshape(-1, 17, 2)
                    savedp = (mm(kernel, w) + ym).reshape(-1, 17, 2)
                    max_direct_prediction = max(
                        max_direct_prediction,
                        numeric_difference(
                            savedp, directp, name="candidate prediction", atol=1e-7
                        ),
                    )
                    valnje = metrics(savedp[val], y[val])["nje"]
                    candidates.append(valnje)
                    expected = freeze["candidates"][key][ai]
                    if not abs(valnje - expected["validation_nje"]) < 1e-12:
                        raise ValueError(
                            "Local audit mismatch: abs(valnje - expected['validation_nje']) < 1e-12"
                        )
                    equations += 1
                selected = min(range(len(candidates)), key=candidates.__getitem__)
                if not selected == state["selected"]:
                    raise ValueError(
                        "Local audit mismatch: selected == state['selected']"
                    )
                prediction = (mm(kernel, state["weights"][selected]) + ym).reshape(
                    -1, 17, 2
                )
                with np.load(
                    base / (key.replace("/", "-") + "-predictions.npz")
                ) as pred:
                    max_saved_prediction = max(
                        max_saved_prediction,
                        numeric_difference(
                            pred["prediction"],
                            prediction,
                            name="saved prediction",
                            atol=1e-11,
                        ),
                    )
                for role, mask in (
                    ("train", train),
                    ("validation", val),
                    ("test", test),
                ):
                    check_score(
                        metrics(prediction[mask], y[mask]),
                        results["results"][key][role],
                    )
                check_score(
                    metrics(prediction[test][perm], y[test]),
                    results["results"][key]["shuffle"],
                )
                independent_scores[key] = {
                    "validation": metrics(prediction[val], y[val]),
                    "test": metrics(prediction[test], y[test]),
                    "shuffle": metrics(prediction[test][perm], y[test]),
                }
                models += 1
        primary = [
            k for k in independent_scores if k.startswith(variant + "/startup_shape/")
        ]
        chosen = min(
            primary, key=lambda key: independent_scores[key]["validation"]["nje"]
        )
        if not chosen == freeze["selected_per_variant"][variant]:
            raise ValueError(
                "Local audit mismatch: chosen == freeze['selected_per_variant'][variant]"
            )
        p = independent_scores[chosen]
        s = independent_scores[variant + "/startup_frozen_raw/constant"]["test"]
        passed = (
            p["test"]["r2"] >= 0.5
            and p["test"]["r2"] - p["shuffle"]["r2"] >= 0.3
            and (p["test"]["nje"] <= 0.75 * s["nje"])
            and (p["test"]["nje"] < s["nje"])
        )
        if not bool(passed) == results["local_gates"][variant]["passed"]:
            raise ValueError(
                "Local audit mismatch: bool(passed) == results['local_gates'][variant]['passed']"
            )
    rawkeys = [k for k in independent_scores if k.startswith("continuous/raw_current/")]
    raw = min(rawkeys, key=lambda key: independent_scores[key]["validation"]["nje"])
    old = read(pre / "results.json")
    oldrows = [z for z in old["results"] if z["feature"] == "congruence_dynamics"]
    oldchoice = min(
        oldrows, key=lambda z: z["validation"]["metrics"]["normalized_joint_error"]
    )
    oldname = "congruence_dynamics-" + oldchoice["kernel"] + "-predictions.npz"
    oldmanifest = read(pre / "artifact-manifest-final.json")
    oldmanifest = oldmanifest.get("sha256", oldmanifest)
    if not sha(pre / oldname) == oldmanifest[oldname]:
        raise ValueError(
            "Local audit mismatch: sha(pre / oldname) == oldmanifest[oldname]"
        )
    with np.load(pre / oldname) as z:
        p = z["prediction"]
    p = p - p[:, [11, 12]].mean(1, keepdims=True)
    oldscore = metrics(p[test], y[test])
    if not (
        max_anchor < 1e-12
        and max_feature < 1e-11
        and np.isfinite(max_normalization)
        and max_normalization < 1e-8
        and (max_residual < 1e-07)
        and (max_saved_prediction < 1e-11)
        and (max_direct_prediction < 1e-07)
        and (max_metric < 1e-11)
    ):
        raise ValueError("Local audit mismatch: numerical acceptance thresholds")
    report = {
        "scope": "Actual local fixedA12 verification only; no full SOURCE fit/rawRF/GPU/E04",
        "primary_continuous_gate": results["local_gates"]["continuous"],
        "reset_stress_gate": results["local_gates"]["reset_stress"],
        "independent_reference": "Hermitian coordinate inversion; SciPy generalized symmetric eigenvalues instead of Cholesky whitening; direct positive-definite solves instead of source eigendecomposition",
        "startup_anchors_verified": 4,
        "spectral_feature_rows_verified": 328,
        "models_verified": models,
        "all_candidate_equations_verified": equations,
        "constant_controls_verified": 2,
        "maximum_anchor_difference": max_anchor,
        "maximum_feature_difference": max_feature,
        "maximum_normalization_difference": max_normalization,
        "saved_normalizer_fields_verified": [
            "mean",
            "scale",
            "training_design",
            "bandwidth_squared",
        ],
        "saved_normalizer_field_tolerances": NORMALIZER_TOLERANCES,
        "maximum_normal_equation_relative_residual": max_residual,
        "maximum_direct_solver_prediction_difference": max_direct_prediction,
        "maximum_saved_prediction_difference": max_saved_prediction,
        "maximum_metric_difference": max_metric,
        "sourceval_selections_and_gates_match": True,
        "controls": {
            "static_mean": metrics(
                np.broadcast_to(y[train].mean(0), y[test].shape), y[test]
            ),
            "raw_current_sourceval_selected": {
                "method": raw,
                **independent_scores[raw],
            },
            "old_temporal_frozen_original_selector": {
                "method": oldname,
                "score": oldscore,
                "limit": "Original absolute-pose validation NJE selected this frozen comparator; predictions were originally clipped to image[0,1], then hip-relative projected only for matched articulation scoring. Same79/13/72 observation endpoints and40-frame validity context; unlike continuous startup, old temporal has no shared startup summary.",
            },
        },
        "consumed_inputs_sha256": reg["consumed_predecessor_sha256"],
        "source_sha256": reg["source_sha256"],
        "verification_source_sha256": sha(Path(__file__)),
        "numeric_readback_source_sha256": sha(
            ROOT / "scripts/research_numeric_readback.py"
        ),
    }
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument(
        "--predecessor", type=Path, default=ROOT / "work/domain-covariance-positive-v1"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.experiment, args.predecessor, args.output)
    print(json.dumps(report, indent=2, allow_nan=False))
