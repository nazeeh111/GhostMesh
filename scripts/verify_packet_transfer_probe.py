"""Independently audit raw or normalized packet-spectrum transfer artifacts.

Rebuild source-only designs and validation choices, directly solve all ridge
systems, replay every prediction, and recheck shuffled/static/clock/covariance
controls. Raw mode additionally checks every raw annotation join and three
fixed actual RF windows per record (all three spectral variants). Shape mode
checks EVERY pointwise cache transformation and the fixed raw comparison.
No fitting changes, new hyperparameters, query adaptation or new data.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import domain_packet_transfer_probe as raw  # noqa:E402
from scripts import domain_spectral_shape_probe as shape  # noqa:E402
from ghostmesh.doppler_features import complex_quotients, spectral_features  # noqa:E402


def load_npz(path):
    with np.load(path, allow_pickle=False) as value:
        return dict(value)


def verify_systems(design, y, model, train, val):
    """Check actual source statistics, direct solutions and four head selections."""
    train, val = np.asarray(train, bool), np.asarray(val, bool)
    if np.any(train & val) or not train.any() or not val.any():
        raise ValueError("Nonempty disjoint source roles required")
    if y.shape != (len(design), 104):
        raise ValueError("Expected four registered target blocks")
    np.testing.assert_array_equal(model["alphas"], raw.CONFIG["alphas"])
    np.testing.assert_array_equal(model["x_mean"], design[train].mean(0))
    np.testing.assert_array_equal(model["y_mean"], y[train].mean(0))
    x = design[train] - model["x_mean"]
    rhs = raw.multiply(x.T, y[train] - model["y_mean"])
    gram = raw.multiply(x.T, x)
    validation = {name: [] for name in raw.SLICES}
    max_residual = max_direct = 0.0
    if model["weights"].shape != (5, design.shape[1], 104):
        raise ValueError("Wrong registered ridge checkpoint dimensions")
    for alpha, weight in zip(model["alphas"], model["weights"]):
        system = gram + train.sum() * alpha * np.eye(design.shape[1])
        residual = float(
            np.linalg.norm(raw.multiply(system, weight) - rhs)
            / max(np.linalg.norm(rhs), 1e-12)
        )
        direct = np.linalg.solve(system, rhs)
        difference = float(
            np.linalg.norm(direct - weight) / max(np.linalg.norm(weight), 1e-12)
        )
        if not np.isfinite(residual + difference) or max(residual, difference) > 1e-6:
            raise ValueError("Stored coefficients fail independent direct ridge solve")
        max_residual, max_direct = (
            max(max_residual, residual),
            max(max_direct, difference),
        )
        p = raw.multiply(design[val] - model["x_mean"], weight) + model["y_mean"]
        for name, sl in raw.SLICES.items():
            value = (
                raw.current_metrics(
                    p[:, sl].reshape(-1, 17, 2).clip(0, 1),
                    y[val, sl].reshape(-1, 17, 2),
                )["normalized_joint_error"]
                if name == "pose_full"
                else float(np.square(p[:, sl] - y[val, sl]).mean())
            )
            validation[name].append(value)
    for i, name in enumerate(raw.SLICES):
        if int(model["selected_index"][i]) != int(np.argmin(validation[name])):
            raise ValueError("Stored checkpoint did not minimize source validation")
    return max_residual, max_direct, validation


def refuse_report_collision(report):
    if report.exists():
        raise FileExistsError(report)


def verify(
    mode,
    out,
    cache,
    baseline,
    report,
    raw_cache=None,
    raw_experiment=None,
    base=None,
    cov=None,
):
    refuse_report_collision(report)
    study = raw if mode == "raw" else shape
    r = result = json.loads((out / "results.json").read_text())
    reg = json.loads((out / "registration.json").read_text())
    for name, expected in json.loads(
        (out / "artifact-manifest.json").read_text()
    ).items():
        if raw.digest(out / name) != expected:
            raise ValueError("Original fit artifact changed: " + name)
    for name, expected in reg["source_sha256"].items():
        if (
            raw.digest(ROOT / name) != expected
            or raw.digest(out / "executed-source" / name) != expected
        ):
            raise ValueError("Frozen source changed: " + name)
    receipt, data, features = study.load_cache(cache)
    if (
        raw.digest(cache / "receipt.json") != reg["spectral_cache_receipt_sha256"]
        or receipt["arrays_sha256"] != reg["spectral_cache_arrays_sha256"]
    ):
        raise ValueError("Registered cache binding differs")
    for name, expected in reg["baseline_sha256"].items():
        if raw.digest(baseline / name) != expected:
            raise ValueError("Frozen covariance reference changed")
    for name, value in study.CONFIG.items():
        if result["config"][name] != json.loads(json.dumps(value)):
            raise ValueError("Executed configuration differs: " + name)
    rows_checked = rf_windows = 0
    if mode == "raw":
        if base is None or cov is None:
            raise ValueError("Raw mode requires raw source and covariance cache")
        with np.load(cov / "targets-and-metadata.npz", allow_pickle=False) as z:
            original = {k: z[k] for k in z.files}
        raw_receipt = json.loads((base / "receipt.json").read_text())
        raw.validate_records(raw_receipt["records"])
        cache_registration = receipt["registration"]
        if (
            raw.digest(base / "receipt.json")
            != cache_registration["raw_receipt_sha256"]
            or raw_receipt["arrays"] != cache_registration["raw_arrays_sha256"]
            or raw.digest(cov / "receipt.json")
            != cache_registration["covariance_receipt_sha256"]
            or raw.digest(cov / "targets-and-metadata.npz")
            != cache_registration["covariance_metadata_sha256"]
        ):
            raise ValueError("Raw RF/annotation/covariance cache lineage differs")
        actual_rf = np.load(base / "rf.npy", mmap_mode="r", allow_pickle=False)
        pose = np.load(base / "pose.npy", mmap_mode="r", allow_pickle=False)
        for name, expected in raw_receipt["arrays"].items():
            if study.digest(base / (name + ".npy")) != expected:
                raise ValueError("Actualsource bytes changed")
        rows_checked = rf_windows = 0
        for record in raw_receipt["records"]:
            ids = np.flatnonzero(data["record"] == record["id"])
            ends = data["end"][ids]
            offset = record["offset"]
            index = data["covariance_index"][ids]
            joined = study.join_labels(
                np.asarray(pose[offset : offset + 297], dtype=np.float64),
                ends,
                original["pose"][index],
                original["delta"][index],
            )
            for name, value in joined.items():
                np.testing.assert_array_equal(value, data[name][ids])
            if len(ids):
                selected = ids[
                    np.unique(np.linspace(0, len(ids) - 1, min(3, len(ids)), dtype=int))
                ]
                q, _ = complex_quotients(
                    actual_rf[offset : offset + 297],
                    magnitude_units="log_magnitude_db",
                    phase_units="radians",
                )
                for row in selected:
                    end = int(data["end"][row])
                    for variant in study.VARIANTS:
                        np.testing.assert_array_equal(
                            spectral_features(q[end - 7 : end + 1], variant).astype(
                                "float32"
                            ),
                            features[variant][row],
                        )
                rf_windows += len(selected)
            rows_checked += len(ids)
            if record["id"].endswith("/A27") and int(record["id"][5:7]) % 5 == 0:
                print("Verifiedsource", record["id"], flush=True)

    else:
        if raw_cache is None or raw_experiment is None:
            raise ValueError(
                "Shape mode requires original spectral cache and raw experiment"
            )
        rr, rd, rf = raw.load_cache(raw_cache)
        for name, value in rd.items():
            np.testing.assert_array_equal(value, data[name])
        for variant in study.VARIANTS:
            np.testing.assert_array_equal(
                study.log_power_fractions(rf[variant]).astype(np.float32),
                features[variant],
            )
            power = (
                10.0
                ** (
                    np.asarray(features[variant], dtype=np.float64).reshape(
                        -1, 2, 19, 10
                    )
                    / 10
                )
            ).sum(-1)
            np.testing.assert_allclose(power, 1, rtol=0, atol=1e-6)
        for name in ("metadata.npz", "quality.json"):
            if raw.digest(cache / name) != raw.digest(raw_cache / name):
                raise ValueError("Shape metadata differs from raw cohort")
        if (
            raw.digest(raw_cache / "receipt.json")
            != receipt["raw_spectral_receipt_sha256"]
        ):
            raise ValueError("Original spectral cache binding differs")
        raw_result = json.loads((raw_experiment / "results.json").read_text())
        if (
            raw.digest(raw_experiment / "results.json")
            != reg["raw_power_reference_results_sha256"]
        ):
            raise ValueError("Fixed raw reference result changed")
        for key in ("selected_natural", "choices"):
            assert result["raw_power_comparison"][key] == raw_result[key]
        for room, methods in raw_result["results"].items():
            for arm, row in methods.items():
                for name in (*study.SLICES, "pose_relative"):
                    assert (
                        result["raw_power_comparison"]["aligned"][room][arm][name]
                        == row["aligned"][name]
                    )
        rows_checked = len(data["room"])
    y = study.target_matrix(data)
    max_residual = max_direct = max_prediction = 0.0
    models = equations = 0
    choices = json.loads((out / "ALL-FOLDS-FROZEN-SELECTIONS.json").read_text())
    assert choices == result["choices"]
    for room in study.CONFIG["folds"]:
        train, val, held = study.fold_masks(data["room"], data["person"], room)
        ids = np.flatnonzero(held)
        for arm, (variant, family) in study.ARMS.items():
            design, mapping = study.fit_map(features[variant], train, family)
            saved_map = load_npz(out / (room + "-" + arm + "-map.npz"))
            for name, value in mapping.items():
                np.testing.assert_array_equal(value, saved_map[name])
            model = load_npz(out / (room + "-" + arm + "-model.npz"))
            residual, difference, validation = verify_systems(
                design, y, model, train, val
            )
            max_residual, max_direct = (
                max(max_residual, residual),
                max(max_direct, difference),
            )
            grid = json.loads((out / (room + "-" + arm + "-grid.json")).read_text())
            for i, alpha in enumerate(model["alphas"]):
                assert grid[i]["alpha"] == float(alpha)
                for name in study.SLICES:
                    assert (
                        grid[i]["source_validation_criteria"][name]
                        == validation[name][i]
                    )
            for i, name in enumerate(study.SLICES):
                index = int(model["selected_index"][i])
                expected = dict(
                    index=index,
                    alpha=float(model["alphas"][index]),
                    validation=validation[name][index],
                )
                assert choices[room]["arms"][arm][name] == expected
            p = study.predict(design[held], model)
            saved = load_npz(out / (room + "-" + arm + "-predictions.npz"))
            np.testing.assert_array_equal(saved["indices"], ids)
            for name, value in p.items():
                max_prediction = max(
                    max_prediction, float(abs(value - saved[name]).max())
                )
            assert (
                study.stratified(p, data, ids)
                == result["results"][room][arm]["aligned"]
            )
            models += 1
            equations += len(model["alphas"])
            print("Verified model", room, arm, flush=True)
        for name in study.SLICES:
            winner = min(
                ("natural_linear", "natural_rff"),
                key=lambda arm: choices[room]["arms"][arm][name]["validation"],
            )
            assert choices[room]["natural_selected"][name] == winner
            assert (
                result["selected_natural"][room][name]
                == result["results"][room][winner]["aligned"][name]
            )
    if models != 12 or equations != 60 or max_prediction > 1e-9:
        raise ValueError("Incomplete replay or changed predictions")
    controls = shuffles = covariance = 0
    for room in study.CONFIG["folds"]:
        train, val, held = study.fold_masks(data["room"], data["person"], room)
        ids = np.flatnonzero(held)
        perm = study.permutation(data["record"][ids])
        np.testing.assert_array_equal(
            perm, np.load(out / (room + "-permutation.npy"), allow_pickle=False)
        )
        for arm in study.ARMS:
            with np.load(
                out / (room + "-" + arm + "-predictions.npz"), allow_pickle=False
            ) as z:
                p = {name: z[name][perm] for name in study.SLICES}
            assert (
                study.stratified(p, data, ids)
                == r["results"][room][arm]["within_record_prediction_shuffle"]
            )
            shuffles += 1
        mean = {
            name: np.repeat(data[name][train].mean(0)[None], len(ids), axis=0)
            for name in study.SLICES
        }
        zero = {
            name: value if name == "pose_full" else np.zeros_like(value)
            for name, value in mean.items()
        }
        endpoint = {name: value.copy() for name, value in mean.items()}
        fallback = []
        for end in np.unique(data["end"][ids]):
            source = np.flatnonzero(train & (data["end"] == end))
            pick = data["end"][ids] == end
            if not len(source):
                fallback.append(int(end))
                continue
            for name in study.SLICES:
                endpoint[name][pick] = data[name][source].mean(0)
        for name, prediction in [
            ("source_mean", mean),
            ("zero_delta", zero),
            ("endpoint_only", endpoint),
        ]:
            assert (
                study.stratified(prediction, data, ids) == r["comparators"][room][name]
            )
            controls += 1
        assert fallback == r["comparators"][room]["endpoint_fallback_ends"]
        index = data["covariance_index"][ids]
        for arm in ("untreated_covariance", "log_shape_tangent", "congruence_dynamics"):
            with np.load(
                baseline / (room + "-" + arm + "-model-prediction.npz"),
                allow_pickle=False,
            ) as z:
                lookup = {
                    int(original): i for i, original in enumerate(z["held_indices"])
                }
                rows = np.array([lookup[int(i)] for i in index])
                p, d = z["pose"][rows], z["delta"][rows]
            expected = r["comparators"][room]["covariance"][arm]
            assert (
                study.current_metrics(p, data["pose_relative"][ids])
                == expected["pose_relative"]
            )
            assert (
                study.displacement_metrics(
                    d,
                    data["delta_relative"][ids],
                    data["pose_full"][ids],
                    data["record"][ids],
                )
                == expected["delta_relative"]
            )
            covariance += 1

    motion = all(
        result["selected_natural"][room]["delta_full"]["raw_zero_reference_r2"]
        is not None
        and result["selected_natural"][room]["delta_full"]["raw_zero_reference_r2"] > 0
        and result["selected_natural"][room]["delta_full"]["raw_zero_reference_r2"]
        > result["results"][room][choices[room]["natural_selected"]["delta_full"]][
            "within_record_prediction_shuffle"
        ]["delta_full"]["raw_zero_reference_r2"]
        and all(
            result["selected_natural"][room]["delta_full"]["raw_mse"]
            <= result["results"][room][c]["aligned"]["delta_full"]["raw_mse"]
            for c in ("within_frame_shuffle", "whole_window_shuffle")
        )
        for room in study.CONFIG["folds"]
    ) and all(
        any(
            result["selected_natural"][room]["delta_full"]["raw_mse"]
            < result["results"][room][c]["aligned"]["delta_full"]["raw_mse"]
            for room in study.CONFIG["folds"]
        )
        for c in ("within_frame_shuffle", "whole_window_shuffle")
    )
    pose = all(
        result["selected_natural"][room]["pose_full"]["normalized_joint_error"]
        < result["comparators"][room]["source_mean"]["pose_full"][
            "normalized_joint_error"
        ]
        and all(
            result["selected_natural"][room]["pose_full"]["pck_curve"][t]
            >= result["comparators"][room]["source_mean"]["pose_full"]["pck_curve"][t]
            for t in ("0.05", "0.2")
        )
        for room in study.CONFIG["folds"]
    )
    assert (
        motion == result["motion_gate_passed"]
        and pose == result["separate_pose_gate_passed"]
    )
    evidence = dict(
        mode=mode,
        source_label_or_transformed_rows_verified=rows_checked,
        actual_RF_windows_spotverified=rf_windows,
        variants_per_window=3,
        models_verified=models,
        normal_equations_direct_solved=equations,
        maximum_normal_equation_residual=max_residual,
        maximum_independent_directsolve_relative_difference=max_direct,
        maximum_saved_prediction_difference=max_prediction,
        shuffled_model_score_readbacks=shuffles,
        static_zero_endpoint_control_readbacks=controls,
        covariance_relative_only_comparison_readbacks=covariance,
        all_source_maps_statistics_bandwidth_validation_choices_and_reported_metrics_exact=True,
        per_person_action_breakdowns_recomputed=True,
        both_gates_recomputed=True,
        input_registration_sha256=raw.digest(out / "registration.json"),
        results_sha256=raw.digest(out / "results.json"),
        verifier_source_sha256=raw.digest(Path(__file__)),
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x") as handle:
        handle.write(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    print(json.dumps(evidence), flush=True)
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("raw", "shape"), required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="New path; existing reports are never overwritten",
    )
    parser.add_argument(
        "--baseline", type=Path, default=ROOT / "work/domain-covariance-full27-v1"
    )
    parser.add_argument(
        "--raw-cache", type=Path, default=ROOT / "work/domain-packet-full-cache-v1"
    )
    parser.add_argument(
        "--raw-experiment", type=Path, default=ROOT / "work/domain-packet-transfer-v1"
    )
    parser.add_argument(
        "--raw-source", type=Path, default=ROOT / "work/domain-source30-rf-v1"
    )
    parser.add_argument(
        "--covariance-cache",
        type=Path,
        default=ROOT / "work/domain-covariance-full-cache-v1",
    )
    args = parser.parse_args()
    verify(
        args.mode,
        args.experiment,
        args.cache,
        args.baseline,
        args.report,
        args.raw_cache,
        args.raw_experiment,
        args.raw_source,
        args.covariance_cache,
    )
