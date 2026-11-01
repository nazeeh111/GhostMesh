"""Bundle/reproduce the frozen local-motion diagnostic. No E04 or transfer claim."""

import argparse
import json
from hashlib import sha256
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.data import sha256 as file_digest  # noqa:E402
from ghostmesh.local_kernel import KIND, design, infer_recording  # noqa:E402


def bundle(cache, experiment, out):
    if out.exists():
        raise FileExistsError(out)
    receipt = json.loads((cache / "receipt.json").read_text())
    registration = json.loads((experiment / "registration.json").read_text())
    if file_digest(cache / "receipt.json") != registration["cache_receipt_sha256"]:
        raise ValueError("Source cache receipt changed")
    manifest = json.loads((experiment / "artifact-manifest-final.json").read_text())[
        "sha256"
    ]
    for name in (
        "held_time-quotient-rbf-weights.npz",
        "held_time-quotient-rbf-predictions.npz",
        "held_time-quotient-indices.npz",
        "selected-record-bindings.json",
        "results.json",
        "verification.json",
    ):
        if file_digest(experiment / name) != manifest[name]:
            raise ValueError("Frozen experiment artifact changed: " + name)
    row = next(r for r in receipt["records"] if r["id"] == "E01/S01/A12")
    binding = json.loads((experiment / "selected-record-bindings.json").read_text())[
        row["id"]
    ]
    sl = slice(row["offset"], row["offset"] + 297)
    rf = np.load(cache / "rf.npy", mmap_mode="r", allow_pickle=False)[sl].copy()
    labels = np.load(cache / "pose.npy", mmap_mode="r", allow_pickle=False)[sl].copy()
    if (
        sha256(rf.tobytes()).hexdigest() != binding["cached_rf_sha256"]
        or sha256(labels.tobytes()).hexdigest() != binding["cached_pose_sha256"]
    ):
        raise ValueError("Actual selected RF/labels changed")
    weights = np.load(
        experiment / "held_time-quotient-rbf-weights.npz", allow_pickle=False
    )
    training = (design(rf[:118]) - weights["input_mean"]) / weights["input_scale"]
    np.testing.assert_array_equal(training, weights["training_design"])
    selected = np.flatnonzero(weights["lambdas"] == weights["selected_lambda"])
    if len(selected) != 1:
        raise ValueError("Ambiguous frozen lambda")
    out.mkdir(parents=True)
    model_path, sample_path = out / "local-kernel.npz", out / "local-csi.npz"
    np.savez_compressed(
        model_path,
        kind=np.asarray(KIND),
        magnitude_units=np.asarray("log_magnitude_db"),
        phase_units=np.asarray("radians"),
        reference_rf=rf[:118],
        input_mean=weights["input_mean"],
        input_scale=weights["input_scale"],
        dual=weights["dual"][selected[0]],
        target_mean=weights["target_mean"],
        bandwidth_squared=weights["bandwidth_squared"],
    )
    np.savez_compressed(
        sample_path,
        rf=rf[186:],
        timestamps=np.arange(186, 297) / 10.0,
        magnitude_units=np.asarray("log_magnitude_db"),
        phase_units=np.asarray("radians"),
    )
    predicted = infer_recording(model_path, sample_path, out / "inference-only.json")
    valid = np.array([f["pose"] for f in predicted["frames"][7:]])
    with np.load(
        experiment / "held_time-quotient-indices.npz", allow_pickle=False
    ) as indices:
        test = indices["split"] == "test"
    expected = np.load(
        experiment / "held_time-quotient-rbf-predictions.npz", allow_pickle=False
    )["prediction"][test]
    difference = float(abs(valid - expected).max())
    if difference > 1e-9:
        raise ValueError("Compact checkpoint does not reproduce frozen predictions")
    predicted["provenance"].update(
        kind="dataset_inference",
        dataset="MM-Fi (NeurIPS2023)",
        subject="S01",
        environment="E01",
        sequence="E01/S01/A12",
        split="same-condition held-time diagnostic",
        evaluation_scope="same_condition_held_time",
        timestamp_basis="Nominal10Hz reconstructed from dataset frame indices",
        context_seconds=0.7,
        note="Same person, room and action as111 reference windows. Dense references span11.8s; separated104 held-time windows. This does not establish room/person/action transfer or a small calibration budget. Query CSI only; reference overlays added separately.",
    )
    # Overlays are attached only AFTER the label-free inference result is saved.
    for frame, truth in zip(predicted["frames"], labels[186:]):
        frame["ground_truth"] = truth.tolist()
        frame["ground_truth_valid"] = [True] * 17
    rows = json.loads((experiment / "results.json").read_text())["results"]
    result = next(
        r
        for r in rows
        if r.get("protocol") == "held_time"
        and r.get("feature") == "quotient"
        and r.get("kernel") == "rbf"
    )
    predicted["metrics"] = {
        **result["test"]["metrics"],
        "scope": "Same E01/S01/A12 held-time only; not room/person/action transfer",
    }
    predicted["metrics"].pop("per_frame_nje", None)
    (out / "replay.json").write_text(
        json.dumps(predicted, allow_nan=False, separators=(",", ":")) + "\n"
    )
    report = dict(
        valid_predictions=104,
        compact_vs_frozen_prediction_max_abs=difference,
        experiment_results_sha256=file_digest(experiment / "results.json"),
        scope="Same person/room/action; no uncertainty estimator, no transfer claim",
        files_sha256={
            p.name: file_digest(p) for p in sorted(out.iterdir()) if p.is_file()
        },
    )
    (out / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("bundle")
    b.add_argument("--cache", type=Path, default=ROOT / "work/domain-source30-rf-v1")
    b.add_argument(
        "--experiment", type=Path, default=ROOT / "work/domain-same-condition-v1"
    )
    b.add_argument("--out", type=Path, required=True)
    i = sub.add_parser("infer")
    i.add_argument("--model", type=Path, required=True)
    i.add_argument("--recording", type=Path, required=True)
    i.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "bundle":
        bundle(args.cache, args.experiment, args.out)
    else:
        infer_recording(args.model, args.recording, args.out)
