"""Independent MM-Fi MAT/keypoint adapter and explicit test-fixture generator."""

from __future__ import annotations
import json
from pathlib import Path
import re
import numpy as np
from scipy.io import loadmat
from .data import sha256

JOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
EDGES = [
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
    [5, 6],
    [5, 7],
    [7, 9],
    [6, 8],
    [8, 10],
    [5, 11],
    [6, 12],
    [11, 12],
    [11, 13],
    [13, 15],
    [12, 14],
    [14, 16],
]


def _write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def _manifest(provenance, records, protocol):
    return {
        "schema_version": 1,
        "coordinate_system": "normalized_image",
        "protocol": protocol,
        "joint_names": JOINT_NAMES,
        "edges": EDGES,
        "provenance": provenance,
        "sequences": records,
    }


def split_subject(subject, environment, protocol):
    n = int(subject[1:])
    if protocol == "cross_environment":
        return (
            "test"
            if environment == "E04"
            else ("validation" if n % 5 == 4 else "train")
        )
    if protocol != "cross_subject":
        raise ValueError("Protocol must be cross_subject or cross_environment")
    return "test" if n % 5 == 0 else ("validation" if n % 5 == 4 else "train")


def _frame_map(directory, suffix):
    result = {}
    for path in Path(directory).glob(f"frame*{suffix}"):
        m = re.fullmatch(r"frame(\d+)" + re.escape(suffix), path.name)
        if not m:
            continue
        number = int(m.group(1))
        if number in result:
            raise ValueError(f"Duplicate frame number {number}")
        result[number] = path
    return result


def prepare_mmfi(
    source,
    output,
    *,
    width,
    height,
    fps=10.0,
    label_modality="rgb",
    provenance=None,
    subjects=None,
    actions=None,
):
    """Only 2D per-frame annotations; never slices ground_truth.npy (3D).

    Image dimensions are explicit, as inferring them from keypoint maxima would
    silently rescale each subject. Frame-number joins prevent positional drift.
    """
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to replace prepared data: {output}")
    if min(width, height, fps) <= 0 or label_modality not in {
        "rgb",
        "infra1",
        "infra2",
    }:
        raise ValueError(
            "Positive image dimensions/fps and an MM-Fi 2D label modality required"
        )
    roots = sorted(p.parent for p in source.glob("E*/S*/A*/wifi-csi") if p.is_dir())
    if subjects:
        roots = [p for p in roots if p.parent.name in set(subjects)]
    if actions:
        roots = [p for p in roots if p.name in set(actions)]
    if not roots:
        raise ValueError("No E*/S*/A*/wifi-csi sequences found")
    provenance = provenance or {}
    for key in ("dataset", "source", "license", "consent_basis"):
        if not provenance.get(key):
            raise ValueError(f"Missing dataset provenance {key}")
    provenance = {
        **provenance,
        "kind": "licensed_dataset",
        "label_modality": label_modality,
        "reference_canvas": [width, height],
        "fps": fps,
        "coordinate_note": "Declared annotation reference plane, not recovered camera calibration. Use equal width/height for isotropic preservation of pixel-space distances when the label canvas is undocumented.",
        "label_note": "Supplied 2D keypoints, not independently manually audited; no RGB images loaded.",
    }
    output.mkdir(parents=True)
    records = []
    for root in roots:
        env, subject, action = root.parts[-3:]
        csi_paths = _frame_map(root / "wifi-csi", ".mat")
        pose_paths = _frame_map(root / label_modality, ".npy")
        if not csi_paths or set(csi_paths) != set(pose_paths):
            raise ValueError(f"Unpaired CSI/2D frame IDs at {root}")
        indices = sorted(csi_paths)
        signals, poses, validity, raw_hashes = [], [], [], []
        for i in indices:
            mat = loadmat(csi_paths[i])
            if "CSIamp" not in mat:
                raise ValueError(f"CSIamp missing: {csi_paths[i]}")
            x = np.asarray(mat["CSIamp"], dtype=np.float32)
            if x.shape != (3, 114, 10):
                raise ValueError(f"Unexpected MM-Fi CSIamp shape {x.shape}")
            y = np.asarray(np.load(pose_paths[i], allow_pickle=False), dtype=np.float32)
            if y.shape != (17, 2):
                raise ValueError(
                    f"Expected 17 2D labels, got {y.shape}; no implicit 3D conversion"
                )
            y = y / np.array([width, height], dtype=np.float32)
            valid = np.isfinite(y).all(-1) & ((y >= 0) & (y <= 1)).all(-1)
            signals.append(x)
            poses.append(np.where(valid[:, None], y, 0))
            validity.append(valid)
            raw_hashes.append(
                {
                    "frame": i,
                    "csi_sha256": sha256(csi_paths[i]),
                    "pose_sha256": sha256(pose_paths[i]),
                }
            )
        name = f"{env}-{subject}-{action}.npz"
        np.savez_compressed(
            output / name,
            csi=np.stack(signals),
            pose=np.stack(poses),
            valid=np.stack(validity),
            timestamps=np.array(indices, dtype=np.float64) / fps,
        )
        records.append(
            {
                "id": f"{env}/{subject}/{action}",
                "path": name,
                "subject": subject,
                "environment": env,
                "action": action,
                "split": split_subject(subject, env, "cross_subject"),
                "sha256": sha256(output / name),
                "frames": len(indices),
                "max_gap_seconds": 1.5 / fps,
                "raw_frame_hashes": raw_hashes,
            }
        )
    _write_json(
        output / "manifest.json", _manifest(provenance, records, "cross_subject")
    )
    scene_records = [
        {
            **r,
            "split": split_subject(r["subject"], r["environment"], "cross_environment"),
        }
        for r in records
    ]
    _write_json(
        output / "cross-environment.json",
        _manifest(provenance, scene_records, "cross_environment"),
    )
    return output / "manifest.json"


def make_fixture(output, seed=17, frames=48):
    """A generated unit-test signal/pose relationship, NOT measured wireless data."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite fixture {output}")
    if frames < 12:
        raise ValueError("At least 12 fixture frames needed")
    output.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    skeleton = np.array(
        [
            [0.5, 0.20],
            [0.48, 0.18],
            [0.52, 0.18],
            [0.45, 0.2],
            [0.55, 0.2],
            [0.4, 0.32],
            [0.6, 0.32],
            [0.34, 0.44],
            [0.66, 0.44],
            [0.3, 0.55],
            [0.7, 0.55],
            [0.44, 0.56],
            [0.56, 0.56],
            [0.43, 0.74],
            [0.57, 0.74],
            [0.42, 0.91],
            [0.58, 0.91],
        ],
        dtype=np.float32,
    )
    projection = rng.normal(0, 0.3, (34, 3 * 24 * 4)).astype(np.float32)
    records = []
    for i in range(9):
        t = np.arange(frames) / 10
        phase = rng.uniform(0, 2 * np.pi)
        pose = np.broadcast_to(skeleton, (frames, 17, 2)).copy()
        pose[:, :, 0] += (0.03 * np.sin(t + phase))[:, None]
        pose[:, [7, 9], 1] -= (0.12 * (1 + np.sin(2 * t + phase)))[:, None]
        projected = np.einsum(
            "nf,fs->ns", pose.reshape(frames, 34), projection, optimize=False
        )
        csi = np.exp(
            1.5
            + projected.reshape(frames, 3, 24, 4) * 0.5
            + rng.normal(0, 0.02, (frames, 3, 24, 4))
        ).astype(np.float32)
        valid = np.ones((frames, 17), dtype=bool)
        # A genuine failure-path fixture: do not silently interpolate this gap.
        if i == 8:
            csi[frames // 2 : frames // 2 + 2] = np.nan
        name = f"fixture-{i:02d}.npz"
        np.savez_compressed(
            output / name, csi=csi, pose=pose, valid=valid, timestamps=t
        )
        split = "train" if i < 5 else ("validation" if i < 7 else "test")
        records.append(
            {
                "id": f"fixture/{i}",
                "path": name,
                "subject": f"F{i:02d}",
                "environment": f"E{i // 3}",
                "split": split,
                "sha256": sha256(output / name),
                "max_gap_seconds": 0.15,
            }
        )
    provenance = {
        "kind": "synthetic_fixture",
        "dataset": "Generated software fixture",
        "seed": seed,
        "note": "Not real Wi-Fi. No efficacy or domain-generalization claims permitted.",
    }
    _write_json(output / "manifest.json", _manifest(provenance, records, "fixture"))
    return output / "manifest.json"
