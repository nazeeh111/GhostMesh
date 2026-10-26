"""Versioned sequence manifest, split audit and causal window loader."""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from collections import Counter, OrderedDict
import numpy as np
import torch
from torch.utils.data import Dataset
from .preprocess import prepare_window, signal_quality

SPLITS = {"train", "validation", "test"}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path) -> dict:
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    provenance = manifest.get("provenance", {})
    if provenance.get("kind") not in {"synthetic_fixture", "licensed_dataset"}:
        raise ValueError("Declare synthetic_fixture or licensed_dataset provenance")
    if provenance["kind"] == "licensed_dataset":
        for key in ("dataset", "license", "source", "consent_basis"):
            if not provenance.get(key):
                raise ValueError(f"Licensed data requires provenance.{key}")
    if manifest.get("coordinate_system") != "normalized_image":
        raise ValueError(
            "2D normalized_image coordinate_system required; no implicit 3D projection"
        )
    records = manifest.get("sequences", [])
    if not records:
        raise ValueError("No sequences in manifest")
    ids, paths, hashes = set(), set(), {}
    subjects = {s: set() for s in SPLITS}
    environments = {s: set() for s in SPLITS}
    for row in records:
        for key in ("id", "path", "subject", "environment", "split", "sha256"):
            if not row.get(key):
                raise ValueError(f"Sequence requires {key}")
        split = row["split"]
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split}")
        p = (path.parent / row["path"]).resolve()
        if not p.is_relative_to(path.parent.resolve()):
            raise ValueError("Sequence path must stay inside manifest directory")
        if row["id"] in ids or p in paths:
            raise ValueError("Sequence duplicated across manifest/splits")
        ids.add(row["id"])
        paths.add(p)
        if not p.is_file() or sha256(p) != row["sha256"]:
            raise ValueError(f"Sequence missing or content hash mismatch: {row['id']}")
        h = row["sha256"]
        if h in hashes and hashes[h] != split:
            raise ValueError("Identical sequence bytes cross split boundary")
        hashes[h] = split
        subjects[split].add(row["subject"])
        environments[split].add(row["environment"])
    protocol = manifest.get("protocol", "custom")
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if (
            protocol in {"cross_subject", "cross_environment"}
            and subjects[a] & subjects[b]
        ):
            raise ValueError(f"Subject leakage between {a} and {b}")
    if protocol == "cross_environment":
        if environments["test"] & (environments["train"] | environments["validation"]):
            raise ValueError("Test environment appears in development data")
    if protocol not in {"cross_subject", "cross_environment", "custom", "fixture"}:
        raise ValueError("Unknown split protocol")
    return manifest


def read_sequence(path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"csi", "pose", "valid", "timestamps"}
        if not required <= set(archive.files):
            raise ValueError(f"Sequence missing {required - set(archive.files)}")
        result = {key: archive[key] for key in required}
    x, y, v, t = (result[key] for key in ("csi", "pose", "valid", "timestamps"))
    if x.ndim != 4 or y.ndim != 3 or y.shape[-1] != 2 or v.shape != y.shape[:-1]:
        raise ValueError("Expected csi[N,C,S,P], pose[N,J,2], valid[N,J]")
    if len(x) != len(y) or t.shape != (len(x),) or min(x.shape) < 1:
        raise ValueError("CSI, labels and timestamps must align exactly")
    if not np.isfinite(t).all() or (np.diff(t) <= 0).any():
        raise ValueError("Timestamps must be finite and strictly increasing")
    if v.dtype != np.bool_:
        raise ValueError("valid must be boolean, not confidence scores")
    if not np.isfinite(y[v]).all() or ((y[v] < 0) | (y[v] > 1)).any():
        raise ValueError("Valid labels must be finite 2D coordinates in [0,1]")
    result["pose"] = np.where(v[..., None], y, 0).astype(np.float32)
    return result


class WindowDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        split,
        window=8,
        normalize=True,
        stride=1,
        cache_windows=False,
    ):
        self.path = Path(manifest_path)
        self.manifest = load_manifest(self.path)
        self.entries = [r for r in self.manifest["sequences"] if r["split"] == split]
        self.window, self.normalize = int(window), normalize
        if self.window < 1 or stride < 1:
            raise ValueError("window and stride must be positive")
        self.cache = OrderedDict()
        self.windows = []
        self.candidate_windows = 0
        self.rejected_windows = 0
        self.rejection_reasons = Counter()
        self.sequence_coverage = {}
        self.warmup_frames = 0
        self.stride = int(stride)
        self.input_shape = None
        self.joints = None
        self.cache_windows = bool(cache_windows)
        self._mapped = {}
        self._cached_coverage = None
        self._storage = {"mode": "sequence_lru", "cache_windows": False}
        if self.cache_windows:
            self._initialize_window_cache(split)
        else:
            self._scan_windows()

    def _scan_windows(self):
        for index, record in enumerate(self.entries):
            seq = self._get(index)
            shape = seq["csi"].shape[1:]
            joints = seq["pose"].shape[1]
            if self.input_shape is not None and (
                shape != self.input_shape or joints != self.joints
            ):
                raise ValueError("Mixed sensor layouts or skeleton conventions")
            self.input_shape, self.joints = shape, joints
            max_gap = float(record.get("max_gap_seconds", 0.25))
            warmup = min(self.window - 1, len(seq["csi"]))
            self.warmup_frames += warmup
            sequence_counts = {
                "candidate_windows": 0,
                "accepted_windows": 0,
                "rejected_windows": 0,
                "warmup_frames": warmup,
                "rejection_reasons": Counter(),
            }
            self.sequence_coverage[record["id"]] = sequence_counts
            for end in range(self.window - 1, len(seq["csi"]), self.stride):
                self.candidate_windows += 1
                sequence_counts["candidate_windows"] += 1
                start = end - self.window + 1
                raw = seq["csi"][start : end + 1]
                ok, reasons = signal_quality(raw)
                if ok:
                    try:
                        prepare_window(raw, self.normalize)
                    except ValueError:
                        reasons.append("preprocessing_rejected")
                if (np.diff(seq["timestamps"][start : end + 1]) > max_gap).any():
                    reasons.append("timestamp_gap")
                if not seq["valid"][end].any():
                    reasons.append("no_valid_target_joints")
                if reasons:
                    self.rejected_windows += 1
                    sequence_counts["rejected_windows"] += 1
                    # One window can have several independent failure reasons.
                    self.rejection_reasons.update(set(reasons))
                    sequence_counts["rejection_reasons"].update(set(reasons))
                    continue
                self.windows.append((index, end))
                sequence_counts["accepted_windows"] += 1
        if not self.windows:
            raise ValueError(
                f"No usable windows (records={len(self.entries)}, rejected={self.rejected_windows})"
            )

    @staticmethod
    def _json_fingerprint(value):
        return hashlib.sha256(
            json.dumps(
                value, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()

    def _cache_request(self, split):
        root = Path(__file__).resolve().parent
        # The manifest hashes bind source array shapes as well as values. The
        # derived layout fingerprint and each NPY header are checked on restore.
        return {
            "schema_version": 1,
            "manifest_sha256": sha256(self.path),
            "split": split,
            "window": self.window,
            "normalize": bool(self.normalize),
            "stride": self.stride,
            "source_sha256": {
                name: sha256(root / name) for name in ("data.py", "preprocess.py")
            },
            "sequences_sha256": self._json_fingerprint(self.entries),
        }

    def _initialize_window_cache(self, split):
        request = self._cache_request(split)
        key = self._json_fingerprint(request)
        root = self.path.parent / ".ghostmesh-cache"
        if root.is_symlink():
            raise ValueError("Window cache root must not be a symlink")
        root.mkdir(exist_ok=True)
        destination = root / key
        if destination.exists() or destination.is_symlink():
            self._restore_window_cache(destination, request, key, reused=True)
            return
        # A stale lock is an explicit incomplete build, never an excuse to reuse
        # partial arrays or overwrite another builder's data.
        lock = root / (key + ".lock")
        try:
            lock.mkdir()
        except FileExistsError as exc:
            raise RuntimeError(
                f"Window cache build is active or incomplete: {lock}"
            ) from exc
        try:
            if destination.exists() or destination.is_symlink():
                self._restore_window_cache(destination, request, key, reused=True)
                return
            self._scan_windows()
            self._write_window_cache(root, destination, request, key)
            self.cache.clear()
            self._restore_window_cache(destination, request, key, reused=False)
        finally:
            lock.rmdir()  # Only the empty lock created by this invocation.

    def _array_specs(self, count):
        return {
            "csi": {
                "shape": [count, self.window, *self.input_shape],
                "dtype": "float32",
            },
            "pose": {"shape": [count, self.joints, 2], "dtype": "float32"},
            "valid": {"shape": [count, self.joints], "dtype": "bool"},
            "timestamps": {"shape": [count], "dtype": "float64"},
            "windows": {"shape": [count, 2], "dtype": "int64"},
        }

    def _write_window_cache(self, root, destination, request, key):
        temporary = Path(tempfile.mkdtemp(prefix=f".{key}.build-", dir=root))
        arrays = {}
        try:
            specs = self._array_specs(len(self.windows))
            for name, spec in specs.items():
                arrays[name] = np.lib.format.open_memmap(
                    temporary / f"{name}.npy",
                    mode="w+",
                    dtype=spec["dtype"],
                    shape=tuple(spec["shape"]),
                )
            # Sequential traversal bounds decoded-sequence storage and writes
            # one prepared window at a time, regardless of total cache size.
            for index, (record_index, end) in enumerate(self.windows):
                seq = self._get(record_index)
                raw = seq["csi"][end - self.window + 1 : end + 1]
                arrays["csi"][index] = prepare_window(raw, self.normalize)
                arrays["pose"][index] = seq["pose"][end]
                arrays["valid"][index] = seq["valid"][end]
                arrays["timestamps"][index] = seq["timestamps"][end]
                arrays["windows"][index] = (record_index, end)
            for array in arrays.values():
                array.flush()
                array._mmap.close()
            arrays.clear()
            files = {}
            for name, spec in specs.items():
                path = temporary / f"{name}.npy"
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
                files[name] = {**spec, "sha256": sha256(path)}
            layout = {
                "input_shape": list(self.input_shape),
                "joints": self.joints,
                "arrays": specs,
            }
            metadata = {
                "schema_version": 1,
                "complete": True,
                "key": key,
                "request": request,
                "layout": layout,
                "layout_sha256": self._json_fingerprint(layout),
                "coverage": self.coverage(),
                "files": files,
            }
            with (temporary / "metadata.json").open("x") as stream:
                stream.write(
                    json.dumps(metadata, sort_keys=True, indent=2, allow_nan=False)
                    + "\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
            with (temporary / "metadata.sha256").open("x") as stream:
                stream.write(sha256(temporary / "metadata.json") + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(
                    f"Refusing to replace existing window cache: {destination}"
                )
            temporary.rename(
                destination
            )  # Directory is visible only when all arrays are complete.
        finally:
            for array in arrays.values():
                array._mmap.close()
            if temporary.exists():
                # This invocation's generated, disposable intermediate only.
                shutil.rmtree(temporary)

    def _restore_window_cache(self, destination, request, key, *, reused):
        try:
            if destination.is_symlink() or not destination.is_dir():
                raise ValueError("cache directory is missing or a symlink")
            metadata_path = destination / "metadata.json"
            metadata_digest = destination / "metadata.sha256"
            if metadata_path.is_symlink() or metadata_digest.is_symlink():
                raise ValueError("cache metadata is a symlink")
            if sha256(metadata_path) != metadata_digest.read_text().strip():
                raise ValueError("cache metadata checksum mismatch")
            metadata = json.loads(metadata_path.read_text())
            if (
                not isinstance(metadata, dict)
                or metadata.get("schema_version") != 1
                or metadata.get("complete") is not True
                or metadata.get("key") != key
                or metadata.get("request") != request
            ):
                raise ValueError("cache request/completion metadata mismatch")
            layout = metadata["layout"]
            if not isinstance(layout, dict):
                raise ValueError("cache layout must be a mapping")
            if metadata.get("layout_sha256") != self._json_fingerprint(layout):
                raise ValueError("cache layout fingerprint mismatch")
            shape, joints = layout["input_shape"], layout["joints"]
            if (
                len(shape) != 3
                or any(type(n) is not int or n < 1 for n in shape)
                or type(joints) is not int
                or joints < 1
            ):
                raise ValueError("cache contains invalid sensor/skeleton layout")
            coverage = metadata["coverage"]
            if not isinstance(coverage, dict) or not isinstance(
                metadata["files"], dict
            ):
                raise ValueError("cache coverage/files must be mappings")
            count = coverage["accepted_windows"]
            if (
                type(count) is not int
                or count < 1
                or coverage["candidate_windows"] - coverage["rejected_windows"] != count
            ):
                raise ValueError("cache coverage accounting is inconsistent")
            self.input_shape, self.joints = tuple(shape), joints
            specs = self._array_specs(count)
            if layout["arrays"] != specs or set(metadata["files"]) != set(specs):
                raise ValueError("cache array layout does not match request")
            mapped = {}
            try:
                for name, spec in specs.items():
                    path = destination / f"{name}.npy"
                    detail = metadata["files"][name]
                    if (
                        not isinstance(detail, dict)
                        or detail.get("shape") != spec["shape"]
                        or detail.get("dtype") != spec["dtype"]
                    ):
                        raise ValueError(f"cache array metadata mismatch: {name}")
                    if (
                        path.is_symlink()
                        or not path.is_file()
                        or sha256(path) != detail.get("sha256")
                    ):
                        raise ValueError(f"cache checksum mismatch: {name}")
                    array = np.load(path, mmap_mode="r", allow_pickle=False)
                    if list(array.shape) != spec["shape"] or array.dtype != np.dtype(
                        spec["dtype"]
                    ):
                        array._mmap.close()
                        raise ValueError(f"cache NPY header mismatch: {name}")
                    mapped[name] = array
                indices = mapped["windows"]
                # Check bounded chunks; never materialize the feature cache.
                for start in range(0, count, 65536):
                    block = indices[start : start + 65536]
                    if (
                        (block[:, 0] < 0)
                        | (block[:, 0] >= len(self.entries))
                        | (block[:, 1] < self.window - 1)
                    ).any():
                        raise ValueError("cache contains invalid window indices")
            except Exception:
                for array in mapped.values():
                    array._mmap.close()
                raise
            self._mapped = mapped
            self.windows = mapped["windows"]
            self.candidate_windows = coverage["candidate_windows"]
            self.rejected_windows = coverage["rejected_windows"]
            self.rejection_reasons = Counter(coverage["rejection_reasons"])
            self.warmup_frames = coverage["warmup_frames"]
            self._cached_coverage = coverage
            self._storage = {
                "mode": "readonly_window_memmap",
                "cache_windows": True,
                "cache_key": key,
                "layout_sha256": metadata["layout_sha256"],
                "reused": reused,
                "checksums_verified": True,
                "feature_bytes": mapped["csi"].nbytes,
            }
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(
                f"Window cache is incomplete or corrupt; it was not overwritten: {destination}: {exc}"
            ) from exc

    def _get(self, index):
        if index not in self.cache:
            self.cache[index] = read_sequence(
                self.path.parent / self.entries[index]["path"]
            )
            if len(self.cache) > (1 if self.cache_windows else 64):
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def __len__(self):
        return len(self.windows)

    def coverage(self) -> dict:
        """Account for every strided, complete causal candidate before filtering.

        Warmup frames and stride are explicit so accepted fraction cannot be
        mistaken for coverage of every original frame. Reasons can overlap.
        One-joint targets remain usable for training, while pose metrics report
        their own additional exclusions for undefined bounding-box scales.
        """
        if self._cached_coverage is not None:
            return json.loads(json.dumps(self._cached_coverage))
        sequences = {}
        for name, counts in self.sequence_coverage.items():
            sequences[name] = {
                **counts,
                "rejection_reasons": dict(counts["rejection_reasons"]),
                "accepted_fraction": counts["accepted_windows"]
                / counts["candidate_windows"]
                if counts["candidate_windows"]
                else None,
            }
        return {
            "candidate_windows": self.candidate_windows,
            "accepted_windows": len(self),
            "rejected_windows": self.rejected_windows,
            "accepted_fraction": len(self) / self.candidate_windows
            if self.candidate_windows
            else None,
            "rejection_reasons": dict(self.rejection_reasons),
            "sequences": sequences,
            "warmup_frames": self.warmup_frames,
            "window_frames": self.window,
            "stride": self.stride,
            "denominator": "strided complete causal windows, before signal/timestamp/target filtering; warmup frames excluded",
            "reason_counting": "each reason counts a window once; different reasons can overlap",
        }

    def storage_info(self):
        return dict(self._storage)

    def __getitem__(self, index):
        record_index, end = map(int, self.windows[index])
        row = self.entries[record_index]
        if self._mapped:
            # The on-disk cache remains read-only. Returning one-window copies
            # also prevents callers from mutating it through Torch tensors.
            return {
                "csi": torch.from_numpy(self._mapped["csi"][index].copy()),
                "pose": torch.from_numpy(self._mapped["pose"][index].copy()),
                "valid": torch.from_numpy(self._mapped["valid"][index].copy()),
                "sequence": row["id"],
                "subject": row["subject"],
                "environment": row["environment"],
                "time": float(self._mapped["timestamps"][index]),
                "frame_index": end,
            }
        seq = self._get(record_index)
        raw = seq["csi"][end - self.window + 1 : end + 1]
        return {
            "csi": torch.from_numpy(prepare_window(raw, self.normalize)),
            "pose": torch.from_numpy(seq["pose"][end]),
            "valid": torch.from_numpy(seq["valid"][end]),
            "sequence": row["id"],
            "subject": row["subject"],
            "environment": row["environment"],
            "time": float(seq["timestamps"][end]),
            "frame_index": end,
        }
