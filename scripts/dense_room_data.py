"""Guarded dense two-source-room data, without target-room annotation access."""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import domain_pose_alignment_probe as base  # noqa:E402
from scripts.verify_geometric_distillation import arrays, bind, digest, read, require  # noqa:E402

VAL_PEOPLE = ("S04", "S09", "S14", "S19")


class HistoryView:
    def __init__(self, moments, centers):
        self.moments = moments
        self.centers = np.asarray(centers)
        require(
            moments.ndim == 3
            and moments.shape[1:] == (12, 114)
            and self.centers.ndim == 1
            and self.centers.dtype.kind in "iu"
            and len(self.centers) > 0
            and self.centers.min() >= 27
            and self.centers.max() + 4 < len(moments),
            "Typed record-bounded RF history centers required",
        )
        self.shape = (len(centers), 32, 12, 114)
        self.dtype = moments.dtype

    def __getitem__(self, rows):
        rows = np.asarray(rows)
        require(
            rows.ndim == 1
            and rows.dtype.kind in "iu"
            and len(rows) > 0
            and rows.min() >= 0
            and rows.max() < len(self.centers),
            "Typed history rows required",
        )
        frames = self.centers[rows, None] + 4 - np.arange(31, -1, -1)[None]
        require(
            frames.min() >= 0 and frames.max() < len(self.moments),
            "RF history bounds differ",
        )
        return self.moments[frames]


def load_source(cache, raw):
    receipt = read(cache / "receipt.json")
    reg = read(cache / "registration.json")
    require(
        receipt["kind"] == reg["kind"] == "E01E02-label-free-perframe-moments"
        and receipt["registration_sha256"] == digest(cache / "registration.json")
        and receipt["frame_rows"] == 160380
        and receipt["annotation_arrays_decoded"] is False
        and reg["annotation_arrays_decoded"] is False,
        "Label-free exactsource20 cache required",
    )
    bind(cache, receipt["arrays_sha256"])
    bind(cache / "executed-source", reg["source_sha256"])
    rr = read(raw / "receipt.json")
    require(
        reg["raw_receipt_sha256"] == digest(raw / "receipt.json")
        and rr["config"]["magnitude_units"] == "log_magnitude_db"
        and rr["config"]["phase_units"] == "radians"
        and rr["config"]["reference_canvas"] == [640, 640],
        "Typedrawprovenance differs",
    )
    meta = arrays(cache / "metadata.npz")
    require(
        set(meta)
        == {"room", "person", "action", "record", "end", "raw_index", "cache_center"}
        and set(meta["room"]) == {"E01", "E02"}
        and all(len(v) == receipt["logical_dense_endpoints"] for v in meta.values()),
        "Held room/malformed metadata entered source",
    )
    held = np.isin(meta["person"], VAL_PEOPLE)
    keep = ~held | ((meta["end"] - 39) % 8 == 0)
    data = {k: v[keep] for k, v in meta.items()}
    data["validation"] = held[keep]
    data["train"] = ~data["validation"]
    require(
        all(
            data[key].dtype.kind in "iu" for key in ("end", "raw_index", "cache_center")
        ),
        "Integer source/frame indices required",
    )
    require(
        (int(data["train"].sum()), int(data["validation"].sum())) == (108956, 3435),
        "DenseTRAIN/stridedVAL cohort differs",
    )
    require(
        set(data["person"]) == {f"S{i:02}" for i in range(1, 21)}
        and not set(data["person"][data["train"]])
        & set(data["person"][data["validation"]]),
        "Source-person isolation differs",
    )
    rows = {r["id"]: r for r in rr["records"]}
    cached = {r["record"]: r for r in read(cache / "records.json")}
    quality = {r["record"]: r for r in read(cache / "quality.json")}
    for room, person, action, name, end, index, center in zip(
        *[
            data[k]
            for k in [
                "room",
                "person",
                "action",
                "record",
                "end",
                "raw_index",
                "cache_center",
            ]
        ]
    ):
        require(
            room in ["E01", "E02"]
            and name == f"{room}/{person}/{action}"
            and name in rows
            and name in cached
            and 39 <= end <= 292
            and index == rows[name]["offset"] + end
            and center == cached[name]["cache_start"] + end
            and 0 <= index < 160380,
            "Source raw/frame join differs",
        )
        require(
            cached[name]["cache_start"]
            <= center + 4 - 31
            <= center + 4
            < cached[name]["cache_start"] + 297,
            "RF history crossed record",
        )
        require(
            int(end) in quality[name]["dense_endpoints"],
            "Endpoint missing RFquality guard",
        )
    require(
        reg["context_frames"] == 32 and reg["output_delay_frames"] == 4,
        "Clock recipe differs",
    )
    for name in ["pose", "valid"]:
        require(
            digest(raw / f"{name}.npy") == rr["arrays"][name], "Raw annotations differ"
        )
    # E03 starts at160380. All roles/joins are checked before any annotation decode.
    y = np.asarray(
        np.load(raw / "pose.npy", mmap_mode="r", allow_pickle=False)[data["raw_index"]],
        dtype=float,
    )
    valid = np.load(raw / "valid.npy", mmap_mode="r", allow_pickle=False)[
        data["raw_index"]
    ]
    require(
        valid.all()
        and y.shape == (len(keep.nonzero()[0]), 17, 2)
        and np.isfinite(y).all(),
        "Complete source teachers required",
    )
    data["pose"] = (y - y[:, [11, 12]].mean(1, keepdims=True)).astype(np.float32)
    moments = np.load(cache / "moments.npy", mmap_mode="r", allow_pickle=False)
    require(
        moments.shape == (160380, 12, 114) and moments.dtype == np.float32,
        "Per-frame RF schema differs",
    )
    feature = HistoryView(moments, data["cache_center"])
    return data, feature, receipt


def statistics(data, feature, cfg):
    ids = np.flatnonzero(data["train"])
    centers = data["cache_center"][ids]
    frames = centers[:, None] + 4 - np.arange(31, -1, -1)[None]
    counts = np.bincount(frames.ravel(), minlength=len(feature.moments))
    require(counts.sum() == len(ids) * 32, "TRAINhistory weights differ")
    active = np.flatnonzero(counts)
    total = float(counts.sum())
    mean = np.zeros((12, 114))
    for first in range(0, len(active), 1024):
        rows = active[first : first + 1024]
        x = np.asarray(feature.moments[rows], dtype=float)
        mean += (x * counts[rows, None, None]).sum(0) / total
    variance = np.zeros_like(mean)
    for first in range(0, len(active), 1024):
        rows = active[first : first + 1024]
        x = np.asarray(feature.moments[rows], dtype=float)
        variance += (np.square(x - mean) * counts[rows, None, None]).sum(0) / total
    target = base.p.training_labels(data, ids).astype(float)
    pairs = base.p.pair_rows(data["record"][ids], data["end"][ids])
    delta = target[pairs[:, 0]] - target[pairs[:, 1]]
    scale = np.maximum(target.std(0), cfg["pose_scale_floor"])
    proto, prior, assignment = base.prototypes(target, scale, cfg["seed"])
    return dict(
        input_mean=mean.astype(np.float32),
        input_scale=np.maximum(np.sqrt(variance), 0.05).astype(np.float32),
        pose_mean=target.mean(0).astype(np.float32),
        pose_scale=scale.astype(np.float32),
        pose_delta_scale=np.maximum(
            np.sqrt(np.square(delta).mean(0)), cfg["delta_scale_floor"]
        ).astype(np.float32),
        prototypes=proto,
        prior=prior,
        TRAIN_prototype_assignment=assignment,
    )


def make_schedule(data, cfg):
    ids = np.flatnonzero(data["train"])
    pairs = ids[base.p.pair_rows(data["record"][ids], data["end"][ids])]
    people = sorted(set(data["person"][ids]))
    require(len(people) == 16, "16TRAINpeople required")
    choices = {
        person: pairs[data["person"][pairs[:, 0]] == person] for person in people
    }
    rng = np.random.default_rng(cfg["seed"])
    schedule = []
    for _ in range(cfg["steps"]):
        schedule.append(
            np.concatenate(
                [
                    choices[person][int(rng.integers(len(choices[person])))]
                    for person in people
                ]
            )
        )
    return np.asarray(schedule, dtype=np.int64)


def references(data, room):
    people = {"E01": ["S01", "S02"], "E02": ["S11", "S12"]}[room]
    selected = []
    for person in people:
        for action in ["A03", "A07", "A12"]:
            rows = np.flatnonzero(
                data["train"]
                & (data["room"] == room)
                & (data["person"] == person)
                & (data["action"] == action)
            )
            rows = rows[np.argsort(data["end"][rows])]
            require(len(rows) >= 20, "Insufficient reference frames")
            selected.extend(rows[np.linspace(0, len(rows) - 1, 20, dtype=np.int64)])
    result = np.asarray(selected, dtype=np.int64)
    require(
        len(set(result)) == 120 and data["train"][result].all(),
        "Shared room reference roles/count differ",
    )
    return result
