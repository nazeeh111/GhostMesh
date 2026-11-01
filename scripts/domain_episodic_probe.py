"""SOURCE-only training of a reference-conditioned CSI adaptation procedure.

Fresh actual cache/config identities replace historical source-ready constants.
Only E01 labels/statistics enter this pilot. E03 metadata are counted but its
annotations are never decoded. This command cannot score held-room queries.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy import __version__ as SCIPY_VERSION
from scipy.linalg import solve
from scipy.linalg.blas import dgemm
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.episodic_calibration import EpisodicStudent, apply_residual, fit_residual  # noqa:E402
from ghostmesh.motion_filter import MotionFilter  # noqa:E402
from scripts import geometric_checkpoint as recovery  # noqa:E402
from scripts.build_moment_cache import digest, require, save  # noqa:E402
from scripts.verify_geometric_distillation import arrays, bind, read, scores  # noqa:E402

ARMS = ("ordinary", "episodic", "support_permuted")


def validate_config(config):
    require(
        config["schema_version"] == 1
        and config["arms"] == list(ARMS)
        and config["primary"] == "episodic",
        "Unknown episodic recipe",
    )
    require(
        config["source_room"] == "E01"
        and config["held_room"] == "E03"
        and config["source_validation_people"] == ["S04", "S09"],
        "New E01->E03 recipe required",
    )
    require(
        config["reference_actions"] == ["A03", "A07", "A12"]
        and config["references_per_record"] == 20
        and config["source_reference_people"] == ["S01", "S02"],
        "Reference contract differs",
    )
    for key in (
        "steps",
        "checkpoint_every",
        "reference_count_convention",
        "expected_source_train",
        "expected_source_validation",
        "expected_held_metadata",
    ):
        require(
            type(config[key]) is int and config[key] > 0,
            "Positive integer config required: " + key,
        )
    for key in ("seed", "permutation_seed"):
        require(
            type(config[key]) is int and config[key] >= 0, "Nonnegative seed required"
        )
    require(
        config["reference_count_convention"] == 24
        and config["steps"] % config["checkpoint_every"] == 0
        and config["steps"] < 10000,
        "Episode or recovery budget differs",
    )
    for key in (
        "learning_rate",
        "ridge",
        "outer_weight",
        "input_scale_floor",
        "pose_scale_floor",
        "delta_scale_floor",
    ):
        require(
            not isinstance(config[key], bool)
            and np.isfinite(config[key])
            and config[key] > 0,
            "Positive finite config required: " + key,
        )
    require(
        not isinstance(config["weight_decay"], bool)
        and np.isfinite(config["weight_decay"])
        and config["weight_decay"] >= 0,
        "Nonnegative finite decay required",
    )
    return config


def load_source(cache, raw, config):
    receipt = read(cache / "receipt.json")
    registration = read(cache / "registration.json")
    require(
        receipt["kind"] == "label-free-causal-moment-cache"
        and receipt["annotation_arrays_decoded"] is False
        and receipt["allocated_rows"] == receipt["logical_rows"],
        "Label-free exact-allocation cache required",
    )
    require(
        receipt["registration_sha256"] == digest(cache / "registration.json"),
        "Cache registration differs",
    )
    require(
        registration["annotation_arrays_decoded"] is False
        and registration["raw_receipt_sha256"] == receipt["raw_receipt_sha256"],
        "Producer annotation/provenance boundary differs",
    )
    bind(cache / "executed-source", registration["source_sha256"])
    bind(cache, receipt["arrays_sha256"])
    require(
        receipt["raw_receipt_sha256"] == digest(raw / "receipt.json"),
        "Raw/cache provenance differs",
    )
    meta = arrays(cache / "metadata.npz")
    require(
        set(meta) == {"room", "person", "record", "action", "end", "raw_index"}
        and all(len(v) == receipt["logical_rows"] for v in meta.values()),
        "Label-free metadata schema differs",
    )
    held = meta["room"] == config["held_room"]
    require(
        held.sum() == config["expected_held_metadata"], "Held metadata count differs"
    )
    own = meta["room"] == config["source_room"]
    ids = np.flatnonzero(own)
    data = {k: v[ids] for k, v in meta.items()}
    data["cache_row"] = ids
    data["validation"] = np.isin(data["person"], config["source_validation_people"])
    data["train"] = ~data["validation"]
    require(
        (int(data["train"].sum()), int(data["validation"].sum()))
        == (config["expected_source_train"], config["expected_source_validation"]),
        "Source cohort count differs",
    )
    require(
        set(data["person"][data["train"]])
        == {"S01", "S02", "S03", "S05", "S06", "S07", "S08", "S10"},
        "Source training people differ",
    )
    require(set(data["room"]) == {"E01"}, "Held room entered source data")
    raw_receipt = read(raw / "receipt.json")
    validate_raw_indices(data, raw_receipt["records"])
    require(
        raw_receipt["config"]["reference_canvas"] == [640, 640]
        and digest(raw / "pose.npy") == raw_receipt["arrays"]["pose"]
        and digest(raw / "valid.npy") == raw_receipt["arrays"]["valid"],
        "Raw annotation schema/hash differs",
    )
    # mmap reads only these E01 rows. Do not materialize the all-room array.
    full = np.asarray(
        np.load(raw / "pose.npy", mmap_mode="r", allow_pickle=False)[data["raw_index"]],
        dtype=float,
    )
    valid = np.load(raw / "valid.npy", mmap_mode="r", allow_pickle=False)[
        data["raw_index"]
    ]
    require(
        full.shape == (len(ids), 17, 2) and np.isfinite(full).all() and valid.all(),
        "Complete finite source teacher poses required",
    )
    data["pose"] = (full - full[:, [11, 12]].mean(1, keepdims=True)).astype(np.float32)
    feature = np.load(cache / "features.npy", mmap_mode="r", allow_pickle=False)
    require(
        feature.shape == (receipt["logical_rows"], 8, 12, 114)
        and feature.dtype == np.float32,
        "Source moment schema differs",
    )
    return data, feature, receipt, registration


def validate_raw_indices(data, records):
    offsets = {r["id"]: r["offset"] for r in records}
    require(
        data["end"].dtype.kind in "iu" and data["raw_index"].dtype.kind in "iu",
        "Integer raw annotation indices required",
    )
    for room, person, action, record, end, index in zip(
        *[data[k] for k in ("room", "person", "action", "record", "end", "raw_index")]
    ):
        require(
            room == "E01"
            and record == f"{room}/{person}/{action}"
            and record in offsets,
            "Source raw record identity differs",
        )
        require(
            39 <= end <= 295 and (end - 39) % 8 == 0 and index == offsets[record] + end,
            "Source raw annotation frame join differs",
        )


def checked_rows(rows, allowed):
    rows = np.asarray(rows)
    require(
        rows.ndim == 1
        and rows.dtype.kind in "iu"
        and len(rows) > 0
        and rows.min() >= 0
        and rows.max() < len(allowed)
        and allowed[rows].all(),
        "Rows outside explicit source role",
    )
    return rows


class SourceReader:
    def __init__(self, feature, cache_rows, allowed):
        self.feature = feature
        self.cache_rows = np.asarray(cache_rows).copy()
        self.allowed = np.asarray(allowed).copy()
        require(
            self.allowed.dtype == bool and self.allowed.shape == self.cache_rows.shape,
            "Boolean RF role required",
        )

    def __call__(self, rows):
        rows = checked_rows(rows, self.allowed)
        value = np.array(self.feature[self.cache_rows[rows]], copy=True)
        require(np.isfinite(value).all(), "Nonfinite source moments")
        return value


def training_labels(data, rows):
    return data["pose"][checked_rows(rows, data["train"])]


def pair_rows(record, ends):
    lookup = {(str(r), int(e)): i for i, (r, e) in enumerate(zip(record, ends))}
    require(len(lookup) == len(record), "Duplicate episode record endpoint")
    return np.array(
        [
            (i, lookup[(str(r), int(e) - 8)])
            for i, (r, e) in enumerate(zip(record, ends))
            if (str(r), int(e) - 8) in lookup
        ],
        dtype=np.int64,
    ).reshape(-1, 2)


def make_episodes(data, steps, seed, permutation_seed):
    train = data["train"]
    people = sorted(set(data["person"][train]))
    require(
        len(people) == 8 and set(data["room"][train]) == {"E01"},
        "E01-only eight-person TRAIN episodes required",
    )
    blocks = {}
    for person in people:
        for action in [f"A{i:02}" for i in range(1, 28)]:
            rows = np.flatnonzero(
                train & (data["person"] == person) & (data["action"] == action)
            )
            rows = rows[np.argsort(data["end"][rows])]
            for length in (3, 4):
                candidates = [
                    rows[start : start + length]
                    for start in range(len(rows) - length + 1)
                    if np.all(np.diff(data["end"][rows[start : start + length]]) == 8)
                ]
                require(bool(candidates), "Missing consecutive TRAIN block")
                blocks[person, action, length] = candidates
    rng = np.random.default_rng(seed)
    shuffle = np.random.default_rng(permutation_seed)
    episodes = []
    permutations = []
    for _ in range(steps):
        chosen = rng.choice(people, 4, replace=False)
        rows = []
        for person in chosen[:2]:
            for action in ("A03", "A07", "A12"):
                candidates = blocks[str(person), action, 4]
                rows.extend(candidates[int(rng.integers(len(candidates)))])
        for person in chosen[2:]:
            for action in rng.choice(
                [f"A{i:02}" for i in range(1, 28)], 4, replace=False
            ):
                candidates = blocks[str(person), str(action), 3]
                rows.extend(candidates[int(rng.integers(len(candidates)))])
        rows = np.array(rows, dtype=np.int64)
        checked_rows(rows, train)
        require(
            len(rows) == 48
            and not set(data["person"][rows[:24]]) & set(data["person"][rows[24:]]),
            "Support/query person leakage",
        )
        episodes.append(rows)
        permutations.append(
            np.concatenate(
                [
                    shuffle.permutation(np.arange(start, start + 4))
                    for start in range(0, 24, 4)
                ]
            )
        )
    return np.stack(episodes), np.stack(permutations)


def statistics(data, reader, config):
    ids = np.flatnonzero(data["train"])
    count = 0
    mean = np.zeros((12, 114))
    m2 = mean.copy()
    for start in range(0, len(ids), 128):
        x = reader(ids[start : start + 128]).astype(float).reshape(-1, 12, 114)
        n = len(x)
        mu = x.mean(0)
        difference = mu - mean
        total = count + n
        m2 += np.square(x - mu).sum(0) + difference**2 * (count * n / total)
        mean += difference * (n / total)
        count = total
    target = training_labels(data, ids).astype(float)
    pairs = pair_rows(data["record"][ids], data["end"][ids])
    delta = target[pairs[:, 0]] - target[pairs[:, 1]]
    return dict(
        input_mean=mean.astype(np.float32),
        input_scale=np.maximum(np.sqrt(m2 / count), config["input_scale_floor"]).astype(
            np.float32
        ),
        pose_mean=target.mean(0).astype(np.float32),
        pose_scale=np.maximum(target.std(0), config["pose_scale_floor"]).astype(
            np.float32
        ),
        pose_delta_scale=np.maximum(
            np.sqrt(np.square(delta).mean(0)), config["delta_scale_floor"]
        ).astype(np.float32),
    )


def make_model(stats, device):
    base = MotionFilter(
        *[
            torch.from_numpy(stats[key])
            for key in ("input_mean", "input_scale", "pose_mean", "pose_scale")
        ]
    )
    return EpisodicStudent(base).to(device)


def optimizer(model, config):
    return torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )


def pose_loss(pred, target, pairs, scale, delta_scale):
    require(
        pred.shape == target.shape and pred.shape[1:] == (17, 2) and len(pairs) > 0,
        "Pose objective schema differs",
    )
    return (
        0.5 * ((pred - target) / scale).square().mean()
        + 0.5
        * (
            (
                (pred[pairs[:, 0]] - pred[pairs[:, 1]])
                - (target[pairs[:, 0]] - target[pairs[:, 1]])
            )
            / delta_scale
        )
        .square()
        .mean()
    )


def train_step(
    model, opt, x, y, pairs, query_pairs, permutation, delta_scale, arm, config
):
    require(
        arm in ARMS and len(x) == 48 and len(y) == 48, "Known48example episode required"
    )
    result = model(x)
    base = result["pose"]
    outer = base[24:]
    if arm != "ordinary":
        support_y = y[:24] if arm == "episodic" else y[permutation]
        state = fit_residual(
            result["code"][:24],
            support_y.flatten(1),
            base[:24].flatten(1),
            ridge=config["ridge"],
            reference_count=config["reference_count_convention"],
        )
        outer = apply_residual(
            state, result["code"][24:], base[24:].flatten(1)
        ).reshape(-1, 17, 2)
        outer = outer - outer[:, [11, 12]].mean(1, keepdim=True)
    loss = pose_loss(base, y, pairs, model.body.pose_scale, delta_scale) + config[
        "outer_weight"
    ] * pose_loss(outer, y[24:], query_pairs, model.body.pose_scale, delta_scale)
    require(bool(torch.isfinite(loss)), "Nonfinite episodic loss before update")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    opt.step()
    return loss.detach()


def encode(model, reader, ids, device):
    model.eval()
    poses = []
    codes = []
    with torch.inference_mode():
        for start in range(0, len(ids), 16):
            result = model(torch.from_numpy(reader(ids[start : start + 16])).to(device))
            poses.append(result["pose"].cpu().numpy())
            codes.append(result["code"].cpu().numpy())
    return np.concatenate(poses), np.concatenate(codes)


def measure(pred, data, ids):
    def group(rows):
        return scores(
            pred[rows],
            data["pose"][ids[rows]],
            data["record"][ids[rows]],
            data["end"][ids[rows]],
        )

    result = dict(pooled=group(np.arange(len(ids))))
    for field in ("person", "action"):
        result["per_" + field] = {
            str(name): group(np.flatnonzero(data[field][ids] == name))
            for name in sorted(set(data[field][ids]))
        }
    perm = np.arange(len(ids))
    rng = np.random.default_rng(90210)
    for record in sorted(set(data["record"][ids])):
        rows = np.flatnonzero(data["record"][ids] == record)
        perm[rows] = rng.permutation(rows)
    result["shuffled"] = scores(
        pred[perm], data["pose"][ids], data["record"][ids], data["end"][ids]
    )
    return result


def references(data, config):
    selected = []
    for person in config["source_reference_people"]:
        for action in config["reference_actions"]:
            ids = np.flatnonzero(
                data["train"] & (data["person"] == person) & (data["action"] == action)
            )
            ids = ids[np.argsort(data["end"][ids])]
            require(len(ids) >= 20, "Insufficient reference endpoints")
            selected.extend(ids[np.linspace(0, len(ids) - 1, 20, dtype=np.int64)])
    selected = np.array(selected, dtype=np.int64)
    checked_rows(selected, data["train"])
    require(len(set(selected)) == 120, "Reference count/repeats differ")
    return selected


def product(a, b):
    return dgemm(
        1.0,
        np.asarray(a, dtype=float, order="F"),
        np.asarray(b, dtype=float, order="F"),
    )


def raw_kernel(train, query, kind):
    dimension = train.shape[1]
    dot = product(query, train.T) / dimension
    if kind == "linear":
        return dot, None
    own = product(train, train.T) / dimension
    square = np.square(train).mean(1)
    distance = np.maximum(square[:, None] + square[None, :] - 2 * own, 0)
    upper = distance[np.triu_indices(len(train), 1)]
    width = float(np.median(upper))
    require(np.isfinite(width) and width > 1e-12, "Degenerate reference RBF bandwidth")
    cross = np.maximum(np.square(query).mean(1)[:, None] + square[None, :] - 2 * dot, 0)
    return np.exp(-cross / (2 * width)), width


def raw_prediction(x, y, q, kind, alpha):
    km, width = raw_kernel(x, x, kind)
    kq, _ = raw_kernel(x, q, kind)
    # Both calls use the same reference-only bandwidth, never query statistics.
    center = y.mean(0)
    system = km + alpha * np.eye(len(x))
    weights = solve(
        system,
        y.reshape(len(y), -1) - center.reshape(1, -1),
        assume_a="pos",
        check_finite=True,
    )
    state = dict(alpha=alpha, reference_mean=center, weights=weights)
    if width is not None:
        state["width"] = width
    return (product(kq, weights) + center.reshape(1, -1)).reshape(-1, 17, 2), state


def raw_controls(ref_feature, ref_target, ref_person, query_feature):
    result = {}
    states = {}
    grids = {}
    for kind in ("linear", "rbf"):
        grid = []
        for alpha in (1e-5, 0.001, 0.1, 10.0):
            errors = []
            for person in sorted(set(ref_person)):
                tr = ref_person != person
                va = ~tr
                pred, _ = raw_prediction(
                    ref_feature[tr], ref_target[tr], ref_feature[va], kind, alpha
                )
                diagonal = np.linalg.norm(
                    ref_target[va].max(1) - ref_target[va].min(1), axis=1
                )
                require((diagonal > 1e-6).all(), "Degenerate reference body")
                errors.append(
                    float(
                        (
                            np.linalg.norm(pred - ref_target[va], axis=2)
                            / diagonal[:, None]
                        ).mean()
                    )
                )
            grid.append(
                dict(alpha=alpha, reference_person_cv_nje=float(np.mean(errors)))
            )
        choice = min(
            grid, key=lambda row: (row["reference_person_cv_nje"], -row["alpha"])
        )
        result[kind], states[kind] = raw_prediction(
            ref_feature, ref_target, query_feature, kind, choice["alpha"]
        )
        grids[kind] = grid
    return result, states, grids


def gate(primary, controls):
    po = primary["pooled"]
    return dict(
        body_motion_useful=po["motion_r2"] is not None and po["motion_r2"] > 0.05,
        delta_useful=po["delta_r2"] is not None and po["delta_r2"] > 0.05,
        both_people_positive=set(primary["per_person"]) == {"S04", "S09"}
        and all(
            r["motion_r2"] is not None and r["motion_r2"] > 0
            for r in primary["per_person"].values()
        ),
        nje_beats_controls=all(po["nje"] < r["pooled"]["nje"] for r in controls),
        pck_not_worse=all(
            po[key] >= r["pooled"][key] for r in controls for key in ("pck05", "pck20")
        ),
        shuffle_worse=po["nje"] < primary["shuffled"]["nje"],
    )


def run(cache, raw, config_path, out, device, resume=False):
    config = validate_config(read(config_path))
    require(resume or not out.exists(), "Existing study preserved")
    require(
        not any(out.resolve().is_relative_to(p.resolve()) for p in (cache, raw)),
        "Output inside immutable input",
    )
    require(
        device == "cpu" or torch.backends.mps.is_available(),
        "Requested native MPS unavailable",
    )
    data, feature, receipt, _ = load_source(cache, raw, config)
    paths = [
        Path(__file__),
        ROOT / "ghostmesh/episodic_calibration.py",
        ROOT / "ghostmesh/motion_filter.py",
        ROOT / "ghostmesh/geometric_distillation.py",
        ROOT / "scripts/geometric_checkpoint.py",
        ROOT / "scripts/build_moment_cache.py",
        ROOT / "scripts/verify_geometric_distillation.py",
        ROOT / "scripts/domain_geometric_distillation_probe.py",
        ROOT / "tests/test_episodic_calibration.py",
        ROOT / "tests/test_episodic_probe.py",
    ]
    registration = dict(
        kind="E01-only-episodic-calibration-source",
        config=config,
        config_sha256=digest(config_path),
        cache_receipt_sha256=digest(cache / "receipt.json"),
        raw_receipt_sha256=digest(raw / "receipt.json"),
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in paths},
        device=device,
        torch=str(torch.__version__),
        numpy=np.__version__,
        scipy=SCIPY_VERSION,
        held_annotation_values_decoded=False,
        scope="E01sourcefeasibility, represented-roomsourcecalproxy, noE03/E02/E04score orfullimage/3Dclaim",
    )
    out.mkdir(parents=True, exist_ok=resume)
    if resume:
        require(
            read(out / "registration.json") == registration,
            "Recovery registration differs",
        )
        require(
            not (out / "source-results.json").exists(),
            "Completed source study preserved",
        )
    else:
        save(out / "registration.json", registration)
        shutil.copyfile(config_path, out / "config.json")
        for path in paths:
            dest = out / "executed-source" / path.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
    lock = out / "writer.lock"
    if lock.exists():
        pid = read(lock)["pid"]
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            require(resume, "Stale writer needs explicit recovery")
            shutil.copyfile(lock, out / f"prior-writer-{pid}.json")
            lock.unlink()
        else:
            raise ValueError("Existing source writer")
    save(lock, dict(pid=os.getpid()))
    started = time.monotonic()
    torch.set_num_threads(1)
    try:
        tr_reader = SourceReader(feature, data["cache_row"], data["train"])
        val_reader = SourceReader(feature, data["cache_row"], data["validation"])
        episodes, permutations = make_episodes(
            data, config["steps"], config["seed"], config["permutation_seed"]
        )
        stats = statistics(data, tr_reader, config)
        if resume:
            np.testing.assert_array_equal(
                np.load(out / "episodes.npy", allow_pickle=False), episodes
            )
            np.testing.assert_array_equal(
                np.load(out / "support-permutations.npy", allow_pickle=False),
                permutations,
            )
            for key, value in stats.items():
                np.testing.assert_array_equal(
                    arrays(out / "TRAIN-statistics.npz")[key], value
                )
        else:
            np.save(out / "episodes.npy", episodes, allow_pickle=False)
            np.save(out / "support-permutations.npy", permutations, allow_pickle=False)
            np.savez(out / "TRAIN-statistics.npz", **stats)
            np.savez(out / "source-data.npz", **data)
        torch.manual_seed(config["seed"])
        initial = make_model(stats, "cpu")
        if resume:
            saved = torch.load(
                out / "initial-model.pt", map_location="cpu", weights_only=True
            )
            require(
                set(saved) == set(initial.state_dict())
                and all(
                    torch.equal(saved[k], v) for k, v in initial.state_dict().items()
                ),
                "Initial model differs",
            )
        else:
            torch.save(initial.state_dict(), out / "initial-model.pt")
        identity = ":".join(
            digest(out / name)
            for name in (
                "registration.json",
                "episodes.npy",
                "support-permutations.npy",
                "TRAIN-statistics.npz",
                "initial-model.pt",
            )
        )
        delta = torch.from_numpy(stats["pose_delta_scale"]).to(device)
        diagnostic = episodes[0]
        for arm in ARMS:
            folder = out / arm
            folder.mkdir(exist_ok=resume)
            net = make_model(stats, device)
            net.load_state_dict(initial.state_dict(), strict=True)
            opt = optimizer(net, config)
            step = (
                recovery.load(folder, net, opt, identity, arm, device) if resume else 0
            )
            require(step <= config["steps"], "Recovery budget exceeded")
            for index in range(step, config["steps"]):
                rows = episodes[index]
                y = training_labels(data, rows)
                x = tr_reader(rows)
                pairs = pair_rows(data["record"][rows], data["end"][rows])
                query_pairs = pair_rows(
                    data["record"][rows[24:]], data["end"][rows[24:]]
                )
                net.train()
                loss = train_step(
                    net,
                    opt,
                    torch.from_numpy(x).to(device),
                    torch.from_numpy(y).to(device),
                    torch.from_numpy(pairs).to(device),
                    torch.from_numpy(query_pairs).to(device),
                    torch.from_numpy(permutations[index]).to(device),
                    delta,
                    arm,
                    config,
                )
                if (index + 1) % config["checkpoint_every"] == 0:
                    recovery.save(folder, net, opt, index + 1, identity, arm, device)
                    pred, _ = encode(net, tr_reader, diagnostic, device)
                    progress = dict(
                        arm=arm,
                        step=index + 1,
                        loss=float(loss.cpu()),
                        TRAIN_diagnostic=scores(
                            pred,
                            data["pose"][diagnostic],
                            data["record"][diagnostic],
                            data["end"][diagnostic],
                        ),
                        seconds=time.monotonic() - started,
                    )
                    save(folder / f"progress-{index + 1:04}.json", progress)
                    print(json.dumps(progress), flush=True)
            complete = dict(
                step=config["steps"],
                identity=identity,
                checkpoint_sha256=digest(folder / f"step-{config['steps']:04}.pt"),
            )
            if (folder / "complete.json").exists():
                require(
                    read(folder / "complete.json") == complete,
                    "Completed arm identity differs",
                )
            else:
                save(folder / "complete.json", complete)
            del net, opt
        # All three fixed-final models now exist; no source validation was scored.
        attempt = 0
        while (out / f"source-evaluation-{attempt:02}").exists():
            attempt += 1
        evaluation = out / f"source-evaluation-{attempt:02}"
        evaluation.mkdir()
        refs = references(data, config)
        ids = np.flatnonzero(data["validation"])
        target = training_labels(data, refs)
        np.save(evaluation / "reference-rows.npy", refs, allow_pickle=False)
        zero = {}
        cal = {}
        for arm in ARMS:
            net = make_model(stats, device)
            opt = optimizer(net, config)
            require(
                recovery.load(out / arm, net, opt, identity, arm, device)
                == config["steps"],
                "Final state missing",
            )
            rp, rc = encode(net, tr_reader, refs, device)
            qp, qc = encode(net, val_reader, ids, device)
            state = fit_residual(
                torch.from_numpy(rc),
                torch.from_numpy(target).flatten(1),
                torch.from_numpy(rp).flatten(1),
                ridge=config["ridge"],
                reference_count=config["reference_count_convention"],
            )
            adapted = (
                apply_residual(
                    state, torch.from_numpy(qc), torch.from_numpy(qp).flatten(1)
                )
                .reshape(-1, 17, 2)
                .numpy()
            )
            adapted -= adapted[:, [11, 12]].mean(1, keepdims=True)
            np.savez(
                evaluation / f"{arm}-reference-map.npz",
                code_mean=state.code_mean.detach().numpy(),
                correction_mean=state.correction_mean.detach().numpy(),
                weights=state.weights.detach().numpy(),
            )
            np.save(evaluation / f"{arm}-VALzero.npy", qp, allow_pickle=False)
            np.save(evaluation / f"{arm}-VALcal.npy", adapted, allow_pickle=False)
            zero[arm] = measure(qp, data, ids)
            cal[arm] = measure(adapted, data, ids)
            del net, opt
        ref_feature = (
            (tr_reader(refs) - stats["input_mean"]) / stats["input_scale"]
        ).reshape(len(refs), -1)
        query_feature = (
            (val_reader(ids) - stats["input_mean"]) / stats["input_scale"]
        ).reshape(len(ids), -1)
        raw_pred, raw_states, raw_grids = raw_controls(
            ref_feature, target, data["person"][refs], query_feature
        )
        for kind, pred in raw_pred.items():
            np.save(evaluation / f"raw-{kind}-VAL.npy", pred, allow_pickle=False)
            np.savez(evaluation / f"raw-{kind}-state.npz", **raw_states[kind])
            cal["raw_" + kind] = measure(pred, data, ids)
        save(evaluation / "raw-reference-cv.json", raw_grids)
        source_target = training_labels(data, np.flatnonzero(data["train"]))
        prior = np.broadcast_to(source_target.mean(0), data["pose"][ids].shape)
        clock = {
            int(e): data["pose"][data["train"] & (data["end"] == e)].mean(0)
            for e in set(data["end"][data["train"]])
        }
        zero["source_mean"] = measure(prior, data, ids)
        zero["source_clock"] = measure(
            np.stack([clock[int(e)] for e in data["end"][ids]]), data, ids
        )
        cal["reference_mean"] = measure(
            np.broadcast_to(target.mean(0), data["pose"][ids].shape), data, ids
        )
        gates = dict(
            zero=gate(
                zero["episodic"], [r for name, r in zero.items() if name != "episodic"]
            ),
            calibrated=gate(
                cal["episodic"], [r for name, r in cal.items() if name != "episodic"]
            ),
        )
        bind(ROOT, registration["source_sha256"])
        bind(cache, receipt["arrays_sha256"])
        require(
            digest(config_path) == registration["config_sha256"],
            "Config changed during fit",
        )
        report = dict(
            registration=registration,
            zero=zero,
            calibrated=cal,
            gates=gates,
            passes={name: all(values.values()) for name, values in gates.items()},
            evaluation_directory=evaluation.name,
            source_reference_rows=120,
            source_query_rows=len(ids),
            invocation_seconds=time.monotonic() - started,
            resumed=resume,
            limits="Only E01optimizer/statistics/weights and validation. Calibration is represented-room source-person proxy, not unseenroom result. No E03annotations decoded orheldroomqueries scored. Rawreference controls use fewer global labels than learnedarms; ordinary/shuffled controls are matched. Hiprelative2Donly.",
        )
        save(out / "source-results.json", report)
        manifest = {
            str(p.relative_to(out)): digest(p)
            for p in out.rglob("*")
            if p.is_file() and p != lock
        }
        save(
            out / "source-ready.json",
            dict(
                kind=registration["kind"],
                held_metrics_computed=False,
                source_validation_computed=True,
                artifacts_sha256=manifest,
            ),
        )
        print(
            json.dumps(
                dict(
                    source_passes=report["passes"], seconds=report["invocation_seconds"]
                )
            ),
            flush=True,
        )
    finally:
        if lock.exists() and read(lock)["pid"] == os.getpid():
            lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "raw", "config", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.cache, args.raw, args.config, args.out, args.device, args.resume)
