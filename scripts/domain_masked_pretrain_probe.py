"""Source-only masked raw-CSI pretraining pilot and frozen pose readouts.

Use --benchmark for synthetic engineering checks. Real fits require an explicit
output directory and preserve code, initialization, schedule, checkpoints and
source-only selections before held-room scores. E04 is never accepted here.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostmesh.masked_csi import (  # noqa:E402
    MaskedCSI,
    fixed_count_mask,
    normalize_frames,
    pretext_loss,
)  # noqa:E402
from scripts.domain_background_probe import digest  # noqa:E402
from scripts.domain_covariance_cached_probe import load_inputs  # noqa:E402
from scripts.domain_covariance_source_probe import fold_masks, rff_transform, save  # noqa:E402
from scripts.domain_covariance_full_cache import verify_hashes  # noqa:E402
from scripts.domain_reference_covariance_probe import fit_readout, predict, scores  # noqa:E402
from ghostmesh.reference_covariance import current_covariances  # noqa:E402
from ghostmesh.covariance_dynamics import hermitian_coordinates  # noqa:E402

ARMS = (
    "random_frozen",
    "masked_only",
    "masked_temporal",
    "masked_shuffled",
    "supervised",
)
CONFIG = dict(
    seed=2027,
    held_room="E02",
    steps=4000,
    batch=16,
    learning_rate=0.001,
    weight_decay=0.01,
    gradient_clip=1.0,
    mask_seed=719,
    mask="Exactly71of95tokens perframe (74.7368percent), full95tokens including masktokens enterencoder; no visible-token deletion",
    input="Log-magnitude dB ONLY, pointwise global frame mean/SD;3receiversx114subcarriersx10packetcolumns, no calibrated packet timing/phase claim",
    patch="3receiversx6subcarriersx2packetcolumns;19x5orderedtokens,2attentionlayers48D/4heads/FF128; no upstream code copied",
    context="8causalframes ending canonicalendpoint, retained40frame quality guard from frozen covariance cache",
    pretext="MeanmaskedpatchMSE+.05*(25adjacentMSE+25variancehinge+covariancedecorrelation); maskedonly omits all temporal terms",
    shuffled="Within same8frame history, roll seven adjacent partner embeddings by4; pairtime distances4or-3, no true adjacent or identical pairing; reconstruction inputs and mask unchanged",
    supervision="Same8rawframes/steps asSSL; standardizedendpoint34coordinateMSE, endpointlabelsTRAINonly; no early stopping. Allarms subsequently same sourceTRAINendpoint labels in frozenRFFprobe",
    selection="Fixed finalstep encoder; independently select source-onlyRFF512 ridgealpha .001/.01/.1/1/10 by equal-source-room validation NJE; ALLchoices saved before heldscores",
    scope="One E02developmentfold/seed, sourceE01/E03TRAINonly optimizer. HIP-ROOT-RELATIVE articulation only, not fullimagepose/hiplocation; no E04,target enrollment,RGB,3D or final efficacy claim",
    limits="Full-frame normalization precedes patch masking: observed values in masked patches contribute two frame mean/SD statistics, so this is not strict masked-raw-target independence. No pose, future, or cross-frame statistics enter. Masked training versus full-unmasked embedding inference distribution differs. Pointwise normalization discards overall frame gain; all encoder frames process independently before8embedding concatenation. Prior source development exposure remains disclosed.",
)


class HistoryReader:
    """RF-only reader with an explicit allowed-example boundary, no label API."""

    def __init__(self, rf, metadata, records, allowed):
        self.rf = rf
        self.ends = np.array(metadata["end"], dtype=np.int64)
        self.allowed = np.asarray(allowed, dtype=bool).copy()
        offsets = {r["id"]: r["offset"] for r in records}
        self.offsets = np.asarray([offsets[str(r)] for r in metadata["record"]])
        if self.allowed.shape != self.ends.shape or (self.ends < 7).any():
            raise ValueError("Invalid role or causal endpoint")

    def __call__(self, ids, *, return_observed=False):
        ids = np.asarray(ids)
        if (
            ids.ndim != 1
            or ids.dtype.kind not in "iu"
            or (ids < 0).any()
            or (ids >= len(self.ends)).any()
            or not self.allowed[ids].all()
        ):
            raise ValueError("RF access outside explicitly allowed role")
        frames = (
            self.offsets[ids, None] + self.ends[ids, None] - np.arange(7, -1, -1)[None]
        )
        return normalize_frames(
            np.asarray(self.rf[frames, :3]), return_observed=return_observed
        )


def make_schedule(room, train, steps=1000, batch=16):
    rng = np.random.default_rng(CONFIG["seed"])
    if len(room) != len(train) or train.dtype != bool:
        raise ValueError("Boolean training mask required")
    groups = [np.flatnonzero(train & (room == name)) for name in ("E01", "E03")]
    if any(len(ids) < batch for ids in groups) or np.any(
        train & ~np.isin(room, ["E01", "E03"])
    ):
        raise ValueError("Only E01/E03 TRAIN examples can be scheduled")
    return np.stack(
        [
            rng.choice(groups[int(rng.integers(2))], batch, replace=False)
            for _ in range(steps)
        ]
    )


def optimizer_step(model, optimizer, histories, mask, arm, target=None, observed=None):
    optimizer.zero_grad(set_to_none=True)
    if arm == "supervised":
        if target is None or target.shape != (len(histories), 34):
            raise ValueError("Supervised endpoint target required")
        loss = torch.nn.functional.mse_loss(
            model.pose_head(model.history(histories)), target
        )
        terms = {"supervised": loss}
    elif arm in ("masked_only", "masked_temporal", "masked_shuffled"):
        if target is not None:
            raise ValueError("Pretext optimizer does not accept pose labels")
        loss, terms = pretext_loss(
            model,
            histories,
            mask,
            temporal=arm != "masked_only",
            shuffled=arm == "masked_shuffled",
            observed=observed,
        )
    else:
        raise ValueError("Unknown trainable arm")
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite objective")
    loss.backward()
    # The global L2 norm is already required for clipping. This check rejects
    # NaN/Inf gradients and finite-gradient overflow BEFORE any parameter update,
    # without one GPU synchronization for every parameter tensor.
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), CONFIG["gradient_clip"], error_if_nonfinite=True
    )
    optimizer.step()
    return float(loss.detach().cpu()), {
        key: float(value.detach().cpu()) for key, value in terms.items()
    }


def benchmark(device="cpu", batch=16):
    torch.set_num_threads(1)
    torch.manual_seed(CONFIG["seed"])
    model = MaskedCSI().to(device)
    x = torch.randn(batch, 8, 3, 114, 10, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    generator = torch.Generator().manual_seed(CONFIG["mask_seed"])
    durations = []
    for _ in range(6):
        mask = fixed_count_mask(batch * 8, generator).to(device)
        started = time.monotonic()
        loss, terms = optimizer_step(model, optimizer, x, mask, "masked_temporal")
        if device == "mps":
            torch.mps.synchronize()
        durations.append(time.monotonic() - started)
    excluded = ("decoder.", "projector.", "pose_head.")
    return dict(
        device=device,
        batch=batch,
        histories_frames=8,
        parameters=sum(p.numel() for p in model.parameters()),
        encoder_parameters=sum(
            p.numel() for n, p in model.named_parameters() if not n.startswith(excluded)
        ),
        seconds=durations,
        median_last5_seconds=float(np.median(durations[1:])),
        loss=loss,
        terms=terms,
        scope="SYNTHETIC engineering only, no dataset trained or scored",
    )


@torch.no_grad()
def extract(model, reader, n, device, out):
    model.eval()
    embedding = np.lib.format.open_memmap(
        out, mode="w+", dtype=np.float32, shape=(n, 512)
    )
    for start in range(0, n, 32):
        ids = np.arange(start, min(n, start + 32))
        x = torch.from_numpy(reader(ids)).to(device)
        embedding[ids] = model.history(x).cpu().numpy()
    embedding.flush()
    return embedding


def query_permutation(records):
    rng = np.random.default_rng(90210)
    perm = np.arange(len(records))
    for record in sorted(set(records)):
        ids = np.flatnonzero(records == record)
        perm[ids] = rng.permutation(ids)
    return perm


def cpu_state(value):
    """Portable snapshots without retaining mutable device tensor references."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(item) for item in value)
    return value


def save_checkpoint(
    out, arm, step, model, optimizer, generator, identity, logs, seconds, device
):
    path = out / f"{arm}-step-{step:05}.pt"
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(
            "Checkpoint already exists; refusing silent overwrite: " + str(path)
        )
    state = dict(
        arm=arm,
        step=step,
        identity=identity,
        model=cpu_state(model.state_dict()),
        optimizer=cpu_state(optimizer.state_dict()),
        mask_generator=generator.get_state(),
        cpu_rng=torch.get_rng_state(),
        mps_rng=torch.mps.get_rng_state().cpu() if device == "mps" else None,
        device=device,
        logs=logs,
        seconds=seconds,
    )
    with path.open("xb") as handle:
        torch.save(state, handle)
    save(
        path.with_suffix(".json"),
        dict(checkpoint_sha256=digest(path), identity=identity, arm=arm, step=step),
    )
    return path


def restore_checkpoint(out, arm, model, optimizer, generator, identity, device):
    candidates = sorted(out.glob(arm + "-step-*.json"))
    # A partially written unsealed checkpoint cannot be silently accepted.
    if any(
        not path.with_suffix(".json").exists() for path in out.glob(arm + "-step-*.pt")
    ):
        raise ValueError(
            "Unsealed partial checkpoint present; preserve and inspect before resuming"
        )
    if not candidates:
        return 0, [], 0.0
    receipt = json.loads(candidates[-1].read_text())
    path = candidates[-1].with_suffix(".pt")
    if receipt["identity"] != identity or receipt["checkpoint_sha256"] != digest(path):
        raise ValueError("Checkpoint identity or bytes changed")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (
        saved["identity"] != identity
        or saved["arm"] != arm
        or saved["device"] != device
        or saved["step"] != receipt["step"]
    ):
        raise ValueError("Checkpoint arm, device, step or identity differs")
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    generator.set_state(saved["mask_generator"])
    torch.set_rng_state(saved["cpu_rng"])
    if device == "mps":
        torch.mps.set_rng_state(saved["mps_rng"])
    return saved["step"], saved["logs"], saved["seconds"]


def seal_source_ready(out):
    """All source-only fitting is complete; no held metrics are computed here."""
    files = {
        str(path.relative_to(out)): digest(path)
        for path in sorted(out.rglob("*"))
        if path.is_file() and path.name != "source-ready.json"
    }
    save(
        out / "source-ready.json",
        dict(
            kind="masked-csi-source-only-ready",
            held_metrics_computed=False,
            artifacts_sha256=files,
        ),
    )


def run(raw, cache, old_probe, out, device, *, resume=False):
    if out.exists() and not resume:
        raise FileExistsError(out)
    if resume and not out.is_dir():
        raise ValueError("Resume requires an existing experiment directory")
    started = time.monotonic()
    receipt, data, covariance = load_inputs(cache)
    raw_receipt = json.loads((raw / "receipt.json").read_text())
    if raw_receipt["config"]["magnitude_units"] != "log_magnitude_db":
        raise ValueError("Expected explicitly typed log-magnitude dB inputs")
    if (
        receipt["raw_receipt_sha256"] != digest(raw / "receipt.json")
        or raw_receipt["arrays"] != receipt["raw_arrays_sha256"]
    ):
        raise ValueError("Raw/cache lineage differs")
    verify_hashes(
        raw, {name + ".npy": sha for name, sha in raw_receipt["arrays"].items()}
    )
    rf = np.load(raw / "rf.npy", mmap_mode="r", allow_pickle=False)
    if rf.shape != (240570, 6, 114, 10):
        raise ValueError("Only complete source30 RF accepted")
    train, validation, held = fold_masks(data["room"], data["person"], "E02")
    schedule = make_schedule(data["room"], train, CONFIG["steps"], CONFIG["batch"])
    train_reader = HistoryReader(rf, data, raw_receipt["records"], train)
    all_reader = HistoryReader(
        rf, data, raw_receipt["records"], np.ones(len(train), bool)
    )
    # No new per-arm quality drops are permitted. Fail before optimizing if the
    # common canonical mask includes a frame unsuitable for this preprocessing.
    quality = {
        name: dict(
            frame_occurrences=0,
            missing_cells=0,
            observed_cells=0,
            frames_with_missing=0,
        )
        for name in ["train", "validation", "held"]
    }
    for name, role in [("train", train), ("validation", validation), ("held", held)]:
        indices = np.flatnonzero(role)
        for start in range(0, len(indices), 128):
            _, observed = all_reader(indices[start : start + 128], return_observed=True)
            quality[name]["frame_occurrences"] += int(observed.shape[0] * 8)
            quality[name]["missing_cells"] += int((~observed).sum())
            quality[name]["observed_cells"] += int(observed.sum())
            quality[name]["frames_with_missing"] += int(
                (~observed).any(axis=(-3, -2, -1)).sum()
            )
    torch.set_num_threads(1)
    torch.manual_seed(CONFIG["seed"])
    initial = MaskedCSI().state_dict()
    old_model_path = old_probe / "E02-congruence_dynamics-model-prediction.npz"
    old_manifest = json.loads((old_probe / "artifact_manifest.json").read_text())
    if old_manifest["targets-and-metadata.npz"] != digest(
        cache / "targets-and-metadata.npz"
    ):
        raise ValueError("Frozen temporal metadata differs from canonical cache")
    with np.load(old_model_path, allow_pickle=False) as old:
        if (
            not np.array_equal(old["held_indices"], np.flatnonzero(held))
            or old["pose"].shape != data["pose"][held].shape
        ):
            raise ValueError("Frozen temporal held query indices differ")
    verify_hashes(old_probe, {old_model_path.name: old_manifest[old_model_path.name]})
    sources = [
        Path(__file__),
        ROOT / "ghostmesh/masked_csi.py",
        ROOT / "tests/test_masked_csi.py",
        ROOT / "tests/test_masked_pretrain_probe.py",
        ROOT / "scripts/domain_covariance_cached_probe.py",
        ROOT / "scripts/domain_covariance_source_probe.py",
        ROOT / "scripts/domain_covariance_full_cache.py",
        ROOT / "scripts/domain_reference_covariance_probe.py",
        ROOT / "scripts/domain_background_probe.py",
        ROOT / "ghostmesh/reference_covariance.py",
        ROOT / "ghostmesh/covariance_dynamics.py",
        ROOT / "ghostmesh/covariance_accelerated.py",
        ROOT / "ghostmesh/causal_channel.py",
        ROOT / "ghostmesh/metrics.py",
    ]
    registration = dict(
        config=CONFIG,
        counts={
            name: int(mask.sum())
            for name, mask in [
                ("train", train),
                ("validation", validation),
                ("held", held),
            ]
        },
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
        cache_receipt_sha256=digest(cache / "receipt.json"),
        raw_receipt_sha256=digest(raw / "receipt.json"),
        device=device,
        torch_version=torch.__version__,
        frozen_temporal_model_sha256=digest(old_model_path),
        missingness_per_role=quality,
        missing_policy="Finite-only mean/SD eachFRAME, missinginputcells0(frameobservedmean); no mask inputchannel; missingtargetcells excluded frommaskedMSE; allarms same RFinput. Allmissing/constantframes rejected. No temporal/person aggregation",
        exposure=dict(
            optimizer_histories_per_trained_arm=int(schedule.size),
            optimizer_frame_occurrences_per_trained_arm=int(schedule.size * 8),
            optimizer_unique_endpoint_rows=int(len(np.unique(schedule))),
            supervised_optimizer_endpoint_label_occurrences=int(schedule.size),
            ssl_optimizer_pose_labels=0,
            frozen_probe_training_endpoint_labels=int(train.sum()),
            feature_preflight="Allrole RF finite/constant checks only; no fit or statistic retained across frames",
        ),
    )
    if resume:
        previous = json.loads((out / "registration.json").read_text())
        if previous != registration:
            raise ValueError("Resume registration/config/input/source/device differs")
        seal = json.loads((out / "preoptimization-seal.json").read_text())
        verify_hashes(out, seal["files_sha256"])
        np.testing.assert_array_equal(
            np.load(out / "schedule.npy", allow_pickle=False), schedule
        )
        with np.load(out / "roles.npz", allow_pickle=False) as roles:
            for name, value in [
                ("train", train),
                ("validation", validation),
                ("held", held),
            ]:
                np.testing.assert_array_equal(roles[name], value)
        initial = torch.load(out / "initial.pt", map_location="cpu", weights_only=True)
        if (out / "results.json").exists():
            raise ValueError(
                "Held results already exist; no prospective training resume allowed"
            )
        if (out / "source-ready.json").exists():
            verify_hashes(
                out,
                json.loads((out / "source-ready.json").read_text())["artifacts_sha256"],
            )
            return out
    else:
        out.mkdir(parents=True)
        save(out / "registration.json", registration)
        for path in sources:
            target = out / "executed-source" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        np.save(out / "schedule.npy", schedule, allow_pickle=False)
        np.savez(out / "roles.npz", train=train, validation=validation, held=held)
        shutil.copyfile(
            cache / "targets-and-metadata.npz", out / "targets-and-metadata.npz"
        )
        torch.save(initial, out / "initial.pt")
        shutil.copyfile(old_model_path, out / "frozen-temporal-control.npz")
        if (
            digest(out / "frozen-temporal-control.npz")
            != registration["frozen_temporal_model_sha256"]
        ):
            raise ValueError("Frozen temporal baseline snapshot changed")
        role_receipt = {
            name: dict(
                people=sorted(set(data["person"][mask])),
                records=sorted(set(data["record"][mask])),
                rows=int(mask.sum()),
            )
            for name, mask in [
                ("train", train),
                ("validation", validation),
                ("held", held),
            ]
        }
        unique_rows = np.unique(schedule)
        frame_ids = (
            train_reader.offsets[unique_rows, None]
            + train_reader.ends[unique_rows, None]
            - np.arange(7, -1, -1)[None]
        )
        save(
            out / "preoptimization-seal.json",
            dict(
                files_sha256={
                    name: digest(out / name)
                    for name in [
                        "registration.json",
                        "schedule.npy",
                        "roles.npz",
                        "initial.pt",
                        "frozen-temporal-control.npz",
                    ]
                },
                roles=role_receipt,
                optimizer_unique_raw_frames=int(len(np.unique(frame_ids))),
                raw_frame_rule="Eachscheduledrow reads exactly end-7throughend within its ownsourceTRAINrecord; no extra denseframe exposure",
            ),
        )
    mean = data["pose"][train].reshape(-1, 34).mean(0)
    scale = np.maximum(data["pose"][train].reshape(-1, 34).std(0), 0.01)
    if resume:
        with np.load(
            out / "supervised-training-statistics.npz", allow_pickle=False
        ) as saved:
            np.testing.assert_array_equal(mean, saved["mean"])
            np.testing.assert_array_equal(scale, saved["scale"])
    else:
        np.savez(out / "supervised-training-statistics.npz", mean=mean, scale=scale)
    identity = dict(
        registration=digest(out / "registration.json"),
        schedule=digest(out / "schedule.npy"),
        initial=digest(out / "initial.pt"),
    )
    for arm in ARMS:
        if resume and (out / (arm + "-complete.json")).exists():
            complete = json.loads((out / (arm + "-complete.json")).read_text())
            if complete["identity"] != identity or complete["model_sha256"] != digest(
                out / (arm + "-final.pt")
            ):
                raise ValueError("Completed arm bytes or experiment identity changed")
            continue
        arm_started = time.monotonic()
        model = MaskedCSI().to(device)
        model.load_state_dict(copy.deepcopy(initial))
        logs = []
        elapsed_previous = 0.0
        if arm != "random_frozen":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=CONFIG["learning_rate"],
                weight_decay=CONFIG["weight_decay"],
            )
            generator = torch.Generator().manual_seed(CONFIG["mask_seed"])
            completed = 0
            if resume:
                completed, logs, elapsed_previous = restore_checkpoint(
                    out, arm, model, optimizer, generator, identity, device
                )
                if not 0 <= completed <= len(schedule):
                    raise ValueError("Checkpoint step outside frozen schedule")
            model.train()
            for step, ids in enumerate(schedule):
                if step < completed:
                    continue
                raw_input, observed_input = train_reader(ids, return_observed=True)
                x = torch.from_numpy(raw_input).to(device)
                observed = torch.from_numpy(observed_input).to(device)
                mask = fixed_count_mask(len(ids) * 8, generator).to(device)
                target = None
                if arm == "supervised":
                    target = torch.as_tensor(
                        (data["pose"][ids].reshape(-1, 34) - mean) / scale,
                        dtype=torch.float32,
                        device=device,
                    )
                loss, terms = optimizer_step(
                    model, optimizer, x, mask, arm, target, observed
                )
                if step % 100 == 0 or step + 1 == len(schedule):
                    logs.append(dict(step=step + 1, loss=loss, terms=terms))
                    print(
                        json.dumps(
                            dict(
                                arm=arm,
                                step=step + 1,
                                loss=loss,
                                elapsed_seconds=elapsed_previous
                                + time.monotonic()
                                - arm_started,
                            )
                        ),
                        flush=True,
                    )
                if (step + 1) % 1000 == 0 or step + 1 == len(schedule):
                    save_checkpoint(
                        out,
                        arm,
                        step + 1,
                        model,
                        optimizer,
                        generator,
                        identity,
                        logs,
                        elapsed_previous + time.monotonic() - arm_started,
                        device,
                    )
            if (out / (arm + "-optimizer.pt")).exists():
                raise FileExistsError(
                    "Unsealed final optimizer exists; inspect interrupted finalization"
                )
            torch.save(cpu_state(optimizer.state_dict()), out / (arm + "-optimizer.pt"))
        if (out / (arm + "-final.pt")).exists():
            raise FileExistsError(
                "Unsealed final model exists; inspect interrupted finalization"
            )
        torch.save(
            {key: value.cpu() for key, value in model.state_dict().items()},
            out / (arm + "-final.pt"),
        )
        save(
            out / (arm + "-training.json"),
            dict(
                logs=logs,
                seconds=elapsed_previous + time.monotonic() - arm_started,
                steps=0 if arm == "random_frozen" else len(schedule),
                selection="Fixed finalstep, no pose-based checkpointselection",
            ),
        )
        save(
            out / (arm + "-complete.json"),
            dict(identity=identity, model_sha256=digest(out / (arm + "-final.pt"))),
        )
        del model
    if resume and any(out.glob("*-embedding.npy")):
        raise ValueError(
            "Interrupted source extraction exists; preserve and inspect before replay"
        )
    # All neural checkpoints precede feature extraction or source probe tuning.
    selections = {}
    for arm in ARMS:
        model = MaskedCSI().to(device)
        model.load_state_dict(
            torch.load(
                out / (arm + "-final.pt"), map_location=device, weights_only=True
            )
        )
        feature = extract(
            model, all_reader, len(train), device, out / (arm + "-embedding.npy")
        )
        z = rff_transform(feature, train, out / (arm + "-rff.npz"))
        state, grid = fit_readout(z, data["pose"], train, validation, data["room"])
        np.savez(out / (arm + "-readout.npz"), **state, prediction=predict(z, state))
        selections[arm] = dict(grid=grid, selected=int(state["selected"]))
        del model, feature
    raw_current = hermitian_coordinates(
        current_covariances(covariance["untreated_covariance"])
    ).reshape(len(train), -1)
    z = rff_transform(raw_current, train, out / "raw_current-rff.npz")
    state, grid = fit_readout(z, data["pose"], train, validation, data["room"])
    np.savez(out / "raw_current-readout.npz", **state, prediction=predict(z, state))
    selections["raw_current"] = dict(grid=grid, selected=int(state["selected"]))
    save(out / "frozen-source-selections.json", selections)
    save(
        out / "source-fit-time.json",
        dict(seconds=time.monotonic() - started, held_metrics_computed=False),
    )
    seal_source_ready(out)
    return out


def score_frozen(out, cache):
    if (out / "results.json").exists() or (out / "query-permutation.npy").exists():
        raise FileExistsError("Held scoring already started; preserve prior output")
    ready = json.loads((out / "source-ready.json").read_text())
    if (
        ready["kind"] != "masked-csi-source-only-ready"
        or ready["held_metrics_computed"] is not False
    ):
        raise ValueError("Expected source-only frozen fit receipt")
    required = {
        "registration.json",
        "initial.pt",
        "schedule.npy",
        "roles.npz",
        "targets-and-metadata.npz",
        "frozen-source-selections.json",
        "frozen-temporal-control.npz",
        "raw_current-rff.npz",
        "raw_current-readout.npz",
    }
    required.update(
        arm + suffix
        for arm in ARMS
        for suffix in ["-final.pt", "-rff.npz", "-readout.npz"]
    )
    if not required <= set(ready["artifacts_sha256"]):
        raise ValueError("Frozen receipt omits required model/map/selection artifacts")
    verify_hashes(out, ready["artifacts_sha256"])
    registration = json.loads((out / "registration.json").read_text())
    verify_hashes(ROOT, registration["source_sha256"])
    if digest(cache / "receipt.json") != registration["cache_receipt_sha256"]:
        raise ValueError("Scoring cache receipt differs")
    _, data, _ = load_inputs(cache)
    if digest(cache / "targets-and-metadata.npz") != digest(
        out / "targets-and-metadata.npz"
    ):
        raise ValueError("Frozen scoring metadata differs from canonical cache")
    train, validation, held = fold_masks(
        data["room"], data["person"], registration["config"]["held_room"]
    )
    with np.load(out / "roles.npz", allow_pickle=False) as saved:
        for name, mask in [
            ("train", train),
            ("validation", validation),
            ("held", held),
        ]:
            np.testing.assert_array_equal(saved[name], mask)
    frozen = digest(out / "frozen-source-selections.json")
    ids = np.flatnonzero(held)
    perm = query_permutation(data["record"][ids])
    np.save(out / "query-permutation.npy", perm, allow_pickle=False)
    predictions = {}
    for arm in (*ARMS, "raw_current"):
        with np.load(out / (arm + "-readout.npz"), allow_pickle=False) as saved:
            predictions[arm] = saved["prediction"][ids]
    predictions["source_mean"] = np.broadcast_to(
        data["pose"][train].mean(0), data["pose"][held].shape
    )
    if (
        digest(out / "frozen-temporal-control.npz")
        != registration["frozen_temporal_model_sha256"]
    ):
        raise ValueError("Frozen temporal snapshot changed before scoring")
    with np.load(out / "frozen-temporal-control.npz", allow_pickle=False) as old:
        if not np.array_equal(old["held_indices"], ids):
            raise ValueError("Frozen temporal query rows differ")
        predictions["frozen_temporal"] = old["pose"]
    results = {name: scores(p, data, ids, perm) for name, p in predictions.items()}
    primary = results["masked_temporal"]["aligned"]
    controls = [
        results[name]["aligned"]
        for name in (
            "source_mean",
            "random_frozen",
            "supervised",
            "raw_current",
            "frozen_temporal",
        )
    ]
    gate = dict(
        nje_better_all_controls=all(
            primary["normalized_joint_error"] < c["normalized_joint_error"]
            for c in controls
        ),
        pck_not_worse_controls=all(
            primary["pck_curve"]["0.05"] >= c["pck_curve"]["0.05"]
            and primary["pck_curve"]["0.2"] >= c["pck_curve"]["0.2"]
            for c in controls
        ),
        positive_motion_r2=primary["record_centered_articulation_r2"] > 0,
        shuffle_worse=primary["normalized_joint_error"]
        < results["masked_temporal"]["shuffled"]["normalized_joint_error"],
    )
    save(
        out / "results.json",
        dict(
            registration=registration,
            frozen_source_selection_sha256=frozen,
            methods=results,
            gate=gate,
            passes=all(gate.values()),
            source_ready_sha256=digest(out / "source-ready.json"),
        ),
    )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--score-frozen-out", type=Path)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--old-probe", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.benchmark:
        print(json.dumps(benchmark(args.device), indent=2))
    elif args.score_frozen_out is not None:
        if args.cache is None:
            parser.error("--cache required for frozen scoring")
        score_frozen(args.score_frozen_out, args.cache)
    else:
        if any(
            value is None for value in (args.raw, args.cache, args.old_probe, args.out)
        ):
            parser.error("--raw --cache --old-probe --out required")
        run(
            args.raw,
            args.cache,
            args.old_probe,
            args.out,
            args.device,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
