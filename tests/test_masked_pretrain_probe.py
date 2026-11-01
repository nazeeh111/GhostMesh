import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from ghostmesh.masked_csi import MaskedCSI, fixed_count_mask
from scripts.domain_masked_pretrain_probe import (
    HistoryReader,
    make_schedule,
    optimizer_step,
    query_permutation,
    run,
)


def synthetic():
    rng = np.random.default_rng(7)
    rf = rng.normal(size=(60, 6, 114, 10)).astype(np.float32)
    meta = dict(
        record=np.array(["train", "train", "val", "held"]),
        end=np.array([7, 15, 10, 12]),
    )
    records = [
        dict(id=name, offset=20 * i) for i, name in enumerate(["train", "val", "held"])
    ]
    allowed = np.array([True, True, False, False])
    return rf, meta, records, allowed


def test_reader_enforces_allowed_rows_and_only_earlier_same_record_eight_frames():
    rf, meta, records, allowed = synthetic()

    class Tracked:
        def __getitem__(self, index):
            self.index = index
            return rf[index]

    tracked = Tracked()
    reader = HistoryReader(tracked, meta, records, allowed)
    before = reader(np.array([1]))
    np.testing.assert_array_equal(tracked.index[0], np.arange(8, 16)[None])
    rf[16:] = 1e9
    np.testing.assert_array_equal(before, reader(np.array([1])))
    for ids in [
        np.array([2]),
        np.array([3]),
        np.array([-1]),
        np.array([4]),
        np.array([1.0]),
    ]:
        with pytest.raises(ValueError, match="outside"):
            reader(ids)
    assert not hasattr(reader, "pose")


def test_matched_schedule_only_reads_source_training_rows_and_balances_rooms():
    room = np.repeat(["E01", "E02", "E03"], 40)
    train = room != "E02"
    train[30:40] = False
    train[110:120] = False
    a = make_schedule(room, train, steps=20, batch=8)
    np.testing.assert_array_equal(a, make_schedule(room, train, steps=20, batch=8))
    assert train[a].all() and set(room[a].ravel()) == {"E01", "E03"}
    assert all(len(set(row)) == 8 and len(set(room[row])) == 1 for row in a)
    with pytest.raises(ValueError):
        make_schedule(room, np.ones(len(room), bool), steps=1, batch=8)


def test_actual_pretext_step_cannot_consume_labels_or_validation_held_rf():
    torch.set_num_threads(1)
    torch.manual_seed(4)
    initial = MaskedCSI().state_dict()
    rf, meta, records, allowed = synthetic()
    reader = HistoryReader(rf, meta, records, allowed)
    mask = fixed_count_mask(16, torch.Generator().manual_seed(5))
    states = []
    for poison in [False, True]:
        if poison:
            rf[20:] = 1e9
            meta["pose"] = np.full((4, 17, 2), np.nan)
        model = MaskedCSI()
        model.load_state_dict(copy.deepcopy(initial))
        opt = torch.optim.AdamW(model.parameters(), lr=0.001)
        histories = torch.from_numpy(reader(np.array([0, 1])))
        optimizer_step(model, opt, histories, mask, "masked_temporal")
        states.append(copy.deepcopy(model.state_dict()))
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="does not accept pose"):
        optimizer_step(
            model, opt, histories, mask, "masked_temporal", torch.ones(2, 34)
        )


def test_query_shuffle_reproducible_preserves_record_membership():
    record = np.repeat(["a", "b", "c"], 12)
    perm = query_permutation(record)
    np.testing.assert_array_equal(record[perm], record)
    np.testing.assert_array_equal(np.sort(perm), np.arange(len(record)))
    np.testing.assert_array_equal(perm, query_permutation(record))


def test_existing_output_refused_before_any_data_access(tmp_path):
    with pytest.raises(FileExistsError):
        run(
            tmp_path / "absent",
            tmp_path / "absent",
            tmp_path / "absent",
            tmp_path,
            "cpu",
        )


def test_actual_rff_and_probe_selection_ignore_poisoned_held_features_and_labels(
    tmp_path, monkeypatch
):
    from scripts import domain_covariance_source_probe as source
    from scripts.domain_reference_covariance_probe import fit_readout
    from scripts.domain_masked_pretrain_probe import rff_transform

    # Small feature count shortens this real-fit boundary test; same map builder,
    # alpha grid, normalization, validation metric and selector as the pilot.
    monkeypatch.setitem(source.CONFIG, "rff_dimensions", 16)
    rng = np.random.default_rng(5)
    feature = rng.normal(size=(168, 12))
    pose = rng.normal(size=(168, 17, 2))
    train = np.arange(168) < 128
    validation = (np.arange(168) >= 128) & (np.arange(168) < 148)
    held = np.arange(168) >= 148
    rooms = np.tile(["E01", "E03"], 84)
    rooms[held] = "E02"
    states, grids, maps = [], [], []
    for poison in [False, True]:
        if poison:
            feature[held] = feature[held] * 1e6 + 1e9
            pose[held] += 1e9
        path = tmp_path / ("poison.npz" if poison else "clean.npz")
        z = rff_transform(feature, train, path)
        state, grid = fit_readout(z, pose, train, validation, rooms)
        states.append(state)
        grids.append(grid)
        with np.load(path, allow_pickle=False) as saved:
            maps.append({key: saved[key] for key in saved.files})
    for key in maps[0]:
        np.testing.assert_array_equal(maps[0][key], maps[1][key])
    for key in states[0]:
        np.testing.assert_array_equal(states[0][key], states[1][key])
    assert grids[0] == grids[1]


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_gradient_refused_before_optimizer_weights_unchanged(bad):
    torch.set_num_threads(1)
    torch.manual_seed(2)
    model = MaskedCSI()
    before = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    model.patch.weight.register_hook(lambda grad: torch.full_like(grad, bad))
    x = torch.randn(2, 8, 3, 114, 10)
    mask = fixed_count_mask(16, torch.Generator().manual_seed(3))
    with pytest.raises(RuntimeError, match="non-finite"):
        optimizer_step(model, optimizer, x, mask, "masked_temporal")
    assert not optimizer.state
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_checkpoint_restores_optimizer_rng_mask_schedule_exactly_and_rejects_tampering(
    tmp_path,
):
    from scripts.domain_masked_pretrain_probe import save_checkpoint, restore_checkpoint

    torch.set_num_threads(1)
    torch.manual_seed(3)
    model = MaskedCSI()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(2, 8, 3, 114, 10)
    identity = dict(registration="reg", schedule="sched", initial="init")
    optimizer_step(
        model, optimizer, x, fixed_count_mask(16, generator), "masked_temporal"
    )
    path = save_checkpoint(
        tmp_path,
        "masked_temporal",
        1,
        model,
        optimizer,
        generator,
        identity,
        [dict(step=1)],
        0.5,
        "cpu",
    )
    mask = fixed_count_mask(16, generator)
    optimizer_step(model, optimizer, x, mask, "masked_temporal")
    continuous = copy.deepcopy(model.state_dict())
    continuous_optimizer = copy.deepcopy(optimizer.state_dict())
    next_random = torch.rand(3)
    resumed = MaskedCSI()
    resumed_optimizer = torch.optim.AdamW(
        resumed.parameters(), lr=0.001, weight_decay=0.01
    )
    resumed_generator = torch.Generator().manual_seed(999)
    step, logs, elapsed = restore_checkpoint(
        tmp_path,
        "masked_temporal",
        resumed,
        resumed_optimizer,
        resumed_generator,
        identity,
        "cpu",
    )
    assert (step, logs, elapsed) == (1, [dict(step=1)], 0.5)
    torch.testing.assert_close(fixed_count_mask(16, resumed_generator), mask)
    optimizer_step(resumed, resumed_optimizer, x, mask, "masked_temporal")
    for key in continuous:
        torch.testing.assert_close(
            resumed.state_dict()[key], continuous[key], rtol=0, atol=0
        )
    torch.testing.assert_close(torch.rand(3), next_random, rtol=0, atol=0)
    for index, state in continuous_optimizer["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(
                resumed_optimizer.state_dict()["state"][index][key],
                value,
                rtol=0,
                atol=0,
            )
    with pytest.raises(ValueError, match="identity"):
        restore_checkpoint(
            tmp_path,
            "masked_temporal",
            resumed,
            resumed_optimizer,
            resumed_generator,
            dict(identity, initial="changed"),
            "cpu",
        )
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="bytes changed"):
        restore_checkpoint(
            tmp_path,
            "masked_temporal",
            resumed,
            resumed_optimizer,
            resumed_generator,
            identity,
            "cpu",
        )


def test_partial_checkpoint_without_seal_refuses_silent_resume(tmp_path):
    from scripts.domain_masked_pretrain_probe import restore_checkpoint

    (tmp_path / "masked_temporal-step-01000.pt").write_bytes(b"incomplete")
    with pytest.raises(ValueError, match="Unsealed partial"):
        restore_checkpoint(tmp_path, "masked_temporal", None, None, None, {}, "cpu")


def frozen_scoring_bundle(tmp_path, monkeypatch):
    import json
    from scripts import domain_masked_pretrain_probe as probe

    out, cache = tmp_path / "out", tmp_path / "cache"
    out.mkdir()
    cache.mkdir()
    (cache / "receipt.json").write_text("{}")
    rng = np.random.default_rng(10)
    pose = rng.normal(size=(12, 17, 2))
    data = dict(
        pose=pose,
        delta=np.zeros_like(pose),
        room=np.array(["E01", "E03"] * 4 + ["E02"] * 4),
        person=np.array(["p" + str(i) for i in range(12)]),
        record=np.array(["train"] * 6 + ["val"] * 2 + ["q1", "q1", "q2", "q2"]),
        action=np.array(["A01"] * 12),
        end=np.arange(12),
    )
    train = np.arange(12) < 6
    validation = (np.arange(12) >= 6) & (np.arange(12) < 8)
    held = np.arange(12) >= 8
    np.savez(cache / "targets-and-metadata.npz", **data)
    (out / "targets-and-metadata.npz").write_bytes(
        (cache / "targets-and-metadata.npz").read_bytes()
    )
    np.savez(out / "roles.npz", train=train, validation=validation, held=held)
    np.savez(
        out / "frozen-temporal-control.npz",
        pose=pose[held] + 0.01,
        held_indices=np.flatnonzero(held),
    )
    source = Path(probe.__file__)
    registration = dict(
        config=dict(held_room="E02"),
        source_sha256={str(source.relative_to(probe.ROOT)): probe.digest(source)},
        cache_receipt_sha256=probe.digest(cache / "receipt.json"),
        frozen_temporal_model_sha256=probe.digest(out / "frozen-temporal-control.npz"),
    )
    (out / "registration.json").write_text(json.dumps(registration))
    (out / "frozen-source-selections.json").write_text("{}")
    for name in ["initial.pt", "schedule.npy", "raw_current-rff.npz"]:
        (out / name).write_bytes(b"sealed-synthetic-placeholder")
    for arm in probe.ARMS:
        (out / (arm + "-final.pt")).write_bytes(b"sealed-synthetic-model")
        (out / (arm + "-rff.npz")).write_bytes(b"sealed-synthetic-map")
    for arm in (*probe.ARMS, "raw_current"):
        np.savez(
            out / (arm + "-readout.npz"),
            prediction=pose if arm == "masked_temporal" else pose + 0.01,
        )
    probe.seal_source_ready(out)
    monkeypatch.setattr(probe, "load_inputs", lambda path: ({}, data, {}))
    monkeypatch.setattr(probe, "fold_masks", lambda *args: (train, validation, held))

    def no_refitting(*args, **kwargs):
        raise AssertionError("Frozen scoring must never fit models or maps")

    for name in ["fit_readout", "rff_transform", "optimizer_step", "run"]:
        monkeypatch.setattr(probe, name, no_refitting)
    return probe, out, cache


def test_separate_frozen_scoring_uses_saved_predictions_no_refitting(
    tmp_path, monkeypatch
):
    import json

    probe, out, cache = frozen_scoring_bundle(tmp_path, monkeypatch)
    assert not (out / "results.json").exists()
    probe.score_frozen(out, cache)
    results = json.loads((out / "results.json").read_text())
    assert (
        results["methods"]["masked_temporal"]["aligned"]["normalized_joint_error"] == 0
    )
    assert "source_ready_sha256" in results
    with pytest.raises(FileExistsError):
        probe.score_frozen(out, cache)


def test_frozen_scoring_refuses_changed_map_before_any_metrics(tmp_path, monkeypatch):
    probe, out, cache = frozen_scoring_bundle(tmp_path, monkeypatch)
    (out / "masked_temporal-rff.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        probe.score_frozen(out, cache)
    assert not (out / "results.json").exists()
