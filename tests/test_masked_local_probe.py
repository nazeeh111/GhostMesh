"""Fabricated software fixtures only, never evidence of sensing accuracy."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ghostmesh.masked_csi import MaskedCSI
from scripts import domain_masked_local_probe as probe


def raw_record():
    rng = np.random.default_rng(19)
    raw = rng.normal(size=(297, 6, 114, 10)).astype(np.float32)
    raw[:, :3] += 40
    return raw


def local_fixture():
    rng = np.random.default_rng(22)
    ends = np.array([7, 15, 23, 31, 39, 47, 133, 141, 149, 157, 193, 201, 209, 217])
    split = np.array(["train"] * 6 + ["validation"] * 4 + ["query"] * 4)
    feature = rng.normal(size=(len(ends), 9))
    pose = rng.uniform(0.3, 0.7, size=(len(ends), 17, 2))
    return feature, pose, ends, split


def test_fixed_purged_causal_supports_are_disjoint_and_have_declared_counts():
    ends, history, split = probe.supports()
    assert [int((split == name).sum()) for name in probe.BLOCKS] == [111, 45, 104]
    supports = []
    for name, (start, stop) in probe.BLOCKS.items():
        ids = split == name
        assert history[ids].min() >= start and history[ids].max() < stop
        np.testing.assert_array_equal(history[ids, -1], ends[ids])
        np.testing.assert_array_equal(
            np.diff(history[ids], axis=1), np.ones((ids.sum(), 7))
        )
        supports.append(set(history[ids].ravel()))
    assert not any(supports[i] & supports[j] for i in range(3) for j in range(i))
    assert min(supports[1]) - max(supports[0]) - 1 == 8
    assert min(supports[2]) - max(supports[1]) - 1 == 8


def test_quality_mask_is_signal_only_common_and_never_bridges_invalid_frames():
    raw = raw_record()
    _, history, split = probe.supports()
    raw[5, :3] = np.nan
    raw[150, 3:] = np.nan
    raw[220, :3] = 4
    keep, quality = probe.common_quality(raw, history)
    assert not keep[(history == 5).any(1)].any()
    assert not keep[(history == 150).any(1)].any()
    assert not keep[(history == 220).any(1)].any()
    assert quality["retained_rows"] == int(keep.sum())
    assert all(np.any(keep & (split == name)) for name in probe.BLOCKS)


def test_partial_missing_values_remain_causal_and_all_raw_controls_match_dimensions():
    raw = raw_record()
    raw[10, 1, 4, 2] = np.nan
    history = np.array([np.arange(8, 16), np.arange(20, 28)])
    keep, _ = probe.common_quality(raw, history)
    assert keep.all()
    before = probe.raw_features(raw, history)
    raw[28:] = 1e6
    after = probe.raw_features(raw, history)
    assert before["quotient"].shape == (2, 54720)
    assert before["raw_log"].shape == (2, 27360)
    for name in before:
        assert np.isfinite(before[name]).all()
        np.testing.assert_array_equal(before[name], after[name])


def test_encoder_optimizer_role_exclusion_is_required():
    people = np.array(["S04", "S04", "S01", "S21"])
    train = np.array([False, False, True, True])
    validation = ~train
    schedule = np.array([[2, 3]])
    probe.check_encoder_exclusion(people, train, validation, schedule)
    with pytest.raises(ValueError, match="exclusively"):
        probe.check_encoder_exclusion(
            people, train | (people == "S04"), np.zeros(4, bool), schedule
        )
    with pytest.raises(ValueError, match="prohibited"):
        probe.check_encoder_exclusion(people, train, validation, np.array([[0, 2]]))


def test_encoder_inference_is_frozen_and_cannot_call_pose_heads_decoders_or_optimizers(
    monkeypatch,
):
    torch.set_num_threads(1)
    torch.manual_seed(3)
    model = MaskedCSI()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    monkeypatch.setattr(
        model.pose_head, "forward", lambda *a: pytest.fail("global pose head called")
    )
    monkeypatch.setattr(
        model.decoder, "forward", lambda *a: pytest.fail("decoder called")
    )
    monkeypatch.setattr(
        model, "forward", lambda *a: pytest.fail("whole model forward called")
    )
    monkeypatch.setattr(
        torch.optim, "AdamW", lambda *a, **k: pytest.fail("optimizer called")
    )
    raw = raw_record()
    history = np.array([np.arange(8), np.arange(8, 16)])
    x = probe.encoder_features(model, raw, history)
    raw[8:] += 10
    changed = probe.encoder_features(model, raw, history)
    np.testing.assert_array_equal(x[0], changed[0])
    assert x.shape == (2, 512)
    assert all(not p.requires_grad for p in model.parameters())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(before[name], value, rtol=0, atol=0)


def test_checkpoint_loader_discards_all_pose_and_pretext_heads(tmp_path):
    model = MaskedCSI()
    state = model.state_dict()
    for name in list(state):
        if name.startswith(("pose_head.", "decoder.", "projector.")):
            state[name] = torch.full_like(state[name], float("nan"))
    path = tmp_path / "frozen-final.pt"
    torch.save(state, path)
    backbone = probe.load_backbone(path)
    assert not any(
        hasattr(backbone, name) for name in ("pose_head", "decoder", "projector")
    )
    assert all(torch.isfinite(value).all() for value in backbone.state_dict().values())
    for name, value in backbone.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)


def test_local_fit_query_label_and_signal_poison_cannot_change_choices_or_coefficients():
    x, y, ends, split = local_fixture()
    before = probe.fit_arrays({"encoder": x}, y, ends, split)
    x2, y2 = x.copy(), y.copy()
    x2[split == "query"] = 1e6
    y2[split == "query"] = np.nan
    after = probe.fit_arrays({"encoder": x2}, y2, ends, split)
    assert before[1] == after[1]
    for tag, state in before[0].items():
        for name in state:
            np.testing.assert_array_equal(state[name], after[0][tag][name])
    for space in ("hip_relative", "full_image"):
        for control in ("source_mean", "source_time_trend"):
            np.testing.assert_array_equal(
                before[2][space + "/" + control], after[2][space + "/" + control]
            )


def test_dual_solves_remain_training_sized_and_match_direct_equation(monkeypatch):
    x, y, _, split = local_fixture()
    x = np.tile(x, (1, 40))
    train, validation = split == "train", split == "validation"
    original = probe.eigh
    dimensions = []

    def checked(matrix, **kwargs):
        dimensions.append(matrix.shape)
        assert matrix.shape == (train.sum(), train.sum())
        return original(matrix, **kwargs)

    monkeypatch.setattr(probe, "eigh", checked)
    state, choice, prediction = probe.choose_head(x, y, train, validation)
    kernels, _ = probe.kernel_inputs(x, train)
    kernel = kernels[str(state["kernel"])]
    expected = np.linalg.solve(
        kernel[train] + float(state["alpha"]) * np.eye(train.sum()),
        y[train].reshape(train.sum(), 34) - state["target_mean"],
    )
    np.testing.assert_allclose(state["dual"], expected, atol=1e-8, rtol=1e-9)
    np.testing.assert_allclose(
        prediction,
        (kernel @ expected + state["target_mean"]).reshape(y.shape),
        atol=1e-10,
    )
    assert dimensions == [(6, 6), (6, 6)] and len(choice["grid"]) == 10


def test_constant_features_do_not_fabricate_an_rbf_bandwidth():
    _, y, _, split = local_fixture()
    state, choice, prediction = probe.choose_head(
        np.ones((len(y), 3)), y, split == "train", split == "validation"
    )
    assert choice["rbf_available"] is False
    assert str(state["kernel"]) == "linear"
    np.testing.assert_allclose(
        prediction, np.broadcast_to(y[split == "train"].mean(0), y.shape), atol=1e-15
    )


@pytest.mark.parametrize("field", ["feature", "target", "raw"])
def test_declared_real_local_inputs_reject_complex_values(field):
    x, y, ends, split = local_fixture()
    with pytest.raises(ValueError, match="real-valued"):
        if field == "raw":
            probe.common_quality(raw_record().astype(complex), np.arange(8)[None])
        else:
            if field == "feature":
                x = x.astype(complex) + 1e6j
            else:
                y = y.astype(complex)
            probe.fit_arrays({"x": x}, y, ends, split)


def source_fixture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    manifest = {
        name: probe.digest(probe.ROOT / name)
        for name in probe.REQUIRED_TRAINING_SOURCES
    }
    reg = dict(
        config=dict(steps=4000, batch=16, held_room="E02"), source_sha256=manifest
    )
    probe.save_json(source / "registration.json", reg)
    for name in manifest:
        destination = source / "executed-source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((probe.ROOT / name).read_bytes())
    people = np.array(["S04", "S04", "S01", "S21"])
    probe.save_npz(source / "targets-and-metadata.npz", person=people)
    probe.save_npz(
        source / "roles.npz",
        train=np.array([False, False, True, True]),
        validation=np.array([True, True, False, False]),
    )
    probe.save_array(source / "schedule.npy", np.tile([2, 3], (4000, 8)))
    for arm in probe.ENCODERS:
        (source / (arm + "-final.pt")).write_bytes(
            b"no model decoding in role validator"
        )
        probe.save_json(source / (arm + "-complete.json"), {})
        probe.save_json(
            source / (arm + "-training.json"),
            dict(
                steps=0 if arm == "random_frozen" else 4000,
                selection="Fixed finalstep, no pose-based checkpointselection",
            ),
        )
    probe.save_json(
        source / "source-ready.json",
        dict(
            kind="masked-csi-source-only-ready",
            held_metrics_computed=False,
            artifacts_sha256={
                str(p.relative_to(source)): probe.digest(p)
                for p in source.rglob("*")
                if p.is_file()
            },
        ),
    )
    return source


def test_role_verification_never_loads_global_pose_heads_or_rff_maps(
    tmp_path, monkeypatch
):
    source = source_fixture(tmp_path)
    original = np.load

    def checked(path, *args, **kwargs):
        assert not str(path).endswith(("-rff.npz", "-readout.npz", "-embedding.npy"))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", checked)
    reg = probe.frozen_encoder_role(source, probe.digest(source / "source-ready.json"))
    assert reg["config"]["steps"] == 4000


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "omitted_producer",
        "unrelated_only",
        "snapshot_not_sealed",
        "foreign_snapshot",
    ],
)
def test_producer_provenance_rejects_incomplete_or_unbound_sources_before_array_reads(
    tmp_path, monkeypatch, case
):
    source = source_fixture(tmp_path)
    registration_path = source / "registration.json"
    reg = json.loads(registration_path.read_text())
    if case == "empty":
        reg["source_sha256"] = {}
    elif case == "omitted_producer":
        del reg["source_sha256"]["scripts/domain_masked_pretrain_probe.py"]
    elif case == "unrelated_only":
        name = "scripts/domain_masked_local_probe.py"
        reg["source_sha256"] = {name: probe.digest(probe.ROOT / name)}
        (source / "executed-source" / name).write_bytes(
            (probe.ROOT / name).read_bytes()
        )
    elif case == "foreign_snapshot":
        path = source / "executed-source/ghostmesh/masked_csi.py"
        path.write_bytes(path.read_bytes() + b"\n# Foreign executed version\n")
    registration_path.write_text(json.dumps(reg))
    ready_path = source / "source-ready.json"
    ready = json.loads(ready_path.read_text())
    ready["artifacts_sha256"] = {
        str(p.relative_to(source)): probe.digest(p)
        for p in source.rglob("*")
        if p.is_file() and p != ready_path
    }
    if case == "snapshot_not_sealed":
        del ready["artifacts_sha256"]["executed-source/ghostmesh/masked_csi.py"]
    ready_path.write_text(json.dumps(ready))
    monkeypatch.setattr(
        np,
        "load",
        lambda *a, **k: pytest.fail(
            "array decoded before producer provenance rejection"
        ),
    )
    monkeypatch.setattr(
        torch,
        "load",
        lambda *a, **k: pytest.fail(
            "checkpoint decoded before producer provenance rejection"
        ),
    )
    with pytest.raises(ValueError, match="producer"):
        probe.frozen_encoder_role(source, probe.digest(ready_path))


@pytest.mark.parametrize(
    "case",
    [
        "complex",
        "zero_imaginary_complex",
        "nan",
        "inf",
        "wrong_dtype",
        "wrong_shape",
        "not_tensor",
        "missing_tensor",
        "extra_tensor",
    ],
)
def test_backbone_loader_rejects_invalid_retained_tensors_before_state_conversion(
    tmp_path, case
):
    state = MaskedCSI().state_dict()
    name = "embedding.weight"
    if case == "complex":
        state[name] = state[name].to(torch.complex64) + 1e6j
    elif case == "zero_imaginary_complex":
        state[name] = state[name].to(torch.complex64)
    elif case == "nan":
        state[name][0, 0] = float("nan")
    elif case == "inf":
        state[name][0, 0] = float("inf")
    elif case == "wrong_dtype":
        state[name] = state[name].double()
    elif case == "wrong_shape":
        state[name] = state[name][1:]
    elif case == "not_tensor":
        state[name] = [1.0, 2.0]
    elif case == "missing_tensor":
        del state[name]
    elif case == "extra_tensor":
        state["unknown.weight"] = torch.ones(1)
    path = tmp_path / "invalid-final.pt"
    torch.save(state, path)
    with pytest.raises(ValueError, match="backbone tensor"):
        probe.load_backbone(path)


def test_existing_local_output_refuses_before_source_or_data_access(tmp_path):
    with pytest.raises(FileExistsError):
        probe.fit(tmp_path / "missing", tmp_path / "missing", tmp_path, "missing")


def test_actual_fit_never_indexes_query_annotations_before_choices_are_sealed(
    tmp_path, monkeypatch
):
    """Small fabricated file pipeline exercises the actual fit and score phases."""
    raw = tmp_path / "raw"
    source = tmp_path / "source"
    out = tmp_path / "out"
    raw.mkdir()
    source.mkdir()
    signal = raw_record()
    pose = np.random.default_rng(4).uniform(0.3, 0.7, (297, 17, 2))
    probe.save_array(raw / "rf.npy", signal)
    probe.save_array(raw / "pose.npy", pose)
    probe.save_array(raw / "valid.npy", np.ones((297, 17), bool))
    receipt = dict(
        frames=240570,
        records=[dict(id=probe.CONFIG["record"], offset=0, frames=297)],
        arrays={
            name: probe.digest(raw / (name + ".npy"))
            for name in ("rf", "pose", "valid")
        },
    )
    probe.save_json(raw / "receipt.json", receipt)
    monkeypatch.setattr(
        probe,
        "frozen_encoder_role",
        lambda *a: dict(raw_receipt_sha256=probe.digest(raw / "receipt.json")),
    )
    model = MaskedCSI().state_dict()
    for arm in probe.ENCODERS:
        torch.save(model, source / (arm + "-final.pt"))
    monkeypatch.setattr(
        probe,
        "encoder_features",
        lambda model, rf, history: np.column_stack(
            [history[:, -1] / 297, np.sin(history[:, -1] / 10)]
        ),
    )
    original_load = np.load
    indexes = []
    may_read_query = False

    class AnnotationRows:
        def __init__(self, values):
            self.values = values

        def __getitem__(self, ids):
            indices = np.asarray(ids)
            indexes.extend(indices.tolist())
            if not may_read_query:
                assert not np.any(indices >= 186), (
                    "query GT consumed during local fitting"
                )
            return self.values[ids]

    def checked_load(path, *args, **kwargs):
        value = original_load(path, *args, **kwargs)
        return AnnotationRows(value) if Path(path) == raw / "pose.npy" else value

    monkeypatch.setattr(np, "load", checked_load)
    probe.fit(source, raw, out, "fixture-source-ready")
    assert max(indexes) <= 177
    assert not (out / "results.json").exists()
    with original_load(out / "local-data.npz", allow_pickle=False) as data:
        assert np.isnan(data["source_target"][data["split"] == "query"]).all()
    monkeypatch.setattr(
        probe, "choose_head", lambda *a: pytest.fail("refit during score")
    )
    ready_digest = probe.digest(out / "local-source-ready.json")
    path = out / "hip_relative-masked_temporal-state.npz"
    original_bytes = path.read_bytes()
    path.write_bytes(b"tampered coefficients")
    with pytest.raises(ValueError):
        probe.score_frozen(out, source, raw, ready_digest)
    assert max(indexes) <= 177 and not (out / "query-permutation.npy").exists()
    path.write_bytes(original_bytes)
    may_read_query = True
    probe.score_frozen(out, source, raw, ready_digest)
    assert (out / "results.json").exists() and max(indexes) == 296
    with pytest.raises(FileExistsError):
        probe.score_frozen(
            out, source, raw, probe.digest(out / "local-source-ready.json")
        )
