"""Synthetic full-image phase contracts, never real pose accuracy evidence."""

import copy
import json
import shutil

import numpy as np
import pytest

from scripts import domain_masked_fullimage_calibration_probe as probe
from scripts.domain_reference_covariance_probe import REF_ENDS


def fixture():
    rng = np.random.default_rng(82)
    names = [f"E02/S{p}/A{a:02}" for p in (11, 12) for a in (3, 7, 12)]
    record = np.concatenate(
        [
            np.repeat(names, 20),
            np.repeat([f"E02/S{p}/A03" for p in range(13, 21)], 3),
            np.repeat("E01/S01/A03", 10),
        ]
    )
    person = np.array([name.split("/")[1] for name in record])
    room = np.array([name.split("/")[0] for name in record])
    action = np.array([name.split("/")[2] for name in record])
    ends = np.concatenate(
        [np.tile(REF_ENDS, 6), np.tile(REF_ENDS[:3], 8), REF_ENDS[:10]]
    )
    n = len(record)
    x = rng.normal(size=(n, 5))
    skeleton = np.linspace(-0.2, 0.4, 34).reshape(17, 2)
    body = (
        skeleton[None]
        + 0.02 * np.sin(x[:, :1, None]) * np.linspace(0.1, 1, 17)[None, :, None]
    )
    body -= body[:, [11, 12]].mean(1, keepdims=True)
    full = body + (np.array([0.4, 0.5]) + 0.03 * x[:, :2])[:, None]
    data = dict(
        record=record, person=person, room=room, action=action, end=ends, pose=full
    )
    reference, query = np.arange(120), np.arange(120, 144)
    train = room == "E01"
    features = {arm: x.copy() for arm in probe.ARMS}
    base = np.broadcast_to(full[train].mean(0), full.shape).copy()
    bases = {arm: base.copy() for arm in probe.ARMS}
    return data, train, reference, query, features, bases


class ReferenceOnly:
    def __init__(self, values, reference):
        self.values, self.reference = values, reference

    def __getitem__(self, ids):
        np.testing.assert_array_equal(ids, self.reference)
        return self.values[ids]


def directories(tmp_path):
    source, cache, raw, full = [
        tmp_path / name for name in ("source", "cache", "raw", "full")
    ]
    for path in (source, cache, raw, full):
        path.mkdir()
    (source / "source-ready.json").write_text("synthetic reviewed body seal")
    (cache / "receipt.json").write_text("synthetic cache")
    (raw / "receipt.json").write_text("synthetic raw labels")
    return source, cache, raw, full


def test_actual_full_fit_query_label_and_rf_poison_leave_choices_and_models_exact(
    tmp_path, monkeypatch
):
    source, cache, raw, full = directories(tmp_path)
    ordinary = fixture()
    outputs = []
    for poison in (False, True):
        data, train, ref, query, features, bases = copy.deepcopy(ordinary)
        if poison:
            data["pose"][query] = np.nan
            for x in features.values():
                x[query] = 1e100
        readers = {arm: ReferenceOnly(x, ref) for arm, x in features.items()}
        monkeypatch.setattr(
            probe,
            "inputs",
            lambda *args, data=data, readers=readers: (
                data,
                train,
                ref,
                query,
                readers,
                bases,
            ),
        )
        out = tmp_path / ("poisoned" if poison else "ordinary")
        probe.fit(source, cache, raw, full, out, "reviewed synthetic full-head seal")
        outputs.append(out)
        assert not (out / "results.json").exists()
        assert (
            json.loads((out / "registration.json").read_text())["config"]["target_mode"]
            == "full_image_2d"
        )
    assert (outputs[0] / "frozen-reference-choices.json").read_bytes() == (
        outputs[1] / "frozen-reference-choices.json"
    ).read_bytes()
    for path in outputs[0].glob("*.npz"):
        with (
            np.load(path, allow_pickle=False) as a,
            np.load(outputs[1] / path.name, allow_pickle=False) as b,
        ):
            assert a.files == b.files
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])


def test_full_score_uses_frozen_models_without_refit_and_names_motion_correctly(
    tmp_path, monkeypatch
):
    source, cache, raw, full = directories(tmp_path)
    values = fixture()
    monkeypatch.setattr(probe, "inputs", lambda *args: values)
    out = tmp_path / "calibration"
    probe.fit(source, cache, raw, full, out, "synthetic full seal")
    monkeypatch.setattr(
        probe.calibration,
        "fit_reference_model",
        lambda *args, **kwargs: pytest.fail("Must not fit in scorer"),
    )
    probe.score(source, cache, raw, full, out)
    result = json.loads((out / "results.json").read_text())
    assert len(result["methods"]) == 34
    primary = result["methods"][probe.PRIMARY]["aligned"]
    assert {
        "record_centered_full_image_r2",
        "record_centered_body_r2",
        "record_centered_root_r2",
    } <= set(primary)
    assert "record_centered_articulation_r2" not in primary
    assert "full_pose_motion_positive" in result["gate"]
    assert "body_motion_positive" in result["gate"]
    data, _, _, query, _, bases = values
    zero = np.load(
        out / "supervised--zero_shot-query-prediction.npy", allow_pickle=False
    )
    np.testing.assert_array_equal(zero, bases["supervised"][query])
    assert not np.allclose(
        zero[:, [11, 12]].mean(1), data["pose"][query][:, [11, 12]].mean(1)
    )
    with pytest.raises(FileExistsError):
        probe.score(source, cache, raw, full, out)
    changed = tmp_path / "changed-fitted-state"
    shutil.copytree(out, changed)
    (changed / "results.json").unlink()
    (changed / "query-permutation.npy").unlink()
    (changed / "masked_temporal--residual_krr.npz").write_bytes(
        b"changed fitted full-image model"
    )
    with pytest.raises(ValueError):
        probe.score(source, cache, raw, full, changed)


def full_source_fixture(tmp_path, monkeypatch):
    source, cache, raw, full = directories(tmp_path)
    values = fixture()
    data, train, ref, query, features, bases = values
    target = data["pose"].copy()
    body_data = dict(data, pose=target - target[:, [11, 12]].mean(1, keepdims=True))
    validation = np.zeros(len(train), dtype=bool)
    held = ~train
    indices = np.arange(len(train))
    save = probe.calibration
    save.save_npz(full / "roles.npz", train=train, validation=validation, held=held)
    save.save_npz(
        full / "canonical-metadata.npz",
        **{key: data[key] for key in ("record", "person", "action", "room", "end")},
    )
    save.save_array(full / "full-targets.npy", target)
    save.save_array(full / "raw-frame-indices.npy", indices)
    save.save_json(
        full / "frozen-fullpose-selections.json",
        {arm: "synthetic source choice" for arm in probe.ARMS},
    )
    for arm in probe.ARMS:
        save.save_npz(
            full / (arm + "-root-state.npz"),
            synthetic=np.asarray("source-only synthetic root"),
        )
        save.save_array(full / (arm + "-full-prediction.npy"), bases[arm])
        if arm != "raw_current":
            save.save_array(
                source / (arm + "-embedding.npy"),
                np.tile(features[arm][:, :1], (1, 512)),
            )
    for arm in ("source_mean", "source_endpoint_clock"):
        save.save_array(full / (arm + "-full-prediction.npy"), target)
    source_sha = save.digest(source / "source-ready.json")
    reg = dict(
        source_ready_sha256=source_sha,
        cache_receipt_sha256=save.digest(cache / "receipt.json"),
        raw_receipt_sha256=save.digest(raw / "receipt.json"),
        source_sha256={},
    )
    save.save_json(full / "registration.json", reg)
    save.save_json(
        full / "fullpose-source-ready.json",
        dict(
            kind="masked-csi-fullimage-source-ready",
            held_metrics_computed=False,
            **{
                k: reg[k]
                for k in (
                    "source_ready_sha256",
                    "cache_receipt_sha256",
                    "raw_receipt_sha256",
                )
            },
            artifacts_sha256={path.name: save.digest(path) for path in full.iterdir()},
        ),
    )
    calls = []
    monkeypatch.setattr(
        probe.fullpose,
        "source_contract",
        lambda *args: calls.append("encoder-contract"),
    )

    def target_loader(*args):
        calls.append("raw-target-join")
        return (
            body_data,
            target,
            indices,
            train,
            validation,
            held,
            dict(untreated_covariance=np.zeros((len(train), 9576))),
        )

    monkeypatch.setattr(probe.fullpose, "inputs", target_loader)
    return (source, cache, raw, full), values, calls


def test_full_source_contract_is_required_before_labels_and_head_tampering_rejects(
    tmp_path, monkeypatch
):
    dirs, values, calls = full_source_fixture(tmp_path, monkeypatch)
    source, cache, raw, full = dirs
    expected = probe.calibration.digest(full / "fullpose-source-ready.json")
    loaded = probe.inputs(*dirs, expected)
    assert calls == ["encoder-contract", "raw-target-join"]
    np.testing.assert_array_equal(loaded[0]["pose"], values[0]["pose"])
    assert len(loaded[4]) == 6 and "frozen_temporal" not in loaded[4]
    calls.clear()
    with pytest.raises(ValueError, match="identity differs"):
        probe.inputs(*dirs, "wrong reviewed root-head seal")
    assert not calls
    (full / "masked_temporal-root-state.npz").write_bytes(b"mutated after seal")
    with pytest.raises(ValueError):
        probe.inputs(*dirs, expected)
    assert not calls


@pytest.mark.parametrize(
    "field", ["full-targets.npy", "raw-frame-indices.npy", "canonical-metadata.npz"]
)
def test_full_target_join_or_metadata_disagreement_rejects_even_if_locally_resealed(
    tmp_path, monkeypatch, field
):
    dirs, _, _ = full_source_fixture(tmp_path, monkeypatch)
    full = dirs[3]
    if field == "canonical-metadata.npz":
        with np.load(full / field, allow_pickle=False) as saved:
            changed = {key: saved[key] for key in saved.files}
        changed["end"] = changed["end"] + 1
        with (full / field).open("wb") as stream:
            np.savez(stream, **changed)
    else:
        value = np.load(full / field, allow_pickle=False)
        with (full / field).open("wb") as stream:
            np.save(stream, value + 1, allow_pickle=False)
    ready_path = full / "fullpose-source-ready.json"
    ready = json.loads(ready_path.read_text())
    ready["artifacts_sha256"][field] = probe.calibration.digest(full / field)
    ready_path.write_text(json.dumps(ready))
    with pytest.raises(AssertionError):
        probe.inputs(*dirs, probe.calibration.digest(ready_path))


def test_missing_full_source_ready_and_already_scored_source_refuse_before_fitting(
    tmp_path, monkeypatch
):
    source, cache, raw, full = directories(tmp_path)
    called = []
    monkeypatch.setattr(
        probe, "inputs", lambda *args: called.append("must not read data")
    )
    (full / "results.json").write_text("already inspected")
    with pytest.raises(ValueError, match="before new zero-shot"):
        probe.fit(source, cache, raw, full, tmp_path / "out", "unused")
    assert not called
    with pytest.raises(FileNotFoundError):
        probe.fullpose.frozen_contract(full, source, cache, raw, "absent")
