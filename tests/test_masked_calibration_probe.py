"""Software fixtures for reference-held selection; never pose efficacy evidence."""

import copy
import json
import shutil

import numpy as np
import pytest

from ghostmesh.reference_kernel import ReferenceRBFKernel
from scripts import domain_masked_calibration_probe as probe


def reference_fixture():
    rng = np.random.default_rng(81)
    x = rng.normal(size=(120, 5))
    people = np.repeat(["S11", "S12"], 60)
    record = np.repeat([f"S{p}/A{a}" for p in (11, 12) for a in (3, 7, 12)], 20)
    skeleton = np.linspace(-0.3, 0.5, 34)
    labels = skeleton + 0.03 * np.sin(x[:, :1]) * np.linspace(0.1, 1, 34)
    base = np.broadcast_to(skeleton, labels.shape).copy()
    return x, labels, base, people, record


def assert_states_equal(a, b):
    assert set(a) == set(b)
    for name in a:
        if isinstance(a[name], dict):
            assert_states_equal(a[name], b[name])
        else:
            np.testing.assert_array_equal(a[name], b[name])


def test_cv_normalizers_and_bandwidth_use_each_training_person_only():
    x, labels, base, people, record = reference_fixture()
    x[60:] += 20
    _, report = probe.fit_reference_model(x, labels, base, people, record)
    for (train, _), row in zip(probe.partitions(people), report["folds"]):
        kernel = ReferenceRBFKernel.from_reference(x[train])
        np.testing.assert_array_equal(row["input_mean"], x[train].mean(0))
        np.testing.assert_array_equal(row["input_scale"], kernel.input_scale)
        assert row["bandwidth_squared"] == kernel.bandwidth_squared
        assert set(row["fitting_people"]).isdisjoint(row["validation_people"])
    assert len(report["grid"]) == 25


def test_cv_grid_is_equal_person_mean_and_ties_choose_zero_beta_strongest_ridge():
    x, labels, base, people, record = reference_fixture()
    state, report = probe.fit_reference_model(x, labels, labels, people, record)
    assert float(state["beta"]) == 0 and float(state["alpha"]) == 10
    assert str(state["correction_kind"]) == "zero"
    for row in report["grid"]:
        assert row["reference_person_cv_nje"] == np.mean(row["per_person_nje"])
    np.testing.assert_array_equal(
        probe.predict_reference_model(state, x[:3], base[:3]), base[:3]
    )


def test_wrapper_matches_direct_kernel_solve_and_has_far_reference_fallback():
    x, labels, base, _, _ = reference_fixture()
    kernel = ReferenceRBFKernel.from_reference(x)
    alpha, beta = 0.1, 0.5
    design = kernel.reference_design
    distance = np.square(design[:, None] - design[None]).sum(-1)
    matrix = np.exp(-distance / (2 * kernel.bandwidth_squared)) + len(
        x
    ) * alpha * np.eye(len(x))
    dual = np.linalg.solve(matrix, labels - base)
    state = dict(
        kernel.state_dict(),
        method=np.asarray("residual_krr"),
        base_kind=np.asarray("frozen_arm"),
        correction_kind=np.asarray("kernel"),
        beta=np.asarray(beta),
        dual=dual,
    )
    actual = probe.predict_reference_model(state, x[:3], base[:3])
    expected = base[:3] + beta * (
        np.exp(-distance[:3] / (2 * kernel.bandwidth_squared))[:, :, None] * dual[None]
    ).sum(1)
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)
    np.testing.assert_array_equal(
        probe.predict_reference_model(state, np.full((3, 5), 1e8), base[:3]), base[:3]
    )


def test_undefined_reference_features_do_not_invent_bandwidth_or_adaptation():
    _, labels, base, people, record = reference_fixture()
    state, report = probe.fit_reference_model(
        np.ones((120, 5)), labels, base, people, record
    )
    assert float(state["beta"]) == 0
    assert str(state["correction_kind"]) == "zero"
    assert "bandwidth_squared" not in state
    assert all(row["available"] == (row["beta"] == 0) for row in report["grid"])
    json.dumps(report, allow_nan=False)


def test_reference_label_permutation_preserves_each_record_and_is_its_own_selector():
    x, labels, base, people, record = reference_fixture()
    perm = probe.permutation(record)
    np.testing.assert_array_equal(record[perm], record)
    np.testing.assert_array_equal(np.sort(perm), np.arange(120))
    assert not np.array_equal(labels, labels[perm])
    a, report_a = probe.fit_reference_model(
        x, labels, base, people, record, method="permuted_residual_krr"
    )
    b, report_b = probe.fit_reference_model(x, labels[perm], base, people, record)
    for key in set(a) - {"method"}:
        np.testing.assert_array_equal(a[key], b[key])
    assert report_a == report_b


def study_fixture():
    x, labels, base, people, records = reference_fixture()
    rng = np.random.default_rng(52)
    query = np.arange(120, 140)
    train = np.arange(140, 150)
    full_x = np.vstack([x, rng.normal(size=(30, 5))])
    poses = np.vstack([labels, base[:30]]).reshape(150, 17, 2)
    persons = np.concatenate(
        [people, np.repeat(["S13", "S14"], 10), np.repeat("S01", 10)]
    )
    record = np.concatenate(
        [records, np.repeat(["S13/A03", "S14/A03"], 10), np.repeat("S01/A03", 10)]
    )
    data = dict(
        pose=poses,
        person=persons,
        record=record,
        action=np.array([name.split("/")[-1] for name in record]),
        end=np.arange(150) % 10,
        room=np.where(persons == "S01", "E01", "E02"),
    )
    reference = np.arange(120)
    train_mask = np.zeros(150, dtype=bool)
    train_mask[train] = True
    features = {name: full_x.copy() for name in probe.ARMS}
    bases = {
        name: np.broadcast_to(base[0], (150, 34)).reshape(150, 17, 2).copy()
        for name in probe.ARMS
    }
    return data, train_mask, reference, query, features, bases


class ReferenceOnly:
    def __init__(self, x, reference):
        self.x, self.reference = x, reference

    def __getitem__(self, ids):
        np.testing.assert_array_equal(ids, self.reference)
        return self.x[ids]


def test_actual_fit_entrypoint_query_poison_cannot_change_choices_or_coefficients(
    tmp_path, monkeypatch
):
    source, cache = tmp_path / "source", tmp_path / "cache"
    source.mkdir()
    cache.mkdir()
    (cache / "receipt.json").write_text("synthetic source-only fixture")
    data, train, reference, query, features, bases = study_fixture()
    outputs = []
    for poison in (False, True):
        d = copy.deepcopy(data)
        f = copy.deepcopy(features)
        if poison:
            d["pose"][query] = np.nan
            for value in f.values():
                value[query] = 1e100
        readers = {name: ReferenceOnly(value, reference) for name, value in f.items()}
        monkeypatch.setattr(
            probe,
            "inputs",
            lambda *args, d=d, readers=readers: (
                {},
                d,
                train,
                reference,
                query,
                readers,
                bases,
            ),
        )
        out = tmp_path / ("poisoned" if poison else "ordinary")
        probe.fit(source, cache, out, "synthetic reviewed source seal")
        outputs.append(out)
        assert (out / "calibration-ready.json").exists()
        assert not (out / "results.json").exists()
        assert not (out / "query-permutation.npy").exists()
    assert (outputs[0] / "frozen-reference-choices.json").read_bytes() == (
        outputs[1] / "frozen-reference-choices.json"
    ).read_bytes()
    for path in outputs[0].glob("*.npz"):
        with (
            np.load(path, allow_pickle=False) as a,
            np.load(outputs[1] / path.name, allow_pickle=False) as b,
        ):
            assert_states_equal({k: a[k] for k in a.files}, {k: b[k] for k in b.files})


def test_actual_fit_refuses_scored_source_before_reading_any_data(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "results.json").write_text("already scored")
    monkeypatch.setattr(
        probe, "inputs", lambda *args: pytest.fail("Data must not be loaded")
    )
    with pytest.raises(ValueError, match="before new zero-shot"):
        probe.fit(source, tmp_path / "cache", tmp_path / "out", "unused")


@pytest.mark.parametrize("field", ["person", "record"])
def test_reference_query_identity_overlap_rejected(field):
    data, _, reference, query, _, _ = study_fixture()
    data[field][query[0]] = data[field][reference[0]]
    with pytest.raises(ValueError, match="overlap"):
        probe.validate_roles(data, reference, query)


def test_actual_score_only_uses_frozen_states_and_refuses_a_changed_state(
    tmp_path, monkeypatch
):
    source, cache = tmp_path / "source", tmp_path / "cache"
    source.mkdir()
    cache.mkdir()
    (cache / "receipt.json").write_text("synthetic")
    values = study_fixture()
    data, train, reference, query, features, bases = values
    monkeypatch.setattr(probe, "inputs", lambda *args: ({}, *values))
    out = tmp_path / "study"
    probe.fit(source, cache, out, "synthetic source seal")
    monkeypatch.setattr(
        probe,
        "fit_reference_model",
        lambda *args, **kwargs: pytest.fail("Scoring must never refit"),
    )
    probe.score(source, cache, out)
    result = json.loads((out / "results.json").read_text())
    assert len(result["methods"]) == 39 and len(result["reversals"]["person"]) == 2
    assert np.load(out / "query-permutation.npy").shape == (20,)
    with pytest.raises(FileExistsError):
        probe.score(source, cache, out)
    changed = tmp_path / "changed"
    shutil.copytree(out, changed)
    (changed / "results.json").unlink()
    (changed / "query-permutation.npy").unlink()
    state = changed / "masked_temporal--residual_krr.npz"
    state.write_bytes(b"changed model bytes")
    with pytest.raises(ValueError):
        probe.score(source, cache, changed)


def test_source_only_seal_must_validate_before_dataset_labels_are_loaded(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        probe, "source_contract", lambda *args: calls.append("source-seal")
    )

    def forbidden_loader(*args):
        calls.append("dataset")
        raise RuntimeError("Synthetic stop after checking order")

    monkeypatch.setattr(probe, "load_inputs", forbidden_loader)
    with pytest.raises(RuntimeError, match="Synthetic stop"):
        probe.inputs(tmp_path, tmp_path, "reviewed hash")
    assert calls == ["source-seal", "dataset"]


def test_exclusive_outputs_preserve_preexisting_bytes(tmp_path):
    path = tmp_path / "existing.npz"
    path.write_bytes(b"owned by another writer")
    with pytest.raises(FileExistsError):
        probe.save_npz(path, x=np.ones(3))
    with pytest.raises(FileExistsError):
        probe.save_array(path, np.ones(3))
    assert path.read_bytes() == b"owned by another writer"


def source_seal_fixture(tmp_path):
    source, cache = tmp_path / "source", tmp_path / "cache"
    source.mkdir()
    cache.mkdir()
    (cache / "receipt.json").write_text("synthetic immutable cache receipt")
    (cache / "targets-and-metadata.npz").write_bytes(b"synthetic metadata bytes")
    names = {
        "roles.npz",
        "frozen-source-selections.json",
        "frozen-temporal-control.npz",
        "raw_current-readout.npz",
        "raw_current-rff.npz",
    }
    names.update(
        arm + suffix
        for arm in probe.ENCODERS
        for suffix in ("-embedding.npy", "-final.pt", "-rff.npz", "-readout.npz")
    )
    for name in names:
        (source / name).write_bytes(b"opaque synthetic frozen bytes")
    (source / "targets-and-metadata.npz").write_bytes(
        (cache / "targets-and-metadata.npz").read_bytes()
    )
    probe.save_json(
        source / "registration.json",
        dict(
            config=dict(held_room="E02"),
            cache_receipt_sha256=probe.digest(cache / "receipt.json"),
        ),
    )
    ready = dict(
        kind="masked-csi-source-only-ready",
        held_metrics_computed=False,
        artifacts_sha256={path.name: probe.digest(path) for path in source.iterdir()},
    )
    probe.save_json(source / "source-ready.json", ready)
    return source, cache, ready


def test_source_seal_binds_models_and_embeddings_and_rejects_missing_manifest_entries(
    tmp_path,
):
    source, cache, ready = source_seal_fixture(tmp_path)
    expected = probe.digest(source / "source-ready.json")
    probe.source_contract(source, cache, expected)
    model = source / "masked_temporal-final.pt"
    model.write_bytes(b"changed after source-only seal")
    with pytest.raises(ValueError):
        probe.source_contract(source, cache, expected)
    model.write_bytes(b"opaque synthetic frozen bytes")
    del ready["artifacts_sha256"]["masked_temporal-embedding.npy"]
    (source / "source-ready.json").write_text(json.dumps(ready))
    with pytest.raises(ValueError, match="lacks required"):
        probe.source_contract(source, cache, probe.digest(source / "source-ready.json"))


def test_nonfinite_frozen_correction_is_rejected_instead_of_withholding_predictions():
    state = dict(
        base_kind=np.asarray("frozen_arm"),
        beta=np.asarray(0.5),
        correction_kind=np.asarray("constant"),
        correction=np.full(34, np.nan),
    )
    with pytest.raises(ValueError, match="finite"):
        probe.predict_reference_model(state, np.ones((3, 1)), np.zeros((3, 34)))
    state["correction"] = np.zeros(34)
    state["beta"] = np.asarray(np.nan)
    with pytest.raises(ValueError, match="beta"):
        probe.predict_reference_model(state, np.ones((3, 1)), np.zeros((3, 34)))


@pytest.mark.parametrize("current,width", [(False, 512), (True, 9576)])
def test_actual_fit_rejects_complex_stored_features_before_cast_or_decoder(
    monkeypatch, current, width
):
    data, train, reference, query, features, bases = study_fixture()
    stored = np.tile(features["masked_temporal"][:, :1], (1, width)).astype(complex)
    stored += 1e6j
    readers = {"masked_temporal": probe.Features(stored, current=current)}
    monkeypatch.setattr(
        probe,
        "current_covariances",
        lambda *args: pytest.fail("Invalid complex coordinates reached the decoder"),
    )
    with pytest.raises(ValueError, match="real-valued"):
        probe.fit_arrays(
            readers,
            bases,
            data,
            reference,
            query,
            data["pose"][train].reshape(train.sum(), 34).mean(0),
        )
