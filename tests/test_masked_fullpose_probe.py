import copy
import json

import numpy as np
import pytest

from scripts import domain_masked_fullpose_probe as probe


def fixture():
    rng = np.random.default_rng(14)
    n = 24
    train, validation, held = (
        np.arange(n) < 12,
        (np.arange(n) >= 12) & (np.arange(n) < 18),
        np.arange(n) >= 18,
    )
    z = rng.normal(size=(n, 3))
    body = rng.normal(scale=0.1, size=(n, 17, 2))
    body -= body[:, [11, 12]].mean(1, keepdims=True)
    root = np.array([0.5, 0.6]) + z[:, :2] * 0.03
    target = body + root[:, None]
    data = dict(
        record=np.repeat(["a", "b", "c", "d", "e", "f"], [6, 6, 3, 3, 3, 3]),
        person=np.repeat(["p1", "p2", "p3", "p4", "p5", "p6"], [6, 6, 3, 3, 3, 3]),
        room=np.repeat(["E01", "E03", "E01", "E03", "E02"], [6, 6, 3, 3, 6]),
        end=np.tile([7, 15, 23], 8),
        action=np.tile(["A01", "A02", "A03"], 8),
        pose=body,
    )
    return z, body, target, data, train, validation, held


def test_actual_source_fit_ignores_poisoned_query_labels_and_query_feature_statistics():
    z, body, target, data, train, validation, held = fixture()
    before = probe.fit_arrays(
        {"masked_temporal": z},
        {"masked_temporal": body},
        target,
        data,
        train,
        validation,
        held,
    )
    altered_target, altered_z = target.copy(), z.copy()
    altered_target[held], altered_z[held] = np.nan, 1e6
    after = probe.fit_arrays(
        {"masked_temporal": altered_z},
        {"masked_temporal": body},
        altered_target,
        data,
        train,
        validation,
        held,
    )
    assert before[1] == after[1]
    for key in before[0]["masked_temporal"]:
        np.testing.assert_array_equal(
            before[0]["masked_temporal"][key], after[0]["masked_temporal"][key]
        )
    for name in ("source_mean", "source_endpoint_clock"):
        np.testing.assert_array_equal(before[2][name], after[2][name])


def test_root_ridge_candidates_match_independent_direct_solutions():
    z, body, target, data, train, validation, _ = fixture()
    state, selection = probe.fit_root(z, body, target, train, validation, data["room"])
    x = z[train] - z[train].mean(0)
    root = target[train, 11:13].mean(1)
    centered = root - root.mean(0)
    for i, alpha in enumerate(probe.ALPHAS):
        expected = np.linalg.solve(x.T @ x + len(x) * alpha * np.eye(3), x.T @ centered)
        np.testing.assert_allclose(
            state["weights"][i], expected, atol=1e-15, rtol=1e-12
        )
    assert len(selection["grid"]) == 25
    assert all(
        row["normal_equation_relative_residual"] < 1e-12 for row in selection["grid"]
    )


def test_static_root_ties_choose_no_motion_and_stronger_regularization():
    z, body, _, data, train, validation, _ = fixture()
    target = body + np.array([0.5, 0.5])[None, None]
    state, selection = probe.fit_root(z, body, target, train, validation, data["room"])
    assert selection["choice"]["beta"] == 0
    assert selection["choice"]["alpha"] == 10
    np.testing.assert_array_equal(
        probe.predict_root(z * 100, state), np.full((len(z), 2), 0.5)
    )


def test_fitting_refuses_people_overlap_and_noncentered_frozen_bodies():
    z, body, target, data, train, validation, held = fixture()
    bad = copy.deepcopy(data)
    bad["person"][held] = data["person"][0]
    with pytest.raises(ValueError, match="overlap"):
        probe.fit_arrays({"x": z}, {"x": body}, target, bad, train, validation, held)
    with pytest.raises(ValueError, match="hip midpoint"):
        probe.fit_arrays(
            {"x": z}, {"x": body + 0.01}, target, data, train, validation, held
        )
    with pytest.raises(ValueError, match="partition"):
        probe.fit_arrays({"x": z}, {"x": body}, target, data, train, train, held)


def test_raw_full_target_join_uses_exact_record_endpoint_and_hip_coordinates():
    rng = np.random.default_rng(3)
    raw = rng.uniform(0.1, 0.9, (30, 17, 2)).astype(np.float32)
    valid = np.ones((30, 17), bool)
    records = [
        dict(id="E01/S01/A01", offset=0, frames=15),
        dict(id="E03/S21/A01", offset=15, frames=15),
    ]
    metadata = dict(
        record=np.array([records[0]["id"], records[1]["id"]]), end=np.array([7, 14])
    )
    expected = raw[[7, 29]].astype(float)
    metadata["pose"] = expected - expected[:, [11, 12]].mean(1, keepdims=True)
    joined, rows = probe.join_full_targets(raw, valid, records, metadata)
    np.testing.assert_array_equal(joined, expected)
    np.testing.assert_array_equal(rows, [7, 29])
    bad = copy.deepcopy(metadata)
    bad["end"][0] = 15
    with pytest.raises(ValueError, match="outside"):
        probe.join_full_targets(raw, valid, records, bad)
    valid[7, 0] = False
    with pytest.raises(ValueError, match="invalid teacher"):
        probe.join_full_targets(raw, valid, records, metadata)
    valid[:] = True
    metadata["pose"][0, 0] += 0.1
    with pytest.raises(ValueError, match="disagrees"):
        probe.join_full_targets(raw, valid, records, metadata)


def test_frozen_rff_transform_matches_formula_without_fitting_statistics():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(7, 3))
    state = dict(
        feature_mean=np.array([1, 2, 3]),
        feature_scale=np.array([2, 3, 4]),
        weight=rng.normal(size=(3, 5)),
        offset=np.arange(5),
        rff_mean=np.arange(5) / 10,
        rff_scale=np.ones(5) * 2,
    )
    expected = (
        np.sqrt(2 / 5)
        * np.cos(
            ((x - state["feature_mean"]) / state["feature_scale"]) @ state["weight"]
            + state["offset"]
        )
        - state["rff_mean"]
    ) / state["rff_scale"]
    np.testing.assert_allclose(probe.apply_rff(x, state), expected, atol=1e-15)
    changed = x.copy()
    changed[-1] = 1e6
    np.testing.assert_array_equal(
        probe.apply_rff(x, state)[:-1], probe.apply_rff(changed, state)[:-1]
    )
    state["rff_scale"][0] = 0
    with pytest.raises(ValueError, match="positive"):
        probe.apply_rff(x, state)


def test_full_metric_separates_static_root_from_body_motion():
    _, body, _, data, _, _, _ = fixture()
    target = body + np.array([0.5, 0.5])[None, None]
    pred = body + np.array([0.6, 0.7])[None, None]
    result = probe.metric(pred, target, data["record"])
    assert result["normalized_joint_error"] > 0
    assert result["record_centered_body_r2"] == pytest.approx(1)
    assert result["record_centered_root_r2"] is None


def fake_source(tmp_path):
    source, cache, raw = (tmp_path / name for name in ("source", "cache", "raw"))
    for path in (source, cache, raw):
        path.mkdir()
    for folder in (cache, raw):
        (folder / "receipt.json").write_text("{}")
    return source, cache, raw


def full_ready(tmp_path):
    z, body, target, data, train, validation, held = fixture()
    out = tmp_path / "out"
    out.mkdir()
    source, cache, raw = fake_source(tmp_path)
    registration = dict(
        config=probe.CONFIG, source_ready_sha256="source-hash", source_sha256={}
    )
    probe.save_json(out / "registration.json", registration)
    probe.save_npz(out / "roles.npz", train=train, validation=validation, held=held)
    probe.save_npz(
        out / "canonical-metadata.npz", **{k: v for k, v in data.items() if k != "pose"}
    )
    probe.save_array(out / "full-targets.npy", target)
    probe.save_array(out / "raw-frame-indices.npy", np.arange(len(z)))
    probe.save_json(out / "frozen-fullpose-selections.json", {})
    for arm in probe.ARMS:
        probe.save_npz(out / (arm + "-root-state.npz"), x=np.ones(1))
    for arm in (*probe.ARMS, "source_mean", "source_endpoint_clock"):
        probe.save_array(out / (arm + "-full-prediction.npy"), target)
    ready = dict(
        kind="masked-csi-fullimage-source-ready",
        held_metrics_computed=False,
        source_ready_sha256="source-hash",
        raw_receipt_sha256=probe.digest(raw / "receipt.json"),
        cache_receipt_sha256=probe.digest(cache / "receipt.json"),
        artifacts_sha256={p.name: probe.digest(p) for p in out.iterdir()},
    )
    probe.save_json(out / "fullpose-source-ready.json", ready)
    return out, source, cache, raw


def test_actual_frozen_scoring_never_refits_and_refuses_repeated_outputs(
    tmp_path, monkeypatch
):
    out, source, cache, raw = full_ready(tmp_path)
    monkeypatch.setattr(probe, "source_contract", lambda *a: {})
    monkeypatch.setattr(probe, "fit_arrays", lambda *a: pytest.fail("refit called"))
    monkeypatch.setattr(probe, "fit_root", lambda *a: pytest.fail("root refit called"))
    probe.score_frozen(
        out, source, cache, raw, probe.digest(out / "fullpose-source-ready.json")
    )
    result = json.loads((out / "results.json").read_text())
    assert len(result["methods"]) == 8
    assert (
        result["methods"]["masked_temporal"]["aligned"]["normalized_joint_error"] == 0
    )
    with pytest.raises(FileExistsError):
        probe.score_frozen(
            out, source, cache, raw, probe.digest(out / "fullpose-source-ready.json")
        )


def test_frozen_scoring_refuses_changed_model_artifacts_before_metric_output(
    tmp_path, monkeypatch
):
    out, source, cache, raw = full_ready(tmp_path)
    monkeypatch.setattr(probe, "source_contract", lambda *a: {})
    expected = probe.digest(out / "fullpose-source-ready.json")
    np.save(out / "masked_temporal-full-prediction.npy", np.zeros((24, 17, 2)))
    with pytest.raises(ValueError):
        probe.score_frozen(out, source, cache, raw, expected)
    assert not (out / "query-permutation.npy").exists()


def test_source_contract_rejects_omitted_required_artifacts(tmp_path):
    source, cache, raw = fake_source(tmp_path)
    probe.save_json(
        source / "source-ready.json",
        dict(
            kind="masked-csi-source-only-ready",
            held_metrics_computed=False,
            artifacts_sha256={},
        ),
    )
    with pytest.raises(ValueError, match="complete"):
        probe.source_contract(
            source, cache, raw, probe.digest(source / "source-ready.json")
        )


def test_exclusive_outputs_preserve_existing_bytes(tmp_path):
    path = tmp_path / "owned.json"
    path.write_text("owned")
    with pytest.raises(FileExistsError):
        probe.save_json(path, {})
    assert path.read_text() == "owned"


@pytest.mark.parametrize("field", ["design", "body", "target"])
@pytest.mark.parametrize("imaginary", [0.0, 1e6])
def test_actual_fit_rejects_complex_schema_before_discarding_imaginary_values(
    field, imaginary
):
    z, body, target, data, train, validation, held = fixture()
    if field == "design":
        z = z.astype(np.complex128) + 1j * imaginary
    elif field == "body":
        body = body.astype(np.complex128) + 1j * imaginary
    else:
        target = target.astype(np.complex128) + 1j * imaginary
    with pytest.raises(ValueError, match="real-valued"):
        probe.fit_arrays({"x": z}, {"x": body}, target, data, train, validation, held)


@pytest.mark.parametrize("field", ["feature_mean", "weight", "input"])
@pytest.mark.parametrize("imaginary", [0.0, 1e6])
def test_frozen_mapping_rejects_complex_fields_before_real_conversion(field, imaginary):
    x = np.ones((4, 3))
    state = dict(
        feature_mean=np.zeros(3),
        feature_scale=np.ones(3),
        weight=np.eye(3),
        offset=np.zeros(3),
        rff_mean=np.zeros(3),
        rff_scale=np.ones(3),
    )
    if field == "input":
        x = x.astype(np.complex128) + 1j * imaginary
    else:
        state[field] = state[field].astype(np.complex128) + 1j * imaginary
    with pytest.raises(ValueError, match="real-valued"):
        probe.apply_rff(x, state)


@pytest.mark.parametrize(
    "value",
    [np.array(["1", "2"]), np.array([1, 2], dtype=object), np.array([True, False])],
)
def test_real_numeric_contract_rejects_non_numeric_and_boolean_values(value):
    with pytest.raises(ValueError, match="real-valued"):
        probe.finite(value)


def test_root_target_and_frozen_selection_require_real_numerical_schema():
    z, body, target, data, train, validation, _ = fixture()
    with pytest.raises(ValueError, match="real-valued"):
        probe.fit_root(
            z, body, target.astype(np.complex128), train, validation, data["room"]
        )
    state, _ = probe.fit_root(z, body, target, train, validation, data["room"])
    for field in ("selected", "beta"):
        altered = copy.deepcopy(state)
        altered[field] = altered[field].astype(np.complex128)
        with pytest.raises(ValueError, match="real-valued"):
            probe.predict_root(z, altered)


def test_raw_pose_join_rejects_complex_coordinates_and_non_boolean_validity():
    raw = np.zeros((2, 17, 2))
    records = [dict(id="E01/S01/A01", offset=0, frames=2)]
    data = dict(
        record=np.array(["E01/S01/A01"]), end=np.array([0]), pose=np.zeros((1, 17, 2))
    )
    with pytest.raises(ValueError, match="real-valued"):
        probe.join_full_targets(
            raw.astype(np.complex128), np.ones((2, 17), bool), records, data
        )
    with pytest.raises(ValueError, match="Boolean"):
        probe.join_full_targets(raw, np.ones((2, 17), np.complex128), records, data)
