"""Synthetic tests qualify the inverse math, not real Wi-Fi efficacy."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import solve

from scripts import domain_conditioned_dynamics_probe as p


def test_conditional_design_and_prediction_are_exactly_odd_and_static_zero():
    rng = np.random.default_rng(4)
    current, previous = rng.normal(size=(100, 64)), rng.normal(size=(100, 64))
    delta, context = p.pair_values(current, previous)
    state = p.fit_map(delta, context)
    swapped_delta, swapped_context = p.pair_values(previous, current)
    np.testing.assert_array_equal(swapped_context, context)
    for rank in p.RANKS:
        design = p.transform(delta, context, state, rank)
        swapped = p.transform(swapped_delta, swapped_context, state, rank)
        np.testing.assert_array_equal(swapped, -design)
        weight = rng.normal(size=(design.shape[1], 34))
        np.testing.assert_array_equal(p.mm(swapped, weight), -p.mm(design, weight))
        zero = p.transform(np.zeros_like(delta), context, state, rank)
        np.testing.assert_array_equal(zero, np.zeros((len(delta), 64 * (1 + rank))))
        np.testing.assert_array_equal(p.mm(zero, weight), np.zeros((len(delta), 34)))


def test_pairs_require_exact_past_endpoint_and_never_bridge_missing_gaps_or_records():
    record = np.array(["a", "a", "a", "a", "b", "b"])
    ends = np.array([39, 47, 63, 71, 39, 47])
    current, previous = p.pair_indices(record, ends)
    np.testing.assert_array_equal(current, [1, 3, 5])
    np.testing.assert_array_equal(previous, [0, 2, 4])
    np.testing.assert_array_equal(ends[current] - ends[previous], np.full(3, 8))
    np.testing.assert_array_equal(record[current], record[previous])
    with pytest.raises(ValueError, match="No exact"):
        p.pair_indices(np.array(["a", "b"]), np.array([39, 47]))
    with pytest.raises(ValueError, match="Duplicate"):
        p.pair_indices(np.array(["a", "a"]), np.array([39, 39]))
    with pytest.raises(ValueError, match="integer"):
        p.pair_indices(record, ends.astype(float))


def fixture():
    rng = np.random.default_rng(13)
    feature = rng.normal(size=(15, 512))
    target = rng.normal(size=(15, 17, 2))
    metadata = dict(
        room=np.repeat(["E01", "E03", "E01", "E03", "E02"], 3),
        person=np.repeat(["S01", "S21", "S04", "S24", "S11"], 3),
        record=np.repeat(
            ["E01/S01/A01", "E03/S21/A01", "E01/S04/A01", "E03/S24/A01", "E02/S11/A01"],
            3,
        ),
        end=np.tile([39, 47, 55], 5),
        action=np.repeat("A01", 15),
    )
    return (
        feature,
        target,
        metadata,
        np.arange(15) < 6,
        (np.arange(15) >= 6) & (np.arange(15) < 12),
    )


def test_training_map_and_all_coefficients_ignore_validation_and_e02_poison():
    feature, target, metadata, train, val = fixture()
    before = p.fit_source_models(feature, target, metadata, train, val)
    feature[val] += np.arange(val.sum())[:, None] * 1e3
    target[val] += np.arange(val.sum())[:, None, None] * 1e3
    feature[~(train | val)], target[~(train | val)] = np.nan, np.nan
    after = p.fit_source_models(feature, target, metadata, train, val)
    for key in before["mapping"]:
        np.testing.assert_array_equal(before["mapping"][key], after["mapping"][key])
    assert set(before["models"]) == set(after["models"])
    for key in before["models"]:
        np.testing.assert_array_equal(
            before["models"][key]["weight"], after["models"][key]["weight"]
        )


def test_actual_source_pair_reader_does_not_decode_e02_values_and_refuses_wrong_roles():
    feature, target, metadata, train, val = fixture()

    class Tracked:
        def __init__(self, values):
            self.values = values

        def __getitem__(self, key):
            ids = key[0] if isinstance(key, tuple) else key
            assert np.isin(metadata["room"][ids], ["E01", "E03"]).all()
            return self.values[key]

    result = p.fit_source_models(
        Tracked(feature), Tracked(target), metadata, train, val
    )
    assert len(result["train_pairs"]["target"]) == 4
    assert len(result["validation_pairs"]["target"]) == 4
    wrong = train.copy()
    wrong[-1] = True
    with pytest.raises(ValueError, match="Prohibited"):
        p.extract_role_pairs(
            Tracked(feature), Tracked(target), metadata, wrong, "train"
        )
    with pytest.raises(ValueError, match="Prohibited"):
        p.extract_role_pairs(Tracked(feature), Tracked(target), metadata, val, "train")


def test_ridge_weights_match_independent_lu_and_zero_reference_score():
    rng = np.random.default_rng(11)
    x, y = rng.normal(size=(150, 64)), rng.normal(size=(150, 34))
    rows, models = p.fit_ridge(x, y)
    gram = np.einsum("ni,nj->ij", x, x)
    right = np.einsum("ni,nj->ij", x, y)
    for row in rows:
        assert row["status"] == "available"
        alpha = row["alpha"]
        direct = solve(gram + len(x) * alpha * np.eye(64), right, assume_a="pos")
        np.testing.assert_allclose(models[alpha], direct, atol=1e-12, rtol=1e-10)
        residual = np.linalg.norm(
            np.einsum("ij,jk->ik", gram + len(x) * alpha * np.eye(64), models[alpha])
            - right
        ) / np.linalg.norm(right)
        assert residual < 1e-12
    assert p.delta_score(np.zeros_like(y), y)["delta_r2"] == pytest.approx(0)


def test_source_pair_metadata_cannot_join_different_people_or_mislabeled_rooms():
    feature, target, metadata, train, _ = fixture()
    bad = copy.deepcopy(metadata)
    bad["person"][0] = "S21"
    with pytest.raises(ValueError, match="metadata disagree"):
        p.extract_role_pairs(feature, target, bad, train, "train")
    bad = copy.deepcopy(metadata)
    bad["record"][0] = "E02/S11/A01"
    with pytest.raises(ValueError, match="metadata disagree"):
        p.extract_role_pairs(feature, target, bad, train, "train")
    bad = copy.deepcopy(metadata)
    bad["end"] = bad["end"].astype(float)
    with pytest.raises(ValueError, match="canonical integer"):
        p.extract_role_pairs(feature, target, bad, train, "train")


def rotated_pairs(n, seed):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi, np.pi, n)
    desired = rng.normal(size=(n, 2))
    c, s = np.cos(theta), np.sin(theta)
    observed = np.column_stack(
        [c * desired[:, 0] - s * desired[:, 1], s * desired[:, 0] + c * desired[:, 1]]
    )
    current, previous = np.zeros((n, 64)), np.zeros((n, 64))
    current[:, :2] = previous[:, :2] = np.column_stack([c, s])
    current[:, 2:4], previous[:, 2:4] = observed / 2, -observed / 2
    target = np.zeros((n, 34))
    target[:, :2] = desired
    delta, context = p.pair_values(current, previous)
    return delta, context, target


def test_conditional_inverse_recovers_synthetic_style_rotation_unconditional_cannot():
    delta, context, target = rotated_pairs(500, 43)
    vd, vc, vy = rotated_pairs(250, 92)
    mapping = p.fit_map(delta, context)
    outcomes = {}
    for mode, rank in [
        ("unconditional", 0),
        ("conditional", 2),
        ("shuffled_context", 2),
    ]:
        tc = (
            context
            if mode != "shuffled_context"
            else context[p.source_permutation(len(context), "train")]
        )
        qc = (
            vc
            if mode != "shuffled_context"
            else vc[p.source_permutation(len(vc), "validation")]
        )
        x, q = p.transform(delta, tc, mapping, rank), p.transform(vd, qc, mapping, rank)
        _, models = p.fit_ridge(x, target)
        outcomes[mode] = max(
            p.delta_score(p.mm(q, weights), vy)["delta_r2"]
            for weights in models.values()
        )
    assert outcomes["conditional"] > 0.99
    assert outcomes["unconditional"] < 0.1
    assert outcomes["shuffled_context"] < 0.1


def score_fixture(value):
    return dict(
        pooled=dict(delta_r2=value),
        equal_room_delta_r2=value,
        per_room={name: dict(delta_r2=value) for name in ("E01", "E03")},
        per_person={
            name: dict(delta_r2=value) for name in ("S04", "S09", "S24", "S29")
        },
    )


def test_source_selection_ties_prefer_rank_zero_then_stronger_alpha_and_checks_shuffle():
    reports = {}
    for rank in p.RANKS:
        for mode in ("natural",) if rank == 0 else ("natural", "shuffled"):
            reports[f"rank{rank}-{mode}"] = [
                dict(
                    status="available",
                    rank=rank,
                    context=mode,
                    alpha=alpha,
                    validation=score_fixture(0.2),
                )
                for alpha in p.ALPHAS
            ]
    chosen = p.selected_and_gate(reports)
    assert chosen["choice"]["rank"] == 0 and chosen["choice"]["alpha"] == 10
    assert chosen["passes"]
    for row in reports["rank2-natural"]:
        row["validation"] = score_fixture(0.3)
    for row in reports["rank2-shuffled"]:
        row["validation"] = score_fixture(0.35)
    chosen = p.selected_and_gate(reports)
    assert chosen["choice"]["rank"] == 2
    assert chosen["checks"]["conditional_context_better_than_matched_shuffle"] is False
    assert not chosen["passes"]


def test_bad_numeric_system_is_unavailable_and_never_receives_validation_scores(
    monkeypatch,
):
    monkeypatch.setattr(p, "eigh", lambda x, **kw: (-np.ones(64) * 1e6, np.eye(64)))
    rows, models = p.fit_ridge(np.ones((10, 64)), np.ones((10, 34)))
    assert not models and len(rows) == len(p.ALPHAS)
    assert all(
        row["status"] == "unavailable" and "reason" in row and "validation" not in row
        for row in rows
    )
    selected = p.selected_and_gate(
        {"rank0-natural": [dict(row, rank=0, context="natural") for row in rows]}
    )
    assert selected["choice"] is None and selected["passes"] is False


@pytest.mark.parametrize("bad", [np.nan, np.inf, 1j, 0j])
def test_real_finite_input_contract_refuses_nonfinite_and_complex_data(bad):
    x = np.ones((5, 64))
    if isinstance(bad, complex):
        x = x.astype(complex)
    x[0, 0] = bad
    with pytest.raises(ValueError, match="real finite"):
        p.fit_map(x, np.ones((5, 64)))


def test_context_map_shapes_scales_and_rank_cannot_be_silently_broadcast():
    rng = np.random.default_rng(3)
    delta, context = rng.normal(size=(30, 64)), rng.normal(size=(30, 64))
    mapping = p.fit_map(delta, context)
    for key in ("delta_rms", "context_scale", "context_pc_scale"):
        bad = copy.deepcopy(mapping)
        bad[key] = np.array([1.0])
        with pytest.raises(ValueError, match="compatible"):
            p.transform(delta, context, bad, 2)
    with pytest.raises(ValueError, match="Undeclared"):
        p.transform(delta, context, mapping, 3)


def source_fixture(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    manifest = {name: p.digest(p.ROOT / name) for name in p.REQUIRED_TRAINING_SOURCES}
    for name in manifest:
        destination = source / "executed-source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((p.ROOT / name).read_bytes())
    p.save_json(
        source / "registration.json",
        dict(config=dict(steps=4000, held_room="E02"), source_sha256=manifest),
    )
    for name in [
        "roles.npz",
        "targets-and-metadata.npz",
        *(arm + "-embedding.npy" for arm in p.ARMS),
    ]:
        (source / name).write_bytes(b"not decoded by provenance-only validation")
    ready = dict(
        kind="masked-csi-source-only-ready",
        held_metrics_computed=False,
        artifacts_sha256={
            str(path.relative_to(source)): p.digest(path)
            for path in source.rglob("*")
            if path.is_file()
        },
    )
    p.save_json(source / "source-ready.json", ready)
    monkeypatch.setattr(p, "EXPECTED_READY", p.digest(source / "source-ready.json"))
    return source


@pytest.mark.parametrize("case", ["empty", "omitted", "unrelated", "missing_snapshot"])
def test_source_producer_manifest_is_mandatory_before_any_array_decode(
    tmp_path, monkeypatch, case
):
    source = source_fixture(tmp_path, monkeypatch)
    reg_path = source / "registration.json"
    reg = json.loads(reg_path.read_text())
    if case == "empty":
        reg["source_sha256"] = {}
    elif case == "omitted":
        del reg["source_sha256"]["ghostmesh/masked_csi.py"]
    elif case == "unrelated":
        reg["source_sha256"] = {
            "scripts/domain_conditioned_dynamics_probe.py": p.digest(Path(p.__file__))
        }
    reg_path.write_text(json.dumps(reg))
    ready_path = source / "source-ready.json"
    ready = json.loads(ready_path.read_text())
    ready["artifacts_sha256"] = {
        str(path.relative_to(source)): p.digest(path)
        for path in source.rglob("*")
        if path.is_file() and path != ready_path
    }
    if case == "missing_snapshot":
        del ready["artifacts_sha256"]["executed-source/ghostmesh/masked_csi.py"]
    ready_path.write_text(json.dumps(ready))
    monkeypatch.setattr(p, "EXPECTED_READY", p.digest(ready_path))
    monkeypatch.setattr(
        np,
        "load",
        lambda *a, **k: pytest.fail(
            "array decoded before required producer validation"
        ),
    )
    with pytest.raises(ValueError, match="provenance|snapshots"):
        p.source_contract(source)


def test_legitimate_producer_manifest_and_executed_sources_validate(
    tmp_path, monkeypatch
):
    source = source_fixture(tmp_path, monkeypatch)
    assert p.source_contract(source)["config"]["steps"] == 4000


def test_existing_output_refuses_before_data_access(tmp_path):
    with pytest.raises(FileExistsError):
        p.run(tmp_path / "absent-study", tmp_path)
