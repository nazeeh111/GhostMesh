"""Synthetic post-score qualification checks, never sensing efficacy evidence."""

import ast
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from scripts import verify_masked_results as audit


def motion_fixture():
    rng = np.random.default_rng(79)
    n = 12
    body = rng.normal(0, 0.1, (n, 17, 2))
    body -= body[:, [11, 12]].mean(1, keepdims=True)
    full = body + rng.uniform(0.3, 0.7, (n, 1, 2))
    data = dict(
        record=np.repeat(["one", "two", "three"], 4),
        person=np.repeat(["p1", "p2", "p3"], 4),
        action=np.tile(["A01", "A02"], 6),
    )
    return body, full, data


def test_bbox_metrics_have_correct_scale_pck_and_per_joint_counts():
    body, _, data = motion_fixture()
    prediction = body + 0.02
    actual = audit.metric(prediction, body, data["record"], "body")
    error = (
        np.linalg.norm(prediction - body, axis=2)
        / np.linalg.norm(body.max(1) - body.min(1), axis=1)[:, None]
    )
    assert actual["normalized_joint_error"] == pytest.approx(error.mean())
    for threshold in (0.05, 0.1, 0.2):
        assert actual["pck_curve"][f"{threshold:g}"] == (error <= threshold).mean()
    assert actual["valid_joints"] == 12 * 17 and actual["valid_frames"] == 12
    assert len(actual["per_joint"]) == 17
    for joint, row in enumerate(actual["per_joint"]):
        assert row["joint"] == joint and row["count"] == 12
        assert row["nje"] == pytest.approx(error[:, joint].mean())
    changed = audit.metric(prediction * 7 + 3, body * 7 + 3, data["record"], "body")
    assert changed["normalized_joint_error"] == pytest.approx(
        actual["normalized_joint_error"]
    )
    assert changed["record_centered_articulation_r2"] == pytest.approx(1)


def test_stable_static_motion_is_undefined_instead_of_roundoff_r2_one():
    skeleton = np.linspace(0.1, 0.9, 34).reshape(17, 2)
    truth = np.broadcast_to(skeleton, (80, 17, 2)).copy()
    pred = truth + np.array([0.123, -0.234])
    records = np.repeat(["a", "b"], 40)
    assert audit.motion_r2(pred, truth, records) is None
    result = audit.metric(pred, truth, records, "full")
    assert result["record_centered_full_image_r2"] is None
    assert result["record_centered_body_r2"] is None
    assert result["record_centered_root_r2"] is None
    # Static predictions for genuinely dynamic targets have zero motion R2.
    _, target, data = motion_fixture()
    baseline = np.broadcast_to(target.mean(0), target.shape)
    assert audit.motion_r2(baseline, target, data["record"]) == pytest.approx(
        0, abs=1e-14
    )


def test_full_body_root_fields_are_separate_and_no_query_hip_is_placed():
    body, full, data = motion_fixture()
    prediction = body + np.array([0.8, 0.9])[None, None]
    result = audit.metric(prediction, full, data["record"], "full")
    assert result["record_centered_body_r2"] == pytest.approx(1)
    assert result["record_centered_root_r2"] == pytest.approx(0, abs=1e-14)
    assert result["normalized_joint_error"] > 0
    assert "record_centered_articulation_r2" not in result


@pytest.mark.parametrize("damage", ["nan", "complex", "bool"])
def test_metric_rejects_nonfinite_and_nonreal_fields(damage):
    body, _, data = motion_fixture()
    bad = (
        np.full_like(body, np.nan)
        if damage == "nan"
        else (
            body.astype(complex) if damage == "complex" else np.ones(body.shape, bool)
        )
    )
    with pytest.raises(ValueError):
        audit.metric(bad, body, data["record"], "body")


def test_permutation_is_exact_within_record_and_role_order_matters():
    body, _, data = motion_fixture()
    perm = audit.permutation(data["record"], 2027)
    np.testing.assert_array_equal(data["record"][perm], data["record"])
    np.testing.assert_array_equal(perm, audit.permutation(data["record"], 2027))
    result = audit.scores(body, body, data, np.arange(12), perm, "body")
    assert len(result["per_person"]) == 3 and len(result["per_action"]) == 2
    assert result["aligned"]["normalized_joint_error"] == 0
    with pytest.raises(ValueError, match="within-record"):
        audit.scores(body, body, data, np.arange(12), np.roll(np.arange(12), 1), "body")
    with pytest.raises(ValueError):
        audit.scores(body, body, data, np.arange(12), perm.astype(float), "body")


def kernel_state():
    rng = np.random.default_rng(91)
    return dict(
        method=np.asarray("residual_krr"),
        base_kind=np.asarray("frozen_arm"),
        correction_kind=np.asarray("kernel"),
        beta=np.asarray(0.5),
        alpha=np.asarray(0.1),
        kind=np.asarray("reference_residual_rbf_v1"),
        input_mean=np.zeros(3),
        input_scale=np.ones(3),
        reference_design=rng.normal(size=(4, 3)),
        bandwidth_squared=np.asarray(2.0),
        scale_floor=np.asarray(0.001),
        floored_columns=np.zeros(3, bool),
        dual=rng.normal(0, 0.02, (4, 34)),
        final_equation_residual=np.asarray(0.0),
    )


def test_own_cdist_kernel_reconstructs_frozen_correction_and_far_reference_base():
    rng = np.random.default_rng(92)
    state = kernel_state()
    x, base = rng.normal(size=(7, 3)), rng.normal(size=(7, 34))
    expected = base + 0.5 * np.einsum(
        "ij,jk->ik",
        np.exp(-cdist(x, state["reference_design"], "sqeuclidean") / 4),
        state["dual"],
        optimize=False,
    )
    np.testing.assert_allclose(
        audit.predict_calibrated(state, x, base), expected, atol=1e-15
    )
    np.testing.assert_array_equal(
        audit.predict_calibrated(state, np.full((7, 3), 1e8), base), base
    )
    changed = x.copy()
    changed[-1] += 100
    np.testing.assert_array_equal(
        audit.predict_calibrated(state, changed, base)[:-1], expected[:-1]
    )


def test_zero_constant_absolute_and_mean_mixture_semantics():
    x, base = np.ones((3, 2)), np.zeros((3, 34))
    zero = dict(
        method=np.asarray("residual_krr"),
        base_kind=np.asarray("frozen_arm"),
        correction_kind=np.asarray("zero"),
        beta=np.asarray(0.0),
        alpha=np.asarray(10.0),
    )
    np.testing.assert_array_equal(audit.predict_calibrated(zero, x, base), base)
    constant = dict(
        zero,
        method=np.asarray("constant_residual"),
        correction_kind=np.asarray("constant"),
        beta=np.asarray(0.75),
        alpha=np.asarray(0.0),
        correction=np.arange(34) / 100,
    )
    np.testing.assert_array_equal(
        audit.predict_calibrated(constant, x, base),
        np.broadcast_to(0.75 * constant["correction"], base.shape),
    )
    absolute = dict(
        zero,
        method=np.asarray("absolute_krr"),
        base_kind=np.asarray("source_mean"),
        source_mean=np.arange(34) / 100,
    )
    np.testing.assert_array_equal(
        audit.predict_calibrated(absolute, x, base),
        np.broadcast_to(absolute["source_mean"], base.shape),
    )
    zero["beta"] = np.asarray(0.5)
    with pytest.raises(ValueError, match="Zero correction"):
        audit.predict_calibrated(zero, x, base)


@pytest.mark.parametrize(
    "field",
    [
        "dual",
        "input_mean",
        "input_scale",
        "reference_design",
        "bandwidth_squared",
        "beta",
    ],
)
@pytest.mark.parametrize("damage", ["nan", "complex"])
def test_frozen_kernel_nonfinite_or_complex_rejected(field, damage):
    state = kernel_state()
    state[field] = (
        np.full_like(state[field], np.nan)
        if damage == "nan"
        else state[field].astype(complex) + 1e6j
    )
    with pytest.raises(ValueError):
        audit.predict_calibrated(state, np.ones((3, 3)), np.zeros((3, 34)))


@pytest.mark.parametrize("damage", ["offset", "nan", "complex"])
def test_saved_query_pose_tampering_rejects(tmp_path, damage):
    expected = np.ones((5, 17, 2))
    name = "masked_temporal/residual_krr"
    path = tmp_path / "masked_temporal--residual_krr-query-prediction.npy"
    np.save(path, expected)
    assert audit.query_prediction_readback(tmp_path, {name: expected})[name] == 0
    changed = (
        expected + 0.01
        if damage == "offset"
        else (
            np.full_like(expected, np.nan)
            if damage == "nan"
            else expected.astype(complex)
        )
    )
    np.save(path, changed)
    with pytest.raises(ValueError):
        audit.query_prediction_readback(tmp_path, {name: expected})


def test_source_rff_root_replay_uses_frozen_coefficients_and_never_query_hips(
    tmp_path, monkeypatch
):
    source, full_source = tmp_path / "source", tmp_path / "full"
    source.mkdir()
    full_source.mkdir()
    rng = np.random.default_rng(19)
    n, ids = 15, np.arange(3, 15)
    train = np.arange(n) < 3
    roles = dict(train=train, validation=np.zeros(n, bool), held=~train)
    full_target = rng.uniform(0.1, 0.9, (n, 17, 2))
    body_target = full_target - full_target[:, [11, 12]].mean(1, keepdims=True)
    data = dict(pose=body_target, end=np.arange(n) % 3)
    features = {
        arm: rng.normal(size=(n, 3))
        for arm in (*audit.ENCODERS, "raw_current", "frozen_temporal")
    }
    expected_full = {}
    for arm in (*audit.ENCODERS, "raw_current"):
        mapping = dict(
            feature_mean=np.zeros(3),
            feature_scale=np.ones(3),
            weight=rng.normal(0, 0.1, (3, 512)),
            offset=rng.uniform(0, 6, 512),
            rff_mean=np.zeros(512),
            rff_scale=np.ones(512),
        )
        z = np.sqrt(2 / 512) * np.cos(
            np.einsum("ij,jk->ik", features[arm], mapping["weight"], optimize=False)
            + mapping["offset"]
        )
        weights = rng.normal(0, 0.001, (5, 512, 17, 2))
        weights -= weights[:, :, [11, 12]].mean(2, keepdims=True)
        weights = weights.reshape(5, 512, 34)
        state = dict(
            x_mean=np.zeros(512),
            y_mean=np.zeros(34),
            weights=weights,
            selected=np.asarray(2),
        )
        body = np.einsum("ij,jk->ik", z, weights[2], optimize=False).reshape(n, 17, 2)
        state["prediction"] = body
        np.savez(source / (arm + "-rff.npz"), **mapping)
        np.savez(source / (arm + "-readout.npz"), **state)
        rs = dict(
            x_mean=np.zeros(512),
            root_mean=np.array([0.3, 0.5]),
            weights=rng.normal(0, 0.001, (5, 512, 2)),
            selected=np.asarray(2),
            beta=np.asarray(0.5),
        )
        np.savez(full_source / (arm + "-root-state.npz"), **rs)
        root = rs["root_mean"] + 0.5 * np.einsum(
            "ij,jk->ik", z, rs["weights"][2], optimize=False
        )
        expected_full[arm] = body + root[:, None]
        np.save(full_source / (arm + "-full-prediction.npy"), expected_full[arm])
    np.savez(
        source / "frozen-temporal-control.npz", held_indices=ids, pose=body_target[ids]
    )
    np.save(
        full_source / "source_mean-full-prediction.npy",
        np.broadcast_to(full_target[train].mean(0), full_target.shape),
    )
    clock = np.stack(
        [full_target[train & (data["end"] == end)].mean(0) for end in data["end"]]
    )
    np.save(full_source / "source_endpoint_clock-full-prediction.npy", clock)
    monkeypatch.setattr(
        audit,
        "frozen_features",
        lambda source, cache, arm, selected, count: features[arm][selected],
    )
    actual_ids, body, full, _, errors = audit.source_predictions(
        source, full_source, tmp_path, data, full_target, roles
    )
    np.testing.assert_array_equal(actual_ids, ids)
    assert max(errors.values()) < 1e-14 and len(body) == len(full) == 8
    for arm in expected_full:
        np.testing.assert_allclose(full[arm], expected_full[arm][ids], atol=1e-15)
        assert not np.allclose(
            full[arm][:, [11, 12]].mean(1), full_target[ids][:, [11, 12]].mean(1)
        )
    poisoned = full_target.copy()
    poisoned[ids] = np.nan
    poisoned_data = dict(data, pose=body_target.copy())
    poisoned_data["pose"][ids] = np.nan
    again = audit.source_predictions(
        source, full_source, tmp_path, poisoned_data, poisoned, roles
    )
    for arm in full:
        np.testing.assert_array_equal(full[arm], again[2][arm])
    state_path = source / "random_frozen-readout.npz"
    state = audit.contracts.npz(state_path)
    state["prediction"][ids[0], 0, 0] += 0.01
    np.savez(state_path, **state)
    with pytest.raises(ValueError, match="source readout"):
        audit.source_predictions(
            source, full_source, tmp_path, data, full_target, roles
        )


def test_report_disagreements_are_preserved_and_none_is_not_a_duplicate_success():
    assert audit.differences(None, None) == []
    finding = audit.differences(1.0, None, "static-root")
    assert finding[0]["independent"] is None and finding[0]["stored"] == 1
    assert audit.differences(-2e-16, 0.0) == []
    assert (
        audit.differences(0.1, 0.3)[0]["reason"]
        == "Independent numerical metric differs"
    )
    assert audit.differences(True, False)[0]["independent"] is False
    for bad in (float("nan"), complex(1, 2)):
        with pytest.raises(ValueError):
            audit.differences(bad, 0.0)
    with pytest.raises(ValueError):
        audit.differences(10.0, 10)


def test_unchanged_formal_gates_require_all_controls_and_positive_defined_motion():
    body, full, data = motion_fixture()
    for protocol in audit.PROTOCOLS:
        mode = "body" if protocol.endswith("body") else "full"
        target = body if mode == "body" else full
        score = audit.scores(
            target,
            target,
            data,
            np.arange(12),
            audit.permutation(data["record"], 2027),
            mode,
        )
        methods = {
            name: copy.deepcopy(score) for name in audit.expected_methods(protocol)
        }
        decisions = audit.gate(protocol, methods, 0.0)
        assert not all(decisions.values())
        primary = "masked_temporal" if protocol.startswith("zero_") else audit.PRIMARY
        field = (
            "record_centered_articulation_r2"
            if mode == "body"
            else "record_centered_body_r2"
        )
        methods[primary]["aligned"][field] = None
        decisions = audit.gate(protocol, methods, 1.0)
        assert not all(decisions.values())
    assert sum(len(audit.expected_methods(p)) for p in audit.PROTOCOLS) == 89
    assert len(audit.expected_methods("calibrated_body")) == 39
    assert len(audit.expected_methods("calibrated_full")) == 34


def freeze_fixture(tmp_path):
    values = {}
    names = {"work/" + d + "/" + f for d, f, _, _ in audit.PROTOCOLS.values()} | set(
        audit.AUDITS.values()
    )
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
        values[name] = audit.contracts.digest(path)
    ledger = tmp_path / "freeze.json"
    data = dict(
        stage="frozen_before_first_E02_query_metrics",
        held_query_metrics_already_computed=False,
        seed=2027,
        development_room="E02",
        reference_labels=120,
        primary_zero="masked_temporal",
        primary_calibrated=audit.PRIMARY,
        reference_people=["S11", "S12"],
        calibrated_query_people=[f"S{i:02}" for i in range(13, 21)],
        zero_query_people=[f"S{i:02}" for i in range(11, 21)],
        frozen_sha256=values,
    )
    ledger.write_text(json.dumps(data))
    return ledger, data


@pytest.mark.parametrize(
    "damage",
    ["hash", "ready-omission", "audit-omission", "already-scored-flag", "person"],
)
def test_ledger_rejects_tampering_or_role_changes_before_array_reads(
    tmp_path, monkeypatch, damage
):
    ledger, data = freeze_fixture(tmp_path)
    if damage == "ready-omission":
        del data["frozen_sha256"][next(iter(data["frozen_sha256"]))]
    elif damage == "audit-omission":
        del data["frozen_sha256"][audit.AUDITS["body"]]
    elif damage == "already-scored-flag":
        data["held_query_metrics_already_computed"] = True
    elif damage == "person":
        data["reference_people"] = ["S13", "S14"]
    ledger.write_text(json.dumps(data))
    expected = "0" * 64 if damage == "hash" else audit.contracts.digest(ledger)
    monkeypatch.setattr(
        np, "load", lambda *a, **k: pytest.fail("Unverified array read")
    )
    with pytest.raises(ValueError):
        audit.ledger_contract(ledger, expected, tmp_path)


def test_upstream_fingerprint_omission_and_changed_code_reject(tmp_path):
    proof = dict(
        kind="independent-masked-csi-source-only-audit",
        model_training_performed=False,
        held_metrics_computed=False,
        source_ready_sha256="source",
        auditor_sources_sha256={},
    )
    with pytest.raises(ValueError, match="fingerprints"):
        audit.audit_receipt(tmp_path, "source", proof, "source", "full")
    for name in (
        "scripts/verify_masked_pretrain_probe.py",
        "scripts/research_numeric_readback.py",
    ):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("frozen code")
        proof["auditor_sources_sha256"][name] = audit.contracts.digest(path)
    audit.audit_receipt(tmp_path, "source", proof, "source", "full")
    (tmp_path / "scripts/verify_masked_pretrain_probe.py").write_text("changed code")
    with pytest.raises(ValueError, match="Hash mismatch"):
        audit.audit_receipt(tmp_path, "source", proof, "source", "full")


def test_post_contract_refuses_an_unscored_stage_before_array_reads(
    tmp_path, monkeypatch
):
    ledger, _ = freeze_fixture(tmp_path)
    monkeypatch.setattr(
        np, "load", lambda *a, **k: pytest.fail("Array read in unscored stage")
    )
    with pytest.raises(ValueError, match="completed scored"):
        audit.scored_contract(
            tmp_path, ledger, audit.contracts.digest(ledger), tmp_path, tmp_path
        )


def test_raw_frame_time_join_perturbation_rejected():
    rng = np.random.default_rng(101)
    raw = rng.normal(size=(20, 17, 2))
    valid = np.ones((20, 17), bool)
    data = dict(record=np.array(["E01/S01/A01", "E03/S21/A01"]), end=np.array([7, 8]))
    data["pose"] = raw[[7, 18]] - raw[[7, 18]][:, [11, 12]].mean(1, keepdims=True)
    records = [
        dict(id="E01/S01/A01", offset=0, frames=10),
        dict(id="E03/S21/A01", offset=10, frames=10),
    ]
    audit.root_contract.raw_target_join(raw, valid, records, data)
    data["end"][0] += 1
    with pytest.raises(ValueError):
        audit.root_contract.raw_target_join(raw, valid, records, data)


def test_reference_metadata_overlap_rejected_before_any_prediction():
    data = dict(
        record=np.array(["a", "b", "c"]), person=np.array(["S11", "S12", "S13"])
    )
    with pytest.raises(ValueError, match="overlap"):
        audit.contracts.validate_reference_query(
            data, np.array([0, 1]), np.array([1, 2])
        )
    data["person"][2] = "S11"
    with pytest.raises(ValueError, match="overlap"):
        audit.contracts.validate_reference_query(data, np.array([0, 1]), np.array([2]))


def test_89_set_post_wrapper_matches_legacy_synthetic_reports_without_fitting(
    tmp_path, monkeypatch
):
    # The real-data/hash loader and model reconstruction are covered separately.
    # This tiny fixture exercises all four scored-report/permutation/output paths
    # with original producer metrics, without any real recording or fitting.
    from scripts import domain_reference_covariance_probe as legacy_body
    from scripts import domain_masked_fullpose_probe as legacy_full

    ledger, frozen = freeze_fixture(tmp_path)
    body, full, data = motion_fixture()
    data["pose"] = body
    query = np.arange(12)
    roles = dict(
        train=np.zeros(12, bool), validation=np.zeros(12, bool), held=np.ones(12, bool)
    )
    paths, ready, registrations, payloads = {}, {}, {}, {}
    for protocol, (directory, filename, _, _) in audit.PROTOCOLS.items():
        path = tmp_path / "work" / directory
        paths[protocol] = path
        target = body if protocol.endswith("body") else full
        predictions = {
            name: target + 0.01 for name in sorted(audit.expected_methods(protocol))
        }
        perm = audit.permutation(
            data["record"], 90210 if protocol.startswith("zero_") else 2027
        )
        np.save(path / "query-permutation.npy", perm)
        registration = dict(synthetic=True, protocol=protocol)
        registrations[protocol] = registration
        (path / "registration.json").write_text(json.dumps(registration))
        ready[protocol] = dict(
            artifacts_sha256={
                "registration.json": audit.contracts.digest(path / "registration.json")
            }
        )
        legacy = legacy_body if protocol.endswith("body") else legacy_full
        if protocol.endswith("body"):
            results = {
                name: legacy.scores(prediction, data, query, perm)
                for name, prediction in predictions.items()
            }
        else:
            results = {
                name: legacy.scores(prediction, full, data, query, perm)
                for name, prediction in predictions.items()
            }
        if protocol == "zero_body":
            gate = dict(
                nje_better_all_controls=False,
                pck_not_worse_controls=True,
                positive_motion_r2=True,
                shuffle_worse=True,
            )
        elif protocol == "zero_full":
            gate = dict(
                nje_beats_matched_controls=False,
                pck_not_worse=True,
                positive_full_motion=True,
                positive_body_motion=True,
                shuffle_worse=True,
            )
        elif protocol == "calibrated_body":
            gate = dict(
                beta_positive=False,
                nje_strictly_better_all_controls=False,
                pck_not_worse_all_controls=True,
                motion_positive=True,
                query_shuffle_worse=True,
            )
        else:
            gate = dict(
                beta_positive=False,
                full_nje_strictly_better_all_controls=False,
                full_pck_not_worse_all_controls=True,
                full_pose_motion_positive=True,
                body_motion_positive=True,
                query_shuffle_worse=True,
            )
        result = dict(
            registration=registration, methods=results, gate=gate, passes=False
        )
        ready_field = {
            "zero_body": "source_ready_sha256",
            "zero_full": "fullpose_source_ready_sha256",
            "calibrated_body": "calibration_ready_sha256",
            "calibrated_full": "fullimage_calibration_ready_sha256",
        }[protocol]
        result[ready_field] = audit.contracts.digest(path / filename)
        if protocol.startswith("calibrated_"):
            (path / "frozen-reference-choices.json").write_text("{}")
            result["frozen_choices_sha256"] = audit.contracts.digest(
                path / "frozen-reference-choices.json"
            )
            for name, prediction in predictions.items():
                np.save(
                    path / (name.replace("/", "--") + "-query-prediction.npy"),
                    prediction,
                )
            result["reversals"] = {
                field: {
                    key: [
                        name
                        for name, value in results.items()
                        if name != audit.PRIMARY
                        and primary["normalized_joint_error"]
                        >= value["per_" + field][key]["normalized_joint_error"]
                    ]
                    for key, primary in results[audit.PRIMARY]["per_" + field].items()
                }
                for field in ("person", "action")
            }
        elif protocol == "zero_body":
            (path / "frozen-source-selections.json").write_text("{}")
            result["frozen_source_selection_sha256"] = audit.contracts.digest(
                path / "frozen-source-selections.json"
            )
        (path / "results.json").write_text(json.dumps(result, allow_nan=False))
        payloads[protocol] = (
            query,
            predictions,
            (None if protocol.startswith("zero_") else 0.0),
        )
    monkeypatch.setattr(
        audit,
        "scored_contract",
        lambda *a: (frozen, paths, ready, registrations, data, full, roles),
    )
    monkeypatch.setattr(
        audit,
        "source_predictions",
        lambda *a: (query, payloads["zero_body"][1], payloads["zero_full"][1], {}, {}),
    )
    monkeypatch.setattr(
        audit,
        "calibrated_predictions",
        lambda path, mode, *a: (*payloads["calibrated_" + mode], {}),
    )
    for module in (legacy_body, legacy_full):
        for name in ("run", "fit_root", "fit_arrays", "scores"):
            if hasattr(module, name):
                monkeypatch.setattr(
                    module,
                    name,
                    lambda *a, **k: pytest.fail("Producer called by post auditor"),
                )
    output = tmp_path / "independent-proof.json"
    proof = audit.verify(
        tmp_path,
        tmp_path / "raw",
        tmp_path / "cache",
        ledger,
        output,
        expected_ledger_sha256=audit.contracts.digest(ledger),
    )
    assert proof["prediction_sets_verified"] == 89
    assert proof["metric_findings"] == []
    assert proof["status"] == "qualified" and not any(proof["gate_passes"].values())
    assert json.loads(output.read_text()) == proof


def test_output_preservation_and_frozen_directory_guard(tmp_path):
    output = tmp_path / "proof.json"
    output.write_text("owned")
    for path in (output, tmp_path / "dangling.json"):
        if path != output:
            path.symlink_to(tmp_path / "absent")
        with pytest.raises(ValueError, match="already exists"):
            audit.verify(
                tmp_path,
                tmp_path / "raw",
                tmp_path / "cache",
                tmp_path / "none",
                path,
                expected_ledger_sha256="unused",
            )
    assert output.read_text() == "owned"
    inside = tmp_path / "work" / audit.PROTOCOLS["zero_body"][0] / "new.json"
    with pytest.raises(ValueError, match="outside frozen"):
        audit.verify(
            tmp_path,
            tmp_path / "raw",
            tmp_path / "cache",
            tmp_path / "none",
            inside,
            expected_ledger_sha256="unused",
        )


def test_post_auditor_has_no_fitter_import_or_training_selection_calls():
    tree = ast.parse(Path(audit.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").split(".")[-1].startswith("domain_")
        if isinstance(node, ast.Call):
            name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else (node.func.attr if isinstance(node.func, ast.Attribute) else "")
            )
            assert name not in {
                "fit",
                "fit_arrays",
                "fit_root",
                "fit_reference_model",
                "backward",
                "Adam",
                "AdamW",
                "lu_solution",
                "solve",
                "reconstruct_reference",
            }
