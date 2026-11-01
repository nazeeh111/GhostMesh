"""Synthetic numerical qualification; no real calibration or query metrics."""

import ast
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import verify_masked_calibration_probe as audit


@pytest.fixture(scope="module")
def reference_data():
    rng = np.random.default_rng(81)
    x = rng.normal(size=(120, 5))
    people = np.repeat(["S11", "S12"], 60)
    records = np.repeat([f"E02/S{p}/A{a:02}" for p in (11, 12) for a in (3, 7, 12)], 20)
    skeleton = np.linspace(-0.3, 0.5, 34)
    labels = skeleton + 0.03 * np.sin(x[:, :1]) * np.linspace(0.1, 1, 34)
    base = np.broadcast_to(skeleton, labels.shape).copy()
    source_mean = base[0].copy()
    return x, labels, base, people, records, source_mean


@pytest.fixture(scope="module")
def frozen_states(reference_data, tmp_path_factory):
    # Build toy artifacts with the actual Cholesky fitter once. The auditor
    # below is forbidden from invoking that implementation during qualification.
    from scripts.domain_masked_calibration_probe import fit_reference_model

    x, labels, base, people, records, source_mean = reference_data
    directory = tmp_path_factory.mktemp("frozen-reference-states")
    result = {}
    for method in audit.METHODS:
        state, choice = fit_reference_model(
            x, labels, base, people, records, method=method, source_mean=source_mean
        )
        np.savez(directory / (method + ".npz"), **state)
        (directory / (method + ".json")).write_text(json.dumps(choice, allow_nan=False))
        result[method] = (
            audit.npz(directory / (method + ".npz")),
            json.loads((directory / (method + ".json")).read_text()),
        )
    return result


@pytest.mark.parametrize("method", audit.METHODS)
def test_actual_fitter_states_match_independent_cv_lu_and_far_fallback(
    reference_data, frozen_states, method, monkeypatch
):
    from scripts import domain_masked_calibration_probe as fitter
    from ghostmesh.reference_kernel import ReferenceRBFKernel

    def forbidden(*args, **kwargs):
        raise AssertionError("Auditor must not call fitter math or scoring")

    for name in [
        "fit_reference_model",
        "fit_arrays",
        "kernel_or_none",
        "partitions",
        "nje",
        "permutation",
        "score",
        "scores",
    ]:
        monkeypatch.setattr(fitter, name, forbidden)
    monkeypatch.setattr(ReferenceRBFKernel, "from_reference", forbidden)
    monkeypatch.setattr(ReferenceRBFKernel, "solve", forbidden)
    x, labels, base, people, records, source_mean = reference_data
    state, choice = frozen_states[method]
    result = audit.audit_reference_state(
        x, labels, base, people, records, method, source_mean, state, choice
    )
    assert result["maximum_state_error"] < 1e-10
    assert result["far_reference_correction_error"] == 0
    assert result["cv_candidates"] == (5 if method == "constant_residual" else 25)


def test_cv_uses_only_each_fitting_person_and_rejects_wrong_partition(
    reference_data, frozen_states
):
    x, labels, base, people, records, source_mean = reference_data
    bad = people.copy()
    bad[0] = "S12"
    with pytest.raises(ValueError, match="60/60"):
        audit.reconstruct_reference(
            x, labels, base, bad, records, "residual_krr", source_mean
        )
    state, choice = copy.deepcopy(frozen_states["residual_krr"])
    choice["folds"][0]["fitting_people"] = ["S12"]
    with pytest.raises(ValueError, match="fitting_people"):
        audit.audit_reference_state(
            x, labels, base, people, records, "residual_krr", source_mean, state, choice
        )


@pytest.mark.parametrize(
    "field",
    ["input_mean", "input_scale", "bandwidth_squared", "reference_design", "dual"],
)
def test_kernel_state_stats_width_and_coefficients_cannot_change(
    reference_data, frozen_states, field
):
    x, labels, base, people, records, source_mean = reference_data
    state, choice = copy.deepcopy(frozen_states["residual_krr"])
    assert str(state["correction_kind"]) == "kernel"
    state[field] = state[field] + 0.01
    with pytest.raises(ValueError, match="Numerical mismatch"):
        audit.audit_reference_state(
            x, labels, base, people, records, "residual_krr", source_mean, state, choice
        )


@pytest.mark.parametrize("field", ["alpha", "beta", "correction"])
def test_selected_alpha_beta_and_constant_residual_are_explicit(
    reference_data, frozen_states, field
):
    x, labels, base, people, records, source_mean = reference_data
    method = "constant_residual" if field == "correction" else "residual_krr"
    if field == "correction":
        from scripts.domain_masked_calibration_probe import fit_reference_model

        labels = base + 0.1
        state, choice = fit_reference_model(
            x, labels, base, people, records, method=method
        )
    else:
        state, choice = copy.deepcopy(frozen_states[method])
    assert field in state
    state[field] = state[field] + 0.25
    with pytest.raises(
        ValueError, match="Numerical mismatch|Invalid exact calibration"
    ):
        audit.audit_reference_state(
            x, labels, base, people, records, method, source_mean, state, choice
        )


@pytest.mark.parametrize("field", ["input_mean", "input_scale", "bandwidth_squared"])
def test_cv_diagnostics_reject_fitting_statistic_and_width_changes(
    reference_data, frozen_states, field
):
    x, labels, base, people, records, source_mean = reference_data
    state, choice = copy.deepcopy(frozen_states["residual_krr"])
    value = np.asarray(choice["folds"][0][field]) + 0.1
    choice["folds"][0][field] = value.tolist()
    with pytest.raises(ValueError, match="Numerical mismatch"):
        audit.audit_reference_state(
            x, labels, base, people, records, "residual_krr", source_mean, state, choice
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1000j])
def test_real_finite_features_labels_and_state_are_required(
    reference_data, frozen_states, bad
):
    x, labels, base, people, records, source_mean = reference_data
    dtype = np.complex128 if isinstance(bad, complex) else np.float64
    damaged_x = x.astype(dtype)
    damaged_x[0, 0] = bad
    with pytest.raises(ValueError, match="finite REAL"):
        audit.reconstruct_reference(
            damaged_x, labels, base, people, records, "residual_krr", source_mean
        )
    state, choice = copy.deepcopy(frozen_states["residual_krr"])
    state["dual"] = state["dual"].astype(dtype)
    state["dual"][0, 0] = bad
    with pytest.raises(ValueError, match="finite REAL"):
        audit.audit_reference_state(
            x, labels, base, people, records, "residual_krr", source_mean, state, choice
        )


def test_ties_choose_beta_zero_and_strongest_alpha_and_no_kernel(reference_data):
    x, labels, _, people, records, source_mean = reference_data
    state, choice, _, _ = audit.reconstruct_reference(
        x, labels, labels, people, records, "residual_krr", source_mean
    )
    assert float(state["beta"]) == 0 and float(state["alpha"]) == 10
    assert str(state["correction_kind"]) == "zero" and "dual" not in state
    bad = copy.deepcopy(choice)
    bad["selected"]["alpha"] = 0.001
    with pytest.raises(ValueError, match="Numerical mismatch"):
        audit.audit_reference_state(
            x, labels, labels, people, records, "residual_krr", source_mean, state, bad
        )


def test_degenerate_reference_features_never_invent_kernel(reference_data):
    _, labels, base, people, records, source_mean = reference_data
    x = np.ones((120, 5))
    state, choice, _, _ = audit.reconstruct_reference(
        x, labels, base, people, records, "residual_krr", source_mean
    )
    assert float(state["beta"]) == 0 and str(state["correction_kind"]) == "zero"
    assert all(row["available"] == (row["beta"] == 0) for row in choice["grid"])
    assert all(row["input_mean"] is None for row in choice["folds"])


def test_permuted_labels_have_their_own_person_held_selector(
    reference_data, frozen_states
):
    x, labels, base, people, records, source_mean = reference_data
    perm = audit.reference_permutation(records)
    np.testing.assert_array_equal(records[perm], records)
    assert not np.array_equal(labels[perm], labels)
    state, choice, _, _ = audit.reconstruct_reference(
        x, labels[perm], base, people, records, "residual_krr", source_mean
    )
    expected, expected_choice = frozen_states["permuted_residual_krr"]
    assert str(state["method"]) == "residual_krr"
    for key in set(state) - {"method"}:
        if state[key].dtype.kind in "US":
            assert str(state[key]) == str(expected[key])
        else:
            audit.close(state[key], expected[key], key)
    audit.compare(choice, expected_choice, "permuted own CV")


def test_residual_kernel_has_no_intercept_and_exact_far_f0(
    reference_data, frozen_states
):
    x, labels, base, people, records, source_mean = reference_data
    state, choice = copy.deepcopy(frozen_states["residual_krr"])
    state["intercept"] = np.ones(34)
    with pytest.raises(ValueError, match="no unregistered intercept"):
        audit.audit_reference_state(
            x, labels, base, people, records, "residual_krr", source_mean, state, choice
        )
    kernel = audit.reference_kernel(x)
    far = kernel["input_mean"][None] + 1e8 * kernel["input_scale"][None]
    np.testing.assert_array_equal(audit.cross_kernel(far, kernel), np.zeros((1, 120)))


def canonical_role_fixture():
    ends = [end for first in range(39, 256, 24) for end in (first, first + 8)]
    rows = [
        (f"E02/S{person:02}/A{action:02}", f"S{person:02}", f"A{action:02}", end)
        for person in (11, 12)
        for action in (3, 7, 12)
        for end in ends
    ]
    rows += [
        (f"E02/S{person:02}/A01", f"S{person:02}", "A01", 39)
        for person in range(13, 21)
    ]
    data = dict(
        record=np.array([r[0] for r in rows]),
        person=np.array([r[1] for r in rows]),
        action=np.array([r[2] for r in rows]),
        end=np.array([r[3] for r in rows]),
        room=np.repeat("E02", len(rows)),
    )
    return data, np.arange(120), np.arange(120, 128)


def test_canonical_reference_ids_and_query_person_or_row_overlap_rejected():
    data, reference, query = canonical_role_fixture()
    audit.validate_reference_query(data, reference, query)
    overlap = query.copy()
    overlap[0] = reference[0]
    with pytest.raises(ValueError, match="rows overlap"):
        audit.validate_reference_query(data, reference, overlap)
    d = copy.deepcopy(data)
    d["person"][120] = "S11"
    with pytest.raises(ValueError, match="person overlap"):
        audit.validate_reference_query(d, reference, query)


def test_raw_current_coordinate_extraction_preserves_slot_zero(tmp_path):
    shape = (26730, 9576)
    array = np.lib.format.open_memmap(
        tmp_path / "untreated_covariance-features.npy",
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )
    rows = np.arange(2)
    values = np.arange(2 * 9576, dtype=np.float32).reshape(2, 9576)
    array[rows] = values
    array.flush()
    got = audit.feature_rows(tmp_path, tmp_path, "raw_current", rows, 2)
    np.testing.assert_array_equal(
        got, values.reshape(2, 8, 19, 7, 9)[:, :, :, 0].reshape(2, 1368)
    )


def test_hash_required_before_audit_and_output_preserved(tmp_path):
    output = tmp_path / "output.json"
    output.write_text("preserve")
    with pytest.raises(ValueError, match="already exists"):
        audit.verify(
            "body", tmp_path, tmp_path, tmp_path, None, tmp_path, output, "x" * 64
        )
    output.unlink()
    (tmp_path / "calibration-ready.json").write_text("{}")
    with pytest.raises(ValueError, match="Reviewed calibration-ready hash"):
        audit.verify(
            "body", tmp_path, tmp_path, tmp_path, None, tmp_path, output, "x" * 64
        )
    assert not output.exists()


def test_ready_seal_required_states_and_bytes_checked(tmp_path):
    (tmp_path / "registration.json").write_text("{}")
    ready = dict(
        kind="test",
        query_metrics_computed=False,
        artifacts_sha256={
            "registration.json": audit.digest(tmp_path / "registration.json")
        },
    )
    (tmp_path / "ready.json").write_text(json.dumps(ready))
    with pytest.raises(ValueError, match="omits required"):
        audit.sealed(
            tmp_path,
            "ready.json",
            "test",
            {"missing-state.npz"},
            "query_metrics_computed",
        )
    (tmp_path / "registration.json").write_text("changed")
    with pytest.raises(ValueError, match="Hash mismatch"):
        audit.sealed(
            tmp_path,
            "ready.json",
            "test",
            {"registration.json"},
            "query_metrics_computed",
        )


def test_no_fitter_kernel_optimizer_or_query_metric_imports():
    tree = ast.parse(Path(audit.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not any(
                name in (node.module or "")
                for name in ("domain_masked", "reference_kernel", "metrics")
            )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {
                "backward",
                "step",
                "fit",
                "cho_solve",
                "cho_factor",
            }


def test_float32_references_are_qualified_in_float64(reference_data):
    from scripts.domain_masked_calibration_probe import fit_reference_model

    x, labels, base, people, records, source_mean = reference_data
    x, labels, base = (a.astype(np.float32) for a in (x, labels, base))
    state, choice = fit_reference_model(x, labels, base, people, records)
    audit.audit_reference_state(
        x, labels, base, people, records, "residual_krr", source_mean, state, choice
    )


@pytest.fixture
def f0_bundle(tmp_path):
    source, full_source = tmp_path / "source", tmp_path / "full"
    source.mkdir()
    full_source.mkdir()
    arm, n = "masked_temporal", 120
    feature = np.zeros((n, 3))
    np.savez(
        source / (arm + "-rff.npz"),
        feature_mean=np.zeros(3),
        feature_scale=np.ones(3),
        weight=np.zeros((3, 512)),
        offset=np.zeros(512),
        rff_mean=np.zeros(512),
        rff_scale=np.ones(512),
    )
    body = np.broadcast_to(np.linspace(-0.3, 0.5, 34).reshape(17, 2), (n, 17, 2)).copy()
    np.savez(
        source / (arm + "-readout.npz"),
        selected=np.asarray(0),
        x_mean=np.full(512, 0.0625),
        weights=np.zeros((5, 512, 34)),
        y_mean=body[0].reshape(34),
        prediction=body,
    )
    np.savez(
        full_source / (arm + "-root-state.npz"),
        selected=np.asarray(0),
        beta=np.asarray(0.5),
        x_mean=np.full(512, 0.0625),
        root_mean=np.array([0.2, 0.3]),
        weights=np.zeros((5, 512, 2)),
    )
    expected = body + np.array([0.2, 0.3])[None, None]
    path = full_source / (arm + "-full-prediction.npy")
    np.save(path, expected)
    return source, full_source, arm, feature, np.arange(n), np.ones(n, bool), expected


def test_full_image_f0_uses_only_frozen_body_and_root_coefficients(f0_bundle):
    source, full_source, arm, feature, ids, held, expected = f0_bundle
    got, error = audit.source_base_reference(
        source, full_source, arm, feature, ids, held, "full"
    )
    np.testing.assert_array_equal(got, expected.reshape(len(ids), 34))
    assert error == 0
    path = full_source / (arm + "-full-prediction.npy")
    np.save(path, expected + 0.01)
    with pytest.raises(ValueError, match="frozen full-image reference f0"):
        audit.source_base_reference(
            source, full_source, arm, feature, ids, held, "full"
        )


def test_floor_flag_schema_matches_kernel_state_loader(reference_data, frozen_states):
    from ghostmesh.reference_kernel import ReferenceRBFKernel

    x, labels, base, people, records, source_mean = reference_data
    state, choices = copy.deepcopy(frozen_states["residual_krr"])
    assert state["floored_columns"].dtype == bool
    ReferenceRBFKernel.from_state(state)
    audit.audit_reference_state(
        x, labels, base, people, records, "residual_krr", source_mean, state, choices
    )
    state["floored_columns"] = state["floored_columns"].astype(np.float64)
    with pytest.raises(ValueError, match="Invalid reference normalization state"):
        ReferenceRBFKernel.from_state(state)
    with pytest.raises(ValueError, match="exact Boolean vector"):
        audit.audit_reference_state(
            x,
            labels,
            base,
            people,
            records,
            "residual_krr",
            source_mean,
            state,
            choices,
        )


@pytest.mark.parametrize(
    "field",
    [
        "kind",
        "method",
        "correction_kind",
        "base_kind",
        "alpha",
        "beta",
        "bandwidth_squared",
        "scale_floor",
    ],
)
def test_prediction_state_requires_scalar_text_and_numeric_shapes(
    reference_data, frozen_states, field
):
    x, labels, base, people, records, source_mean = reference_data
    state, choices = copy.deepcopy(frozen_states["residual_krr"])
    state[field] = state[field].reshape(1)
    with pytest.raises(ValueError, match="scalar|Unexpected shape"):
        audit.audit_reference_state(
            x,
            labels,
            base,
            people,
            records,
            "residual_krr",
            source_mean,
            state,
            choices,
        )


def test_near_grid_beta_is_not_a_loadable_exact_beta(reference_data, frozen_states):
    from scripts.domain_masked_calibration_probe import predict_reference_model

    x, labels, base, people, records, source_mean = reference_data
    state, choices = copy.deepcopy(frozen_states["residual_krr"])
    state["beta"] = state["beta"] + 1e-10
    with pytest.raises(ValueError, match="Invalid frozen calibration beta"):
        predict_reference_model(state, x[:1], base[:1])
    with pytest.raises(ValueError, match="Invalid exact calibration beta"):
        audit.audit_reference_state(
            x,
            labels,
            base,
            people,
            records,
            "residual_krr",
            source_mean,
            state,
            choices,
        )


@pytest.mark.parametrize(
    "target,field,shape",
    [
        ("body", "x_mean", (1, 512)),
        ("body", "y_mean", (1, 34)),
        ("body", "weights", (1, 512, 34)),
        ("root", "x_mean", (1, 512)),
        ("root", "root_mean", (1, 2)),
        ("root", "weights", (1, 512, 2)),
        ("mapping", "feature_mean", (1, 3)),
        ("mapping", "offset", (1, 512)),
    ],
)
def test_frozen_source_f0_rejects_broadcastable_wrong_protocol_shapes(
    f0_bundle, target, field, shape
):
    source, full_source, arm, feature, ids, held, _ = f0_bundle
    path = (
        source / (arm + ("-rff.npz" if target == "mapping" else "-readout.npz"))
        if target != "root"
        else full_source / (arm + "-root-state.npz")
    )
    state = audit.npz(path)
    state[field] = (
        state[field][:1].reshape(shape)
        if field == "weights"
        else state[field].reshape(shape)
    )
    np.savez(path, **state)
    with pytest.raises(ValueError, match="Unexpected shape"):
        audit.source_base_reference(
            source, full_source, arm, feature, ids, held, "full"
        )


@pytest.mark.parametrize("target", ["body", "root", "mapping"])
def test_source_parser_nonboolean_numeric_contract(f0_bundle, target):
    source, full_source, arm, feature, ids, held, _ = f0_bundle
    path = (
        full_source / (arm + "-root-state.npz")
        if target == "root"
        else source / (arm + ("-rff.npz" if target == "mapping" else "-readout.npz"))
    )
    state = audit.npz(path)
    field = "weight" if target == "mapping" else "weights"
    state[field] = state[field].astype(bool)
    np.savez(path, **state)
    with pytest.raises(ValueError, match="non-Boolean real source"):
        audit.source_base_reference(
            source, full_source, arm, feature, ids, held, "full"
        )


@pytest.mark.parametrize(
    "mode,omitted",
    [
        ("body", "scripts/domain_masked_calibration_probe.py"),
        ("body", "ghostmesh/reference_kernel.py"),
        ("full", "scripts/domain_masked_fullimage_calibration_probe.py"),
        ("full", "ghostmesh/reference_kernel.py"),
        ("body", "all"),
    ],
)
def test_resealed_ready_cannot_omit_producer_or_kernel_before_array_reads(
    tmp_path, monkeypatch, mode, omitted
):
    kind = (
        "masked-csi-reference-calibration-ready"
        if mode == "body"
        else "masked-csi-fullimage-calibration-ready"
    )
    filename = (
        "calibration-ready.json"
        if mode == "body"
        else "fullimage-calibration-ready.json"
    )
    production = (
        audit.PRODUCTION_BY_KIND[kind] - {omitted}
        if omitted != "all"
        else {"unrelated.txt"}
    )
    source_hashes = {}
    for name in production:
        path = tmp_path / "executed-source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("preserved synthetic source")
        source_hashes[name] = audit.digest(path)
    (tmp_path / "registration.json").write_text(
        json.dumps(dict(source_sha256=source_hashes))
    )
    arms = audit.ENCODERS + (
        ("raw_current", "frozen_temporal") if mode == "body" else ("raw_current",)
    )
    required = {
        "roles-and-priors.npz",
        "frozen-reference-choices.json",
        "reference_mean_mixture.npz",
    }
    required.update(
        arm + "--" + method + ".npz" for arm in arms for method in audit.METHODS
    )
    for name in required:
        (tmp_path / name).write_bytes(b"must not open synthetic arrays")
    manifest = {
        str(path.relative_to(tmp_path)): audit.digest(path)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    (tmp_path / filename).write_text(
        json.dumps(
            dict(kind=kind, query_metrics_computed=False, artifacts_sha256=manifest)
        )
    )
    monkeypatch.setattr(
        np,
        "load",
        lambda *args, **kwargs: pytest.fail(
            "Omitted producer/kernel must fail before array reads"
        ),
    )
    with pytest.raises(ValueError, match="omits required production dependencies"):
        audit.verify(
            mode,
            tmp_path,
            tmp_path,
            tmp_path,
            tmp_path,
            tmp_path,
            tmp_path / "report.json",
            audit.digest(tmp_path / filename),
        )


def test_prospective_audit_refuses_already_scored_experiment(tmp_path, monkeypatch):
    (tmp_path / "results.json").write_text("already scored")
    monkeypatch.setattr(
        audit, "sealed", lambda *args: pytest.fail("No data reads after scoring begins")
    )
    with pytest.raises(ValueError, match="Query scoring already started"):
        audit.verify(
            "full",
            tmp_path,
            tmp_path,
            tmp_path,
            tmp_path,
            tmp_path,
            tmp_path / "report.json",
            "x" * 64,
        )
