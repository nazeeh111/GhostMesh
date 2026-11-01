"""Synthetic software fixtures only; these tests make no accuracy claim."""

import numpy as np
import pytest
from scipy.linalg import solve
from scipy.linalg.blas import dgemm

from ghostmesh.reference_kernel import ReferenceRBFKernel


def fixture():
    rng = np.random.default_rng(212)
    return rng.normal(size=(12, 5)), rng.normal(size=(12, 3)), rng.normal(size=(9, 5))


def direct(reference, query, targets, alpha):
    std = np.maximum(reference.std(axis=0), 0.001)
    x = (reference - reference.mean(axis=0)) / std
    q = (query - reference.mean(axis=0)) / std
    distance = np.square(x[:, None] - x[None, :]).sum(axis=2)
    width = np.median(distance[np.triu_indices(len(x), 1)])
    kernel = np.exp(-distance / (2 * width))
    dual = solve(kernel + len(x) * alpha * np.eye(len(x)), targets)
    cross = np.exp(-np.square(q[:, None] - x[None, :]).sum(axis=2) / (2 * width))
    return dual, dgemm(1.0, cross, dual), width


def test_matches_independent_dense_formula_and_regularization_convention():
    reference, targets, query = fixture()
    kernel = ReferenceRBFKernel.from_reference(reference)
    for alpha in (0.001, 0.01, 0.1, 1, 10):
        dual, residual = kernel.solve(targets, alpha)
        expected_dual, expected, width = direct(reference, query, targets, alpha)
        assert kernel.bandwidth_squared == pytest.approx(width, rel=1e-15)
        np.testing.assert_allclose(dual, expected_dual, atol=2e-12, rtol=2e-12)
        np.testing.assert_allclose(kernel.predict(query, dual), expected, atol=1e-12)
        assert residual <= 1e-8
        assert dual.dtype == np.float64


def test_reference_only_stats_query_batches_prefixes_and_immutable_inputs():
    reference, targets, query = fixture()
    original = reference.copy()
    kernel = ReferenceRBFKernel.from_reference(reference)
    dual, _ = kernel.solve(targets, 0.01)
    before = kernel.predict(query, dual)
    np.testing.assert_array_equal(kernel.input_mean, original.mean(axis=0))
    np.testing.assert_array_equal(kernel.input_scale, original.std(axis=0))
    reference[:] = -1000
    changed = query.copy()
    changed[4:] = 1000
    np.testing.assert_array_equal(kernel.predict(changed, dual)[:4], before[:4])
    np.testing.assert_array_equal(kernel.predict(query[:4], dual), before[:4])
    np.testing.assert_allclose(
        np.concatenate([kernel.predict(q[None], dual) for q in query]),
        before,
        atol=2e-15,
        rtol=2e-15,
    )
    for value in (
        kernel.input_mean,
        kernel.input_scale,
        kernel.reference_design,
        kernel.floored_columns,
    ):
        assert not value.flags.writeable
    np.testing.assert_array_equal(kernel.input_mean, original.mean(axis=0))
    assert kernel.predict(query[:0], dual).shape == (0, targets.shape[1])


def test_nonzero_target_mean_is_not_centered_and_far_correction_vanishes():
    reference, _, query = fixture()
    kernel = ReferenceRBFKernel.from_reference(reference)
    targets = np.full((len(reference), 2), [7.0, -3.0])
    dual, _ = kernel.solve(targets, 0.01)
    _, expected, _ = direct(reference, query, targets, 0.01)
    np.testing.assert_allclose(kernel.predict(query, dual), expected, atol=1e-12)
    assert np.max(np.abs(expected)) > 1
    np.testing.assert_array_equal(kernel.predict(np.full((3, 5), 1e6), dual), 0)
    zero_dual, residual = kernel.solve(np.zeros_like(targets), 0.01)
    np.testing.assert_array_equal(zero_dual, 0)
    assert residual == 0


def test_column_floor_is_disclosed_and_does_not_remove_columns():
    reference = np.array([[0.0, 0.0, 10.0], [2.0, 0.0001, 10.0], [4.0, 0.0002, 10.0]])
    kernel = ReferenceRBFKernel.from_reference(reference)
    np.testing.assert_array_equal(kernel.floored_columns, [False, True, True])
    np.testing.assert_array_equal(kernel.input_scale[1:], [0.001, 0.001])
    assert kernel.reference_design.shape == reference.shape
    np.testing.assert_allclose(kernel.reference_design[:, 1], [-0.1, 0, 0.1])
    query = reference.copy()
    query[0, 2] += 0.003
    assert not np.array_equal(
        kernel.cross_kernel(query), kernel.cross_kernel(reference)
    )


def test_npz_roundtrip_predictions_are_exact_and_state_has_no_target_fields(tmp_path):
    reference, targets, query = fixture()
    kernel = ReferenceRBFKernel.from_reference(reference)
    dual, _ = kernel.solve(targets, 0.1)
    state = kernel.state_dict()
    assert set(state) == {
        "kind",
        "input_mean",
        "input_scale",
        "reference_design",
        "bandwidth_squared",
        "scale_floor",
        "floored_columns",
    }
    path = tmp_path / "reference.npz"
    np.savez(path, **state, dual=dual)
    with np.load(path, allow_pickle=False) as saved:
        restored = ReferenceRBFKernel.from_state(saved)
        np.testing.assert_array_equal(
            restored.predict(query, saved["dual"]), kernel.predict(query, dual)
        )
    state["input_mean"][:] = 100
    np.testing.assert_array_equal(kernel.input_mean, reference.mean(axis=0))


def test_degenerate_reference_invalid_floor_and_zero_median_are_rejected():
    for reference in (
        np.ones((3, 2)),
        np.arange(6).reshape(3, 2) * 1e-6,
        np.zeros((1, 2)),
        np.zeros((3, 0)),
        np.zeros(3),
    ):
        with pytest.raises(ValueError):
            ReferenceRBFKernel.from_reference(reference)
    # Most unordered pairs are duplicates despite one nonconstant column.
    with pytest.raises(ValueError, match="median bandwidth"):
        ReferenceRBFKernel.from_reference(np.array([[0.0], [0.0], [0.0], [0.0], [2.0]]))
    for floor in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError):
            ReferenceRBFKernel.from_reference(fixture()[0], scale_floor=floor)


def test_nonfinite_values_shapes_and_alpha_are_rejected():
    reference, targets, query = fixture()
    kernel = ReferenceRBFKernel.from_reference(reference)
    for invalid in (np.nan, np.inf, -np.inf):
        bad = reference.copy()
        bad[0, 0] = invalid
        with pytest.raises(ValueError):
            ReferenceRBFKernel.from_reference(bad)
        bad = query.copy()
        bad[0, 0] = invalid
        with pytest.raises(ValueError):
            kernel.cross_kernel(bad)
        bad = targets.copy()
        bad[0, 0] = invalid
        with pytest.raises(ValueError):
            kernel.solve(bad, 0.1)
        with pytest.raises(ValueError):
            kernel.predict(query, bad)
    for alpha in (0, -1, np.nan, np.inf, [0.1]):
        with pytest.raises(ValueError):
            kernel.solve(targets, alpha)
    for bad in (targets[:, 0], targets[:-1], np.empty((12, 0))):
        with pytest.raises(ValueError):
            kernel.solve(bad, 0.1)
    for bad in (query[:, 0], query[:, :-1]):
        with pytest.raises(ValueError):
            kernel.cross_kernel(bad)
    for field, value in (
        ("kind", "wrong"),
        ("bandwidth_squared", 0),
        ("input_scale", np.zeros(5)),
        ("reference_design", targets),
    ):
        with pytest.raises(ValueError):
            ReferenceRBFKernel.from_state(kernel.state_dict() | {field: value})
    state = kernel.state_dict()
    del state["input_mean"]
    with pytest.raises(ValueError, match="Missing"):
        ReferenceRBFKernel.from_state(state)


def test_protocol_sized_synthetic_system_and_extreme_finite_inputs():
    rng = np.random.default_rng(89)
    reference = rng.normal(size=(120, 1368))
    targets = rng.normal(size=(120, 34))
    kernel = ReferenceRBFKernel.from_reference(reference)
    dual, residual = kernel.solve(targets, 0.001)
    assert dual.shape == (120, 34)
    assert residual <= 1e-8
    assert kernel.predict(reference[:3], dual).shape == (3, 34)
    np.testing.assert_array_equal(kernel.predict(np.full((1, 1368), 1e100), dual), 0)
    with pytest.raises(ValueError, match="distance|normalization"):
        kernel.cross_kernel(np.full((1, 1368), 1e308))
    with pytest.raises(ValueError, match="alpha"):
        kernel.solve(targets, 1e308)
    # A positive subnormal median remains valid. Its far kernels become zero.
    tiny = ReferenceRBFKernel.from_reference(
        np.array([[0.0, 0.0], [0.0, 1e-160], [0.0, 2e-160], [0.0, 3e-160], [1.0, 0.0]])
    )
    assert 0 < tiny.bandwidth_squared < np.finfo(float).tiny
    assert np.isfinite(tiny.cross_kernel(np.array([[0.0, 0.0], [1.0, 0.0]]))).all()
