import numpy as np
import pytest

from ghostmesh.motion_filter import packet_moments
from scripts.verify_pose_alignment_source import (
    independent_moments,
    raw_fit,
    raw_kernel,
)


def test_raw_packet_moment_reconstruction_matches_guarded_producer():
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(2, 32, 6, 114, 10)).astype(np.float32)
    raw[:, :, :3] *= 20
    raw[0, 4, 0, 1, 2] = -np.inf
    actual = independent_moments(raw)
    expected = np.stack([packet_moments(r) for r in raw])
    np.testing.assert_allclose(actual, expected, atol=5e-7, rtol=2e-6)
    with pytest.raises(ValueError):
        independent_moments(raw[:, :8])


def test_reference_rbf_bandwidth_cannot_use_query_distribution():
    rng = np.random.default_rng(2)
    refs = rng.normal(size=(12, 24))
    queries = rng.normal(size=(3, 24))
    _, first = raw_kernel(refs, queries, "rbf")
    _, second = raw_kernel(refs, queries * 1000, "rbf")
    assert first == second
    k, _ = raw_kernel(refs, refs, "rbf")
    np.testing.assert_allclose(np.diag(k), 1, atol=1e-14)


def test_raw_reference_equation_keeps_coco_coordinates_in_c_order():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(40, 50))
    y = rng.normal(size=(40, 17, 2))
    y[:, :, 0] += 3
    y[:, :, 1] -= 7
    pred, state = raw_fit(x, y, x, "rbf", 0.001)
    kernel, _ = raw_kernel(x, x, "rbf")
    left = kernel + 0.001 * np.eye(40)
    right = y.reshape(40, 34) - y.mean(0).reshape(1, 34)
    np.testing.assert_allclose(
        np.einsum("ij,jk->ik", left, state["weights"]), right, atol=1e-12
    )
    assert np.max(np.abs(pred - y)) < 0.03
