"""Geometric identities, not empirical radio accuracy."""

import numpy as np
import pytest

from ghostmesh.pose_shape import PAIR_I, PAIR_J, canonical_distances


def test_shape_is_rigid_reflection_and_isotropic_scale_invariant():
    rng = np.random.default_rng(2027)
    cloud = rng.normal(size=(5, 17, 3))
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    transformed = np.einsum("nji,ik->njk", cloud, rotation)
    transformed[:, :, 0] *= -1  # Reflection is intentionally not identifiable.
    transformed = transformed * np.arange(1, 6)[:, None, None] + [8, -10, 2]
    np.testing.assert_allclose(
        canonical_distances(cloud),
        canonical_distances(transformed),
        atol=3e-14,
        rtol=3e-14,
    )


def test_pairwise_distances_match_independent_scalar_calculation_and_edm_identity():
    rng = np.random.default_rng(4)
    cloud = rng.normal(size=(17, 3))
    centered = cloud - cloud.mean(0)
    radius = np.sqrt(sum(sum(float(v) ** 2 for v in row) for row in centered) / 17)
    direct = [
        sum(float(a - b) ** 2 for a, b in zip(cloud[i], cloud[j])) ** 0.5 / radius
        for i in range(17)
        for j in range(i + 1, 17)
    ]
    actual = canonical_distances(cloud)
    np.testing.assert_allclose(actual, direct, atol=1e-14, rtol=1e-14)
    assert np.square(actual).sum() == pytest.approx(17**2, abs=3e-12)
    matrix = np.zeros((17, 17))
    matrix[PAIR_I, PAIR_J] = actual**2
    matrix += matrix.T
    centered_matrix = matrix - matrix.mean(0) - matrix.mean(1)[:, None] + matrix.mean()
    eigen = np.linalg.eigvalsh(-0.5 * centered_matrix)
    assert eigen.min() > -1e-12
    assert np.count_nonzero(eigen > 1e-10) <= 3


def test_extreme_positive_scales_preserve_shape_without_squaring_raw_coordinates():
    cloud = np.arange(51, dtype=float).reshape(17, 3) - 25
    expected = canonical_distances(cloud)
    for scale in [1e-280, 1e280]:
        np.testing.assert_allclose(
            canonical_distances(cloud * scale), expected, atol=1e-14, rtol=1e-14
        )
    shifted = cloud + 1e9
    np.testing.assert_allclose(
        canonical_distances(shifted), expected, atol=1e-14, rtol=1e-14
    )


def test_joint_permutations_reindex_distances_and_are_never_a_coco_conversion():
    rng = np.random.default_rng(9)
    cloud = rng.normal(size=(17, 3))
    permutation = rng.permutation(17)
    original = canonical_distances(cloud)
    matrix = np.zeros((17, 17))
    matrix[PAIR_I, PAIR_J] = original
    matrix += matrix.T
    np.testing.assert_allclose(
        canonical_distances(cloud[permutation]),
        matrix[permutation[:, None], permutation[None, :]][PAIR_I, PAIR_J],
        atol=1e-14,
        rtol=1e-14,
    )
    assert not np.allclose(canonical_distances(cloud[permutation]), original)


@pytest.mark.parametrize("bad", [np.nan, np.inf, 1j, 0j, True, "1"])
def test_nonfinite_complex_boolean_and_text_targets_refused(bad):
    dtype = (
        complex
        if isinstance(bad, complex)
        else bool
        if isinstance(bad, bool)
        else str
        if isinstance(bad, str)
        else float
    )
    cloud = np.ones((17, 3), dtype=dtype)
    cloud[0, 0] = bad
    with pytest.raises(ValueError, match="Finite real"):
        canonical_distances(cloud)


def test_collapsed_cloud_and_wrong_layout_refused_without_imputing_targets():
    for bad in [np.ones((17, 3)), np.zeros((3, 17, 3))]:
        with pytest.raises(ValueError, match="Collapsed"):
            canonical_distances(bad)
    with pytest.raises(ValueError, match="17,3"):
        canonical_distances(np.zeros((17, 2)))
    with pytest.raises(ValueError, match="17,3"):
        canonical_distances(np.zeros((0, 17, 3)))
