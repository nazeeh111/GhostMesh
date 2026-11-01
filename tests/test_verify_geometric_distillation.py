"""Analytic oracle and corrupt-report refusal for the independent auditor."""

import numpy as np
import pytest

from scripts.verify_geometric_distillation import close_scores, metadata_only, scores


def fixture():
    target = np.tile(
        np.stack([np.linspace(-0.2, 0.2, 17), np.linspace(-0.4, 0.4, 17)], axis=1),
        (4, 1, 1),
    )
    target[:, 0, 0] += [0, 0.02, 0.04, 0.06]
    return target, np.array(["a", "a", "b", "b"]), np.array([39, 47, 39, 47])


def test_identity_static_and_constant_bias_have_independent_analytic_motion():
    y, r, e = fixture()
    s = scores(y, y, r, e)
    assert s == dict(motion_r2=1.0, delta_r2=1.0, nje=0.0, pck05=1.0, pck20=1.0)
    static = np.empty_like(y)
    for label in set(r):
        static[r == label] = y[r == label].mean(0)
    assert scores(static, y, r, e)["motion_r2"] == 0
    shifted = scores(y + 0.01, y, r, e)
    assert shifted["motion_r2"] == pytest.approx(1.0) and shifted[
        "delta_r2"
    ] == pytest.approx(1.0)
    assert shifted["nje"] > 0


def test_geometry_constant_target_has_undefined_motion_not_fabricated_score():
    s = scores(
        np.ones((2, 136)), np.ones((2, 136)), np.array(["a", "a"]), np.array([39, 47])
    )
    assert s == dict(motion_r2=None, delta_r2=None, mse=0.0)


@pytest.mark.parametrize("corruption", ["nan", "complex", "duplicate", "degenerate"])
def test_invalid_metrics_fail_closed(corruption):
    y, r, e = fixture()
    p = y.copy()
    if corruption == "nan":
        p[0, 0, 0] = np.nan
    elif corruption == "complex":
        p = p.astype(complex)
    elif corruption == "duplicate":
        e[1] = e[0]
    else:
        y[:] = 0
    with pytest.raises(ValueError):
        scores(p, y, r, e)


def test_corrupt_or_undefined_reports_are_rejected():
    with pytest.raises(ValueError, match="disagreement"):
        close_scores(dict(nje=0.1), dict(nje=0.11))
    with pytest.raises(ValueError, match="Undefined"):
        close_scores(dict(motion=None), dict(motion=0.0))
    with pytest.raises(ValueError, match="Nonfinite"):
        close_scores(dict(nje=0.1), dict(nje=np.nan))


def test_metadata_reader_does_not_decode_target_annotation_members(tmp_path):
    path = tmp_path / "source-container.npz"
    fields = {
        key: np.array(["identifier"]) for key in ("record", "person", "room", "action")
    }
    fields["end"] = np.array([39])
    # A forbidden member raises with allow_pickle=False if the reader touches it.
    np.savez(path, **fields, pose=np.array([{"forbidden": True}], dtype=object))
    actual = metadata_only(path)
    assert set(actual) == set(fields)
    for key, value in fields.items():
        np.testing.assert_array_equal(actual[key], value)
