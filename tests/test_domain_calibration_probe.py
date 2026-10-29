"""Analytical checks of calibration math and held-person selection."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("domain_calibration_probe", Path(__file__).parents[1] / "scripts/domain_calibration_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_dual_ridge_matches_primal_with_intercept():
    rng = np.random.default_rng(9)
    x, y, z = rng.normal(size=(7, 4)), rng.normal(size=(7, 17, 2)), rng.normal(size=(3, 4))
    alpha = .3
    fit = probe.ridge_fit(x, y, alpha)
    xc, yc = x - x.mean(0), y.reshape(7, -1) - y.reshape(7, -1).mean(0)
    weight = np.linalg.solve(xc.T @ xc + alpha * len(x) * np.eye(4), xc.T @ yc)
    expected = ((z - x.mean(0)) @ weight + y.reshape(7, -1).mean(0)).reshape(3, 17, 2)
    np.testing.assert_allclose(probe.ridge_predict(fit, z), expected, atol=1e-12)


def test_shrink_endpoints_are_target_mean_and_joint_residual():
    rng = np.random.default_rng(3)
    p, cp, y = rng.normal(size=(5, 17, 2)), rng.normal(size=(8, 17, 2)), rng.normal(size=(8, 17, 2))
    np.testing.assert_allclose(probe.shrink(p, cp, y, 0), np.broadcast_to(y.mean(0), p.shape), atol=1e-12)
    np.testing.assert_allclose(probe.shrink(p, cp, y, 1), p + y.mean(0) - cp.mean(0), atol=1e-12)


def test_selection_excludes_forbidden_rooms_and_motion_static_oracle():
    with pytest.raises(ValueError, match="Forbidden"):
        probe.calibration_rows({"sequences": []}, "E04")
    assert probe.ENDS == tuple(range(31, 293, 29))
    rng = np.random.default_rng(7)
    y = rng.normal(size=(74, 17, 2))
    groups = np.arange(74).reshape(2, 37)
    static = np.repeat(y.reshape(2, 37, 17, 2).mean(1), 37, axis=0)
    for item in probe.motion(static, y, groups).values():
        assert item["r2"] == pytest.approx(0, abs=1e-12)
    for item in probe.motion(y, y, groups).values():
        assert item["r2"] == pytest.approx(1, abs=1e-12)
