"""Independent reference algebra, metric semantics and auditor failure guards."""

import numpy as np
import pytest
import torch

from ghostmesh.amplitude_pose import normalize_amplitude
from ghostmesh.episodic_calibration import fit_residual
from scripts import verify_amplitude_source as v


def test_explicit_flattening_and_general_solve_reproduce_reference_state():
    rng = np.random.default_rng(4)
    code = rng.normal(size=(7, 5))
    y = rng.normal(size=(7, 34))
    base = rng.normal(size=(7, 34))
    mean, bias, w = v.reference_solution(code, y, base, 0.1)
    state = fit_residual(torch.tensor(code), torch.tensor(y), torch.tensor(base))
    np.testing.assert_allclose(mean, state.code_mean.numpy(), atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(
        bias, state.correction_mean.numpy(), atol=2e-12, rtol=2e-12
    )
    np.testing.assert_allclose(w, state.weights.numpy(), atol=2e-12, rtol=2e-12)
    with pytest.raises(ValueError, match="flattened"):
        v.reference_solution(code, y.reshape(7, 17, 2), base.reshape(7, 17, 2), 0.1)
    with pytest.raises(ValueError, match="Explicit2D"):
        v.product(code.T, y.reshape(7, 17, 2))


def test_independent_packet_column_imputation_and_frame_normalization():
    raw = np.random.default_rng(8).normal(30, 2, (2, 8, 3, 114, 10)).astype(np.float32)
    raw[0, 0, 0, 0, 0] = np.nan
    raw[1, 4, 1, 5, 9] = np.inf
    np.testing.assert_allclose(
        v.independent_normalize(raw), normalize_amplitude(raw), atol=1e-7, rtol=0
    )


def test_independent_motion_metric_distinguishes_static_perfect_and_biased():
    y = np.tile(
        np.stack([np.linspace(-0.3, 0.3, 17), np.linspace(-0.4, 0.4, 17)], 1), (4, 1, 1)
    )
    y[:, 0, 0] += [0, 0.02, 0.03, 0.05]
    r = np.array(["a", "a", "b", "b"])
    ends = np.array([39, 47, 39, 47])
    result = v.metric(y, y, r, ends)
    assert result["motion_r2"] == 1 and result["nje"] == 0
    static = y.copy()
    for label in set(r):
        static[r == label] = y[r == label].mean(0)
    assert v.metric(static, y, r, ends)["motion_r2"] == 0
    biased = v.metric(y + 0.01, y, r, ends)
    assert biased["motion_r2"] == pytest.approx(1) and biased["nje"] > 0
    with pytest.raises(ValueError, match="disagreement"):
        v.compare(result, {**result, "nje": 0.01})


def test_undefined_motion_never_passes_the_scientific_gate():
    score = dict(motion_r2=None, delta_r2=None, nje=0.1, pck05=0.5, pck20=0.9)
    primary = dict(
        pooled=score,
        per_person={"S04": score, "S09": score},
        shuffled={**score, "nje": 0.2},
    )
    gate = v.gate(primary, [primary])
    assert gate["body_motion_useful"] is False and gate["both_people_positive"] is False
    assert gate["nje_beats_controls"] is False and all(gate.values()) is False
