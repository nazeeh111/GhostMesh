"""Signal/gradient/causality contracts for a new unvalidated raw-amplitude model."""

import numpy as np
import pytest
import torch

from ghostmesh.amplitude_pose import (
    AmplitudeStudent,
    CausalBlock,
    normalize_amplitude,
    reverse_gradient,
)


def test_frame_normalization_uses_no_other_frames_and_cancels_common_affine_change():
    rng = np.random.default_rng(4)
    x = rng.normal(30, 3, (2, 8, 3, 114, 10))
    actual = normalize_amplitude(x)
    np.testing.assert_allclose(
        normalize_amplitude(2 * x + 9), actual, atol=1e-7, rtol=0
    )
    changed = x.copy()
    changed[:, 1:] *= 10
    np.testing.assert_array_equal(normalize_amplitude(changed)[:, 0], actual[:, 0])
    assert actual.min() == 0 and actual.max() == 1


def test_missing_packet_cells_are_imputed_from_same_frame_column_only():
    x = np.arange(2 * 8 * 3 * 114 * 10, dtype=float).reshape(2, 8, 3, 114, 10) + 1
    x[0, 0, 0, 0, 0] = np.nan
    result = normalize_amplitude(x)
    column = x[0, 0, :, :, 0]
    fill = np.nanmean(column)
    fixed = x.copy()
    fixed[0, 0, 0, 0, 0] = fill
    np.testing.assert_array_equal(result, normalize_amplitude(fixed))
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="missing magnitude"):
        normalize_amplitude(x)


def test_receiver_gain_augmentation_is_history_constant_and_not_a_room_transform_claim():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(2, 8, 3, 114, 10))
    gain = np.array([[1.0, -2.0, 3.0], [-1.0, 2.0, -3.0]])
    np.testing.assert_array_equal(
        normalize_amplitude(x, gain_db=gain),
        normalize_amplitude(x + gain[:, None, :, None, None]),
    )
    with pytest.raises(ValueError, match="gains"):
        normalize_amplitude(x, gain_db=np.zeros((2, 8, 3)))


def test_gradient_reversal_preserves_forward_and_has_exact_analytic_backward():
    x = torch.tensor([1.0, -2.0], requires_grad=True)
    out = reverse_gradient(x, 0.25)
    torch.testing.assert_close(out, x, rtol=0, atol=0)
    out.square().sum().backward()
    torch.testing.assert_close(x.grad, -0.5 * x.detach(), atol=0, rtol=0)
    with pytest.raises(ValueError):
        reverse_gradient(x, float("nan"))


def test_temporal_filter_does_not_read_future_positions():
    torch.manual_seed(5)
    net = CausalBlock(8, 2).eval()
    x = torch.randn(2, 8, 8)
    changed = x.clone()
    changed[:, :, 4:] = torch.randn_like(changed[:, :, 4:]) * 100
    with torch.inference_mode():
        torch.testing.assert_close(
            net(x)[:, :, :4], net(changed)[:, :, :4], atol=0, rtol=0
        )


def test_initial_pose_is_source_prior_and_nuisance_classifier_cannot_backprop_at_zero_reversal():
    torch.manual_seed(6)
    mean = torch.randn(17, 2)
    model = AmplitudeStudent(mean, torch.ones(17, 2))
    x = torch.randn(2, 8, 3, 114, 10)
    result = model(x)
    expected = mean - mean[[11, 12]].mean(0, keepdim=True)
    torch.testing.assert_close(
        result["pose"], expected.expand(2, -1, -1), atol=0, rtol=0
    )
    result["nuisance_logits"].square().mean().backward()
    assert (
        sum(
            float(p.grad.abs().sum())
            for p in model.body.parameters()
            if p.grad is not None
        )
        == 0
    )
    assert (
        sum(
            float(p.grad.abs().sum())
            for p in model.nuisance.parameters()
            if p.grad is not None
        )
        > 0
    )


def test_constant_frames_and_complex_inputs_are_refused():
    with pytest.raises(ValueError, match="Constant"):
        normalize_amplitude(np.ones((1, 8, 3, 114, 10)))
    with pytest.raises(ValueError, match="Real"):
        normalize_amplitude(np.ones((1, 8, 3, 114, 10), complex))
