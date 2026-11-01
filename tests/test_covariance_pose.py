import numpy as np
import pytest
import torch
from torch.nn import functional as F

from ghostmesh.covariance_pose import CovariancePose, root_relative


def test_two_heads_initial_prior_and_pose_projection():
    model = CovariancePose()
    mean = torch.arange(34, dtype=torch.float32) / 100
    model.fit_source_stats(
        np.zeros((18, 1)), np.ones((18, 1)), mean, np.ones(34), np.ones(34)
    )
    result = model(torch.randn(3, 8, 18, 19))
    assert result["current"].shape == result["past"].shape == (3, 34)
    assert not result["current"].any() and not result["past"].any()
    expected = root_relative(mean.reshape(1, 17, 2)).expand(3, -1, -1)
    torch.testing.assert_close(model.decode(result["current"]), expected)
    torch.testing.assert_close(
        model.decode(result["current"]) - model.decode(result["past"]),
        torch.zeros_like(expected),
    )
    with pytest.raises(ValueError):
        model(torch.randn(2, 8, 19, 18))
    with pytest.raises(ValueError):
        model.fit_source_stats(
            np.zeros((18, 1)), np.zeros((18, 1)), mean, np.ones(34), np.ones(34)
        )


def test_temporal_causality_and_no_cross_query_batch_statistics():
    torch.manual_seed(4)
    model = CovariancePose().eval()
    latent = torch.randn(2, 8, 64)
    changed = latent.clone()
    changed[:, 4:] += 100
    torch.testing.assert_close(
        model.temporal(latent)[:, :4], model.temporal(changed)[:, :4], rtol=0, atol=0
    )
    x = torch.randn(3, 8, 18, 19)
    y = torch.randn(2, 8, 18, 19) * 10
    torch.testing.assert_close(
        model.encode(x), model.encode(torch.cat((x, y)))[:3], rtol=1e-5, atol=1e-6
    )


def test_matched_objectives_use_both_original_endpoint_labels_and_reach_encoder():
    torch.manual_seed(5)
    model = CovariancePose()
    with torch.no_grad():
        model.current.weight.normal_(0, 0.1)
        model.past.weight.normal_(0, 0.1)
    model.delta_scale.fill_(0.2)
    x = torch.randn(4, 8, 18, 19)
    y = root_relative(torch.randn(4, 17, 2) * 0.1)
    past = y + root_relative(torch.randn_like(y) * 0.01)
    prediction = model(x)
    ordinary = model.loss(prediction, y, past, "absolute")
    expected = 0.5 * F.mse_loss(
        prediction["current"], model.standardized_target(y)
    ) + 0.5 * F.mse_loss(prediction["past"], model.standardized_target(past))
    torch.testing.assert_close(ordinary, expected)
    motion = model.loss(prediction, y, past, "motion")
    expected = 0.5 * F.mse_loss(
        prediction["current"], model.standardized_target(y)
    ) + 0.5 * F.mse_loss(
        (
            model.decode(prediction["current"]) - model.decode(prediction["past"])
        ).flatten(-2)
        / 0.2,
        (y - past).flatten(-2) / 0.2,
    )
    torch.testing.assert_close(motion, expected)
    assert not torch.isclose(ordinary, motion)
    motion.backward()
    for parameter in [
        model.spectral[0].weight,
        model.current.weight,
        model.past.weight,
    ]:
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        model.loss(prediction, y, past, "other")
