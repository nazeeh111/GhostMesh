import numpy as np
import pytest
import torch

from ghostmesh.masked_csi import (
    MASKED_TOKENS,
    MaskedCSI,
    fixed_count_mask,
    masked_reconstruction,
    normalize_frames,
    patchify,
    pretext_loss,
    temporal_regularizer,
    unpatchify,
)


def test_patch_axes_roundtrip_and_specific_receiver_frequency_packet_coordinates():
    x = torch.arange(2 * 3 * 114 * 10).reshape(2, 3, 114, 10)
    patches = patchify(x)
    torch.testing.assert_close(unpatchify(patches), x)
    for n, r, f, p in [(0, 0, 0, 0), (1, 2, 113, 9), (0, 1, 47, 6)]:
        assert (
            patches[n, f // 6 * 5 + p // 2, r * 12 + f % 6 * 2 + p % 2] == x[n, r, f, p]
        )


def test_frame_normalization_is_pointwise_affine_invariant_and_rejects_no_signal():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(3, 3, 114, 10)).astype(np.float32)
    y = normalize_frames(x)
    np.testing.assert_allclose(y, normalize_frames(x * 2 + 15), atol=2e-6)
    altered = x.copy()
    altered[1:] *= 500
    np.testing.assert_array_equal(y[0], normalize_frames(altered)[0])
    for bad in [np.ones_like(x), np.full_like(x, np.nan), x[:, :, :-1]]:
        with pytest.raises(ValueError):
            normalize_frames(bad)


def test_mask_has_fixed_count_is_seeded_and_only_masked_targets_enter_loss():
    generator = torch.Generator().manual_seed(3)
    mask = fixed_count_mask(2, generator)
    assert MASKED_TOKENS == 71
    assert (mask.sum(1) == MASKED_TOKENS).all()
    torch.testing.assert_close(
        mask, fixed_count_mask(2, torch.Generator().manual_seed(3))
    )
    result = dict(target=torch.zeros(2, 95, 36), reconstruction=torch.ones(2, 95, 36))
    before = masked_reconstruction(result, mask)
    result["reconstruction"][~mask] = 100
    assert masked_reconstruction(result, mask) == before == 1


def test_collapse_penalty_and_pair_shuffle_change_alignment_without_changing_reconstruction():
    collapsed = torch.ones(8, 64)
    terms = temporal_regularizer(collapsed, collapsed)
    assert terms["invariance"] == terms["covariance"] == 0
    assert terms["variance"] > 0.98
    torch.manual_seed(6)
    model = MaskedCSI()
    x = torch.randn(2, 8, 3, 114, 10)
    mask = fixed_count_mask(16, torch.Generator().manual_seed(4))
    _, natural = pretext_loss(model, x, mask)
    _, shuffled = pretext_loss(model, x, mask, shuffled=True)
    torch.testing.assert_close(natural["reconstruction"], shuffled["reconstruction"])
    assert not torch.equal(natural["invariance"], shuffled["invariance"])


def test_small_model_finite_reconstruction_and_temporal_gradients_and_history_causality():
    torch.set_num_threads(1)
    torch.manual_seed(9)
    model = MaskedCSI()
    assert 50_000 <= sum(p.numel() for p in model.parameters()) <= 150_000
    x = torch.randn(2, 8, 3, 114, 10)
    mask = fixed_count_mask(16, torch.Generator().manual_seed(4))
    loss, _ = pretext_loss(model, x, mask)
    loss.backward()
    assert torch.isfinite(loss)
    for name in [
        "patch.weight",
        "mask_token",
        "encoder.layers.0.self_attn.in_proj_weight",
        "decoder.weight",
        "projector.0.weight",
    ]:
        gradient = dict(model.named_parameters())[name].grad
        assert (
            gradient is not None
            and torch.isfinite(gradient).all()
            and gradient.abs().sum() > 0
        )
    model.eval()
    z = model.history(x)
    assert z.shape == (2, 512)
    appended = torch.cat([x, torch.randn_like(x)], dim=0)
    torch.testing.assert_close(model.history(appended)[:2], z)


def test_partial_missing_values_use_frame_only_finite_statistics_and_never_targets():
    raw = np.arange(3 * 114 * 10, dtype=np.float32).reshape(1, 3, 114, 10)
    raw[0, 1, 17, 4] = np.nan
    raw[0, 2, 83, 7] = np.inf
    normalized, observed = normalize_frames(raw, return_observed=True)
    finite = raw[np.isfinite(raw)].astype(np.float64)
    np.testing.assert_allclose(
        normalized[observed],
        (finite - finite.mean()) / finite.std(),
        rtol=1e-6,
        atol=1e-7,
    )
    assert np.isfinite(normalized).all() and (normalized[~observed] == 0).all()
    assert observed.sum() == raw.size - 2
    # Force both missing cells into masked patches. Their imputed target values
    # cannot contribute to loss or gradients, regardless of predicted values.
    mask = torch.ones(1, 95, dtype=torch.bool)
    prediction = torch.ones(1, 95, 36, requires_grad=True)
    result = dict(
        target=torch.from_numpy(normalized).reshape(1, 3, 114, 10),
        reconstruction=prediction,
    )
    result["target"] = patchify(result["target"])
    loss = masked_reconstruction(result, mask, torch.from_numpy(observed))
    loss.backward()
    target_observed = patchify(torch.from_numpy(observed))
    assert (prediction.grad[~target_observed] == 0).all()
    assert prediction.grad[target_observed].abs().sum() > 0
    with torch.no_grad():
        prediction[~target_observed] = 1e9
    torch.testing.assert_close(
        masked_reconstruction(result, mask, torch.from_numpy(observed)), loss
    )


def test_encoder_only_history_equals_original_embedding_and_never_runs_decoder(
    monkeypatch,
):
    torch.set_num_threads(1)
    model = MaskedCSI().eval()
    histories = torch.randn(2, 8, 3, 114, 10)
    with torch.no_grad():
        expected = model(histories.flatten(0, 1))["embedding"].reshape(2, 512)

        def forbidden(*args, **kwargs):
            raise AssertionError(
                "Decoder must not run during frozen embedding inference"
            )

        monkeypatch.setattr(model.decoder, "forward", forbidden)
        torch.testing.assert_close(model.history(histories), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_weighted_masked_loss_matches_boolean_reference_values_and_gradients(dtype):
    torch.manual_seed(4)
    target = torch.randn(2, 95, 36, dtype=dtype)
    a = torch.randn_like(target, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    mask = fixed_count_mask(2, torch.Generator().manual_seed(5))
    observed = torch.rand(2, 3, 114, 10) > 0.03
    valid = mask[..., None] & patchify(observed)
    reference = (a - target).square()[valid].mean()
    actual = masked_reconstruction(
        dict(target=target, reconstruction=b), mask, observed
    )
    reference.backward()
    actual.backward()
    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
