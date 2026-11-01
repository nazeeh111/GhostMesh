"""Independent masked log-magnitude CSI representation, without pose labels.

Axes remain receiver/subcarrier/packet throughout patching. Packet timing is not
assumed calibrated. This is a research encoder, not a validated transfer model.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TOKENS = 95
PATCH_VALUES = 36
MASKED_TOKENS = 71  # nearest integer to 75 percent of 95, fixed in every arm


def normalize_frames(raw, *, return_observed=False):
    """Finite-only per-frame affine normalization, with no temporal statistics.

    Missing cells become zero (the frame's observed-value mean), but retain an
    explicit target-validity mask so reconstruction never learns fabricated
    target values. All-missing and constant observed frames are refused.
    """
    x = np.asarray(raw, dtype=np.float32)
    if x.ndim < 3 or x.shape[-3:] != (3, 114, 10):
        raise ValueError("Expected log-magnitude frames[...,3,114,10]")
    observed = np.isfinite(x)
    count = observed.sum(axis=(-3, -2, -1), keepdims=True)
    if (count == 0).any():
        raise ValueError("All-missing log-magnitude frame")
    finite = np.where(observed, x, 0).astype(np.float64)
    mean = finite.sum(axis=(-3, -2, -1), keepdims=True) / count
    centered = np.where(observed, finite - mean, 0)
    scale = np.sqrt(np.square(centered).sum(axis=(-3, -2, -1), keepdims=True) / count)
    if (scale < 1e-6).any():
        raise ValueError("Constant observed log-magnitude frame")
    normalized = np.asarray(centered / scale, dtype=np.float32)
    return (normalized, observed) if return_observed else normalized


def patchify(x):
    if x.ndim != 4 or x.shape[1:] != (3, 114, 10):
        raise ValueError("Expected frames[N,3,114,10]")
    return (
        x.reshape(-1, 3, 19, 6, 5, 2)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(-1, TOKENS, PATCH_VALUES)
    )


def unpatchify(tokens):
    if tokens.ndim != 3 or tokens.shape[1:] != (TOKENS, PATCH_VALUES):
        raise ValueError("Expected patches[N,95,36]")
    return (
        tokens.reshape(-1, 19, 5, 3, 6, 2)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(-1, 3, 114, 10)
    )


def fixed_count_mask(batch, generator):
    order = torch.rand(batch, TOKENS, generator=generator).argsort(dim=1)
    mask = torch.zeros(batch, TOKENS, dtype=torch.bool)
    mask.scatter_(1, order[:, :MASKED_TOKENS], True)
    return mask


class MaskedCSI(nn.Module):
    """Small ordered-patch encoder; full frames are used by frozen readouts."""

    def __init__(self):
        super().__init__()
        self.patch = nn.Linear(PATCH_VALUES, 48)
        self.position = nn.Parameter(torch.zeros(1, TOKENS, 48))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 48))
        layer = nn.TransformerEncoderLayer(
            48,
            4,
            128,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(48)
        self.embedding = nn.Linear(48, 64)
        self.decoder = nn.Linear(48, PATCH_VALUES)
        self.projector = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 64))
        self.pose_head = nn.Linear(8 * 64, 34)
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def encode_frames(self, frames, mask=None):
        patches = patchify(frames)
        tokens = self.patch(patches)
        if mask is not None:
            if mask.shape != patches.shape[:2] or mask.dtype != torch.bool:
                raise ValueError("Expected Boolean per-patch mask")
            tokens = torch.where(mask[..., None], self.mask_token, tokens)
        tokens = self.norm(self.encoder(tokens + self.position))
        return tokens

    def forward(self, frames, mask=None):
        tokens = self.encode_frames(frames, mask)
        return dict(
            embedding=self.embedding(tokens.mean(1)),
            reconstruction=self.decoder(tokens),
            target=patchify(frames),
        )

    def history(self, histories):
        if histories.ndim != 5 or histories.shape[1:] != (8, 3, 114, 10):
            raise ValueError("Expected causal histories[N,8,3,114,10]")
        tokens = self.encode_frames(histories.flatten(0, 1))
        return self.embedding(tokens.mean(1)).reshape(len(histories), 8 * 64)


def masked_reconstruction(result, mask, observed=None):
    if mask.shape != result["target"].shape[:2] or not mask.any():
        raise ValueError("Nonempty per-patch mask required")
    target_mask = mask[..., None].expand_as(result["target"])
    if observed is not None:
        if observed.shape != (len(mask), 3, 114, 10) or observed.dtype != torch.bool:
            raise ValueError("Boolean observed-cell frame mask required")
        target_mask = target_mask & patchify(observed)
    if not target_mask.any():
        raise ValueError("No observed masked reconstruction targets")
    return (
        (result["reconstruction"] - result["target"]).square() * target_mask
    ).sum() / target_mask.sum()


def temporal_regularizer(a, b):
    """Adjacent alignment and explicit variance/covariance anti-collapse terms."""
    if a.shape != b.shape or a.ndim != 2 or len(a) < 2:
        raise ValueError("At least two paired embeddings required")
    invariant = F.mse_loss(a, b)
    variance, covariance = [], []
    for z in (a, b):
        centered = z - z.mean(0)
        variance.append(F.relu(1 - torch.sqrt(z.var(0, unbiased=False) + 1e-4)).mean())
        matrix = centered.T @ centered / (len(z) - 1)
        off_diagonal = matrix - torch.diag_embed(matrix.diag())
        covariance.append(off_diagonal.square().sum() / z.shape[1])
    return dict(
        invariance=invariant, variance=sum(variance) / 2, covariance=sum(covariance) / 2
    )


def pretext_loss(
    model, histories, mask, *, shuffled=False, temporal=True, observed=None
):
    """No pose argument exists. Every arm reconstructs identical raw histories."""
    if histories.ndim != 5 or histories.shape[1:] != (8, 3, 114, 10):
        raise ValueError("Expected causal histories[N,8,3,114,10]")
    result = model(histories.flatten(0, 1), mask)
    reconstruction = masked_reconstruction(
        result, mask, None if observed is None else observed.flatten(0, 1)
    )
    projected = model.projector(result["embedding"]).reshape(-1, 8, 64)
    a, b = projected[:, :-1], projected[:, 1:]
    if shuffled:
        # Same seven partner observations, cyclic displacement four within the
        # SAME causal history. No additional RF or record/action label is read.
        b = b.roll(4, dims=1)
    terms = temporal_regularizer(a.flatten(0, 1), b.flatten(0, 1))
    regularizer = (
        25 * terms["invariance"] + 25 * terms["variance"] + terms["covariance"]
    )
    total = reconstruction + (0.05 * regularizer if temporal else 0)
    return total, dict(reconstruction=reconstruction, **terms)
