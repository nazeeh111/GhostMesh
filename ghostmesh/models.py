"""Original compact CSI encoders. No weights or code from reference models."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FrameEncoder(nn.Module):
    """Shared encoder: antennas/links are channels, subcarriers/packets are axes."""

    def __init__(self, channels: int, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, width // 2, (7, 3), stride=(2, 1), padding=(3, 1)),
            nn.GroupNorm(4, width // 2),
            nn.SiLU(),
            nn.Conv2d(width // 2, width, (5, 3), stride=(2, 1), padding=(2, 1)),
            nn.GroupNorm(4, width),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 2)),
            nn.Flatten(),
            nn.Linear(width * 8, width),
            nn.LayerNorm(width),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PacketMomentEncoder(nn.Module):
    """Learn antenna/subcarrier structure while marginalizing packet order.

    MM-Fi supplies short blocks without original packet timestamps. Packet
    mean/deviation retain RF variation without inventing within-block timing.
    Pooling preserves coarse subcarrier positions; antennas remain distinct.
    """

    def __init__(self, channels: int, width: int, bins: int = 32):
        super().__init__()
        if bins < 1:
            raise ValueError("Subcarrier bins must be positive")
        self.bins = bins
        self.net = nn.Sequential(
            nn.Linear(channels * 2 * bins, width * 2),
            nn.LayerNorm(width * 2),
            nn.SiLU(),
            nn.Linear(width * 2, width),
            nn.LayerNorm(width),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1)
        deviation = (x.var(dim=-1, unbiased=False) + 1e-6).sqrt()
        features = F.adaptive_avg_pool1d(torch.cat((mean, deviation), dim=1), self.bins)
        return self.net(features.flatten(1))


class CausalBlock(nn.Module):
    """Left padding makes every timestep independent of future frames."""

    def __init__(self, width: int, dilation: int, kernel: str = "native"):
        super().__init__()
        if dilation < 1 or kernel not in {"native", "direct"}:
            raise ValueError("Positive dilation and native/direct kernel required")
        self.kernel = kernel
        self.padding = 2 * dilation
        self.depthwise = nn.Conv1d(width, width, 3, dilation=dilation, groups=width)
        self.mix = nn.Conv1d(width, width, 1)
        # LayerNorm acts across features within one timestep, never across time.
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel == "native":
            y = self.mix(F.silu(self.depthwise(F.pad(x, (self.padding, 0)))))
        else:
            # Same three-tap convolution and state-dict schema, with less tiny
            # grouped-convolution dispatch overhead on CPU. Verified against
            # native forward values, gradients, causality and checkpoint reload.
            length, dilation = x.shape[-1], self.depthwise.dilation[0]
            padded = F.pad(x, (2 * dilation, 0))
            weights = self.depthwise.weight[:, 0, :]
            y = (
                padded[..., :length] * weights[:, 0][None, :, None]
                + padded[..., dilation : dilation + length]
                * weights[:, 1][None, :, None]
                + padded[..., 2 * dilation : 2 * dilation + length]
                * weights[:, 2][None, :, None]
            )
            y = y + self.depthwise.bias[None, :, None]
            y = F.linear(
                F.silu(y).transpose(1, 2), self.mix.weight[:, :, 0], self.mix.bias
            ).transpose(1, 2)
        return x + F.silu(self.norm(y.transpose(1, 2))).transpose(1, 2)


class CSIPose(nn.Module):
    """Temporal model or last-frame baseline, with isotropic joint uncertainty.

    ``log_sigma`` is the log of coordinate standard deviation in normalized
    image coordinates. It is not a calibrated probability of being correct.
    """

    def __init__(
        self,
        channels: int = 3,
        joints: int = 17,
        width: int = 64,
        temporal: bool = True,
        dilations: tuple[int, ...] = (1, 2, 4),
        encoder: str = "cnn",
        bins: int = 32,
        temporal_kernel: str = "native",
    ):
        super().__init__()
        if channels < 1 or joints < 1 or width < 8 or width % 8:
            raise ValueError(
                "channels/joints must be positive; width must be a multiple of 8"
            )
        self.channels, self.joints, self.temporal = channels, joints, temporal
        if encoder not in {"cnn", "packet_moment"}:
            raise ValueError("Encoder must be cnn or packet_moment")
        self.encoder = (
            FrameEncoder(channels, width)
            if encoder == "cnn"
            else PacketMomentEncoder(channels, width, bins)
        )
        self.temporal_net = (
            nn.Sequential(*(CausalBlock(width, d, temporal_kernel) for d in dilations))
            if temporal
            else nn.Identity()
        )
        self.head = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, joints * 3)
        )

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[2] != self.channels or x.shape[1] < 1:
            raise ValueError(
                f"Expected [B,T,{self.channels},S,P], got {tuple(x.shape)}"
            )
        b, t, c, s, p = x.shape
        z = self.encoder(x.reshape(b * t, c, s, p)).reshape(b, t, -1)
        return self.temporal_net(z.transpose(1, 2)).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if not self.temporal:
            x = x[:, -1:]
        raw = self.head(self.encode_sequence(x)[:, -1]).reshape(-1, self.joints, 3)
        return {
            "pose": raw[..., :2].sigmoid(),
            "log_sigma": raw[..., 2].clamp(-6.0, 1.0),
        }


def build_model(config: dict) -> CSIPose:
    model = config.get("model", config)
    return CSIPose(
        channels=int(model.get("channels", 3)),
        joints=int(model.get("joints", 17)),
        width=int(model.get("width", 64)),
        temporal=bool(model.get("temporal", True)),
        dilations=tuple(model.get("dilations", [1, 2, 4])),
        encoder=model.get("encoder", "cnn"),
        bins=int(model.get("bins", 32)),
        temporal_kernel=model.get("temporal_kernel", "native"),
    )


def supervised_loss(
    prediction: dict[str, torch.Tensor], target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Masked 2D Gaussian negative log likelihood, up to a constant."""
    mask = valid.bool() & torch.isfinite(target).all(-1)
    if not mask.any():
        raise ValueError("Batch has no finite, valid supervision")
    error2 = (prediction["pose"][mask] - target[mask]).square().sum(-1)
    log_sigma = prediction["log_sigma"][mask]
    return (0.5 * error2 * torch.exp(-2 * log_sigma) + 2 * log_sigma).mean()


def radius_confidence(log_sigma: torch.Tensor, radius: float = 0.05) -> torch.Tensor:
    """Gaussian mass within an explicit image-coordinate radius (uncalibrated)."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    return -torch.expm1(-0.5 * radius**2 * torch.exp(-2 * log_sigma))
