"""Refiner U-Net: 6-channel input [RGB, p1, p2, p3] -> K raw correction heads.

Default widths 24-48-96-192 give 1,097,282 parameters for the two-head (M05/M07) model.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class DoubleConv(nn.Sequential):
    """(Conv3x3 -> GroupNorm -> ReLU) x 2."""

    def __init__(self, in_ch: int, out_ch: int, groups: int = 8) -> None:
        g = min(groups, out_ch)
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.GroupNorm(g, out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.GroupNorm(g, out_ch), nn.ReLU(inplace=True),
        )


class RefinerUNet(nn.Module):
    """Shared encoder/decoder with `n_heads` zero-initialised 1-channel 1x1 heads (no activation)."""

    def __init__(self, in_channels: int = 6, n_heads: int = 2,
                 widths: Sequence[int] = (24, 48, 96, 192)) -> None:
        super().__init__()
        if len(widths) != 4:
            raise ValueError("widths must have 4 levels")
        self.inc = DoubleConv(in_channels, widths[0])
        self.downs = nn.ModuleList(DoubleConv(widths[i], widths[i + 1]) for i in range(3))
        self.ups = nn.ModuleList(DoubleConv(widths[i + 1] + widths[i], widths[i]) for i in reversed(range(3)))
        self.heads = nn.ModuleList(nn.Conv2d(widths[0], 1, 1) for _ in range(n_heads))
        for head in self.heads:  # step 0 predicts exactly p̄
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw signed corrections [B, n_heads, H, W]."""
        h = self.inc(x)
        skips = [h]
        for down in self.downs:
            h = down(F.max_pool2d(h, 2))
            skips.append(h)
        skips.pop()
        for up in self.ups:
            skip = skips.pop()
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up(torch.cat([h, skip], dim=1))
        return torch.cat([head(h) for head in self.heads], dim=1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
