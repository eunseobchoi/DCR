"""The seven refinement conditions M01–M07 behind one interface.

All methods read the same 6-channel input and the same frozen base triplet.
M05 (primary comparator) and M07 (DCR) share an identical two-head network.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from .geometry import Geometry, make_geometry, make_random_geometry, project, split_correction
from .losses import component_losses, keep_loss, residual_mse, segmentation_loss
from .models.refiner import RefinerUNet

METHOD_NAMES = {
    "M01": "pixel-wise weighted fusion",
    "M02": "disagreement-space correction only",
    "M03": "complement-space correction only",
    "M04": "generic residual correction",
    "M05": "unrestricted two-head correction (primary comparator)",
    "M06": "same-rank random-space correction",
    "M07": "DCR",
}
_N_HEADS = {"M01": 3, "M02": 1, "M03": 1, "M04": 1, "M05": 2, "M06": 2, "M07": 2}
_NEEDS_GEOMETRY = {"M02", "M03", "M06", "M07"}


@dataclass(frozen=True)
class LossWeights:
    """Default loss weights; tune them on the development split (C4 dev_select)."""
    lambda_parallel: float = 0.1
    lambda_perp: float = 0.1
    lambda_residual: float = 0.1
    mu_keep: float = 0.01


class RefinementMethod(nn.Module):
    def __init__(self, method_id: str, widths: Sequence[int] = (24, 48, 96, 192),
                 block_size: int = 32, random_master_seed: int = 0, zero_rgb: bool = False) -> None:
        super().__init__()
        if method_id not in METHOD_NAMES:
            raise ValueError(f"unknown method {method_id!r}")
        self.method_id = method_id
        self.block_size = block_size
        self.random_master_seed = random_master_seed
        self.zero_rgb = zero_rgb  # ablation A03: keep 6 channels, feed zeros for RGB
        self.net = RefinerUNet(in_channels=6, n_heads=_N_HEADS[method_id], widths=widths)

    @torch.no_grad()
    def geometry(self, probs: torch.Tensor, valid: torch.Tensor,
                 sample_ids: Sequence[str]) -> Geometry | None:
        if self.method_id not in _NEEDS_GEOMETRY:
            return None
        g = make_geometry(probs, valid, self.block_size)
        if self.method_id == "M06":
            g = make_random_geometry(g, valid, sample_ids, self.random_master_seed)
        return g

    def forward(self, image: torch.Tensor, probs: torch.Tensor, valid: torch.Tensor,
                geometry: Geometry | None = None, beta: float = 1.0) -> dict[str, torch.Tensor]:
        if self.method_id in _NEEDS_GEOMETRY and geometry is None:
            raise ValueError(f"{self.method_id} needs a geometry")
        probs = probs.detach().float()  # base models are frozen
        rgb = torch.zeros_like(image) if self.zero_rgb else image
        heads = self.net(torch.cat([rgb, probs], dim=1))
        pbar = probs.mean(dim=1, keepdim=True)
        valid_f = valid.float()
        out: dict[str, torch.Tensor] = {"pbar": pbar, "heads": heads}
        m = self.method_id
        if m == "M01":
            fused = (torch.softmax(heads, dim=1) * probs).sum(dim=1, keepdim=True)
            correction = (fused - pbar) * valid_f
        elif m == "M02":
            out["pa"] = correction = project(heads[:, :1], geometry, valid)
        elif m == "M03":
            b = heads[:, :1]
            out["pb_perp"] = correction = b * valid_f - project(b, geometry, valid)
        elif m in ("M04", "M05"):
            correction = heads.sum(dim=1, keepdim=True) * valid_f
        else:  # M06, M07
            out["pa"], out["pb_perp"] = split_correction(heads[:, :1], heads[:, 1:2], geometry, valid)
            correction = out["pa"] + out["pb_perp"]
        out["correction"] = correction
        out["q"] = (pbar + float(beta) * correction).clamp(0.0, 1.0)
        return out

    def loss(self, out: dict, target: torch.Tensor, valid: torch.Tensor,
             geometry: Geometry | None, weights: LossWeights) -> dict[str, torch.Tensor]:
        q, pbar = out["q"], out["pbar"]
        seg = segmentation_loss(q, target, valid)
        keep = keep_loss(q, pbar, target, valid)
        error = ((target.float() - pbar) * valid.float()).detach()
        para = perp = resid = q.new_zeros(())
        if self.method_id in _NEEDS_GEOMETRY:
            para, perp = component_losses(out.get("pa"), out.get("pb_perp"), error, geometry, valid)
        if self.method_id == "M05":
            resid = residual_mse(out["correction"], error, valid)
        total = (seg["seg"] + weights.lambda_parallel * para + weights.lambda_perp * perp
                 + weights.lambda_residual * resid + weights.mu_keep * keep)
        return dict(total=total, **seg, parallel=para, perpendicular=perp, residual=resid, keep=keep)
