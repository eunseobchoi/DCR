"""DCR training losses.

L = Lseg(q,y) + λ1/r ||Pa − Pe||² + λ2/(n−r) ||(I−P)b − (I−P)e||² + μ Lkeep
with Lseg = BCE + soft Dice on valid pixels. Reductions are recorded as
`equal_mean_of_eligible_tiles` in the config.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .geometry import Geometry, make_prediction, project, to_tiles


def segmentation_loss(q: torch.Tensor, y: torch.Tensor, valid: torch.Tensor,
                      eps: float = 1e-6) -> dict[str, torch.Tensor]:
    """BCE on all valid pixels; soft Dice on positive images only."""
    valid = valid.float()
    y = y.float()
    n = valid.sum().clamp_min(1.0)
    bce = (F.binary_cross_entropy(q.clamp(eps, 1 - eps), y, reduction="none") * valid).sum() / n
    dims = (1, 2, 3)
    intersection = (q * y * valid).sum(dim=dims)
    qsum = (q * valid).sum(dim=dims)
    ysum = (y * valid).sum(dim=dims)
    positive = (ysum > 0).float()
    dice_loss = 1 - (2 * intersection + eps) / (qsum + ysum + eps)
    soft_dice = (dice_loss * positive).sum() / positive.sum().clamp_min(1)
    return {"seg": bce + soft_dice, "bce": bce, "soft_dice": soft_dice}


def keep_mask(pbar: torch.Tensor, y: torch.Tensor, valid: torch.Tensor,
              low: float = 0.1, high: float = 0.9) -> torch.Tensor:
    """Pixels where p̄ is confident AND correct. Confidently wrong pixels are excluded."""
    confident = (pbar <= low) | (pbar >= high)
    correct = (pbar >= 0.5) == (y >= 0.5)
    return confident & correct & (valid > 0.5)


def keep_loss(q: torch.Tensor, pbar: torch.Tensor, y: torch.Tensor, valid: torch.Tensor,
              low: float = 0.1, high: float = 0.9) -> torch.Tensor:
    mask = keep_mask(pbar, y, valid, low, high).float()
    return ((q - pbar).square() * mask).sum() / mask.sum().clamp_min(1)


def _mean_over_eligible(patch_sum: torch.Tensor, denominator: torch.Tensor,
                        eligible: torch.Tensor) -> torch.Tensor:
    per_patch = patch_sum / denominator.clamp_min(1).float()
    eligible_f = eligible.float()
    return (per_patch * eligible_f).sum() / eligible_f.sum().clamp_min(1.0)


def component_losses(pa: torch.Tensor | None, pb_perp: torch.Tensor | None, error: torch.Tensor,
                     geometry: Geometry, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Parallel term (rank>0 tiles, /r) and perpendicular term (n−r>0 tiles, /(n−r))."""
    s = geometry.block_size
    zero = error.new_zeros(())
    pe = project(error, geometry, valid)
    rank = geometry.rank.to(error.device)
    n_valid = geometry.n_valid.to(error.device)
    loss_para = loss_perp = zero
    if pa is not None:
        patch = to_tiles((pa - pe).square() * valid, s)[..., 0].sum(-1)
        loss_para = _mean_over_eligible(patch, rank, (rank > 0) & (n_valid > 0))
    if pb_perp is not None:
        e_perp = error * valid - pe
        patch = to_tiles((pb_perp - e_perp).square() * valid, s)[..., 0].sum(-1)
        loss_perp = _mean_over_eligible(patch, n_valid - rank, (n_valid - rank) > 0)
    return loss_para, loss_perp


def residual_mse(correction: torch.Tensor, error: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """M05 auxiliary loss: masked MSE between the unconstrained correction and y − p̄."""
    valid = valid.float()
    return ((correction - error).square() * valid).sum() / valid.sum().clamp_min(1)


def dcr_loss(prob_maps: torch.Tensor, y: torch.Tensor, valid: torch.Tensor,
             a: torch.Tensor, b: torch.Tensor, geometry: Geometry,
             lambda_parallel: float, lambda_perp: float, mu_keep: float,
             beta: float = 1.0, eps: float = 1e-6) -> dict[str, torch.Tensor]:
    """Full DCR objective (reference form)."""
    if not (y.shape == valid.shape == a.shape == b.shape):
        raise ValueError("y, valid, a, b must all be [B,1,H,W]")
    pbar = prob_maps.detach().float().mean(dim=1, keepdim=True)
    error = ((y.float() - pbar) * valid.float()).detach()
    q, pa, pb_perp = make_prediction(prob_maps, a, b, geometry, valid, beta)
    loss_para, loss_perp = component_losses(pa, pb_perp, error, geometry, valid)
    seg = segmentation_loss(q, y, valid, eps)
    loss_keep = keep_loss(q, pbar, y, valid)
    total = seg["seg"] + lambda_parallel * loss_para + lambda_perp * loss_perp + mu_keep * loss_keep
    return dict(total=total, **seg, parallel=loss_para, perpendicular=loss_perp, keep=loss_keep, q=q)
