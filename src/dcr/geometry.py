"""Local disagreement-subspace geometry for DCR.

Inside every non-overlapping S×S tile the valid pixels form a vector space R^n.
The centred base predictions D = [p1-p̄, p2-p̄, p3-p̄] span a subspace of rank
r ≤ 2 with orthonormal basis Q, and P v = Q(Qᵀv). The dense n×n projector is
never built.

The basis is built once on the CPU (SVD) and passed as a frozen [B,T,S*S,2]
tensor; on any accelerator the projection is then two batched matmuls
(coef = Qᵀv, Pv = Q·coef) with no data-dependent branching.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import torch

MAX_RANK = 2


@dataclass(frozen=True)
class Geometry:
    q_basis: torch.Tensor  # [B,T,S*S,2]; padded columns and invalid rows are zero
    rank: torch.Tensor     # [B,T] long, values in {0,1,2}
    n_valid: torch.Tensor  # [B,T] long, values in 0..S*S
    block_size: int


def to_tiles(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """[B,C,H,W] -> [B,T,S*S,C], row-major tile and pixel order."""
    if x.ndim != 4:
        raise ValueError("expected [B,C,H,W]")
    b, c, h, w = x.shape
    s = block_size
    if h % s or w % s:
        raise ValueError("H and W must be divisible by block_size")
    return (x.reshape(b, c, h // s, s, w // s, s)
             .permute(0, 2, 4, 3, 5, 1).reshape(b, (h // s) * (w // s), s * s, c))


def from_tiles(x: torch.Tensor, height: int, width: int, block_size: int) -> torch.Tensor:
    """[B,T,S*S,C] -> [B,C,H,W]."""
    if x.ndim != 4:
        raise ValueError("expected [B,T,S*S,C]")
    b, t, area, c = x.shape
    s = block_size
    if height % s or width % s or t != (height // s) * (width // s) or area != s * s:
        raise ValueError("inconsistent tile shape")
    return (x.reshape(b, height // s, width // s, s, s, c)
             .permute(0, 5, 1, 3, 2, 4).reshape(b, c, height, width))


def _check_inputs(prob_maps: torch.Tensor, valid: torch.Tensor) -> None:
    if prob_maps.ndim != 4 or prob_maps.shape[1] != 3:
        raise ValueError("prob_maps must be [B,3,H,W]")
    if valid.shape != (prob_maps.shape[0], 1, *prob_maps.shape[-2:]):
        raise ValueError("valid must be [B,1,H,W]")
    if not bool(torch.isfinite(prob_maps).all()):
        raise ValueError("non-finite probability map")
    if bool(((prob_maps < 0) | (prob_maps > 1)).any()):
        raise ValueError("probability map outside [0,1]")


@torch.no_grad()
def make_geometry(prob_maps: torch.Tensor, valid: torch.Tensor, block_size: int = 32,
                  rel_tol: float = 1e-5, abs_tol: float = 1e-7) -> Geometry:
    """Per-tile orthonormal basis of the centred three-model differences."""
    _check_inputs(prob_maps, valid)
    p = to_tiles(prob_maps.detach().float().cpu(), block_size)          # [B,T,L,3]
    m = to_tiles(valid.detach().float().cpu(), block_size)[..., 0] > 0.5  # [B,T,L]
    b, t, area, _ = p.shape
    q = torch.zeros((b, t, area, MAX_RANK), dtype=torch.float32)
    rank = torch.zeros((b, t), dtype=torch.long)
    n_valid = m.sum(dim=-1).long()
    differences = (p - p.mean(dim=-1, keepdim=True)) * m.unsqueeze(-1)
    for bi in range(b):
        for ti in range(t):
            chosen = m[bi, ti]
            if not bool(chosen.any()):
                continue
            u, s, _ = torch.linalg.svd(differences[bi, ti, chosen], full_matrices=False)
            tol = max(abs_tol, rel_tol * float(s[0]))
            r = min(int((s > tol).sum()), MAX_RANK)
            rank[bi, ti] = r
            if r:
                cols = u[:, :r].clone()
                # Sign convention for reproducible caches; P itself is sign-invariant.
                for k in range(r):
                    if cols[int(cols[:, k].abs().argmax()), k] < 0:
                        cols[:, k] *= -1
                q[bi, ti, chosen, :r] = cols
    return Geometry(q, rank, n_valid, block_size)


def _stable_seed(*parts: object) -> int:
    """Process-independent seed (Python's hash() is salted per run)."""
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


@torch.no_grad()
def make_random_geometry(geometry: Geometry, valid: torch.Tensor, sample_ids: Sequence[str],
                         master_seed: int = 0, augmentation_id: str = "none") -> Geometry:
    """M06 control: fixed random subspace with the same per-tile rank.

    The basis depends only on (master_seed, sample_id, augmentation_id, tile)
    and never on the target, so it stays fixed across epochs.
    """
    s = geometry.block_size
    m = to_tiles(valid.detach().float().cpu(), s)[..., 0] > 0.5
    if len(sample_ids) != m.shape[0]:
        raise ValueError("one sample_id per batch element is required")
    q = torch.zeros_like(geometry.q_basis)
    for bi, sample_id in enumerate(sample_ids):
        for ti in range(m.shape[1]):
            r = int(geometry.rank[bi, ti])
            if r == 0:
                continue
            chosen = m[bi, ti]
            gen = torch.Generator().manual_seed(_stable_seed(master_seed, sample_id, augmentation_id, ti))
            z = torch.randn(int(chosen.sum()), r, generator=gen)
            q[bi, ti, chosen, :r] = torch.linalg.qr(z).Q
    return Geometry(q, geometry.rank.clone(), geometry.n_valid.clone(), s)


def project(v: torch.Tensor, geometry: Geometry, valid: torch.Tensor) -> torch.Tensor:
    """P v per tile as Q(Qᵀv) in FP32; v and valid are [B,1,H,W]."""
    if v.ndim != 4 or v.shape[1] != 1 or v.shape != valid.shape:
        raise ValueError("v and valid must be [B,1,H,W]")
    _, _, h, w = v.shape
    s = geometry.block_size
    vt = to_tiles(v.float() * valid.float(), s)[..., 0]
    q = geometry.q_basis.to(device=v.device, dtype=torch.float32)
    if q.shape[:3] != vt.shape:
        raise ValueError("geometry and v shapes disagree")
    coefficients = torch.einsum("btli,btl->bti", q, vt)
    projected = torch.einsum("btli,bti->btl", q, coefficients)
    return from_tiles(projected.unsqueeze(-1), h, w, s) * valid.float()


def split_correction(a: torch.Tensor, b: torch.Tensor, geometry: Geometry,
                     valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (P a, (I-P) b)."""
    pa = project(a, geometry, valid)
    return pa, b.float() * valid.float() - project(b, geometry, valid)


def make_prediction(prob_maps: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                    geometry: Geometry, valid: torch.Tensor, beta: float = 1.0):
    """q = clip(p̄ + β[Pa + (I-P)b]); returns (q, Pa, (I-P)b)."""
    p_mean = prob_maps.detach().float().mean(dim=1, keepdim=True)
    pa, pb_perp = split_correction(a, b, geometry, valid)
    q = (p_mean + float(beta) * (pa + pb_perp)).clamp(0.0, 1.0)
    return q, pa, pb_perp


@torch.no_grad()
def capture_stats(error: torch.Tensor, geometry: Geometry, valid: torch.Tensor,
                  eps: float = 1e-12) -> dict:
    """Error-energy capture η = ||Pe||²/||e||² and κ = η / (r/n) per tile.

    Tiles with n=0, r=0 or ||e||=0 are NaN and reported as separate counts.
    """
    s = geometry.block_size
    pe = project(error, geometry, valid)
    e2 = to_tiles((error.float() * valid.float()).square(), s)[..., 0].sum(-1).cpu()
    pe2 = to_tiles(pe.square(), s)[..., 0].sum(-1).cpu()
    r = geometry.rank.float()
    n = geometry.n_valid.float()
    defined = (r > 0) & (n > 0) & (e2 > eps)
    nan = torch.tensor(float("nan"))
    eta = torch.where(defined, pe2 / e2.clamp_min(eps), nan)
    kappa = torch.where(defined, eta / (r / n.clamp_min(1)), nan)
    return {
        "eta": eta, "kappa": kappa,
        "n_defined": int(defined.sum()),
        "n_empty_tiles": int((n == 0).sum()),
        "n_rank0_tiles": int(((n > 0) & (r == 0)).sum()),
        "n_zero_error_tiles": int(((n > 0) & (r > 0) & (e2 <= eps)).sum()),
    }
