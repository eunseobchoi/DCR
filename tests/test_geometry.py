import pytest
import torch

from dcr.geometry import (capture_stats, from_tiles, make_geometry, make_prediction, make_random_geometry,
                          project, to_tiles)
from dcr.losses import dcr_loss


def test_tile_roundtrip():
    x = torch.arange(2 * 3 * 8 * 12.).reshape(2, 3, 8, 12)
    assert torch.equal(from_tiles(to_tiles(x, 4), 8, 12, 4), x)


def test_rank2_orthonormal_and_pythagoras():
    torch.manual_seed(1)
    p, v = torch.rand(2, 3, 8, 8), torch.ones(2, 1, 8, 8)
    g = make_geometry(p, v, 4)
    assert torch.all(g.rank == 2)
    for b in range(2):
        for t in range(4):
            q = g.q_basis[b, t]
            assert torch.allclose(q.T @ q, torch.eye(2), atol=1e-5)
    e = torch.randn(2, 1, 8, 8)
    pe = project(e, g, v)
    ep = e - pe
    assert torch.allclose(project(pe, g, v), pe, atol=1e-5)
    assert torch.allclose(e.square().sum(), pe.square().sum() + ep.square().sum(), atol=2e-5)
    assert torch.allclose(project(ep, g, v), torch.zeros_like(pe), atol=1e-5)


def test_centred_differences_sum_to_zero():
    p = torch.rand(1, 3, 8, 8)
    d = p - p.mean(dim=1, keepdim=True)
    assert torch.allclose(d.sum(dim=1), torch.zeros(1, 8, 8), atol=1e-6)


def test_rank0_rank1_and_empty_tiles():
    p, v = torch.full((1, 3, 8, 8), 0.3), torch.ones(1, 1, 8, 8)
    g = make_geometry(p, v, 4)
    assert int(g.rank.max()) == 0
    x = torch.randn(1, 1, 8, 8)
    assert torch.equal(project(x, g, v), torch.zeros_like(x))
    p[:, 0] = torch.linspace(0.1, 0.6, 64).reshape(8, 8)
    p[:, 1] = 0.3
    p[:, 2] = 0.6 - p[:, 0]
    g = make_geometry(p, v, 4)
    assert int(g.rank.min()) == 1 and int(g.rank.max()) == 1
    v[:, :, :, 4:] = 0
    g = make_geometry(p, v, 4)
    assert int(g.rank[:, [1, 3]].max()) == 0 and int(g.n_valid[:, [1, 3]].max()) == 0


def test_beta0_and_gradients():
    p, v = torch.rand(1, 3, 8, 8), torch.ones(1, 1, 8, 8)
    g = make_geometry(p, v, 4)
    a = torch.randn(1, 1, 8, 8, requires_grad=True)
    b = torch.randn(1, 1, 8, 8, requires_grad=True)
    q0, _, _ = make_prediction(p, a, b, g, v, beta=0)
    assert torch.allclose(q0, p.mean(1, keepdim=True))
    y = torch.randint(0, 2, (1, 1, 8, 8)).float()
    dcr_loss(p, y, v, a, b, g, 0.1, 0.1, 0.01)["total"].backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()


def test_hand_computed_rank1_example():
    # one 2x2 tile = four valid pixels in row-major order
    p = torch.tensor([[0.1, 0.8, 0.5, 0.4], [0.3, 0.6, 0.5, 0.4], [0.5, 0.4, 0.5, 0.4]]).reshape(1, 3, 2, 2)
    y = torch.tensor([1.0, 0.0, 1.0, 0.0]).reshape(1, 1, 2, 2)
    v = torch.ones(1, 1, 2, 2)
    g = make_geometry(p, v, 2)
    assert int(g.rank) == 1
    e = y - p.mean(1, keepdim=True)
    pe = project(e, g, v)
    assert torch.allclose(pe.flatten(), torch.tensor([0.65, -0.65, 0.0, 0.0]), atol=1e-5)
    assert torch.allclose((e - pe).flatten(), torch.tensor([0.05, 0.05, 0.5, -0.4]), atol=1e-5)
    q, _, _ = make_prediction(p, e, e, g, v, beta=1.0)  # ideal a=b=e recovers y
    assert torch.allclose(q, y, atol=1e-5)
    q_half, _, _ = make_prediction(p, e, e, g, v, beta=0.5)
    assert torch.allclose(q_half.flatten(), torch.tensor([0.65, 0.30, 0.75, 0.20]), atol=1e-5)


def test_invalid_pixels_do_not_change_valid_output():
    torch.manual_seed(0)
    p, v = torch.rand(1, 3, 8, 8), torch.ones(1, 1, 8, 8)
    v[..., 6:] = 0
    x = torch.randn(1, 1, 8, 8)
    p2, x2 = p.clone(), x.clone()
    p2[..., 6:] = torch.rand(1, 3, 8, 2)
    x2[..., 6:] = 100.0
    out1 = project(x, make_geometry(p, v, 4), v)
    out2 = project(x2, make_geometry(p2, v, 4), v)
    assert torch.allclose(out1, out2, atol=1e-6)
    assert torch.equal(out1[..., 6:], torch.zeros(1, 1, 8, 2))


@pytest.mark.parametrize("block", [16, 32, 64])
def test_tile_sizes(block):
    p, v = torch.rand(1, 3, 64, 64), torch.ones(1, 1, 64, 64)
    g = make_geometry(p, v, block)
    assert g.q_basis.shape == (1, (64 // block) ** 2, block * block, 2)
    x = torch.randn(1, 1, 64, 64)
    assert torch.allclose(project(project(x, g, v), g, v), project(x, g, v), atol=1e-5)


def test_512_shape():
    p, v = torch.rand(1, 3, 512, 512), torch.ones(1, 1, 512, 512)
    g = make_geometry(p, v, 32)
    assert g.q_basis.shape == (1, 256, 1024, 2) and g.rank.shape == (1, 256)


def test_random_geometry_matches_rank_and_is_deterministic():
    torch.manual_seed(3)
    p, v = torch.rand(1, 3, 16, 16), torch.ones(1, 1, 16, 16)
    p[:, :, :8, :8] = 0.4  # one rank-0 tile
    g = make_geometry(p, v, 8)
    r1 = make_random_geometry(g, v, ["s0"], master_seed=7)
    r2 = make_random_geometry(g, v, ["s0"], master_seed=7)
    assert torch.equal(r1.rank, g.rank) and torch.equal(r1.q_basis, r2.q_basis)
    assert int(g.rank[0, 0]) == 0 and torch.equal(r1.q_basis[0, 0], torch.zeros(64, 2))
    for t in range(1, 4):
        q = r1.q_basis[0, t]
        assert torch.allclose(q.T @ q, torch.eye(2), atol=1e-5)
    assert not torch.allclose(r1.q_basis, g.q_basis)
    assert not torch.equal(make_random_geometry(g, v, ["s1"], 7).q_basis, r1.q_basis)


def test_capture_stats_excludes_undefined_tiles():
    p, v = torch.rand(1, 3, 16, 16), torch.ones(1, 1, 16, 16)
    p[:, :, :8, :8] = 0.4
    v[..., 8:, 8:] = 0
    g = make_geometry(p, v, 8)
    stats = capture_stats(torch.randn(1, 1, 16, 16), g, v)
    assert stats["n_rank0_tiles"] == 1 and stats["n_empty_tiles"] == 1 and stats["n_defined"] == 2
    defined = stats["eta"][~torch.isnan(stats["eta"])]
    assert ((defined >= 0) & (defined <= 1 + 1e-6)).all()
