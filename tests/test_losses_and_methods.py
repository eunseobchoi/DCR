import pytest
import torch

from dcr.geometry import make_geometry
from dcr.losses import dcr_loss, keep_loss, keep_mask
from dcr.methods import METHOD_NAMES, LossWeights, RefinementMethod
from dcr.models.refiner import count_parameters


def _batch(size=32, b=2, seed=0):
    torch.manual_seed(seed)
    return (torch.randn(b, 3, size, size), torch.rand(b, 3, size, size),
            (torch.rand(b, 1, size, size) > 0.7).float(), torch.ones(b, 1, size, size))


def test_keep_excludes_confident_wrong_pixels():
    pbar = torch.full((1, 1, 4, 4), 0.95)
    assert not keep_mask(pbar, torch.zeros(1, 1, 4, 4), torch.ones(1, 1, 4, 4)).any()
    assert keep_mask(pbar, torch.ones(1, 1, 4, 4), torch.ones(1, 1, 4, 4)).all()
    q = torch.full((1, 1, 4, 4), 0.5)
    assert float(keep_loss(q, pbar, torch.zeros(1, 1, 4, 4), torch.ones(1, 1, 4, 4))) == 0.0
    assert float(keep_loss(q, pbar, torch.ones(1, 1, 4, 4), torch.ones(1, 1, 4, 4))) > 0.0


def test_rank0_everywhere_gives_finite_loss():
    p, v = torch.full((1, 3, 8, 8), 0.3), torch.ones(1, 1, 8, 8)
    g = make_geometry(p, v, 4)
    a = torch.randn(1, 1, 8, 8, requires_grad=True)
    b = torch.randn(1, 1, 8, 8, requires_grad=True)
    terms = dcr_loss(p, torch.ones(1, 1, 8, 8), v, a, b, g, 1.0, 1.0, 1.0)
    assert float(terms["parallel"]) == 0.0 and torch.isfinite(terms["total"])
    terms["total"].backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()


def test_negative_image_uses_bce_only():
    p, v = torch.rand(1, 3, 8, 8), torch.ones(1, 1, 8, 8)
    g = make_geometry(p, v, 4)
    z = torch.zeros(1, 1, 8, 8)
    terms = dcr_loss(p, z, v, z, z, g, 0.1, 0.1, 0.01)
    assert float(terms["soft_dice"]) == 0.0 and float(terms["bce"]) > 0.0


@pytest.mark.parametrize("method", list(METHOD_NAMES))
def test_zero_init_returns_pbar_for_every_method(method):
    image, probs, _, valid = _batch()
    model = RefinementMethod(method, block_size=16)
    g = model.geometry(probs, valid, ["a", "b"])
    out = model(image, probs, valid, g)
    assert torch.allclose(out["q"], probs.mean(1, keepdim=True).clamp(0, 1), atol=1e-6)


@pytest.mark.parametrize("method", list(METHOD_NAMES))
def test_gradients_reach_heads_not_base_probs(method):
    image, probs, target, valid = _batch()
    probs = probs.requires_grad_(True)
    model = RefinementMethod(method, block_size=16)
    g = model.geometry(probs, valid, ["a", "b"])
    out = model(image, probs, valid, g)
    model.loss(out, target, valid, g, LossWeights())["total"].backward()
    assert probs.grad is None
    for head in model.net.heads:
        assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
        assert head.weight.grad.abs().sum() > 0


def test_m05_and_m07_have_identical_networks():
    assert count_parameters(RefinementMethod("M05")) == count_parameters(RefinementMethod("M07"))
    assert count_parameters(RefinementMethod("M06")) == count_parameters(RefinementMethod("M07"))


def test_zero_rgb_ablation_keeps_parameter_count():
    assert count_parameters(RefinementMethod("M07", zero_rgb=True)) == count_parameters(RefinementMethod("M07"))


def test_refiner_512_forward_shape():
    image, probs, _, valid = torch.randn(1, 3, 512, 512), torch.rand(1, 3, 512, 512), None, torch.ones(1, 1, 512, 512)
    model = RefinementMethod("M07")
    with torch.no_grad():
        out = model(image, probs, valid, model.geometry(probs, valid, ["x"]))
    assert out["q"].shape == (1, 1, 512, 512) and out["heads"].shape == (1, 2, 512, 512)
