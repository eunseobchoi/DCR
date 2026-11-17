import pytest
import torch

from dcr.models.base import BASE_ARCHITECTURES, build_base_model, predict_probability, predict_triplet


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(0)
    return [build_base_model(name, pretrained=False) for name in BASE_ARCHITECTURES]


def test_logits_shape_and_probability_range(models):
    image = torch.randn(2, 3, 128, 128)
    for model in models:
        model.eval()
        with torch.no_grad():
            assert model(image).shape == (2, 1, 128, 128)
        prob = predict_probability(model, image)
        assert torch.isfinite(prob).all() and prob.min() >= 0 and prob.max() <= 1


def test_triplet_order_and_shape(models):
    probs = predict_triplet(models, torch.randn(1, 3, 64, 64))
    assert probs.shape == (1, 3, 64, 64)


def test_unknown_architecture_rejected():
    with pytest.raises(ValueError):
        build_base_model("resnet18_unet")
