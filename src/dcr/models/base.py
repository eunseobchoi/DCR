"""Three frozen base segmentation models. Each returns raw logits [B,1,H,W].

Pretrained loading uses ImageNet *encoder* weights only (no segmentation
checkpoints); it downloads weights and is never exercised in tests.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models as tvm
from torchvision.models.segmentation import deeplabv3_resnet50

from .refiner import DoubleConv

BASE_ARCHITECTURES = ("unet_r34", "deeplab_r50", "segformer_b0")  # p1, p2, p3 order


class _Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None, size=None) -> torch.Tensor:
        size = skip.shape[-2:] if skip is not None else size
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return self.conv(x if skip is None else torch.cat([x, skip], dim=1))


class ResNet34UNet(nn.Module):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        r = tvm.resnet34(weights=tvm.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)
        self.pool = r.maxpool
        self.layers = nn.ModuleList([r.layer1, r.layer2, r.layer3, r.layer4])
        self.up4, self.up3 = _Up(512, 256, 256), _Up(256, 128, 128)
        self.up2, self.up1 = _Up(128, 64, 64), _Up(64, 64, 32)
        self.up0 = _Up(32, 0, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)
        f1 = self.layers[0](self.pool(s0))
        f2 = self.layers[1](f1)
        f3 = self.layers[2](f2)
        f4 = self.layers[3](f3)
        d = self.up1(self.up2(self.up3(self.up4(f4, f3), f2), f1), s0)
        return self.head(self.up0(d, size=x.shape[-2:]))


class DeepLabV3R50(nn.Module):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        backbone = tvm.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        # torchvision defaults weights_backbone to ImageNet, so pass it explicitly.
        self.net = deeplabv3_resnet50(weights=None, weights_backbone=backbone, num_classes=1, aux_loss=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)["out"]


class SegFormerB0(nn.Module):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        if pretrained:
            self.net = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/mit-b0", num_labels=1, ignore_mismatched_sizes=True)
        else:
            config = SegformerConfig(num_labels=1, hidden_sizes=[32, 64, 160, 256],
                                     depths=[2, 2, 2, 2], decoder_hidden_size=256)
            self.net = SegformerForSemanticSegmentation(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(pixel_values=x).logits  # H/4 x W/4
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


_BUILDERS = {"unet_r34": ResNet34UNet, "deeplab_r50": DeepLabV3R50, "segformer_b0": SegFormerB0}


def build_base_model(name: str, pretrained: bool = False) -> nn.Module:
    if name not in _BUILDERS:
        raise ValueError(f"unknown base architecture {name!r}; expected one of {BASE_ARCHITECTURES}")
    return _BUILDERS[name](pretrained=pretrained)


@torch.no_grad()
def predict_probability(model: nn.Module, image: torch.Tensor) -> torch.Tensor:
    """Sigmoid applied exactly once to the logits of a frozen model."""
    model.eval()
    logits = model(image)
    if logits.shape != (image.shape[0], 1, *image.shape[-2:]):
        raise ValueError(f"unexpected logits shape {tuple(logits.shape)}")
    return torch.sigmoid(logits).float()


@torch.no_grad()
def predict_triplet(models: Sequence[nn.Module], image: torch.Tensor) -> torch.Tensor:
    """[B,3,H,W] probabilities in BASE_ARCHITECTURES order."""
    if len(models) != 3:
        raise ValueError("exactly three base models are required")
    return torch.cat([predict_probability(m, image) for m in models], dim=1)
