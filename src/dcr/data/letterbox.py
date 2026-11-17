"""Aspect-preserving 512×512 letterbox with an explicit valid mask."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class LetterboxMeta:
    orig_h: int
    orig_w: int
    new_h: int
    new_w: int
    top: int
    left: int
    size: int

    def to_dict(self) -> dict:
        return asdict(self)


def letterbox(image: np.ndarray, mask: np.ndarray | None = None, size: int = 512):
    """image HxWx3 (uint8 or float in [0,1]) -> (image float32 [size,size,3], mask uint8|None, valid uint8, meta).

    RGB uses bilinear interpolation, the mask nearest-neighbour; padding is centred
    and marked valid=0 so it never counts as background in losses or metrics.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be HxWx3")
    h, w = image.shape[:2]
    if mask is not None and mask.shape[:2] != (h, w):
        raise ValueError("MASK_ALIGNMENT_ERROR: mask and image sizes differ")
    scale = min(size / w, size / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    top, left = (size - new_h) // 2, (size - new_w) // 2
    img = image.astype(np.float32) / (255.0 if image.dtype == np.uint8 else 1.0)
    out = np.zeros((size, size, 3), np.float32)
    out[top:top + new_h, left:left + new_w] = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    valid = np.zeros((size, size), np.uint8)
    valid[top:top + new_h, left:left + new_w] = 1
    out_mask = None
    if mask is not None:
        resized = cv2.resize((mask > 0).astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        out_mask = np.zeros((size, size), np.uint8)
        out_mask[top:top + new_h, left:left + new_w] = resized
    return out, out_mask, valid, LetterboxMeta(h, w, new_h, new_w, top, left, size)


def unletterbox_probability(prob: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    """Crop the padding and resize a [size,size] probability map back to the original resolution."""
    crop = prob[meta.top:meta.top + meta.new_h, meta.left:meta.left + meta.new_w].astype(np.float32)
    return cv2.resize(crop, (meta.orig_w, meta.orig_h), interpolation=cv2.INTER_LINEAR)


def normalize_imagenet(image: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """HxWx3 float [0,1] -> 3xHxW ImageNet-normalised; padding stays exactly 0."""
    out = (image - IMAGENET_MEAN) / IMAGENET_STD * valid[..., None]
    return np.ascontiguousarray(out.transpose(2, 0, 1), dtype=np.float32)
