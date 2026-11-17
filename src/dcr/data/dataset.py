"""Datasets yielding the fixed-shape batch dict used by the trainer.

`SyntheticPolypDataset` drives quick runs and tests without real data.
`CachedPredictionDataset` reads a validated manifest plus a frozen base-prediction cache.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

from ..errors import DCRError
from .letterbox import letterbox, normalize_imagenet
from .splits import assert_oof_contract, cluster_key


class SyntheticPolypDataset(Dataset):
    """Elliptical "polyps", letterbox-style padding and three noisy base maps that share a common miss."""

    def __init__(self, n: int = 32, size: int = 64, seed: int = 0,
                 negative_fraction: float = 0.2, site_id: str = "SYN") -> None:
        self.n, self.size, self.seed = n, size, seed
        self.negative_fraction, self.site_id = negative_fraction, site_id

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng([self.seed, idx])
        h = w = self.size
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        pad = int(rng.integers(0, h // 8 + 1))
        valid = np.zeros((h, w), np.float32)
        valid[pad:h - pad] = 1
        target = np.zeros((h, w), np.float32)
        shared_miss = np.zeros((h, w), np.float32)
        if rng.random() >= self.negative_fraction:
            cy, cx = rng.uniform(0.3, 0.7, 2) * (h, w)
            ry, rx = rng.uniform(0.1, 0.25, 2) * (h, w)
            target = ((((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1).astype(np.float32) * valid
            shared_miss = target * (xx > cx + 0.3 * rx)
        image = 0.45 + 0.05 * rng.standard_normal((3, h, w)).astype(np.float32)
        image[0] += 0.25 * target
        probs = []
        for _ in range(3):
            noise = gaussian_filter(rng.standard_normal((h, w)), 2.0) * 4.0
            logits = 8.0 * (target * (1 - shared_miss) - 0.5) + noise
            probs.append(1.0 / (1.0 + np.exp(-logits)))
        return {
            "image": torch.from_numpy(image * valid),
            "probs": torch.from_numpy(np.stack(probs).astype(np.float32)),
            "target": torch.from_numpy(target[None]),
            "valid": torch.from_numpy(valid[None]),
            "sample_id": f"syn_{idx:05d}",
            "site_id": self.site_id,
            "group_id": f"syn_g{idx // 4:04d}",
        }


class CachedPredictionDataset(Dataset):
    """Manifest rows of one split + per-sample cache `{cache_dir}/{safe_id}.npz`.

    Cache layout: `probs` float32 [3,S,S]
    in BASE_ARCHITECTURES order and `source_train_sites` (str array) for the OOF check.
    """

    def __init__(self, manifest: pd.DataFrame, cache_dir: str | Path, split: str,
                 size: int = 512, allow_self_site: bool = False) -> None:
        rows = manifest[(manifest.split == split) & manifest.has_verified_mask.fillna(False).astype(bool)]
        if rows.empty:
            raise DCRError("SPLIT_UNVERIFIED", f"no verified rows for split {split!r}")
        self.rows = rows.reset_index(drop=True)
        self.cache_dir, self.size, self.allow_self_site = Path(cache_dir), size, allow_self_site

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def cache_name(sample_id: str) -> str:
        return sample_id.replace("/", "__").replace(":", "--") + ".npz"

    def __getitem__(self, idx: int) -> dict:
        row = self.rows.iloc[idx]
        image = cv2.cvtColor(cv2.imread(row.image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(row.mask_path, cv2.IMREAD_GRAYSCALE) > 127
        img, mask_lb, valid, meta = letterbox(image, mask, self.size)
        cache_path = self.cache_dir / self.cache_name(row.sample_id)
        if not cache_path.is_file():
            raise DCRError("INCOMPLETE_BASE_TRIPLET", f"missing cache {cache_path}")
        with np.load(cache_path) as cache:
            probs = cache["probs"].astype(np.float32)
            sources = [str(s) for s in cache["source_train_sites"]]
        if probs.shape != (3, self.size, self.size):
            raise DCRError("INCOMPLETE_BASE_TRIPLET", f"bad cache shape {probs.shape}")
        assert_oof_contract(row.site_id, sources, self.allow_self_site)
        return {
            "image": torch.from_numpy(normalize_imagenet(img, valid)),
            "probs": torch.from_numpy(probs),
            "target": torch.from_numpy(mask_lb[None].astype(np.float32)),
            "valid": torch.from_numpy(valid[None].astype(np.float32)),
            "sample_id": row.sample_id,
            "site_id": row.site_id,
            "group_id": cluster_key(row),
            "letterbox_meta": meta.to_dict(),
        }
