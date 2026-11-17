"""Evaluation metrics. All inputs are 2-D numpy arrays on the 512 grid; `valid` masks padding.

Undefined quantities (empty denominators, negative images for Dice) return None,
never 0, so they cannot silently enter an average.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import ndimage

COMMON_ERROR_THRESHOLD = 0.5
EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


def _bool(x) -> np.ndarray:
    return np.asarray(x).astype(bool)


def dice(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float | None:
    """Dice on a positive image; None when the ground truth is empty."""
    v = _bool(valid)
    p, t = _bool(pred) & v, _bool(target) & v
    if not t.any():
        return None
    tp = (p & t).sum()
    return float(2 * tp / (p.sum() + t.sum()))


def common_error_masks(probs: np.ndarray, target: np.ndarray, valid: np.ndarray,
                       threshold: float = COMMON_ERROR_THRESHOLD) -> tuple[np.ndarray, np.ndarray]:
    """probs [3,H,W] -> (common miss, common false alarm) where all three base models agree and are wrong."""
    b = np.asarray(probs) >= threshold
    t, v = _bool(target), _bool(valid)
    miss = t & ~b.any(axis=0) & v
    false_alarm = ~t & b.all(axis=0) & v
    return miss, false_alarm


def _rate(num: int, den: int) -> float | None:
    return float(num / den) if den else None


def recovery_and_harm(probs: np.ndarray, q: np.ndarray, target: np.ndarray, valid: np.ndarray,
                      t_eval: float = 0.5) -> dict:
    """Common-error recovery, harm to correct baseline pixels, and net error reduction."""
    t, v = _bool(target), _bool(valid)
    miss, false_alarm = common_error_masks(probs, target, valid)
    pbar = np.asarray(probs).mean(axis=0)
    baseline_correct = ((pbar >= 0.5) == t) & v
    corrected = np.asarray(q) >= t_eval
    fixed = corrected == t
    common = miss | false_alarm
    harmed = baseline_correct & ~fixed
    baseline_errors = int((~((pbar >= 0.5) == t) & v).sum())
    corrected_errors = int((~fixed & v).sum())
    return {
        "common_miss_px": int(miss.sum()), "common_miss_recovered_px": int((miss & fixed).sum()),
        "common_false_alarm_px": int(false_alarm.sum()),
        "common_false_alarm_recovered_px": int((false_alarm & fixed).sum()),
        "recovery_rate": _rate(int((common & fixed).sum()), int(common.sum())),
        "baseline_correct_px": int(baseline_correct.sum()), "harm_count": int(harmed.sum()),
        "harm_rate": _rate(int(harmed.sum()), int(baseline_correct.sum())),
        "net_error_reduction": _rate(baseline_errors - corrected_errors, int(v.sum())),
    }


def negative_false_positive(q: np.ndarray, valid: np.ndarray, threshold: float = 0.5,
                            min_area_ratio: float = 0.001) -> dict:
    """A negative frame is a false-positive frame if any 8-connected predicted component
    covers at least ceil(min_area_ratio · n_valid) valid pixels."""
    v = _bool(valid)
    n_valid = int(v.sum())
    pred = (np.asarray(q) >= threshold) & v
    labels, n = ndimage.label(pred, structure=EIGHT_CONNECTED)
    sizes = np.bincount(labels.ravel())[1:] if n else np.zeros(0, int)
    min_px = math.ceil(min_area_ratio * n_valid - 1e-9)
    return {
        "is_fp_frame": bool((sizes >= min_px).any()) if n_valid else None,
        "fp_pixels": int(pred.sum()), "n_valid": n_valid,
        "largest_component_px": int(sizes.max()) if sizes.size else 0, "min_component_px": min_px,
    }


def _boundary(mask: np.ndarray, valid: np.ndarray) -> np.ndarray:
    # Treat padding as foreground for erosion so the letterbox edge never becomes a lesion boundary.
    return mask & ~ndimage.binary_erosion(mask | ~valid, border_value=1)


def boundary_f1(pred: np.ndarray, target: np.ndarray, valid: np.ndarray, tolerance: float = 2.0) -> float | None:
    """Boundary F1 with a pixel tolerance; None when the ground truth is empty."""
    v = _bool(valid)
    p, t = _bool(pred) & v, _bool(target) & v
    if not t.any():
        return None
    bp, bt = _boundary(p, v), _boundary(t, v)
    if not bp.any():
        return 0.0
    dist_to_t = ndimage.distance_transform_edt(~bt)
    dist_to_p = ndimage.distance_transform_edt(~bp)
    precision = float((dist_to_t[bp] <= tolerance).mean())
    recall = float((dist_to_p[bt] <= tolerance).mean())
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def site_equal_dice(records: pd.DataFrame, score_col: str = "dice") -> dict:
    """image scores -> group mean -> site mean -> equal-weight macro over sites."""
    scored = records.dropna(subset=[score_col])
    group_means = scored.groupby(["site_id", "group_id"])[score_col].mean()
    site_means = group_means.groupby(level="site_id").mean()
    return {"per_site": site_means.to_dict(), "site_equal": float(site_means.mean()),
            "n_groups": group_means.groupby(level="site_id").size().to_dict()}
