import numpy as np
import pandas as pd
import pytest

from dcr.errors import DCRError
from dcr.evaluation.bootstrap import group_scores, paired_cluster_bootstrap
from dcr.evaluation.metrics import (boundary_f1, common_error_masks, dice, negative_false_positive,
                                    recovery_and_harm, site_equal_dice)

V = np.ones((4, 4), bool)


def test_dice_positive_only():
    t = np.zeros((4, 4))
    t[0, :2] = 1
    p = np.zeros((4, 4))
    p[0, 1:3] = 1
    assert dice(p, t, V) == pytest.approx(0.5)
    assert dice(p, np.zeros((4, 4)), V) is None


def test_common_errors_recovery_and_harm():
    target = np.array([[1, 1, 0, 0]] * 4, float)
    probs = np.zeros((3, 4, 4))
    probs[:, :, 2] = 0.8          # column 2: all three say polyp, truth background -> common false alarm
    probs[:, :, 0] = 0.1          # column 0: all three miss -> common miss
    probs[:, :, 1] = 0.9          # column 1: correct; column 3: correct background
    miss, fa = common_error_masks(probs, target, V)
    assert miss.sum() == 4 and fa.sum() == 4
    q = probs.mean(0).copy()
    q[:, 0] = 0.9      # recover all misses
    q[:2, 1] = 0.1     # harm 2 of 8 baseline-correct pixels
    r = recovery_and_harm(probs, q, target, V)
    assert r["recovery_rate"] == pytest.approx(4 / 8)
    assert r["harm_count"] == 2 and r["harm_rate"] == pytest.approx(2 / 8)
    assert r["net_error_reduction"] == pytest.approx((8 - 6) / 16)
    none = recovery_and_harm(np.zeros((3, 4, 4)), np.zeros((4, 4)), np.zeros((4, 4)), V)
    assert none["recovery_rate"] is None


def test_negative_component_threshold_is_point_one_percent():
    valid = np.ones((100, 100), bool)  # n_valid = 10_000 -> min component 10 px
    q = np.zeros((100, 100))
    q[0, :9] = 1
    assert negative_false_positive(q, valid)["is_fp_frame"] is False
    q[0, 9] = 1
    assert negative_false_positive(q, valid)["is_fp_frame"] is True
    diag = np.zeros((100, 100))
    np.fill_diagonal(diag[:10, :10], 1)  # 8-connected diagonal of 10 px
    assert negative_false_positive(diag, valid)["is_fp_frame"] is True


def _square(shift=0):
    m = np.zeros((64, 64), bool)
    m[10:30, 10 + shift:30 + shift] = True
    return m


def test_boundary_f1_tolerance():
    valid = np.ones((64, 64), bool)
    assert boundary_f1(_square(0), _square(0), valid) == pytest.approx(1.0)
    assert boundary_f1(_square(2), _square(0), valid) == pytest.approx(1.0)
    assert boundary_f1(_square(3), _square(0), valid) < 1.0
    assert boundary_f1(_square(0), np.zeros((64, 64), bool), valid) is None


def test_boundary_ignores_padding_edge():
    valid = np.zeros((64, 64), bool)
    valid[10:, :] = True
    m = np.zeros((64, 64), bool)
    m[10:20, 20:40] = True  # touches the padding edge at row 10
    assert boundary_f1(m, m, valid) == pytest.approx(1.0)


def test_site_equal_macro_ignores_image_counts():
    rec = pd.DataFrame({"site_id": ["C5"] * 4 + ["C6"], "group_id": ["a", "a", "a", "b", "c"],
                        "dice": [1.0, 1.0, 1.0, 0.0, 0.2]})
    out = site_equal_dice(rec)
    assert out["per_site"] == {"C5": pytest.approx(0.5), "C6": pytest.approx(0.2)}
    assert out["site_equal"] == pytest.approx(0.35)


def _records(gain=0.02):
    rng = np.random.default_rng(0)
    rows = []
    for site in ("C5", "C6"):
        for g in range(12):
            for i in range(3):
                base = rng.uniform(0.5, 0.9)
                for seed in range(3):
                    for method, d in (("M05", base), ("M07", base + gain)):
                        rows.append(dict(site_id=site, group_id=f"{site}g{g}", sample_id=f"{site}g{g}i{i}",
                                         method=method, seed=seed, dice=d))
    return pd.DataFrame(rows)


def test_bootstrap_is_paired_and_deterministic():
    groups = group_scores(_records(0.02))
    r1 = paired_cluster_bootstrap(groups, n_boot=500, seed=1)
    r2 = paired_cluster_bootstrap(groups, n_boot=500, seed=1)
    assert r1 == r2
    # a constant paired gain gives a degenerate CI only if clusters are drawn jointly for both methods
    assert r1.delta == pytest.approx(0.02) and r1.ci_low == pytest.approx(0.02) and r1.ci_high == pytest.approx(0.02)
    assert r1.verdicts() == {"statistics_success": True, "engineering_target": True,
                             "per_site_drop_exceeds_guardrail": False}


def test_bootstrap_requires_all_seeds():
    rec = _records()
    with pytest.raises(DCRError):
        group_scores(rec[~((rec.seed == 2) & (rec.sample_id == "C5g0i0"))])
