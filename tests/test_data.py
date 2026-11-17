import numpy as np
import pandas as pd
import pytest
import torch

from dcr.data.dataset import SyntheticPolypDataset
from dcr.data.letterbox import letterbox, normalize_imagenet, unletterbox_probability
from dcr.data.manifest import MANIFEST_COLUMNS, parse_site, validate_manifest
from dcr.data.splits import assert_oof_contract, assign_splits, fold_for_target
from dcr.errors import DCRError


def test_letterbox_preserves_aspect_and_marks_padding():
    image = np.random.default_rng(0).integers(0, 255, (300, 600, 3), dtype=np.uint8)
    mask = np.zeros((300, 600), np.uint8)
    mask[100:200, 200:400] = 1
    img, m, valid, meta = letterbox(image, mask, 512)
    assert img.shape == (512, 512, 3) and (meta.new_h, meta.new_w) == (256, 512)
    assert valid.sum() == 256 * 512 and valid[:meta.top].sum() == 0
    assert set(np.unique(m)) <= {0, 1}
    assert unletterbox_probability(m.astype(np.float32), meta).shape == (300, 600)
    norm = normalize_imagenet(img, valid)
    assert norm.shape == (3, 512, 512) and np.all(norm[:, :meta.top] == 0)


def test_letterbox_rejects_misaligned_mask():
    with pytest.raises(ValueError, match="MASK_ALIGNMENT_ERROR"):
        letterbox(np.zeros((10, 20, 3), np.uint8), np.zeros((20, 10), np.uint8))


def _manifest(rows):
    base = {c: None for c in MANIFEST_COLUMNS}
    base.update(dataset_name="PolypGen", has_verified_mask=True, is_negative=False, label_area_px=10,
                audit_state="verified")
    return pd.DataFrame([{**base, **r} for r in rows], columns=list(MANIFEST_COLUMNS))


def _sites():
    return _manifest([
        {"sample_id": f"s{i}", "site_id": site, "sequence_id": f"{site}_seq{i % 2}"}
        for i, site in enumerate(["C1", "C2", "C3", "C4", "C4", "C4", "C4", "C5", "C6"])
    ])


def test_assign_splits_is_valid_and_deterministic():
    a, b = assign_splits(_sites()), assign_splits(_sites())
    assert validate_manifest(a) == []
    assert a.split.tolist() == b.split.tolist()
    assert set(a[a.site_id == "C4"].split) <= {"dev_select", "dev_mid"}
    assert a[a.site_id.isin(["C5", "C6"])].split.eq("external").all()


def test_validate_flags_external_site_in_train_and_group_crossing():
    df = assign_splits(_sites())
    df.loc[df.site_id == "C5", "split"] = "train"
    assert any("C5" in e for e in validate_manifest(df))
    df = assign_splits(_sites())
    df.loc[df.sample_id == "s7", "sequence_id"] = "C1_seq0"  # same sequence as a C1 train row
    assert any("sequence_id crosses splits" in e for e in validate_manifest(df))


def test_validate_flags_unlabeled_negative():
    df = assign_splits(_sites())
    df.loc[0, ["is_negative", "has_verified_mask"]] = [True, False]
    errors = validate_manifest(df)
    assert any("negative without verified" in e for e in errors)
    assert any("unverified or unlabeled" in e for e in errors)


def test_oof_contract():
    assert fold_for_target("C1") == "leave_C1_out" and fold_for_target("C5") == "full_C123"
    assert_oof_contract("C1", ("C2", "C3"))
    with pytest.raises(DCRError) as exc:
        assert_oof_contract("C1", ("C1", "C2", "C3"))
    assert exc.value.code == "LEAKAGE_OOF_SELF_SITE"
    assert_oof_contract("C1", ("C1", "C2", "C3"), allow_self_site=True)  # ablation A04 only
    with pytest.raises(DCRError):
        assert_oof_contract("C4", ("C1", "C5"))


def test_parse_site():
    assert parse_site("positive/positive_C3_0012.jpg") == "C3"
    assert parse_site("kvasir/cju0qkwl35piu0993l0dewei2.jpg") == "UNKNOWN"


def test_synthetic_dataset_contract():
    item = SyntheticPolypDataset(n=4, size=32, negative_fraction=0.0)[0]
    assert item["image"].shape == (3, 32, 32) and item["probs"].shape == (3, 32, 32)
    assert item["target"].shape == item["valid"].shape == (1, 32, 32)
    assert item["probs"].min() >= 0 and item["probs"].max() <= 1
    assert torch.equal(item["target"] * item["valid"], item["target"])
