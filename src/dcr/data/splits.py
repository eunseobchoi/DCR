"""Institution-separated splits and the out-of-fold (OOF) contract."""
from __future__ import annotations

import hashlib
from typing import Sequence

import pandas as pd

from ..errors import DCRError

TRAIN_SITES = ("C1", "C2", "C3")
DEV_SITE = "C4"
EXTERNAL_SITES = ("C5", "C6")

# fold_id -> (base training sites, sites whose predictions this fold produces)
OOF_FOLDS = {
    "leave_C1_out": (("C2", "C3"), ("C1",)),
    "leave_C2_out": (("C1", "C3"), ("C2",)),
    "leave_C3_out": (("C1", "C2"), ("C3",)),
    "full_C123": (("C1", "C2", "C3"), ("C4", "C5", "C6")),
}


def cluster_key(row) -> str:
    """Patient, else sequence, else near-duplicate group, else the image itself."""
    for col in ("patient_id", "sequence_id", "near_duplicate_group"):
        value = getattr(row, col, None)
        if value is not None and not pd.isna(value):
            return f"{col}:{value}"
    return f"sample:{row.sample_id}"


def _unit_interval(salt: str, key: str) -> float:
    return int(hashlib.sha256(f"{salt}|{key}".encode()).hexdigest()[:8], 16) / 2**32


def _split_for(row, key: str, dev_mid_fraction: float, salt: str) -> str:
    if row.audit_state != "verified" or not bool(row.has_verified_mask):
        return "unassigned"
    if row.dataset_name != "PolypGen":
        return "secondary"
    if row.site_id in TRAIN_SITES:
        return "train"
    if row.site_id == DEV_SITE:
        return "dev_mid" if _unit_interval(salt, key) < dev_mid_fraction else "dev_select"
    if row.site_id in EXTERNAL_SITES:
        return "external"
    return "unassigned"


def assign_splits(df: pd.DataFrame, dev_mid_fraction: float = 0.5, salt: str = "split_v1") -> pd.DataFrame:
    """Return a copy with `split` set; C4 is cut by cluster so no cluster spans select/mid."""
    out = df.copy()
    rows = list(out.itertuples(index=False))
    out["split"] = [_split_for(r, cluster_key(r), dev_mid_fraction, salt) for r in rows]
    out["split_version"] = salt
    return out


def fold_for_target(site: str) -> str:
    for fold_id, (_, targets) in OOF_FOLDS.items():
        if site in targets:
            return fold_id
    raise DCRError("SPLIT_UNVERIFIED", f"no fold produces predictions for site {site}")


def assert_oof_contract(target_site: str, source_train_sites: Sequence[str],
                        allow_self_site: bool = False) -> None:
    """A prediction used for refiner training/eval must come from a base model that never saw its site.

    `allow_self_site=True` exists only for the deliberate leakage ablation A04.
    """
    if target_site in set(source_train_sites) and not allow_self_site:
        raise DCRError("LEAKAGE_OOF_SELF_SITE",
                       f"prediction for {target_site} comes from a model trained on {sorted(source_train_sites)}")
    if set(source_train_sites) & set(EXTERNAL_SITES + (DEV_SITE,)):
        raise DCRError("LEAKAGE_OOF_SELF_SITE", "base models may only be trained on C1–C3")
