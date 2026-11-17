"""Canonical one-row-per-image manifest and its leakage constraints."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ..errors import DCRError
from ..utils import sha256_file

MANIFEST_COLUMNS = (
    "sample_id", "dataset_name", "site_id", "patient_id", "sequence_id", "near_duplicate_group",
    "image_path", "mask_path", "image_sha256", "mask_sha256", "width", "height", "channels",
    "has_verified_mask", "is_negative", "label_area_px", "label_area_ratio",
    "audit_state", "exclude_reason", "split", "split_version", "license_ref", "source_revision",
)
SPLITS = {"train", "dev_select", "dev_mid", "external", "secondary", "unassigned"}
GROUP_COLUMNS = ("patient_id", "sequence_id", "near_duplicate_group")
SITE_SPLIT_RULES = {
    "C1": {"train"}, "C2": {"train"}, "C3": {"train"},
    "C4": {"dev_select", "dev_mid"}, "C5": {"external"}, "C6": {"external"},
}
_SITE_RE = re.compile(r"(?:^|[_/\-])C([1-6])(?=[_/.\-]|$)")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _as_bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def validate_manifest(df: pd.DataFrame) -> list[str]:
    """Return a list of violated constraints (empty list = valid)."""
    missing = set(MANIFEST_COLUMNS) - set(df.columns)
    if missing:
        return [f"missing columns: {sorted(missing)}"]
    errors: list[str] = []
    dup = df.loc[df.sample_id.duplicated(), "sample_id"].tolist()
    if dup:
        errors.append(f"duplicate sample_id: {dup[:5]}")
    unknown = set(df.split.dropna()) - SPLITS
    if unknown:
        errors.append(f"unknown split values: {sorted(unknown)}")
    negative = df[_as_bool(df.is_negative)]
    bad_neg = negative[~_as_bool(negative.has_verified_mask) | (negative.label_area_px.fillna(-1) != 0)]
    if len(bad_neg):
        errors.append(f"negative without verified all-zero mask: {bad_neg.sample_id.tolist()[:5]}")
    in_use = df[df.split != "unassigned"]
    bad_use = in_use[(in_use.audit_state != "verified") | ~_as_bool(in_use.has_verified_mask)]
    if len(bad_use):
        errors.append(f"unverified or unlabeled rows assigned to a split: {bad_use.sample_id.tolist()[:5]}")
    active = df[df.audit_state == "verified"]
    polypgen = active[active.dataset_name == "PolypGen"]
    for site, allowed in SITE_SPLIT_RULES.items():
        wrong = polypgen[(polypgen.site_id == site) & ~polypgen.split.isin(allowed)]
        if len(wrong):
            errors.append(f"{site} rows outside {sorted(allowed)}: {wrong.sample_id.tolist()[:5]}")
    for col in GROUP_COLUMNS:
        n_splits = active.dropna(subset=[col]).groupby(col).split.nunique()
        crossing = n_splits[n_splits > 1].index.tolist()
        if crossing:
            errors.append(f"{col} crosses splits: {crossing[:5]}")
    return errors


def assert_valid_manifest(df: pd.DataFrame) -> None:
    errors = validate_manifest(df)
    if errors:
        raise DCRError("SPLIT_UNVERIFIED", "; ".join(errors))


def read_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def write_manifest(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False) if path.suffix == ".parquet" else df.to_csv(path, index=False)
    return path


def parse_site(path: str) -> str:
    match = _SITE_RE.search(path)
    return f"C{match.group(1)}" if match else "UNKNOWN"


def _find_mask(image_path: Path) -> Path | None:
    # Generic layout guesses; adapt to the exact folder structure of each dataset release.
    candidates = [image_path.parent.parent / "masks" / image_path.name,
                  image_path.parent.parent / "masks" / f"{image_path.stem}.png",
                  image_path.with_name(f"{image_path.stem}_mask{image_path.suffix}"),
                  image_path.with_name(f"{image_path.stem}_mask.png")]
    return next((c for c in candidates if c.is_file()), None)


def _verify_mask(image_size: tuple[int, int], mask_path: Path) -> tuple[bool, int, str | None]:
    mask = np.asarray(Image.open(mask_path).convert("L"))
    if mask.shape[::-1] != image_size:
        return False, 0, "MASK_ALIGNMENT_ERROR"
    return True, int((mask > 127).sum()), None


def inventory_directory(root: str | Path, dataset_name: str, license_ref: str = "",
                        source_revision: str = "") -> pd.DataFrame:
    """Scan a raw dataset folder into manifest rows (audit_state=needs_review, split=unassigned)."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    rows = []
    for image_path in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES):
        if "mask" in image_path.parent.name.lower() or image_path.stem.endswith("_mask"):
            continue
        with Image.open(image_path) as im:
            width, height = im.size
            channels = len(im.getbands())
        mask_path = _find_mask(image_path)
        verified, area, reason = _verify_mask((width, height), mask_path) if mask_path else (False, 0, "no_mask")
        rel = image_path.relative_to(root).as_posix()
        rows.append({
            "sample_id": f"{dataset_name}:{rel}", "dataset_name": dataset_name, "site_id": parse_site(rel),
            "patient_id": None, "sequence_id": None, "near_duplicate_group": None,
            "image_path": str(image_path), "mask_path": str(mask_path) if mask_path else None,
            "image_sha256": sha256_file(image_path),
            "mask_sha256": sha256_file(mask_path) if mask_path else None,
            "width": width, "height": height, "channels": channels,
            "has_verified_mask": verified, "is_negative": (area == 0) if verified else None,
            "label_area_px": area if verified else None,
            "label_area_ratio": area / (width * height) if verified else None,
            "audit_state": "needs_review", "exclude_reason": reason,
            "split": "unassigned", "split_version": None,
            "license_ref": license_ref, "source_revision": source_revision,
        })
    return pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))
