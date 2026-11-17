"""The 90-run training matrix and the one-shot external-evaluation lock."""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable

from .data.splits import OOF_FOLDS
from .errors import DCRError
from .methods import METHOD_NAMES
from .models.base import BASE_ARCHITECTURES
from .utils import sha256_json, write_json_atomic

SEED_IDS = (0, 1, 2)
REFERENCE_MODELS = {"R01": "nnU-Net v2 (2D)", "R02": "Polyp-PVT", "R03": "SegRefiner"}
# Ablations are expressed as overrides of the M07 configuration.
ABLATIONS = {
    "A01": {"block_size": 16}, "A02": {"block_size": 64}, "A03": {"zero_rgb": True},
    "A04": {"train_predictions": "self_site"}, "A05": {"lambda_parallel": 0.0, "lambda_perp": 0.0},
    "A06": {"mu_keep": 0.0}, "A07": {"refiner_label_fraction": 0.25}, "A08": {"refiner_label_fraction": 0.50},
}
EXPECTED_COUNTS = {"BASE": 36, "MAIN": 21, "REF": 9, "ABL": 24}
REQUIRED_LOCK_KEYS = ("model_checkpoint_sha256", "beta", "threshold", "postprocessing",
                      "split_sha256", "config_sha256", "code_commit")


def run_template(run_id: str, family: str, seed_id: int, **fields) -> dict:
    """run.json skeleton; unknown values stay None until produced."""
    return {
        "run_id": run_id, "family": family, "attempt_id": 1, "status": "PLANNED", "seed_id": seed_id,
        "dataset_split_sha256": None, "base_triplet_checkpoint_sha256": None,
        "input_prediction_manifest_sha256": None, "git_commit": None, "config_sha256": None,
        "training_beta": 1.0, "optimizer_updates_completed": 0, "effective_batch_size": None,
        "started_at": None, "ended_at": None, "checkpoint_sha256": None,
        "dev_select_metrics": None, "issues": [], **fields,
    }


def generate_runs(families: Iterable[str] = tuple(EXPECTED_COUNTS)) -> list[dict]:
    runs: list[dict] = []
    for family in families:
        for s in SEED_IDS:
            if family == "BASE":
                runs += [run_template(f"BASE-{arch}-{fold}-S{s:02d}", family, s, architecture=arch, fold_id=fold,
                                      train_sites=list(OOF_FOLDS[fold][0]))
                         for arch in BASE_ARCHITECTURES for fold in OOF_FOLDS]
            elif family == "MAIN":
                runs += [run_template(f"MAIN-{m}-S{s:02d}", family, s, method=m, tile_size=32) for m in METHOD_NAMES]
            elif family == "REF":
                runs += [run_template(f"REF-{r}-S{s:02d}", family, s, reference=name)
                         for r, name in REFERENCE_MODELS.items()]
            elif family == "ABL":
                runs += [run_template(f"ABL-{a}-S{s:02d}", family, s, method="M07", overrides=o)
                         for a, o in ABLATIONS.items()]
            else:
                raise ValueError(f"unknown family {family!r}")
    return sorted(runs, key=lambda r: r["run_id"])


def validate_runs(runs: list[dict], expect_total: int | None = None) -> Counter:
    ids = Counter(r["run_id"] for r in runs)
    dup = [k for k, v in ids.items() if v > 1]
    if dup:
        raise DCRError("RUN_REGISTRY_CONFLICT", f"duplicate run_id {dup[:5]}")
    by_family = Counter(r["family"] for r in runs)
    for family, n in by_family.items():
        if n != EXPECTED_COUNTS.get(family):
            raise DCRError("RUN_REGISTRY_CONFLICT", f"{family}: {n} runs, expected {EXPECTED_COUNTS.get(family)}")
    if expect_total is not None and len(runs) != expect_total:
        raise DCRError("RUN_REGISTRY_CONFLICT", f"{len(runs)} runs, expected {expect_total}")
    return by_family


def write_evaluation_lock(path: str | Path, payload: dict) -> dict:
    """Freeze model/β/threshold/post-processing before any C5/C6 access."""
    missing = [k for k in REQUIRED_LOCK_KEYS if payload.get(k) is None]
    if missing:
        raise DCRError("EVALUATION_NOT_LOCKED", f"lock payload lacks {missing}")
    path = Path(path)
    if path.exists():
        raise DCRError("EVALUATION_NOT_LOCKED", f"{path} already exists; locks are never rewritten")
    lock = {**payload, "lock_sha256": sha256_json(payload)}
    write_json_atomic(path, lock)
    os.chmod(path, 0o444)
    return lock


def verify_lock(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise DCRError("EVALUATION_NOT_LOCKED", f"no lock at {path}")
    lock = json.loads(path.read_text())
    payload = {k: v for k, v in lock.items() if k != "lock_sha256"}
    if lock.get("lock_sha256") != sha256_json(payload):
        raise DCRError("EVALUATION_NOT_LOCKED", "lock content does not match its hash")
    return lock


def authorize_external_evaluation(lock_path: str | Path, marker_dir: str | Path) -> Path:
    """Allow exactly one C5/C6 evaluation per lock; a second call fails."""
    lock = verify_lock(lock_path)
    marker = Path(marker_dir) / f"external_evaluated_{lock['lock_sha256'][:16]}.marker"
    marker.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(marker, "x") as fh:
            fh.write(lock["lock_sha256"])
    except FileExistsError:
        raise DCRError("EXTERNAL_ALREADY_EVALUATED", f"marker {marker} exists") from None
    return marker
