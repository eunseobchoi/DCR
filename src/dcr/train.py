"""Minimal refiner trainer: β=1, frozen base probabilities, geometry precomputed on CPU per batch."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .errors import DCRError
from .methods import LossWeights, RefinementMethod
from .registry import run_template
from .utils import sha256_file, sha256_json, write_json_atomic


@dataclass(frozen=True)
class TrainConfig:
    method: str = "M07"
    steps: int = 100
    batch_size: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    block_size: int = 32
    widths: tuple = (24, 48, 96, 192)
    zero_rgb: bool = False
    loss: LossWeights = LossWeights()
    grad_clip: float = 1.0
    seed: int = 202601
    save_every: int = 50


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _save_checkpoint(path: Path, model, optimizer, step: int, cfg: TrainConfig) -> str:
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                "rng": torch.get_rng_state(), "config": asdict(cfg)}, tmp)
    os.replace(tmp, path)
    return sha256_file(path)


def _batches(loader: DataLoader):
    while True:
        yield from loader


def train_refiner(cfg: TrainConfig, dataset: Dataset, out_dir: str | Path, run_id: str | None = None) -> dict:
    """Train (or resume) one refiner run and return its run.json record."""
    if len(dataset) < cfg.batch_size:
        raise ValueError("dataset smaller than one batch")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path, record_path = out_dir / "last.pt", out_dir / "run.json"
    torch.manual_seed(cfg.seed)
    model = RefinementMethod(cfg.method, cfg.widths, cfg.block_size, cfg.seed, cfg.zero_rgb)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    step = 0
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
        torch.set_rng_state(state["rng"])
    record = {**run_template(run_id or f"SMOKE-{cfg.method}", "MAIN", 0, method=cfg.method),
              "status": "RUNNING", "started_at": _now(), "config_sha256": sha256_json(asdict(cfg)),
              "effective_batch_size": cfg.batch_size, "optimizer_updates_completed": step}
    write_json_atomic(record_path, record)
    # Note: the DataLoader cursor is not restored on resume (fresh shuffle); step counts are exact.
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    batches = _batches(loader)
    model.train()
    last_terms: dict = {}
    try:
        while step < cfg.steps:
            batch = next(batches)
            geometry = model.geometry(batch["probs"], batch["valid"], batch["sample_id"])
            out = model(batch["image"], batch["probs"], batch["valid"], geometry, beta=1.0)
            terms = model.loss(out, batch["target"], batch["valid"], geometry, cfg.loss)
            if not torch.isfinite(terms["total"]):
                raise DCRError("NONFINITE_LOSS", f"step {step}")
            optimizer.zero_grad(set_to_none=True)
            terms["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            step += 1
            last_terms = {k: float(v) for k, v in terms.items()}
            if step % cfg.save_every == 0:
                _save_checkpoint(ckpt_path, model, optimizer, step, cfg)
        digest = _save_checkpoint(ckpt_path, model, optimizer, step, cfg)
    except BaseException as exc:
        write_json_atomic(record_path, {**record, "status": "FAILED", "ended_at": _now(),
                                        "optimizer_updates_completed": step, "issues": [repr(exc)]})
        raise
    record = {**record, "status": "COMPLETED", "ended_at": _now(), "optimizer_updates_completed": step,
              "checkpoint_sha256": digest, "last_loss_terms": last_terms}
    write_json_atomic(record_path, record)
    return record
