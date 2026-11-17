"""`dcr` command-line entry point. Fail-closed errors print their code and exit 2."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch

from .errors import DCRError


def _cmd_geometry_check(args) -> int:
    from .geometry import make_geometry, project

    s, size = args.block_size, 2 * args.block_size
    gen = torch.Generator().manual_seed(0)
    ramp = torch.linspace(0.1, 0.5, size * size).reshape(size, size)
    rank1 = torch.stack([ramp, torch.full_like(ramp, 0.3), 0.6 - ramp])[None]
    padded = torch.ones(1, 1, size, size)
    padded[..., s:] = 0
    cases = {
        "rank0_identical": (torch.full((1, 3, size, size), 0.3), torch.ones(1, 1, size, size)),
        "rank1_dependent": (rank1, torch.ones(1, 1, size, size)),
        "rank2_random": (torch.rand(1, 3, size, size, generator=gen), torch.ones(1, 1, size, size)),
        "rank2_half_padded": (torch.rand(1, 3, size, size, generator=gen), padded),
    }
    all_ok = True
    for name, (probs, valid) in cases.items():
        g = make_geometry(probs, valid, s)
        v = torch.randn(1, 1, size, size, generator=gen)
        pv = project(v, g, valid)
        perp = v * valid - pv
        ok = (torch.allclose(project(pv, g, valid), pv, atol=1e-5)
              and abs(float((pv * perp).sum())) < 1e-3
              and torch.allclose((v * valid).square().sum(), pv.square().sum() + perp.square().sum(), atol=1e-3))
        all_ok &= ok
        print(f"{name:18s} ranks={sorted(set(g.rank.flatten().tolist()))} "
              f"n_valid={sorted(set(g.n_valid.flatten().tolist()))} identities={'PASS' if ok else 'FAIL'}")
    return 0 if all_ok else 1


def _cmd_runs_generate(args) -> int:
    from .registry import EXPECTED_COUNTS, generate_runs

    families = tuple(EXPECTED_COUNTS) if args.family == "ALL" else (args.family,)
    runs = generate_runs(families)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in runs))
    print(f"wrote {len(runs)} runs to {args.out}")
    return 0


def _cmd_runs_validate(args) -> int:
    from .registry import validate_runs

    runs = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    counts = validate_runs(runs, args.expect_total)
    print(json.dumps({"total": len(runs), **counts}))
    return 0


def _cmd_data_validate(args) -> int:
    from .data.manifest import read_manifest, validate_manifest

    errors = validate_manifest(read_manifest(args.manifest))
    for e in errors:
        print(f"SPLIT_UNVERIFIED: {e}", file=sys.stderr)
    print("manifest OK" if not errors else f"{len(errors)} violation(s)")
    return 0 if not errors else 2


def _cmd_data_split(args) -> int:
    from .data.manifest import assert_valid_manifest, read_manifest, write_manifest
    from .data.splits import assign_splits

    df = assign_splits(read_manifest(args.manifest), args.dev_mid_fraction, args.salt)
    assert_valid_manifest(df)
    write_manifest(df, args.out)
    print(df.split.value_counts().to_string())
    return 0


def _cmd_train_refiner(args) -> int:
    from .train import TrainConfig, train_refiner

    if not args.synthetic:
        raise DCRError("SPLIT_UNVERIFIED", "real-data training needs a locked split and OOF cache; use --synthetic for now")
    from .data.dataset import SyntheticPolypDataset

    cfg = replace(TrainConfig(), method=args.method, steps=args.steps, batch_size=args.batch_size,
                  block_size=args.block_size)
    record = train_refiner(cfg, SyntheticPolypDataset(n=max(8, 2 * args.batch_size), size=args.size),
                           args.out, run_id=args.run_id)
    print(json.dumps({k: record[k] for k in ("run_id", "status", "optimizer_updates_completed", "checkpoint_sha256")}))
    return 0


def _cmd_evaluate_lock(args) -> int:
    from .registry import write_evaluation_lock

    lock = write_evaluation_lock(args.out, json.loads(Path(args.payload).read_text()))
    print(f"locked {args.out} sha256={lock['lock_sha256']}")
    return 0


def _cmd_evaluate_external(args) -> int:
    from .registry import authorize_external_evaluation

    marker = authorize_external_evaluation(args.lock, args.marker_dir)
    print(f"external evaluation authorized once (marker {marker}); run external inference with the locked bundle now")
    return 0


def _cmd_evaluate_bootstrap(args) -> int:
    from .evaluation.bootstrap import group_scores, paired_cluster_bootstrap

    result = paired_cluster_bootstrap(group_scores(pd.read_csv(args.scores)), args.method_a, args.method_b,
                                      n_boot=args.n_boot, seed=args.seed)
    print(json.dumps({"delta": result.delta, "ci95": [result.ci_low, result.ci_high],
                      "site_delta": result.site_delta, "n_clusters": result.n_clusters, **result.verdicts()}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dcr", description="DCR research pipeline (research use only)")
    top = parser.add_subparsers(dest="group", required=True)

    geo = top.add_parser("geometry").add_subparsers(dest="cmd", required=True)
    p = geo.add_parser("check", help="rank 0/1/2 and projection identities on synthetic tiles")
    p.add_argument("--block-size", type=int, default=32)
    p.set_defaults(func=_cmd_geometry_check)

    runs = top.add_parser("runs").add_subparsers(dest="cmd", required=True)
    p = runs.add_parser("generate", help="write the run matrix (36 BASE / 21 MAIN / 9 REF / 24 ABL)")
    p.add_argument("--family", default="ALL", choices=["ALL", "BASE", "MAIN", "REF", "ABL"])
    p.add_argument("--out", required=True)
    p.set_defaults(func=_cmd_runs_generate)
    p = runs.add_parser("validate")
    p.add_argument("--input", required=True)
    p.add_argument("--expect-total", type=int)
    p.set_defaults(func=_cmd_runs_validate)

    data = top.add_parser("data").add_subparsers(dest="cmd", required=True)
    p = data.add_parser("validate", help="check manifest leakage constraints")
    p.add_argument("--manifest", required=True)
    p.set_defaults(func=_cmd_data_validate)
    p = data.add_parser("split", help="assign institution splits and validate")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dev-mid-fraction", type=float, default=0.5)
    p.add_argument("--salt", default="split_v1")
    p.set_defaults(func=_cmd_data_split)

    train = top.add_parser("train").add_subparsers(dest="cmd", required=True)
    p = train.add_parser("refiner", help="train one refinement method")
    p.add_argument("--method", default="M07", choices=[f"M0{i}" for i in range(1, 8)])
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--run-id")
    p.add_argument("--out", required=True)
    p.set_defaults(func=_cmd_train_refiner)

    ev = top.add_parser("evaluate").add_subparsers(dest="cmd", required=True)
    p = ev.add_parser("lock", help="freeze weights/beta/threshold before external evaluation")
    p.add_argument("--payload", required=True)
    p.add_argument("--out", default="locks/evaluation_lock.json")
    p.set_defaults(func=_cmd_evaluate_lock)
    p = ev.add_parser("external", help="one-shot C5/C6 guard")
    p.add_argument("--lock", default="locks/evaluation_lock.json")
    p.add_argument("--marker-dir", default="locks")
    p.set_defaults(func=_cmd_evaluate_external)
    p = ev.add_parser("bootstrap", help="paired cluster bootstrap of the primary comparison")
    p.add_argument("--scores", required=True, help="CSV: site_id,group_id,sample_id,method,seed,dice")
    p.add_argument("--method-a", default="M07")
    p.add_argument("--method-b", default="M05")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=1027)
    p.set_defaults(func=_cmd_evaluate_bootstrap)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except DCRError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
