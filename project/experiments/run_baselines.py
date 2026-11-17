#!/usr/bin/env python3
"""
Experiment A (implementation guide section 8A): baseline comparison under the shared protocol.

Trains every requested model on the same train/val/test split with the same hyperparameters,
selects each checkpoint on validation, evaluates test once, and writes one summary table.
Replaces the former run_gat_baseline.py, run_plain_moe.py, run_graph_moe.py, run_track_b_full.py.

Usage (from project/):
    python experiments/run_baselines.py --device cuda                         # all models, seed 42
    python experiments/run_baselines.py --models gat,ug_moe --device cuda     # subset
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from common import configure_torch, load_splits, trivial_baseline_metrics
from registry import add_common_args, print_table, run_suite

DEFAULT_MODELS = "muad,gat,plain_moe,graph_moe,ug_moe"


def main():
    parser = add_common_args(argparse.ArgumentParser(description="Baseline comparison (Exp. A)"))
    parser.add_argument("--models", type=str, default=DEFAULT_MODELS, help=f"Comma-separated (default: {DEFAULT_MODELS})")
    args = parser.parse_args()

    configure_torch()
    names = [m.strip() for m in args.models.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device(args.device)

    out_dir = Path(args.output_dir) / f"baselines_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    loaders = load_splits(Path(args.data_dir), args.batch_size, args.val_ratio, args.split_seed,
                          split_out=out_dir / "split.json", normalize=args.normalize, split=args.split, test_run=args.test_run, val_run=args.val_run)
    print(f"Sizes: {loaders['sizes']}  culprit distribution: {loaders['culprit_dist']}")
    print(f"Models: {names} | seeds: {seeds}")

    trivial = trivial_baseline_metrics(loaders["test"])
    suite = run_suite(names, seeds, loaders, device, args, out_dir)

    summary = {
        "experiment": "baselines",
        "protocol": "train/val/test; checkpoint selected on val macro-F1 (tie: PR-AUC); test evaluated once",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "config": vars(args),
        "sizes": loaders["sizes"],
        "culprit_dist": loaders["culprit_dist"],
        "trivial_always_anomalous_test": trivial,
        "models": suite,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nTEST metrics at validation-selected epoch (mean over seeds)")
    print_table(suite, trivial)
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
