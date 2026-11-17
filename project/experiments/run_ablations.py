#!/usr/bin/env python3
"""
Component ablations of the Graph-Aware MoE under the shared protocol (experiments/common.py).

The full model (ug_moe) is always trained alongside the ablations so every row is compared
against a reference trained on the same split, seed and hyperparameters.

CAVEAT (PROJECT_STATUS.md section 3.7): with the current model these ablations do not isolate
the components their names suggest. The set will be redefined to match implementation guide
section 8B once Phase 2 fixes routing / uncertainty / fusion.

Usage (from project/):
    python experiments/run_ablations.py --device cuda                           # all ablations, seed 42
    python experiments/run_ablations.py --ablations no_graph,no_fusion --device cuda
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from common import configure_torch, load_splits, trivial_baseline_metrics
from registry import ABLATIONS, add_common_args, print_table, run_suite


def main():
    parser = add_common_args(argparse.ArgumentParser(description="Ablation study"))
    parser.add_argument("--ablations", type=str, default=",".join(ABLATIONS),
                        help=f"Comma-separated subset of: {', '.join(ABLATIONS)}")
    parser.add_argument("--skip-full", action="store_true", help="Do not train the full ug_moe reference")
    args = parser.parse_args()

    configure_torch()
    abls = [a.strip() for a in args.ablations.split(",")]
    unknown = [a for a in abls if a not in ABLATIONS]
    if unknown:
        parser.error(f"unknown ablation(s) {unknown}; options: {list(ABLATIONS)}")
    names = ([] if args.skip_full else ["ug_moe"]) + [f"abl:{a}" for a in abls]
    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device(args.device)

    out_dir = Path(args.output_dir) / f"ablations_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    loaders = load_splits(Path(args.data_dir), args.batch_size, args.val_ratio, args.split_seed,
                          split_out=out_dir / "split.json")
    print(f"Sizes: {loaders['sizes']} | runs: {names} | seeds: {seeds}")

    trivial = trivial_baseline_metrics(loaders["test"])
    suite = run_suite(names, seeds, loaders, device, args, out_dir)

    summary = {
        "experiment": "ablations",
        "protocol": "train/val/test; checkpoint selected on val F1 (tie: PR-AUC); test evaluated once",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "config": vars(args),
        "ablation_overrides": {a: ABLATIONS[a] for a in abls},
        "sizes": loaders["sizes"],
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
