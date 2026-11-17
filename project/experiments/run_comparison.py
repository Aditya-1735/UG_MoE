#!/usr/bin/env python3
"""
Track A (MUAD) vs Track B (Graph-Aware MoE) under the shared protocol in experiments/common.py.

* Same data, same train/val/test split (val carved from chunk_train.pkl with a fixed split seed),
  same seeds, same optimizer / lr / weight decay / epochs / early stopping for both tracks.
* Track A is trained with MUAD's own objective (MMLoss + 0.6 * confidence loss + KL), i.e. the
  'loss' returned by vendor/muad/models/main_model.py.
* The checkpoint is selected on validation F1 (ties: PR-AUC); test is evaluated once.
* Paired t-test over seeds on test F1 and test macro-F1.

Usage (from project/):
    python experiments/run_comparison.py --seeds 42 --device cuda               # smoke test
    python experiments/run_comparison.py --seeds 42,123,456,789,999 --device cuda
"""
import argparse
import json
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from common import (
    DEFAULT_DATA_DIR,
    PROJECT_ROOT,
    configure_torch,
    load_splits,
    seed_everything,
    summarize,
    train_and_select,
    trivial_baseline_metrics,
)
from tracks.proposed_model import DEFAULT_CONFIG, GraphAwareMoE
from vendor.muad.models.main_model import MainModel


def build_track_a(metadata, device):
    return MainModel(device, hidden_dim=[64, 64], event_num=metadata["event_num"])


def build_track_b(metadata, device):
    config = DEFAULT_CONFIG.copy()
    config["log_dim"] = metadata["event_num"]
    config["metric_dim"] = metadata["metric_num"]
    config["num_nodes"] = metadata["node_num"]
    return GraphAwareMoE(**config)


TRACKS = {
    "track_a": ("MUAD baseline", build_track_a),
    "track_b": ("Graph-Aware MoE", build_track_b),
}


def paired_test(res_a, res_b, metric):
    a = [r["test"][metric] for r in res_a]
    b = [r["test"][metric] for r in res_b]
    if len(a) < 2:
        return {"note": "need >= 2 seeds for a paired t-test"}
    t, p = stats.ttest_rel(b, a)
    return {
        "metric": metric,
        "mean_diff_b_minus_a": float(np.mean(b) - np.mean(a)),
        "t_statistic": float(t),
        "p_value": float(p),
        "significant_p<0.05": bool(p < 0.05),
    }


def main():
    parser = argparse.ArgumentParser(description="Track A vs Track B comparison (validation-selected)")
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated seeds")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "experiments" / "runs"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=15, help="Early-stopping patience on val (<=0 disables)")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--tracks", type=str, default="track_a,track_b")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    configure_torch()
    seeds = [int(s) for s in args.seeds.split(",")]
    tracks = [t.strip() for t in args.tracks.split(",")]
    device = torch.device(args.device)
    patience = args.patience if args.patience > 0 else None

    out_dir = Path(args.output_dir) / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    loaders = load_splits(
        Path(args.data_dir), args.batch_size, args.val_ratio, args.split_seed, split_out=out_dir / "split.json"
    )
    metadata = loaders["metadata"]
    print("=" * 70)
    print("TRACK A vs TRACK B (selection on validation, test evaluated once)")
    print("=" * 70)
    print(f"Data: {args.data_dir}")
    print(f"Sizes: {loaders['sizes']}  culprit distribution: {loaders['culprit_dist']}")
    print(f"Seeds: {seeds} | epochs={args.epochs} lr={args.lr} wd={args.weight_decay} "
          f"batch={args.batch_size} patience={patience}")

    trivial = trivial_baseline_metrics(loaders["test"])
    print(f"Trivial 'always anomalous' on test: F1={trivial['f1']:.4f} macroF1={trivial['macro_f1']:.4f}")

    results = {t: [] for t in tracks}
    errors = {t: [] for t in tracks}
    for seed in seeds:
        for track in tracks:
            name, builder = TRACKS[track]
            print(f"\n--- {track} ({name}) | seed {seed} ---")
            try:
                seed_everything(seed)
                model = builder(metadata, device)
                res = train_and_select(
                    model, loaders, device,
                    epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                    patience=patience, ckpt_path=out_dir / f"{track}_seed{seed}_best.pth",
                    log_prefix="  ",
                )
                res["seed"] = seed
                results[track].append(res)
                with open(out_dir / f"{track}_seed{seed}.json", "w") as f:
                    json.dump(res, f, indent=2)
            except Exception as e:  # keep the other seeds/tracks running
                traceback.print_exc()
                errors[track].append({"seed": seed, "error": repr(e)})

    summary = {
        "experiment": "track_a_vs_track_b_comparison",
        "protocol": "train/val/test; checkpoint selected on val F1 (tie: PR-AUC); test evaluated once",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "data_dir": args.data_dir,
        "sizes": loaders["sizes"],
        "culprit_dist": loaders["culprit_dist"],
        "config": vars(args),
        "trivial_always_anomalous_test": trivial,
        "tracks": {
            t: {
                "test": summarize(results[t], "test"),
                "val": summarize(results[t], "val"),
                "best_epochs": [r["best_epoch"] for r in results[t]],
                "errors": errors[t],
            }
            for t in tracks
        },
    }
    if "track_a" in results and "track_b" in results:
        by_seed_a = {r["seed"]: r for r in results["track_a"]}
        paired = [(by_seed_a[r["seed"]], r) for r in results["track_b"] if r["seed"] in by_seed_a]
        ra, rb = [p[0] for p in paired], [p[1] for p in paired]
        summary["paired_tests"] = {m: paired_test(ra, rb, m) for m in ("f1", "macro_f1", "pr_auc")}

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY (test metrics at the validation-selected epoch)")
    print("=" * 70)
    for t in tracks:
        s = summary["tracks"][t]["test"]
        if s:
            print(f"{t}: F1={s['f1']['mean']:.4f}±{s['f1']['std']:.4f}  "
                  f"macroF1={s['macro_f1']['mean']:.4f}±{s['macro_f1']['std']:.4f}  "
                  f"F1_normal={s['f1_normal']['mean']:.4f}  PR-AUC={s['pr_auc']['mean']:.4f}")
    for m, r in summary.get("paired_tests", {}).items():
        print(f"paired t-test [{m}]: {r}")
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
