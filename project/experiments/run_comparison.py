#!/usr/bin/env python3
"""
Track A vs Track B: Apples-to-Apples Comparison.
Runs MUAD (Track A) and Graph-Aware MoE (Track B) on the SAME data,
same seeds, same hyperparameters. Reports mean+/-std F1 and paired statistical test.
"""
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "vendor/muad"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vendor.muad.data.utils import load_data, create_dataloader
from vendor.muad.data.dataset import ChunkDataset
from vendor.muad.utils.general_utils import seed_everything
from vendor.muad.training.base_model import BaseModel
from tracks.proposed_model import GraphAwareMoE, DEFAULT_CONFIG

# Disable Flash SDPA and mem-efficient SDPA — known NaN issues in backward pass
# with Transformer attention. Forces standard (stable) attention implementation.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

EVAL_EVERY = 5
WEIGHT_DECAY = 1e-5


def load_common_data(data_dir: Path, batch_size: int):
    """Load train/test data once, shared by both models."""
    train_data, node_num, edges = load_data(
        str(data_dir / "chunk_train.pkl"),
        str(data_dir / "metadata.json")
    )
    test_data, _, _ = load_data(
        str(data_dir / "chunk_test.pkl"),
        str(data_dir / "metadata.json")
    )

    train_dataset = ChunkDataset(train_data, list(train_data.keys()), node_num, edges)
    test_dataset = ChunkDataset(test_data, list(test_data.keys()), node_num, edges)

    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = create_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, len(train_data), len(test_data)


def run_track_a(seed, train_loader, test_loader, output_dir, epochs, lr, event_num, device):
    """Run MUAD (Track A) with a single seed."""
    seed_everything(seed)
    device = torch.device(device)

    model = BaseModel(
        device=device,
        lr=lr,
        epochs=epochs,
        patience=15,
        result_dir=str(output_dir / f"track_a_seed{seed}"),
        hash_id=f"comparison_track_a_seed{seed}",
        hidden_dim=[64, 64],
        event_num=event_num,
    )

    best_f1, best_epoch = model.fit(train_loader, test_loader, evaluation_epoch=EVAL_EVERY)
    final_results = model.evaluate(test_loader)

    return {
        "seed": seed,
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "final_f1": final_results["F1"],
        "final_precision": final_results["Pre"],
        "final_recall": final_results["Rec"],
    }


def run_track_b(seed, train_loader, test_loader, output_dir, epochs, lr, batch_size, weight_decay, log_dim, device):
    """Run Graph-Aware MoE (Track B) with a single seed."""
    seed_everything(seed)
    device = torch.device(device)

    config = DEFAULT_CONFIG.copy()
    config.pop("epochs", None)
    # log_dim uses DEFAULT_CONFIG default (14) which matches actual data
    model = GraphAwareMoE(**config).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    best_f1 = 0.0
    best_epoch = 0
    nan_batches = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for graph, labels in train_loader:
            graph, labels = graph.to(device), labels.to(device)
            binary_labels = (labels >= 1).long()

            optimizer.zero_grad()
            outputs = model(graph, binary_labels)
            logits = outputs["MMlogit"]
            loss = outputs["loss"]

            # Skip batch if model outputs or loss are non-finite
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                nan_batches += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            train_loss += loss.item()

        if nan_batches > 0:
            print(f"  Warning: {nan_batches} batches skipped due to NaN/Inf loss in epoch {epoch}")
            nan_batches = 0

        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        test_loss = 0.0
        with torch.no_grad():
            for graph, labels in test_loader:
                graph, labels = graph.to(device), labels.to(device)
                binary_labels = (labels >= 1).long()
                outputs = model(graph, binary_labels)
                loss = outputs.get("loss", criterion(outputs["MMlogit"], binary_labels))
                test_loss += loss.item()
                preds = outputs["y_pred"]
                probs = torch.softmax(outputs["MMlogit"], dim=-1)[:, 1]
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(binary_labels.cpu().tolist())
                all_probs.extend(probs.cpu().tolist())

        avg_test_loss = test_loss / len(test_loader)

        from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
        f1 = f1_score(all_labels, all_preds, average="binary")
        precision = precision_score(all_labels, all_preds, average="binary")
        recall = recall_score(all_labels, all_preds, average="binary")
        roc_auc = roc_auc_score(all_labels, all_probs)
        pr_auc = average_precision_score(all_labels, all_probs)

        print(f"  Epoch {epoch:2d}: Train Loss={train_loss / len(train_loader):.4f}, Test Loss={avg_test_loss:.4f}, F1={f1:.4f}, Pre={precision:.4f}, Rec={recall:.4f}, ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch

    return {
        "seed": seed,
        "best_f1": best_f1,
        "best_epoch": best_epoch,
    }


def main():
    parser = argparse.ArgumentParser(description="Track A vs Track B Apples-to-Apples Comparison")
    parser.add_argument("--seeds", type=str, default="42",
                        help="Comma-separated list of seeds (default: 42 for single-seed test)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to data dir with chunk_train.pkl/chunk_test.pkl/metadata.json")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Base output directory (default: experiments/runs)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size (default: 50)")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate (default: 0.0005)")
    parser.add_argument("--event-num", type=int, default=15, help="Number of event templates (default: 15)")
    parser.add_argument("--log-dim", type=int, default=14, help="Log dim for Track B (default: 14)")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu (default: cuda)")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    project_root = Path(__file__).resolve().parent.parent
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = project_root / "recovered_artifact"

    if args.output_dir:
        output_base = Path(args.output_dir) / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        output_base = project_root / "experiments" / "runs" / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_base.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TRACK A vs TRACK B - APPLES-TO-APPLES COMPARISON")
    print("=" * 70)
    print(f"Data: {data_dir} (event_num={args.event_num})")
    print(f"Seeds: {seeds} ({len(seeds)} seed(s))")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}, Weight decay: {WEIGHT_DECAY}")
    print(f"Eval every {EVAL_EVERY} epochs")
    print()

    # Load shared data once
    train_loader, test_loader, n_train, n_test = load_common_data(data_dir, args.batch_size)
    print(f"Train chunks: {n_train}, Test chunks: {n_test}")
    print()

    track_a_results = []
    track_b_results = []

    for seed in seeds:
        print(f"{'=' * 70}")
        print(f"SEED {seed}")
        print(f"{'=' * 70}")

        # Track A
        print("\n--- Track A (MUAD Baseline) ---")
        try:
            res_a = run_track_a(seed, train_loader, test_loader, output_base,
                                args.epochs, args.lr, args.event_num, args.device)
            track_a_results.append(res_a)
            print(f"  Best F1: {res_a['best_f1']:.4f} (epoch {res_a['best_epoch']})")
            print(f"  Final F1: {res_a['final_f1']:.4f}, Pre: {res_a['final_precision']:.4f}, Rec: {res_a['final_recall']:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            track_a_results.append({"seed": seed, "error": str(e)})

        # Track B
        print("\n--- Track B (Graph-Aware MoE) ---")
        try:
            res_b = run_track_b(seed, train_loader, test_loader, output_base,
                                args.epochs, args.lr, args.batch_size, WEIGHT_DECAY, args.log_dim, args.device)
            track_b_results.append(res_b)
            print(f"  Best F1: {res_b['best_f1']:.4f} (epoch {res_b['best_epoch']})")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            track_b_results.append({"seed": seed, "error": str(e)})

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - TRACK A vs TRACK B")
    print("=" * 70)

    successful_a = [r for r in track_a_results if "error" not in r]
    successful_b = [r for r in track_b_results if "error" not in r]

    if successful_a and successful_b:
        f1s_a = [r["best_f1"] for r in successful_a]
        f1s_b = [r["best_f1"] for r in successful_b]

        mean_a = np.mean(f1s_a)
        std_a = np.std(f1s_a)
        mean_b = np.mean(f1s_b)
        std_b = np.std(f1s_b)

        # Paired t-test (same seeds)
        t_stat, p_value = stats.ttest_rel(f1s_b, f1s_a)

        print(f"\nTrack A (MUAD):      Mean F1 = {mean_a:.4f} +/- {std_a:.4f}")
        print(f"Track B (Graph MoE): Mean F1 = {mean_b:.4f} +/- {std_b:.4f}")
        print(f"Improvement:         {mean_b - mean_a:+.4f} ({(mean_b - mean_a) / mean_a * 100:+.2f}%)")
        print(f"Paired t-test:       t={t_stat:.4f}, p={p_value:.6f}")
        print(f"Significant (p<0.05): {'YES' if p_value < 0.05 else 'NO'}")

        print(f"\nPer-seed comparison:")
        print(f"{'Seed':<8} {'Track A':<10} {'Track B':<10} {'Diff':<10}")
        for ra, rb in zip(successful_a, successful_b):
            diff = rb["best_f1"] - ra["best_f1"]
            print(f"{ra['seed']:<8} {ra['best_f1']:<10.4f} {rb['best_f1']:<10.4f} {diff:+.4f}")

        summary = {
            "experiment": "track_a_vs_track_b_comparison",
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "data_dir": str(data_dir),
            "config": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": WEIGHT_DECAY,
                "eval_every": EVAL_EVERY,
                "seeds": seeds,
                "event_num": args.event_num,
                "log_dim": args.log_dim,
            },
            "track_a": {
                "mean_f1": float(mean_a),
                "std_f1": float(std_a),
                "best_f1s": f1s_a,
                "results": successful_a,
            },
            "track_b": {
                "mean_f1": float(mean_b),
                "std_f1": float(std_b),
                "best_f1s": f1s_b,
                "results": successful_b,
            },
            "statistics": {
                "improvement": float(mean_b - mean_a),
                "improvement_pct": float((mean_b - mean_a) / mean_a * 100),
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05),
            },
        }
    else:
        print("Insufficient successful runs for comparison.")
        summary = {"error": "Insufficient successful runs"}

    with open(output_base / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_base}")


if __name__ == "__main__":
    main()
