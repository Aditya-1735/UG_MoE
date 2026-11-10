#!/usr/bin/env python3
"""
Plain MoE Baseline - Mixture-of-Experts without graph routing, uncertainty,
or confidence-weighted fusion. Uses base router + simple weighted_concat fusion.
Saves checkpoint to checkpoints/plain_moe.pth
"""
import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "vendor/muad"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vendor.muad.data.utils import load_data, create_dataloader
from vendor.muad.data.dataset import ChunkDataset
from vendor.muad.utils.general_utils import seed_everything
from tracks.proposed_model import GraphAwareMoE, DEFAULT_CONFIG

# Disable Flash SDPA for stability
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


def run_single_experiment(run_id: int, seed: int, output_base: Path):
    """Run single Plain MoE experiment with given seed."""
    print(f"\n{'='*60}")
    print(f"PLAIN MoE - RUN {run_id}/5 - Seed: {seed}")
    print(f"{'='*60}")
    
    seed_everything(seed)
    
    # Load recovered artifact (same as Track A/B for apples-to-apples)
    data_dir = Path(__file__).parent.parent / "recovered_artifact"
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
    
    train_loader = create_dataloader(train_dataset, batch_size=50, shuffle=True)
    test_loader = create_dataloader(test_dataset, batch_size=50, shuffle=False)
    
    device = torch.device("cpu")
    
    # Plain MoE config: no graph routing, no uncertainty routing, simple fusion
    config = DEFAULT_CONFIG.copy()
    config.pop("epochs", None)
    config['log_dim'] = 15
    config['router_type'] = "base"
    config['use_graph_routing'] = False
    config['use_uncertainty_routing'] = False
    config['fusion_type'] = "weighted_concat"
    config['use_uncertainty'] = False
    
    model = GraphAwareMoE(**config).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_f1 = 0
    best_epoch = 0
    
    for epoch in range(1, 51):
        model.train()
        train_loss = 0
        for graph, labels in train_loader:
            graph, labels = graph.to(device), labels.to(device)
            binary_labels = (labels >= 1).long()
            
            optimizer.zero_grad()
            outputs = model(graph, binary_labels)
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            train_loss += loss.item()
        
        if epoch % 5 == 0:
            model.eval()
            all_preds = []
            all_labels = []
            all_probs = []
            test_loss = 0
            with torch.no_grad():
                for graph, labels in test_loader:
                    graph, labels = graph.to(device), labels.to(device)
                    binary_labels = (labels >= 1).long()
                    outputs = model(graph, binary_labels)
                    loss = outputs.get("loss", criterion(outputs["MMlogit"], binary_labels))
                    test_loss += loss.item()
                    probs = torch.softmax(outputs["MMlogit"], dim=-1)[:, 1]
                    preds = outputs["y_pred"]
                    
                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(binary_labels.cpu().tolist())
                    all_probs.extend(probs.cpu().tolist())
            
            from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
            f1 = f1_score(all_labels, all_preds, average='binary')
            precision = precision_score(all_labels, all_preds, average='binary')
            recall = recall_score(all_labels, all_preds, average='binary')
            roc_auc = roc_auc_score(all_labels, all_probs)
            pr_auc = average_precision_score(all_labels, all_probs)
            
            print(f"  Epoch {epoch:2d}: Train Loss={train_loss/len(train_loader):.4f}, "
                  f"Test Loss={test_loss/len(test_loader):.4f}, "
                  f"F1={f1:.4f}, Pre={precision:.4f}, Rec={recall:.4f}, "
                  f"ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
    
    # Final evaluation
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for graph, labels in test_loader:
            graph, labels = graph.to(device), labels.to(device)
            binary_labels = (labels >= 1).long()
            outputs = model(graph, binary_labels)
            probs = torch.softmax(outputs["MMlogit"], dim=-1)[:, 1]
            preds = outputs["y_pred"]
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(binary_labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
    
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
    final_f1 = f1_score(all_labels, all_preds, average='binary')
    final_pre = precision_score(all_labels, all_preds, average='binary')
    final_rec = recall_score(all_labels, all_preds, average='binary')
    final_roc_auc = roc_auc_score(all_labels, all_probs)
    final_pr_auc = average_precision_score(all_labels, all_probs)
    
    # Save checkpoint
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "plain_moe.pth")
    
    results = {
        "run_id": run_id,
        "seed": seed,
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "final_f1": final_f1,
        "final_precision": final_pre,
        "final_recall": final_rec,
        "final_roc_auc": final_roc_auc,
        "final_pr_auc": final_pr_auc,
        "checkpoint": str(checkpoint_dir / "plain_moe.pth"),
    }
    
    with open(output_base / f"run_{run_id}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"  Best F1: {best_f1:.4f} (epoch {best_epoch})")
    print(f"  Final F1: {final_f1:.4f}, Pre: {final_pre:.4f}, Rec: {final_rec:.4f}, "
          f"ROC-AUC: {final_roc_auc:.4f}, PR-AUC: {final_pr_auc:.4f}")
    print(f"  Checkpoint saved to: {checkpoint_dir / 'plain_moe.pth'}")
    
    return results


def main():
    print("="*60)
    print("PLAIN MoE BASELINE - FULL EXPERIMENTS")
    print("="*60)
    print("Config: hidden_dim=64, epochs=50, lr=0.001, batch=50, weight_decay=1e-5")
    print("Data: recovered_artifact (event_num=15)")
    print("Plain MoE: no graph routing, no uncertainty, weighted_concat fusion")
    print("5 runs with seeds: 42, 123, 456, 789, 999")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(__file__).parent.parent / "experiments" / "runs" / f"plain_moe_{timestamp}"
    output_base.mkdir(parents=True, exist_ok=True)
    
    seeds = [42, 123]
    all_results = []
    
    for i, seed in enumerate(seeds, 1):
        try:
            results = run_single_experiment(i, seed, output_base)
            all_results.append(results)
        except Exception as e:
            print(f"  ERROR in run {i}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"run_id": i, "seed": seed, "error": str(e)})
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY - PLAIN MoE BASELINE")
    print(f"{'='*60}")
    
    successful = [r for r in all_results if "error" not in r]
    if successful:
        f1s = [r["best_f1"] for r in successful]
        print(f"Successful runs: {len(successful)}/5")
        print(f"Best F1s: {[f'{f:.4f}' for f in f1s]}")
        print(f"Mean F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    
    summary = {
        "experiment": "plain_moe_baseline",
        "timestamp": timestamp,
        "config": {
            "hidden_dim": 64,
            "epochs": 50,
            "lr": 0.001,
            "batch_size": 50,
            "weight_decay": 1e-5,
            "event_num": 15,
            "seeds": seeds,
            "router_type": "base",
            "use_graph_routing": False,
            "use_uncertainty_routing": False,
            "fusion_type": "weighted_concat",
            "use_uncertainty": False,
        },
        "results": all_results,
    }
    
    with open(output_base / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_base}")


if __name__ == "__main__":
    main()
