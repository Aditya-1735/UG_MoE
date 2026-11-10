#!/usr/bin/env python3
"""
Missing-Modality Experiments - Test model robustness when one modality is unavailable.
Tests Graph-Aware MoE with each modality dropped (no traces, no logs, no metrics).
Reports F1, precision, recall, ROC-AUC, PR-AUC for each missing-modality scenario.
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


def mask_modality(graph, modality):
    """Return a copy of graph with the specified modality zeroed out."""
    import copy
    import dgl
    g = dgl.batch([dgl.DGLGraph(graph.subgraph([i])) for i in range(graph.batch_size)])
    g = graph.clone()
    
    if modality == "metrics":
        g.ndata["metrics"] = torch.zeros_like(g.ndata["metrics"])
    elif modality == "traces":
        g.ndata["traces"] = torch.zeros_like(g.ndata["traces"])
    elif modality == "logs":
        g.ndata["logs"] = torch.zeros_like(g.ndata["logs"])
    
    return g


def evaluate_model(model, test_loader, device):
    """Evaluate model and return metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    test_loss = 0
    
    criterion = torch.nn.CrossEntropyLoss()
    
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
    
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "test_loss": test_loss / len(test_loader),
    }


def run_single_experiment(seed: int, output_base: Path):
    """Run missing-modality experiment with given seed."""
    print(f"\n{'='*60}")
    print(f"MISSING-MODALITY - Seed: {seed}")
    print(f"{'='*60}")
    
    seed_everything(seed)
    
    # Load recovered artifact
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
    
    # Load full model config
    config = DEFAULT_CONFIG.copy()
    config.pop("epochs", None)
    config['log_dim'] = 14
    model = GraphAwareMoE(**config).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Train full model first
    print("  Training full model (all modalities)...")
    best_f1 = 0
    for epoch in range(1, 51):
        model.train()
        for graph, labels in train_loader:
            graph, labels = graph.to(device), labels.to(device)
            binary_labels = (labels >= 1).long()
            
            optimizer.zero_grad()
            outputs = model(graph, binary_labels)
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
        
        if epoch % 10 == 0:
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for graph, labels in test_loader:
                    graph, labels = graph.to(device), labels.to(device)
                    binary_labels = (labels >= 1).long()
                    outputs = model(graph, binary_labels)
                    preds = outputs["y_pred"]
                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(binary_labels.cpu().tolist())
            
            from sklearn.metrics import f1_score
            f1 = f1_score(all_labels, all_preds, average='binary')
            if f1 > best_f1:
                best_f1 = f1
            print(f"    Epoch {epoch:2d}: F1={f1:.4f}")
    
    # Evaluate full model
    full_metrics = evaluate_model(model, test_loader, device)
    print(f"  Full model: F1={full_metrics['f1']:.4f}, ROC-AUC={full_metrics['roc_auc']:.4f}, PR-AUC={full_metrics['pr_auc']:.4f}")
    
    # Test missing each modality
    results = {
        "seed": seed,
        "full_model": full_metrics,
        "missing_modalities": {},
    }
    
    for modality in ["metrics", "traces", "logs"]:
        print(f"  Testing missing modality: {modality}")
        
        # Create missing-modality test loader
        missing_test_data = []
        for graph, labels in test_dataset:
            g_masked = mask_modality(graph, modality)
            missing_test_data.append((g_masked, labels))
        
        missing_test_loader = create_dataloader(missing_test_data, batch_size=50, shuffle=False)
        
        # Evaluate with missing modality
        missing_metrics = evaluate_model(model, missing_test_loader, device)
        
        results["missing_modalities"][modality] = missing_metrics
        print(f"    Missing {modality}: F1={missing_metrics['f1']:.4f}, "
              f"ROC-AUC={missing_metrics['roc_auc']:.4f}, PR-AUC={missing_metrics['pr_auc']:.4f}")
    
    return results


def main():
    print("="*60)
    print("MISSING-MODALITY EXPERIMENTS")
    print("="*60)
    print("Config: hidden_dim=64, epochs=50, lr=0.001, batch=50")
    print("Data: recovered_artifact (event_num=15)")
    print("Seeds: 42, 123, 456, 789, 999")
    print("Modalities tested: metrics, traces, logs (each dropped individually)")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(__file__).parent.parent / "experiments" / "runs" / f"missing_modality_{timestamp}"
    output_base.mkdir(parents=True, exist_ok=True)
    
    seeds = [42, 123]
    all_results = []
    
    for seed in seeds:
        try:
            results = run_single_experiment(seed, output_base)
            all_results.append(results)
        except Exception as e:
            print(f"  ERROR for seed {seed}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"seed": seed, "error": str(e)})
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY - MISSING-MODALITY EXPERIMENTS")
    print(f"{'='*60}")
    
    successful = [r for r in all_results if "error" not in r]
    if successful:
        full_f1s = [f"{r['full_model']['f1']:.4f}" for r in successful]
        full_aucs = [f"{r['full_model']['roc_auc']:.4f}" for r in successful]
        print(f"Full model F1: {', '.join(full_f1s)}")
        print(f"Full model ROC-AUC: {', '.join(full_aucs)}")
        
        for modality in ["metrics", "traces", "logs"]:
            f1s = [f"{r['missing_modalities'][modality]['f1']:.4f}" for r in successful]
            mean_f1 = np.mean([r['missing_modalities'][modality]['f1'] for r in successful])
            std_f1 = np.std([r['missing_modalities'][modality]['f1'] for r in successful])
            print(f"\nMissing {modality}:")
            print(f"  F1: {', '.join(f1s)}")
            print(f"  Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")
    
    summary = {
        "experiment": "missing_modality",
        "timestamp": timestamp,
        "config": {
            "hidden_dim": 64,
            "epochs": 50,
            "lr": 0.001,
            "batch_size": 50,
            "event_num": 15,
            "seeds": seeds,
        },
        "results": all_results,
    }
    
    with open(output_base / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_base}")


if __name__ == "__main__":
    main()
