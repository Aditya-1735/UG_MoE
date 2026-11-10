#!/usr/bin/env python3
"""
Noise Robustness Experiments - Test model robustness under Gaussian noise injection.
Injects noise into metric/trace/log features at varying levels and reports
performance degradation.
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


def inject_noise(graph, noise_level, modality):
    """Return a copy of graph with Gaussian noise added to specified modality."""
    import dgl
    g = graph.clone()
    
    if modality == "metrics":
        feat = g.ndata["metrics"]
        noise = torch.randn_like(feat) * noise_level
        g.ndata["metrics"] = feat + noise
    elif modality == "traces":
        feat = g.ndata["traces"]
        noise = torch.randn_like(feat) * noise_level
        g.ndata["traces"] = feat + noise
    elif modality == "logs":
        feat = g.ndata["logs"]
        noise = torch.randn_like(feat) * noise_level
        g.ndata["logs"] = feat + noise
    
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
    """Run noise robustness experiment with given seed."""
    print(f"\n{'='*60}")
    print(f"NOISE ROBUSTNESS - Seed: {seed}")
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
    config['log_dim'] = 15
    model = GraphAwareMoE(**config).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Train full model first
    print("  Training full model (no noise)...")
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
    
    # Evaluate on clean test data
    clean_metrics = evaluate_model(model, test_loader, device)
    print(f"  Clean model: F1={clean_metrics['f1']:.4f}, ROC-AUC={clean_metrics['roc_auc']:.4f}, PR-AUC={clean_metrics['pr_auc']:.4f}")
    
    # Test with noise at different levels
    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5]
    modalities = ["metrics", "traces", "logs"]
    
    results = {
        "seed": seed,
        "clean_model": clean_metrics,
        "noise_results": {},
    }
    
    for modality in modalities:
        results["noise_results"][modality] = {}
        for noise_level in noise_levels:
            print(f"  Testing noise: modality={modality}, level={noise_level}")
            
            # Create noisy test data
            noisy_test_data = []
            for graph, labels in test_dataset:
                g_noisy = inject_noise(graph, noise_level, modality)
                noisy_test_data.append((g_noisy, labels))
            
            noisy_test_loader = create_dataloader(noisy_test_data, batch_size=50, shuffle=False)
            
            # Evaluate with noise
            noisy_metrics = evaluate_model(model, noisy_test_loader, device)
            
            results["noise_results"][modality][str(noise_level)] = noisy_metrics
            print(f"    Level {noise_level}: F1={noisy_metrics['f1']:.4f}, "
                  f"ROC-AUC={noisy_metrics['roc_auc']:.4f}, PR-AUC={noisy_metrics['pr_auc']:.4f}")
    
    return results


def main():
    print("="*60)
    print("NOISE ROBUSTNESS EXPERIMENTS")
    print("="*60)
    print("Config: hidden_dim=64, epochs=50, lr=0.001, batch=50")
    print("Data: recovered_artifact (event_num=15)")
    print("Seeds: 42, 123, 456, 789, 999")
    print("Noise levels: 0.01, 0.05, 0.1, 0.2, 0.5")
    print("Modalities: metrics, traces, logs")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(__file__).parent.parent / "experiments" / "runs" / f"noise_robustness_{timestamp}"
    output_base.mkdir(parents=True, exist_ok=True)
    
    seeds = [42, 123, 456, 789, 999]
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
    print("SUMMARY - NOISE ROBUSTNESS EXPERIMENTS")
    print(f"{'='*60}")
    
    successful = [r for r in all_results if "error" not in r]
    if successful:
        clean_f1s = [f"{r['clean_model']['f1']:.4f}" for r in successful]
        print(f"Clean model F1: {', '.join(clean_f1s)}")
        
        for modality in ["metrics", "traces", "logs"]:
            print(f"\nModality={modality}:")
            for noise_level in [0.01, 0.05, 0.1, 0.2, 0.5]:
                f1_vals = [r['noise_results'][modality][str(noise_level)]['f1'] for r in successful]
                f1s = [f"{v:.4f}" for v in f1_vals]
                mean_f1 = np.mean(f1_vals)
                std_f1 = np.std(f1_vals)
                print(f"  Level {noise_level}: Mean F1={mean_f1:.4f} ± {std_f1:.4f}")
    
    summary = {
        "experiment": "noise_robustness",
        "timestamp": timestamp,
        "config": {
            "hidden_dim": 64,
            "epochs": 50,
            "lr": 0.001,
            "batch_size": 50,
            "event_num": 15,
            "seeds": seeds,
            "noise_levels": [0.01, 0.05, 0.1, 0.2, 0.5],
            "modalities": ["metrics", "traces", "logs"],
        },
        "results": all_results,
    }
    
    with open(output_base / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_base}")


if __name__ == "__main__":
    main()
