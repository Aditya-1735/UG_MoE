#!/usr/bin/env python3
"""
Track B: Graph-Aware MoE (Proposed Model) - Full Experiments
Runs 5 seeds × 50 epochs on OUR regenerated data (event_num=14)
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

def run_single_experiment(run_id: int, seed: int, output_base: Path):
    """Run single Track B experiment with given seed."""
    print(f"\n{'='*60}")
    print(f"RUN {run_id}/5 - Seed: {seed}")
    print(f"{'='*60}")
    
    seed_everything(seed)
    
    # Load OUR regenerated data (event_num=14)
    data_dir = Path(__file__).parent.parent / "preprocessing/muad_compat/output"
    train_data, node_num, edges = load_data(
        str(data_dir / "chunk_train.pkl"),
        str(data_dir / "metadata.json")
    )
    test_data, _, _ = load_data(
        str(data_dir / "chunk_test.pkl"),
        str(data_dir / "metadata.json")
    )
    
    print(f"  Train: {len(train_data)} chunks, Test: {len(test_data)} chunks")
    
    train_dataset = ChunkDataset(train_data, list(train_data.keys()), node_num, edges)
    test_dataset = ChunkDataset(test_data, list(test_data.keys()), node_num, edges)
    
    train_loader = create_dataloader(train_dataset, batch_size=50, shuffle=True)
    test_loader = create_dataloader(test_dataset, batch_size=50, shuffle=False)
    
    device = torch.device("cpu")
    config = DEFAULT_CONFIG.copy()
    config['log_dim'] = 14  # Our regenerated data has 14 templates
    model_config = {k: v for k, v in config.items() if k != 'epochs'}
    
    model = GraphAwareMoE(**model_config).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_f1 = 0
    best_epoch = 0
    epoch_results = []
    
    print(f"  Training for 50 epochs...")
    
    for epoch in range(1, 51):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for graph, labels in train_loader:
            graph, labels = graph.to(device), labels.to(device)
            binary_labels = (labels >= 1).long()
            
            optimizer.zero_grad()
            outputs = model(graph, binary_labels)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = outputs["y_pred"]
            train_correct += (preds == binary_labels).sum().item()
            train_total += labels.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # Evaluation every 5 epochs (matching MUAD evaluation_epoch=5)
        if epoch % 5 == 0:
            model.eval()
            test_loss = 0
            test_correct = 0
            test_total = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for graph, labels in test_loader:
                    graph, labels = graph.to(device), labels.to(device)
                    binary_labels = (labels >= 1).long()
                    
                    outputs = model(graph, binary_labels)
                    loss = outputs.get("loss", criterion(outputs["MMlogit"], binary_labels))
                    
                    test_loss += loss.item()
                    preds = outputs["y_pred"]
                    test_correct += (preds == binary_labels).sum().item()
                    test_total += labels.size(0)
                    
                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(binary_labels.cpu().tolist())
            
            avg_test_loss = test_loss / len(test_loader)
            test_acc = test_correct / test_total
            
            from sklearn.metrics import f1_score, precision_score, recall_score
            f1 = f1_score(all_labels, all_preds, average='binary')
            precision = precision_score(all_labels, all_preds, average='binary')
            recall = recall_score(all_labels, all_preds, average='binary')
            
            print(f"  Epoch {epoch:2d}: Train Loss={avg_train_loss:.4f}, Acc={train_acc:.4f} | "
                  f"Test Loss={avg_test_loss:.4f}, Acc={test_acc:.4f}, "
                  f"F1={f1:.4f}, Pre={precision:.4f}, Rec={recall:.4f}")
            
            epoch_results.append({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "train_acc": train_acc,
                "test_loss": avg_test_loss,
                "test_acc": test_acc,
                "f1": f1,
                "precision": precision,
                "recall": recall
            })
            
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
                # Save best model
                torch.save(model.state_dict(), output_base / f"run_{run_id}_best.pth")
        else:
            # Quick eval without full metrics
            model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for graph, labels in test_loader:
                    graph, labels = graph.to(device), labels.to(device)
                    binary_labels = (labels >= 1).long()
                    outputs = model(graph, binary_labels)
                    preds = outputs["y_pred"]
                    test_correct += (preds == binary_labels).sum().item()
                    test_total += labels.size(0)
            
            test_acc = test_correct / test_total
            print(f"  Epoch {epoch:2d}: Train Loss={avg_train_loss:.4f}, Acc={train_acc:.4f} | Test Acc={test_acc:.4f}")
    
    # Final evaluation
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
    
    from sklearn.metrics import f1_score, precision_score, recall_score
    final_f1 = f1_score(all_labels, all_preds, average='binary')
    final_pre = precision_score(all_labels, all_preds, average='binary')
    final_rec = recall_score(all_labels, all_preds, average='binary')
    
    results = {
        "run_id": run_id,
        "seed": seed,
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "final_f1": final_f1,
        "final_precision": final_pre,
        "final_recall": final_rec,
        "epoch_results": epoch_results
    }
    
    with open(output_base / f"run_{run_id}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"  Best F1: {best_f1:.4f} (epoch {best_epoch})")
    print(f"  Final F1: {final_f1:.4f}, Pre: {final_pre:.4f}, Rec: {final_rec:.4f}")
    
    return results

def main():
    print("="*60)
    print("TRACK B: GRAPH-AWARE MoE - FULL EXPERIMENTS")
    print("="*60)
    print("Config: hidden_dim=64, epochs=50, lr=0.001, batch=50, weight_decay=1e-5")
    print("Data: Our regenerated data (event_num=14)")
    print("5 runs with seeds: 42, 123, 456, 789, 999")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path("project/experiments/runs") / f"track_b_full_{timestamp}"
    output_base.mkdir(parents=True, exist_ok=True)
    
    seeds = [42, 123, 456, 789, 999]
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
    print("SUMMARY - TRACK B FULL")
    print(f"{'='*60}")
    
    successful = [r for r in all_results if "error" not in r]
    if successful:
        f1s = [r["best_f1"] for r in successful]
        print(f"Successful runs: {len(successful)}/5")
        print(f"Best F1s: {[f'{f:.4f}' for f in f1s]}")
        print(f"Mean F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(f"Max F1: {np.max(f1s):.4f}, Min F1: {np.min(f1s):.4f}")
    else:
        print("No successful runs")
    
    summary = {
        "experiment": "track_b_full",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "config": {
            "hidden_dim": 64,
            "epochs": 50,
            "lr": 0.001,
            "batch_size": 50,
            "weight_decay": 1e-5,
            "event_num": 14,
            "seeds": seeds,
            "router_type": "uncertainty_guided",
            "fusion_type": "confidence",
            "uncertainty_type": "variational"
        },
        "results": all_results
    }
    
    with open(output_base / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_base}")

if __name__ == "__main__":
    main()