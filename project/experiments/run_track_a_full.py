#!/usr/bin/env python3
"""
Track A: MUAD Faithful Reproduction - Full-label Experiments
Runs 5 seeds × 50 epochs on recovered artifact (commit 6031ccf config)
"""
import sys
import os
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "vendor/muad"))

from vendor.muad.training.base_model import BaseModel
from vendor.muad.data.utils import load_data, create_dataloader
from vendor.muad.data.dataset import ChunkDataset
from vendor.muad.utils.general_utils import seed_everything

def run_single_experiment(run_id: int, seed: int, output_base: Path):
    """Run single full-label experiment with given seed."""
    print(f"\n{'='*60}")
    print(f"RUN {run_id}/5 - Seed: {seed}")
    print(f"{'='*60}")
    
    # Set seed for reproducibility
    seed_everything(seed)
    
    # Load RECOVERED artifact (commit 6031ccf)
    data_dir = Path(__file__).parent.parent / "recovered_artifact"
    train_data, node_num, edges = load_data(
        str(data_dir / "chunk_train.pkl"),
        str(data_dir / "metadata.json")
    )
    test_data, _, _ = load_data(
        str(data_dir / "chunk_test.pkl"),
        str(data_dir / "metadata.json")
    )
    
    # Datasets
    train_dataset = ChunkDataset(train_data, list(train_data.keys()), node_num, edges)
    test_dataset = ChunkDataset(test_data, list(test_data.keys()), node_num, edges)
    
    train_loader = create_dataloader(train_dataset, batch_size=50, shuffle=True)
    test_loader = create_dataloader(test_dataset, batch_size=50, shuffle=False)
    
    # Model - commit 6031ccf config
    device = torch.device("cpu")
    model = BaseModel(
        device=device,
        lr=0.001,
        epochs=50,
        patience=15,
        result_dir=str(output_base / f"run_{run_id}"),
        hash_id=f"faithful_full_run{run_id}",
        hidden_dim=[64, 64],
        event_num=15  # Recovered artifact has 15 templates
    )
    
    # Training
    print(f"  Training on {len(train_data)} chunks, testing on {len(test_data)} chunks")
    best_f1, best_epoch = model.fit(train_loader, test_loader, evaluation_epoch=5)
    
    # Final evaluation
    final_results = model.evaluate(test_loader)
    
    # Save results
    results = {
        "run_id": run_id,
        "seed": seed,
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "final_f1": final_results["F1"],
        "final_precision": final_results["Pre"],
        "final_recall": final_results["Rec"],
        "config": {
            "hidden_dim": [64, 64],
            "epochs": 50,
            "lr": 0.001,
            "batch_size": 50,
            "patience": 15,
            "event_num": 15,
            "evaluation_epoch": 5
        }
    }
    
    with open(output_base / f"run_{run_id}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"  Best F1: {best_f1:.4f} (epoch {best_epoch})")
    print(f"  Final F1: {final_results['F1']:.4f}, Pre: {final_results['Pre']:.4f}, Rec: {final_results['Rec']:.4f}")
    
    return results

def main():
    print("="*60)
    print("TRACK A: MUAD FAITHFUL REPRODUCTION - FULL-LABEL")
    print("="*60)
    print("Config: hidden_dim=[64,64], epochs=50, lr=0.001, batch=50")
    print("Data: Recovered artifact (commit 6031ccf), event_num=15")
    print("5 runs with seeds: 42, 123, 456, 789, 999")
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path("project/experiments/runs") / f"track_a_full_{timestamp}"
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Seeds for 5 runs
    seeds = [42, 123, 456, 789, 999]
    all_results = []
    
    for i, seed in enumerate(seeds, 1):
        try:
            results = run_single_experiment(i, seed, output_base)
            all_results.append(results)
        except Exception as e:
            print(f"  ERROR in run {i}: {e}")
            all_results.append({"run_id": i, "seed": seed, "error": str(e)})
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY - TRACK A FULL-LABEL")
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
    
    # Save summary
    summary = {
        "experiment": "track_a_full_label",
        "timestamp": timestamp,
        "config": {
            "hidden_dim": [64, 64],
            "epochs": 50,
            "lr": 0.001,
            "batch_size": 50,
            "patience": 15,
            "event_num": 15,
            "seeds": seeds
        },
        "results": all_results
    }
    
    with open(output_base / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_base}")

if __name__ == "__main__":
    main()