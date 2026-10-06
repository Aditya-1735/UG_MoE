#!/usr/bin/env python3
"""
Track A: MUAD Faithful Reproduction - Active Learning Experiments
Runs 5 seeds × 30 iterations × 5 epochs on recovered artifact (commit 6031ccf config)
"""
import sys
import os
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "vendor/muad"))

from vendor.muad.training.base_model import BaseModel
from vendor.muad.data.utils import load_data, create_dataloader
from vendor.muad.data.dataset import ChunkDataset
from vendor.muad.utils.general_utils import seed_everything, split_data
from vendor.muad.active_learning.strategies import entropy_based_selection, confidence_based_selection

def run_single_al_experiment(run_id: int, seed: int, output_base: Path):
    """Run single active learning experiment with given seed."""
    print(f"\n{'='*60}")
    print(f"RUN {run_id}/5 - Seed: {seed}")
    print(f"{'='*60}")
    
    seed_everything(seed)
    
    data_dir = Path(__file__).parent.parent / "recovered_artifact"
    train_data, node_num, edges = load_data(
        str(data_dir / "chunk_train.pkl"),
        str(data_dir / "metadata.json")
    )
    test_data, _, _ = load_data(
        str(data_dir / "chunk_test.pkl"),
        str(data_dir / "metadata.json")
    )
    
    # Active learning split: 10% labeled
    train_keys, val_keys = split_data(train_data, train_ratio=0.1)
    
    test_dataset = ChunkDataset(test_data, list(test_data.keys()), node_num, edges)
    test_loader = create_dataloader(test_dataset, batch_size=50, shuffle=False)
    
    device = torch.device("cpu")
    model = BaseModel(
        device=device,
        lr=0.001,
        epochs=5,  # 5 epochs per iteration as per commit 6031ccf
        patience=15,
        result_dir=str(output_base / f"run_{run_id}"),
        hash_id=f"faithful_al_run{run_id}",
        hidden_dim=[64, 64],
        event_num=15
    )
    
    best_overall_f1 = 0
    iteration_results = []
    
    for iteration in range(30):  # max_iter = 30 per commit 6031ccf
        print(f"  Iteration {iteration + 1}/30: Train={len(train_keys)}, Val={len(val_keys)}", end=" ")
        
        train_dataset = ChunkDataset(train_data, train_keys, node_num, edges)
        val_dataset = ChunkDataset(train_data, val_keys, node_num, edges)
        
        train_loader = create_dataloader(train_dataset, batch_size=50, shuffle=True)
        val_loader = create_dataloader(val_dataset, batch_size=50, shuffle=False)
        
        # Train
        best_f1, best_epoch = model.fit(train_loader, test_loader, evaluation_epoch=5)
        test_results = model.evaluate(test_loader)
        
        # Active learning selection
        uncertain_indices = entropy_based_selection(model.model, val_loader, device, n_samples=5)
        high_conf_indices, pseudo_labels = confidence_based_selection(model.model, val_loader, device, confidence_threshold=0.9)
        
        uncertain_indices = uncertain_indices[:min(200, len(val_keys))]
        
        # Remove selected from val_keys (reverse order to preserve indices)
        all_selected = sorted(set(uncertain_indices + high_conf_indices), reverse=True)
        for idx in all_selected:
            del val_keys[idx]
        
        # Add high-conf to train_keys (with the bug - uses already-modified val_keys)
        if high_conf_indices:
            # Bug: uses modified val_keys, not original
            high_conf_keys = [val_keys[i] for i in high_conf_indices] if high_conf_indices else []
            train_keys.extend(high_conf_keys)
        
        iteration_results.append({
            "iteration": iteration + 1,
            "train_size": len(train_keys),
            "val_size": len(val_keys),
            "uncertain_selected": len(uncertain_indices),
            "high_conf_selected": len(high_conf_indices),
            "test_f1": test_results["F1"],
            "test_pre": test_results["Pre"],
            "test_rec": test_results["Rec"],
            "best_f1": best_f1,
            "best_epoch": best_epoch
        })
        
        print(f" F1={test_results['F1']:.4f} (Best={best_f1:.4f}) Unc={len(uncertain_indices)} Conf={len(high_conf_indices)}")
        
        if test_results["F1"] > best_f1:
            best_f1 = test_results["F1"]
        
        # Stopping condition
        if test_results["Pre"] >= 0.9999:
            print(f"  Precision threshold reached, stopping early")
            break
    
    # Final evaluation
    final_results = model.evaluate(test_loader)
    
    results = {
        "run_id": run_id,
        "seed": seed,
        "best_f1": best_f1,
        "final_f1": final_results["F1"],
        "final_pre": final_results["Pre"],
        "final_rec": final_results["Rec"],
        "iterations": iteration_results
    }
    
    with open(output_base / f"run_{run_id}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"  Best F1: {best_f1:.4f}, Final F1: {final_results['F1']:.4f}")
    return results

def main():
    print("="*60)
    print("TRACK A: MUAD FAITHFUL REPRODUCTION - ACTIVE LEARNING")
    print("="*60)
    print("Config: hidden_dim=[64,64], epochs=5/iter, max_iter=30, event_num=15")
    print("Data: Recovered artifact (commit 6031ccf)")
    print("5 runs with seeds: 42, 123, 456, 789, 999")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path("project/experiments/runs") / f"track_a_al_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_base.mkdir(parents=True, exist_ok=True)
    
    seeds = [42, 123, 456, 789, 999]
    all_results = []
    
    for i, seed in enumerate(seeds, 1):
        try:
            results = run_single_al_experiment(i, seed, output_base)
            all_results.append(results)
        except Exception as e:
            print(f"  ERROR in run {i}: {e}")
            all_results.append({"run_id": i, "seed": seed, "error": str(e)})
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY - TRACK A ACTIVE LEARNING")
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
        "experiment": "track_a_active_learning",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "config": {
            "hidden_dim": [64, 64],
            "epochs_per_iter": 5,
            "max_iter": 30,
            "lr": 0.001,
            "batch_size": 50,
            "patience": 15,
            "event_num": 15,
            "initial_ratio": 0.1,
            "n_select": 200,
            "confidence_threshold": 0.9,
            "entropy_n_samples": 5,
            "seeds": seeds
        },
        "results": all_results
    }
    
    with open(output_base / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_base}")

if __name__ == "__main__":
    main()