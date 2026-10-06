#!/usr/bin/env python3
"""
Ablation Studies for Graph-Aware MoE
Runs 6 ablation configs × 5 seeds on OUR regenerated data (event_num=14)
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

# Ablation configurations
ABLATIONS = {
    "no_graph": {
        "use_graph_routing": False,
        "router_type": "base",
        "description": "No graph routing - base router only"
    },
    "no_uncertainty": {
        "use_uncertainty": False,
        "description": "No uncertainty blocks"
    },
    "no_fusion": {
        "fusion_type": "concat_mlp",
        "description": "Simple concat+MLP fusion instead of confidence-weighted"
    },
    "single_expert": {
        "expert_types": ["temporal"],
        "description": "Single temporal expert (no MoE)"
    },
    "plain_moe": {
        "router_type": "base",
        "use_graph_routing": False,
        "use_uncertainty_routing": False,
        "fusion_type": "concat_mlp",
        "description": "Plain MoE: no graph, no uncertainty, simple fusion"
    },
    "graph_transformer": {
        "description": "Graph Transformer instead of GAT in dependency expert"
        # Note: This requires modifying DependencyExpert expert_type
    }
}

def run_ablation(ablation_name: str, ablation_config: dict, seed: int, output_base: Path):
    """Run single ablation experiment with given seed."""
    print(f"\n{'='*60}")
    print(f"ABLATION: {ablation_name} | Seed: {seed}")
    print(f"{'='*60}")
    print(f"Config: {ablation_config.get('description', '')}")
    
    seed_everything(seed)
    
    data_dir = Path(__file__).parent.parent / "preprocessing/muad_compat/output"
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
    config = DEFAULT_CONFIG.copy()
    config['log_dim'] = 14
    
    # Apply ablation config
    for k, v in ablation_config.items():
        if k != "description":
            config[k] = v
    
    model_config = {k: v for k, v in config.items() if k != 'epochs'}
    model = GraphAwareMoE(**model_config).to(device)
    
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
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        if epoch % 5 == 0:
            model.eval()
            all_preds = []
            all_labels = []
            test_loss = 0
            with torch.no_grad():
                for graph, labels in test_loader:
                    graph, labels = graph.to(device), labels.to(device)
                    binary_labels = (labels >= 1).long()
                    outputs = model(graph, binary_labels)
                    loss = outputs.get("loss", criterion(outputs["MMlogit"], binary_labels))
                    test_loss += loss.item()
                    preds = outputs["y_pred"]
                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(binary_labels.cpu().tolist())
            
            avg_test_loss = test_loss / len(test_loader)
            
            from sklearn.metrics import f1_score, precision_score, recall_score
            f1 = f1_score(all_labels, all_preds, average='binary')
            precision = precision_score(all_labels, all_preds, average='binary')
            recall = recall_score(all_labels, all_preds, average='binary')
            
            print(f"  Epoch {epoch:2d}: Test Loss={avg_test_loss:.4f}, F1={f1:.4f}, Pre={precision:.4f}, Rec={recall:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
    
    # Final eval
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
    
    return {
        "ablation": ablation_name,
        "seed": seed,
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "final_f1": final_f1,
        "final_precision": final_pre,
        "final_recall": final_rec
    }

def main():
    print("="*60)
    print("ABLATION STUDIES - GRAPH-AWARE MoE")
    print("="*60)
    print(f"Ablations: {list(ABLATIONS.keys())}")
    print("5 seeds per ablation: 42, 123, 456, 789, 999")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path("project/experiments/runs") / f"ablations_{timestamp}"
    output_base.mkdir(parents=True, exist_ok=True)
    
    seeds = [42, 123]
    all_results = {}
    
    for ablation_name, ablation_config in ABLATIONS.items():
        print(f"\n{'='*60}")
        print(f"ABLATION: {ablation_name} - {ablation_config.get('description', '')}")
        print(f"{'='*60}")
        
        ablation_results = []
        for i, seed in enumerate(seeds, 1):
            try:
                result = run_ablation(ablation_name, ablation_config, seed, output_base)
                ablation_results.append(result)
                print(f"  Run {i}/2 (seed={seed}): Best F1={result['best_f1']:.4f} (epoch {result['best_epoch']})")
            except Exception as e:
                print(f"  ERROR in run {i} (seed={seed}): {e}")
                import traceback
                traceback.print_exc()
                ablation_results.append({"ablation": ablation_name, "seed": seed, "error": str(e)})
        
        all_results[ablation_name] = ablation_results
        
        # Save intermediate
        with open(output_base / f"{ablation_name}_results.json", "w") as f:
            json.dump(ablation_results, f, indent=2)
    
    # Summary
    print(f"\n{'='*60}")
    print("ABLATION SUMMARY")
    print(f"{'='*60}")
    
    for ablation_name, results in all_results.items():
        successful = [r for r in results if "error" not in r]
        if successful:
            f1s = [r["best_f1"] for r in successful]
            print(f"{ablation_name:20s}: Mean F1={np.mean(f1s):.4f} ± {np.std(f1s):.4f}  (Max: {np.max(f1s):.4f})")
        else:
            print(f"{ablation_name:20s}: FAILED")
    
    summary = {
        "experiment": "ablations",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "config": {
            "hidden_dim": 64,
            "epochs": 50,
            "lr": 0.001,
            "batch_size": 50,
            "weight_decay": 1e-5,
            "event_num": 14,
            "seeds": [42, 123]
        },
        "results": all_results
    }
    
    with open(output_base / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_base}")

if __name__ == "__main__":
    main()