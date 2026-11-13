#!/usr/bin/env python3
"""
Uncertainty & Fusion Weight Analysis - Analyze per-expert uncertainty estimates
and modality fusion weights from the trained Graph-Aware MoE model.
"""
import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

MODALITY_NAMES = ["metric", "trace", "log"]


def collect_uncertainty_and_fusion(model, test_loader, device):
    """Collect per-expert uncertainty and fusion weights from test set."""
    model.eval()
    all_uncertainties = []  # per-expert KL losses
    all_fusion_weights = []  # per-modality fusion weights
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for graph, labels in test_loader:
            graph, labels = graph.to(device), labels.to(device)
            binary_labels = (labels >= 1).long()
            
            outputs = model(graph, binary_labels, return_routing=True, return_expert_outputs=True)
            
            # Debug: print available output keys on first batch
            if len(all_fusion_weights) == 0 and len(all_uncertainties) == 0:
                print(f"  [DEBUG] Output keys: {list(outputs.keys())}")
                if "fusion_weights" in outputs:
                    print(f"  [DEBUG] fusion_weights type: {type(outputs['fusion_weights'])}")
                if "expert_uncertainties" in outputs:
                    print(f"  [DEBUG] expert_uncertainties type: {type(outputs['expert_uncertainties'])}")
            
            # Try multiple key names for fusion weights
            fusion_weights = None
            for key in ["fusion_weights", "fusion_weights_dict", "fusion"]:
                if key in outputs and isinstance(outputs[key], dict):
                    fusion_weights = outputs[key]
                    break
            
            if fusion_weights is not None:
                all_fusion_weights.append({k: v.cpu().numpy() for k, v in fusion_weights.items()})
            
            # Try multiple key names for expert uncertainties
            expert_uncs = None
            for key in ["expert_uncertainties", "uncertainties", "kl_losses", "uncertainty_losses"]:
                if key in outputs and isinstance(outputs[key], dict):
                    expert_uncs = outputs[key]
                    break
            
            if expert_uncs is not None:
                all_uncertainties.append({k: v.cpu().numpy() for k, v in expert_uncs.items()})
            
            all_preds.extend(outputs["y_pred"].cpu().numpy())
            all_labels.extend(binary_labels.cpu().numpy())
    
    return all_uncertainties, all_fusion_weights, np.array(all_preds), np.array(all_labels)


def run_analysis(seed: int, output_base: Path):
    """Run uncertainty/fusion analysis with given seed."""
    print(f"\n{'='*60}")
    print(f"UNCERTAINTY & FUSION ANALYSIS - Seed: {seed}")
    print(f"{'='*60}")
    
    seed_everything(seed)
    
    # Load recovered artifact
    data_dir = Path(__file__).parent.parent / "recovered_artifact"
    train_data, node_num, edges, metadata = load_data(
        str(data_dir / "chunk_train.pkl"),
        str(data_dir / "metadata.json")
    )
    test_data, _, _, _ = load_data(
        str(data_dir / "chunk_test.pkl"),
        str(data_dir / "metadata.json")
    )
    
    print(f"  Train: {len(train_data)} chunks, Test: {len(test_data)} chunks")
    print(f"  Data dims: log_dim={metadata['event_num']}, metric_dim={metadata['metric_num']}, num_nodes={metadata['node_num']}")
    
    train_dataset = ChunkDataset(train_data, list(train_data.keys()), node_num, edges)
    test_dataset = ChunkDataset(test_data, list(test_data.keys()), node_num, edges)
    
    train_loader = create_dataloader(train_dataset, batch_size=50, shuffle=True)
    test_loader = create_dataloader(test_dataset, batch_size=50, shuffle=False)
    
    device = torch.device("cpu")
    
    config = DEFAULT_CONFIG.copy()
    config.pop("epochs", None)
    config['log_dim'] = metadata["event_num"]
    model = GraphAwareMoE(**config).to(device)
    
    # Train model briefly
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    
    print("  Training model for uncertainty/fusion analysis...")
    for epoch in range(1, 31):
        model.train()
        train_loss = 0.0
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
        
        # Evaluate every 5 epochs
        if epoch % 5 == 0:
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
                    probs = torch.softmax(outputs["MMlogit"], dim=-1)[:, 1]
                    preds = outputs["y_pred"]
                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(binary_labels.cpu().tolist())
                    all_probs.extend(probs.cpu().tolist())
            
            avg_test_loss = test_loss / len(test_loader)
            from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
            f1 = f1_score(all_labels, all_preds, average='binary')
            precision = precision_score(all_labels, all_preds, average='binary')
            recall = recall_score(all_labels, all_preds, average='binary')
            roc_auc = roc_auc_score(all_labels, all_probs)
            pr_auc = average_precision_score(all_labels, all_probs)
            
            print(f"  Epoch {epoch:2d}: Train Loss={train_loss/len(train_loader):.4f}, Test Loss={avg_test_loss:.4f}, F1={f1:.4f}, Pre={precision:.4f}, Rec={recall:.4f}, ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")
    
    # Collect uncertainty and fusion weights
    print("  Collecting uncertainty and fusion weights...")
    uncertainties, fusion_weights, preds, labels = collect_uncertainty_and_fusion(
        model, test_loader, device
    )
    
    results = {
        "seed": seed,
        "has_uncertainty_data": len(uncertainties) > 0,
        "has_fusion_data": len(fusion_weights) > 0,
    }
    
    if uncertainties:
        # Average KL loss per expert
        expert_names = ["temporal", "semantic", "dependency", "cross_modal"]
        avg_uncs = {}
        for name in expert_names:
            vals = [u.get(name, np.array([0])) for u in uncertainties if name in u]
            if vals:
                avg_uncs[name] = float(np.mean(np.concatenate(vals)))
        
        print(f"\n  Average KL uncertainty per expert:")
        for name in expert_names:
            if name in avg_uncs:
                print(f"    {name:15s}: {avg_uncs[name]:.6f}")
        
        results["avg_uncertainties"] = avg_uncs
    
    if fusion_weights:
        # Average fusion weight per modality
        avg_fusion = {}
        for name in MODALITY_NAMES:
            vals = [fw.get(name, np.array([0])) for fw in fusion_weights if name in fw]
            if vals:
                avg_fusion[name] = float(np.mean(np.concatenate(vals)))
        
        print(f"\n  Average fusion weight per modality:")
        for name in MODALITY_NAMES:
            if name in avg_fusion:
                print(f"    {name:15s}: {avg_fusion[name]:.4f}")
        
        results["avg_fusion_weights"] = avg_fusion
        
        # Create fusion weight bar chart
        if avg_fusion:
            fig, ax = plt.subplots(figsize=(6, 4))
            names = list(avg_fusion.keys())
            values = list(avg_fusion.values())
            ax.bar(names, values, color='steelblue', alpha=0.7)
            ax.set_ylabel('Average Fusion Weight')
            ax.set_title('Average Fusion Weight per Modality')
            ax.set_ylim(0, 1)
            plt.tight_layout()
            plot_path = output_base / f"fusion_weights_seed{seed}.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"  Plot saved to: {plot_path}")
            results["fusion_plot_path"] = str(plot_path)
    
    return results


def main():
    print("="*60)
    print("UNCERTAINTY & FUSION WEIGHT ANALYSIS")
    print("="*60)
    print("Config: hidden_dim=64, epochs=30, lr=0.001, batch=50")
    print("Data: recovered_artifact (event_num=15)")
    print("Seeds: 42, 123, 456, 789, 999")
    print("Modalities: metric, trace, log")
    print("Experts: temporal, semantic, dependency, cross_modal")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(__file__).parent.parent / "experiments" / "runs" / f"uncertainty_fusion_analysis_{timestamp}"
    output_base.mkdir(parents=True, exist_ok=True)
    
    seeds = [42, 123]
    all_results = []
    
    for seed in seeds:
        try:
            results = run_analysis(seed, output_base)
            all_results.append(results)
        except Exception as e:
            print(f"  ERROR for seed {seed}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"seed": seed, "error": str(e)})
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY - UNCERTAINTY & FUSION ANALYSIS")
    print(f"{'='*60}")
    
    successful = [r for r in all_results if "error" not in r]
    if successful:
        # Average fusion weights across seeds
        fusion_keys = ["metric", "trace", "log"]
        for name in fusion_keys:
            vals = [r.get("avg_fusion_weights", {}).get(name, 0) for r in successful if name in r.get("avg_fusion_weights", {})]
            if vals:
                print(f"  Fusion weight {name:10s}: Mean = {np.mean(vals):.4f} ± {np.std(vals):.4f}")
        
        # Average uncertainties across seeds
        expert_names = ["temporal", "semantic", "dependency", "cross_modal"]
        for name in expert_names:
            vals = [r.get("avg_uncertainties", {}).get(name, 0) for r in successful if name in r.get("avg_uncertainties", {})]
            if vals:
                print(f"  Uncertainty {name:15s}: Mean = {np.mean(vals):.6f} ± {np.std(vals):.6f}")
    
    summary = {
        "experiment": "uncertainty_fusion_analysis",
        "timestamp": timestamp,
        "config": {
            "hidden_dim": 64,
            "epochs": 30,
            "lr": 0.001,
            "batch_size": 50,
            "event_num": 15,
            "seeds": seeds,
            "modalities": MODALITY_NAMES,
            "experts": expert_names,
        },
        "results": all_results,
    }
    
    with open(output_base / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_base}")


if __name__ == "__main__":
    main()
