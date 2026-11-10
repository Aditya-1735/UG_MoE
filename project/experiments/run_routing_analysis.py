#!/usr/bin/env python3
"""
Routing Analysis - Analyze expert routing weights from trained models.
Plots routing weight distributions per expert, computes routing entropy,
and shows how routing changes across epochs and seeds.
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

EXPERT_NAMES = ["temporal", "semantic", "dependency", "cross_modal"]


def collect_routing_weights(model, test_loader, device):
    """Collect routing weights for all test batches."""
    model.eval()
    all_routing_weights = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for graph, labels in test_loader:
            graph, labels = graph.to(device), labels.to(device)
            binary_labels = (labels >= 1).long()
            
            # Forward pass to get routing weights
            # We need to intercept the router output
            B = graph.batch_size
            
            # Encode modalities
            modality_embeddings = model.encode_modalities(graph)
            
            # Build routing features
            metric_emb = modality_embeddings["metric"]
            trace_emb = modality_embeddings["trace"]
            log_emb = modality_embeddings["log"]
            routing_features = torch.cat([metric_emb, trace_emb, log_emb], dim=-1)
            
            # Get routing weights
            graph_info = model._build_graph_info(graph)
            expert_uncertainties = model._get_expert_uncertainties(modality_embeddings)
            
            routing_weights, _ = model.router(
                routing_features,
                graph_info=graph_info,
                expert_uncertainties=expert_uncertainties,
                return_uncertainty=True,
            )
            
            all_routing_weights.append(routing_weights.cpu().numpy())
            all_preds.extend(model.classifier(
                model.fusion(
                    {k: model.expert_outputs[k] for k in model.expert_types},
                    return_weights=False,
                )[0]
            ).argmax(dim=-1).cpu().numpy())
            all_labels.extend(binary_labels.cpu().numpy())
    
    routing_weights = np.vstack(all_routing_weights)  # [N, num_experts]
    return routing_weights, np.array(all_preds), np.array(all_labels)


def compute_routing_entropy(routing_weights):
    """Compute routing entropy per batch."""
    # routing_weights: [N, num_experts]
    entropy = -np.sum(routing_weights * np.log(routing_weights + 1e-8), axis=-1)
    return entropy  # [N]


def run_analysis(seed: int, output_base: Path):
    """Run routing analysis with given seed."""
    print(f"\n{'='*60}")
    print(f"ROUTING ANALYSIS - Seed: {seed}")
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
    
    config = DEFAULT_CONFIG.copy()
    config.pop("epochs", None)
    config['log_dim'] = 15
    model = GraphAwareMoE(**config).to(device)
    
    # Train model briefly
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    
    print("  Training model for routing analysis...")
    for epoch in range(1, 31):  # Shorter training for analysis
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
    
    # Collect routing weights from test set
    print("  Collecting routing weights...")
    routing_weights, preds, labels = collect_routing_weights(model, test_loader, device)
    
    # Compute routing entropy
    entropy = compute_routing_entropy(routing_weights)
    
    # Average routing weight per expert
    avg_weights = routing_weights.mean(axis=0)
    std_weights = routing_weights.std(axis=0)
    
    print(f"\n  Average routing weights per expert:")
    for i, name in enumerate(EXPERT_NAMES):
        print(f"    {name:15s}: {avg_weights[i]:.4f} ± {std_weights[i]:.4f}")
    
    print(f"\n  Routing entropy: mean={entropy.mean():.4f}, std={entropy.std():.4f}")
    print(f"  (Lower entropy = more confident routing, higher entropy = more uniform routing)")
    
    # Create routing weight heatmap
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Bar chart of average weights
    axes[0].bar(EXPERT_NAMES, avg_weights, yerr=std_weights, capsize=5, color='steelblue', alpha=0.7)
    axes[0].set_ylabel('Average Routing Weight')
    axes[0].set_title('Average Routing Weight per Expert')
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Entropy histogram
    axes[1].hist(entropy, bins=20, color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Routing Entropy')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of Routing Entropy')
    
    plt.tight_layout()
    plot_path = output_base / f"routing_analysis_seed{seed}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    results = {
        "seed": seed,
        "avg_routing_weights": {name: float(w) for name, w in zip(EXPERT_NAMES, avg_weights)},
        "std_routing_weights": {name: float(s) for name, s in zip(EXPERT_NAMES, std_weights)},
        "routing_entropy_mean": float(entropy.mean()),
        "routing_entropy_std": float(entropy.std()),
        "plot_path": str(plot_path),
    }
    
    print(f"  Plot saved to: {plot_path}")
    
    return results


def main():
    print("="*60)
    print("ROUTING ANALYSIS")
    print("="*60)
    print("Config: hidden_dim=64, epochs=30, lr=0.001, batch=50")
    print("Data: recovered_artifact (event_num=15)")
    print("Seeds: 42, 123, 456, 789, 999")
    print("Experts: temporal, semantic, dependency, cross_modal")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(__file__).parent.parent / "experiments" / "runs" / f"routing_analysis_{timestamp}"
    output_base.mkdir(parents=True, exist_ok=True)
    
    seeds = [42, 123, 456, 789, 999]
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
    print("SUMMARY - ROUTING ANALYSIS")
    print(f"{'='*60}")
    
    successful = [r for r in all_results if "error" not in r]
    if successful:
        for name in EXPERT_NAMES:
            weights = [r["avg_routing_weights"][name] for r in successful]
            print(f"  {name:15s}: Mean weight = {np.mean(weights):.4f} ± {np.std(weights):.4f}")
        
        entropies = [r["routing_entropy_mean"] for r in successful]
        print(f"\n  Routing entropy: Mean = {np.mean(entropies):.4f} ± {np.std(entropies):.4f}")
    
    summary = {
        "experiment": "routing_analysis",
        "timestamp": timestamp,
        "config": {
            "hidden_dim": 64,
            "epochs": 30,
            "lr": 0.001,
            "batch_size": 50,
            "event_num": 15,
            "seeds": seeds,
            "experts": EXPERT_NAMES,
        },
        "results": all_results,
    }
    
    with open(output_base / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_base}")


if __name__ == "__main__":
    main()
