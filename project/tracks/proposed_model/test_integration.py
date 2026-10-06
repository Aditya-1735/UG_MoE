"""
Quick integration test for GraphAwareMoE with real data.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import torch
from pathlib import Path

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), "../../..")
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "vendor/muad"))

from vendor.muad.data.utils import load_data, create_dataloader
from vendor.muad.data.dataset import ChunkDataset
from tracks.proposed_model import GraphAwareMoE, DEFAULT_CONFIG

def test_graph_aware_moe():
    print("Testing GraphAwareMoE with real data...")
    
    # Load data
    data_dir = Path("project/preprocessing/muad_compat/output")
    
    train_data, node_num, edges = load_data(
        str(Path("project/preprocessing/muad_compat/output/chunk_train.pkl")),
        str(Path("project/preprocessing/muad_compat/output/metadata.json"))
    )
    test_data, _, _ = load_data(
        str(Path("project/preprocessing/muad_compat/output/chunk_test.pkl")),
        str(Path("project/preprocessing/muad_compat/output/metadata.json"))
    )
    
    print(f"Train: {len(train_data)} chunks, Test: {len(test_data)} chunks")
    print(f"Nodes: {node_num}, Edges: {len(edges[0])}")
    
    # Create datasets
    train_dataset = ChunkDataset(train_data, list(train_data.keys())[:100], node_num, edges)
    test_dataset = ChunkDataset(test_data, list(test_data.keys())[:50], node_num, edges)
    
    train_loader = create_dataloader(train_dataset, batch_size=8, shuffle=True)
    test_loader = create_dataloader(test_dataset, batch_size=8, shuffle=False)
    
    # Create model
    config = DEFAULT_CONFIG.copy()
    config["log_dim"] = 14  # Our regenerated data has 14 templates
    model = GraphAwareMoE(**DEFAULT_CONFIG)
    print(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    device = torch.device("cpu")
    model = model.to(device)
    model.train()
    
for graph, labels in train_loader:
    graph = graph.to(device)
    labels = labels.to(device)
    outputs = model(graph, labels, return_routing=True)
        print(f"Forward pass successful!")
        print(f"  MMlogit: {outputs['MMlogit'].shape}")
        print(f"  loss: {outputs.get('loss', 'N/A')}")
        print(f"  y_pred: {outputs['y_pred'].shape}")
        print(f"  routing_weights: {outputs['routing_weights'].shape}")
        print(f"  expert_outputs keys: {list(outputs.get('expert_outputs', {}).keys())}")
        break
    
    # Test eval
    model.eval()
    with torch.no_grad():
        for graph, labels in test_loader:
            graph = graph.to(device)
            labels = labels.to(device)
            
            outputs = model(graph, labels)
            print(f"\nEval pass successful!")
            print(f"  MMlogit: {outputs['MMlogit'].shape}")
            break
    
    print("\n[OK] GraphAwareMoE integration test passed!")
    return True


if __name__ == "__main__":
    test_graph_aware_moe()