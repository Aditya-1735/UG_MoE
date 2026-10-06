#!/usr/bin/env python3
"""
Gate 10: DataLoader compatibility
Verifies vendored ChunkDataset/create_dataloader successfully loads our generated chunks
and produces correctly-shaped batches (matches Phase C of runtime verification).
"""
import sys
from pathlib import Path
sys.path.insert(0, 'C:/Users/as999/OneDrive/Desktop/trash/impl/project/vendor/muad')

import torch
from data.utils import load_data, create_dataloader
from data.dataset import ChunkDataset

REGENERATED_DIR = Path("C:/Users/as999/OneDrive/Desktop/trash/impl/project/preprocessing/muad_compat/output")

def main():
    print("=" * 60)
    print("GATE 10: DataLoader Compatibility")
    print("=" * 60)
    
    # Load data
    train_data, node_num, edges = load_data(
        str(REGENERATED_DIR / "chunk_train.pkl"),
        str(REGENERATED_DIR / "metadata.json")
    )
    test_data, _, _ = load_data(
        str(REGENERATED_DIR / "chunk_test.pkl"),
        str(REGENERATED_DIR / "metadata.json")
    )
    
    print(f"[OK] load_data: train={len(train_data)} chunks, test={len(test_data)} chunks")
    print(f"[OK] node_num: {node_num}")
    print(f"[OK] edges: {len(edges[0])} edges")
    
    # Create datasets
    train_dataset = ChunkDataset(train_data, list(train_data.keys()), node_num, edges)
    test_dataset = ChunkDataset(test_data, list(test_data.keys()), node_num, edges)
    
    print(f"[OK] ChunkDataset created: train={len(train_dataset)}, test={len(test_dataset)}")
    
    # Create dataloaders
    train_loader = create_dataloader(train_dataset, batch_size=5, shuffle=True)
    test_loader = create_dataloader(test_dataset, batch_size=5, shuffle=False)
    
    print(f"[OK] DataLoader created")
    
    # Test one batch from train
    for graph, labels in train_loader:
        print(f"\n--- Train Batch ---")
        print(f"Graph: {graph}")
        print(f"Labels: {labels} (shape: {labels.shape})")
        print(f"Metrics shape: {graph.ndata['metrics'].shape}")
        print(f"Traces shape: {graph.ndata['traces'].shape}")
        print(f"Logs shape: {graph.ndata['logs'].shape}")
        print(f"Batch size: {graph.batch_size}")
        
        # Verify shapes match Phase C of runtime verification
        expected_batch_size = 5
        expected_nodes = expected_batch_size * node_num  # 5 * 12 = 60
        expected_edges = expected_batch_size * 24  # 5 * 24 = 120
        
        if graph.num_nodes() != expected_nodes:
            print(f"[FAIL] Graph nodes: {graph.num_nodes()} != {expected_nodes}")
            return 1
        if graph.num_edges() != expected_edges:
            print(f"[FAIL] Graph edges: {graph.num_edges()} != {expected_edges}")
            return 1
        if labels.shape[0] != expected_batch_size:
            print(f"[FAIL] Labels shape: {labels.shape} != [{expected_batch_size}]")
            return 1
        if graph.ndata["metrics"].shape != (expected_nodes, 10, 7):
            print(f"[FAIL] Metrics shape: {graph.ndata['metrics'].shape} != ({expected_nodes}, 10, 7)")
            return 1
        if graph.ndata["traces"].shape != (expected_nodes, 10, 1):
            print(f"[FAIL] Traces shape: {graph.ndata['traces'].shape} != ({expected_nodes}, 10, 1)")
            return 1
        if graph.ndata["logs"].shape != (expected_nodes, 14):
            print(f"[FAIL] Logs shape: {graph.ndata['logs'].shape} != ({expected_nodes}, 14)")
            return 1
        
        print(f"[OK] All shapes match Phase C runtime verification expectations")
        break
    
    # Test one batch from test
    for graph, labels in test_loader:
        print(f"\n--- Test Batch ---")
        print(f"Graph: {graph}")
        print(f"Labels: {labels} (shape: {labels.shape})")
        break
    
    print("\n[GATE 10 PASSED] DataLoader compatibility verified")
    return 0


if __name__ == "__main__":
    from pathlib import Path
    exit(main())