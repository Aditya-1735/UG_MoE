"""
Build log co-occurrence graph from HDFS sequences.
Creates graph where nodes are templates and edges represent co-occurrence.
"""
import sys
import os
import pickle
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def build_log_graph(sequences_pkl, output_dir, graph_type="cooccurrence", window_size=5):
    """
    Build log co-occurrence graph from sequences.
    
    Args:
        sequences_pkl: Path to sequences.pkl from create_sequences.py
        output_dir: Output directory
        graph_type: "cooccurrence", "sequential", "attention"
        window_size: Window size for co-occurrence
    
    Returns:
        log_graph: Dict with adjacency matrix and node features
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load sequences
    print(f"Loading sequences from {sequences_pkl}...")
    with open(sequences_pkl, 'rb') as f:
        sequences = pickle.load(f)
    
    # Load vocabulary
    vocab_file = Path(sequences_pkl).parent / "template_vocab.json"
    with open(vocab_file, 'r') as f:
        template_vocab = json.load(f)
    
    vocab_size = len(template_vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    if graph_type == "cooccurrence":
        # Build co-occurrence matrix within sliding window
        print("Building co-occurrence graph...")
        cooccur = np.zeros((vocab_size, vocab_size), dtype=np.int32)
        
        for blk_id, seq_data in sequences.items():
            templates = seq_data["templates"]
            # Remove padding
            templates = [t for t in templates if t != 0]
            
            for i, t1 in enumerate(templates):
                for j in range(i+1, min(i+window_size+1, len(templates))):
                    t2 = templates[j]
                    if t1 != t2:
                        cooccur[t1, t2] += 1
                        cooccur[t2, t1] += 1
        
        # Create adjacency (threshold for sparsity)
        threshold = 1
        adjacency = (cooccur > threshold).astype(np.float32)
        
        # Add self-loops
        np.fill_diagonal(adjacency, 1.0)
        
        # Normalize
        row_sums = adjacency.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        adjacency = adjacency / row_sums
        
    elif graph_type == "sequential":
        # Sequential graph: edges between consecutive templates
        print("Building sequential graph...")
        adj = np.zeros((vocab_size, vocab_size), dtype=np.float32)
        
        for blk_id, seq_data in sequences.items():
            templates = [t for t in seq_data["templates"] if t != 0]
            for i in range(len(templates) - 1):
                t1, t2 = templates[i], templates[i+1]
                if t1 != t2:
                    adj[t1, t2] += 1
                    adj[t2, t1] += 1
        
        # Normalize
        row_sums = adj.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        adjacency = adj / row_sums
        
    elif graph_type == "attention":
        # Attention-based: use template embeddings similarity
        print("Building attention-based graph (requires embeddings)...")
        # This would require pre-trained embeddings
        # For now, fall back to co-occurrence
        print("Falling back to co-occurrence...")
        return build_log_graph(sequences_pkl, output_dir, "cooccurrence", window_size)
    
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")
    
    # Save adjacency matrix
    adj_file = Path(output_dir) / f"adjacency_{graph_type}.npy"
    np.save(adj_file, adjacency)
    
    # Create node features (one-hot or learned later)
    node_features = np.eye(len(template_vocab), dtype=np.float32)
    feat_file = Path(output_dir) / "node_features.npy"
    np.save(feat_file, node_features)
    
    # Save graph metadata
    meta = {
        "graph_type": graph_type,
        "vocab_size": vocab_size,
        "num_nodes": len(template_vocab),
        "window_size": window_size,
        "adjacency_shape": adjacency.shape,
        "density": float(adjacency.sum()) / (vocab_size * vocab_size),
    }
    
    meta_file = Path(output_dir) / f"graph_meta_{graph_type}.json"
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"Graph built: {adjacency.shape}, density: {meta['density']:.4f}")
    print(f"Saved to {output_dir}")
    
    return {
        "adjacency": adjacency,
        "node_features": node_features,
        "meta": meta
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build log co-occurrence graph")
    parser.add_argument("--sequences", required=True, help="Path to sequences.pkl")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--type", choices=["cooccurrence", "sequential", "attention"], default="cooccurrence")
    parser.add_argument("--window", type=int, default=5, help="Window size for co-occurrence")
    args = parser.parse_args()
    
    build_log_graph(args.sequences, args.output, args.type, args.window)


if __name__ == "__main__":
    main()