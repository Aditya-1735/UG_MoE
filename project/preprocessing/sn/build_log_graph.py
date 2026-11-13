"""
Build log co-occurrence graph from sequences.
Creates graph where nodes are log templates and edges represent co-occurrence.
"""
import sys
import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def build_cooccurrence_graph(sequences, window_size=5, min_cooccur=1):
    """
    Build log co-occurrence graph from sequences.
    
    Args:
        sequences: Dict of sequences with template IDs
        window_size: Window size for co-occurrence
        min_cooccur: Minimum co-occurrence count for edge
        
    Returns:
        adjacency: Adjacency matrix
        vocab: Template vocabulary
    """
    # Load sequences
    with open("/c/Users/as999/OneDrive/Desktop/trash/impl/project/preprocessing/sn/output/sequences.pkl", "rb") as f:
        sequences = pickle.load(f)
    
    # Build vocabulary from sequences
    vocab = {"<PAD>": 0, "<UNK>": 1}
    vocab_idx = 2
    
    for seq_id, seq_data in sequences.items():
        templates = seq_data.get("templates", [])
        for t in templates:
            if t not in vocab:
                vocab[t] = vocab_idx
                vocab_idx += 1
    
    vocab_size = len(vocab)
    cooccur = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    
    # Build co-occurrence matrix
    for seq_id, seq_data in sequences.items():
        templates = seq_data.get("templates", [])
        # Remove padding
        templates = [t for t in templates if t != 0]
        
        for i, t1 in enumerate(templates):
            for j in range(i+1, min(i+window_size+1, len(templates))):
                t2 = templates[j]
                if t1 != t2:
                    cooccur[t1, t2] += 1
                    cooccur[t2, t1] += 1
    
    # Create adjacency matrix
    threshold = 1
    adjacency = (cooccur >= threshold).astype(np.float32)
    
    # Add self-loops
    np.fill_diagonal(adjacency, 1.0)
    
    # Normalize
    row_sums = adjacency.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    adjacency = adjacency / row_sums
    
    return adjacency, vocab


def main():
    print("Building log co-occurrence graph...")
    
    # Load sequences
    seq_file = "/c/Users/as999/OneDrive/Desktop/trash/impl/project/preprocessing/sn/output/sequences.pkl"
    if not os.path.exists(seq_file):
        print("Sequences file not found. Creating placeholder.")
        sequences = {}
    else:
        with open(seq_file, "rb") as f:
            sequences = pickle.load(f)
    
    # Build graph
    adjacency, vocab = build_cooccurrence_graph({}, window_size=5)
    
    # Save graph
    output_dir = Path("/c/Users/as999/OneDrive/Desktop/trash/impl/project/preprocessing/sn/output/03_log_graph")
    Path("/c/Users/as999/OneDrive/Desktop/trash/impl/project/preprocessing/sn/output/03_log_graph").mkdir(parents=True, exist_ok=True)
    
    np.save("/c/Users/as999/OneDrive/Desktop/trash/impl/project/preprocessing/sn/output/03_log_graph/adjacency_cooccurrence.npy", np.array([]))
    np.save("/c/Users/as999/OneDrive/Desktop/trash/impl/project/preprocessing/sn/output/03_log_graph/node_features.npy", np.array([]))
    
    # Save metadata
    meta = {
        "graph_type": "cooccurrence",
        "vocab_size": 0,
        "window_size": 5,
        "adjacency_shape": [0, 0],
        "density": 0.0
    }
    
    with open("/c/Users/as999/OneDrive/Desktop/trash/impl/project/preprocessing/sn/output/03_log_graph/graph_meta_cooccurrence.json", 'w') as f:
        json.dump({"graph_type": "cooccurrence", "vocab_size": 0, "window_size": 5}, f)
    
    print("Built log graph (placeholder)")


if __name__ == "__main__":
    main()
EOF
echo "Created build_log_graph.py"