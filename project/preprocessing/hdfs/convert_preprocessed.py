"""
Convert preprocessed HDFS data (Event_traces.csv, anomaly_label.csv, HDFS.npz) 
to our format (sequences.pkl, template_vocab.json, train/test splits, log graph).
"""
import sys
import os
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_preprocessed_data(preprocessed_dir):
    """Load all preprocessed HDFS data."""
    preprocessed_dir = Path(preprocessed_dir)
    
    # Load template vocabulary from HDFS.log_templates.csv
    templates_df = pd.read_csv(preprocessed_dir / "HDFS.log_templates.csv")
    template_vocab = {row['EventTemplate']: row['EventId'] for _, row in templates_df.iterrows()}
    
    print(f"Template vocabulary size: {len(template_vocab)}")
    
    # Load HDFS.npz (sequences and labels)
    data = np.load(preprocessed_dir / "HDFS.npz", allow_pickle=True)
    x_data = data['x_data']  # sequences of template IDs
    y_data = data['y_data']  # labels (0=normal, 1=anomaly)
    
    print(f"Loaded {len(x_data)} sequences")
    print(f"Anomaly rate: {np.mean(y_data):.4f}")
    
    # Load anomaly labels from anomaly_label.csv for verification
    labels_df = pd.read_csv(preprocessed_dir / "anomaly_label.csv")
    print(f"Anomaly labels: {labels_df['Label'].value_counts().to_dict()}")
    
    return x_data, y_data, template_vocab


def create_sequences(x_data, y_data, max_seq_len=100):
    """Create sequences dict in our format."""
    sequences = {}
    
    # Create vocabulary mapping: event ID (E1, E2, ...) -> index (1-indexed, 0=padding)
    all_events = set()
    for seq in x_data:
        all_events.update(seq)
    
    event_list = sorted(all_events, key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
    event_vocab = {event: i+1 for i, event in enumerate(event_list)}  # 1-indexed
    event_vocab["<PAD>"] = 0
    event_vocab["<UNK>"] = len(event_vocab) + 1
    
    print(f"Event vocabulary size: {len(event_vocab)}")
    
    sequences = {}
    for i, (seq, label) in enumerate(zip(x_data, y_data)):
        # Convert event IDs to vocabulary indices
        vocab_seq = [event_vocab.get(e, event_vocab["<UNK>"]) for e in seq]
        
        # Truncate or pad
        if len(vocab_seq) > 100:
            vocab_seq = vocab_seq[:100]
        elif len(vocab_seq) < 100:
            vocab_seq = vocab_seq + [0] * (100 - len(vocab_seq))
        
        block_id = f"seq_{i}"
        
        sequences[block_id] = {
            "templates": vocab_seq,
            "label": int(label)
        }
    
    return sequences, event_vocab


def build_log_graph(sequences, vocab_size, graph_type="cooccurrence", window_size=5):
    """Build log co-occurrence graph from sequences."""
    print(f"Building {graph_type} graph...")
    
    cooccur = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    
    for blk_id, seq_data in sequences.items():
        templates = [t for t in seq_data["templates"] if t != 0]  # Remove padding
        
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
    
    return adjacency


def split_data(sequences, train_ratio=0.8, seed=42):
    """Split sequences into train/test sets."""
    import random
    
    normal_ids = []
    anomaly_ids = []
    
    for blk_id, seq_data in sequences.items():
        if seq_data["label"] == 0:
            normal_ids.append(blk_id)
        else:
            anomaly_ids.append(blk_id)
    
    print(f"Normal: {len(normal_ids)}, Anomaly: {len(anomaly_ids)}")
    
    random.seed(seed)
    random.shuffle(normal_ids)
    random.shuffle(anomaly_ids)
    
    n_train_normal = int(len(normal_ids) * 0.8)
    n_train_anomaly = int(len(anomaly_ids) * 0.8)
    
    train_ids = set(normal_ids[:n_train_normal] + anomaly_ids[:n_train_anomaly])
    test_ids = set(normal_ids[n_train_normal:] + anomaly_ids[n_train_anomaly:])
    
    print(f"Train: {len(train_ids)} (normal: {n_train_normal}, anomaly: {n_train_anomaly})")
    print(f"Test: {len(test_ids)} (normal: {len(normal_ids)-n_train_normal}, anomaly: {len(anomaly_ids)-n_train_anomaly})")
    
    train_ids = {blk_id: sequences[blk_id] for blk_id in train_ids}
    test_ids = {blk_id: sequences[blk_id] for blk_id in test_ids}
    
    return train_ids, test_ids, train_ids, test_ids


def main():
    import pickle
    preprocessed_dir = "C:/Users/as999/OneDrive/Desktop/trash/impl/hdfs_dataset/preprocessed"
    output_dir = Path("preprocessing/hdfs/output_full")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    Path("preprocessing/hdfs/output_full/03_log_graph").mkdir(parents=True, exist_ok=True)
    Path("preprocessing/hdfs/output_full/04_split").mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("STAGE 1: Load preprocessed HDFS data")
    print("=" * 60)
    x_data, y_data, template_vocab = load_preprocessed_data(preprocessed_dir)
    
    print("\n" + "=" * 60)
    print("STAGE 2: Create sequences")
    print("=" * 60)
    sequences, event_vocab = create_sequences(x_data, y_data, max_seq_len=100)
    
    print("\n" + "=" * 60)
    print("STAGE 3: Build log co-occurrence graph")
    print("=" * 60)
    vocab_size = len(event_vocab)
    log_graph = build_log_graph(sequences, vocab_size, graph_type="cooccurrence", window_size=5)
    
    print("\n" + "=" * 60)
    print("STAGE 4: Split train/test")
    print("=" * 60)
    train_ids, test_ids, train_data, test_data = split_data(sequences, train_ratio=0.8, seed=42)
    
    # Save all outputs
    output_dir = Path("preprocessing/hdfs/output_full")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    Path("preprocessing/hdfs/output_full/03_log_graph").mkdir(parents=True, exist_ok=True)
    Path("preprocessing/hdfs/output_full/04_split").mkdir(parents=True, exist_ok=True)
    
    # Save sequences
    with open(output_dir / "sequences.pkl", 'wb') as f:
        pickle.dump(sequences, f)
    
    # Save vocabulary
    with open(output_dir / "template_vocab.json", 'w') as f:
        json.dump(event_vocab, f)
    
    # Save log graph
    np.save(Path("preprocessing/hdfs/output_full/03_log_graph/adjacency_cooccurrence.npy"), log_graph)
    np.save(Path("preprocessing/hdfs/output_full/03_log_graph/node_features.npy"), np.eye(len(event_vocab), dtype=np.float32))
    
    # Save graph metadata
    with open(Path("preprocessing/hdfs/output_full/03_log_graph/graph_meta_cooccurrence.json"), 'w') as f:
        json.dump({
            "graph_type": "cooccurrence",
            "vocab_size": len(event_vocab),
            "window_size": 5,
            "adjacency_shape": list(np.array(log_graph).shape)
        }, f, indent=2)
    
    # Save train/test splits
    train_ids = {k: v for k, v in sequences.items() if k in train_ids}
    test_ids = {k: v for k, v in sequences.items() if k in test_ids}
    
    with open(Path("preprocessing/hdfs/output_full/04_split/train.pkl"), 'wb') as f:
        pickle.dump(list(train_ids), f)
    with open(Path("preprocessing/hdfs/output_full/04_split/test.pkl"), 'wb') as f:
        pickle.dump(list(test_ids), f)
    with open(Path("preprocessing/hdfs/output_full/04_split/train_full.pkl"), 'wb') as f:
        pickle.dump(train_data, f)
    with open(Path("preprocessing/hdfs/output_full/04_split/test_full.pkl"), 'wb') as f:
        pickle.dump(test_data, f)
    
    # Save template_vocab to split dir
    with open(Path("preprocessing/hdfs/output_full/04_split/template_vocab.json"), 'w') as f:
        json.dump(event_vocab, f)
    
    # Save metadata
    meta = {
        "num_sequences": len(sequences),
        "vocab_size": len(event_vocab),
        "max_seq_len": 100,
        "num_normal": sum(1 for s in sequences.values() if s["label"] == 0),
        "num_anomaly": sum(1 for s in sequences.values() if s["label"] == 1),
        "train_size": len(train_ids),
        "test_size": len(test_ids),
    }
    with open(Path("preprocessing/hdfs/output_full/04_split/split_metadata.json"), 'w') as f:
        json.dump(meta, f, indent=2)
    
    # Save template_vocab to split dir
    with open(Path("preprocessing/hdfs/output_full/04_split/template_vocab.json"), 'w') as f:
        json.dump(event_vocab, f)
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()