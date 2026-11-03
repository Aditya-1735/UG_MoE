"""
Split HDFS data into train/test sets.
Splits by block_id to avoid data leakage.
"""
import sys
import os
import pickle
import json
import random
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def split_data(sequences_pkl, output_dir, train_ratio=0.8, seed=42):
    """
    Split sequences into train/test sets by block_id.
    
    Args:
        sequences_pkl: Path to sequences.pkl
        output_dir: Output directory
        train_ratio: Fraction for training
        seed: Random seed for reproducibility
    
    Returns:
        train.pkl, test.pkl, metadata.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load sequences
    print(f"Loading sequences from {sequences_pkl}...")
    with open(sequences_pkl, 'rb') as f:
        sequences = pickle.load(f)
    
    # Separate by label
    normal_ids = []
    anomaly_ids = []
    
    for blk_id, seq_data in sequences.items():
        if seq_data["label"] == 0:
            normal_ids.append(blk_id)
        else:
            anomaly_ids.append(blk_id)
    
    print(f"Normal: {len(normal_ids)}, Anomaly: {len(anomaly_ids)}")
    
    # Stratified split
    random.seed(42)
    random.shuffle(normal_ids)
    random.shuffle(anomaly_ids)
    
    n_train_normal = int(len(normal_ids) * 0.8)
    n_train_anomaly = int(len(anomaly_ids) * 0.8)
    
    train_ids = set(normal_ids[:n_train_normal] + anomaly_ids[:n_train_anomaly])
    test_ids = set(normal_ids[n_train_normal:] + anomaly_ids[n_train_anomaly:])
    
    print(f"Train: {len(train_ids)} (normal: {n_train_normal}, anomaly: {n_train_anomaly})")
    print(f"Test: {len(test_ids)} (normal: {len(normal_ids)-n_train_normal}, anomaly: {len(anomaly_ids)-n_train_anomaly})")
    
    # Create train/test dicts
    train_data = {blk_id: sequences[blk_id] for blk_id in train_ids}
    test_data = {blk_id: sequences[blk_id] for blk_id in test_ids}
    
    # Verify no overlap
    assert len(train_ids & set(sequences.keys()) & set(sequences.keys())) == len(train_ids)
    assert len(train_ids & test_ids) == 0
    
    # Save
    train_file = Path(output_dir) / "train.pkl"
    test_file = Path(output_dir) / "test.pkl"
    
    with open(train_file, 'wb') as f:
        pickle.dump(train_ids, f)
    with open(test_file, 'wb') as f:
        pickle.dump(test_ids, f)
    
    # Save full data with labels
    train_data_full = {blk_id: sequences[blk_id] for blk_id in train_ids}
    test_data_full = {blk_id: sequences[blk_id] for blk_id in test_ids}
    
    train_full_file = Path(output_dir) / "train_full.pkl"
    test_full_file = Path(output_dir) / "test_full.pkl"
    
    with open(train_full_file, 'wb') as f:
        pickle.dump(train_data_full, f)
    with open(test_full_file, 'wb') as f:
        pickle.dump(test_data_full, f)
    
    # Metadata
    meta = {
        "train_size": len(train_ids),
        "test_size": len(test_ids),
        "train_normal": sum(1 for b in train_ids if sequences[b]["label"] == 0),
        "train_anomaly": sum(1 for b in train_ids if sequences[b]["label"] == 1),
        "test_normal": sum(1 for b in test_ids if sequences[b]["label"] == 0),
        "test_anomaly": sum(1 for b in test_ids if sequences[b]["label"] == 1),
    }
    
    meta_file = Path(output_dir) / "split_metadata.json"
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"Train: {meta['train_size']} (normal: {meta['train_normal']}, anomaly: {meta['train_anomaly']})")
    print(f"Test: {meta['test_size']} (normal: {meta['test_normal']}, anomaly: {meta['test_anomaly']})")
    print(f"Saved to {output_dir}")
    
    return train_ids, test_ids


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Split HDFS data into train/test")
    parser.add_argument("--sequences", required=True, help="Path to sequences.pkl")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    split_data(args.sequences, args.output, args.train_ratio, args.seed)


if __name__ == "__main__":
    main()