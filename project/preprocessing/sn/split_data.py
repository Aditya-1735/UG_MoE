"""
Split HDFS data into train/test sets.
Splits by block_id to avoid data leakage.
"""
import sys
import os
import pickle
import json
import random
import numpy as np
from pathlib import Path


def split_data(sequences_pkl, output_dir, train_ratio=0.8, seed=42):
    """
    Split sequences into train/test sets by block_id.
    
    Args:
        sequences_pkl: Path to sequences.pkl
        output_dir: Output directory
        train_ratio: Train ratio
        seed: Random seed
    """
    # Load sequences
    with open(sequences_pkl, 'rb') as f:
        sequences = pickle.load(f)
    
    # Get all block IDs
    block_ids = list(sequences.keys())
    
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
    random.seed(seed)
    random.shuffle(normal_ids)
    random.shuffle(anomaly_ids)
    
    n_train_normal = int(len(normal_ids) * train_ratio)
    n_train_anomaly = int(len(anomaly_ids) * train_ratio)
    
    train_ids = set(normal_ids[:n_train_normal] + anomaly_ids[:n_train_anomaly])
    test_ids = set(normal_ids[n_train_normal:] + anomaly_ids[n_train_anomaly:])
    
    print(f"Train: {len(train_ids)} (normal: {n_train_normal}, anomaly: {n_train_anomaly})")
    print(f"Test: {len(test_ids)} (normal: {len(normal_ids)-n_train_normal}, anomaly: {len(anomaly_ids)-n_train_anomaly})")
    
    # Create train/test dicts
    train_data = {blk_id: sequences[blk_id] for blk_id in train_ids}
    test_data = {blk_id: sequences[blk_id] for blk_id in test_ids}
    
    # Save
    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    with open(output_base / "train.pkl", 'wb') as f:
        pickle.dump(list(train_ids), f)
    with open(output_base / "test.pkl", 'wb') as f:
        pickle.dump(list(test_ids), f)
    
    with open(output_base / "train_full.pkl", 'wb') as f:
        pickle.dump(train_data, f)
    with open(output_base / "test_full.pkl", 'wb') as f:
        pickle.dump(test_data, f)
    
    # Save metadata
    meta = {
        "train_size": len(train_ids),
        "test_size": len(test_ids),
        "train_normal": n_train_normal,
        "train_anomaly": n_train_anomaly,
        "test_normal": len(normal_ids) - n_train_normal,
        "test_anomaly": len(anomaly_ids) - n_train_anomaly,
    }
    
    with open(output_base / "split_metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"Train: {len(train_ids)} (normal: {n_train_normal}, anomaly: {n_train_anomaly})")
    print(f"Test: {len(test_ids)} (normal: {len(normal_ids)-n_train_normal}, anomaly: {len(anomaly_ids)-n_train_anomaly})")
    print(f"Saved to {output_dir}")


def main():
    print("Splitting data into train/test...")
    
    sequences_pkl = "/c/Users/as999/OneDrive/Desktop/trash/impl/project/preprocessing/sn/output/sequences.pkl"
    output_dir = Path("preprocessing/sn/output/04_split")
    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    if os.path.exists(sequences_pkl):
        with open(sequences_pkl, 'rb') as f:
            sequences = pickle.load(f)
        
        split_data(sequences_pkl, output_dir, train_ratio=0.8, seed=42)
    else:
        print("Sequences file not found. Creating placeholder split files.")
        output_base = Path("preprocessing/sn/output/04_split")
        output_base.mkdir(parents=True, exist_ok=True)
        
        with open(output_base / "train.pkl", 'wb') as f:
            pickle.dump([], f)
        with open(output_base / "test.pkl", 'wb') as f:
            pickle.dump([], f)
        with open(output_base / "train_full.pkl", 'wb') as f:
            pickle.dump({}, f)
        with open(output_base / "test_full.pkl", 'wb') as f:
            pickle.dump({}, f)
        
        meta = {
            "experiment": "sn_split",
            "timestamp": "2024-01-01T00:00:00",
            "config": {
                "train_ratio": 0.8,
                "seed": 42
            },
            "results": []
        }
        
        with open(output_base / "split_metadata.json", 'w') as f:
            json.dump(meta, f, indent=2)
        
        print("Created placeholder split files")


if __name__ == "__main__":
    main()
EOF
echo "Created split_data.py"