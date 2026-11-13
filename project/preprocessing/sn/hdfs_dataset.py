"""
HDFS Dataset for Graph-Aware MoE (log_only_mode).
PyTorch Dataset for HDFS log anomaly detection.
"""
import sys
import os
import pickle
import json
import torch
import dgl
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class HDFSLogDataset(Dataset):
    """
    HDFS Dataset for log-only anomaly detection.
    
    Expected input:
    - sequences_pkl: Path to sequences.pkl with {blk_id: {"templates": [template_ids], "label": 0/1}}
    - template_vocab_json: Path to template_vocab.json
    - log_graph_dir: Directory with adjacency_cooccurrence.npy and node_features.npy
    """
    
    def __init__(self, sequences_pkl, template_vocab_json, log_graph_dir, max_seq_len=100, split="train"):
        """
        Args:
            sequences_pkl: Path to train_full.pkl or test_full.pkl (full sequences dict)
            template_vocab_json: Path to template_vocab.json
            log_graph_dir: Directory with adjacency_cooccurrence.npy and node_features.npy
            max_seq_len: Maximum sequence length
            split: "train" or "test"
        """
        # Load sequences
        with open(sequences_pkl, 'rb') as f:
            self.sequences = pickle.load(f)
        
        # Load template vocabulary
        with open(template_vocab_json, 'r') as f:
            self.template_vocab = json.load(f)
        
        self.max_seq_len = max_seq_len
        self.vocab_size = len(self.template_vocab)
        
        # Load log graph
        adj_file = Path(log_graph_dir) / "adjacency_cooccurrence.npy"
        feat_file = Path(log_graph_dir) / "node_features.npy"
        
        if adj_file.exists():
            self.adjacency = torch.from_numpy(np.load(adj_file)).float()
            self.node_features = torch.from_numpy(np.load(feat_file)).float()
            self.has_graph = True
        else:
            self.adjacency = None
            self.node_features = None
            self.has_graph = False
        
        # Load sequences
        with open(sequences_pkl, 'rb') as f:
            self.sequences = pickle.load(f)
        
        self.block_ids = list(self.sequences.keys())
        self.labels = {blk_id: seq["label"] for blk_id, seq in self.sequences.items()}
        
        print(f"Loaded {len(self.block_ids)} sequences for {split}")
    
    def __len__(self):
        return len(self.block_ids)
    
    def __getitem__(self, idx):
        blk_id = self.block_ids[idx]
        seq_data = self.sequences[blk_id]
        
        templates = seq_data["templates"]
        label = seq_data["label"]
        
        # Ensure correct length
        if len(templates) > self.max_seq_len:
            templates = templates[:self.max_seq_len]
        elif len(templates) < self.max_seq_len:
            templates = templates + [0] * (self.max_seq_len - len(templates))
        
        template_ids = torch.tensor(templates, dtype=torch.long)
        attention_mask = (torch.tensor(templates) != 0).long()
        label = torch.tensor(label, dtype=torch.long)
        
        return {
            "template_ids": template_ids,
            "attention_mask": attention_mask,
            "label": label
        }


def create_hdfs_dataloaders(train_sequences_pkl, test_sequences_pkl, template_vocab_json, log_graph_dir, 
                           batch_size=32, max_seq_len=100, num_workers=0):
    """
    Create train and test dataloaders for HDFS.
    """
    train_dataset = HDFSLogDataset(
        sequences_pkl=train_sequences_pkl,
        template_vocab_json=template_vocab_json,
        log_graph_dir=log_graph_dir,
        max_seq_len=max_seq_len,
        split="train"
    )
    
    test_dataset = HDFSLogDataset(
        sequences_pkl=test_sequences_pkl,
        template_vocab_json=template_vocab_json,
        log_graph_dir=log_graph_dir,
        max_seq_len=max_seq_len,
        split="test"
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test HDFS Dataset")
    parser.add_argument("--train", required=True, help="Path to train_full.pkl")
    parser.add_argument("--test", required=True, help="Path to test_full.pkl")
    parser.add_argument("--vocab", required=True, help="Path to template_vocab.json")
    parser.add_argument("--graph", required=True, help="Path to log graph directory")
    parser.add_argument("--max-seq", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    
    args = parser.parse_args()
    
    train_loader, test_loader = create_hdfs_dataloaders(
        args.train, args.test, args.vocab, args.graph,
        batch_size=args.batch, max_seq_len=args.max_seq
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test one batch
    for batch in train_loader:
        print(f"template_ids: {batch['template_ids'].shape}")
        print(f"attention_mask: {batch['attention_mask'].shape}")
        print(f"label: {batch['label'].shape}")
        break
EOF
echo "Created hdfs_dataset.py"