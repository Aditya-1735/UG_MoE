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


class HDFSDataset(Dataset):
    """
    HDFS Dataset for log-only anomaly detection.
    
    Expected input:
    - sequences: {blk_id: {"templates": [template_ids], "label": 0/1}}
    - template_vocab: {template_str: template_id}
    - log_graph: {"adjacency": [V, V], "node_features": [V, D]}
    """
    
    def __init__(self, sequences_pkl, template_vocab_json, log_graph_dir, max_seq_len=100, split="train"):
        """
        Args:
            sequences_pkl: Path to train.pkl or test.pkl (block_ids) or full sequences
            template_vocab_json: Path to template_vocab.json
            log_graph_dir: Directory with adjacency.npy and node_features.npy
            max_seq_len: Maximum sequence length
            split: "train" or "test"
        """
        # Load data
        with open(sequences_pkl, 'rb') as f:
            self.sequences = pickle.load(f)
        
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
        
        # If sequences is a dict of block_ids (train.pkl/test.pkl), load full data
        if isinstance(self.sequences, (set, list)):
            # It's a set of block_ids - need to load full sequences
            # This assumes sequences.pkl is the full sequences dict
            pass  # sequences already loaded correctly
        
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
        
        # Create log graph for this sample (use global log graph)
        if self.has_graph:
            # Use global log graph
            g = dgl.graph((self.adjacency > 0).nonzero(as_tuple=True))
            g.ndata['feat'] = self.node_features
        else:
            # Create empty graph
            g = dgl.graph(([], []), num_nodes=1)
            g.ndata['feat'] = torch.zeros(1, 64)
        
        return {
            "template_ids": template_ids,
            "attention_mask": attention_mask,
            "graph": g,
            "label": label
        }


def create_hdfs_dataloader(sequences_pkl, template_vocab_json, log_graph_dir, 
                          batch_size=32, max_seq_len=100, shuffle=True, num_workers=0):
    """
    Create DataLoader for HDFS dataset.
    """
    dataset = HDFSDataset(
        sequences_pkl=sequences_pkl,
        template_vocab_json=template_vocab_json,
        log_graph_dir=log_graph_dir,
        max_seq_len=max_seq_len
    )
    
    def collate_fn(batch):
        template_ids = torch.stack([item["template_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["label"] for item in batch])
        graphs = [item["graph"] for item in batch]
        
        # Batch graphs
        batched_graph = dgl.batch(graphs)
        
        return {
            "template_ids": template_ids,
            "attention_mask": attention_mask,
            "graph": batched_graph,
            "label": labels
        }
    
    from torch.utils.data import DataLoader
    return DataLoader(
        HDFSDataset(sequences_pkl, template_vocab_json, log_graph_dir),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )


class HDFSLogDataset(Dataset):
    """
    Alternative HDFS Dataset that works with pre-built sequences.
    Simpler interface for training.
    """
    
    def __init__(self, sequences_dict, template_vocab, log_graph=None, max_seq_len=100):
        """
        Args:
            sequences_dict: {blk_id: {"templates": [...], "label": 0/1}}
            template_vocab: {template_str: idx}
            log_graph: DGL graph with node features (optional)
            max_seq_len: Maximum sequence length
        """
        self.sequences = sequences_dict
        self.template_vocab = template_vocab
        self.log_graph = log_graph
        self.max_seq_len = max_seq_len
        self.vocab_size = len(template_vocab)
        
        self.block_ids = list(sequences_dict.keys())
        self.labels = {blk_id: seq["label"] for blk_id, seq in sequences_dict.items()}
        
        if log_graph is not None:
            self.log_graph = log_graph
            self.has_graph = True
        else:
            self.has_graph = False
        
        print(f"Initialized HDFSLogDataset with {len(self.block_ids)} sequences")
    
    def __len__(self):
        return len(self.block_ids)
    
    def __getitem__(self, idx):
        blk_id = self.block_ids[idx]
        seq_data = self.sequences[blk_id]
        
        templates = seq_data["templates"]
        label = seq_data["label"]
        
        # Pad/truncate
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
    
    def get_graph(self):
        """Return the log graph for batching."""
        return self.log_graph


def create_hdfs_dataloaders(train_sequences_pkl, test_sequences_pkl, 
                           template_vocab_json, log_graph_dir,
                           batch_size=32, max_seq_len=100, num_workers=0):
    """
    Create train and test dataloaders for HDFS.
    """
    train_dataset = HDFSLogDataset(
        sequences_dict=pickle.load(open(train_sequences_pkl, 'rb')),
        template_vocab=json.load(open(template_vocab_json)),
        max_seq_len=100
    )
    
    test_dataset = HDFSLogDataset(
        sequences_dict=pickle.load(open(test_sequences_pkl, 'rb')),
        template_vocab=json.load(open(template_vocab_json)),
        max_seq_len=100
    )
    
    def collate_fn(batch):
        template_ids = torch.stack([item["template_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["label"] for item in batch])
        return {
            "template_ids": template_ids,
            "attention_mask": attention_mask,
            "label": labels
        }
    
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test HDFS Dataset")
    parser.add_argument("--sequences", required=True, help="Path to sequences.pkl")
    parser.add_argument("--vocab", required=True, help="Path to template_vocab.json")
    args = parser.parse_args()
    
    dataset = HDFSLogDataset(
        sequences_dict=pickle.load(open(args.sequences, 'rb')),
        template_vocab=json.load(open(args.vocab)),
    )
    
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"template_ids shape: {sample['template_ids'].shape}")
    print(f"label: {sample['label']}")