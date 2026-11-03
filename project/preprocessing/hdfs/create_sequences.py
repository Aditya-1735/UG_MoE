"""
Create sequences from parsed HDFS logs.
Groups log lines by block_id (blk_id) to create sequences per block.
"""
import sys
import os
import re
import pickle
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def create_sequences(parsed_lines_pkl, anomaly_label_csv, output_dir, max_seq_len=100):
    """
    Create sequences grouped by block_id from parsed HDFS logs.
    
    Args:
        parsed_lines_pkl: Path to parsed_lines.pkl from parse_hdfs.py
        anomaly_label_csv: Path to anomaly_label.csv (block_id, label)
        output_dir: Output directory
        max_seq_len: Maximum sequence length (truncate/pad)
    
    Returns:
        sequences: Dict {blk_id: {"templates": [template_ids], "label": 0/1}}
        template_vocab: Dict {template_str: template_id}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load parsed lines
    print(f"Loading parsed lines from {parsed_lines_pkl}...")
    with open(parsed_lines_pkl, 'rb') as f:
        parsed_lines = pickle.load(f)
    
    # Load anomaly labels
    print(f"Loading anomaly labels from {anomaly_label_csv}...")
    labels = {}
    with open(anomaly_label_csv, 'r') as f:
        header = f.readline().strip()
        for line in f:
            blk_id, label = line.strip().split(',')
            labels[blk_id] = int(label)  # 0=normal, 1=anomaly
    
    print(f"Total labels: {len(labels)} (normal: {sum(1 for v in labels.values() if v==0)}, anomaly: {sum(1 for v in labels.values() if v==1)})")
    
    # Group by block_id (blk_id)
    # HDFS log format: ... blk_-1234567890 ...
    blk_sequences = defaultdict(list)
    blk_id_pattern = re.compile(r'blk_[-]?\d+')
    
    print("Grouping by block_id...")
    for entry in parsed_lines:
        raw_msg = entry.get("raw_message", "")
        match = re.search(r'(blk_[-]?\d+)', raw_msg)
        if match:
            blk_id = match.group(1)
            blk_sequences[blk_id].append(entry["template_id"])
    
    print(f"Found {len(blk_sequences)} unique block_ids")
    
    # Build template vocabulary
    all_templates = set()
    for entry in parsed_lines:
        all_templates.add(entry["template"])
    
    template_vocab = {t: i+1 for i, t in enumerate(sorted(all_templates))}  # 0 = padding
    template_vocab["<PAD>"] = 0
    template_vocab["<UNK>"] = len(template_vocab)
    
    print(f"Template vocabulary size: {len(template_vocab)}")
    
    # Build sequences with labels
    sequences = {}
    unmatched_blocks = 0
    max_len = 0
    
    for blk_id, template_ids in blk_sequences.items():
        if blk_id not in labels:
            unmatched_blocks += 1
            continue
        
        # Convert to template vocab indices
        vocab_ids = [template_vocab.get(tid, template_vocab["<UNK>"]) for tid in template_ids]
        
        # Truncate or pad
        if len(vocab_ids) > max_seq_len:
            vocab_ids = vocab_ids[:max_seq_len]
        elif len(vocab_ids) < max_seq_len:
            vocab_ids = vocab_ids + [0] * (max_seq_len - len(vocab_ids))
        
        max_len = max(max_len, len(vocab_ids))
        
        sequences[blk_id] = {
            "templates": vocab_ids,
            "label": labels[blk_id]
        }
    
    print(f"Created {len(sequences)} sequences")
    print(f"Unmatched blocks (no label): {unmatched_blocks}")
    print(f"Max sequence length (before padding): {max_len}")
    
    # Label distribution
    normal = sum(1 for s in sequences.values() if s["label"] == 0)
    anomaly = sum(1 for s in sequences.values() if s["label"] == 1)
    print(f"Normal: {normal}, Anomaly: {anomaly}")
    
    # Save sequences
    seq_file = Path(output_dir) / "sequences.pkl"
    with open(seq_file, 'wb') as f:
        pickle.dump(sequences, f)
    
    # Save vocabulary
    vocab_file = Path(output_dir) / "template_vocab.json"
    with open(vocab_file, 'w') as f:
        json.dump(template_vocab, f, indent=2)
    
    # Save metadata
    meta_file = Path(output_dir) / "metadata.json"
    metadata = {
        "num_sequences": len(sequences),
        "vocab_size": len(template_vocab),
        "max_seq_len": max_seq_len,
        "num_normal": normal,
        "num_anomaly": anomaly,
        "unmatched_blocks": unmatched_blocks,
    }
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved to {output_dir}")
    return sequences, template_vocab


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Create HDFS sequences from parsed logs")
    parser.add_argument("--parsed", required=True, help="Path to parsed_lines.pkl")
    parser.add_argument("--labels", required=True, help="Path to anomaly_label.csv")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--max-len", type=int, default=100, help="Max sequence length")
    args = parser.parse_args()
    
    create_sequences(args.parsed, args.labels, args.output, args.max_len)


if __name__ == "__main__":
    main()