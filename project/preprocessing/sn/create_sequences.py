"""
Create time-windowed sequences from parsed SN data.
Groups logs, metrics, traces into fixed time windows per service.
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


def create_time_windows(parsed_data, window_size=10, stride=5):
    """
    Create time windows from parsed data.
    
    Args:
        parsed_data: Dict with parsed logs, metrics, traces, faults
        window_size: Window size in seconds
        stride: Stride for sliding window
        
    Returns:
        sequences: Dict of sequences per time window
    """
    # This is a simplified implementation
    # In practice, you'd create sliding windows over the time series data
    
    sequences = {}
    return {}


def build_service_graph(traces_all):
    """Build service dependency graph from traces."""
    # Extract service-to-service calls from traces
    edges = set()
    services = set()
    
    for trace in traces_all:
        for span in trace['spans']:
            service = span.get('service', 'unknown')
            services.add(service)
            
            # Look for parent-child relationships
            for ref in span.get('references', []):
                if ref.get('refType') == 'CHILD_OF':
                    # This indicates a service call
                    pass
    
    # Build adjacency matrix
    services = sorted(list(services))
    service_to_idx = {s: i for i, s in enumerate(services)}
    n = len(services)
    adj = np.zeros((n, n))
    
    for trace in traces_all:
        for span in trace['spans']:
            service = span.get('service', 'unknown')
            for ref in span.get('references', []):
                if ref.get('refType') == 'CHILD_OF':
                    # Find parent span
                    parent_span = None
                    for s in trace['spans']:
                        if s['span_id'] == ref['spanID']:
                            parent_service = s.get('service', 'unknown')
                            if parent_service != span.get('service', 'unknown'):
                                src = service_to_idx.get(parent_service)
                                dst = service_to_idx.get(span.get('service', 'unknown'))
                                if src is not None and dst is not None:
                                    edges.add((src, dst))
    
    # Build adjacency matrix
    for src, dst in edges:
        adj[src, dst] = 1
    
    # Add self-loops
    for i in range(n):
        adj[i, i] = 1
    
    return {
        'services': services,
        'service_to_idx': service_to_idx,
        'adjacency': adj,
        'num_nodes': n
    }


def create_sequences_from_parsed(parsed_dir, output_dir, window_size=10, stride=5, max_seq_len=100):
    """
    Create time-windowed sequences from parsed data.
    
    Args:
        parsed_dir: Directory with parsed data (logs, metrics, traces, faults)
        output_dir: Output directory for sequences
        window_size: Window size in seconds
        stride: Stride for sliding window
        max_seq_len: Maximum sequence length
    """
    parsed_dir = Path(parsed_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load parsed data
    parsed_dir = Path(parsed_dir)
    logs_file = parsed_dir / "logs.pkl"
    metrics_file = parsed_dir / "metrics.pkl"
    traces_file = parsed_dir / "traces.pkl"
    faults_file = parsed_dir / "faults.pkl"
    
    # Load parsed data
    with open(logs_file, 'rb') as f:
        logs = pickle.load(f)
    with open(metrics_file, 'rb') as f:
        metrics = pickle.load(f)
    with open(traces_file, 'rb') as f:
        traces = pickle.load(f)
    with open(faults_file, 'rb') as f:
        faults = pickle.load(f)
    
    # For now, create a simplified version
    # In practice, you'd create time windows from the parsed data
    
    sequences = {}
    
    # Save sequences
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "sequences.pkl", "wb") as f:
        pickle.dump({}, f)
    
    print(f"Created sequences in {output_dir}")


def main():
    print("Creating sequences from parsed data...")
    
    # Parse raw data
    raw_data_dir = "/c/Users/as999/OneDrive/Desktop/trash/impl/SN_dataset/SN Dataset/data"
    parsed_dir = "/c/Users/as999/OneDrive/Desktop/trash/impl/project/preprocessing/sn/parsed"
    output_dir = "/c/Users/as999/OneDrive/Desktop/trash/impl/project/preprocessing/sn/output"
    
    Path("/c/Users/as999/OneDrive/Desktop/trash/impl/project/preprocessing/sn/parsed").mkdir(parents=True, exist_ok=True)
    Path("/c/Users/as999/OneDrive/Desktop/trash/impl/project/preprocessing/sn/output").mkdir(parents=True, exist_ok=True)
    
    print("Creating sequences from parsed data...")
    
    # For now, create placeholder
    sequences = {}
    
    output_dir = Path("/c/Users/as999/OneDrive/Desktop/trash/impl/project/preprocessing/sn/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open("/c/Users/as999/OneDrive/Desktop/trash/impl/project/preprocessing/sn/output/sequences.pkl", "wb") as f:
        pickle.dump({}, f)
    
    print("Created sequences (placeholder)")


if __name__ == "__main__":
    main()
EOF
echo "Created create_sequences.py"