#!/usr/bin/env python3
"""
Convert SN dataset raw data to Kilo MoE chunk format.

Reads SN dataset from:
  C:/Users/as999/OneDrive/Desktop/trash/impl/SN_dataset/SN Dataset/data/

Outputs:
  C:/Users/as999/OneDrive/Desktop/trash/impl/project/preprocessing/sn/
    chunk_train.pkl
    chunk_test.pkl
    metadata.json

Each chunk directory contains:
  metrics/*.csv - per service, per second (7 metrics)
  logs.json    - list of log lines with timestamps
  spans.json   - list of trace spans with timestamps

Output chunks are split into CHUNK_SIZE=10 timestep sub-chunks.
"""

import sys
import json
import pickle
import numpy as np
import csv
from pathlib import Path
from collections import defaultdict
from datetime import datetime

SN_DATA_DIR = Path(r"C:/Users/as999/OneDrive/Desktop/trash/impl/SN_dataset/SN Dataset/data")
OUTPUT_DIR = Path("C:/Users/as999/OneDrive/Desktop/trash/impl/project/preprocessing/sn")

CHUNK_SIZE = 10  # Split each 1411-step sequence into 10-timestep sub-chunks


def load_sn_data(sn_data_dir):
    """Load all SN chunk directories."""
    chunks = []

    for chunk_dir in sorted(sn_data_dir.iterdir()):
        if not chunk_dir.is_dir():
            continue

        chunk_id = chunk_dir.name
        print(f"Processing chunk: {chunk_id}")

        # Load metrics for all services
        metrics_dir = chunk_dir / "metrics"
        metrics_files = {}
        for csv_file in sorted(metrics_dir.glob("*.csv")):
            service_name = csv_file.stem
            metrics_files[service_name] = csv_file

        # Load logs
        logs_path = chunk_dir / "logs.json"
        with open(logs_path, 'r') as f:
            logs_data = json.load(f)

        # Load traces
        spans_path = chunk_dir / "spans.json"
        with open(spans_path, 'r') as f:
            traces_data = json.load(f)

        yield {
            "chunk_id": chunk_id,
            "metrics": metrics_files,
            "logs": logs_data,
            "traces": traces_data,
        }


def read_metrics_csv(csv_path):
    """Read metrics CSV and return list of row dicts."""
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def build_chunks(chunk_data):
    """Convert raw chunk data into multiple sub-chunks of CHUNK_SIZE time steps."""
    metrics_files = chunk_data["metrics"]
    logs_data = chunk_data["logs"]
    traces_data = chunk_data["traces"]

    # Get service names
    service_names = sorted(metrics_files.keys())
    num_nodes = len(service_names)
    node_to_idx = {name: i for i, name in enumerate(service_names)}

    # Process metrics: read each service CSV
    metric_arrays = []
    metric_cols = ["cpu_usage_system", "cpu_usage_total", "cpu_usage_user",
                   "memory_usage", "memory_working_set", "rx_bytes", "tx_bytes"]

    for service_name in service_names:
        csv_path = metrics_files[service_name]
        rows = read_metrics_csv(csv_path)
        metric_values = []
        for row in rows:
            row_vals = []
            for col in metric_cols:
                try:
                    row_vals.append(float(row[col]))
                except (ValueError, KeyError):
                    row_vals.append(0.0)
            metric_values.append(row_vals)
        metric_arrays.append(np.array(metric_values, dtype=np.float32))

    metrics_array = np.stack(metric_arrays, axis=0)  # [num_nodes, chunk_length, metric_num]
    num_nodes, num_timesteps, metric_num = metrics_array.shape
    num_subchunks = num_timesteps // CHUNK_SIZE

    # Process logs: count template IDs per service per sub-chunk
    log_templates = set()
    for service_name, log_lines in logs_data.items():
        for line in log_lines:
            parts = line.split()
            if parts:
                log_templates.add(parts[0])

    log_templates = sorted(log_templates)
    template_to_idx = {t: i for i, t in enumerate(log_templates)}
    num_templates = len(log_templates)

    # Parse log timestamps to assign to sub-chunks
    sub_chunk_logs = {}  # {sub_idx: {node_idx: {template_id: count}}}
    first_log_ts = None

    for service_name, log_lines in logs_data.items():
        if service_name not in node_to_idx:
            continue
        node_idx = node_to_idx[service_name]
        for line in log_lines:
            # Extract timestamp: [2022-Apr-17 11:27:00.063087]
            ts_match = line.split(']')[0].strip('[') if ']' in line else None
            if ts_match:
                try:
                    ts = datetime.strptime(ts_match, "%Y-%b-%d %H:%M:%S.%f")
                    if first_log_ts is None:
                        first_log_ts = ts
                    sub_idx = int((ts.timestamp() - first_log_ts.timestamp()) // CHUNK_SIZE)
                    if 0 <= sub_idx < num_subchunks:
                        if sub_idx not in sub_chunk_logs:
                            sub_chunk_logs[sub_idx] = defaultdict(lambda: defaultdict(int))
                        sub_chunk_logs[sub_idx][node_idx][parts[0]] += 1
                except:
                    pass

    # Process traces: count spans per service per sub-chunk
    traces_data_list = traces_data if isinstance(traces_data, list) else traces_data.get("spans", [])
    trace_counts = defaultdict(lambda: defaultdict(int))  # {sub_idx: {node_idx: count}}
    for span in traces_data_list:
        service_name = span.get("serviceName", "")
        if service_name in node_to_idx:
            node_idx = node_to_idx[service_name]
            # Parse start time from span
            start_time_str = span.get("startTime", "")
            try:
                ts = datetime.strptime(start_time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
                sub_idx = int((ts.timestamp() - first_log_ts.timestamp()) // CHUNK_SIZE)
                if 0 <= sub_idx < num_subchunks:
                    trace_counts[sub_idx][node_idx] += 1
            except:
                pass  # Can't parse timestamp, skip

    # Determine culprit from log content
    culprit = 0  # default normal
    for service_name, log_lines in logs_data.items():
        for line in log_lines:
            line_lower = line.lower()
            if "error" in line_lower or "exception" in line_lower or "fail" in line_lower:
                if service_name in node_to_idx:
                    culprit = node_to_idx[service_name] + 1  # 1-indexed anomaly
                    break

    # Build per-sub-chunk dicts
    all_chunks = []
    for sub_idx in range(num_subchunks):
        start = sub_idx * CHUNK_SIZE
        end = start + CHUNK_SIZE
        sub_metrics = metrics_array[:, start:end, :]  # [num_nodes, CHUNK_SIZE, metric_num]

        # Build sub-chunk log counts
        sub_logs = np.zeros((num_nodes, num_templates), dtype=np.float32)
        for service_name in service_names:
            if service_name not in node_to_idx:
                continue
            node_idx = node_to_idx[service_name]
            for template_id, count in sub_chunk_logs.get(sub_idx, {}).get(node_idx, {}).items():
                if template_id in template_to_idx:
                    sub_logs[node_idx, template_to_idx[template_id]] = float(count)

        # Sub-chunk trace counts
        sub_traces = np.zeros((num_nodes, 1), dtype=np.float32)
        for service_name, count in trace_counts.get(sub_idx, {}).items():
            if service_name in node_to_idx:
                sub_traces[node_to_idx[service_name], 0] = float(count)

        chunk_dict = {
            "metrics": sub_metrics,
            "traces": sub_traces,
            "logs": sub_logs,
            "culprit": culprit,
        }
        all_chunks.append(chunk_dict)

    return all_chunks


def main():
    sn_data_dir = SN_DATA_DIR
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SN dataset...")
    chunks = list(load_sn_data(sn_data_dir))
    print(f"Loaded {len(chunks)} raw chunks")

    # Convert each chunk into sub-chunks of CHUNK_SIZE time steps
    all_chunks = []
    for chunk_data in chunks:
        sub_chunks = build_chunks(chunk_data)
        all_chunks.extend(sub_chunks)

    print(f"Total sub-chunks: {len(all_chunks)}")

    # Split train/test (80/20)
    n_chunks = len(all_chunks)
    n_train = int(n_chunks * 0.8)
    train_chunks = all_chunks[:n_train]
    test_chunks = all_chunks[n_train:]

    print(f"Train chunks: {len(train_chunks)}, Test chunks: {len(test_chunks)}")

    # Save as dict keyed by chunk index
    train_dict = {i: chunk for i, chunk in enumerate(train_chunks)}
    test_dict = {i: chunk for i, chunk in enumerate(test_chunks)}

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "chunk_train.pkl", "wb") as f:
        pickle.dump(train_dict, f)

    with open(output_dir / "chunk_test.pkl", "wb") as f:
        pickle.dump(test_dict, f)

    # Create metadata.json
    metadata = {
        "node_num": 12,
        "edges": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]],
        "event_num": 14,
        "metric_num": 7,
        "chunk_length": CHUNK_SIZE,
        "chunk_num": len(all_chunks),
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! Saved to {output_dir}")
    print(f"Train chunks: {len(train_chunks)}, Test chunks: {len(test_chunks)}")
    print(f"Metadata: {metadata}")


if __name__ == "__main__":
    main()