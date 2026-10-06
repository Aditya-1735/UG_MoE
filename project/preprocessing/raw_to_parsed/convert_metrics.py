#!/usr/bin/env python3
"""
RECONSTRUCTED COMPONENT - NOT ORIGINAL SOURCE CODE
Converts raw metrics CSVs to parsed_data format.
Input: raw_data/SN Dataset/data/SN.<ts>/metrics/<service>.csv
Output: parsed_data/SN/metrics<idx>/<service>.csv (same schema, copy only)
"""
import os
import shutil
import pandas as pd
from pathlib import Path


SERVICE_NAMES = [
    'social-graph-service', 'compose-post-service', 'post-storage-service',
    'user-timeline-service', 'url-shorten-service', 'user-service',
    'media-service', 'text-service', 'unique-id-service', 'user-mention-service',
    'home-timeline-service', 'nginx-web-server'
]

EXPECTED_METRICS = [
    'cpu_usage_system', 'cpu_usage_total', 'cpu_usage_user',
    'memory_usage', 'memory_working_set', 'rx_bytes', 'tx_bytes'
]

RAW_BASE = Path("C:/Users/as999/OneDrive/Desktop/trash/impl/raw_data/SN Dataset")
PARSED_BASE = Path("C:/Users/as999/OneDrive/Desktop/trash/impl/project/parsed_data")


def find_experiment_dirs(data_root: Path, prefix: str) -> list:
    """Find experiment directories matching prefix (e.g., 'SN.2022-04-17T181245D2022-04-17T183616')"""
    dirs = []
    for item in data_root.iterdir():
        if item.is_dir() and item.name.startswith(prefix):
            dirs.append(item)
    return sorted(dirs)


def convert_metrics_for_experiment(exp_dir: Path, exp_idx: int, fault_type: str) -> bool:
    """
    Convert metrics for a single experiment.
    fault_type: 'fault' or 'no_fault'
    """
    metrics_src = exp_dir / "metrics"
    if not metrics_src.exists():
        print(f"  [ERROR] No metrics directory in {exp_dir}")
        return False
    
    metrics_dst = PARSED_BASE / "SN" / f"metrics{exp_idx}"
    metrics_dst.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    for service in SERVICE_NAMES:
        src_file = metrics_src / f"{service}.csv"
        dst_file = metrics_dst / f"{service}.csv"
        
        if not src_file.exists():
            print(f"  [WARN] Missing {service}.csv in {metrics_src}")
            continue
        
        try:
            df = pd.read_csv(src_file)
            
            # Validate schema
            if list(df.columns) != ['timestamp'] + EXPECTED_METRICS:
                print(f"  [WARN] Schema mismatch in {service}.csv: {list(df.columns)}")
            
            # Validate timestamp format (Unix epoch integer)
            if not pd.api.types.is_integer_dtype(df['timestamp']):
                print(f"  [WARN] Non-integer timestamps in {service}.csv")
            
            # Copy as-is (schema already matches what Eadro expects)
            df.to_csv(dst_file, index=False)
            success_count += 1
            
        except Exception as e:
            print(f"  [ERROR] Failed to process {service}.csv: {e}")
            return False
    
    print(f"  [OK] Converted {success_count}/12 service metrics for {fault_type} experiment {exp_idx}")
    return success_count == 12


def main():
    print("=" * 60)
    print("RECONSTRUCTED: convert_metrics.py")
    print("Raw metrics -> parsed_data/SN/metrics<idx>/<service>.csv")
    print("=" * 60)
    
    PARSED_BASE.mkdir(parents=True, exist_ok=True)
    (PARSED_BASE / "SN").mkdir(parents=True, exist_ok=True)
    
    # Process fault experiments (data/)
    fault_data_dir = RAW_BASE / "data"
    fault_experiments = []
    for item in fault_data_dir.iterdir():
        if item.is_dir() and item.name.startswith("SN.") and not item.name.endswith(".json"):
            fault_experiments.append(item)
    fault_experiments = sorted(fault_experiments)
    
    print(f"\nFound {len(fault_experiments)} fault experiments")
    
    for i, exp_dir in enumerate(fault_experiments):
        print(f"\nProcessing fault experiment {i}: {exp_dir.name}")
        if not convert_metrics_for_experiment(exp_dir, i, "fault"):
            print(f"  [FAILED] Experiment {i}")
            return 1
    
    # Process no-fault experiments (no fault/)
    nofault_data_dir = RAW_BASE / "no fault"
    nofault_experiments = []
    for item in nofault_data_dir.iterdir():
        if item.is_dir() and item.name.startswith("SN."):
            nofault_experiments.append(item)
    nofault_experiments = sorted(nofault_experiments)
    
    print(f"\nFound {len(nofault_experiments)} no-fault experiments")
    
    for i, exp_dir in enumerate(nofault_experiments):
        exp_idx = len(fault_experiments) + i
        print(f"\nProcessing no-fault experiment {exp_idx}: {exp_dir.name}")
        if not convert_metrics_for_experiment(exp_dir, exp_idx, "no_fault"):
            print(f"  [FAILED] Experiment {exp_idx}")
            return 1
    
    print("\n" + "=" * 60)
    print("ALL METRICS CONVERSION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())