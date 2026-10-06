#!/usr/bin/env python3
"""
Gate 3: parsed_data schema correctness
Verifies our reconstructed parsed_data matches what Eadro's single_process.py/align.py expect.
"""
import json
import pandas as pd
from pathlib import Path

PARSED_BASE = Path("C:/Users/as999/OneDrive/Desktop/trash/impl/project/parsed_data/SN")

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

def main():
    print("=" * 60)
    print("GATE 3: parsed_data Schema Correctness")
    print("=" * 60)
    
    # Check experiments 0-6
    for exp_idx in range(7):
        print(f"\n--- Experiment {exp_idx} ---")
        
        # records<idx>.json
        records_file = PARSED_BASE / f"records{exp_idx}.json"
        if not records_file.exists():
            print(f"[FAIL] Missing records{exp_idx}.json")
            return 1
        with open(records_file, 'r') as f:
            records = json.load(f)
        required_keys = {"faults", "start", "end"}
        if set(records.keys()) != required_keys:
            print(f"[FAIL] records{exp_idx}.json keys mismatch: {set(records.keys())} vs {required_keys}")
            return 1
        for fault in records.get("faults", []):
            if not all(k in fault for k in ["s", "e", "service"]):
                print(f"[FAIL] Fault record missing keys: {fault}")
                return 1
            if fault["service"] not in SERVICE_NAMES:
                print(f"[FAIL] Unknown service in fault: {fault['service']}")
                return 1
        print(f"[OK] records{exp_idx}.json schema valid ({len(records['faults'])} faults)")
        
        # metrics<idx>/<service>.csv
        metrics_dir = PARSED_BASE / f"metrics{exp_idx}"
        if not metrics_dir.exists():
            print(f"[FAIL] Missing metrics{exp_idx}/ directory")
            return 1
        for service in SERVICE_NAMES:
            csv_file = metrics_dir / f"{service}.csv"
            if not csv_file.exists():
                print(f"[FAIL] Missing {service}.csv in metrics{exp_idx}")
                return 1
            df = pd.read_csv(csv_file)
            expected_cols = ['timestamp'] + EXPECTED_METRICS
            if list(df.columns) != expected_cols:
                print(f"[FAIL] Schema mismatch in {service}.csv: {list(df.columns)}")
                return 1
            if not pd.api.types.is_integer_dtype(df['timestamp']):
                print(f"[FAIL] Non-integer timestamps in {service}.csv")
                return 1
        print(f"[OK] metrics{exp_idx}/: 12 CSVs with correct schema")
        
        # logs<idx>.csv
        logs_file = PARSED_BASE / f"logs{exp_idx}.csv"
        if not logs_file.exists():
            print(f"[FAIL] Missing logs{exp_idx}.csv")
            return 1
        df = pd.read_csv(logs_file)
        if list(df.columns) != ['timestamp', 'service', 'event']:
            print(f"[FAIL] logs{exp_idx}.csv columns mismatch: {list(df.columns)}")
            return 1
        # Check services are valid
        for svc in df['service'].unique():
            if svc not in SERVICE_NAMES:
                print(f"[FAIL] Unknown service in logs: {svc}")
                return 1
        print(f"[OK] logs{exp_idx}.csv: {len(df)} entries, valid services")
        
        # traces<idx>.json
        traces_file = PARSED_BASE / f"traces{exp_idx}.json"
        if not traces_file.exists():
            print(f"[FAIL] Missing traces{exp_idx}.json")
            return 1
        with open(traces_file, 'r') as f:
            traces = json.load(f)
        # Should be dict with string keys (second timestamps)
        for ts_key, edges in traces.items():
            if not isinstance(ts_key, str):
                print(f"[FAIL] Trace key not string: {ts_key}")
                return 1
            for edge_key, latencies in edges.items():
                if not isinstance(edge_key, str) or '-' not in edge_key:
                    print(f"[FAIL] Invalid edge key format: {edge_key}")
                    return 1
                if not isinstance(latencies, list):
                    print(f"[FAIL] Latencies not list: {edge_key}")
                    return 1
        print(f"[OK] traces{exp_idx}.json: {len(traces)} second buckets")
    
    # templates.json
    templates_file = PARSED_BASE / "templates.json"
    if not templates_file.exists():
        print(f"[FAIL] Missing templates.json")
        return 1
    with open(templates_file, 'r') as f:
        templates = json.load(f)
    if not isinstance(templates, list):
        print(f"[FAIL] templates.json not a list")
        return 1
    print(f"[OK] templates.json: {len(templates)} templates")
    
    print("\n[GATE 3 PASSED] parsed_data schema verified")
    return 0


if __name__ == "__main__":
    exit(main())