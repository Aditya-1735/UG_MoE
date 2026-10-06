#!/usr/bin/env python3
"""
RECONSTRUCTED COMPONENT - NOT ORIGINAL SOURCE CODE
Converts raw logs.json to parsed_data format:
- UTC+8 local timestamp → Unix epoch UTC
- Drain log template parsing
Output: parsed_data/SN/templates.json + parsed_data/SN/logs<idx>.csv (timestamp,service,event)
"""
import os
import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict
import pandas as pd

# Drain3 for log template extraction
try:
    from drain3 import TemplateMiner
    from drain3.file_persistence import FilePersistence
    from drain3.template_miner_config import TemplateMinerConfig
except ImportError:
    print("[ERROR] drain3 not installed. Run: pip install drain3")
    exit(1)

SERVICE_NAMES = [
    'social-graph-service', 'compose-post-service', 'post-storage-service',
    'user-timeline-service', 'url-shorten-service', 'user-service',
    'media-service', 'text-service', 'unique-id-service', 'user-mention-service',
    'home-timeline-service', 'nginx-web-server'
]

RAW_BASE = Path("C:/Users/as999/OneDrive/Desktop/trash/impl/raw_data/SN Dataset")
PARSED_BASE = Path("C:/Users/as999/OneDrive/Desktop/trash/impl/project/parsed_data")

# UTC+8 timezone
UTC_PLUS_8 = timezone(timedelta(hours=8))


def parse_log_timestamp(ts_str: str) -> int:
    """
    Parse log timestamp string like '[2022-Apr-17 10:12:50.490796]' 
    from UTC+8 local time to Unix epoch UTC (integer seconds).
    """
    # Remove brackets
    ts_str = ts_str.strip('[]')
    # Parse as UTC+8 local time
    dt = datetime.strptime(ts_str, "%Y-%b-%d %H:%M:%S.%f")
    dt = dt.replace(tzinfo=UTC_PLUS_8)
    # Convert to UTC
    dt_utc = dt.astimezone(timezone.utc)
    return int(dt_utc.timestamp())


def extract_message(log_line: str) -> str:
    """Extract the message part after the log level and context."""
    # Format: [timestamp] <level>: (file:line:func) message
    match = re.match(r'\[.*?\]\s*<\w+>:\s*\([^)]+\)\s*(.*)', log_line)
    if match:
        return match.group(1).strip()
    return log_line


def convert_logs_for_experiment(exp_dir: Path, exp_idx: int, fault_type: str, 
                                 template_miner: TemplateMiner) -> bool:
    """Convert logs for a single experiment."""
    logs_src = exp_dir / "logs.json"
    if not logs_src.exists():
        print(f"  [ERROR] No logs.json in {exp_dir}")
        return False
    
    with open(logs_src, 'r') as f:
        logs_data = json.load(f)
    
    # Validate service keys
    for service in logs_data.keys():
        if service not in SERVICE_NAMES:
            print(f"  [WARN] Unknown service in logs: {service}")
    
    all_rows = []  # (timestamp, service, event_id)
    
    for service, log_lines in logs_data.items():
        for log_line in log_lines:
            # Extract timestamp from bracketed prefix
            ts_match = re.match(r'\[(.*?)\]', log_line)
            if not ts_match:
                continue
            
            try:
                timestamp = parse_log_timestamp(ts_match.group(1))
            except Exception as e:
                print(f"  [WARN] Failed to parse timestamp: {log_line[:50]}... Error: {e}")
                continue
            
            # Extract message and get template ID
            message = extract_message(log_line)
            template_miner.add_log_message(message)
            cluster = template_miner.match(message)
            event_id = cluster.cluster_id if cluster else 0  # 0 = Unseen
            
            all_rows.append({
                'timestamp': timestamp,
                'service': service,
                'event': event_id
            })
    
    if not all_rows:
        print(f"  [WARN] No valid log entries found")
        return False
    
    # Write logs<idx>.csv
    logs_csv = PARSED_BASE / "SN" / f"logs{exp_idx}.csv"
    df = pd.DataFrame(all_rows)
    df.to_csv(logs_csv, index=False)
    
    print(f"  [OK] Converted {len(all_rows)} log entries for {fault_type} experiment {exp_idx}")
    return True


def main():
    print("=" * 60)
    print("RECONSTRUCTED: convert_logs.py")
    print("Raw logs.json (UTC+8) -> templates.json + logs<idx>.csv")
    print("=" * 60)
    
    PARSED_BASE.mkdir(parents=True, exist_ok=True)
    (PARSED_BASE / "SN").mkdir(parents=True, exist_ok=True)
    
    # Initialize Drain template miner
    config = TemplateMinerConfig()
    config.load("drain3.ini") if os.path.exists("drain3.ini") else None
    # Use default config if no ini file
    persistence = FilePersistence("drain3_state.bin")
    template_miner = TemplateMiner(persistence, config)
    
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
        if not convert_logs_for_experiment(exp_dir, i, "fault", template_miner):
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
        if not convert_logs_for_experiment(exp_dir, exp_idx, "no_fault", template_miner):
            print(f"  [FAILED] Experiment {exp_idx}")
            return 1
    
    # Save templates.json
    templates = [cluster.get_template() for cluster in template_miner.drain.clusters]
    templates_json = PARSED_BASE / "SN" / "templates.json"
    with open(templates_json, 'w') as f:
        json.dump(templates, f, indent=2)
    
    print(f"\n[OK] Saved {len(templates)} templates to {templates_json}")
    print("=" * 60)
    print("ALL LOGS CONVERSION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())