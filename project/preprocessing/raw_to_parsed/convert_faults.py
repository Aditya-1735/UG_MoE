#!/usr/bin/env python3
"""
RECONSTRUCTED COMPONENT - NOT ORIGINAL SOURCE CODE
Converts raw SN.fault-*.json to parsed_data format:
- s = fault["start"], e = fault["start"] + fault["duration"]
- service = fault["name"] with socialnetwork- prefix / -1 suffix stripped + nginx alias
Output: parsed_data/SN/records<idx>.json matching align.py::get_basic expected schema {"faults":[{"s":..., "e":..., "service":...}], ...}
"""
import os
import json
from pathlib import Path

SERVICE_NAMES = [
    'social-graph-service', 'compose-post-service', 'post-storage-service',
    'user-timeline-service', 'url-shorten-service', 'user-service',
    'media-service', 'text-service', 'unique-id-service', 'user-mention-service',
    'home-timeline-service', 'nginx-web-server'
]

SERVICE_TO_NID = {s: i for i, s in enumerate(SERVICE_NAMES)}

# Alias mapping
ALIAS_MAP = {
    'socialnetwork-nginx-thrift-1': 'nginx-web-server',
    'nginx-thrift': 'nginx-web-server',
}

RAW_BASE = Path("C:/Users/as999/OneDrive/Desktop/trash/impl/raw_data/SN Dataset")
PARSED_BASE = Path("C:/Users/as999/OneDrive/Desktop/trash/impl/project/parsed_data")


def resolve_service_name(raw_name: str) -> str:
    """Apply alias mapping to resolve service name to Eadro service name."""
    name = raw_name
    if name.startswith('socialnetwork-'):
        name = name[len('socialnetwork-'):]
    if name.endswith('-1'):
        name = name[:-2]
    return ALIAS_MAP.get(name, name)


def convert_faults_for_experiment(fault_file: Path, exp_idx: int, fault_type: str) -> bool:
    """Convert faults for a single experiment."""
    if not fault_file.exists():
        print(f"  [ERROR] No fault file: {fault_file}")
        return False
    
    with open(fault_file, 'r') as f:
        fault_data = json.load(f)
    
    faults = fault_data.get('faults', [])
    # Convert to int for range() compatibility in Eadro align.py
    start = int(fault_data.get('start', 0))
    end = int(fault_data.get('end', 0))
    
    if fault_type == "no_fault":
        # No-fault experiments have empty faults list
        records = {"faults": [], "start": start, "end": end}
    else:
        converted_faults = []
        for fault in faults:
            raw_name = fault.get('name', '')
            fault_type_str = fault.get('fault', '')
            fault_start = int(fault.get('start', 0))
            duration = fault.get('duration', 120)
            
            # Resolve service name
            service = resolve_service_name(raw_name)
            
            if service not in SERVICE_TO_NID:
                print(f"  [WARN] Unknown service after alias resolution: {raw_name} -> {service}")
                continue
            
            s = fault_start
            e = fault_start + duration
            
            converted_faults.append({
                "s": s,
                "e": e,
                "service": service
            })
        
        records = {
            "faults": converted_faults,
            "start": start,
            "end": end
        }
    
    # Write records<idx>.json
    records_json = PARSED_BASE / "SN" / f"records{exp_idx}.json"
    with open(records_json, 'w') as f:
        json.dump(records, f, indent=2)
    
    print(f"  [OK] Converted {len(records['faults'])} fault records for {fault_type} experiment {exp_idx}")
    return True


def main():
    print("=" * 60)
    print("RECONSTRUCTED: convert_faults.py")
    print("Raw SN.fault-*.json -> records<idx>.json")
    print("=" * 60)
    
    PARSED_BASE.mkdir(parents=True, exist_ok=True)
    (PARSED_BASE / "SN").mkdir(parents=True, exist_ok=True)
    
    # Process fault experiments (data/)
    fault_data_dir = RAW_BASE / "data"
    fault_files = sorted(fault_data_dir.glob("SN.fault-*.json"))
    
    print(f"\nFound {len(fault_files)} fault experiment fault files")
    
    for i, fault_file in enumerate(fault_files):
        print(f"\nProcessing fault experiment {i}: {fault_file.name}")
        if not convert_faults_for_experiment(fault_file, i, "fault"):
            print(f"  [FAILED] Experiment {i}")
            return 1
    
    # Process no-fault experiments (no fault/)
    nofault_data_dir = RAW_BASE / "no fault"
    nofault_files = sorted(nofault_data_dir.glob("SN.fault-*.json"))
    
    print(f"\nFound {len(nofault_files)} no-fault experiment fault files")
    
    for i, fault_file in enumerate(nofault_files):
        exp_idx = len(fault_files) + i
        print(f"\nProcessing no-fault experiment {exp_idx}: {fault_file.name}")
        if not convert_faults_for_experiment(fault_file, exp_idx, "no_fault"):
            print(f"  [FAILED] Experiment {exp_idx}")
            return 1
    
    print("\n" + "=" * 60)
    print("ALL FAULTS CONVERSION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())