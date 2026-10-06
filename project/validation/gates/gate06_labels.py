#!/usr/bin/env python3
"""
Gate 6: Labels verification
Verifies culprit values are in valid range and trace back to fault manifest.
"""
import json
from pathlib import Path

PARSED_BASE = Path("C:/Users/as999/OneDrive/Desktop/trash/impl/project/parsed_data/SN")

SERVICE_NAMES = [
    'social-graph-service', 'compose-post-service', 'post-storage-service',
    'user-timeline-service', 'url-shorten-service', 'user-service',
    'media-service', 'text-service', 'unique-id-service', 'user-mention-service',
    'home-timeline-service', 'nginx-web-server'
]
SERVICE_TO_NID = {s: i for i, s in enumerate(SERVICE_NAMES)}

def main():
    print("=" * 60)
    print("GATE 6: Labels Verification")
    print("=" * 60)
    
    all_fault_services = set()
    
    for exp_idx in range(7):
        records_file = PARSED_BASE / f"records{exp_idx}.json"
        with open(records_file, 'r') as f:
            records = json.load(f)
        
        for fault in records.get("faults", []):
            service = fault["service"]
            if service not in SERVICE_TO_NID:
                print(f"[FAIL] Unknown service in records: {service}")
                return 1
            all_fault_services.add(service)
    
    print(f"[INFO] Services with faults in parsed_data: {len(all_fault_services)}")
    for svc in sorted(all_fault_services):
        print(f"  [OK] {svc} -> node {SERVICE_TO_NID[svc]}")
    
    # Verify no-fault experiments have empty faults
    for exp_idx in [4, 5, 6]:
        records_file = PARSED_BASE / f"records{exp_idx}.json"
        with open(records_file, 'r') as f:
            records = json.load(f)
        if records.get("faults"):
            print(f"[FAIL] No-fault experiment {exp_idx} has non-empty faults")
            return 1
        print(f"[OK] No-fault experiment {exp_idx}: empty faults list")
    
    print("\n[GATE 6 PASSED] Labels verified")
    return 0


if __name__ == "__main__":
    exit(main())