#!/usr/bin/env python3
"""
Gate 2: Fault manifest correctness
Verifies raw fault record count == 36, type/service distribution matches.
"""
import json
from pathlib import Path

RAW_BASE = Path("C:/Users/as999/OneDrive/Desktop/trash/impl/raw_data/SN Dataset")

EXPECTED_TOTAL_FAULTS = 36
EXPECTED_FAULT_TYPES = {"cpu_load": 12, "network_delay": 12, "network_loss": 12}
EXPECTED_SERVICES = 12

def main():
    print("=" * 60)
    print("GATE 2: Fault Manifest Correctness")
    print("=" * 60)
    
    data_dir = RAW_BASE / "data"
    fault_jsons = sorted(data_dir.glob("SN.fault-*.json"))
    
    total_faults = 0
    fault_type_counts = {}
    service_fault_counts = {}
    
    for fault_file in fault_jsons:
        with open(fault_file, 'r') as f:
            fault_data = json.load(f)
        
        faults = fault_data.get('faults', [])
        total_faults += len(faults)
        
        for fault in faults:
            ftype = fault.get('fault', 'unknown')
            fault_type_counts[ftype] = fault_type_counts.get(ftype, 0) + 1
            
            # Resolve service name
            raw_name = fault.get('name', '')
            service = raw_name
            if service.startswith('socialnetwork-'):
                service = service[len('socialnetwork-'):]
            if service.endswith('-1'):
                service = service[:-2]
            # Apply nginx alias
            if service == 'nginx-thrift':
                service = 'nginx-web-server'
            
            service_fault_counts[service] = service_fault_counts.get(service, 0) + 1
    
    print(f"[INFO] Total fault records: {total_faults}")
    if total_faults != EXPECTED_TOTAL_FAULTS:
        print(f"[FAIL] Expected {EXPECTED_TOTAL_FAULTS} faults, got {total_faults}")
        return 1
    print(f"[OK] Total fault count: {total_faults}")
    
    print(f"\n[INFO] Fault type distribution:")
    for ftype, expected in EXPECTED_FAULT_TYPES.items():
        actual = fault_type_counts.get(ftype, 0)
        if actual != expected:
            print(f"[FAIL] {ftype}: expected {expected}, got {actual}")
            return 1
        print(f"[OK] {ftype}: {actual}")
    
    print(f"\n[INFO] Per-service fault distribution:")
    for service in sorted(service_fault_counts.keys()):
        count = service_fault_counts[service]
        if count != 3:
            print(f"[FAIL] {service}: expected 3, got {count}")
            return 1
        print(f"[OK] {service}: {count}")
    
    if len(service_fault_counts) != EXPECTED_SERVICES:
        print(f"[FAIL] Expected {EXPECTED_SERVICES} services with faults, got {len(service_fault_counts)}")
        return 1
    print(f"[OK] All {EXPECTED_SERVICES} services targeted")
    
    # Verify all 36 records have duration=120
    for fault_file in fault_jsons:
        with open(fault_file, 'r') as f:
            fault_data = json.load(f)
        for fault in fault_data.get('faults', []):
            if fault.get('duration') != 120:
                print(f"[FAIL] Fault duration not 120: {fault}")
                return 1
    print(f"[OK] All faults have duration=120s")
    
    print("\n[GATE 2 PASSED] Fault manifest verified")
    return 0


if __name__ == "__main__":
    exit(main())