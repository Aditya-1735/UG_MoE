#!/usr/bin/env python3
"""
Gate 1: Raw dataset integrity
Verifies the raw archive extracts to the documented 7-experiment structure.
"""
import json
from pathlib import Path

RAW_BASE = Path("C:/Users/as999/OneDrive/Desktop/trash/impl/raw_data/SN Dataset")

EXPECTED_FAULT_EXPS = 4
EXPECTED_NOFAULT_EXPS = 3
EXPECTED_SERVICES = 12

def main():
    print("=" * 60)
    print("GATE 1: Raw Dataset Integrity")
    print("=" * 60)
    
    # Check SN-all.tgz exists
    sn_all = RAW_BASE / "SN-all.tgz"
    if not sn_all.exists():
        print("[FAIL] SN-all.tgz not found")
        return 1
    print(f"[OK] SN-all.tgz exists ({sn_all.stat().st_size} bytes)")
    
    # Check fault experiments (data/)
    data_dir = RAW_BASE / "data"
    fault_exps = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("SN.")]
    print(f"[INFO] Found {len(fault_exps)} fault experiment directories")
    
    if len(fault_exps) != EXPECTED_FAULT_EXPS:
        print(f"[FAIL] Expected {EXPECTED_FAULT_EXPS} fault experiments, got {len(fault_exps)}")
        return 1
    print(f"[OK] Fault experiment count: {len(fault_exps)}")
    
    # Check no-fault experiments (no fault/)
    nofault_dir = RAW_BASE / "no fault"
    nofault_exps = [d for d in nofault_dir.iterdir() if d.is_dir() and d.name.startswith("SN.")]
    print(f"[INFO] Found {len(nofault_exps)} no-fault experiment directories")
    
    if len(nofault_exps) != EXPECTED_NOFAULT_EXPS:
        print(f"[FAIL] Expected {EXPECTED_NOFAULT_EXPS} no-fault experiments, got {len(nofault_exps)}")
        return 1
    print(f"[OK] No-fault experiment count: {len(nofault_exps)}")
    
    # Verify each experiment has required files
    for exp_dir in fault_exps + nofault_exps:
        required = ["logs.json", "spans.json", "metrics"]
        for req in required:
            path = exp_dir / req
            if not path.exists():
                print(f"[FAIL] Missing {req} in {exp_dir.name}")
                return 1
        # Check metrics has 12 CSVs
        metrics_dir = exp_dir / "metrics"
        csv_files = list(metrics_dir.glob("*.csv"))
        if len(csv_files) != EXPECTED_SERVICES:
            print(f"[FAIL] Expected {EXPECTED_SERVICES} metric CSVs in {exp_dir.name}, got {len(csv_files)}")
            return 1
    
    print(f"[OK] All experiments have required files (logs.json, spans.json, metrics/ with 12 CSVs)")
    
    # Check fault JSON files
    fault_jsons = list(data_dir.glob("SN.fault-*.json"))
    if len(fault_jsons) != EXPECTED_FAULT_EXPS:
        print(f"[FAIL] Expected {EXPECTED_FAULT_EXPS} fault JSONs, got {len(fault_jsons)}")
        return 1
    print(f"[OK] Fault JSON count: {len(fault_jsons)}")
    
    nofault_jsons = list(nofault_dir.glob("SN.fault-*.json"))
    if len(nofault_jsons) != EXPECTED_NOFAULT_EXPS:
        print(f"[FAIL] Expected {EXPECTED_NOFAULT_EXPS} no-fault JSONs, got {len(nofault_jsons)}")
        return 1
    print(f"[OK] No-fault JSON count: {len(nofault_jsons)}")
    
    print("\n[GATE 1 PASSED] Raw dataset integrity verified")
    return 0


if __name__ == "__main__":
    exit(main())