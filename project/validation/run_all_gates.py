#!/usr/bin/env python3
"""
Run all validation gates.
"""
import subprocess
import sys
from pathlib import Path

GATES_DIR = Path(__file__).parent / "gates"

GATES = [
    ("gate01_raw_integrity.py", "Raw dataset integrity"),
    ("gate02_fault_manifest.py", "Fault manifest correctness"),
    ("gate03_parsed_data_schema.py", "parsed_data schema correctness"),
    ("gate04_chunk_schema.py", "Eadro chunk schema correctness"),
    ("gate05_tensor_dims.py", "Tensor dimensions"),
    ("gate06_labels.py", "Labels verification"),
    ("gate07_graph_structure.py", "Graph structure verification"),
    ("gate08_split.py", "Train/test split"),
    ("gate09_artifact_comparison.py", "Recovered artifact comparison"),
    ("gate10_dataloader_compat.py", "DataLoader compatibility"),
]

def run_gate(gate_file: str, description: str) -> bool:
    print(f"\n{'='*60}")
    print(f"RUNNING: {description} ({gate_file})")
    print(f"{'='*60}")
    
    gate_path = GATES_DIR / gate_file
    result = subprocess.run([sys.executable, str(gate_path)], capture_output=False)
    
    if result.returncode == 0:
        print(f"\n[OK] {description} PASSED")
        return True
    else:
        print(f"\n[FAIL] {description} FAILED (return code {result.returncode})")
        return False


def main():
    print("=" * 60)
    print("VALIDATION GATES RUNNER - ALL 10 GATES")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for gate_file, description in GATES:
        if run_gate(gate_file, description):
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())