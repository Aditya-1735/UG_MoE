#!/usr/bin/env python3
"""
RECONSTRUCTED COMPONENT - NOT ORIGINAL SOURCE CODE
Orchestrates the raw → parsed_data conversion pipeline.
Runs: convert_metrics.py, convert_logs.py, convert_traces.py, convert_faults.py
"""
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent


def run_script(script_name: str) -> bool:
    """Run a conversion script and return success status."""
    script_path = SCRIPTS_DIR / script_name
    print(f"\n{'='*60}")
    print(f"RUNNING: {script_name}")
    print(f"{'='*60}")
    
    result = subprocess.run([sys.executable, str(script_path)], capture_output=False)
    
    if result.returncode == 0:
        print(f"\n[OK] {script_name} completed successfully")
        return True
    else:
        print(f"\n[FAIL] {script_name} FAILED with return code {result.returncode}")
        return False


def main():
    print("=" * 60)
    print("RECONSTRUCTED: run_conversion.py")
    print("Orchestrates raw -> parsed_data conversion")
    print("=" * 60)
    
    scripts = [
        "convert_metrics.py",
        "convert_faults.py",  # Faults first (simpler, validates experiment structure)
        "convert_logs.py",    # Logs (needs drain3)
        "convert_traces.py",  # Traces (most complex)
    ]
    
    for script in scripts:
        if not run_script(script):
            print(f"\n[FATAL] Conversion pipeline failed at {script}")
            return 1
    
    print("\n" + "=" * 60)
    print("ALL CONVERSIONS COMPLETED SUCCESSFULLY")
    print("Output: parsed_data/SN/")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())