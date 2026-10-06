#!/usr/bin/env python3
"""
Gate 5: Tensor dimensions
Verifies metrics[12,10,7], traces[12,10,1] (pre-trim) or [12,10,2] (Eadro native)
"""
import pickle
from pathlib import Path

CHUNKS_DIR = Path("C:/Users/as999/OneDrive/Desktop/trash/impl/project/preprocessing/muad_compat/output")

def main():
    print("=" * 60)
    print("GATE 5: Tensor Dimensions")
    print("=" * 60)
    
    for split_name in ["chunk_train.pkl", "chunk_test.pkl"]:
        split_path = CHUNKS_DIR / split_name
        if not split_path.exists():
            print(f"[FAIL] Missing {split_name}")
            return 1
        
        with open(split_path, 'rb') as f:
            chunks = pickle.load(f)
        
        print(f"\n--- {split_name} ---")
        
        sample_chunk = list(chunks.values())[0]
        
        # Check metrics: [12, 10, 7]
        metrics_shape = sample_chunk["metrics"].shape
        expected_metrics = (12, 10, 7)
        if metrics_shape != expected_metrics:
            print(f"[FAIL] metrics shape: {metrics_shape} != {expected_metrics}")
            return 1
        print(f"[OK] metrics: {metrics_shape}")
        
        # Check traces: [12, 10, 1] (MUAD compat - post trim)
        traces_shape = sample_chunk["traces"].shape
        expected_traces = (12, 10, 1)
        if traces_shape != expected_traces:
            print(f"[FAIL] traces shape: {traces_shape} != {expected_traces}")
            return 1
        print(f"[OK] traces (MUAD compat): {traces_shape}")
        
        # Check logs: [12, event_num] - event_num may differ from recovered 15
        logs_shape = sample_chunk["logs"].shape
        if logs_shape[0] != 12 or logs_shape[1] < 1:
            print(f"[FAIL] logs shape: {logs_shape} (expected [12, event_num])")
            return 1
        print(f"[OK] logs: {logs_shape} (event_num={logs_shape[1]})")
        
        # Check culprit is int
        if not isinstance(sample_chunk["culprit"], int):
            print(f"[FAIL] culprit not int: {type(sample_chunk['culprit'])}")
            return 1
        print(f"[OK] culprit: int")
        
        # Verify all chunks have same shapes
        for chunk_id, chunk in chunks.items():
            if chunk["metrics"].shape != expected_metrics:
                print(f"[FAIL] metrics shape mismatch in {chunk_id}")
                return 1
            if chunk["traces"].shape != expected_traces:
                print(f"[FAIL] traces shape mismatch in {chunk_id}")
                return 1
            if chunk["logs"].shape[0] != 12:
                print(f"[FAIL] logs shape mismatch in {chunk_id}")
                return 1
        
        print(f"[OK] All {len(chunks)} chunks have correct shapes")
    
    print("\n[GATE 5 PASSED] Tensor dimensions verified")
    return 0


if __name__ == "__main__":
    exit(main())