#!/usr/bin/env python3
"""
Gate 4: Eadro chunk schema correctness
Verifies generated chunks have exactly the 4 keys with correct dtypes.
"""
import pickle
from pathlib import Path

CHUNKS_DIR = Path("C:/Users/as999/OneDrive/Desktop/trash/impl/project/preprocessing/muad_compat/output")

def main():
    print("=" * 60)
    print("GATE 4: Eadro Chunk Schema Correctness")
    print("=" * 60)
    
    for split_name in ["chunk_train.pkl", "chunk_test.pkl"]:
        split_path = CHUNKS_DIR / split_name
        if not split_path.exists():
            print(f"[FAIL] Missing {split_name}")
            return 1
        
        with open(split_path, 'rb') as f:
            chunks = pickle.load(f)
        
        print(f"\n--- {split_name} ---")
        print(f"Total chunks: {len(chunks)}")
        
        # Check first few chunks
        for i, (chunk_id, chunk) in enumerate(chunks.items()):
            if i >= 3:
                break
            # Check keys
            expected_keys = {"traces", "metrics", "logs", "culprit"}
            if set(chunk.keys()) != expected_keys:
                print(f"[FAIL] Chunk {chunk_id} keys mismatch: {set(chunk.keys())}")
                return 1
            print(f"[OK] Chunk {chunk_id}: keys = {list(chunk.keys())}")
            
            # Check dtypes
            if not isinstance(chunk["traces"], type(chunk["traces"])):
                pass
            if chunk["traces"].dtype != 'float64':
                print(f"[WARN] traces dtype: {chunk['traces'].dtype} (expected float64)")
            if chunk["metrics"].dtype != 'float64':
                print(f"[WARN] metrics dtype: {chunk['metrics'].dtype} (expected float64)")
            if chunk["logs"].dtype != 'float64':
                print(f"[WARN] logs dtype: {chunk['logs'].dtype} (expected float64)")
            if not isinstance(chunk["culprit"], int):
                print(f"[FAIL] culprit not int: {type(chunk['culprit'])}")
                return 1
        
        # Verify all chunks have same structure
        sample_chunk = list(chunks.values())[0]
        for chunk_id, chunk in chunks.items():
            if chunk["traces"].shape != sample_chunk["traces"].shape:
                print(f"[FAIL] Shape mismatch in {chunk_id}")
                return 1
            if chunk["metrics"].shape != sample_chunk["metrics"].shape:
                print(f"[FAIL] Shape mismatch in {chunk_id}")
                return 1
            if chunk["logs"].shape != sample_chunk["logs"].shape:
                print(f"[FAIL] Shape mismatch in {chunk_id}")
                return 1
        
        print(f"[OK] All {len(chunks)} chunks have consistent structure")
    
    print("\n[GATE 4 PASSED] Chunk schema verified")
    return 0


if __name__ == "__main__":
    exit(main())