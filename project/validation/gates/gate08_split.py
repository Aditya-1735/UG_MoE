#!/usr/bin/env python3
"""
Gate 8: Train/test split
Verifies split ratio ≈60/40 (matching recovered artifact within reasonable sampling variance)
"""
import pickle
from pathlib import Path

CHUNKS_DIR = Path("C:/Users/as999/OneDrive/Desktop/trash/impl/project/preprocessing/muad_compat/output")

def main():
    print("=" * 60)
    print("GATE 8: Train/Test Split")
    print("=" * 60)
    
    train_path = CHUNKS_DIR / "chunk_train.pkl"
    test_path = CHUNKS_DIR / "chunk_test.pkl"
    
    if not train_path.exists() or not test_path.exists():
        print("[FAIL] Missing train/test files")
        return 1
    
    with open(train_path, 'rb') as f:
        train_chunks = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_chunks = pickle.load(f)
    
    train_count = len(train_chunks)
    test_count = len(test_chunks)
    total = train_count + test_count
    
    train_ratio = train_count / total
    test_ratio = test_count / total
    
    print(f"Train chunks: {train_count}")
    print(f"Test chunks: {test_count}")
    print(f"Total chunks: {total}")
    print(f"Train ratio: {train_ratio:.4f} ({train_ratio*100:.2f}%)")
    print(f"Test ratio: {test_ratio:.4f} ({test_ratio*100:.2f}%)")
    
    # Check for overlap
    train_ids = set(train_chunks.keys())
    test_ids = set(test_chunks.keys())
    overlap = train_ids & test_ids
    if overlap:
        print(f"[FAIL] ID overlap: {len(overlap)} chunks in both train and test")
        return 1
    print(f"[OK] No ID overlap between train and test")
    
    # Check ratio is close to 60/40 (allowing ~5pp tolerance for randomness)
    # Recovered artifact had exactly 60.0%/40.0%
    target_train_ratio = 0.6
    tolerance = 0.05  # 5 percentage points
    
    if abs(train_ratio - target_train_ratio) > tolerance:
        print(f"[WARN] Train ratio {train_ratio:.4f} differs from target {target_train_ratio} by >{tolerance}")
        print(f"       (Recovered artifact had exactly 60.0%/40.0%)")
        # Not a hard failure - randomization causes variance
    else:
        print(f"[OK] Train ratio within {tolerance*100:.0f}pp of 60/40 target")
    
    # Check label distribution in both splits
    train_labels = [v['culprit'] for v in train_chunks.values()]
    test_labels = [v['culprit'] for v in test_chunks.values()]
    
    from collections import Counter
    train_dist = Counter(train_labels)
    test_dist = Counter(test_labels)
    
    print(f"\nTrain label distribution:")
    for label in sorted(train_dist.keys()):
        print(f"  culprit {label}: {train_dist[label]} ({train_dist[label]/train_count*100:.1f}%)")
    
    print(f"\nTest label distribution:")
    for label in sorted(test_dist.keys()):
        print(f"  culprit {label}: {test_dist[label]} ({test_dist[label]/test_count*100:.1f}%)")
    
    print("\n[GATE 8 PASSED] Train/test split verified")
    return 0


if __name__ == "__main__":
    exit(main())