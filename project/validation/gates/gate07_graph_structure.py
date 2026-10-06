#!/usr/bin/env python3
"""
Gate 7: Graph structure verification
Verifies node_num=12, edges match the documented 24-edge list.
"""
import json
from pathlib import Path

# Expected edges from forensic doc / recovered metadata.json
EXPECTED_EDGES_SRC = [1,1,1,1,1,1,1,1,10,10,10,2,0,0,7,7,7,5,3,11,11,11,11,11]
EXPECTED_EDGES_DST = [1,10,6,2,7,8,5,3,10,2,0,2,0,5,7,4,9,5,3,1,10,11,0,5]

def main():
    print("=" * 60)
    print("GATE 7: Graph Structure Verification")
    print("=" * 60)
    
    # Check recovered metadata.json
    metadata_file = Path("C:/Users/as999/OneDrive/Desktop/trash/impl/project/recovered_artifact/metadata.json")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # node_num
    if metadata.get("node_num") != 12:
        print(f"[FAIL] node_num: expected 12, got {metadata.get('node_num')}")
        return 1
    print(f"[OK] node_num: 12")
    
    # edges
    edges = metadata.get("edges", [])
    if len(edges) != 2:
        print(f"[FAIL] edges format invalid")
        return 1
    
    src, dst = edges[0], edges[1]
    if src != EXPECTED_EDGES_SRC:
        print(f"[FAIL] Edge sources mismatch")
        print(f"  Expected: {EXPECTED_EDGES_SRC}")
        print(f"  Got:      {src}")
        return 1
    if dst != EXPECTED_EDGES_DST:
        print(f"[FAIL] Edge destinations mismatch")
        print(f"  Expected: {EXPECTED_EDGES_DST}")
        print(f"  Got:      {dst}")
        return 1
    print(f"[OK] Edges match exactly (24 directed edges)")
    
    # Check for self-loops
    self_loops = [(s, d) for s, d in zip(src, dst) if s == d]
    print(f"[INFO] Self-loops found: {self_loops}")
    
    # metric_num
    if metadata.get("metric_num") != 7:
        print(f"[FAIL] metric_num: expected 7, got {metadata.get('metric_num')}")
        return 1
    print(f"[OK] metric_num: 7")
    
    # chunk_lenth
    if metadata.get("chunk_lenth") != 10:
        print(f"[FAIL] chunk_lenth: expected 10, got {metadata.get('chunk_lenth')}")
        return 1
    print(f"[OK] chunk_lenth: 10")
    
    print("\n[GATE 7 PASSED] Graph structure verified")
    return 0


if __name__ == "__main__":
    exit(main())