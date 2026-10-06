#!/usr/bin/env python3
"""
RECONSTRUCTED COMPONENT - NOT ORIGINAL SOURCE CODE
MUAD compatibility trim: traces[..., :1] to drop the always-zero second channel.
Input: Eadro-generated chunks with traces shape (12, 10, 2)
Output: MUAD-compatible chunks with traces shape (12, 10, 1)
This is an undocumented transformation required because:
- Eadro's deal_traces allocates 2 channels but only populates channel 0
- MUAD's TraceEncoder expects in_size=1 (GRUEncoder(in_size=1, out_dim=64))
"""
import os
import pickle
import shutil
from pathlib import Path

CHUNKS_DIR = Path("C:/Users/as999/OneDrive/Desktop/trash/impl/project/vendor/eadro/codes/chunks/SN")
OUTPUT_DIR = Path("C:/Users/as999/OneDrive/Desktop/trash/impl/project/preprocessing/muad_compat/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def trim_traces_in_chunks(input_path: Path, output_path: Path) -> bool:
    """Load chunks.pkl, trim traces channel, save to output."""
    try:
        with open(input_path, 'rb') as f:
            chunks = pickle.load(f)
    except Exception as e:
        print(f"  [ERROR] Failed to load {input_path}: {e}")
        return False
    
    trimmed_count = 0
    for chunk_id, chunk_data in chunks.items():
        if 'traces' in chunk_data:
            # traces is directly a numpy array [12, 10, 2]
            latency = chunk_data['traces']
            if latency.ndim == 3 and latency.shape[-1] == 2:
                chunk_data['traces'] = latency[..., :1]
                trimmed_count += 1
    
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(chunks, f)
        print(f"  [OK] Trimmed {trimmed_count} chunks in {input_path.name}")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to save {output_path}: {e}")
        return False


def main():
    print("=" * 60)
    print("RECONSTRUCTED: trim_trace_channel.py")
    print("Eadro chunks (traces[...,2]) -> MUAD compat (traces[...,1])")
    print("=" * 60)
    
    # Process the combined chunks.pkl
    chunks_pkl = CHUNKS_DIR / "chunks.pkl"
    if not chunks_pkl.exists():
        print(f"[ERROR] No chunks.pkl found in {CHUNKS_DIR}")
        print("Run Eadro preprocessing first to generate chunks.pkl")
        return 1
    
    output_pkl = OUTPUT_DIR / "chunks.pkl"
    if not trim_traces_in_chunks(chunks_pkl, output_pkl):
        return 1
    
    # Also copy metadata.json
    metadata_src = CHUNKS_DIR / "metadata.json"
    if metadata_src.exists():
        metadata_dst = OUTPUT_DIR / "metadata.json"
        shutil.copy2(metadata_src, metadata_dst)
        print(f"  [OK] Copied metadata.json")
    
    # Also copy train/test splits
    for split_name in ["chunk_train.pkl", "chunk_test.pkl"]:
        split_src = CHUNKS_DIR / split_name
        if split_src.exists():
            split_dst = OUTPUT_DIR / split_name
            # Need to trim these too
            trim_traces_in_chunks(split_src, split_dst)
            print(f"  [OK] Trimmed and copied {split_name}")
    
    print("\n" + "=" * 60)
    print("MUAD COMPATIBILITY TRIM COMPLETED")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())