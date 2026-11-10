# Data Directory

## Overview
This directory contains all datasets used by the project. Code and data are
separated so experiments can switch datasets without modifying code.

## Datasets

### 1. `original/` (recommended for Track A vs Track B comparison) — RENAMED to `recovered_artifact/`
- Source: MUAD paper original dataset (recovered from commit 6031ccf)
- Location: `project/recovered_artifact/`
- Files: `chunk_train.pkl`, `chunk_test.pkl`, `metadata.json`
- `event_num`: 15, `node_num`: 12, `train chunks`: 5,610, `test chunks`: 2,244
- Used by: Track A (`run_track_a_full.py`), Track B (`run_track_b_full.py`), 
  `run_comparison.py`, `run_ablations.py`
- Git commit: `c0aef10` (repo-initialize)

### 2. `regenerated/` (muad_compat output) — NOT recommended for comparison
- Source: Regenerated preprocessing pipeline
- Location: `project/preprocessing/muad_compat/output/`
- Files: `chunk_train.pkl`, `chunk_test.pkl`, `metadata.json`
- `event_num`: 14, `node_num`: 12, `train chunks`: 9,907
- Used by: Older versions of Track B and ablation scripts
- Git commit: `97e33a0` (add: preprocessing)

### 3. `vendor/` (vendor/muad/Dataset/chunk/)
- Copy of `original/` used by vendored MUAD baseline
- Location: `project/vendor/muad/Dataset/chunk/`
- Identical to `original/` byte-for-byte

## IMPORTANT: Do Not Mix Datasets
Track A and Track B MUST use the same dataset for a valid comparison.
- Use `recovered_artifact/` (event_num=15) for both models — this ensures apples-to-apples comparison
- `muad_compat/output/` has different event_num (14) — mixing invalidates comparison

## How to Switch Datasets
Edit the experiment script or pass `--data-dir`:
```bash
# Use recovered_artifact (recommended)
python experiments/run_track_b_full.py

# Or explicitly specify
python experiments/run_track_b_full.py --data-dir /path/to/recovered_artifact --event-num 15
```

## Data Generation Pipelines
- `project/preprocessing/raw_to_parsed/` — Convert raw logs/metrics/traces to parsed format (Drain3)
- `project/preprocessing/hdfs/` — HDFS log-specific preprocessing
- `project/preprocessing/sn/` — SN (Eadro) dataset preprocessing
- `project/preprocessing/muad_compat/` — MUAD-compatible data regeneration
