# Data Directory

## Overview
This directory contains all datasets used by the project. Code and data are
separated so experiments can switch datasets without modifying code.

## Datasets

### 1. `original/` (recommended for Track A vs Track B comparison) — RENAMED to `recovered_artifact/`
- Source: MUAD paper original dataset (recovered from commit 6031ccf)
- Location: `project/recovered_artifact/`
- Files: `chunk_train.pkl`, `chunk_test.pkl`, `metadata.json`
- `event_num`: 15, `node_num`: 12, train chunks: 3,366 (5,610 = train + test), test chunks: 2,244
- Labels: `culprit` ∈ {0, 1, 4, 5}; 0 = normal, ≥1 = anomalous (MUAD convention); test is ~83% anomalous
- Used by: every script built on `experiments/common.py` (default `--data-dir`), which carves a
  stratified 15% validation split out of `chunk_train.pkl`
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
Every experiment script takes `--data-dir` (default: `recovered_artifact/`). Model dimensions
(`event_num`, `metric_num`, `node_num`) are read from the directory's `metadata.json`:
```bash
python experiments/run_comparison.py                                   # recovered_artifact (recommended)
python experiments/run_comparison.py --data-dir /path/to/other/chunks  # any dir with chunk_{train,test}.pkl + metadata.json
```

## Data Generation Pipelines
- `project/preprocessing/raw_to_parsed/` — Convert raw logs/metrics/traces to parsed format (Drain3)
- `project/preprocessing/sn/` — SN (Eadro) dataset preprocessing
- `project/preprocessing/muad_compat/` — MUAD-compatible data regeneration
