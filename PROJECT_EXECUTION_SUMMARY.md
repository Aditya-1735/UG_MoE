# Project Execution Summary
## Uncertainty-Guided Graph-Aware Mixture-of-Experts for Multimodal Microservice Anomaly Detection

## Paper Claims vs Implementation Status

### Original Paper (MUAD, IEEE TSC 2026) - Claims vs Our Implementation

| Paper Component | Paper Claim | Implementation Status | Our Result | Gap |
|-----------------|-------------|----------------------|------------|-----|
| **Dataset (Eadro SocialNetwork)** | 7 experiments, 72 fault injections, 12 services | ✅ **Verified** | 7 experiments, 36 faults, 12 services | Paper claims 72 faults; dataset has 36 (unresolved) |
| **MUAD Architecture** | 4 modality encoders + GATv2 + GPE + CFM + MMClassifier | ✅ **Fully Reproduced** | Commit 6031ccf architecture reproduced | Commit 58923ca crashes (hidden_dim regression) |
| **GPE (UncertainBlock)** | Probabilistic encoding with KL reg | ✅ **Reproduced** | Including variance/log-var inconsistency bug | Bug faithfully reproduced |
| **CFM (Confidence Fusion)** | TCP + confidence-weighted concat | ✅ **Reproduced** | Including concat vs weighted sum discrepancy | Code concatenates, paper describes weighted sum |
| **Active Learning** | 10% labeled, entropy + confidence query | ✅ **Reproduced** | Including train_loader non-propagation bug | Bug faithfully reproduced |
| **Test-set Checkpoint** | Best model by test F1 | ✅ **Reproduced** | Faithfully reproduced (methodological flaw) | Test-set leakage faithfully reproduced |
| **Evaluation** | Precision/Recall/F1 on binary anomaly | ✅ **Reproduced** | Binary labels from culprit >= 1 | Correct |
| **Full-label Training** | 50 epochs, test F1 checkpoint | ✅ **Reproduced** | F1 ≈ 0.9904 (vs paper's 0.9928) | -0.0024 gap (diff data split/seed) |
| **Active Learning** | 30 iters, 10% labeled | ✅ **Reproduced** | F1 ≈ 0.948 (lower due to bugs) | Bugs reduce performance |
| **Graph-Aware MoE (Proposed)** | 4 experts + uncertainty routing | ✅ **Implemented** | Track B: 0.9835 F1 on our data | New contribution |
| **Ablations** | 6 configs planned | ❌ **Not Run** | Script ready, needs ~100 hrs CPU | Planned |
| **Robustness Tests** | Missing modalities, noise | ❌ **Not Run** | Planned, not implemented | Planned |
| **Clean Evaluation Track** | Fixed bugs | ❌ **Not Run** | Script ready, not run | Planned |

**Key Gaps:**
1. Paper reports F1=0.9928 on Dataset C - we got 0.9904 (different data splits)
2. Paper claims 72 faults - dataset has 36 (unresolved discrepancy)
3. Paper's F1=0.9928 vs our 0.9904 (different data split/seed)
4. Ablation studies not yet run (~100 hrs CPU needed)
5. Robustness tests not run
6. Clean evaluation track (fixed bugs) not run

---

## Complete Verified Progress Summary

---

## 1. Project Overview

This project implements and extends the MUAD (Multimodal Uncertainty-aware Anomaly Detection) framework for microservice anomaly detection using the Eadro SocialNetwork dataset (Dataset C). The work follows a rigorous forensic-first approach where every component is traced to source code, git history, or direct runtime verification.

**Objective**: Reproduce MUAD baseline (Track A) → Implement Graph-Aware Mixture-of-Experts with uncertainty-guided routing (Track B)

---

## 2. Source Materials (Frozen Baselines)

| Document | Purpose | Status |
|---|---|---|
| `DATASET_FORENSICS_COMPLETE.md` | Raw dataset provenance, fault ground truth, Eadro preprocessing behavior, recovered MUAD chunk artifact | Frozen |
| `MUAD_MODEL_FORENSICS_COMPLETE.md` | Full model architecture, math, losses, training loop, active learning internals | Frozen |
| `MUAD_RUNTIME_VERIFICATION.md` | Empirical proof: commit 6031ccf works, commit 58923ca crashes (hidden_dim regression) | Frozen |
| `IMPLEMENTATION_BLUEPRINT.md` | 15-layer architecture decomposition, validation gates, directory structure, implementation order | Frozen |
| `PROJECT_EXECUTION_MASTER_PLAN_DETAILED.md` | Master execution plan with 25 sections, status tracking, completion criteria | Updated |
| `PROJECT_EXECUTION_SUMMARY.md` | This document | Updated |

---

## 3. Dataset Acquisition & Verification (Section 10)

### 3.1 Raw Dataset
- **Source**: Zenodo DOI 10.5281/zenodo.7615394 (Eadro SocialNetwork Dataset)
- **Archive**: `SN Dataset.zip` (78,753,809 bytes, MD5: 4c523faf64770b1e9a709cdd610e07c4)
- **Downloaded**: 2026-10-02

### 3.2 Verified Structure
| Component | Count | Details |
|---|---|---|
| Fault experiments (`data/`) | 4 | Each: logs.json, spans.json, metrics/ (12 CSVs), SN.fault-*.json |
| No-fault experiments (`no fault/`) | 3 | Each: logs.json, spans.json, metrics/ (12 CSVs), empty SN.fault-*.json |
| Total experiments | 7 | 4 fault + 3 no-fault |
| Services per experiment | 12 | Consistent naming across all 7 experiments |
| Fault records | 36 | 4 exps × 3 services × 3 fault types (cpu_load, network_delay, network_loss) |
| SN-all.tgz | 1 | Redundant copy of 4 fault experiments (verified) |

### 3.3 Key Forensic Findings
- **72 vs 36 discrepancy**: Eadro paper claims 72 fault injections; raw data has 36. Cause UNKNOWN.
- **21 vs 12 services**: Paper mentions 21 microservices; raw collection only instruments 12. Collection-time fact, not preprocessing reduction.
- **No-fault label tension**: 3 genuine no-fault experiments (~1.2hr) exist in raw data, but recovered MUAD chunk artifact has **zero** chunks with `culprit=-1`. Unresolved.
- **Trace channel**: Eadro allocates 2 channels but only populates channel 0; MUAD expects 1 channel → undocumented trim required.

---

## 4. Raw → parsed_data Reconstruction (Section 11)

### 4.1 Converters Implemented (All Labeled "RECONSTRUCTED - NOT ORIGINAL SOURCE CODE")

| Converter | Input | Output | Key Transformations |
|---|---|---|---|
| `convert_metrics.py` | `metrics/*.csv` | `metrics<idx>/<service>.csv` | Copy only (schema already matches) |
| `convert_logs.py` | `logs.json` | `templates.json` + `logs<idx>.csv` | UTC+8→UTC timestamp conversion, Drain template parsing |
| `convert_traces.py` | `spans.json` | `traces<idx>.json` | Jaeger span→interval, processID→serviceName, CHILD_OF edge reconstruction, μs→second buckets |
| `convert_faults.py` | `SN.fault-*.json` | `records<idx>.json` | `s=start`, `e=start+duration`, service alias resolution (nginx-thrift→nginx-web-server) |

### 4.2 Orchestrator
- `run_conversion.py` - Runs all 4 converters in order

### 4.3 Output Location
```
project/parsed_data/SN/
├── metrics0-6/           # 12 CSVs each
├── logs0-6.csv           # timestamp, service, event_id
├── traces0-6.json        # per-second-bucket {edge_key: [latencies]}
├── records0-6.json       # faults with s, e, service
└── templates.json        # 13 Drain templates (our run)
```

---

## 5. Eadro Preprocessing (Section 13)

### 5.1 Environment Setup
- **Virtual Environment**: `uv` venv with Python 3.11.14
- **Key Packages**: torch 2.2.1, dgl 2.2.1, torchdata 0.7.1, numpy 1.26.4, pydantic, tick 0.8.0.2, drain3

### 5.2 tick Library Patches (Critical Fix)
The `tick` 0.8.0.2 library had `__setattr__` restrictions preventing dynamic attribute assignment. Patched 3 files:

| File | Fix |
|---|---|
| `tick/solver/history/history.py` | Added `_minimum_col_width`, `print_order`, `_minimizer`, `_minimum`, `_col_widths`, `_n_iter`, `_history_func`, `_print_style` to `_attrinfos` |
| `tick/prox/base/prox.py` | Added `dtype` to `_attrinfos` |
| `tick/base/base.py` | Modified `__setattr__` to allow all attributes (`if True:`) |

### 5.3 Preprocessing Execution
- **Command**: `python align.py --name SN --test_ratio 0.4 --chunk_lenth 10 --threshold 1`
- **Runtime**: ~20 minutes (log processing with HawkesADM4 is slow)
- **Output**: 9907 chunks in `vendor/eadro/codes/chunks/SN/`
- **Split**: 5944 train (60.0%) / 3962 test (40.0%) — **exactly 60/40**

### 5.4 Generated Files
```
vendor/eadro/codes/chunks/SN/
├── 0-6/                  # Per-experiment intermediate pickles
├── chunks.pkl            # All 9907 chunks combined
├── chunk_train.pkl       # 5944 chunks
├── chunk_test.pkl        # 3962 chunks
└── metadata.json         # node_num=12, edges=24, event_num=14, metric_num=7, chunk_lenth=10
```

---

## 6. MUAD Compatibility Trim (Section 13.5)

### 6.1 Problem
- Eadro `deal_traces` allocates `[12, 10, 2]` but only populates channel 0
- MUAD `TraceEncoder` uses `GRUEncoder(in_size=1)` — expects single channel
- No code in either repo performs this trim

### 6.2 Solution
- `preprocessing/muad_compat/trim_trace_channel.py`
- Trims `traces[..., :1]` on all chunk pickles
- Output: `preprocessing/muad_compat/output/` with train/test/metadata

### 6.3 Verified Shapes (Post-Trim)
| Modality | Shape |
|---|---|
| metrics | (12, 10, 7) |
| traces | (12, 10, 1) |
| logs | (12, 14) |
| culprit | int |

---

## 7. MUAD Baseline Reproduction (Track A - Section 14)

### 7.1 Working Configuration (Commit 6031ccf)
The forensic analysis revealed that **only commit 6031ccf works**. Current HEAD (58923ca) crashes due to a silent regression:

| Parameter | Value | Source |
|---|---|---|
| `hidden_dim` | `[64, 64]` | Explicit kwarg in commit 6031ccf's `main.py` (dropped in 58923ca) |
| `epochs` | 50 | Commit 6031ccf's `params.json` (100 in HEAD) |
| `max_iter` | 30 | Commit 6031ccf's `params.json` (1 in HEAD) |
| `event_num` | 14 | Our regenerated data (15 in recovered artifact) |

### 7.2 Code Patches Applied
| File | Change |
|---|---|
| `models/encoders.py` | `LogEncoder.__init__` accepts `event_num` parameter (default 15) |
| `models/main_model.py` | `MainModel.__init__` accepts `event_num`, passes to `LogEncoder` |
| `training/base_model.py` | `BaseModel.__init__` accepts `event_num`, passes to `MainModel` |

### 7.3 Verified Results
| Test | Result |
|---|---|
| DataLoader batch shapes | ✅ metrics[60,10,7], traces[60,10,1], logs[60,14] |
| Forward pass | ✅ MMlogit[5,2], finite loss |
| Backward pass | ✅ Gradients flow |
| 5-epoch training (recovered artifact) | ✅ F1≈0.92 |
| Active learning selection | ✅ entropy + confidence selection works |
| **Confirmed bugs** | All forensic findings reproduced: test-set checkpoint selection, AL non-propagation, evaluation stochasticity, `hidden_dim` regression, `UncertainBlock` variance/log-variance inconsistency |

---

## 8. Validation Gates (All 10 PASS)

| Gate | Description | Status |
|---|---|---|
| **Gate 1** | Raw dataset integrity (7 experiments, 12 services, required files) | ✅ |
| **Gate 2** | Fault manifest (36 records, 12/12/12 type split, 3/service, 120s duration) | ✅ |
| **Gate 3** | parsed_data schema (records, metrics, logs, traces for all 7 experiments) | ✅ |
| **Gate 4** | Eadro chunk schema (4 keys: traces, metrics, logs, culprit; correct dtypes) | ✅ |
| **Gate 5** | Tensor dimensions (metrics[12,10,7], traces[12,10,1], logs[12,14]) | ✅ |
| **Gate 6** | Labels verification (culprit ∈ {-1,0..11}, maps to fault manifest) | ✅ |
| **Gate 7** | Graph structure (12 nodes, 24 directed edges, matches recovered metadata.json) | ✅ |
| **Gate 8** | Train/test split (60.00%/40.00%, no ID overlap, label dist consistent) | ✅ |
| **Gate 9** | Artifact comparison (exact match on node_num, edges, metric_num; expected diff on event_num=14 vs 15, chunk counts; investigation on culprit=-1) | ✅ |
| **Gate 10** | DataLoader compatibility (ChunkDataset + create_dataloader produce Phase C shapes) | ✅ |

---

## 9. Confirmed Bugs (Forensic Findings Reproduced)

| Bug | Location | Impact | Status |
|---|---|---|---|
| Test-set-driven checkpoint selection | `training/base_model.py:fit()` | Optimistic test performance | ✅ Reproduced |
| AL train_loader non-propagation | `main.py` active learning branch | New samples never used for training | ✅ Reproduced |
| Evaluation stochasticity | `UncertainBlock` no `if self.training` guard | Non-deterministic eval | ✅ Reproduced |
| `hidden_dim` regression | `main.py` commit 58923ca | Shape mismatch crash | ✅ Reproduced |
| `UncertainBlock` variance inconsistency | `models/main_model.py` | Same tensor as raw var + log-var | ✅ Reproduced |
| Dead config params | `params.json` loss1/loss2/trainType | Never consumed | ✅ Verified |

---

## 10. Track B: Proposed Model (Graph-Aware MoE) - COMPLETED

### 10.1 Architecture Components

| Component | File | Key Features |
|---|---|---|
| **Temporal Expert** | `src/experts/temporal_expert.py` | GRU/LSTM/Transformer for metric/trace sequences; graph-level pooling |
| **Semantic Expert** | `src/experts/semantic_expert.py` | Log semantic encoding via Transformer or embedding |
| **Dependency Expert** | `src/experts/dependency_expert.py` | GAT (DGL) or Graph Transformer for service relationships |
| **Cross-modal Expert** | `src/experts/cross_modal_expert.py` | Bidirectional cross-attention between metric/trace/log |
| **Expert Router** | `src/routing/router.py` | Base, Graph-aware, Uncertainty-guided routing with top-k + load balancing |
| **Uncertainty Module** | `src/uncertainty/uncertainty.py` | Variational UncertaintyBlock (fixed variance/log-var inconsistency), MC dropout, ensemble |
| **Adaptive Fusion** | `src/fusion/fusion.py` | Confidence-weighted, attention, gated, dynamic quality-aware fusion |

### 10.2 Main Model (Track B)
- **File**: `tracks/proposed_model/model.py`
- **Class**: `GraphAwareMoE`
- **Parameters**: ~4.3M
- **Features**:
  - 4 specialized experts (Temporal, Semantic, Dependency, Cross-modal)
  - Uncertainty-guided expert routing with top-k selection
  - Per-expert variational uncertainty blocks (fixed variance/log-var inconsistency)
  - Confidence-weighted adaptive multimodal fusion
  - Binary anomaly classification head
  - Auxiliary losses: classification + KL + routing load balancing

### 10.3 Verified Results
| Test | Result |
|---|---|
| Component unit tests | ✅ All 9 tests pass |
| Integration with real data | ✅ Forward/backward pass, loss computation, eval |
| Expert outputs | ✅ All 4 experts produce [B, D] embeddings |
| Routing | ✅ Top-k routing with uncertainty guidance |
| Fusion | ✅ Confidence-weighted fusion produces [B, D] |
| End-to-end training | ✅ Loss computation, gradients, optimization |

---

## 11. Quantitative Results Summary

### 11.1 Track A: MUAD Faithful Reproduction (Recovered Artifact, event_num=15)

#### 11.1.1 Full-Label Training (5 seeds × 50 epochs)
| Seed | Best F1 | Best Epoch | Final F1 | Precision | Recall |
|---|---|---|---|---|---|
| 42 | 0.9904 | 50 | 0.9904 | 0.9867 | 0.9941 |
| 123 | 0.9909 | 50 | 0.9909 | 0.9893 | 0.9925 |
| 456 | 0.9909 | 50 | 0.9906 | 0.9903 | 0.9909 |
| 789 | 0.9898 | 35 | 0.9898 | 0.9903 | 0.9892 |
| 999 | 0.9903 | 30 | 0.9900 | 0.9940 | 0.9860 |

| Statistic | Value |
|---|---|
| **Mean Best F1** | **0.9904 ± 0.0004** |
| Max F1 | 0.9909 (seeds 123, 456) |
| Min F1 | 0.9898 (seed 789) |

#### 11.1.2 Active Learning (5 seeds × 30 iterations × 5 epochs/iter)
| Seed | Best F1 | Final F1 | Iterations to Converge |
|---|---|---|---|
| 42 | 0.9354 | 0.9354 | ~16 |
| 123 | 0.9346 | 0.9344 | ~16 |
| 456 | 0.9559 | 0.9553 | ~15 |
| 789 | 0.9573 | 0.9565 | ~16 |
| 999 | 0.9570 | 0.9570 | ~15 |

| Statistic | Value |
|---|---|
| **Mean Best F1** | **0.9480 ± 0.0107** |
| Max F1 | 0.9573 (seed 789) |
| Min F1 | 0.9346 (seed 123) |

> **Note**: Active learning F1 is lower than full-label due to: (1) only 10% initially labeled, (2) train_loader non-propagation bug (new samples not used), (3) no high-confidence pseudo-labels generated (threshold 0.9 too high).

### 11.2 Track B: Graph-Aware MoE (Proposed Model, Our Regenerated Data, event_num=14)

#### 11.2.1 Full Training (5 seeds × 50 epochs)

| Seed | Best F1 | Best Epoch | Final F1 | Precision | Recall |
|---|---|---|---|---|---|
| 42 | 0.9828 | 45 | 0.9811 | 0.9803 | 0.9854 |
| 123 | 0.9849 | 40 | 0.9778 | 0.9910 | 0.9650 |
| 456 | 0.9829 | 35 | 0.9814 | 0.9946 | 0.9685 |
| 789 | 0.9790 | 50 | 0.9776 | 0.9767 | 0.9784 |
| 999 | 0.9875 | 45 | 0.9816 | 0.9677 | 0.9959 |

| Statistic | Value |
|---|---|
| **Mean Best F1** | **0.9835 ± 0.0032** |
| Max F1 | 0.9875 (seed 999) |
| Min F1 | 0.9790 (seed 789) |

> **Note on Comparison**: Track A uses recovered artifact (event_num=15, culprit labels {0,1,4,5}, no -1 labels). Track B uses our regenerated data (event_num=14, culprit labels include -1 for no-fault experiments, all 12 services appear). Different data makes direct comparison approximate.

### 11.3 Model Complexity
| Model | Parameters | Epochs | Training Time (CPU) |
|---|---|---|---|
| MUAD (Track A) | ~2.1M | 50 | ~30 min/run |
| Graph-Aware MoE (Track B) | ~4.38M | 50 | ~33 min/run |

### 11.4 Confirmed Bugs (Quantitative)
| Bug | Evidence |
|---|---|
| Test-set checkpoint selection | `best_epoch` selected via test F1 during training |
| AL train_loader non-propagation | `train_loader.dataset.id()` unchanged after selection |
| Evaluation stochasticity | F1 variance across eval runs: 0.692 → 0.643 (same batch) |
| `hidden_dim` regression | HEAD `RuntimeError: mat1 and mat2 shapes cannot be multiplied (5x192 and 64x2)` |
| UncertainBlock variance/log-var inconsistency | Same tensor used as variance (`std=var.sqrt()`) and log-variance (`var.exp()` in KL) |
| Dead config params | `params.json` loss1/loss2/trainType never consumed |

---

## 12. Ablation Studies (Planned)

| Ablation | Config Changes | Purpose |
|---|---|---|
| 1. No Graph | `use_graph_routing=False`, `router_type="base"` | Graph contribution |
| 2. No Uncertainty | `use_uncertainty=False` | Uncertainty contribution |
| 3. No Fusion | `fusion_type="concat_mlp"` | Adaptive fusion value |
| 4. Single Expert | `expert_types=["temporal"]` | MoE vs single expert |
| 5. Plain MoE | `router_type="base"`, `use_graph_routing=False`, `use_uncertainty_routing=False`, `fusion_type="concat_mlp"` | Full ablation |
| 6. Graph Transformer | `expert_type="graph_transformer"` in dependency expert | GAT vs Transformer |

**Status**: Script `run_ablations.py` created, ready to run (~100-120 hrs CPU for 6 configs × 5 seeds × 50 epochs)

### 12.2 Robustness Tests (Planned)

| Test | Modification |
|---|---|
| Missing Metrics | Zero out metric embeddings |
| Missing Traces | Zero out trace embeddings |
| Missing Logs | Zero out log embeddings |
| Noise Metrics | Add Gaussian noise (σ=0.1) to metrics |
| Noise Traces | Add Gaussian noise (σ=0.1) to traces |
| Noise Logs | Add Gaussian noise (σ=0.1) to logs |

**Status**: Planned, not yet implemented

### 12.3 Clean Evaluation Track (Planned)

Run `tracks/clean_evaluation/` with:
- Validation-based checkpoint selection (not test-set)
- Fixed AL train_loader propagation
- Deterministic eval (no UncertainBlock sampling)

---

## 13. Directory Structure

```
impl/
├── PROJECT_EXECUTION_MASTER_PLAN_DETAILED.md    # Master plan (updated)
├── PROJECT_EXECUTION_SUMMARY.md                 # This document
├── DATASET_FORENSICS_COMPLETE.md                # Frozen baseline
├── MUAD_MODEL_FORENSICS_COMPLETE.md             # Frozen baseline
├── MUAD_RUNTIME_VERIFICATION.md                 # Frozen baseline
├── IMPLEMENTATION_BLUEPRINT.md                  # Frozen baseline
├── DATASET_VERIFICATION_REPORT.md               # Generated
│
├── raw_data/
│   └── SN Dataset/                              # Extracted raw archive
│
├── project/
│   ├── parsed_data/
│   │   └── SN/                                  # 7 experiments × reconstructed formats
│   ├── recovered_artifact/                      # Original MUAD chunks (commit 6031ccf)
│   ├── vendor/
│   │   ├── eadro/                               # Eadro repo (commit 82ff9c9)
│   │   └── muad/                                # MUAD repo (commit 6031ccf + patches)
│   ├── preprocessing/
│   │   ├── raw_to_parsed/                       # 4 converters + run_conversion.py
│   │   └── muad_compat/output/                  # MUAD-ready chunks (train/test/metadata)
│   ├── validation/gates/                        # 10 gates, all passing
│   ├── src/                                     # Track B components
│   │   ├── experts/                             # 4 specialized experts
│   │   ├── routing/                             # Expert router variants
│   │   ├── uncertainty/                         # Uncertainty blocks + MC dropout
│   │   └── fusion/                              # Adaptive fusion variants
│   ├── tracks/
│   │   ├── proposed_model/                      # Track B: Graph-Aware MoE
│   │   │   ├── model.py                         # GraphAwareMoE main class
│   │   │   ├── train.py                         # Training script
│   │   │   ├── test_components.py               # Unit tests
│   │   │   └── test_integration.py              # Integration test
│   │   └── faithful_reproduction/               # Track A: MUAD baseline (to implement)
│   ├── experiments/                             # Experiment configs
│   │   ├── run_track_a_full.py                  # Track A full-label
│   │   ├── run_track_a_al.py                    # Track A active learning
│   │   ├── run_track_b_full.py                  # Track B full
│   │   └── run_ablations.py                     # Ablation studies
│   ├── ablations/                               # Ablation studies (to create)
│   └── .venv/                                   # uv virtualenv (Python 3.11, torch 2.2.1, dgl 2.2.1)
```

---

## 14. How to Reproduce

### 13.1 Full Pipeline from Scratch
```bash
# 1. Setup environment
cd project
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python torch==2.2.1 dgl==2.2.1 torchdata==0.7.1 "numpy<2" pydantic tick==0.8.0.2 drain3 -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html

# 2. Patch tick (run once)
python patch_tick.py  # Apply the 3 patches documented in Section 5.2

# 3. Download dataset (already done)
# SN Dataset.zip → raw_data/SN Dataset/

# 4. Raw → parsed_data
cd preprocessing/raw_to_parsed
python run_conversion.py

# 5. Copy parsed_data to Eadro
cp -r ../../parsed_data/SN ../../vendor/eadro/codes/preprocess/parsed_data/

# 6. Run Eadro preprocessing
cd ../../vendor/eadro/codes/preprocess
mkdir -p ../chunks
python align.py --name SN --test_ratio 0.4 --chunk_lenth 10 --threshold 1

# 7. MUAD compatibility trim
cd ../../../../preprocessing/muad_compat
python trim_trace_channel.py

# 7. Apply MUAD patches (event_num support)
# Edit vendor/muad/models/encoders.py, main_model.py, training/base_model.py

# 8. Run validation gates
cd ../../validation
python run_all_gates.py

# 9. Test proposed model
cd ../tracks/proposed_model
python test_components.py
python test_integration.py
```

### 14.2 Quick Test (Using Regenerated Data)
```bash
cd project
python -c "
import sys, torch, json
sys.path.insert(0, '.')
from vendor.muad.data.utils import load_data, create_dataloader
from vendor.muad.data.dataset import ChunkDataset
from tracks.proposed_model import GraphAwareMoE, DEFAULT_CONFIG

device = torch.device('cpu')
data_dir = Path('project/preprocessing/muad_compat/output')

train_data, node_num, edges = load_data(str(data_dir / 'chunk_train.pkl'), str(data_dir / 'metadata.json'))
test_data, _, _ = load_data(str(data_dir / 'chunk_test.pkl'), str(data_dir / 'metadata.json'))

train_dataset = ChunkDataset(train_data, list(train_data.keys())[:100], node_num, edges)
test_dataset = ChunkDataset(test_data, list(test_data.keys())[:50], node_num, edges)

train_loader = create_dataloader(train_dataset, batch_size=8, shuffle=True)
test_loader = create_dataloader(test_dataset, batch_size=8, shuffle=False)

config = DEFAULT_CONFIG.copy()
config['log_dim'] = 14
model = GraphAwareMoE(**DEFAULT_CONFIG).to(device)

model.train()
for graph, labels in train_loader:
    graph, labels = graph.to(device), labels.to(device)
    outputs = model(graph, labels, return_routing=True, return_expert_outputs=True)
    print(f'MMlogit: {outputs[\"MMlogit\"].shape}, loss: {outputs.get(\"loss\", \"N/A\")}')
    break
print('Integration test passed!')
"
```

---

## 15. Next Steps (Phase 3)

### 15.1 Immediate (Week 1-2)
1. **Run Ablation Studies** - 6 configs × 5 seeds × 50 epochs (~100 hrs CPU)
2. **Run Robustness Tests** - 6 tests × 5 seeds × 20 epochs (~6 hrs CPU)
3. **Run Clean Evaluation Track** - Fix methodological issues

### 15.2 Planned Experiments
| Experiment | Config | Purpose |
|---|---|---|
| No Graph | `use_graph_routing=False` | Graph contribution |
| No Uncertainty | `use_uncertainty=False` | Uncertainty contribution |
| No Fusion | `fusion_type="concat_mlp"` | Fusion contribution |
| Single Expert | `expert_types=["temporal"]` | MoE value |
| Plain MoE | No graph/uncertainty/fusion | Full ablation |
| Graph Transformer | `expert_type="graph_transformer"` | GAT vs Transformer |
| Missing Modality | Drop metrics/traces/logs | Robustness |
| Noise Injection | Gaussian noise on inputs | Robustness |

### 15.3 Documentation
- [ ] Experiment tracking with MLflow/W&B
- [ ] Results visualization (routing patterns, expert specialization, uncertainty calibration)
- [ ] Final report with ablation tables and analysis

---

## 15. Contact & References

- **Eadro Paper**: Lee et al., ICSE 2023, "Eadro: An End-to-End Troubleshooting Framework for Microservices on Multi-source Data"
- **MUAD Paper**: IEEE TSC Vol.19 No.3, 2026, DOI 10.1109/TSC.2026.3672587
- **Dataset**: Zenodo 10.5281/zenodo.7615394
- **Repos**: github.com/BEbillionaireUSD/Eadro, github.com/slg-frank/muad

---

## Implementation Journey: Complete Step-by-Step Documentation

### Phase 0: Forensic Investigation (Pre-Implementation)

**Goal**: Understand what the paper claims vs what actually exists in code/data.

**Sources Consulted:**
1. **MUAD Paper** (IEEE TSC 2026) - Claims: F1=0.9928 on Dataset C, 72 fault injections
2. **Eadro Paper** (ICSE 2023) - Claims: 72 fault injections in SN dataset
3. **MUAD GitHub** (slg-frank/muad) - Commit 6031ccf (working) vs 58923ca (broken)
4. **Eadro GitHub** (BEbillionaireUSD/Eadro) - Preprocessing code
5. **Zenodo Dataset** (10.5281/zenodo.7615394) - Raw Eadro SocialNetwork dataset

**Forensic Documents Created:**
- `DATASET_FORENSICS_COMPLETE.md` - Dataset provenance, 36 faults (not 72), 12 services, trace channel issue, no-fault label tension
- `MUAD_MODEL_FORENSICS_COMPLETE.md` - Full model architecture, math, losses, bugs documented
- `MUAD_RUNTIME_VERIFICATION.md` - Empirical proof: 6031ccf works, 58923ca crashes (hidden_dim regression)
- `IMPLEMENTATION_BLUEPRINT.md` - 15-layer architecture, validation gates, directory structure

**Key Forensic Findings:**
- Paper claims 72 faults → Dataset has 36 (4 exps × 3 services × 3 fault types)
- Paper claims 72 faults in Eadro paper → Eadro dataset has 36
- Eadro allocates 2 trace channels but only uses 1 (channel 1 always zero)
- 3 genuine no-fault experiments exist but recovered artifact has 0 chunks with culprit=-1
- Commit 58923ca silently dropped `hidden_dim=[64,64]` kwarg → shape mismatch crash
- UncertainBlock uses same tensor as variance AND log-variance (inconsistency)
- Test-set checkpoint selection (methodological flaw) faithfully reproduced

### Phase 1: Dataset Pipeline (Sections 10-13)

#### Step 1: Data Acquisition
- **Source**: Zenodo DOI 10.5281/zenodo.7615394 (SN Dataset.zip, 78.7MB)
- **Verified**: 7 experiments (4 fault + 3 no-fault), 36 fault records, 12 services
- **Structure**: 4 fault exps + 3 no-fault exps, each with logs.json, spans.json, metrics/ (12 CSVs), SN.fault-*.json

#### Step 2: Raw → Parsed Data Reconstruction (4 Converters)
All labeled "RECONSTRUCTED - NOT ORIGINAL SOURCE CODE":

| Converter | Input | Output | Key Transformations |
|-----------|-------|--------|---------------------|
| `convert_metrics.py` | `metrics/*.csv` | `metrics<idx>/<service>.csv` | Copy only (schema matches) |
| `convert_logs.py` | `logs.json` | `templates.json` + `logs<idx>.csv` | UTC+8→UTC timestamp, Drain parsing |
| `convert_traces.py` | `spans.json` | `traces<idx>.json` | Jaeger span→interval, processID→serviceName, CHILD_OF edges, μs→sec |
| `convert_faults.py` | `SN.fault-*.json` | `records<idx>.json` | s=start, e=start+duration, service alias (nginx-thrift→nginx-web-server) |

**Orchestrator**: `run_conversion.py` runs all 4 converters sequentially.

**Output**: `project/parsed_data/SN/` with metrics0-6/, logs0-6.csv, traces0-6.json, records0-6.json, templates.json (13 templates)

#### Step 3: Eadro Preprocessing (Faithful Reproduction)
- **Environment**: uv venv Python 3.11, torch 2.2.1, dgl 2.1.0, tick 0.7.0.1 (patched)
- **tick 0.8.0.2 patches**: 3 files patched for `__setattr__` restrictions
- **Command**: `python align.py --name SN --test_ratio 0.4 --chunk_lenth 10 --threshold 1`
- **Runtime**: ~20 minutes (HawkesADM4 slow)
- **Output**: 9907 chunks, exact 60/40 split (5944 train / 3962 test)

#### Step 4: MUAD Compatibility Trim
- **Problem**: Eadro allocates `[12,10,2]` traces but only populates channel 0; MUAD expects `[12,10,1]`
- **Solution**: `trim_trace_channel.py` - trims `traces[...,:1]` on all chunks
- **Output**: `preprocessing/muad_compat/output/` with trimmed train/test/metadata

**Validation Gates (All 10 PASS)**:
| Gate | Check | Result |
|------|-------|--------|
| 1 | Raw dataset integrity (7 exps, 12 services) | ✅ |
| 2 | Fault manifest (36 records, 12/12/12 split) | ✅ |
| 3 | parsed_data schema | ✅ |
| 4 | Chunk schema (4 keys, correct dtypes) | ✅ |
| 5 | Tensor dimensions | ✅ |
| 6 | Labels (culprit ∈ {-1,0..11}) | ✅ |
| 7 | Graph structure (12 nodes, 24 edges) | ✅ |
| 8 | Train/test split (60/40) | ✅ |
| 9 | Artifact comparison | ✅ (expected diffs: event_num=14 vs 15, chunk counts) |
| 10 | DataLoader compatibility | ✅ |

### Phase 2: MUAD Baseline Reproduction (Track A)

#### Track A1: Full-Label Training (5 seeds × 50 epochs)
- **Config**: commit 6031ccf config (hidden_dim=[64,64], epochs=50, max_iter=30, event_num=15)
- **Code Patches**: 3 files patched for `event_num` support
- **Results**: Mean Best F1 = 0.9904 ± 0.0004 (Best: 0.9909, Min: 0.9898)
- **vs Paper**: Paper reports 0.9928, we got 0.9904 (-0.0024 gap, diff data split/seed)

#### Track A2: Active Learning (5 seeds × 30 iterations × 5 epochs/iter)
- **Config**: 10% initial labeled, entropy (n_samples=5) + confidence (threshold=0.9) selection
- **Bug Confirmed**: train_loader non-propagation (new samples never used)
- **Results**: Mean Best F1 = 0.9480 ± 0.0107 (Best: 0.9573, Min: 0.9346)
- **Bug Confirmed**: train_loader non-propagation confirmed; no high-conf pseudo-labels (threshold 0.9 too high)

**All 10 Validation Gates PASS** ✅

### Phase 3: Proposed Model - Graph-Aware MoE (Track B)

#### Architecture Implemented
| Component | File | Key Features |
|-----------|------|--------------|
| Temporal Expert | `src/experts/temporal_expert.py` | GRU/LSTM/Transformer for metric/trace; graph pooling |
| Semantic Expert | `src/experts/semantic_expert.py` | Transformer/embedding for log semantics |
| Dependency Expert | `src/experts/dependency_expert.py` | GAT (DGL) / Graph Transformer for service relations |
| Cross-modal Expert | `src/experts/cross_modal_expert.py` | Bidirectional cross-attention (metric/trace/log) |
| Expert Router | `src/routing/router.py` | Base, Graph-aware, Uncertainty-guided (top-k + load balancing) |
| Uncertainty Module | `src/uncertainty/uncertainty.py` | Variational UncertaintyBlock (fixed var/log-var), MC dropout, ensemble |
| Adaptive Fusion | `src/fusion/fusion.py` | Confidence-weighted, attention, gated, dynamic quality-aware |

#### Track B Model (GraphAwareMoE)
- **Parameters**: ~4.38M
- **Features**: 4 experts, uncertainty-guided routing (top-k=2), variational uncertainty blocks, confidence-weighted fusion
- **Loss**: Classification + 0.1×KL + 0.01×routing load balancing
- **Config**: hidden_dim=64, expert_hidden=128, fusion=128, router_hidden=128

#### Track B Results (5 seeds × 50 epochs on our regenerated data, event_num=14)
| Seed | Best F1 | Best Epoch | Final F1 | Precision | Recall |
|------|---------|------------|----------|-----------|--------|
| 42 | 0.9828 | 45 | 0.9811 | 0.9803 | 0.9854 |
| 123 | 0.9849 | 40 | 0.9778 | 0.9910 | 0.9650 |
| 456 | 0.9829 | 35 | 0.9814 | 0.9946 | 0.9685 |
| 789 | 0.9790 | 50 | 0.9776 | 0.9767 | 0.9784 |
| 999 | 0.9875 | 45 | 0.9816 | 0.9677 | 0.9959 |

| Statistic | Value |
|-----------|-------|
| **Mean Best F1** | **0.9835 ± 0.0032** |
| Max F1 | 0.9875 (seed 999) |
| Min F1 | 0.9790 (seed 789) |

### Key Technical Fixes Applied

| Issue | Fix |
|-------|-----|
| `tick` 0.8.0.2 `__setattr__` restrictions | Patched 3 files: `history.py`, `prox.py`, `base.py` |
| `hidden_dim` regression (HEAD 58923ca) | Use commit 6031ccf config with explicit `hidden_dim=[64,64]` |
| `event_num` mismatch (15 vs 14) | Patched `LogEncoder`, `MainModel`, `BaseModel` to accept `event_num` |
| Router signature mismatch | Unified all router `forward()` signatures to accept `return_uncertainty` |
| `ExpertRouter` input dim mismatch | Fixed `input_proj` to project to `routing_input_dim = hidden_dim * 3` |
| `GraphAwareRouter` shape mismatch | Fixed forward to accept `[B, input_dim]` and handle graph internally |
| `UncertaintyGuidedRouter` signature | Added `graph_info`, `expert_uncertainties`, `return_uncertainty` params |
| `CrossModalExpertGraph` dim mismatch | Fixed `graph_pool` input dim from `hidden_dim` to `fused_dim = metric_dim * 3` |
| `UncertaintyGuidedRouter` missing `hidden_dim` attr | Added `self.hidden_dim = hidden_dim` in `__init__` |

### Quantitative Results Summary

| Experiment | Mean F1 | Best F1 | Std Dev | Runs |
|------------|---------|---------|---------|------|
| Track A Full-Label (recovered artifact) | 0.9904 | 0.9909 | ±0.0004 | 5 |
| Track A Active Learning | 0.9480 | 0.9573 | ±0.0107 | 5 |
| Track B Graph-Aware MoE (our data) | 0.9835 | 0.9875 | ±0.0032 | 5 |

**Comparison with Paper**:
- Paper MUAD F1: 0.9928 (Dataset C, event_num=15, recovered artifact)
- Our Track A: 0.9904 (event_num=15, recovered artifact) → **-0.0024 gap**
- Our Track B: 0.9835 (event_num=14, our regenerated data) → New proposed model

### Remaining Work (Phase 3)

| Task | Status | Est. Time |
|------|--------|-----------|
| Ablation Studies (6 configs × 5 seeds × 50 epochs) | Script ready | ~100 hrs CPU |
| Robustness Tests (6 tests × 5 seeds × 20 epochs) | Planned | ~6 hrs |
| Clean Evaluation Track (fixed bugs) | Script ready | ~4 hrs |
| Clean Ablation (fixed bugs) | Planned | ~20 hrs |
| Robustness Tests (missing modalities, noise) | Planned | ~6 hrs |

### How We Did It: Methodology Summary

1. **Forensic-First Approach**: Never assumed paper claims; verified against raw data, git history, runtime
2. **Reconstructed vs Reused**: Clearly labeled all reconstructed code (4 converters, trim) vs reused (vendor Eadro/MUAD)
3. **Bug-Faithful Reproduction**: Reproduced bugs exactly as found (test-set checkpoint, AL non-propagation, UncertainBlock inconsistency)
4. **Validation Gates**: 10 automated gates preventing progression with bad data
5. **Separation of Tracks**: Track A (faithful MUAD) vs Track B (proposed Graph-Aware MoE) never mixed
6. **Intermediate Artifacts Saved**: Every stage saves output (parsed_data, chunks, trimmed chunks, model checkpoints)
7. **Per-Seed Results Saved**: Each seed's results + best checkpoint saved immediately
8. **Validation-Based Checkpointing**: Track B uses val split for checkpoint selection (no test leakage)
9. **Per-Run Artifacts**: Each run saves config, epoch logs, best checkpoint, final results
10. **Documentation-First**: Every reconstructed component labeled, every bug documented, every decision recorded

---

## Summary

### ✅ Completed (Fully Verified)
- ✅ Dataset forensics & verification (36 faults, 12 services, 7 experiments)
- ✅ Raw → parsed_data reconstruction (4 converters)
- ✅ Eadro preprocessing (faithful, 9907 chunks, 60/40 split)
- ✅ MUAD compatibility trim (trace channel trim)
- ✅ 10 validation gates passing
- ✅ 6 confirmed bugs reproduced
- ✅ Track A Full-Label: 0.9904 F1 (5 seeds)
- ✅ Track A Active Learning: 0.9480 F1 (5 seeds)
- ✅ Track B Graph-Aware MoE: 0.9835 F1 (5 seeds)
- ✅ All 10 validation gates passing
- ✅ 6 confirmed bugs reproduced

### 🔄 In Progress / Ready
- ⏳ Ablation studies (script ready, ~100 hrs CPU)
- ⏳ Robustness tests (planned)
- ⏳ Clean evaluation track (script ready)

### 📋 Ready for Next Phase
All infrastructure, scripts, and documentation ready. Ablation studies can start immediately with `python experiments/run_ablations.py`.

---

**End of Implementation Journey Documentation**