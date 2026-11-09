# UG-MoE Implementation Plan
## Based on UG-MoE_Implementation_Guide.md and Actual Codebase Analysis

---

## Executive Summary

Project: Uncertainty-Guided Graph-Aware MoE for Multimodal Microservice Anomaly Detection
Primary Dataset: Eadro SN (SocialNetwork) - Multimodal (Metrics + Traces + Logs)
Secondary Dataset: HDFS (Log-only) - Already working, 0.9945 F1
Primary Goal: Reproduce MUAD baseline (Track A) + Implement Graph-Aware MoE improvements (Track B)

---

## Current Implementation Status

### FULLY IMPLEMENTED (Ready for Training)

| Component | Status | Location |
|----------|--------|----------|
| Core Model Architecture | Complete | tracks/proposed_model/model.py - GraphAwareMoE with 4 experts + 3 log-only experts |
| 4 Experts (Track B) | Complete | src/experts/ - Temporal, Semantic, Dependency, CrossModal |
| 3 Log-Only Experts | Complete | src/experts/log_expert.py - LogExpert, LogGraphExpert, LogContextExpert |
| MoE Router | Complete | src/routing/router.py - 3 router types with log_only_mode |
| Uncertainty Modeling | Complete | src/uncertainty/uncertainty.py - Standard + Variational |
| Uncertainty-Guided Routing | Implemented | ExpertRouter with uncertainty-guided routing |
| Multimodal Fusion | Complete | AdaptiveFusion, DynamicFusion, HierarchicalFusion, ModalityDropout |
| HDFS Preprocessing Pipeline | Complete | preprocessing/hdfs/ - all 4 stages working |
| HDFS Dataset/Loader | Complete | HDFSLogDataset, create_hdfs_dataloaders |
| HDFS Training Script | Complete | experiments/run_hdfs.py
| Track B Full Training | Complete | experiments/run_track_b_full.py
| Ablation Framework | Partial | run_ablations.py with 6 configs (missing some) |
| Track B Full Training | Complete | tracks/proposed_model/train.py |
| HDFS Preprocessing Pipeline | Complete | preprocessing/hdfs/ - all 4 stages working |

---

## MISSING / INCOMPLETE (Per Implementation Guide)

### 1. Track A Baseline (MUAD) on Eadro SN - NOT RUN
- Track A exists (run_track_a_full.py) but uses vendor/muad/ original code
- Need to verify it runs on Eadro SN data (event_num=15) and produces baseline results
- No evidence it has been run successfully on Eadro SN dataset

### 2. Primary Dataset (Eadro SN) - Track B Training
- NOT RUN YET - Primary dataset per Implementation Guide is Eadro SN
- Current successful run is on HDFS (log-only), but primary target is Eadro SN multimodal dataset
- Track B training script (run_track_b_full.py) uses Eadro data but never been run successfully

### 3. Required Experiments (Section 8 of Guide) - NOT RUN
| Experiment | Status |
|------------|--------|
| A. Baseline comparison (GAT, Plain MoE, Graph-MoE, Full) | Not run |
| B. Component ablation (w/o graph, w/o MoE, w/o routing, w/o uncertainty, w/o fusion) | Partially (ablation script exists but missing configs) |
| C. Modality ablation (Log only, Trace only, Metric only, combinations) | Not run |
| D. Missing modality tests (10%, 20%, 40%, 60%) | Not run |
| E. Noise robustness tests | Not run |
| F. Expert-routing analysis (weight plots, entropy) | Not run |
| G. Uncertainty analysis (clean vs noisy) | Not run |
| H. Hyperparameter sensitivity | Not run |

### 4. Missing ModalityDropout Integration
- ModalityDropout class exists in fusion.py but not integrated into training pipeline

### 6. Missing Evaluation Metrics
- PR-AUC, ROC-AUC not implemented in evaluation (only F1, Precision, Recall)

---

## PHASE 1: Core Training Pipeline Verification (Week 1)

### 1.1 Verify Track B Full Training on Eadro SN (Primary Dataset)
```bash
# Run Track B on Eadro SN (primary dataset)
cd ~/AHA_BTP/UG_MoE/project
python -u experiments/run_track_b_full.py --device cuda --epochs 50
```
Expected: Successful training on Eadro SN data (5610 train / ~1400 test chunks)
Target: F1 > 0.98 on Eadro SN (Dataset C in MUAD paper claims 0.9928)

### 1.2 Verify Track A Baseline (MUAD Reproduction)
```bash
python -u experiments/run_track_a_full.py --device cuda --epochs 50
```
Expected: Reproduce paper 0.9928 F1 on Eadro SN (Dataset C)
Critical: Must work before claiming improvements

### 1.3 Fix Any Training Issues
- Monitor for NaN losses, gradient explosions
- Verify GPU memory usage (batch_size=50 may need adjustment)

---

## PHASE 2: Required Experiments (Section 8 of Guide) - Weeks 2-3

### 2.1 Experiment A: Baseline Comparison
| Model | Status | Action |
|-------|--------|--------|
| GAT Baseline | Need to implement/run | Implement in run_ablations.py |
| Plain MoE | Need to implement | Router only, no graph |
| Graph-MoE | GraphAwareMoE (Track B) | Done |
| Full Proposed | GraphAwareMoE (Track B) | Done |

Action: Implement GAT baseline and Plain MoE variants in run_ablations.py

### 2.2 Component Ablation (Section 8B) - Partially Done
| Ablation | Status | Action |
|----------|--------|--------|
| w/o graph | In ablation script | Done |
| w/o MoE | Need to implement | Add config |
| w/o dynamic routing | Need to implement | Add config |
| w/o entropy reg | In ablation | Done |
| w/o diversity loss | Need to implement | Add config |
| w/o adaptive fusion | Need to implement | Add config |

Action: Implement missing ablation configs in run_ablations.py

### 2.3 Modality Ablation (Section 8C)
| Config | Status | Action |
|--------|--------|--------|
| Log only | Need config | Add config |
| Trace only | Need config | Add config |
| Metric only | Need config | Add config |
| Log+Trace | Need config | Add config |
| Log+Metric | Need config | Add config |
| Trace+Metric | Need config | Add config |
| All three | Default | Done |

### 2.4 Missing Modality Tests (Section 8D)
| Test | Status | Action |
|------|--------|--------|
| 10% missing | | Add config |
| 20% missing | | Add config |
| 40% missing | | Add config |
| 60% missing | | Add config |
| Missing specific modality | | Add config |

### 2.5 Noise Robustness (Section 8E) - NOT IMPLEMENTED
### 2.6 Expert Routing Analysis (Section 8F) - NOT IMPLEMENTED
### 2.7 Uncertainty Analysis (Section 8G) - NOT IMPLEMENTED
### 2.9 Hyperparameter Sensitivity (Section 8H) - NOT IMPLEMENTED

---

## Phase 3: Missing Features Integration (Week 3)

### 3.1 ModalityDropout Integration
- Location: src/fusion/fusion.py - ModalityDropout class exists
- Action: Integrate into training pipeline in run_track_b_full.py and run_hdfs.py

### 3.2 PR-AUC and ROC-AUC Metrics
- Currently only F1, Precision, Recall implemented
- Need to add PR-AUC, ROC-AUC to evaluation

### 3.3 ModalityDropout Integration
- Add to training pipeline in run_track_b_full.py and run_hdfs.py

---

## Phase 4: Track A Baseline Verification (Week 4)

### 4.1 Run Track A (MUAD Baseline) on Eadro SN
```bash
python -u experiments/run_track_a_full.py --device cuda --epochs 50
```
Verify it reproduces paper 0.9928 F1 on Eadro SN (Dataset C)

### 4.2 Compare Track A vs Track B
Generate comparison tables matching paper Table II format

---

## Phase 5: Generate Required Outputs (Section 10)

### 5.1 Generate All Required Outputs
| Output | Status |
|--------|--------|
| Preprocessed dataset with schema | |
| Service dependency graph file | |
| GAT baseline checkpoint | |
| Plain MoE checkpoint | |
| Graph-MoE checkpoint | |
| Full model checkpoint | |
| Overall performance table | |
| Ablation table | |
| Missing/noisy modality plots | |
| Expert-weight heatmaps | |
| Uncertainty/fusion-weight plots | |
| Hyperparameter sensitivity plots | |
| Reproducible evaluation script | |

---

## PRIORITIZED ACTION PLAN

### IMMEDIATE (This Week)
1. Run Track B on Eadro SN - Verify full pipeline works on primary dataset
2. Run Track A baseline - Validate MUAD reproduction
3. Fix any training issues (NaN loss, OOM, etc.)

### Week 2: Ablation Studies
1. Implement missing ablation configs in run_ablations.py
2. Run full ablation study (6 configs x 5 seeds)
3. Implement ModalityDropout in training pipeline
4. Add PR-AUC/ROC-AUC metrics

### Week 3: Advanced Experiments
1. Implement missing modality ablations
2. Implement missing modality tests
3. Implement noise robustness tests
4. Implement expert routing analysis
5. Implement u

---

## PHASE 3: Missing Features Integration (Week 3)

### 3.1 ModalityDropout Integration
- Location: src/fusion/fusion.py - ModalityDropout class exists
- Action: Integrate into training pipeline in run_track_b_full.py and run_hdfs.py

### 3.2 PR-AUC and ROC-AUC Metrics
- Currently only F1, Precision, Recall implemented
- Need to add PR-AUC, ROC-AUC to evaluation

### 3.3 ModalityDropout Integration
- Add to training pipeline in run_track_b_full.py and run_hdfs.py

---

## Phase 4: Track A Baseline Verification (Week 4)

### 4.1 Run Track A (MUAD Baseline) on Eadro SN
```bash
python -u experiments/run_track_a_full.py --device cuda --epochs 50
```
Verify it reproduces paper 0.9928 F1 on Eadro SN (Dataset C)

### 4.2 Compare Track A vs Track B
Generate comparison tables matching paper Table II format

---

## Phase 5: Generate Required Outputs (Section 10)

### 5.1 Generate All Required Outputs
| Output | Status |
|--------|--------|
| Preprocessed dataset with schema | |
| Service dependency graph file | |
| GAT baseline checkpoint | |
| Plain MoE checkpoint | |
| Graph-MoE checkpoint | |
| Full model checkpoint | |
| Overall performance table | |
| Ablation table | |
| Missing/noisy modality plots | |
| Expert-weight heatmaps | |
| Uncertainty/fusion-weight plots | |
| Hyperparameter sensitivity plots | |
| Reproducible evaluation script | |

---

## PRIORITIZED ACTION PLAN

### IMMEDIATE (This Week)
1. Run Track B on Eadro SN - Verify full pipeline works on primary dataset
2. Run Track A baseline - Validate MUAD reproduction
3. Fix any training issues (NaN loss, OOM, etc.)

### Week 2: Ablation Studies
1. Implement missing ablation configs in run_ablations.py
2. Run full ablation study (6 configs x 5 seeds)
3. Implement ModalityDropout in training pipeline
4. Add PR-AUC/ROC-AUC metrics

### Week 3: Advanced Experiments
1. Implement missing modality ablations
2. Implement missing modality tests
3. Implement noise robustness tests
4. Implement expert routing analysis
5. Implement uncertainty analysis

### Week 4: Track A Baseline & Outputs
1. Run Track A baseline successfully
2. Generate all required outputs (Section 10)
3. Generate comparison tables matching paper format

---

## FILES THAT NEED CHANGES

| File | Changes Needed |
|------|----------------|
| experiments/run_ablations.py | Add missing ablation configs (w/o MoE, w/o dynamic routing, w/o probabilistic, w/o fusion, modality ablations, missing modality tests) |
| experiments/run_track_b_full.py | Add ModalityDropout, PR-AUC/ROC-AUC |
| experiments/run_hdfs.py | Same as above |
| tracks/proposed_model/train.py | Add ModalityDropout, PR-AUC/ROC-AUC |
| experiments/run_ablations.py | Add missing ablation configs (w/o MoE, w/o dynamic routing, w/o probabilistic, w/o fusion, modality ablations, missing modality tests) |
| tracks/proposed_model/train.py | Add PR-AUC/ROC-AUC, ModalityDropout |

Files Already Working (Don't Touch)
- tracks/proposed_model/model.py - Core architecture
- src/experts/log_expert.py - Log-only experts
- src/routing/router.py - Router with log_only_mode
- src/fusion/fusion.py - All fusion types
- src/uncertainty/uncertainty.py - Uncertainty blocks
- preprocessing/hdfs/*.py - HDFS pipeline
- experiments/run_hdfs.py - HDFS training
- experiments/run_ablations.py - Ablation framework (partial)
- experiments/run_track_b_full.py - Track B training
- tracks/proposed_model/train.py - Track B training

IMMEDIATE NEXT STEPS

1. Run Track B on Eadro SN (10 epochs to verify pipeline)
2. Run Track A baseline (validate MUAD reproduction)
3. Implement missing ablation configs
4. Run full ablation study (5 seeds x 6 configs)
5. Implement ModalityDropout in training pipeline
7. Add PR-AUC/ROC-AUC metrics
8. Run all required experiments
9. Generate all required outputs (Section 10)

CURRENT BEST RESULT: HDFS 0.9945 F1 (beats paper 0.9928 on different dataset)
TARGET: Eadro SN 0.9945+ F1 (beat paper 0.9928 on Dataset C)

The log-only MoE architecture is validated - now need to validate on primary Eadro SN dataset and run all required experiments.
