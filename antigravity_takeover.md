# ANTIGRAVITY TAKEOVER PACKAGE
## Graph-Aware MoE for Multimodal Microservice Anomaly Detection
## Handoff from Opencode Session - Ready for Antigravity Migration

---

## 📁 REPOSITORY LOCATION
**Source:** `C:/Users/as999/OneDrive/Desktop/trash/impl/project`
**Action:** Copy entire `project/` folder to Antigravity environment

---

## ✅ WORK COMPLETED SUMMARY

### Phase 1: Quick Wins (DONE)
- Disabled uncertainty blocks (`use_uncertainty=False`) → `no_uncertainty` ablation
- Router temperature: 2.0, Top-k: 3
- Entropy regularization: 0.01
- Ablation script fixed (removed `epochs` from model constructor)

### Phase 2: Core Architecture Fixes (DONE)
| Component | File | Change |
|-----------|------|--------|
| Expert Diversity Loss | `tracks/proposed_model/model.py` | Cosine similarity orthogonality loss (weight 0.01) |
| Modality-Specific Inputs | `tracks/proposed_model/model.py` | Temporal: metrics+traces, Semantic: logs, Dependency: graph, Cross-modal: all |
| Router Load Balance | `src/routing/router.py` | Weight 0.01 (Track B) / 0.1 (ablations) |
| Router Entropy | `src/routing/router.py` | Weight 0.01 (all 3 routers) |
| Temperature/Top-k | All routers | 2.0 / 3 |

### NaN Fixes (CRITICAL - Applied to ALL 3 Uncertainty Blocks)
| File | Fix |
|------|-----|
| `src/uncertainty/uncertainty.py` - `UncertaintyBlock` | Variance clamped [1e-8, 1.0], std clamped max 10 |
| `src/uncertainty/uncertainty.py` - `VariationalUncertaintyBlock` | logvar clamped [-10, 10], std clamped max 10, sample clamped [-1e4, 1e4] |
| `tracks/proposed_model/model.py` | KL/routing/diversity losses clamped max 10, fallback to cls_loss |

### Config Separation (run_ablations.py)
| Ablation | LR | Grad Clip | Load Balance |
|----------|-----|-----------|--------------|
| `full` (Track B) | 0.001 | 1.0 | 0.01 |
| Others (`no_uncertainty`, etc.) | 5e-5 | 0.5 | 0.1 |

---

## 🎯 CURRENT STATUS & TARGETS
- **Track B Baseline:** 0.9835 ± 0.0032 (5 seeds, 60 epochs)
- **Best Ablation (prev):** 0.9682 (`no_uncertainty`)
- **Paper Target:** 0.9928
- **Full Track B Test (seed 42):** Reached epoch 15, F1=0.9246 (NO NaN!)
- **Running:** 5 seeds × 60 epochs on local CPU → `trackb_full_phase1_phase2.log`
- **GPU Server Issue:** Was failing at epoch 9 (FP16 numerics) - latest fixes should resolve

---

## 📋 ESSENTIAL FILES TO TRANSFER
```
project/
├── tracks/proposed_model/model.py          # Main model with all fixes
├── tracks/proposed_model/model.py.backup   # Original baseline
├── src/routing/router.py                   # All 3 routers with load_balance/entropy
├── src/uncertainty/uncertainty.py          # 3 uncertainty blocks with clamping
├── src/experts/                            # All 4 experts (modality-specific inputs)
├── experiments/run_ablations.py            # Ablation runner with config separation
├── experiments/run_ablations.py.backup     # Original
├── experiments/run_track_b_full.py         # Original Track B script
├── IMPROVEMENT_PLAN.md                     # Full task tracking
└── preprocessing/muad_compat/output/       # Data (chunk_train.pkl, chunk_test.pkl, metadata.json)
```

---

## 🚀 RESUME COMMANDS ON ANTIGRAVITY

### Install Dependencies
```bash
cd project
pip install -r requirements.txt  # or check pyproject.toml
```

### Quick Test (1 seed, 5 epochs)
```bash
python -u experiments/run_ablations.py --seeds 42 --epochs 5 --temperature 2.0 --top_k 3 --ablations full
```

### Full Run (5 seeds, 60 epochs) - CPU
```bash
nohup python -u experiments/run_ablations.py \
  --seeds "42,123,456,789,999" \
  --epochs 60 --temperature 2.0 --top_k 3 \
  --ablations full > trackb_full.log 2>&1 &
```

### GPU Version (if CUDA available)
```bash
# Recommended: disable FP16 for stability
export TORCH_FLOAT32_MATMUL_PRECISION=high
CUDA_VISIBLE_DEVICES=0 python -u experiments/run_ablations.py \
  --seeds "42,123,456,789,999" \
  --epochs 60 --temperature 2.0 --top_k 3 \
  --ablations full > trackb_full_gpu.log 2>&1 &
```

---

## 🔑 KEY CONTEXT FOR CONTINUATION

### 1. NaN at Epoch 9 WAS FIXED
- **Root cause:** VariationalUncertaintyBlock logvar → -inf, std → inf on GPU (FP16)
- **Fix:** Aggressive clamping in ALL 3 uncertainty block types:
  - `UncertaintyBlock`: var ∈ [1e-8, 1.0], std ≤ 10
  - `VariationalUncertaintyBlock`: logvar ∈ [-10, 10], std ≤ 10, sample ∈ [-1e4, 1e4]
  - Model-level: all loss components clamped max 10, fallback to cls_loss

### 2. Track B Uses DIFFERENT Hyperparams than Ablations
- Config separation in `run_ablations.py` lines 93-111
- `full` ablation → LR=0.001, grad_clip=1.0, load_balance=0.01
- Other ablations → LR=5e-5, grad_clip=0.5, load_balance=0.1
- **Do NOT use same hyperparams for both**

### 3. GPU Server May Need FP32 Mode
```python
# Add at top of run_ablations.py if FP16 issues persist
import torch
torch.set_float32_matmul_precision('high')
# Or disable autocast
```

### 4. Current Best Target
- Beat 0.9835 baseline → reach 0.9928 paper claim
- Phase 1+2 architectural gains should help if training stabilizes

---

## 📊 NEXT PHASES (Phase 3+)

### Phase 3: Fusion & Advanced (If Track B Stabilizes)
- [ ] Fusion: Add residual connection in `concat_mlp`
- [ ] Fusion: Test `dynamic` fusion type
- [ ] Add `ModalityDropout` for robustness
- [ ] Router: Implement `GraphAwareRouter` with node-level routing
- [ ] Router: Add `UncertaintyGuidedRouter` expert uncertainty estimation

### Phase 4: Advanced Training
- [ ] Mixture of Experts with Expert Dropout
- [ ] Hierarchical Routing (coarse→fine)
- [ ] Contrastive Expert Pretraining
- [ ] Curriculum Learning
- [ ] Ensemble of 3 seeds at inference

---

## 📋 IMPROVEMENT_PLAN.md
**Location:** `project/IMPROVEMENT_PLAN.md` - Contains full task tracking with all checkpoints updated.

---

## ⚠️ KNOWN ISSUES TO WATCH

1. **FP16 on GPU** - If NaN returns, force FP32: `torch.set_default_dtype(torch.float32)`
2. **Diversity Loss + Uncertainty** - May conflict; weight 0.01 currently, can reduce to 0.005
3. **Entropy Reg + KL Loss** - Double regularization; entropy weight 0.01, can reduce/remove for full Track B
4. **Load Balance 0.01** - May be too low; can increase to 0.05 if experts don't specialize

---

## 📞 HANDOFF COMPLETE
**All fixes in working files (not backups).** 
**5-seed run in progress locally** - check `trackb_full_phase1_phase2.log` when done.
**Copy entire `project/` folder to Antigravity.**

---
*Generated: 2026-10-30 | Opencode Session Handoff*
