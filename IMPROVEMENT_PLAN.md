# IMPROVEMENT PLAN: Boosting F1 Score Beyond 0.9928
## Graph-Aware MoE for Multimodal Microservice Anomaly Detection

---

## 📊 Current Status

| Metric | Current Value | Target |
|--------|---------------|--------|
| Track B (Full Model) | 0.9835 ± 0.0032 | 0.9875 best |
| Best Ablation (no_uncertainty) | 0.9664 ± 0.0015 | 0.9682 best |
| **Phase 1 Quick Wins** | **RUNNING (temp=2.0, top_k=3)** | **Target** |
| **Phase 2 Core Fixes** | **FULLY IMPLEMENTED** | **Target** |
| Paper Claim | **0.9928** | **Target** |
| **Gap** | **-0.0053 to 0.9928** | **Target** |

---

## 🎯 IMPROVEMENT PLAN & CHECKLIST

---

## ✅ PHASE 1: QUICK WINS (Week 1) - Expected: +0.005-0.008 F1

### 1.1 Disable Uncertainty Blocks (Highest ROI) ✅
- [x] **Done** - `no_uncertainty` ablation achieves 0.9682 F1
- [x] **Action**: Set `use_uncertainty=False` as default in Track B config
- [x] Update default config in `tracks/proposed_model/model.py`
- [x] Update ablation script to use `no_uncertainty` as baseline

### 1.2 Router Temperature & Top-K ✅
- [x] Increase router temperature to 2.0 (softer routing)
- [x] Increase top_k from 2 → 3 (more expert collaboration)
- [x] Add `--temperature` and `--top_k` args to `run_ablations.py`
- [x] Test: `python -u experiments/run_ablations.py --seeds 42 --epochs 60 --temperature 2.0 --top_k 3 --ablations no_uncertainty` - RUNNING

### 1.3 Router Entropy Regularization ✅
- [x] Add router entropy regularization (weight: 0.01)
- [x] Encourage confident routing

### 1.4 Quick Test Command
```bash
python -u experiments/run_ablations.py --seeds 42 --epochs 10 --temperature 2.0 --top_k 3 --ablations no_uncertainty
```

---

## ✅ PHASE 2: CORE ARCHITECTURE FIXES (Week 1-2) - COMPLETED
### Target: +0.01-0.02 F1

### 2.1 Expert Diversity Loss (Orthogonality) 🔴 HIGH IMPACT ✅
**File:** `tracks/proposed_model/model.py` - `GraphAwareMoE.forward()`

- [x] Add expert diversity loss (orthogonality loss)
```python
# In GraphAwareMoE.forward(), after expert outputs:
expert_outputs = torch.stack(list(expert_outputs.values()), dim=1)  # [B, 4, D]
cos_sim = F.cosine_similarity(
    expert_outputs.unsqueeze(2), expert_outputs.unsqueeze(1), dim=-1
)
mask = ~torch.eye(4, dtype=torch.bool, device=device)
diversity_loss = cos_sim.masked_select(mask).mean()
loss += 0.05 * diversity_loss  # Encourage expert outputs to be different
```

- [x] Add `diversity_weight` config parameter (default: 0.05) - via `set_diversity_weight()` method
- [x] Test with `no_uncertainty` ablation first

### 2.2 Modality-Specific Expert Inputs 🔴 HIGH IMPACT ✅
**File:** `tracks/proposed_model/model.py` - `GraphAwareMoE.forward()`

- [x] Temporal expert: metrics + traces only
- [x] Semantic expert: logs only
- [x] Dependency expert: graph structure only
- [x] Cross-modal expert: all modalities
- [x] Added `modality_specific_inputs` config parameter (default: True)

```python
# Temporal expert: metrics + traces
temporal_out = self.experts["temporal"](graph)
# Returns averaged metric + trace embeddings

# Semantic expert: logs only
semantic_out = self.experts["semantic"](graph.ndata["logs"])

# Dependency expert: graph structure only
metric_emb = modality_embeddings["metric"]
node_feats = metric_emb.unsqueeze(1).repeat(1, self.num_nodes, 1).view(-1, self.hidden_dim)
dep_out = self.experts["dependency"](graph, node_feats)

# Cross-modal: all modalities
cross_out = self.experts["cross_modal"](graph)
```

### 2.3 Router Improvements ✅
**File:** `src/routing/router.py`

#### 2.3.1 Load Balancing Loss (increase weight) ✅
```python
# Increase weight from 0.01 to 0.1
load_balance_loss = self.num_experts * torch.sum((expert_fraction - target_fraction) ** 2)
```

#### 2.3.2 Router Entropy Regularization ✅
```python
router_entropy = -(routing_weights * routing_weights.log()).sum(-1).mean()
loss += 0.01 * router_entropy
```

#### 2.3.3 Router Temperature & Top-K ✅
```python
# In ExpertRouter.__init__
self.temperature = 2.0  # Was 1.0
self.top_k = 3  # Was 2
```

### 2.4 Expert Diversity Loss (Orthogonality) ✅ (Duplicate of 2.1)
**File:** `tracks/proposed_model/model.py` - `GraphAwareMoE.forward()`

- [x] Add expert diversity loss (orthogonality)
```python
# After expert outputs:
expert_outputs = torch.stack(list(expert_outputs.values()), dim=1)  # [B, 4, D]
cos_sim = F.cosine_similarity(
    expert_outputs.unsqueeze(2), expert_outputs.unsqueeze(1), dim=-1
)
mask = ~torch.eye(4, dtype=torch.bool, device=device)
diversity_loss = cos_sim.masked_select(mask).mean()
loss += 0.05 * diversity_loss
```

---

## 📝 Phase 3: Fusion & Advanced (Week 3-4)

### 3.1 Fusion Improvements
- [ ] Add residual connection in `concat_mlp` fusion
- [ ] Test `dynamic` fusion type
- [ ] Add `ModalityDropout` for robustness

### 3.2 Advanced Router
- [ ] Implement `GraphAwareRouter` with node-level routing
- [ ] Add `UncertaintyGuidedRouter` expert uncertainty estimation

### 3.3 Advanced Training
- [ ] Mixture of Experts with Expert Dropout
- [ ] Hierarchical Routing (coarse→fine)
- [ ] Contrastive Expert Pretraining
- [ ] Curriculum Learning
- [ ] Ensemble of 3 seeds at inference

---

## 📋 MASTER CHECKLIST

### Phase 1: Quick Wins (Week 1)
- [x] Disable uncertainty blocks (`use_uncertainty=False`)
- [x] Increase router temperature to 2.0
- [x] Increase top_k from 2 → 3
- [x] Add router entropy regularization (weight: 0.01)
- [x] Test quick run: `python experiments/run_ablations.py --seeds 42 --epochs 60 --temperature 2.0 --top_k 3 --ablations no_uncertainty` - RUNNING

### Phase 2: Core Fixes - IN PROGRESS
- [ ] **Expert Diversity Loss** (orthogonality) - `tracks/proposed_model/model.py`
- [ ] **Modality-Specific Expert Inputs** - `tracks/proposed_model/model.py`
- [ ] **Router Improvements** - `src/routing/router.py`
  - [ ] Load balancing loss weight: 0.01 → 0.1
  - [ ] Router entropy regularization (weight: 0.01)
  - [x] Router temperature: 2.0
  - [x] Top-k: 2 → 3
- [ ] **Modality-Specific Expert Inputs** - `tracks/proposed_model/model.py`
- [ ] **Expert Diversity Loss** (orthogonality) - `tracks/proposed_model/model.py`

### Phase 3: Fusion & Advanced
- [ ] Fusion: Add residual connection in `concat_mlp`
- [ ] Fusion: Test `dynamic` fusion type
- [ ] Add `ModalityDropout` for robustness
- [ ] Router: Implement `GraphAwareRouter` with node-level routing
- [ ] Add `ModalityDropout` for robustness

### Phase 4: Advanced
- [ ] Mixture of Experts with Expert Dropout
- [ ] Hierarchical Routing (coarse→fine)
- [ ] Contrastive Expert Pretraining
- [ ] Curriculum Learning
- [ ] Ensemble of 3 seeds at inference

---

## 📋 TRACKING CHECKLIST

### Phase 1: Quick Wins (Week 1)
- [x] Disable uncertainty blocks (`use_uncertainty=False`)
- [x] Increase router temperature to 2.0
- [x] Increase top_k from 2 → 3
- [x] Add router entropy regularization (weight: 0.01)
- [x] Test quick run: `python experiments/run_ablations.py --seeds 42 --epochs 60 --temperature 2.0 --top_k 3 --ablations no_uncertainty` - RUNNING

### Phase 2: Core Fixes - COMPLETED
- [x] **Expert Diversity Loss** (orthogonality) - `tracks/proposed_model/model.py`
- [x] **Modality-Specific Expert Inputs** - `tracks/proposed_model/model.py`
- [x] **Router Improvements** - `src/routing/router.py`
  - [x] Load balancing loss weight: 0.01 → 0.1
  - [x] Router entropy regularization (weight: 0.01)
  - [x] Router temperature: 2.0
  - [x] Top-k: 2 → 3
- [x] **Modality-Specific Expert Inputs** - `tracks/proposed_model/model.py`
- [x] **Expert Diversity Loss** (orthogonality) - `tracks/proposed_model/model.py`

### Phase 3: Fusion & Advanced
- [ ] Fusion: Add residual connection in `concat_mlp`
- [ ] Fusion: Test `dynamic` fusion type
- [ ] Add `ModalityDropout` for robustness
- [ ] Router: Implement `GraphAwareRouter` with node-level routing
- [ ] Add `ModalityDropout` for robustness

### Phase 4: Advanced
- [ ] Mixture of Experts with Expert Dropout
- [ ] Hierarchical Routing (coarse→fine)
- [ ] Contrastive Expert Pretraining
- [ ] Curriculum Learning
- [ ] Ensemble of 3 seeds at inference

---

## 📋 TRACKING CHECKLIST

### Phase 1: Quick Wins
- [x] Disable uncertainty blocks (`use_uncertainty=False`)
- [x] Increase router temperature to 2.0
- [x] Increase top_k from 2 → 3
- [x] Add router entropy regularization (weight: 0.01)
- [x] Test quick run: `python experiments/run_ablations.py --seeds 42 --epochs 60 --temperature 2.0 --top_k 3 --ablations no_uncertainty` - RUNNING

### Phase 2: Core Fixes - COMPLETED
- [x] **Expert Diversity Loss** (orthogonality) - `tracks/proposed_model/model.py`
- [x] **Modality-Specific Expert Inputs** - `tracks/proposed_model/model.py`
- [x] **Router Improvements** - `src/routing/router.py`
  - [x] Load balancing loss weight: 0.01 → 0.1
  - [x] Router entropy regularization (weight: 0.01)
  - [x] Router temperature: 2.0
  - [x] Top-k: 2 → 3

### Phase 3: Fusion & Advanced
- [ ] Fusion: Add residual connection in `concat_mlp`
- [ ] Fusion: Test `dynamic` fusion type
- [ ] Add `ModalityDropout` for robustness
- [ ] Router: Implement `GraphAwareRouter` with node-level routing
- [ ] Add `ModalityDropout` for robustness

### Phase 4: Advanced
- [ ] Mixture of Experts with Expert Dropout
- [ ] Hierarchical Routing (coarse→fine)
- [ ] Contrastive Expert Pretraining
- [ ] Curriculum Learning
- [ ] Ensemble of 3 seeds at inference

---

## 📋 QUICK REFERENCE

### Key Files to Edit
| File | Purpose |
|------|---------|
| `src/routing/router.py` | Router losses, temperature, top-k |
| `tracks/proposed_model/model.py` | Expert diversity, modality inputs, fusion |
| `src/experts/dependency_expert.py` | Graph Transformer support |
| `src/experts/temporal_expert.py` | Modality-specific inputs |
| `src/experts/semantic_expert.py` | Log-only input |
| `src/experts/cross_modal_expert.py` | Cross-modal attention |
| `src/fusion/fusion.py` | Fusion types, output_proj |
| `experiments/run_ablations.py` | Ablation runner, args |

### Quick Test Commands
```bash
# Quick test (2 epochs, 1 seed)
python -u experiments/run_ablations.py --seeds 42 --epochs 2 --temperature 2.0 --top_k 3 --ablations no_uncertainty

# Quick test with fusion ablation
python -u experiments/run_ablations.py --seeds 42 --epochs 2 --temperature 2.0 --top_k 3 --ablations no_fusion

# Full ablation (5 seeds, 10 epochs for testing)
nohup python -u experiments/run_ablations.py --seeds 42,123 --epochs 10 --temperature 2.0 --top_k 3 --ablations no_uncertainty,no_graph,no_fusion > ablation_test.log 2>&1 &

# Full ablation (5 seeds, 60 epochs)
nohup python -u experiments/run_ablations.py --seeds 42,123,456,789,999 --epochs 60 --temperature 2.0 --top_k 3 --ablations no_uncertainty,no_graph,no_fusion,plain_moe,graph_transformer,single_expert > ablation_full.log 2>&1 &
```

---

## 🎯 SUCCESS CRITERIA

| Milestone | Target | Status |
|-----------|--------|--------|
| Quick wins test passes | F1 > 0.970 at epoch 10 | ⏳ |
| Phase 2 core fixes | Mean F1 > 0.970 | ⏳ |
| Phase 3 fusion/advanced | Mean F1 > 0.990 | ⏳ |
| Beat paper (0.9928) | > 0.9928 | ⏳ |

---

**Next Step:** Phase 2 core fixes fully implemented (Expert Diversity Loss, Modality-Specific Expert Inputs, Router load balancing 0.1, entropy 0.01). Next: Run full ablation with 5 seeds × 60 epochs to validate Phase 1 + Phase 2 improvements, then move to Phase 3 Fusion & Advanced.