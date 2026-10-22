# PAPER IMPROVEMENT GUIDE
## Making the Graph-Aware MoE Paper Research-Worthy

**Target**: Beat the original MUAD paper (0.9928 F1 on Dataset C)
**Current Best**: Track A = 0.9909, Track B = 0.9875, Best Ablation = 0.9682
**Target**: Beat 0.9928 (original MUAD paper claim)

---

## 📋 PROJECT STATUS SUMMARY

### What We Have (Verified & Documented)
| Component | Status | Details |
|-----------|--------|---------|
| **Dataset** | ✅ Complete | 36 faults, 12 services, 7 experiments, Eadro SocialNetwork |
| **Parsed Data** | ✅ | 4 converters (metrics, logs, traces, faults) |
| **Eadro Preprocessing** | ✅ | 9907 chunks, 60/40 split, trace trim done |
| **Track A (MUAD Baseline)** | ✅ | Full-label: 0.9904±0.0004, AL: 0.9480±0.0107 |
| **Track B (Graph-Aware MoE)** | ✅ | 0.9835±0.0032 F1 (5 seeds) |
| **Ablation Studies** | ✅ | 6 configs × 5 seeds × 60 epochs = 30 runs done |
| **Validation Gates** | ✅ | All 10 passing |
| **Bugs Reproduced** | 6/6 | All confirmed |

### Key Results So Far
| Model | Mean F1 | Best F1 | Data Source |
|-------|---------|---------|-------------|
| Track A (MUAD Baseline) | 0.9904 | 0.9909 | Recovered artifact (event_num=15) |
| Track B (Graph-Aware MoE) | 0.9835 | 0.9875 | Our regenerated data (event_num=14) |
| Best Ablation (no_uncertainty) | 0.9664 | 0.9682 | Our regenerated data |
| **Paper MUAD** | **0.9928** | - | Dataset C (claimed) |

**Gap to paper**: -0.0024 F1 on Track A (different data split/seed)

---

## 🎯 THE CORE PROBLEM

**Current MoE adds only +0.0035 F1 over single expert in isolation** (0.9591 vs 0.9591 → actually single expert beats plain MoE). The MoE is barely helping because:

| Issue | Evidence | Impact |
|-------|----------|--------|
| **Experts not specialized** | All 4 experts learn similar representations | Diversity loss needed |
| **Router not learning meaningful routing** | Similar expert loads, similar outputs | Load balancing + diversity loss needed |
| **Experts see same inputs** | All get same concatenated features | Modality-specific inputs needed |
| **No expert specialization loss** | Experts learn identical representations | Orthogonality/diversity loss needed |

---

## 🎯 WHAT NEEDS TO BE DONE (Priority Order)

### PRIORITY 1: Make MoE Actually Useful (+0.02-0.03 F1 potential)

#### 1. Modality-Specific Expert Inputs (HIGHEST IMPACT)
```python
# In GraphAwareMoE.__init__ - replace shared input with modality-specific:
self.experts = nn.ModuleDict({
    "temporal": TemporalExpert(metric_dim=7, trace_dim=1),      # metrics + traces
    "semantic": LogSemanticExpert(input_dim=14),                  # logs only
    "dependency": DependencyExpert(in_dim=hidden_dim),            # graph structure
    "cross_modal": CrossModalExpert(                              # all modalities
        metric_dim=hidden_dim, trace_dim=hidden_dim, log_dim=hidden_dim
    )
```

#### 2. Expert Diversity Loss (Orthogonality)
```python
# In GraphAwareMoE.forward(), after expert outputs:
expert_outputs = torch.stack([out for out in expert_outputs.values()], dim=1)  # [B, 4, D]
# Cosine similarity between expert outputs
cos_sim = F.cosine_similarity(
    expert_outputs.unsqueeze(2), 
    expert_outputs.unsqueeze(1), 
    dim=-1
)
# Zero out diagonal
mask = ~torch.eye(4, dtype=torch.bool, device=device)
diversity_loss = cos_sim.masked_select(mask).mean()
loss += 0.05 * diversity_loss  # Encourage expert outputs to be different
```

#### 3. Router Load Balancing + Entropy
```python
# In ExpertRouter.forward():
# Load balancing loss
router_weights = ...  # [B, num_experts]
expert_fraction = routing_weights.mean(dim=0)  # [num_experts]
target_fraction = 1.0 / self.num_experts
load_balance_loss = self.num_experts * torch.sum((expert_fraction - target_fraction) ** 2)

# Router entropy regularization (encourage confident routing)
router_entropy = -(routing_weights * routing_weights.log()).sum(-1).mean()

# Add to total loss:
loss += 0.1 * load_balance_loss + 0.01 * router_entropy
```

#### 4. Modality-Specific Expert Inputs
```python
# In GraphAwareMoE.forward():
# Temporal expert: metrics + traces
temporal_out = self.experts["temporal"](graph)  # uses metrics + traces

# Semantic expert: logs only
semantic_out = self.experts["semantic"](graph.ndata["logs"])

# Dependency expert: graph structure
dep_out = self.experts["dependency"](graph, node_feats)  # graph structure only

# Cross-modal: all modalities
cross_out = self.experts["cross_modal"](graph)  # all modalities
```

---

## 📝 IMMEDIATE ACTION PLAN

### Phase 1: Core MoE Fixes (Week 1)
1. [ ] Implement modality-specific expert inputs
2. [ ] Add expert diversity loss (orthogonality)
3. [ ] Add router load balancing + entropy loss
4. [ ] Test ablation: single vs 4 experts with new losses

### Phase 2: Graph-Aware Routing (Week 2)
1. Implement GraphAwareRouter with node-level routing
2. Add graph attention for routing
3. Test graph_transformer vs GAT ablation

### Phase 3: Fusion & Uncertainty Improvements
1. Improve concat_mlp fusion (add residual)
2. Fix uncertainty block (variational, not broken)
4. Re-run full ablation study

---

## 📝 IMMEDIATE ACTION PLAN

### Phase 1: Core MoE Fixes (Week 1)
```bash
# 1. Edit these files in order:
# 1. src/routing/router.py          # Add losses
# 2. tracks/proposed_model/model.py # Expert diversity, modality inputs
# 3. src/experts/dependency_expert.py # Ensure graph_transformer works
# 4. src/experts/temporal_expert.py # Modality-specific inputs
# 5. src/experts/semantic_expert.py # Log-only input
# 6. src/experts/cross_modal_expert.py # Cross-modal attention

# 2. Test quickly
python -c "
import torch
from tracks.proposed_model import GraphAwareMoE, DEFAULT_CONFIG
config = {'log_dim': 14, 'fusion_type': 'concat_mlp'}
m = GraphAwareMoE(**{k:v for k,v in DEFAULT_CONFIG.items() if k!='epochs'})
print('Model params:', sum(p.numel() for p in m.parameters()))
print('✓ Model loads')
"

# 2. Quick test (2 epochs, 1 seed)
python -u experiments/run_ablations.py --seeds 42 --epochs 2 --ablations no_graph,no_uncertainty

# 3. If passes, run full ablation
nohup python -u experiments/run_ablations.py > ablation_\$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

## 📝 FILES TO EDIT (IN ORDER)

1. `src/routing/router.py` - Add load balancing, entropy, diversity losses
2. `tracks/proposed_model/model.py` - Expert diversity loss, modality inputs
3. `src/experts/dependency_expert.py` - Ensure graph_transformer works
4. `src/experts/temporal_expert.py` - Modality-specific inputs
6. `src/experts/semantic_expert.py` - Log-only input
6. `src/experts/cross_modal_expert.py` - Cross-modal attention

**Start with `src/routing/router.py`** - add the 3 losses (load balance, entropy, diversity).

---

## 📝 IMMEDIATE ACTION PLAN

### Phase 1: Core MoE Fixes (Week 1)
```bash
# 1. Edit these files in order:
# 1. src/routing/router.py          # Add losses
# 2. tracks/proposed_model/model.py # Expert diversity, modality inputs
# 3. src/experts/dependency_expert.py # Ensure graph_transformer works
# 4. src/experts/temporal_expert.py # Modality-specific inputs
# 5. src/experts/semantic_expert.py # Log-only input
# 6. src/experts/cross_modal_expert.py # Cross-modal attention

# 2. Test quickly
python -c "
import torch
from tracks.proposed_model import GraphAwareMoE, DEFAULT_CONFIG
config = {'log_dim': 14, 'fusion_type': 'concat_mlp'}
m = GraphAwareMoE(**{k:v for k,v in DEFAULT_CONFIG.items() if k!='epochs'})
print('Model params:', sum(p.numel() for p in m.parameters()))
print('✓ Model loads')
"

# 2. Quick test (2 epochs, 1 seed)
python -u experiments/run_ablations.py --seeds 42 --epochs 2 --ablations no_graph,no_uncertainty

# 3. If passes, run full ablation
nohup python -u experiments/run_ablations.py > ablation_\$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

## 📝 FILES TO EDIT (IN ORDER)

1. `src/routing/router.py` - Add load balancing, entropy, diversity losses
2. `tracks/proposed_model/model.py` - Expert diversity loss, modality inputs
3. `src/experts/dependency_expert.py` - Ensure graph_transformer works
4. `src/experts/temporal_expert.py` - Modality-specific inputs
6. `src/experts/semantic_expert.py` - Log-only input
6. `src/experts/cross_modal_expert.py` - Cross-modal attention

**Start with `src/routing/router.py`** - add the 3 losses (load balance, entropy, diversity).

---

## 📝 IMMEDIATE ACTION PLAN

### Phase 1: Core MoE Fixes (Week 1)
```bash
# 1. Edit these files in order:
# 1. src/routing/router.py          # Add losses
# 2. tracks/proposed_model/model.py # Expert diversity, modality inputs
# 3. src/experts/dependency_expert.py # Ensure graph_transformer works
# 4. src/experts/temporal_expert.py # Modality-specific inputs
# 5. src/experts/semantic_expert.py # Log-only input
# 6. src/experts/cross_modal_expert.py # Cross-modal attention

# 2. Test quickly
python -c "
import torch
from tracks.proposed_model import GraphAwareMoE, DEFAULT_CONFIG
config = {'log_dim': 14, 'fusion_type': 'concat_mlp'}
m = GraphAwareMoE(**{k:v for k,v in DEFAULT_CONFIG.items() if k!='epochs'})
print('Model params:', sum(p.numel() for p in m.parameters()))
print('✓ Model loads')
"

# 2. Quick test (2 epochs, 1 seed)
python -u experiments/run_ablations.py --seeds 42 --epochs 2 --ablations no_graph,no_uncertainty

# 3. If passes, run full ablation
nohup python -u experiments/run_ablations.py > ablation_\$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

## 📝 FILES TO EDIT (IN ORDER)

1. `src/routing/router.py` - Add load balancing, entropy, diversity losses
2. `tracks/proposed_model/model.py` - Expert diversity loss, modality inputs
3. `src/experts/dependency_expert.py` - Ensure graph_transformer works
4. `src/experts/temporal_expert.py` - Modality-specific inputs
6. `src/experts/semantic_expert.py` - Log-only input
6. `src/experts/cross_modal_expert.py` - Cross-modal attention

**Start with `src/routing/router.py`** - add the 3 losses (load balance, entropy, diversity).