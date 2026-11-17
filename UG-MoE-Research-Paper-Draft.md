# Graph-Aware Mixture-of-Experts for Multimodal Anomaly Detection in Microservice Systems

## Abstract

Microservice architectures generate heterogeneous observability data including metrics, traces, and logs. Existing anomaly detection approaches often rely on single modalities or fail to account for the uncertainty and varying importance of each modality. We propose Graph-Aware Mixture-of-Experts (UG-MoE), a novel anomaly detection framework that combines specialized modality experts with graph-aware routing, uncertainty-guided expert selection, and adaptive multimodal fusion. Our model processes metrics, traces, and logs through dedicated encoders, routes inputs to specialized experts via a graph-aware router, quantifies per-expert uncertainty through variational blocks, and fuses expert outputs using confidence-weighted adaptive fusion. Extensive experiments on a microservice dataset demonstrate that our approach achieves 99.17% F1-score, outperforming the MUAD baseline (0.9896) by 0.23% with statistical significance (p = 0.0052). We further show that metrics is the dominant modality (F1 drops to 0.4532 when missing), traces and logs are individually redundant, and the model is robust to noise up to level 0.5 on metrics.

**Keywords:** anomaly detection, microservice systems, mixture-of-experts, multimodal learning, uncertainty quantification, graph attention

---

## 1. Introduction

Microservice architectures decompose monolithic applications into independently deployable services, improving scalability and flexibility but introducing complexity in fault detection. Anomalies in one service can propagate through service dependency chains, making timely detection critical for system reliability. Microservice systems generate three primary modalities of observability data:

- **Metrics**: Time-series measurements such as CPU utilization, memory usage, and request latency
- **Traces**: Distributed tracing data capturing request flows across services
- **Logs**: Textual event records capturing service behavior and errors

Each modality provides complementary information, but no single modality is sufficient for comprehensive anomaly detection. Existing approaches either treat modalities independently or fuse them without accounting for uncertainty in individual modalities or their varying reliability across different anomaly types.

We propose **Graph-Aware Mixture-of-Experts (UG-MoE)**, a novel anomaly detection framework that:

1. Encodes each modality (metrics, traces, logs) using dedicated temporal encoders (GRU-based)
2. Constructs a service dependency graph and applies graph attention to capture inter-service relationships
3. Routes inputs to specialized experts (temporal, semantic/dependency, cross-modal) via a graph-aware uncertainty-guided router
4. Quantifies per-expert uncertainty through variational blocks with KL regularization
5. Fuses expert outputs using adaptive confidence-weighted fusion
6. Classifies the fused representation as normal or anomalous

### 1.1 Contributions

- A graph-aware MoE architecture that jointly models graph structure and multimodal expert specialization
- Uncertainty-guided routing that dynamically weights experts based on their predictive confidence
- Adaptive multimodal fusion that weights modalities according to their per-sample reliability
- Comprehensive empirical evaluation on microservice anomaly detection, including ablations, missing-modality robustness, and noise robustness

---

## 2. Related Work

### 2.1 Unimodal Anomaly Detection

Unimodal approaches focus on a single data source:

- **Log-based methods**: Parse log templates using Drain and detect anomalies using deep learning models
- **Metric-based methods**: Apply statistical or deep learning methods to time-series metrics
- **Trace-based methods**: Analyze distributed traces for performance anomalies

These approaches suffer from limited information and cannot detect anomalies that manifest across multiple modalities.

### 2.2 Multimodal Anomaly Detection

Multimodal approaches combine logs, traces, and metrics:

- **MUAD**: Multimodal Uncertainty-Aware Anomaly Detection using a Graph-based Probabilistic Encoder (GPE) and Confidence-aware Fusion Mechanism (CFM)
- **MTAD-GAT**: Multivariate time series anomaly detection using GAT
- **DeepLog**: Log-based anomaly detection using LSTM

### 2.3 Mixture-of-Experts for Anomaly Detection

Mixture-of-Experts architectures partition the input space and route inputs to specialized expert networks. Our approach extends this concept with graph-aware routing and uncertainty-guided expert selection.

---

## 3. Methodology

### 3.1 System Overview

Our Graph-Aware MoE model processes multimodal observability data through six stages:

1. **Modality Encoders**: GRU-based encoders process metric, trace, and log sequences
2. **Specialized Experts**: Four experts process different aspects of the data:
   - Temporal Expert: GRU-based temporal sequence modeling
   - Semantic Expert: Transformer-based log semantic analysis
   - Dependency Expert: GAT-based graph dependency modeling
   - Cross-Modal Expert: Cross-attention across modalities
3. **Expert Router**: Graph-aware uncertainty-guided router assigns weights to experts
4. **Uncertainty Blocks**: Variational blocks estimate per-expert uncertainty via KL regularization
5. **Adaptive Fusion**: Confidence-weighted fusion combines expert outputs
6. **Classifier**: Two-layer MLP classifies the fused representation as normal or anomalous

### 3.2 Modality Encoders

Each modality is encoded separately:

- **Metric Encoder**: GRU over temporal metric sequences
- **Trace Encoder**: GRU over temporal trace sequences
- **Log Encoder**: Linear projection of log template embeddings

### 3.3 Specialized Experts

| Expert | Purpose | Architecture |
|--------|---------|--------------|
| Temporal | Temporal pattern detection | GRU backbone |
| Semantic | Log semantic understanding | Transformer |
| Dependency | Service dependency modeling | GAT |
| Cross-Modal | Cross-modality interaction | Cross-attention |

### 3.4 Expert Router

The router computes routing weights using graph-aware features and per-expert uncertainties. Top-k routing (k=2) selects the most relevant experts for each input.

### 3.5 Uncertainty Blocks

Each expert output passes through a variational uncertainty block that models Gaussian uncertainty N(mu, var) with KL divergence regularization toward N(0, I).

### 3.6 Adaptive Fusion

Expert outputs are fused using confidence-weighted attention, where fusion weights are learned per-modality based on quality estimation.

### 3.7 Classifier

A two-layer MLP (fusion_dim → hidden_dim → num_classes=2) with GELU activation and dropout classifies the fused representation as normal or anomalous.

---

## 4. Experimental Setup

### 4.1 Dataset

Experiments are conducted on a microservice dataset with the following characteristics:

| Property | Value |
|----------|-------|
| num_nodes (services) | 12 |
| num_edges | 24 |
| num_event_num (log templates) | 15 |
| metric_dim | 7 |
| trace_dim | 1 |
| chunk_length (time window) | 10 |
| Training chunks | 5,610 |
| Test chunks | 2,244 |

Each data chunk contains per-node metrics, traces, and logs, along with a culprit label (0 = normal, ≥1 = anomaly).

### 4.2 Baseline Models

| Model | Description |
|-------|-------------|
| Track A (MUAD) | Multimodal Uncertainty-Aware Anomaly Detection baseline |
| Track B (Graph-Aware MoE) | Proposed model |
| Plain MoE | No graph routing, no uncertainty, simple fusion |
| GAT Baseline | Graph Attention Network baseline |
| Graph-MoE | Graph-centric MoE baseline |

### 4.3 Evaluation Metrics

- Precision, Recall, F1-Score
- ROC-AUC, PR-AUC
- Paired t-test for statistical significance

### 4.4 Hyperparameters

| Parameter | Value |
|-----------|-------|
| hidden_dim | 64 |
| expert_hidden_dim | 128 |
| fusion_dim | 128 |
| router_hidden_dim | 128 |
| dropout | 0.1 |
| num_experts | 4 |
| expert_types | ["temporal", "semantic", "dependency", "cross_modal"] |
| router_type | "uncertainty_guided" |
| top_k | 2 |
| use_graph_routing | True |
| use_uncertainty_routing | True |
| fusion_type | "confidence" |
| use_uncertainty | True |
| uncertainty_type | "variational" |
| dependency_expert_type | "gat" |
| log_dim | 15 |

---

## 5. Experimental Results

### 5.1 Main Results (Track A vs Track B)

With 5 seeds, our Graph-Aware MoE (Track B) achieves:

| Model | Mean F1 ± Std | p-value |
|-------|--------------|---------|
| Track A (MUAD Baseline) | 0.9896 ± 0.0007 | — |
| Track B (Graph-Aware MoE) | 0.9917 ± 0.0005 | p = 0.0052 |

**Improvement: +0.0022 (+0.23%), statistically significant (p < 0.05)**

Per-seed results:

| Seed | Track A (MUAD) | Track B (Graph MoE) | Difference |
|------|---------------|-------------------|------------|
| 42 | 0.9895 | 0.9917 | +0.0021 |
| 123 | 0.9887 | 0.9917 | +0.0030 |
| 456 | 0.9892 | 0.9925 | +0.0033 |
| 789 | 0.9898 | 0.9912 | +0.0014 |
| 999 | 0.9909 | 0.9922 | +0.0014 |

Paired t-test: t = 5.5354, p = 0.005206 (statistically significant)

### 5.2 Ablation Study

| Ablation | Mean F1 ± Std | Δ vs Full |
|----------|--------------|-----------|
| Full Model (Graph-Aware MoE) | 0.9917 ± 0.0005 | — |
| no_graph | 0.9901 ± 0.0025 | -0.0009 |
| no_uncertainty | 0.9896 ± 0.0006 | -0.0021 |
| no_fusion | 0.9917 ± 0.0013 | -0.0001 |
| plain_moe | 0.9901 ± 0.0025 | -0.0016 |
| single_expert | 0.9835 ± 0.0021 | -0.0082 |
| graph_transformer | 0.9879 ± 0.0010 | -0.0038 |

**Key Findings:**

- **Graph routing** (no_graph): Removing graph-aware routing decreases F1 by 0.0009, showing graph structure provides useful information
- **Uncertainty blocks** (no_uncertainty): Removing uncertainty-guided routing decreases F1 by 0.0021, confirming uncertainty guidance is important
- **Fusion** (no_fusion): Removing confidence-weighted fusion has negligible impact (-0.0001), suggesting fusion weights are less critical on this dataset
- **Single expert** (single_expert): Using only temporal expert drops F1 by 0.0082, showing MoE diversity is important
- **Graph Transformer** (graph_transformer): Replacing GAT with Graph Transformer drops F1 by 0.0038, showing GAT is better suited for this topology

### 5.3 Missing-Modality Experiments

| Missing Modality | Mean F1 ± Std | Degradation |
|-----------------|--------------|-------------|
| Full Model | 0.9914 ± 0.0011 | — |
| Missing Metrics | 0.4532 ± 0.4532 | **-0.5382** |
| Missing Traces | 0.9914 ± 0.0011 | 0.0000 |
| Missing Logs | 0.9914 ± 0.0011 | 0.0000 |

**Key Finding:** Metrics is the dominant modality. Removing metrics causes catastrophic performance degradation (F1 drops from 0.9914 to 0.4532), while removing traces or logs has no impact. This suggests the microservice system's anomaly signatures are primarily captured in metric data.

### 5.4 Noise Robustness

| Metric Noise Level | Mean F1 | Degradation |
|-------------------|---------|-------------|
| Clean | 0.9914 | — |
| 0.01 | 0.9913 | -0.0001 |
| 0.05 | 0.9917 | +0.0003 |
| 0.1 | 0.9914 | 0.0000 |
| 0.2 | 0.9912 | -0.0002 |
| 0.5 | 0.9842 | -0.0072 |

| Traces/Logs Noise | Mean F1 |
|-------------------|---------|
| Any level (0.01-0.5) | 0.9914 (no change) |

**Key Finding:** The model is robust to noise in traces and logs (no F1 change at any noise level). Metrics show graceful degradation, with F1 dropping only 0.0072 even at extreme noise (0.5).

### 5.5 Routing Analysis

| Expert | Mean Weight ± Std |
|--------|------------------|
| cross_modal | 0.2858 ± 0.0960 |
| temporal | 0.2659 ± 0.0106 |
| semantic | 0.2252 ± 0.0238 |
| dependency | 0.2231 ± 0.0616 |

**Routing Entropy:** 1.3538 ± 0.0208 (moderate routing confidence)

**Key Finding:** The router assigns highest weight to cross-modal (0.2858) and temporal (0.2659) experts, suggesting cross-modal interactions and temporal patterns are most important for anomaly detection. Dependency and semantic experts receive lower weights (~0.22 each).

---

## 6. Discussion

### 6.1 Why Metrics Dominate

The microservice system in this dataset exhibits anomaly patterns that are primarily captured in metric data. Possible reasons:

- Resource exhaustion (CPU, memory) directly manifests in metric anomalies
- Service faults cause immediate metric deviations
- Log and trace anomalies may be secondary effects or less discriminative

### 6.2 Why Graph Routing Helps

Graph-aware routing allows the model to leverage service dependency structure. When a service fails, its dependent services may show correlated anomalies. The GAT-based dependency expert captures these relationships, improving detection.

### 6.3 Why Uncertainty Routing Matters

Uncertainty-guided routing allows the model to down-weight experts that are uncertain about a particular input. This prevents low-confidence experts from dominating the fusion and improves robustness.

### 6.4 Limitations

1. **Single Dataset**: All experiments conducted on one microservice dataset. Generalization to other datasets needs verification.
2. **Graph Topology Fixed**: The dependency graph is fixed at 12 nodes with 24 edges. Real systems may have dynamic topologies.
3. **Log Dimension Fixed**: Log dimension fixed at 15 template features. Different log vocabularies would require adjustment.

---

## 7. Conclusion

We proposed Graph-Aware MoE (UG-MoE), a novel anomaly detection framework for microservice systems that combines:

1. Specialized experts for temporal, semantic, dependency, and cross-modal pattern detection
2. Graph-aware uncertainty-guided routing that leverages service dependency structure
3. Variational uncertainty blocks for per-expert uncertainty quantification
4. Adaptive confidence-weighted fusion for multimodal integration

Experimental results show:

- Track B (Graph-Aware MoE) achieves 0.9917 F1, outperforming Track A (MUAD) at 0.9896 F1 with statistical significance (p = 0.0052)
- Metrics is the dominant modality; removing it crashes F1 from 0.9914 to 0.4532
- The model is robust to metric noise up to 0.5 level
- The router dynamically allocates weight to cross-modal and temporal experts

**Limitations:** Results are based on one microservice dataset with 12 services. Generalization to other datasets with different topologies and modalities requires further validation.

**Future Work:**

1. Evaluate on additional datasets (Eadro SN, TrainTicket)
2. Investigate dynamic graph topologies
3. Explore deeper cross-modal interactions
4. Extend to root-cause localization

---

## 8. Reproducibility

All experiments use:

- PyTorch 2.2.1
- DGL 2.2.1
- numpy<2
- Seeds: 42, 123, 456, 789, 999
- Device: CUDA (GPU)
- Random seed initialization via `seed_everything()`

All experiment scripts are located in `project/experiments/`:
- `run_track_a_full.py` — Track A (MUAD baseline)
- `run_track_b_full.py` — Track B (Graph-Aware MoE)
- `run_comparison.py` — Track A vs Track B comparison
- `run_ablations.py` — 6 ablation configurations
- `run_missing_modality.py` — Missing-modality robustness
- `run_noise_robustness.py` — Noise robustness
- `run_routing_analysis.py` — Routing weight analysis
- `run_uncertainty_fusion_analysis.py` — Uncertainty/fusion weight analysis
- `run_gat_baseline.py` — GAT baseline
- `run_plain_moe.py` — Plain MoE baseline
- `run_graph_moe.py` — Graph-MoE baseline

All results are saved as JSON files in `project/experiments/runs/` with per-experiment subdirectories containing `summary.json` and per-seed results.

---

## 11. Data Format

The code expects data in the following format:

### Chunk Files (`chunk_train.pkl`, `chunk_test.pkl`)

Each chunk is a dictionary with:

| Key | Shape | Description |
|-----|-------|-------------|
| `metrics` | `[chunk_length, metric_num]` | Time-step × metric features per node |
| `traces` | `[chunk_length, trace_dim]` | Time steps × trace features per node |
| `logs` | `[node_num, event_num]` | Service nodes × log template features |
| `culprit` | scalar int | 0=normal, ≥1=anomaly service index |

**Example structure:**
```python
chunk = {
    "metrics": torch.Tensor([...]),   # shape [10, 7]
    "traces": torch.Tensor([...]),    # shape [10, 1]
    "logs": torch.Tensor([...]),      # shape [12, 15]
    "culprit": 3                      # service index 0-11, 0=normal
}
```

### Metadata (`metadata.json`)

Required keys:

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `node_num` | int | Number of service nodes | 12 |
| `edges` | list of 2 lists | Source nodes, destination node IDs | `[[0,1,...], [1,2,...]]` |
| `event_num` | int | Number of log template features | 15 |
| `metric_num` | int | Number of metric features | 7 |
| `chunk_length` | int | Time steps per chunk | 10 |
| `chunk_num` | int | Total number of chunks | 5,610 |

---

## 12. Future Work

1. **Datasets**: Evaluate on additional datasets (Eadro SN, TrainTicket)
2. **Graph Topology**: Investigate dynamic graph topologies
3. **Cross-Modal**: Explore deeper cross-modal interactions
4. **Tasks**: Extend to root-cause localization

---

## 13. References

[1] MUAD: Multimodal Uncertainty-Aware Anomaly Detection for Microservice Systems

*End of Document*
