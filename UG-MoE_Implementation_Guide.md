# UG-MoE Implementation Guide

**Uncertainty-Guided Graph-Aware Mixture-of-Experts for Multimodal Microservice Anomaly Detection**

**Scope:** Logs + Traces + Metrics -> Normal / Anomalous

> **Important scope:** This implementation is only for multimodal anomaly detection. Do not implement root-cause analysis, root-cause ranking, failure attribution, or counterfactual RCA.

---

## 1. What You Have to Build

Build a model that uses three observability sources together: logs, traces, and metrics. The main research change is to replace the base paper's GAT-based representation with a Graph-Aware Mixture-of-Experts (MoE) representation. The model must finally predict only **Normal** or **Anomalous**.

### Figure 1. Step-by-step implementation workflow

```
1. Collect Data            Logs + Traces + Metrics
        |
2. Preprocess              Parse logs, clean metrics, parse traces
        |
3. Temporal Alignment      Group observations by service + time window
        |
4. Build Service Graph     Node = service; Edge = dependency/call relation
        |
5. Encode Modalities       Log encoder | Trace encoder | Metric encoder
        |
6. Graph-Aware MoE         Router + Temporal + Semantic + Dependency + Cross-Modal experts
        |
7. Uncertainty Modeling    Estimate mu, sigma; sample probabilistic representation
        |
8. Multimodal Fusion       Learn weights for logs, traces, metrics
        |
9. Anomaly Detection       Predict Normal / Anomalous
        |
10. Evaluation             Baselines + ablation + robustness + sensitivity
```

---

## 2. Base Paper and Our Change

| Component | Base paper (MUAD) | This implementation |
|---|---|---|
| Input | Logs + traces + metrics | Logs + traces + metrics |
| Unimodal encoding | Dedicated encoders | Dedicated encoders |
| Graph representation | GAT inside GPE | Graph-Aware MoE |
| Experts | Not explicit | Temporal, Semantic, Dependency, Cross-Modal |
| Routing | Graph attention | Learned MoE router |
| Uncertainty | Gaussian representation | Gaussian representation + uncertainty-guided routing |
| Fusion | Confidence-aware fusion | Uncertainty-aware adaptive fusion |
| Output | Normal / anomalous | Normal / anomalous |

The base paper uses logs, traces, and metrics; a GAT-based Graph-based Probabilistic Encoder; confidence-aware fusion; and active learning. The proposed implementation keeps the anomaly-detection task and uncertainty idea, but the main architectural replacement is **GAT -> Graph-Aware MoE**.

---

## 3. Dataset to Use

### Primary dataset: Eadro multimodal dataset - SocialNetwork (SN)

Start with the public SN dataset from the Eadro project. The Zenodo record provides an `SN Dataset.zip` and a `TT Dataset.zip`, and the dataset contains multimodal observability data used for anomaly detection research.

- Dataset page: https://zenodo.org/records/7615394
- DOI: 10.5281/zenodo.7615394

Use SN first because it is directly available as a public download. After the complete pipeline works, repeat the same experiments on the TT dataset from the same Zenodo record for cross-dataset validation.

- TT dataset (same record): https://zenodo.org/records/7615394

### Optional secondary dataset: TrainTicket multimodal dataset

A separate public TrainTicket dataset on Zenodo contains monitoring data with logs, Jaeger traces, and Prometheus KPI data. It can be used as an additional validation dataset if required.

- Dataset page: https://zenodo.org/records/6979726
- DOI: 10.5281/zenodo.6979726

---

## 4. Software and Environment

- Python 3.10 or 3.11
- PyTorch
- PyTorch Geometric (for graph representation)
- Pandas and NumPy
- scikit-learn
- Matplotlib / Seaborn for analysis
- tqdm
- Jupyter Notebook or VS Code

GPU is strongly recommended because the model contains multiple neural experts. For the initial debugging stage, a CPU run on a small subset is sufficient.

---

## 5. Recommended Project Structure

```
UG_MoE/
├── datasets/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── models/
│   ├── encoders.py
│   ├── experts.py
│   ├── moe_router.py
│   ├── uncertainty.py
│   ├── fusion.py
│   └── model.py
├── scripts/
│   ├── preprocess_logs.py
│   ├── preprocess_traces.py
│   ├── preprocess_metrics.py
│   ├── build_graph.py
│   ├── train.py
│   └── evaluate.py
├── configs/
├── checkpoints/
├── results/
└── notebooks/
```

---

## 6. Step-by-Step Implementation

### Step 1 - Download and inspect the dataset
Download the `SN Dataset.zip` from the Zenodo page. Unzip it and identify the log, trace, metric, and label files. Do not start model training until you understand which file corresponds to which modality and how anomaly labels are represented.

### Step 2 - Create a common time window
Choose one observation window length, for example 10 seconds initially. All data belonging to the same service and time window should become one multimodal sample. Store a unique `sample_id`, `timestamp/window_id`, and `service_id`.

### Step 3 - Prepare logs
Convert raw log messages into structured events. At minimum keep timestamp, service, event/template ID, and message-derived features. Start with the structured log information available in the dataset. Later, a stronger log encoder can be added if needed.

### Step 4 - Prepare traces
For each trace/span, keep `trace_id`, `span_id`, service, start time, duration/latency, parent-child relation, and status/error information where available. Aggregate these features into the selected time window.

### Step 5 - Prepare metrics
For each service and time window, build a vector of numerical metrics such as CPU, memory, latency, network, and other available KPIs. Normalize using statistics computed **only from the training set**.

### Step 6 - Build the service dependency graph
Create one graph G=(V,E). A node represents a microservice. An edge represents a service-to-service dependency/call relationship. Use trace/service relationship information when available. Save the graph in a reusable format so that every sample can reference it.

### Step 7 - Implement the baseline first
Implement a simple baseline using the prepared multimodal features and a GAT-based graph encoder. This baseline is required because the research claim is a replacement of the GAT-centered representation.

### Step 8 - Implement the four experts
Create four small neural modules: Temporal Expert, Semantic Expert, Dependency Expert, and Cross-Modal Expert. Keep the first version simple. The objective is to have four independently trainable transformations, not four huge models.

### Step 9 - Implement the MoE router
The router receives the sample representation and graph information and outputs K expert weights. Use softmax so the weights sum to 1. Compute the mixture as `H = sum(g_k * E_k)`. Log the expert weights during validation so they can later be visualized.

### Step 10 - Add uncertainty modeling
Map the MoE output to a mean vector and variance vector: `mu` and `sigma^2`. Use the reparameterization trick `z = mu + sigma * epsilon`. Add KL regularization so the variance does not collapse.

### Step 11 - Add uncertainty-guided routing
Extend the router so that uncertainty is also an input to the routing network. Compare this against routing without uncertainty. This comparison is essential for validating the proposed novelty.

### Step 12 - Add multimodal fusion
Produce log, trace, and metric latent representations and learn modality weights. Fuse them as `z_f = w_log * z_log + w_trace * z_trace + w_metric * z_metric`.

### Step 13 - Add anomaly classifier
Send `z_f` to a small classification head and predict Normal vs Anomalous. Use binary cross-entropy or two-class cross-entropy depending on the implementation.

### Step 14 - Train carefully
Split the data into train/validation/test sets without leaking future information. Fit normalization and any learned preprocessing only on the training set. Save the best checkpoint based on validation F1 or PR-AUC.

### Step 15 - Run required experiments
After the full model works, run the baseline, MoE, Graph-MoE, uncertainty ablations, modality ablations, missing-modality tests, noise tests, and sensitivity experiments listed below.

---

## 7. Model Components and Losses

A practical first implementation can use the following overall objective:

```
L_total = L_anomaly + lambda_KL * L_KL
```

Where `L_anomaly` is the anomaly classification loss and `L_KL` regularizes the probabilistic representation. If an additional routing regularizer is introduced later, report it separately and justify its need.

---

## 8. Required Experiments

- **A. Baseline comparison:** GAT baseline; Plain MoE; Graph-MoE; Full proposed model.
- **B. Component ablation:** w/o graph; w/o MoE; w/o dynamic routing; w/o uncertainty routing; w/o probabilistic representation; w/o adaptive fusion.
- **C. Modality ablation:** Log only; Trace only; Metric only; Log+Trace; Log+Metric; Trace+Metric; all three.
- **D. Missing modality:** Randomly mask 10%, 20%, 40%, and 60% of multimodal observations. Also test missing one specific modality at a time.
- **E. Noise robustness:** Add controlled noise to logs, traces, metrics, and combinations of modalities.
- **F. Expert-routing analysis:** Plot expert weights for different anomaly types and compute routing entropy.
- **G. Uncertainty analysis:** Compare uncertainty values for clean versus noisy samples and inspect fusion weights.
- **H. Hyperparameter sensitivity:** Vary number of experts, embedding dimension, window size, uncertainty-loss weight, routing temperature if used, and learning rate.

---

## 9. Evaluation Metrics

For the final anomaly detection evaluation, report at minimum:

- Precision
- Recall
- F1-score
- PR-AUC
- ROC-AUC (optional, as a secondary metric)
- Accuracy (only as a supplementary metric if class imbalance is not misleading)

For robustness experiments, also report **relative F1 degradation** from the clean-data baseline.

---

## 10. Required Outputs

- [ ] Preprocessed multimodal dataset with a documented schema.
- [ ] Service dependency graph file.
- [ ] GAT baseline checkpoint.
- [ ] Plain MoE checkpoint.
- [ ] Graph-MoE checkpoint.
- [ ] Full proposed model checkpoint.
- [ ] Overall performance table.
- [ ] Ablation table.
- [ ] Missing/noisy modality robustness plots.
- [ ] Expert-weight heatmaps.
- [ ] Uncertainty/fusion-weight plots.
- [ ] Hyperparameter sensitivity plots.
- [ ] A reproducible evaluation script that regenerates all major tables and plots.

---

## 11. High-Level Pseudocode

```
Input: Logs L, Traces T, Metrics M

1. Preprocess L, T, M
2. Align by service and time window
3. Build service graph G
4. hL = LogEncoder(L)
5. hT = TraceEncoder(T)
6. hM = MetricEncoder(M)
7. For each modality v:
     g_v = Router(h_v, G, uncertainty_v)
     e_v = sum_k g_v[k] * Expert_k(h_v, G)
     mu_v, sigma_v = ProbabilisticHead(e_v)
     z_v = mu_v + sigma_v * epsilon
8. Estimate modality weights wL, wT, wM
9. z = wL*zL + wT*zT + wM*zM
10. y_hat = AnomalyClassifier(z)
11. Optimize classification + uncertainty regularization
12. Evaluate on held-out test data
```

---

## 12. Implementation Order (Do Not Skip Steps)

1. First: load and visualize the raw multimodal data.
2. Second: create aligned samples and verify labels.
3. Third: build and validate the service graph.
4. Fourth: train a simple baseline classifier.
5. Fifth: implement the GAT baseline.
6. Sixth: implement plain MoE without uncertainty.
7. Seventh: add graph-aware routing.
8. Eighth: add probabilistic uncertainty.
9. Ninth: add uncertainty-guided routing.
10. Tenth: add adaptive multimodal fusion.
11. Eleventh: freeze the implementation and run all ablations.
12. Twelfth: run robustness and sensitivity experiments.

---

## 13. Common Mistakes to Avoid

- Do not mix future test information into training normalization or feature construction.
- Do not randomly split individual rows if multiple rows belong to the same time window; split at the sample/window level.
- Do not claim improvement until the same train/validation/test protocol is used for all methods.
- Do not compare models using different feature sets unless that difference is explicitly part of the experiment.
- Do not start with a very large expert architecture; first make the smallest complete pipeline work.
- Do not add active learning to the first implementation. It is optional and should only be attempted after the main anomaly detector is stable.

---

## 14. Completion Criteria

- [ ] The pipeline runs end-to-end from dataset files to anomaly predictions.
- [ ] The GAT baseline is reproducible.
- [ ] Plain MoE and Graph-MoE versions run with the same evaluation protocol.
- [ ] The full uncertainty-guided model runs without numerical instability.
- [ ] All required ablations are completed.
- [ ] Missing-modality and noise experiments are completed.
- [ ] Hyperparameter sensitivity plots are generated.
- [ ] All final metrics are reproducible from saved checkpoints and scripts.

---

## 15. Data and Reference Sources

- Eadro multimodal dataset (SN + TT), Zenodo: https://zenodo.org/records/7615394
- TrainTicket multimodal anomaly dataset, Zenodo: https://zenodo.org/records/6979726
- TrainTicket benchmark system, GitHub: https://github.com/FudanSELab/train-ticket

**Dataset note:** The Eadro Zenodo record provides an `SN Dataset.zip` and `TT Dataset.zip` and states that the data were used for the Eadro multimodal microservice troubleshooting study. The TrainTicket Zenodo record separately documents logs, Jaeger traces, and Prometheus KPI data.

**Base paper used for methodology reference:** H. Dou et al., "Uncertainty-Aware Multimodal Anomaly Detection for Microservice Systems With Active Learning," IEEE Transactions on Services Computing, 2026, DOI 10.1109/TSC.2026.3672587.
