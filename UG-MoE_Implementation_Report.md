# UG-MoE Implementation Report

**Uncertainty-Guided Graph-Aware Mixture-of-Experts for Multimodal Microservice Anomaly Detection:
what we built, how, why, and what it shows**

**Scope:** logs + traces + metrics → Normal / Anomalous, the same scope as the implementation guide
(`7. UG-MoE_Implementation_Guide.md`). No root-cause analysis.

**Status (2026-11-18):** all experiments required by the guide (§8 A–H) are implemented and have been
run. The main comparison, component ablations and robustness have **5 seeds** with paired t-tests
(`make_report.py`, `runs/report_20261118_194054`). Modality ablation, routing and uncertainty analysis,
and sensitivity are single-seed.

**How to read this document.** It follows the guide's structure section by section. For every step it
says **what the guide asks**, **what we did**, **how** (with the file that does it) and **which concepts
are involved**. Section 10 explains every concept from first principles. Numbers come from runs on the
GPU server; the run directories are named so every number can be checked. The day-by-day log of every
round is in `PROJECT_STATUS.md`; the argument against MUAD's reported numbers is in `MUAD_vs_UG-MoE.md`.

---

## Contents
1. What we built
2. Base paper and our change
3. Datasets
4. Software and environment
5. Project structure
6. Step-by-step implementation (guide §6, steps 1–15)
7. Model components and losses
8. Evaluation protocol
9. Experiments and results (guide §8 A–H)
10. Concepts explained
11. Decisions, deviations and corrections, in order
12. Limitations and threats to validity
13. How to reproduce everything
14. Status against the guide's completion criteria
15. File reference

---

## 1. What we built

A model that reads three observability sources of a microservice system per **10-second window**:
- **metrics:** 7 container metrics per service per second;
- **traces:** span latency per service per second;
- **logs:** log-template counts per service.

It outputs whether the window is **normal or anomalous**. The guide's central change, replacing
MUAD's GAT-based representation with a **Graph-Aware Mixture-of-Experts**, is implemented together
with the two uncertainty mechanisms the guide adds: **uncertainty-guided routing** and
**uncertainty-aware adaptive fusion**.

The workflow as implemented (guide Figure 1, with our files):

```
1. Collect data        Zenodo 10.5281/zenodo.7615394, SN + TT, MD5-checked      data_pipeline/fetch_eadro.py
2. Preprocess          metrics (raw), traces (span latency), logs (Drain3)      data_pipeline/build_eadro.py
3. Temporal alignment  1-s grid per run, per-run clock-offset detection,
                       10-s windows, labels from the fault-injection files      data_pipeline/build_eadro.py
4. Service graph       Eadro's dependency graph (SN 12/24, TT 27/44)            data_pipeline/systems.py
5. Encode modalities   per-modality linear encoders + modality embedding        tracks/proposed_model/ug_moe_v2.py
6. Graph-aware MoE     4 shared experts, one router per modality                ug_moe_v2.py
7. Uncertainty         per-expert mean + log-variance, mixture moments, z ~ N   ug_moe_v2.py
8. Fusion              softmax weights over modalities from [mu, uncertainty]   ug_moe_v2.py
9. Detection           MLP classifier, 2 classes                                ug_moe_v2.py
10. Evaluation         leak-free split, val selection, A-H, report             experiments/*
```

Two versions of the proposed model exist and both are kept:
- **UG-MoE v2** (`UGMoEv2`, 166k parameters): a rewrite that follows the guide's pseudocode (§11)
  line by line. **This is the model all final results refer to.**
- **Legacy UG-MoE** (`GraphAwareMoE`, 4.4M parameters): the earlier implementation. It is kept to
  reproduce earlier results. Its defects, found in our audit, are listed in §7.4 and are why v2 was
  written.

---

## 2. Base paper and our change

| Component | MUAD (base paper) | Guide asks for | What we implemented (v2) |
|---|---|---|---|
| Input | logs + traces + metrics | same | same; logs as Drain3 template counts, traces as span latency |
| Unimodal encoding | dedicated encoders | dedicated encoders | one linear encoder per modality + a learned modality embedding |
| Graph representation | GAT inside the Graph-based Probabilistic Encoder (GPE) | Graph-Aware MoE | 4 experts shared by all modalities, one of them a GATv2 over the service graph; the router also sees a graph-convolution context vector |
| Experts | none | Temporal, Semantic, Dependency, Cross-Modal | Temporal = GRU over time; Semantic = self-attention across services; Dependency = GATv2 over the dependency graph; Cross-Modal = attention from one modality to all three of the same service |
| Routing | graph attention | learned MoE router | one router per modality, softmax over experts; input = pooled features + graph context + each expert's predicted uncertainty |
| Uncertainty | Gaussian representation | Gaussian + uncertainty-guided routing | each expert predicts mean and log-variance; mixture moments give μ, σ² per modality; sampling in training, μ at test; correct KL to N(0, I) |
| Fusion | confidence-aware fusion (TCP) | uncertainty-aware adaptive fusion | softmax weights over the 3 modalities computed from [μ_v, mean log σ²_v] |
| Output | normal / anomalous | normal / anomalous | normal / anomalous |
| Training | active learning | active learning optional, not in the first version | fully supervised (MUAD's active-learning script is vendored but not used) |

What MUAD does and where its code goes wrong is in §7.5 and in `MUAD_vs_UG-MoE.md` §2.

---

## 3. Datasets

### 3.1 What the guide asks
Eadro SN (SocialNetwork) first, then TT (TrainTicket) from the same Zenodo record for cross-dataset
validation.

### 3.2 Raw data (Zenodo 10.5281/zenodo.7615394)

| Archive | Size | MD5 (verified) | Content |
|---|---|---|---|
| `SN Dataset.zip` | 78.8 MB | `4c523faf64770b1e9a709cdd610e07c4` | 4 fault-injection runs (~23.5 min each) + 3 fault-free runs |
| `TT Dataset.zip` | 48.9 MB | `2d77861e4a025781e598d51866fd6f4e` | 9 fault-injection runs (~1.8 h each) + 2 fault-free runs |

Each run folder contains:
- one metrics CSV per service;
- a `spans.json` file (traces);
- the log files;
- a fault file listing each injected fault: service, type, start time, duration.

There are three fault types: `cpu_load`, `network_delay` and `network_loss`. Each fault lasts about
two minutes.

### 3.3 What we found when inspecting the raw data
- **Clock offsets.** Span and log timestamps are shifted from the metric clock by whole hours (+8 h on
  SN, +7 or +8 h on TT, differing per run). Without correcting this, traces and logs would be matched
  to the wrong windows. We detect the offset per run (§6, step 2).
- **Faults suppress traffic.** In SN fault runs only 58–74% of seconds have any span, against 93–95% in
  fault-free runs. "Missing traces" is therefore itself a weak anomaly signal.
- **TT traces are very sparse.** They cover only 2–4% of seconds, and only about the first hour of
  each 1.8-hour run. TT is effectively a metrics + logs dataset.
- **The two TT fault-free runs** have identical request counts but different files (different MD5).
  This is consistent with a replayed load generator; they are not duplicates.

### 3.4 Processed datasets we built

| Dataset | Window / stride | Windows | Services / edges | Log templates (event_num) | Normal share |
|---|---|---|---|---|---|
| SN (`SN_w10_s1`) | 10 s / 1 s | 9,907 | 12 / 24 | 13 (12 templates + "other") | 53% |
| TT (`TT_w10_s5`) | 10 s / 5 s | 14,859 | 27 / 44 | 31 (30 + "other") | 34% |
| SN with 5-s windows (sensitivity) | 5 s / 1 s | 9,942 | 12 / 24 | 13 | 55% |
| SN with 20-s windows (sensitivity) | 20 s / 1 s | 9,837 | 12 / 24 | 13 | 50% |

TT uses stride 5 because its runs are about 4.5× longer than SN's, which keeps the dataset size and
training time comparable.

### 3.5 MUAD's own data (`recovered_artifact`) and why it is not the main benchmark
`project/recovered_artifact/` holds MUAD's processed SN chunks:
- 3,366 train / 2,244 test windows, `event_num` 15;
- only the 4 fault runs, no fault-free runs;
- 83% of windows anomalous.

It was built by Eadro's preprocessing with **stride-1 windows split at random** (the MUAD paper's
protocol: "We randomly selected 60% … for training … 40% for testing"). Consecutive windows share 9 of
10 seconds, so **83.9% of test windows have a near-identical training window** (§8.1, §10.6). Two
further problems:
- its log features are Hawkes intensities of size ~6·10⁻¹⁰, numerically invisible without
  normalization;
- it has no run or time information per window. We recovered it by chaining windows (§8.1).

We keep it only to reproduce the MUAD paper's protocol. All final results use the rebuilt SN and TT.

---

## 4. Software and environment

| Guide | Ours | Why |
|---|---|---|
| Python 3.10 / 3.11 | Python 3.11 (server, conda env `muad`) | — |
| PyTorch | PyTorch | — |
| PyTorch Geometric | **DGL** (Deep Graph Library) | The vendored MUAD and Eadro code is written in DGL. Using the same graph library for every model keeps the comparison identical down to the graph batching. GATv2Conv and GraphConv exist in DGL. |
| pandas, NumPy, scikit-learn, Matplotlib, tqdm | same, plus SciPy (t-test), Drain3 (log parsing) | — |

Work happens on a local Windows machine, where we only check that files compile and import, plus
light data inspection. Training runs on a shared GPU server. Two server-specific details matter:
- flash and memory-efficient scaled-dot-product attention are disabled (`common.configure_torch`),
  because they produced NaN gradients in Transformer attention;
- stdout is line-buffered, so `nohup` logs show progress live.

---

## 5. Project structure

```
impl/
├── 7. UG-MoE_Implementation_Guide.md         the plan (supervisor)
├── 7. Uncertainty-Aware_...Active_Learning.pdf/.md   MUAD paper
├── UG-MoE_Implementation_Report.md           this document
├── PROJECT_STATUS.md                         log of every round, issues, decisions
├── MUAD_vs_UG-MoE.md                         evidence brief for the supervisor
└── project/
    ├── data_pipeline/       systems.py (services, graph, metric names), fetch_eadro.py, build_eadro.py
    ├── datasets/eadro/      raw/ and processed/ (gitignored, rebuilt by the pipeline)
    ├── recovered_artifact/  MUAD's original chunks
    ├── tracks/
    │   ├── proposed_model/  ug_moe_v2.py (UG-MoE v2), model.py (legacy GraphAwareMoE)
    │   └── baselines/       gat_baseline.py, graph_moe.py
    ├── src/                 legacy model components: experts/, routing/, uncertainty/, fusion/
    ├── experiments/
    │   ├── common.py        shared protocol: loading, splits, normalization, training, metrics
    │   ├── registry.py      every model / variant / ablation by name
    │   ├── run_baselines.py   train + evaluate any list of models (A, B, C, H)
    │   ├── run_comparison.py  MUAD vs UG-MoE with paired t-test
    │   ├── run_ablations.py   legacy-model ablations
    │   ├── run_robustness.py  D + E from a checkpoint
    │   ├── run_analysis.py    F + G from a checkpoint
    │   ├── make_report.py     every table and figure from saved runs
    │   └── runs/              all outputs
    └── vendor/              muad/ (baseline), eadro/ (preprocessing reference)
```

This differs from the guide's suggested layout (§5) in naming only. Every role in the guide's tree has
a counterpart above: encoders, experts, router, uncertainty and fusion are inside `ug_moe_v2.py`;
preprocessing is in `data_pipeline/`; training and evaluation are in `experiments/`.

---

## 6. Step-by-step implementation (guide §6)

### Step 1: Download and inspect the dataset
- **Guide:** download SN, identify the log, trace, metric and label files, and understand the labels
  before training.
- **What we did:**
  - `python -m data_pipeline.fetch_eadro` downloads both archives, checks the MD5 sums, extracts them
    safely (it rejects paths that would escape the target folder) and unpacks the nested archives of
    the fault-free runs into `datasets/eadro/raw/{SN,TT}`.
  - We then read the fault files and decoded the labels. Each fault has a target service, a type
    (cpu_load, network_delay, network_loss), a start time and a duration.
  - In MUAD's artifact, labels are encoded as `culprit` 0 / 1 / 4 / 5 (normal / cpu / delay / loss). We
    confirmed this by matching the label sequence along the recovered runs to the raw injection order.

### Step 2: Create a common time window
- **Guide:** one window length (10 s); one multimodal sample per service and window, with ids and
  timestamps.
- **What we did** (`build_eadro.py`):
  1. **Time grid:** integer seconds over the span that every service's metric file covers.
  2. **Clock-offset detection:** for spans and for logs separately, try every whole-hour shift from
     −14 to +14 h and keep the one that puts the most events inside the run's metric window.
     Out-of-window events are dropped. The detected offset and the share of events kept are written
     to `quality.json` for every run.
  3. **Windows:** length 10 s (`--window`), stride 1 s for SN and 5 s for TT (`--stride`).
  4. **Labels:** a window is anomalous if it overlaps an injected fault by at least 1 second (Eadro's
     rule, `--threshold`). If several faults overlap, the one with the largest overlap gives the fault
     type and the faulty service. `culprit` is 0 = normal, 1 = cpu_load, 2 = network_delay,
     3 = network_loss.
  5. **Per-window metadata:** each window (chunk) also stores its run, run index, run kind
     (fault / fault-free), start time `t0`, fault type, faulty service and `trace_frac` (share of
     seconds with any span). The id is `{SYS}-{run:02d}-{start:05d}`.
- **Sample structure:** the guide suggests one sample per (service, window). We use one sample per
  window that contains all services as a graph (nodes = services), which is what MUAD and Eadro do and
  what a graph model needs. The label is per window.

### Step 3: Prepare logs
- **Guide:** structured events with timestamp, service, template id and message features.
- **What we did:**
  1. **Parse timestamps** with the SN and TT log formats (two regexes).
  2. **Mask variables** before template mining: UUIDs, IP:port, long hex ids and numbers become
     `<UUID>`, `<IP>`, `<HEX>`, `<NUM>`.
  3. **Template mining with Drain3**, a fixed-depth parse tree that groups log lines into templates
     (§10.1). We keep the 30 most frequent templates plus one "other" bucket. SN has only 12 templates,
     giving 13 features; TT gives 31.
  4. **Features:** per window and service, the count of each template, then `log1p`.
- **Concept:** a template turns free text into a categorical event type
  (`"Connection to <IP> failed after <NUM> ms"`). Counting templates per window is the standard
  log-anomaly representation (as in DeepLog-style pipelines).
- **Note:** MUAD used Hawkes-process intensities instead of counts (§3.5). Counts are simpler, have a
  sensible scale and need no fitted process. Templates are mined over all runs; this is label-free but
  does use test log text (listed in §12).

### Step 4: Prepare traces
- **Guide:** keep trace and span ids, service, start time, duration, parent / child and status; aggregate
  per window.
- **What we did:** each span is assigned to the service that served it and to its second (after the
  clock-offset correction). The per-second feature is `log1p(mean span latency in ms)`, and 0 when the
  service had no span in that second, as in Eadro.
- **Why this aggregate:** it is what MUAD and Eadro use, so the comparison stays like for like. The
  dependency structure that parent / child relations would provide enters through the service graph
  (step 6). The dataset has no reliable error / status field to use.
- **Shape:** traces are a per-service time series `[services, 10 s, 1]`.

### Step 5: Prepare metrics
- **Guide:** per service and window, a vector of metrics, normalized with **training-set statistics
  only**.
- **What we did:**
  - The 7 container metrics per service per second are `cpu_usage_system`, `cpu_usage_total`,
    `cpu_usage_user`, `memory_usage`, `memory_working_set`, `rx_bytes` and `tx_bytes`.
  - Gaps are forward-filled, then back-filled; values are stored **raw**.
  - Normalization happens in the experiment loader (`common.fit_normalizer`): a per-channel z-score
    for every modality, with mean and std computed **on the training split only**, then applied to
    train, validation and test. The fitted statistics are saved to `normalization.json` in every run
    directory.
- **Why per channel and per modality:** without it, MUAD's log features (~6·10⁻¹⁰) were invisible to
  every network. That was the root cause of "traces and logs have no effect" in early results (§11).

### Step 6: Build the service dependency graph
- **Guide:** one graph, nodes = services, edges = dependencies, saved for reuse.
- **What we did:** `data_pipeline/systems.py` takes the service lists and the dependency edges from the
  vendored Eadro code (`vendor/eadro/codes/preprocess/util.py`), the graph the dataset's authors
  published:
  - SN: **12 services / 24 edges**, identical to MUAD's artifact;
  - TT: **27 services / 44 edges**.

  The graph is stored in every dataset's `metadata.json`. Each window becomes one DGL graph with the
  per-service features as node data, and windows are batched as DGL graph batches.
- **Size-agnostic models:** the legacy code assumed 12 nodes and 24 edges in several places. All models
  now read the node and edge counts from the data, which made TT possible.

### Step 7: Implement the baseline first
- **Guide:** a simple baseline with a GAT-based graph encoder, because the claim is replacing GAT.
- **What we did:**
  - **GAT baseline** (`tracks/baselines/gat_baseline.py`): per-service features of all three
    modalities, concatenated, go through 2 graph-attention layers (4 heads, hidden 64), are pooled and
    classified.
  - **MUAD** (`vendor/muad`): the full base-paper model, trained on its own loss and used unchanged.
    It is the main baseline (Track A).
  - **Graph-MoE baseline** (`tracks/baselines/graph_moe.py`): an MoE over GAT experts. The guide's
    "Plain MoE" and "Graph-MoE" baselines are also realized as the v2 steps `plain` and `graph`
    (step 15).

### Step 8: Implement the four experts
- **Guide:** Temporal, Semantic, Dependency and Cross-Modal experts; small, independently trainable.
- **What we did** (`ug_moe_v2.py`, hidden size D = 64). Each expert takes one modality's encoded data
  and returns one vector per service:

| Expert | Module | What it captures | Input |
|---|---|---|---|
| Temporal | GRU over the 10 time steps of each service, last hidden state | how a service's signal evolves in the window | `[B·N, T, D]` |
| Semantic | Transformer encoder layer (4 heads, pre-norm) across the N services | content-based relations between services, regardless of the graph | `[B, N, D]` |
| Dependency | GATv2 graph attention over the service graph (4 heads, residual) + LayerNorm | propagation along real call dependencies | `[B·N, D]` + graph |
| Cross-modal | multi-head attention from this modality's service vector to the 3 modality vectors of the same service | agreement or conflict between metrics, traces and logs of one service | `[B·N, 1, D]` → `[B·N, 3, D]` |

- **Sharing:** the experts are **shared by all three modalities**. A learned modality embedding is added
  to the encoded input, so the same expert knows which modality it is processing. This keeps the model
  small: 4 experts instead of 12.
- **Output heads:** each expert's per-service output is pooled over services (mean and max,
  concatenated, then a linear layer). Two linear heads then predict the **mean** `μ_k` and
  **log-variance** `s_k` of that expert's representation (step 10).

### Step 9: Implement the MoE router
- **Guide:** the router gets the sample representation and graph information and outputs K weights
  through a softmax; `H = Σ g_k E_k`; log the weights.
- **What we did:**
  - There is **one router per modality**: a 2-layer MLP (GELU, dropout).
  - Its input is the concatenation of the pooled modality features and the **graph context**: a
    GraphConv over the service graph applied to the modality's node features, ELU, pooled.
  - The output is `g_v = softmax(router_v(input) / temperature)`, K weights summing to 1.
  - The mixture is formed on the experts' distributions (step 10), so `H = Σ g_k E_k` holds for the
    means.
  - Routing weights are returned by `forward(..., return_details=True)` and analysed in Exp. F.
- **Optional load balancing:** `lb_weight` × K · Σ (mean g_k)² − 1 penalizes using few experts. It is
  0 by default; the guide says to report any routing regularizer separately, and we did not need one.

### Step 10: Add uncertainty modeling
- **Guide:** map the MoE output to μ and σ², use z = μ + σ·ε, add a KL term so the variance does not
  collapse.
- **What we did:**
  - Each expert k predicts `μ_k` and `s_k = log σ²_k`, with `s_k` clamped to [−8, 8].
  - The routed mixture of Gaussians is summarized by its **first two moments** (§10.4):
    - `μ_v = Σ_k g_k μ_k`
    - `σ²_v = Σ_k g_k (σ²_k + μ_k²) − μ_v²`, clamped ≥ 10⁻⁶.
  - **Training:** `z_v = μ_v + σ_v · ε`, with ε ~ N(0, I) (reparameterization trick, §10.3).
    **Evaluation:** `z_v = μ_v`, which makes predictions deterministic.
  - **KL term:** `KL(N(μ_v, σ²_v) ‖ N(0, I)) = ½ Σ (μ² + σ² − 1 − log σ²)`, averaged over modalities and
    dimensions.
- **Contrast with MUAD:**
  - MUAD samples even at test time, so its predictions are random.
  - Its KL formula puts the variance where the log-variance belongs, which pushes the variance to 0
    instead of 1 (§7.5).

### Step 11: Add uncertainty-guided routing
- **Guide:** the router also receives uncertainty; compare with routing without it.
- **What we did:** each modality's router input also gets `u_v[k] = mean(s_k)`, each expert's
  predicted log-variance averaged over dimensions. So the router can move weight away from an expert
  that is uncertain about this window.
- **Comparison:** this is the ablation `w/o uncertainty routing`, and the incremental step from `prob`
  to `ug_routing` (§9).

### Step 12: Add multimodal fusion
- **Guide:** learn modality weights; `z_f = w_log z_log + w_trace z_trace + w_metric z_metric`.
- **What we did:**
  - A shared fusion network `fusion(·)` (Linear → GELU → Linear → 1 score) is applied to each
    modality's feature `[μ_v, mean log σ²_v]`.
  - Then `w = softmax over the 3 modalities`, and `z_f = Σ_v w_v z_v`.
  - Because the predicted uncertainty is an input, a modality that looks uncertain can receive a
    lower weight. This is the "uncertainty-aware" part.
- **Without adaptive fusion** (ablation): `w_v = 1/3`.

### Step 13: Add the anomaly classifier
- **Guide:** a small classification head; binary or two-class cross-entropy.
- **What we did:** Linear → GELU → Dropout → Linear to 2 logits, trained with two-class cross-entropy on
  `y = 1[culprit ≥ 1]`. Predictions are the arg-max, i.e. a 0.5 probability threshold.

### Step 14: Train carefully
- **Guide:** a train / validation / test split without leaking future information; normalization only
  on train; best checkpoint by validation F1 or PR-AUC.
- **What we did** is described in full in §8. In short:
  - run-level chronological split;
  - normalization on train only;
  - checkpoint selected on **validation macro-F1** (tie-break PR-AUC), with early stopping;
  - test evaluated once;
  - every model trained identically.

  Training settings (all models): Adam, lr 5·10⁻⁴, weight decay 10⁻⁵, batch 50, gradient clipping
  0.5, at most 50 epochs, early-stopping patience 15.

### Step 15: Run the required experiments
All of guide §8 (A–H) has been run; results are in §9. Every model and variant is a name in
`experiments/registry.py`:

| Name | Meaning |
|---|---|
| `muad`, `gat`, `graph_moe` | baselines |
| `ug_moe` | legacy model |
| `ug_moe2:plain` | experts + learned router (no graph, no uncertainty); the guide's "Plain MoE" |
| `ug_moe2:graph` | + graph context and the dependency expert; the guide's "Graph-MoE" |
| `ug_moe2:prob` | + probabilistic representation |
| `ug_moe2:ug_routing` | + uncertainty-guided routing |
| `ug_moe2:full` | + adaptive fusion: **the proposed model** |
| `abl2:<name>` | full model minus one component (guide §8B) |
| `<model>@<modalities>` | trained and tested on a modality subset (guide §8C) |
| `ug_moe2:full#key=value` | full model with one hyperparameter changed (guide §8H) |

---

## 7. Model components and losses

### 7.1 Forward pass of UG-MoE v2 (the guide's §11 pseudocode, as implemented)

```
for each modality v in {metric, trace, log}:
    h_v        = Linear_v(x_v) + modality_embedding_v          # [B·N, T, D]; logs have T = 1
    for each expert k:
        e_k    = Expert_k(h_v, graph)                          # [B·N, D]
        p_k    = Pool(e_k)                                     # mean+max over services -> [B, D]
        mu_k   = MuHead_k(p_k);  s_k = clamp(LogVarHead_k(p_k), -8, 8)
    g_v        = softmax( Router_v([Pool(h_v), Pool(GraphConv(h_v)), mean(s_1..s_K)]) / tau )
    mu_v       = sum_k g_v[k] mu_k
    var_v      = sum_k g_v[k] (exp(s_k) + mu_k^2) - mu_v^2
    z_v        = mu_v + sqrt(var_v) * eps   (training)   |   mu_v   (evaluation)
w              = softmax_v( Fusion([mu_v, mean(log var_v)]) )
z              = sum_v w_v z_v
logits         = Classifier(z)
```

### 7.2 Loss

```
L_total = CE(logits, y) + lambda_KL * mean_v KL( N(mu_v, var_v) || N(0, I) )   [+ lb_weight * L_balance]
lambda_KL = 0.01, lb_weight = 0
```

This is exactly the guide's §7 objective. There are no hidden terms. That is unlike the legacy model,
which added an undocumented router loss, and unlike MUAD, which adds a confidence loss (§7.5).

### 7.3 Switches, incremental steps and ablations
Every component can be switched off on its own:
- `use_graph`: dependency expert and graph context;
- `use_moe`: off leaves a single temporal expert and no router;
- `dynamic_routing`: off gives uniform weights;
- `uncertainty_routing`;
- `probabilistic`: off gives z = μ, no variance, no KL;
- `adaptive_fusion`: off gives w = 1/3.

The guide's §12 incremental order and its §8B ablations are just combinations of these switches
(`INCREMENTAL_STEPS`, `ABLATIONS_V2`), so every comparison differs in exactly the named component.

| Default hyperparameter | Value |
|---|---|
| Hidden size D | 64 |
| Experts | 4 |
| Attention heads | 4 |
| Dropout | 0.1 |
| Routing temperature τ | 1.0 |
| λ_KL | 0.01 |
| Parameters (SN) | 166,031 |

### 7.4 The legacy model (`GraphAwareMoE`) and why v2 was written
Our audit of the earlier implementation (`PROJECT_STATUS.md` §3) found:
1. **The routed mixture never reached the classifier.** Routing weights scaled the expert outputs into
   a tensor that was computed and then not used; the logits came from a separate fusion of the raw
   expert outputs. Routing influenced training only through an auxiliary loss.
2. **No graph-aware routing in any experiment.** The only router that used the graph was switched off
   wherever it was selected.
3. **"Uncertainty-guided" routing did not use the model's uncertainty.** It used a batch-level scalar.
4. **The model was 18× larger than the baselines** (4.4M parameters).
5. **Ablation names did not match what they changed** (e.g. `no_graph` swapped the router type).

v2 fixes all five by construction, and is 25× smaller. The legacy model is kept for reproducibility.
On rebuilt SN it performs well (0.984 macro-F1, single seed), but on TT it never left the
"all anomalous" state.

### 7.5 What MUAD does, and its code defects
MUAD encodes each modality and passes it through an "uncertain block": an encoder, then sigmoid heads
for μ and σ², softmax-weighted attention, sampling, and KL. It then fuses the modalities with
**TCP confidence** (a learned true-class-probability estimate per modality that scales the features),
classifies, and trains on `CE + 0.6·confidence_loss + KL`.

Verified defects in the vendored code:
- **The best epoch is chosen on the test set** (`training/base_model.py`, lines 86–91).
- **It samples random noise at test time**, so predictions are non-deterministic
  (`models/main_model.py`, lines 28 and 31–34).
- **The KL term is the wrong formula.** It uses the variance where the log-variance belongs (lines
  36–37). With f(v) = 1 + v − eᵛ ≤ 0, the term is minimized at σ² = 0 instead of 1, so it pushes
  towards zero variance, the opposite of the intended prior.
- **The shipped log features are about 6·10⁻¹⁰**, so the logs have no effect on the model.

We use MUAD **unchanged**. It receives the same data, split, normalization and selection rule as every
other model. We also trained it with the paper's own hyperparameters (lr 10⁻³, 120 epochs): it scored
0.793 vs 0.922 with our settings on SN, so our settings favour MUAD.

---

## 8. Evaluation protocol (`experiments/common.py`)

### 8.1 Leak-free splits
- **Rebuilt SN and TT, chronological run-level split** (`assign_pipeline_runs`):
  1. Fault runs are sorted by start time. The last ⌈20%⌉ are test, the ⌈20%⌉ before them validation,
     the rest training. So training never sees later fault runs.
  2. Fault-free runs:
     - with at least 3 (SN), they are assigned whole, round-robin train → test → val;
     - with fewer (TT has 2), each is cut chronologically into 60 / 20 / 20% train / val / test
       segments, and windows straddling a cut are dropped (8 on TT). Every split must contain normal
       operation, otherwise validation becomes anomaly-dominated (§11, item 6).

| | Train | Validation | Test |
|---|---|---|---|
| SN | 2 fault + 1 fault-free run: 4,236 windows | 1 + 1: 2,834 | 1 + 1: 2,837 (41% anomalous) |
| TT | 5 fault runs + first 60% of both fault-free: 8,397 | 2 + 60–80% segments: 3,228 | 2 + last 20%: 3,226 (68% anomalous) |

- **MUAD's artifact, run-level split:** the artifact has no run ids, so we **reconstructed the 4 runs**
  (`reconstruct_runs`). Window t+1's first 9 metric steps equal window t's last 9, so the windows chain
  unambiguously into 4 runs of about 1,402 windows. Then train = runs 0–1, validation = run 2, test =
  run 3. `--split random` reproduces the paper's leaky split for comparison only.

### 8.2 Normalization
Per-channel z-score of all modalities, fitted on the training split only, then applied to every split
(saved to `normalization.json`). Constant channels are centred but not scaled.

### 8.3 Training and model selection (`train_and_select`)
- Every epoch is evaluated on the **validation** split.
- The kept checkpoint is the one with the best **validation macro-F1**, tie-break validation PR-AUC.
- Early stopping after 15 epochs without improvement.
- The best weights are restored, saved (`<model>_seed<N>_best.pth`) and evaluated **once** on test.
- Batches with non-finite loss are skipped and counted.

Why macro-F1 and not anomaly-F1: when validation is mostly anomalous, a model that calls everything
anomalous scores high anomaly-F1 and wins the selection (this happened, §11). Its macro-F1 is ≤ 0.5.
The guide allows "validation F1 or PR-AUC"; macro-F1 is an F1.

### 8.4 Metrics (`compute_metrics`)
- Anomaly-class F1, precision and recall, as the guide and the MUAD paper report.
- Normal-class F1.
- **Macro-F1** (mean of both classes' F1), our primary metric.
- PR-AUC (average precision) and ROC-AUC, from the anomaly probability.
- Per fault type: recall for each culprit class, plus the false-positive rate on normal windows.
- Every table includes a **trivial "all anomalous" row**, which shows how much F1 is free from class
  imbalance.
- Robustness also reports the relative drop `(clean − perturbed) / clean` (guide §9).

### 8.5 Seeds, determinism, significance
- **Exploration:** single seed 42 (time constraint).
- **Final runs:** seeds 42, 123, 456, 789 and 999, reported as mean ± std. We use paired t-tests over
  seeds against UG-MoE v2 (`make_report.py`); with 5 seeds these are indicative only.
- Training is deterministic for a given seed: two different configurations that happen to be the same
  model reproduced each other's result exactly (0.9459 on SN, 0.8399 on TT).

---

## 9. Experiments and results (guide §8)

All numbers are test **macro-F1** unless stated otherwise. "5 seeds" = mean over 42 / 123 / 456 / 789 /
999, ± the standard deviation over seeds. All tables come from `make_report.py`
(`runs/report_20261118_194054/tables.md`). p = paired t-test over the 5 seeds against UG-MoE v2 full; with
5 seeds the test has little power, so p is indicative.

### A. Baseline comparison (5 seeds)

| Model | Params | SN | p | TT | p |
|---|---|---|---|---|---|
| Trivial (all anomalous) | — | 0.289 | | 0.404 | |
| MUAD | 242k | 0.925 ± 0.005 | **0.002** | 0.626 ± 0.032 | **0.042** |
| GAT | 303k | 0.931 ± 0.035 | **0.024** | 0.661 ± 0.044 | 0.281 |
| Plain MoE (`ug_moe2:plain`) | 115k | 0.971 ± 0.017 | 0.948 | 0.698 ± 0.073 | 0.724 |
| Graph-MoE (`ug_moe2:graph`) | 144k | 0.960 ± 0.018 | 0.241 | 0.667 ± 0.063 | 0.446 |
| **UG-MoE v2 (full)** | 166k | **0.972 ± 0.013** | — | **0.723 ± 0.090** | — |

- **UG-MoE v2 is the best model on both datasets:** +4.6 / +9.7 points over MUAD (significant on both)
  and +4.1 / +6.3 over GAT (significant on SN).
- **Over its own plain variant:** +0.1 on SN, +2.5 on TT, p ≥ 0.7: a tie. TT varies strongly between
  seeds (std 0.09).
- **On TT the PR-AUC is not better** (0.917 vs MUAD 0.914, GAT 0.925). The macro-F1 lead comes from the
  operating point (more faults flagged), not from ranking windows better.
- **Which anomalies (recall per fault type, 5 seeds):**

  | | cpu_load | network_delay | network_loss | false alarms (normal) |
  |---|---|---|---|---|
  | SN: UG-MoE v2 | **0.984** | 0.954 | 0.891 | 0.7% |
  | SN: MUAD | 0.658 | 0.988 | 0.910 | 1.8% |
  | TT: UG-MoE v2 | 0.998 | **0.594** | **0.508** | 18.3% |
  | TT: MUAD | 0.914 | 0.331 | 0.273 | 10.9% |
  | TT: GAT | 0.876 | 0.461 | 0.353 | 11.9% |

  On SN, MUAD misses a third of the CPU faults, which is most of the gap. On TT, UG-MoE v2 catches about
  twice as many network faults as MUAD, at the cost of more false alarms. (Seed 42 alone gave 0.90 / 0.88
  at 30% false alarms; that seed was favourable.)
- **Reproducing MUAD's paper:** under the paper's own leaky protocol on its own data, our MUAD scores
  F1 0.988 (paper: 0.993). The gap to our leak-free numbers is the protocol (§3.5).

### B. Component ablation (5 seeds; change vs full)

| Removed | SN | Δ (p) | TT | Δ (p) |
|---|---|---|---|---|
| Graph | 0.978 ± 0.008 | +0.6 (0.210) | 0.741 ± 0.082 | +1.8 (0.777) |
| MoE (single temporal expert) | 0.965 ± 0.028 | −0.6 (0.752) | 0.724 ± 0.070 | +0.1 (0.968) |
| Dynamic routing (uniform weights) | 0.977 ± 0.004 | +0.5 (0.428) | 0.673 ± 0.030 | −5.0 (0.435) |
| Uncertainty-guided routing | 0.974 ± 0.015 | +0.2 (0.851) | 0.698 ± 0.071 | −2.6 (0.651) |
| Probabilistic representation | 0.980 ± 0.008 | +0.8 (0.361) | 0.704 ± 0.068 | −1.9 (0.440) |
| **Adaptive fusion** | **0.948 ± 0.012** | **−2.4 (0.061)** | **0.690 ± 0.132** | **−3.3 (0.684)** |

(Δ = ablation − full, so negative means the component helps.)

- **No component is significant on its own at 5 seeds.** Uncertainty-aware adaptive fusion comes closest:
  it is the only component whose removal hurts on both datasets, and on SN p = 0.061.
- On TT, dynamic routing and uncertainty routing point the right way, but within large seed variance.
- **Lesson:** the single-seed ablations (`PROJECT_STATUS.md`, rounds 9 and 15b) showed large effects
  that disappeared over 5 seeds. Examples: "w/o MoE" on SN went from −7.8 to −0.6, and "w/o graph" on
  TT from −18.8 to +1.8. Only multi-seed ablations should be cited.

### C. Modality ablation (SN, single seed; model trained on the subset)

| Modalities | Macro-F1 |
|---|---|
| metrics | 0.883 |
| traces | 0.935 |
| logs | 0.832 |
| traces + logs | 0.938 |
| metrics + logs | 0.959 |
| metrics + traces | 0.929 |
| **all three** | **0.987** |

Every modality carries real signal on its own, and fusing all three adds +2.8 over the best pair.

### D. Missing modality and E. Noise (5 seeds)
Saved checkpoints (one per seed) are perturbed only at test time, with the same random masks for every
model:
- D1: drop one modality entirely;
- D2: zero each (window, service, modality) observation with probability p;
- E: Gaussian noise with σ = level × training std per channel, on every modality combination.

Test macro-F1, 5-seed mean (`runs/final_rob_*`):

| Model | Clean | 20% missing | 40% missing | 60% missing | Noise on all, σ = 2 | No metrics |
|---|---|---|---|---|---|---|
| SN: MUAD | 0.925 | 0.850 | 0.710 | 0.543 | 0.735 | 0.289 |
| SN: v2 plain | 0.971 | 0.885 | 0.780 | 0.683 | 0.737 | 0.289 |
| SN: v2 full w/o probabilistic | 0.980 | 0.902 | 0.797 | 0.663 | 0.810 | 0.289 |
| **SN: UG-MoE v2 full** | 0.972 | **0.903** | **0.811** | **0.696** | **0.808** | 0.274 |
| TT: MUAD | 0.627 | **0.613** | **0.568** | **0.526** | **0.537** | 0.243 |
| TT: GAT | 0.661 | 0.605 | 0.529 | 0.452 | 0.523 | 0.404 |
| TT: UG-MoE v2 full | **0.723** | 0.573 | 0.506 | 0.471 | 0.470 | 0.406 |

- **On SN, UG-MoE v2 stays ahead of MUAD under every perturbation, and the gap grows** with the amount
  of corruption: +4.6 clean, +5.3 / +10.1 / +15.3 at 20 / 40 / 60% missing, +7.3 under strong noise.
- **The probabilistic representation does not explain this.** At seed 42 the model without it lost
  12.8 / 14.9 F1 points at 20 / 40% missing; over 5 seeds it is level with the full model (0.902 vs 0.903,
  0.797 vs 0.811). That seed-42 checkpoint was a fragile one. Full vs plain MoE: +1.8 / +3.1 under
  masking and +7.1 under strong noise, a trend that is not consistent across seeds.
- **Every model fails completely without metrics** (at or below the trivial score). No model sees a
  window without metrics in training, so none learns to fall back on traces and logs, even though a
  traces-only model reaches 0.935 (C). Modality dropout would address this; it was deferred.
- **On TT the clean lead does not survive perturbation.** From 20% missing on, UG-MoE v2 is below MUAD
  and GAT. Two of its five seeds (123, 456, the cautious ones with low network-fault recall) collapse to
  "everything is anomalous" at only 10% missing; the high-recall seeds (42, 789) degrade gradually. On
  TT, macro-F1 must be used: the trivial predictor has anomaly-F1 0.809, so a collapsing model's
  anomaly-F1 rises.

### F. Expert-routing analysis (single seed)
- **Routing specializes by fault type.**
  - On SN, metric-routing entropy is 0.41 (0 = one expert, 1 = uniform). Normal windows go mainly to
    the dependency (GAT) expert (0.87), CPU-load faults to the cross-modal expert (0.95), network faults
    mainly to the temporal expert (about 0.5).
  - On TT, CPU-load windows again get a dedicated metric expert (semantic, 0.996).
- **Traces route mostly to the cross-modal expert** (SN 0.67, TT 0.87), consistent with sparse traces
  drawing on the other modalities.
- **Logs** are routed almost uniformly on SN (entropy 0.97; they are barely used), and to the semantic
  expert on TT (0.79).

### G. Uncertainty and fusion analysis (single seed)
- **The direction is right.** A corrupted modality's predicted uncertainty rises and its fusion weight
  falls:
  - on SN, metric noise moves the metric log-variance from −0.22 to −0.07 and its weight from 0.96 to
    0.91;
  - on TT, dropping metrics moves the metric log-variance from −0.28 to +0.80 and its weight from 0.61
    to 0.13, with traces taking over at 0.79.
- **The size is too small on SN.** Metrics keep 88% of the weight even when they are missing.
- **Failure case (TT):** with noise on the logs, the model becomes *more* confident in the logs and
  raises their weight from 0.11 to 0.65, costing 8 F1 points. Learned variance can be overconfident on
  inputs unlike the training data.
- **Errors:** on SN the uncertainty is higher on wrong predictions (−0.04 vs −0.23); on TT it is not.

### H. Hyperparameter sensitivity (SN, single seed, default marked *)

| Parameter | Values → macro-F1 |
|---|---|
| Number of experts | 1 → 0.909 · 2 → 0.988 · 3 → 0.984 · **4* → 0.987** |
| Embedding dimension | 16 → 0.980 · 32 → 0.978 · **64* → 0.987** · 128 → 0.973 |
| KL weight λ | 0.001 → 0.979 · **0.01* → 0.987** · 0.1 → 0.953 · 1.0 → 0.972 |
| Routing temperature | 0.5 → 0.969 · **1.0* → 0.987** · 2.0 → 0.983 |
| Learning rate | 1e-4 → 0.982 · **5e-4* → 0.987** · 1e-3 → 0.985 |
| Window length (s) | 5 → 0.956 · **10* → 0.987** · 20 → 0.973 |

- The defaults are best or within 0.2 points everywhere. Only a single expert (−7.8) and λ = 0.1 (−3.4)
  hurt clearly; the rest is within single-seed noise.
- The defaults were fixed before these runs, not tuned on test.
- A 4× larger model (dimension 128) is not better, so capacity is not what drives the results. The
  planned shrinking check found dimension 32 0.9 points worse, so we keep 64.

---

## 10. Concepts explained

### 10.1 Log template mining (Drain)
Log lines are free text with variable parts. Drain groups lines into **templates** by passing each
line down a fixed-depth tree:
1. first by token count;
2. then by the leading tokens;
3. then by token-wise similarity to existing templates.

Variable tokens become wildcards. Pre-masking numbers, IPs and ids makes templates cleaner. A window
is then described by how often each template occurred: a fixed-length vector regardless of log volume.

### 10.2 Graph neural networks: GCN, GAT, GATv2
A graph layer updates each node from its neighbours:
- **GCN (GraphConv)** uses a fixed, degree-normalized average.
- **GAT** learns an attention weight per edge, `α_ij = softmax_j(LeakyReLU(aᵀ[W h_i ‖ W h_j]))`, so a
  node decides which neighbours matter.
- **GATv2** applies the non-linearity before the dot product, `aᵀ LeakyReLU(W[h_i ‖ h_j])`, which makes
  the attention truly depend on the query node ("dynamic attention"). Multiple heads are concatenated.

In anomaly detection, faults propagate along calls, so the dependency graph is a strong prior.

### 10.3 Mixture of Experts, routing, and the reparameterization trick
- A **Mixture of Experts** combines K specialised sub-networks with input-dependent weights
  `g = softmax(router(x))`. Each input can rely on different experts, which is called conditional
  computation.
- **Routing entropy** `H = −Σ g_k log g_k / log K` measures specialization: 0 means one expert takes
  all the weight, 1 means the weight is spread evenly.
- **Temperature** τ sharpens (τ < 1) or flattens (τ > 1) the routing.
- **Reparameterization:** sampling `z ~ N(μ, σ²)` is not differentiable, but `z = μ + σ·ε` with
  `ε ~ N(0, I)` is. Gradients flow into μ and σ, so the network can learn a distribution instead of a
  point.

### 10.4 Mixture moments
A routed mixture `Σ_k g_k N(μ_k, σ²_k)` has mean `Σ g_k μ_k` and variance
`Σ g_k (σ²_k + μ_k²) − (Σ g_k μ_k)²`. That is the average within-expert variance **plus the
disagreement between experts**. So the modality's uncertainty grows both when experts are individually
unsure and when they disagree. This is a principled way to turn several experts into one uncertainty.

### 10.5 KL regularization and learned (aleatoric) uncertainty
- `KL(N(μ, σ²) ‖ N(0, 1)) = ½(μ² + σ² − 1 − log σ²)` is zero at μ = 0, σ² = 1. It keeps the
  representation from collapsing to a point (σ → 0) or exploding.
- A variance predicted by the network is **aleatoric** (data) uncertainty. It is learned from training
  data and can be wrong on inputs that look nothing like it. That is the failure we saw with log noise
  on TT. **Epistemic** (model) uncertainty would need ensembles or Bayesian methods (§12).

### 10.6 Data leakage through overlapping windows
With stride-1 windows of length 10, window t and window t+1 share 90% of their data. A random split puts
many such pairs on both sides, so the model is tested on near-copies of training data. Scores then
measure memorization, not detection. The fix is to split by **run** (whole recordings), so no second of
data appears in two splits. The guide warns about this in §13.

### 10.7 Metrics under class imbalance
- **Anomaly-class F1** ignores how many normal windows are correctly passed. If 83% of windows are
  anomalous, "always anomalous" scores F1 0.91.
- **Macro-F1** averages the F1 of both classes, so a trivial predictor scores at most 0.5.
- **PR-AUC** and **ROC-AUC** measure ranking quality independent of the 0.5 threshold. A model can rank
  well (high PR-AUC) and still have a bad threshold (low F1). That happened on TT.

### 10.8 Seeds, variance and paired t-tests
- Neural network results vary with the random seed (initialization, batch order).
- With one seed, differences of a few points can be pure chance. Our single-seed ablations showed
  "effects" of up to 19 points that vanished over 5 seeds.
- A **paired t-test** compares two models on the same seeds and splits, testing whether the mean
  per-seed difference is non-zero. With 5 seeds it detects only large, consistent effects.

### 10.9 Robustness evaluation
Robustness tests perturb **inputs at test time only**, using a model trained on clean data:
- **Missing data:** zeroed observations. After z-scoring, zero is the training mean, so it acts as
  "no information".
- **Noise:** Gaussian noise scaled to each channel's training spread.
- The same random perturbations are used for every model, so the comparison is paired.

---

## 11. Decisions, deviations and corrections, in order

| # | What happened | What we changed | Effect |
|---|---|---|---|
| 1 | Audit: every training script picked the best epoch **on the test set**; no validation split | shared protocol `common.py`: validation split, selection on validation, test once | "UG-MoE beats MUAD" disappeared under fair selection (0.988 vs 0.988 F1) |
| 2 | Robustness showed traces and logs had **no effect** on any model | found log features ≈ 6·10⁻¹⁰; train-only z-score of all modalities (guide step 5) | traces and logs start contributing |
| 3 | MUAD's data split stride-1 windows at random: **83.9% of test windows leaked** | recovered the 4 runs by chaining windows; run-level split | numbers dropped to their honest level; differences between models became visible |
| 4 | Legacy model defects (§7.4) | wrote UG-MoE v2 from the guide's pseudocode | 25× smaller, every component testable |
| 5 | Old SN preprocessing pipelines were broken (malformed, single-class chunks) | new verified pipeline from the Zenodo raw data (`data_pipeline/`), with clock-offset detection and quality reports; TT added | realistic benchmark with fault-free runs; TT cross-dataset validation |
| 6 | TT's first split had no fault-free run in validation, so validation was 85% anomalous and selection on anomaly-F1 picked "all anomalous" epochs (4 of 5 models) | selection on **validation macro-F1**; fault-free TT runs cut into train / val / test segments | TT results became valid |
| 7 | Proposal: modality dropout to fix the "fails without metrics" weakness | **deferred** by decision (not in the guide) | reported as a limitation |
| 8 | Graph library | DGL instead of PyTorch Geometric | same library as the vendored baselines |
| 9 | Single-seed exploration for speed | final runs with 5 seeds | several single-seed "effects" shown to be noise: the large ablation effects, "the probabilistic representation causes robustness", and the TT robustness / network-fault-recall figures |
| 10 | Sample definition | one graph per window (all services) instead of one sample per service | matches MUAD / Eadro and graph models |

---

## 12. Limitations and threats to validity
1. **Variance on TT is high.** UG-MoE v2 has std 0.09 macro-F1 across seeds (0.62–0.84), and its
   seeds fall into two regimes (high recall with many false alarms vs cautious).
2. **The individual UG components are not shown to add accuracy.** Full ≈ plain MoE (p ≥ 0.7) and no
   ablation is significant at 5 seeds; adaptive fusion is the closest (SN p = 0.061).
3. **The robustness on SN is not attributable to the probabilistic representation** (5-seed check), and
   **on TT UG-MoE v2 is less robust than MUAD / GAT** from 20% missing data.
4. **No fall-back without metrics:** every model collapses when metrics are missing (deferred modality
   dropout).
5. **Learned uncertainty can be overconfident** on out-of-distribution inputs (log noise on TT).
6. **More false alarms on TT:** 18% of normal windows flagged vs 11–12% for the baselines, the price
   of catching network faults; TT PR-AUC is not better than the baselines'.
6. **Split caveats:**
   - on TT, fault-free test segments come from the same runs as fault-free training segments;
   - there is no time gap at the cuts (adjacent windows share no seconds but are similar);
   - fault-free runs were recorded after the fault runs;
   - log templates are mined over all runs (label-free).

   Planned hardening: a 60-s gap at the cuts and templates mined on training runs only.
7. **Small datasets:** SN has 4 fault runs, so its test set is 1 fault run + 1 fault-free run.
8. Modality ablation, routing / uncertainty analysis and sensitivity are single-seed.
9. TT traces cover only 2–4% of seconds, so TT cannot test trace-based fall-back.

---

## 13. How to reproduce everything
From `project/` on the GPU server (the full command list is in `README.md`):

```bash
python -m data_pipeline.fetch_eadro
python -m data_pipeline.build_eadro --system SN
python -m data_pipeline.build_eadro --system TT --stride 5
# main comparison, ablations, modality subsets, sensitivity: experiments/run_baselines.py --models ...
# robustness / analysis from a checkpoint: experiments/run_robustness.py, experiments/run_analysis.py
python experiments/make_report.py      # all tables + figures, pooled over seeds
```

Every run directory contains:
- `split.json`, with the exact window ids per split;
- `normalization.json`;
- `summary.json`;
- one `<model>_seed<N>.json` per run, with the full validation history;
- `<model>_seed<N>_best.pth`.

---

## 14. Status against the guide's completion criteria (§14)

| Criterion | Status |
|---|---|
| Pipeline runs end to end from dataset files to predictions | ✅ `fetch_eadro` → `build_eadro` → `run_baselines` |
| GAT baseline reproducible | ✅ deterministic per seed, 5 seeds |
| Plain MoE and Graph-MoE under the same protocol | ✅ `ug_moe2:plain`, `ug_moe2:graph`, 5 seeds |
| Full model runs without numerical instability | ✅ no NaN / skipped batches in any training log we inspected; log-variance clamped, variance ≥ 10⁻⁶ |
| All required ablations completed | ✅ 6 ablations × 5 seeds × 2 datasets |
| Missing-modality and noise experiments completed | ✅ 5 seeds on SN + TT |
| Hyperparameter sensitivity plots generated | ✅ 6 parameters (SN), `sensitivity_sn.png` |
| Final metrics reproducible from saved checkpoints and scripts | ✅ checkpoints per seed; `make_report.py` |

Guide §10 required outputs:
- preprocessed dataset with schema ✅ (`metadata.json`, `data_README.md`);
- graph file ✅;
- checkpoints of GAT / Plain MoE / Graph-MoE / full ✅;
- overall, ablation and robustness tables and plots ✅;
- expert-weight heatmaps ✅;
- uncertainty / fusion plots ✅;
- sensitivity plots ✅;
- reproducible evaluation script ✅ (`make_report.py`).

---

## 15. File reference

| Topic | File |
|---|---|
| Raw data download | `project/data_pipeline/fetch_eadro.py` |
| Windowing, alignment, features, labels | `project/data_pipeline/build_eadro.py` |
| Services, graph, metric names, fault types | `project/data_pipeline/systems.py` |
| Dataset documentation | `project/data_README.md` |
| UG-MoE v2 | `project/tracks/proposed_model/ug_moe_v2.py` |
| Legacy UG-MoE | `project/tracks/proposed_model/model.py`, `project/src/` |
| Baselines | `project/tracks/baselines/`, `project/vendor/muad/` |
| Protocol (split, normalization, training, metrics) | `project/experiments/common.py` |
| Model names, variants, ablations | `project/experiments/registry.py` |
| Experiments | `project/experiments/run_*.py` |
| Tables and figures | `project/experiments/make_report.py` |
| Round-by-round log and decisions | `PROJECT_STATUS.md` |
| MUAD critique with evidence | `MUAD_vs_UG-MoE.md` |
