# The UG-MoE project, explained

A study guide to the whole project: what problem we solve, how the data becomes model input, how UG-MoE v2
works (step by step, with real tensor shapes), whether it follows the implementation guide, what we found, and
how to answer the questions a supervisor or reviewer is likely to ask.
For the formal write-up see `UG-MoE_Implementation_Report.md`; for every experiment round see `PROJECT_STATUS.md`.

---

## Part 1. The problem in one page

A microservice system (SocialNetwork = 12 services, TrainTicket = 27 services) is monitored in three ways:

| Modality | What it is | Example signal of a fault |
|---|---|---|
| **Metrics** | numbers sampled every second per service (CPU, memory, network bytes) | CPU jumps to 100% during a `cpu_load` fault |
| **Traces** | timing of each request as it passes through services (spans) | latency rises during `network_delay` |
| **Logs** | text lines services write | new error messages, or message rates change |

Researchers injected faults (CPU load, network delay, network loss; about 2 minutes each) and recorded
everything. **Our task:** look at a 10-second window of all three modalities from all services and say
*normal* or *anomalous*.

Why it is hard:
- the signals are noisy;
- a fault in one service shows up in others, because faults propagate along calls;
- each modality can be missing or misleading;
- some faults (network) barely change the metrics.

That is the motivation for the three ideas in UG-MoE:
1. **experts:** different fault types need different "views" of the data;
2. **the graph:** faults propagate along dependencies;
3. **uncertainty:** know when a modality can't be trusted, and lean on the others.

---

## Part 2. From raw files to one training sample

```
raw run (metrics CSVs, spans.json, log files, fault file)
   │  1. 1-second grid; fix clock offsets (spans/logs are +7/+8 h shifted)
   │  2. per second: 7 metrics / service, log1p(mean span latency) / service
   │  3. logs -> Drain3 templates -> counts per service in the window
   │  4. cut into 10-s windows; label = anomalous if it overlaps a fault by ≥ 1 s
   ▼
one window = one graph:  12 nodes (services), 24 edges (call dependencies)
   node features:  metrics [10 s × 7]   traces [10 s × 1]   logs [13 template counts]
   label: 0 normal / 1 cpu / 2 delay / 3 loss   (model predicts 0 vs ≥1)
```

Then (in `experiments/common.py`):
- **split by whole runs** (never by random windows: neighbouring windows overlap 90%, which leaks);
- **z-score** every feature with training-split statistics only.

---

## Part 3. How UG-MoE v2 works, step by step

Code: `project/tracks/proposed_model/ug_moe_v2.py`. Shapes below are for one SN batch:
**B = 50 windows, N = 12 services, T = 10 seconds, D = 64 hidden size, K = 4 experts.**
DGL stores a batch of 50 graphs as one big graph with B·N = 600 nodes.

### Step 0: Inputs
```
metrics  [600, 10, 7]      600 service-windows, 10 seconds, 7 metrics
traces   [600, 10, 1]
logs     [600, 13]         no time axis (counts over the window) -> treated as T = 1: [600, 1, 13]
```

### Step 1: Encode each modality
A separate `Linear` per modality maps its features to D = 64. A learned **modality embedding** is added,
a 64-dim "tag" so later shared modules know which modality they are looking at.
```
seq[metric] = Linear_metric(metrics) + emb[metric]   -> [600, 10, 64]
seq[trace]  = Linear_trace(traces)   + emb[trace]    -> [600, 10, 64]
seq[log]    = Linear_log(logs)       + emb[log]      -> [600,  1, 64]
node[v]     = mean over time of seq[v]               -> [600, 64]   (one vector per service)
```

### Step 2: Four experts look at the modality, each in its own way
The same 4 experts are used for all 3 modalities (shared weights), so per batch they run 4 × 3 = 12 times.

| Expert | Operation | Input → output | Intuition |
|---|---|---|---|
| Temporal | GRU over the 10 seconds, last state | `[600,10,64] → [600,64]` | "how did this service's signal evolve?" |
| Semantic | Transformer layer across the 12 services of a window | `[50,12,64] → [600,64]` | "which services look alike right now?" (content, not graph) |
| Dependency | GATv2 over the call graph | `[600,64] + graph → [600,64]` | "what are my callers / callees doing?" |
| Cross-modal | attention from this modality's service vector to the 3 modality vectors of that service | `[600,1,64] → [600,64]` | "do metrics, traces and logs of this service agree?" |

### Step 3: Each expert gives a *distribution*, not just a vector
Each expert's per-service output is pooled to one vector per window: mean and max over the 12 services,
concatenated (128), then `Linear → 64`. That gives `[50, 64]`. Two heads per expert then produce:
```
mu_k    = MuHead_k(pooled)        [50, 64]   the expert's representation
s_k     = LogVarHead_k(pooled)    [50, 64]   its log-variance = how unsure it is (clamped to [-8, 8])
```
Stacked over the 4 experts: `mu_k [50, 4, 64]`, `s_k [50, 4, 64]`.

### Step 4: The router decides how much to trust each expert, per modality and per window
There is one router per modality (a small MLP). Its input (132 numbers per window):
```
pooled modality features          [50, 64]   "what does this modality look like?"
graph context (GraphConv, pooled) [50, 64]   "what does it look like propagated over the call graph?"
expert uncertainties u[k]=mean(s_k)[50, 4]   "how unsure is each expert?"     <- uncertainty-guided routing
```
Output: `g_v = softmax(router_v(input) / τ)`, shape `[50, 4]`, weights summing to 1.
Example from our SN analysis: for CPU-fault windows the metric router puts **0.95** on the cross-modal
expert; for normal windows **0.87** on the dependency expert.

### Step 5: Combine the experts into one Gaussian per modality (mixture moments)
```
mu_v  = Σ_k g[k] · mu_k                                   [50, 64]
var_v = Σ_k g[k] · (exp(s_k) + mu_k²) − mu_v²             [50, 64]
```
The variance is the experts' own uncertainty **plus their disagreement**. If two trusted experts disagree,
the modality becomes uncertain even if each expert is confident.

Then:
- **training:** `z_v = mu_v + sqrt(var_v) · ε`, with ε ~ N(0, 1) (the reparameterization trick). The
  noise teaches the classifier to work with imprecise representations, which helps robustness.
- **testing:** `z_v = mu_v`, so predictions are deterministic.

### Step 6: Fuse the three modalities, trusting the less uncertain ones more
For each modality, the score is `fusion([mu_v, mean(log var_v)])`, using the same small MLP for all three.
Then `w = softmax over the 3 scores`, shape `[50, 3]`, and `z = w_metric·z_metric + w_trace·z_trace + w_log·z_log`.
Example: on TT, when metrics are removed the metric uncertainty jumps from −0.28 to +0.80 and its weight
falls from 0.61 to 0.13.

### Step 7: Classify and train
- `logits = Classifier(z)`, shape `[50, 2]`: Linear → GELU → Dropout → Linear.
- `loss = CrossEntropy(logits, anomalous?) + 0.01 · KL`, where
  `KL = ½ mean(mu² + var − 1 − log var)` pulls each modality's Gaussian towards N(0, 1), so the variance
  neither collapses to 0 nor explodes.

### Sizes and switches
- 166,031 parameters on SN.
- Every component can be switched off: graph, MoE, dynamic routing, uncertainty routing, probabilistic,
  adaptive fusion. That gives the guide's incremental steps (`ug_moe2:plain → graph → prob → ug_routing →
  full`) and its ablations (`abl2:no_*`), each differing in exactly one thing.

---

## Part 4. Does v2 follow the implementation guide?

**Yes, with three interpretive choices**, explained below. Line-by-line check against the guide's §11
pseudocode and §6 steps:

| Guide | v2 | Match |
|---|---|---|
| §11.4–6 `hL, hT, hM = Encoder(L, T, M)` | per-modality Linear + modality embedding | ✅ (guide step 8: "keep the first version simple") |
| §6.8 four experts: Temporal, Semantic, Dependency, Cross-Modal | GRU, cross-service Transformer, GATv2 on the call graph, cross-modal attention | ✅ |
| §11.7 `e_v = Σ_k g_v[k] · Expert_k(h_v, G)`: experts shared across modalities | experts shared across modalities, weighted by `g_v` | ✅ |
| §6.9 router gets the representation + graph info, softmax, `H = Σ g_k E_k`, log weights | pooled features + GraphConv context, softmax, mixture, weights returned for analysis | ✅ |
| §11.7 `g_v = Router(h_v, G, uncertainty_v)` | router input includes graph context and expert uncertainty | ✅ (choice 1: one router per modality) |
| §11.7 `mu_v, sigma_v = ProbabilisticHead(e_v)`; `z_v = mu_v + sigma_v·ε` | per-expert μ / log-variance heads; mixture moments give μ_v, σ²_v; z sampled in training | ✅ (choice 2, below) |
| §6.10 KL so the variance doesn't collapse | `½(μ² + σ² − 1 − log σ²)` | ✅ |
| §6.11 uncertainty as router input; compare with and without | `u_v[k]`; ablation `w/o uncertainty routing` | ✅ |
| §11.8–9 modality weights; `z = Σ w_v z_v` | softmax weights from [μ_v, uncertainty_v] | ✅ |
| §6.13 / §11.10 classifier, 2-class CE | ✅ | ✅ |
| §7 `L_total = L_anomaly + λ_KL·L_KL`; report any routing regularizer separately | exactly this; load balancing exists but is off (weight 0) | ✅ |
| §12 order: plain MoE → graph routing → probabilistic → uncertainty routing → fusion | `INCREMENTAL_STEPS`, the same five steps | ✅ |
| §8B ablations: w/o graph, MoE, dynamic routing, uncertainty routing, probabilistic, fusion | `ABLATIONS_V2`, the same six | ✅ |
| §4 PyTorch Geometric | DGL (choice 3: same library as the MUAD baseline) | ⚠️ tooling only |

**Choice 1, one router per modality.** The pseudocode writes `Router(h_v, G, uncertainty_v)` inside a
loop over modalities, so a single shared router is also a valid reading. We gave each modality its own
router, because the best experts differ by modality (our analysis confirms it: traces prefer the
cross-modal expert, metrics are fault-specific). This is cheap: 3 small MLPs.

**Choice 2, uncertainty before routing (this resolves a circularity in the guide).** As written:
```
g_v = Router(h_v, G, uncertainty_v)       # routing needs the uncertainty ...
e_v = Σ g_v[k] · Expert_k(h_v, G)
mu_v, sigma_v = ProbabilisticHead(e_v)    # ... but the uncertainty is only computed after routing
```
The uncertainty used for routing cannot be the one produced after routing. v2 gives **each expert its own
probabilistic head**, so each expert's uncertainty exists *before* routing and can guide it. The modality's
μ_v and σ_v are then the **moments of the routed mixture**, which is exactly "ProbabilisticHead applied to
the mixture", computed in closed form. It is the most literal implementation that is actually computable.

**Choice 3, evaluation uses μ (no sampling at test time).** The pseudocode doesn't say what happens at test
time. Sampling would make predictions random (MUAD's bug), so we use μ. This is the standard convention
for variational models.

One more detail: the guide's step 2 says "one sample per service and window". We use one sample per window
containing all services as graph nodes, which is what a graph model needs and what MUAD / Eadro do. The
per-service view still exists inside the model (the experts work per service).

---

## Part 5. The baselines in one line each
- **MUAD** (base paper): per-modality encoders → "uncertain blocks" (μ, σ, sampling, KL) → confidence (TCP)
  fusion → classifier. Used unchanged. Its code bugs: test-set selection, test-time sampling, wrong KL
  formula. See `MUAD_vs_UG-MoE.md`.
- **GAT:** concatenate all features per service → 2 graph-attention layers → classifier. The guide's
  "baseline first" (step 7).
- **Plain MoE** = `ug_moe2:plain`: v2 without graph, uncertainty or adaptive fusion.
- **Graph-MoE** = `ug_moe2:graph`: plain + graph context and the dependency expert.
- **Legacy UG-MoE** (`GraphAwareMoE`, 4.4M parameters): our earlier model. Kept for reproducibility; its
  routing never reached the classifier, which is why v2 was written.

---

## Part 6. The story of the project (what happened and why it matters)
1. **Inherited code claimed UG-MoE > MUAD.** Our audit found the best epoch was picked on the test set.
   With a validation split, the advantage vanished (0.988 vs 0.988 F1).
2. **Traces and logs had zero effect on every model.** Cause: MUAD's log features were ~0.0000000006 in
   size. Fix: train-only z-scoring.
3. **The MUAD data leaks.** Stride-1 windows split randomly put 84% of test windows next to a
   near-identical training window. The MUAD paper does exactly this ("randomly selected 60% … 40%").
   Fix: split by whole runs (we reconstructed the runs by chaining overlapping windows).
4. **The legacy model didn't do what it claimed** (routing unused, no real graph or uncertainty routing).
   So we wrote v2 directly from the guide.
5. **We rebuilt SN and TT from the raw Zenodo data**: clock-offset fixes, fault-free runs included, TT added.
6. **A selection bug on TT:** validation without normal data → models picked "all anomalous" epochs.
   Fix: select on macro-F1; give every split normal data.
7. **Single-seed exploration** of every guide experiment (A–H), then **5-seed final runs**, which showed
   several single-seed effects were noise.

---

## Part 7. Results cheat sheet (5 seeds where marked)

| Question | Answer |
|---|---|
| Is UG-MoE v2 better than MUAD? | **Yes, on both datasets (5 seeds):** SN 0.972 vs 0.925 (+4.7), TT 0.723 vs 0.626 (+9.7) macro-F1 |
| Better than GAT? | Yes: +4.1 (SN), +6.3 (TT), 5 seeds |
| Which component matters most? | **Uncertainty-aware adaptive fusion**: removing it costs 2.4 (SN) / 3.3 (TT), 5 seeds |
| Do the uncertainty parts improve clean accuracy over plain MoE? | Not clearly: +0.1 SN, +2.5 TT (TT seeds vary a lot) |
| Where do they help? | **Missing data (SN, single seed, 5-seed rerun running):** at 40% missing, full 0.856 vs 0.707 without the probabilistic part; plain MoE is no better than MUAD |
| Which anomalies does v2 catch that others miss? | Network faults on TT: recall 0.88–0.90 vs 0.30–0.65; but more false alarms (30% vs 13–17%) |
| Do experts specialize? | Yes: e.g. CPU faults → cross-modal expert (0.95), normal → dependency expert (0.87) on SN |
| Does the uncertainty behave sensibly? | In direction yes (a corrupted modality gets more uncertain and less weight). Too weakly when metrics vanish; overconfident under log noise on TT |
| Is it sensitive to hyperparameters? | No: all settings within 3.4 points of the default |
| Why are our MUAD numbers below the paper's 0.99? | The paper's random split leaks and its test set is 83% anomalous; under its protocol we reproduce 0.988 |

---

## Part 8. Questions you may be asked, and short answers

**"Why not just report F1 like MUAD?"**
With 83% anomalous windows, predicting "anomalous" always gives F1 0.91. Macro-F1 averages both classes,
so a trivial model scores ≤ 0.5. We report all metrics plus a trivial row.

**"Isn't your split unfair to MUAD?"**
Every model gets the identical split, normalization and selection rule. MUAD also did *worse* with its own
paper hyperparameters (0.793 vs 0.922), so our settings favour it.

**"Why does the full model not beat plain MoE on clean SN?"**
SN is near its ceiling (0.97+). The uncertainty components pay off when data is missing (+13 points at
40% missing on SN) and through adaptive fusion (the consistent ablation effect).

**"Why do your single-seed and 5-seed results differ?"**
TT especially varies across seeds (the same model scored 0.65–0.82). That is why the paper uses 5 seeds
and paired t-tests, and why we flag single-seed results.

**"What are the limitations?"**
- Both models fail when metrics are entirely missing (modality dropout would fix this; deferred).
- Learned uncertainty can be overconfident on unfamiliar noise.
- More false alarms on TT.
- Small datasets (SN has 4 fault runs).
- Single-seed analysis and sensitivity.

**"Is v2 what the guide asked for?"**
Yes. Part 4: every guide element maps to v2. The three interpretive choices: per-modality routers,
per-expert uncertainty heads (which resolve the guide's routing-uncertainty circularity), and deterministic
evaluation.

---

## Part 9. Where to look in the code

| Want to understand… | Open |
|---|---|
| how a window is built | `project/data_pipeline/build_eadro.py` (docstring at the top) |
| the model | `project/tracks/proposed_model/ug_moe_v2.py` (`forward`, about 70 lines) |
| how training / selection / metrics work | `project/experiments/common.py` (`load_splits`, `train_and_select`, `compute_metrics`) |
| how a model name maps to a configuration | `project/experiments/registry.py` (`get_model`) |
| robustness tests | `project/experiments/run_robustness.py` |
| routing / uncertainty analysis | `project/experiments/run_analysis.py` |
| every table and figure | `project/experiments/make_report.py` |
