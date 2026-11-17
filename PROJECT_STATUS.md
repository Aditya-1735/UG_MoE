# UG-MoE — Project Status vs. Implementation Plan

**Last updated:** 2026-11-17 · **Branch:** `hdfs` · **Plan:** `7. UG-MoE_Implementation_Guide.md` · **Baseline paper:** MUAD (`7. Uncertainty-Aware_Multimodal_Anomaly_Detection_..._Active_Learning.pdf`)

> **How this was assessed.** Only the Python code, the data files, and the result JSONs in `project/experiments/runs/` were treated as evidence. Claims in the older `.md` files (opencode-era plans, `UG-MoE-Research-Paper-Draft.md`, status tables marked "Complete") were **not** taken at face value. Every finding below points to the code that supports it. Line numbers refer to the working copy on 2026-11-17.

---

## 0. Progress log

| Date | Change | Server-verified? |
|---|---|---|
| 2026-11-17 | **Phase 0.1:** uncertainty blocks (`src/uncertainty/uncertainty.py`) now sample with reparameterization only in `train()` mode and return deterministic μ in `eval()`. Variance / log-variance clamps restored. Working-copy `model.py` semantic-expert fix (`expert(graph.ndata["logs"], B)`) kept. Fixes §3.6. | ❌ pending |
| 2026-11-17 | **Phase 1 (part):** new shared protocol `project/experiments/common.py`: stratified val split from `chunk_train.pkl` (fixed `split_seed=2026`, 15%, ids saved to `split.json`), selection on val F1 (tie: PR-AUC), early stopping on val, test evaluated once, per-seed checkpoints, metrics incl. macro-F1 / normal-class F1 / trivial baseline. `run_comparison.py` rewritten on top of it, with Track A using MUAD's real objective. Fixes §3.4 and §3.8 **for the comparison script only**. | ❌ pending |
| 2026-11-17 | **Decision:** because of time constraints, all experiments run with a **single seed (42)** for now. Multi-seed runs (5 seeds plus paired t-test) are deferred to the final paper runs (Phase 3–4). Single-seed differences below about 1 F1 point are not evidence of improvement. | n/a |
| 2026-11-17 | **Phase 1 (cont.):** `experiments/registry.py` (single definition of every model / ablation + forward adapter + `run_suite`). GAT and Graph-MoE model classes moved to `tracks/baselines/` (node count now read from the graph, not hard-coded). New scripts on the shared protocol: `run_baselines.py` (Exp. A), `run_ablations.py` (rewritten; always trains the `ug_moe` reference too), `run_robustness.py` (Exp. D + E as specified: single-modality drop, random 10/20/40/60% masking, noise scaled to train std on all 7 modality combinations, relative F1 drop). `run_comparison.py` now builds models through the registry. **Removed** (superseded): `run_gat_baseline.py`, `run_plain_moe.py`, `run_graph_moe.py`, `run_track_b_full.py`, `run_missing_modality.py`, `run_noise_robustness.py`. Not migrated yet: routing / uncertainty-fusion analysis (they depend on Phase 2), `run_track_a_al.py`, `run_hdfs.py`. | ❌ pending |
| 2026-11-17 | **First new-protocol result** (server): Track A vs Track B, seed 42. Test F1 0.9882 vs 0.9879, macro-F1 0.9654 vs 0.9645, so there is **no advantage** for UG-MoE once selection moves off the test set (§4). This is the "before" baseline for the Phase 2 fixes. | ✅ |
| 2026-11-17 | **Leakage found and fixed (§3.12):** the artifact uses stride-1 windows split at random, so **83.9% of test windows have a 90%-identical training neighbour.** All results so far are inflated. Reconstructed the 4 original runs from the chunks and added a leak-free **run-level split** (`--split run`, the new default: train runs 0–1 / val run 2 / test run 3, 0 overlapping windows). Decoded labels: culprit = fault type (0 normal, 1 cpu, 4 delay, 5 loss). **All previous numbers need rerunning under `--split run`.** | ❌ pending |
| 2026-11-17 | **Round 2 (normalized inputs)**, server, seed 42: plain MoE 0.9906 / GAT 0.9869 / Graph-MoE 0.9898 / MUAD 0.9882 / UG-MoE 0.9857 test F1. **Traces and logs are still unused by both MUAD and UG-MoE**, so the problem is a metrics shortcut, not only scale (§4). Added trained modality ablation (Exp. C) as `<model>@<mods>` in `registry.py`. | ✅ |
| 2026-11-17 | **Robustness** (server, seed 42) on the comparison checkpoints: **MUAD also ignores traces and logs completely**, so the problem lies in the data rather than in our model. **Root cause:** the log features are ~6e-10 in magnitude, numerically constant to the networks. Fix: train-split z-score normalization of all modalities in `common.load_splits` (plan Step 5, now the default; `--normalize none` for the old behaviour). All "before" numbers above use raw inputs, so rerun comparison and baselines with normalization (§3.5). | ✅ results / ❌ fix pending |
| 2026-11-17 | **Baselines** (server, seed 42): GAT 0.9848 / Graph-MoE 0.9866 / **plain MoE 0.9917** test F1 (macro-F1 0.9543 / 0.9608 / **0.9756**). Plain MoE beats full UG-MoE (0.9879 / 0.9645), so the uncertainty and routing components currently hurt (§4). | ✅ |
| 2026-11-17 | **Data:** the SN chunks in `preprocessing/sn/` (from `convert_sn_dataset.py`) turned out to be malformed and single-class (§3.11). Deleted them; the converter is marked broken. | n/a |
| 2026-11-17 | **Scope:** removed the HDFS log-only mode and all HDFS code and runs (out of plan scope, §7), plus `UG-MoE-Research-Paper-Draft.md` (invalid numbers and claims). `model.py` and `router.py` now have a single multimodal path. | ❌ pending (compile-checked only) |
| 2026-11-17 | **Clean-up:** deleted 8 opencode-era planning/forensics `.md` files at the root, `run2`, `project/ablations.log`, `project/project/` (119 MB of October runs), `tracks/proposed_model/train.py` and `test_integration.py`, and `experiments/run_track_a_full.py` (superseded by the `muad` registry entry). Untracked 18 `.pyc` files. Rewrote `README.md` and fixed `data_README.md` (train count, labels, commands). | n/a |

---

## 1. Summary: where we are

**Position in the plan's Implementation Order (§12):** every step from 1 to 12 has *something* in the repo. Most steps are only partly done or have correctness problems. **In practice we are around Step 7–9: graph-aware routing, probabilistic uncertainty and uncertainty-guided routing need to be redone so they actually work as the plan describes.** The ablation and robustness results we already have (Steps 11–12) were produced on a model with these defects, so they cannot be used as paper results yet.

What works:
- An end-to-end training/evaluation pipeline runs on the GPU server on `project/recovered_artifact` (12 services, 24 edges, metrics + traces + logs). It produces F1 around 0.99 for every model.
- All four experts (temporal, semantic, dependency, cross-modal), three router classes, Gaussian uncertainty blocks with KL, and confidence-weighted fusion exist as code.
- Scripts and results exist for: Track A vs Track B (5 seeds), 6 ablations (5 seeds), GAT / plain-MoE / Graph-MoE baselines (2 seeds), missing-modality, noise and routing analysis (2 seeds).

What is wrong or missing (details in §3):
1. **The router has no effect on predictions.** The routed mixture `H = Σ g_k E_k` is computed and then thrown away. The classifier only sees the fusion output.
2. **The "full" model's router uses neither the graph nor the uncertainty blocks.** It ignores the graph input and the per-expert uncertainties it is given.
3. **Model selection happens on the test set,** and there is no validation split. Every "best F1" we have is optimistic.
4. **The saved results show traces and logs have no effect.** Zeroing them, or adding noise up to σ=0.5, leaves F1 unchanged. The model behaves as if it only sees metrics.
5. **The task is easy and imbalanced.** About 83% of test windows are anomalous, so predicting "anomalous" for everything already scores F1 = 0.906. All the differences between models (≈0.2–0.8 F1 points) fall inside this narrow range.
6. **Track A in the comparison is not faithful MUAD.** It is trained with plain cross-entropy instead of MUAD's own loss.
7. Several required experiments do not exist yet: modality ablation (C), random-percentage missing modality (D), combined-modality noise (E), sensitivity (H), and the TT dataset.

---

## 2. Status by plan step

Legend: ✅ done and verified in code · 🟡 partial / done with problems · ❌ not done · ⚪ out of scope / optional

### 2.1 Implementation Order (§12 of the plan)

| # | Plan step | Status | Evidence / notes |
|---|---|---|---|
| 1 | Load and visualize raw multimodal data | 🟡 | Raw SN data and conversion scripts exist (`preprocessing/raw_to_parsed/`, `experiments/convert_sn_dataset.py`), plus validation gates (`validation/gates/gate01–10`). No visualization or EDA code found. |
| 2 | Aligned samples + verified labels | 🟡 | Model-ready data = `recovered_artifact/` (10-step windows × 12 services). Labels: `culprit ∈ {0,1,4,5}`, binarized as `culprit >= 1` (same rule as MUAD, `vendor/muad/models/main_model.py:84`). Our own pipeline from raw data gives *different* data (`muad_compat/output`: `event_num=14`), so `recovered_artifact` was **not reproduced** from raw. Its provenance comes only from `data_README.md`. |
| 3 | Build and validate service graph | 🟡 | 24-edge graph lives in `recovered_artifact/metadata.json` and includes self-loops. Gate 07 checks its structure. No standalone graph file (plan §10). |
| 4 | Simple baseline classifier | ❌ | No non-graph MLP / per-modality baseline script. |
| 5 | GAT baseline | 🟡 | `experiments/run_gat_baseline.py` (GATv2, mean-pooled time). It ran with **2 seeds** and lr=0.001, while the main comparison used 5 seeds and lr=0.0005, so the protocols differ. |
| 6 | Plain MoE without uncertainty | 🟡 | `run_plain_moe.py`, plus the `plain_moe` ablation. Same `GraphAwareMoE` class, so it inherits the "router is ignored" defect (§3.1): this "MoE" has no effective gating. |
| 7 | Graph-aware routing | ❌ (code exists, ineffective) | See §3.2. The graph input to the base router is never enabled by any experiment config, and it is computed incorrectly. `GraphAwareRouter` doesn't take a graph and can't be constructed by the factory. |
| 8 | Probabilistic uncertainty | 🟡 | `VariationalUncertaintyBlock` (μ, log σ², reparameterization, KL), one per expert. Plan Step 10 maps the **MoE output** to μ/σ; the code does it **per expert, before routing**. The working copy has uncommitted changes that flip it between deterministic μ and sampling (§3.6). |
| 9 | Uncertainty-guided routing | ❌ (code exists, ineffective) | `UncertaintyGuidedRouter` ignores the `expert_uncertainties` it receives. It predicts its own "uncertainty" from features (`src/routing/router.py:338–372`). The "uncertainty" passed to the base router is a batch-mean scalar KL, not a per-sample value. |
| 10 | Adaptive multimodal fusion | 🟡 | `AdaptiveFusion(fusion_type="confidence")` fuses the **4 expert outputs**, not the **3 modality latents** (`z_log, z_trace, z_metric`) the plan specifies (Step 12, pseudocode lines 8–9). |
| 11 | Freeze implementation, run all ablations | 🟡 (premature) | 6 ablations × 5 seeds ran (`runs/ablations_20261111_231146`), but on the defective model. Their labels don't match what they change (§3.7). |
| 12 | Robustness + sensitivity | 🟡 | Missing-modality and noise ran (2 seeds). No sensitivity experiments. |

### 2.2 Detailed steps (§6 of the plan)

| Plan step | Status | Notes |
|---|---|---|
| S1 Download/inspect SN | 🟡 | Raw SN is present locally (`SN_dataset/`, `raw_data/`, gitignored). |
| S2 Common time window, `sample_id/window_id/service_id` | 🟡 | Chunks have random string IDs (e.g. `MDgXKHl5`) with no window timestamp. **Whether the train/test split is time-ordered cannot be verified from the data** (plan §13: "do not leak future information"). |
| S3 Logs → templates | 🟡 | Drain3 in `raw_to_parsed/convert_logs.py`. Model input is a per-service count vector over 15 templates (`logs: [12, 15]`). |
| S4 Traces | 🟡 | `convert_traces.py`. Model input is a single latency channel per service per step (`traces: [12, 10, 1]`). Status codes and parent/child relations are not used as features. |
| S5 Metrics, train-only normalization | ❓ | `metrics: [12, 10, 7]`. **Not verified** where or whether normalization was fit on train only (nothing is done in the dataloader; it would have happened upstream when the chunks were built). |
| S6 Service graph | 🟡 | See Step 3 above. |
| S7 Baseline (GAT) first | 🟡 | See Step 5 above. |
| S8 Four experts | ✅ | `src/experts/`: `TemporalExpertGraph` (GRU on metrics/traces), `LogSemanticExpert`, `DependencyExpert` (GAT or graph transformer on per-node metric embeddings), `CrossModalExpertGraph` (cross-attention over metric/trace/log node embeddings). |
| S9 Router, softmax, `H = Σ g_k E_k`, log weights | ❌ | Softmax: yes. Mixture: computed, then **unused** (`model.py:595–601`). Weights are logged by `run_routing_analysis.py`. |
| S10 μ, σ², reparam, KL | 🟡 | Yes, but per expert (see Step 8). |
| S11 Uncertainty → router, compared vs. without | ❌ | See Step 9. No valid "with vs. without uncertainty routing" comparison exists yet: `no_uncertainty` removes the uncertainty *blocks* and leaves the router unchanged. |
| S12 Fusion `z_f = w_log z_log + w_trace z_trace + w_metric z_metric` | 🟡 | See Step 10. |
| S13 Classifier, CE | ✅ | 2-class CE (`model.py:300–305, 643–644`). |
| S14 Train/val/test split, select on validation F1 / PR-AUC | ❌ | **No validation set.** All scripts pick the best epoch by **test** F1 (e.g. `run_comparison.py:148`, `run_ablations.py:166`, `vendor/muad/training/base_model.py:86–91`). |
| S15 Run required experiments | 🟡 | See §4. |

---

## 3. Critical issues found in the code

Ordered by impact on the paper's claims.

### 3.1 The routed mixture never reaches the classifier
`tracks/proposed_model/model.py:595–614`: `combined = Σ routing_weights[:, i] * expert_outputs[i]` is computed, then `fused = self.fusion(expert_outputs)` and `MMlogit = self.classifier(fused)`. `combined` is never used. The router's only influence is an auxiliary loss term. **Consequence:** "MoE routing" contributes nothing to any prediction. The routing-analysis plot shows near-uniform weights (entropy 1.333 against a maximum of ln 4 = 1.386), which fits this.
**Fix:** feed the routed mixture (or routing-weighted expert outputs) into uncertainty and fusion, as the plan's pseudocode does.

### 3.2 No graph-aware routing in any experiment
- `DEFAULT_CONFIG["router_type"] = "uncertainty_guided"` (`model.py:730`). `UncertaintyGuidedRouter.forward` ignores `graph_info` (`router.py:339`). `use_graph_routing=True` has no effect for this router type (`create_router` drops it, `router.py:405–406`).
- The only router that uses graph info is `ExpertRouter` (`router_type="base"`). Every experiment that selects `base` also sets `use_graph_routing=False` (`no_graph`, `plain_moe`).
- Even if enabled, `graph_info` is built from **batched global node IDs**, cast to float (`model.py:552–560`). Sample *i* in a batch gets edge IDs offset by 12·i, so the "graph feature" encodes the sample's position in the batch, not the topology (which is the same for every sample anyway).
- `GraphAwareRouter` (`router.py:189–276`) never receives a graph, runs self-attention over a single token, and can't be built by `create_router` (no `input_dim` → hard-coded `node_feat_dim=64` against a 192-dim input).
- **Consequence:** the "no_graph" ablation is not removing anything from the full model; it swaps the router class.

### 3.3 Uncertainty-guided routing does not use the model's uncertainty
- `UncertaintyGuidedRouter` takes `expert_uncertainties` as an argument but never reads it. It estimates "uncertainty" with its own softplus heads from the input features (`router.py:356–372`).
- The tensor passed in is `kl.mean()`, a **scalar per expert for the whole batch**, expanded to `[B, num_experts]` (`model.py:563–572`). It carries no per-sample information.
- `model.py:576–584` expects the router's second return value to be an aux-loss dict. For `UncertaintyGuidedRouter` it is the `[B, 4]` uncertainty tensor, so the loss silently adds `0.01 * mean(router_uncertainty)` (`model.py:655–663`). That term pushes the router's own uncertainty toward 0. It is not load balancing and is not described anywhere.

### 3.4 Test-set model selection and no validation split
Every training loop evaluates on the test set each epoch and reports `best_f1 = max over epochs`. Track A via `BaseModel.fit` does the same. This breaks plan §6 S14 and §13 ("do not claim improvement until the same train/validation/test protocol is used"). `final_f1` (last epoch) is consistently 0.3–1.1 F1 points below `best_f1` in the ablations, which shows how much selection on test inflates the numbers.

### 3.5 In the saved results, traces and logs have no effect
`runs/missing_modality_20261111_110025` and `runs/noise_robustness_20261111_141641` (Track B full model, seeds 42 and 123):

| Perturbation at test time | F1 seed 42 | F1 seed 123 |
|---|---|---|
| None (clean) | 0.9903 | 0.9925 |
| Zero traces | 0.9903 (identical) | 0.9925 (identical) |
| Zero logs | 0.9903 (identical) | 0.9925 (identical) |
| Zero metrics | 0.9064 (= all-anomalous, ROC-AUC 0.5) | 0.0000 (= all-normal) |
| Noise σ=0.5 on traces / logs | unchanged | unchanged |
| Noise σ=0.5 on metrics | 0.9832 | 0.9851 |

Test loss changes only around the 8th decimal. **The trained model is metric-only.**

**Confirmed on the new protocol, and for MUAD too** (2026-11-17, `runs/robustness_ug_moe_20261117_112943`, `runs/robustness_muad_20261117_113500`, seed 42, test set):

| Perturbation | UG-MoE F1 / macro-F1 | MUAD F1 / macro-F1 |
|---|---|---|
| Clean | 0.9879 / 0.9645 | 0.9884 / 0.9663 |
| Drop metrics | 0.9064 / 0.4532 (= all-anomalous) | 0.9064 / 0.4532 (= all-anomalous) |
| Drop traces | 0.9879 / 0.9645 (identical) | 0.9884 / 0.9663 (identical) |
| Drop logs | 0.9879 / 0.9645 (identical) | 0.9884 / 0.9663 (identical) |
| Random mask p = 0.1 / 0.2 / 0.4 / 0.6 | 0.971 / 0.952 / 0.907 / 0.862 F1; 0.920 / 0.878 / 0.796 / 0.723 mF1 | 0.977 / 0.970 / 0.944 / 0.925 F1; 0.929 / 0.900 / 0.783 / 0.654 mF1 |
| Noise on metrics σ = 0.5 / 1 / 2 × train std | 0.987 / 0.966 / 0.929 F1 | 0.963 / 0.935 / 0.908 F1 |
| Noise on traces or logs or traces+logs, any σ up to 2 | unchanged | unchanged (±0.0003) |

**Root cause found: the log features are numerically invisible, and traces are dominated by metrics.** Descriptive statistics of `recovered_artifact/chunk_train.pkl` (no training involved):
- **logs**: every value lies in [0, 6.06e-10] (mean 5.9e-10, std 9.4e-11). Rounded to 6 decimals the whole modality is a single constant. A linear layer with ~0.1 weights turns this into ~1e-11, effectively zero, so **no model can use logs as provided**. The signal does exist *relative to its own scale*: event channels 1, 2, 3 and 5 differ between normal and anomalous windows by 0.36–0.51 standardized mean difference.
- **traces**: already z-scored (mean 0, std 0.98), with signal (SMD 0.70 between classes). Metrics are stronger (channels 1, 5, 6: SMD 0.96–1.06), so the models learn a metric-only shortcut.
- **Additional UG-MoE wiring issue:** `TemporalExpertGraph` returns both `metric_embedding` and `trace_embedding`, but `model.py` keeps only `metric_embedding` (`out.get("metric_embedding", ...)`). Traces reach UG-MoE only through the cross-modal expert (and through routing features, which don't affect the output, §3.1).

**Fix applied (plan Step 5 / §13, so not a deviation):** `common.load_splits` now z-scores every modality per channel with statistics fitted on the **train split only** (`--normalize train_zscore`, the default; `--normalize none` reproduces the old raw inputs). This applies to all models, including MUAD, and the fitted stats are saved to `normalization.json` in each run directory. Checked locally: in a batch, logs go from std ≈ 1e-10 to ≈ 1. The temporal-expert trace issue is left for Phase 2.

**Other readings:** UG-MoE is more robust than MUAD to metric noise (σ = 1: 0.966 vs 0.935 F1), but degrades faster in F1 under random masking (p = 0.6: 0.862 vs 0.925), while keeping a higher macro-F1 (0.723 vs 0.654). With one seed and both models effectively metric-only, none of these is a claimable result yet. Rerun after normalization.

Note: MUAD's clean test F1 here (0.9884) differs slightly from the comparison run (0.9882) because MUAD's vendored uncertainty block samples even in eval mode. Its eval is not deterministic; ours is (§3.6).

### 3.6 Uncommitted changes to the uncertainty block and model
`git diff` shows the working copy differs from HEAD (`d8bbcb9 "revert some bugs"`):
- `src/uncertainty/uncertainty.py`: HEAD returns **deterministic μ** with clamped log-variance. The working copy returns **reparameterized samples** with no clamping. With sampling there is no `self.training` check, so test-time predictions are random, and removing the clamp risks NaN/Inf.
- `tracks/proposed_model/model.py`: semantic-expert call and router output-shape handling changed.
- The bit-identical F1s in §3.5 suggest the saved runs used the **deterministic** version. Which version is intended has to be decided and committed.

### 3.7 Ablations don't isolate what their names say
(`experiments/run_ablations.py:30–58`)

| Ablation | What it actually changes vs. full |
|---|---|
| `no_graph` | Swaps `UncertaintyGuidedRouter` for the base `ExpertRouter` (which adds top-k and a load-balancing loss). Graph routing was never on in the full model. |
| `no_uncertainty` | Removes the per-expert uncertainty blocks and their KL. The uncertainty-guided router stays. |
| `no_fusion` | Swaps confidence fusion for `weighted_concat`. Fusion is still present. |
| `single_expert` | Temporal expert only. This one is a real "w/o MoE". |
| `plain_moe` | Base router, no graph, no uncertainty routing, `weighted_concat` fusion, **but uncertainty blocks still on** (unlike `run_plain_moe.py`, which also sets `use_uncertainty=False`). |
| `graph_transformer` | Dependency expert uses a graph transformer instead of GAT. |

Plan §8B needs: w/o graph, w/o MoE, w/o dynamic routing, w/o uncertainty routing, w/o probabilistic representation, w/o adaptive fusion. Of these, only "w/o MoE" (`single_expert`) maps cleanly.

### 3.8 Track A in `run_comparison.py` is not MUAD's training objective
`run_comparison.py:85–92` trains the MUAD model with `CrossEntropyLoss(MMlogit)` only. MUAD's own loss is `MMLoss + 0.6·confidence_loss + mean_KL` (`vendor/muad/models/main_model.py:111`). `run_track_a_full.py` does use MUAD's loss through `BaseModel.fit`, but that script has 2 seeds hard-coded (`run_track_a_full.py:114`) despite printing "5 runs". Track B, meanwhile, trains with its own KL and router terms. The comparison is therefore not like-for-like on the objective.

### 3.9 The dataset is easy and imbalanced; the metric hides this
`recovered_artifact`: train 3,366 / test 2,244 windows. Test labels: 384 normal (`culprit=0`) and 1,860 anomalous (`culprit ∈ {1,4,5}`), so **83% positive**. A constant "anomalous" predictor gets F1 = 0.906. Reported F1 is for the positive (anomalous) class only. Report normal-class or macro F1 and PR-AUC as well, and add a trivial baseline row. (`data_README.md` says "train chunks: 5,610". That is train + test combined; the doc is wrong.)

### 3.10 The proposed model is ~18× larger than the baselines
Counted from the registry builders: `ug_moe` 4.38M parameters, `plain_moe` 4.18M, `single_expert` 0.25M, MUAD 0.24M, GAT 0.30M, Graph-MoE 0.32M. Most of the size sits in the expert Transformers and cross-attention. Any gain over MUAD/GAT could come from capacity rather than from the architecture. Plan §13 also says "do not start with a very large expert architecture". Phase 2 should shrink the experts, or the paper should report a size-matched baseline.

**Decision (2026-11-17, user):** model size is not a goal in itself. Shrink the experts **only if the loss is negligible**. Planned as step 13b (after the Phase 2 fixes, so we size the final architecture): a single-seed sweep over `hidden_dim` / `expert_hidden_dim` / `fusion_dim` / `router_hidden_dim` (e.g. 64/128 → 32/64 → 16/32) and the expert depth. Pick the smallest config whose **validation** F1 and macro-F1 stay within ~0.5 points of the full-size model; test is evaluated only for the chosen config. With one seed, a gap of that size is within noise, so it would count as "no measurable loss", not as evidence that the two sizes are equal.

### 3.12 The artifact's train/test split leaks: stride-1 windows, split at random
Found on 2026-11-17 by reading Eadro's preprocessing (`vendor/eadro/codes/preprocess/align.py`), which produced `recovered_artifact`: the 8-character random chunk ids and `chunk_lenth=10` match. The raw SN fault files and a reconstruction of the chunk order confirm it.
- **Windows have stride 1:** `intervals = [(s, s+9) for s in range(start, end-9)]`. The 4 SN fault runs (≈1,410 s each) give 5,607 windows; the artifact has 5,610.
- **The split is a random shuffle** of those windows (`np.random.shuffle(chunk_idx)` in `split_chunks`).
- **Reconstruction:** consecutive windows share 9 of 10 metric steps, so the chunks chain unambiguously into exactly **4 runs** (1,402 / 1,402 / 1,404 / 1,402 windows, every chunk covered). See `common.reconstruct_runs`.
- **Leakage measured: 83.9% of test chunks (1,882 / 2,244) have a training chunk as their immediate, 1-second-shifted (90% identical) neighbour.** Every result so far, ours and MUAD's (and very likely the MUAD paper's), is measured on near-duplicates of training data. This explains why all models sit at F1 ≈ 0.99.
- **Labels decoded:** along a run the labels follow the raw injection order `cpu_load → network_delay → network_loss` per service. So **culprit = fault type: 0 normal, 1 cpu_load, 4 network_delay, 5 network_loss.** The binary `culprit >= 1` target is correct. 12 services are faulted (3 per run), 3 fault types each, 120 s per fault.
- **Fix (plan §13: "do not randomly split rows belonging to the same time window", "don't leak future information"):** `load_splits(split="run")`, now the default. Run 3 is test, run 2 is validation, runs 0–1 are training. Sizes: train 2,804 / val 1,404 / test 1,402, with class balance ≈ 17% normal in every split and all three fault types in each. Verified: 0 test windows overlap any training window. The test run's faults hit **services never faulted in training**, so the setting is realistic and harder. `--split random` reproduces the old leaky protocol for comparison. With more time, report all 4 test-run choices (leave-one-run-out).
- **Residual caveat:** Eadro z-scored metrics and traces per run over the whole run, test windows included, before chunking. That is a mild, label-free normalization leak we can't undo without regenerating from raw.
- **Unused data:** `SN_dataset/SN Dataset/no fault/` holds 3 fault-free runs that the artifact never used. They are a possible later fix for the class imbalance.

### 3.13 Smaller issues
- Hard-coded 12-node / 24-edge assumptions: `nn.Linear(24*2, …)` in the router, `range(12)` routing heads, `.view(B, 12, -1)`, `N = 12` in the baselines. These block running on TT (27 services) or the SN variant without code changes.
- ~~`run_ablations.py`: `final_roc_auc` / `final_pr_auc` reused `all_probs` from the last epoch loop~~ (fixed by the rewrite onto `common.py`).
- **`experiments/convert_sn_dataset.py` produces unusable data** (checked 2026-11-17 on its output, 451/113 chunks): `traces` `[12,1]` instead of `[12,10,1]`, `logs` `[12,1]` instead of `[12,14]`, 12 edges instead of 24, and **every chunk labelled `culprit=2`** (no normal class). A comparison run on it (server, 2026-11-14) crashed in both tracks. The chunks were deleted. Fix or rewrite the converter before any SN-from-raw or TT experiment.
- `experiments/convert_sn_dataset.py:31–32` and `validation/gates/gate01_raw_integrity.py` hard-code Windows absolute paths, so they will not run on the server.
- Checkpoints from the baseline runs were written to the server's `project/checkpoints/` (gitignored). Every seed overwrites the same file (e.g. `baseline_gat.pth`), so only the last seed's checkpoint survives.
- `EVAL_EVERY = 5` is defined in `run_comparison.py` but not used (evaluation runs every epoch).

---

## 4. Required experiments (plan §8): status

| Exp. | Requirement | Status | What exists |
|---|---|---|---|
| A | GAT, Plain MoE, Graph-MoE, Full | 🟡 | All four ran. Baselines: 2 seeds at lr=1e-3. Full: 5 seeds at lr=5e-4. Plain MoE and Full are the same class with a non-functional router. `GraphMoE` (`run_graph_moe.py`) is a separate model with real per-node top-k routing over class logits. |
| B | 6 component ablations | 🟡 | 6 ran (5 seeds), but they don't match the required list (§3.7). |
| C | Modality ablation (7 subsets, trained) | ❌ | No script. (Missing-modality at test time is a different experiment.) |
| D | Missing modality: 10/20/40/60% random masking + one modality at a time | 🟡 | One modality at a time only, at test time, 2 seeds. No percentage masking. |
| E | Noise on each modality and combinations | 🟡 | Each single modality, σ ∈ {0.01…0.5}, 2 seeds. No combinations. No relative-F1-degradation column. |
| F | Routing weights per anomaly type + entropy | 🟡 | Global mean weights and entropy (2 seeds, 30 epochs). **No per-anomaly-type breakdown** (the script never splits by `culprit`). Meaningless until §3.1 is fixed. |
| G | Uncertainty clean vs. noisy; fusion weights | 🟡 | Script exists (`run_uncertainty_fusion_analysis.py`, still has debug prints), but **no output in `runs/`**. Its "uncertainty" is the batch-scalar KL. |
| H | Hyperparameter sensitivity | ❌ | Nothing. |
| — | Cross-dataset (TT) | ❌ | Not started. |

### Current numbers (all on `recovered_artifact`; all "best F1" = **max over epochs on test**)

| Model | Seeds | lr | Best F1 (mean ± std) |
|---|---|---|---|
| Trivial "all anomalous" | — | — | 0.906 |
| GAT baseline | 2 | 1e-3 | 0.9868 |
| Graph-MoE baseline | 2 | 1e-3 | 0.9870 |
| Plain MoE (`run_plain_moe.py`) | 2 | 1e-3 | 0.9916 |
| Track A, MUAD (CE-only, in comparison) | 5 | 5e-4 | 0.9896 ± 0.0007 |
| **Track B, full UG-MoE** | 5 | 5e-4 | **0.9918 ± 0.0005** (paired t-test p = 0.005 vs A) |
| Ablation `no_graph` | 5 | 5e-4 | 0.9908 ± 0.0025 |
| Ablation `no_uncertainty` | 5 | 5e-4 | 0.9896 ± 0.0006 |
| Ablation `no_fusion` | 5 | 5e-4 | 0.9918 ± 0.0013 |
| Ablation `single_expert` | 5 | 5e-4 | 0.9835 ± 0.0021 |
| Ablation `plain_moe` | 5 | 5e-4 | 0.9901 ± 0.0025 |
| Ablation `graph_transformer` | 5 | 5e-4 | 0.9879 ± 0.0010 |

#### New protocol: validation-selected, test evaluated once (seed 42, `runs/comparison_20261117_104222`)

Split: train 2,861 / val 505 / test 2,244. lr 5e-4, 50 epochs, patience 15, no NaN batches skipped.

| Model | Best epoch (val) | Val F1 | Test F1 | P | R | F1 normal | Macro-F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| Trivial "all anomalous" | — | — | 0.9064 | 0.8289 | 1.0000 | 0.0000 | 0.4532 | 0.8289 | 0.5000 |
| Track A, MUAD (own objective) | 36 | 0.9941 | **0.9882** | 0.9882 | 0.9882 | 0.9427 | **0.9654** | 0.9980 | 0.9913 |
| Track B, UG-MoE (current code) | 49 | 0.9929 | **0.9879** | 0.9871 | 0.9887 | 0.9412 | **0.9645** | 0.9986 | 0.9945 |

Baselines on the same split, seed and hyperparameters (`runs/baselines_20261117_105856`):

| Model | Params | Best epoch (val) | Val F1 | Test F1 | P | R | F1 normal | Macro-F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| GAT baseline | 0.30M | 44 | 0.9929 | 0.9848 | 0.9798 | 0.9898 | 0.9239 | 0.9543 | 0.9989 | 0.9947 |
| Graph-MoE baseline | 0.32M | 46 | 0.9928 | 0.9866 | 0.9871 | 0.9860 | 0.9351 | 0.9608 | 0.9983 | 0.9933 |
| **Plain MoE** (no graph / UG routing, no uncertainty blocks, weighted-concat fusion) | 4.18M | 50 | 0.9953 | **0.9917** | 0.9909 | 0.9925 | **0.9595** | **0.9756** | 0.9992 | 0.9962 |

**Reading (baselines):** the best model on every metric is **plain MoE**, i.e. UG-MoE *with its proposed components switched off*. It beats full UG-MoE by +0.38 F1, +1.83 normal-class F1 and +1.11 macro-F1 points (seed 42 only, but the macro-F1 gap is above the ~1-point single-seed threshold). The components that make up the "UG" part (variational uncertainty blocks plus KL, the uncertainty-guided router with its undocumented loss term (§3.3), confidence fusion) are currently **hurting**. Phase 2 therefore has to show that each component, once fixed, adds value over plain MoE. That is the plan's own incremental order (§12 steps 6→10). Plain MoE also peaked at the last epoch (50), so a longer budget is needed in the next round.

**Reading (Track A vs B):** with model selection moved off the test set, **the claimed UG-MoE advantage disappears.** On F1 and macro-F1 Track B is level with MUAD or marginally below it (−0.03 / −0.09 points, well within single-seed noise). It is slightly higher only on the ranking metrics (PR-AUC +0.06, ROC-AUC +0.32 points), and that is not a significant edge either. This happens with an 18× larger model. It is consistent with §3.1–3.3: as currently wired, the MoE parts contribute nothing. Track B's best epoch is 49 of 50, so it may still be improving; allow more epochs in the next comparison. The old-protocol numbers below (0.9896 vs 0.9918, "p = 0.005") should be treated as an artifact of test-set selection.

#### Round 2: same protocol + train-split z-score normalization (seed 42, 2026-11-17)

`runs/round2_tracks/baselines_20261117_115635` (MUAD, UG-MoE) and `runs/round2_baselines/baselines_20261117_115645` (others). Same split, 50 epochs, lr 5e-4.

| Model | Test F1 | P | R | F1 normal | Macro-F1 | PR-AUC | ROC-AUC | Δ macro-F1 vs raw inputs |
|---|---|---|---|---|---|---|---|---|
| Trivial "all anomalous" | 0.9064 | 0.8289 | 1.0000 | 0.0000 | 0.4532 | 0.8289 | 0.5000 | — |
| GAT | 0.9869 | 0.9829 | 0.9909 | 0.9349 | 0.9609 | 0.9993 | 0.9964 | +0.66 |
| Graph-MoE | 0.9898 | 0.9861 | 0.9935 | 0.9496 | 0.9697 | 0.9992 | 0.9962 | +0.89 |
| MUAD (Track A) | 0.9882 | 0.9866 | 0.9898 | 0.9423 | 0.9652 | 0.9985 | 0.9935 | −0.02 |
| UG-MoE (Track B) | 0.9857 | 0.9881 | 0.9833 | 0.9318 | 0.9588 | 0.9991 | 0.9954 | −0.57 |
| **Plain MoE** | **0.9906** | 0.9882 | 0.9930 | **0.9539** | **0.9723** | 0.9988 | 0.9949 | −0.33 |

Robustness on the round-2 checkpoints (`runs/round2_tracks/robustness_*`):

| Perturbation (test) | UG-MoE F1 / macro-F1 | MUAD F1 / macro-F1 |
|---|---|---|
| Clean | 0.9857 / 0.9588 | 0.9882 / 0.9652 |
| Drop metrics / traces / logs | 0.9064 (trivial) / **unchanged** / **unchanged** | 0.9064 (trivial) / 0.9879 / **unchanged** |
| Random mask p = 0.6 | 0.8774 / 0.6672 | 0.9340 / 0.7213 |
| Noise on traces or logs, σ ≤ 2 | unchanged (logs: +0.0008 at σ = 2) | unchanged (±0.0005) |
| Noise on metrics, σ = 1 / 2 | 0.9605 / 0.9085 | 0.9313 / 0.9047 |

**Reading:** normalization made logs numerically visible, but **neither model started using traces or logs.** So scale was not the only cause. Metrics alone already reach about 0.99 on this dataset, and the models take that shortcut. Changes from normalization are within single-seed noise. The ranking is unchanged: plain MoE is still the best and UG-MoE is still below plain MoE and MUAD. **Next diagnostic:** the plan's Exp. C (modality ablation, *trained* on each subset), which shows whether traces and logs carry standalone signal. It is implemented as the registry suffix `<model>@<mods>`, e.g. `muad@traces+logs`.

#### Old protocol (test-set selection, for reference only)

**Do not put these numbers in the paper.** Plain MoE with no graph and no uncertainty (0.9916) and `no_fusion` (0.9918) match the full model, which is what you would expect given §3.1–3.3.

---

## 5. Required outputs (plan §10)

| Output | Status |
|---|---|
| Preprocessed dataset with documented schema | 🟡 `recovered_artifact` + `data_README.md` (schema partly documented, provenance not reproduced) |
| Service dependency graph file | ❌ (embedded in `metadata.json` only) |
| GAT / Plain MoE / Graph-MoE / Full checkpoints | 🟡 Only on the server; last seed only; gitignored; full-model checkpoints not saved by `run_comparison.py` |
| Overall performance table | 🟡 Exists in JSON; not valid (§3.4) |
| Ablation table | 🟡 Exists; not valid (§3.7) |
| Missing/noisy robustness plots | ❌ JSON only, no plots |
| Expert-weight heatmaps | 🟡 Histograms per seed (`routing_analysis_*.png`), no per-anomaly heatmap |
| Uncertainty / fusion-weight plots | ❌ |
| Sensitivity plots | ❌ |
| Reproducible evaluation script regenerating all tables/plots | ❌ |

## 6. Completion criteria (plan §14)

| Criterion | Status |
|---|---|
| End-to-end from dataset files to predictions | 🟡 From pre-built chunks, yes. From raw SN, not reproducibly (different `event_num`). |
| GAT baseline reproducible | 🟡 Runs; 2 seeds; only last checkpoint kept |
| Plain MoE and Graph-MoE under the same protocol | ❌ Different lr and seed count from the full model; test-set selection |
| Full model runs without numerical instability | 🟡 Runs, but needs SDPA workarounds, NaN-skip logic and grad clipping 0.5. The current working copy removed the clamps. |
| All ablations completed | ❌ (wrong set) |
| Missing-modality and noise completed | 🟡 Partial |
| Sensitivity plots | ❌ |
| Metrics reproducible from checkpoints + scripts | ❌ |

---

## 7. Out of scope or beyond the plan

- **HDFS log-only mode: removed 2026-11-17.** The plan covers **multimodal** detection only, and the log-only branches went through the exact files Phase 2 rewrites (`model.py`, `router.py`, experts). Removed: `log_only_mode` and its parameters, `src/experts/log_expert.py`, `preprocessing/hdfs/`, `experiments/run_hdfs.py`, `configs/hdfs_full.json`, `project/runs/` (old HDFS runs), and the HDFS-derived copies in `preprocessing/sn/`. It can be recovered from git history or the `backup-hdfs` branch if the supervisor asks for a log-only experiment.
- **Active learning** (`run_track_a_al.py`). Plan §13: "do not add active learning to the first implementation". It's fine as a MUAD reproduction, but it isn't part of UG-MoE.
- **Framework:** the plan suggests PyTorch Geometric; the code uses DGL (inherited from MUAD). This is an acceptable deviation, but mention it.
- **Project layout** differs from plan §5 (`src/` + `tracks/` + `experiments/` instead of `models/` + `scripts/`). This is fine.

---

## 8. What's next (proposed order)

Each phase should end with a commit pushed to GitHub and a server run whose outputs we check before moving on.

### Phase 0: Decide and stabilize (local, no training)
1. **Settle the uncertainty-block version** (deterministic μ at eval and sampling at train, with clamping, is the usual choice). Commit it, and make `forward` respect `self.training`.
2. ~~HDFS scope~~ (removed, §7). Active-learning scope still to confirm with the supervisor.

### Phase 1: Evaluation protocol (before any further results)
3. Add a **validation split** from the train set, at window level and time-ordered if timestamps can be recovered. Select the epoch/checkpoint on **validation** F1 or PR-AUC. Report test once, at the selected checkpoint.
4. Use one shared training/eval loop (one module used by every script) so every model gets the same seeds (5), lr, epochs, early stopping and metrics. Currently each script duplicates its own loop.
5. Report P, R, F1 (anomaly class), macro-F1, PR-AUC, ROC-AUC, plus a trivial-baseline row. Save the per-seed best checkpoint under a per-run directory.
6. Make Track A use MUAD's real objective in the comparison (or call `BaseModel`), with 5 seeds.

### Phase 2: Make the model match the plan
7. **Routing actually used:** `H = Σ g_k · E_k` feeds the probabilistic head and the classifier path (pseudocode lines 7–10).
8. **Graph-aware routing:** route using a real graph signal, e.g. pooled GAT or node-level features of the dependency graph, or per-node routing. Drop the flattened edge-ID vector and the broken `GraphAwareRouter`.
9. **Uncertainty-guided routing:** pass **per-sample** σ (e.g. mean predicted variance per expert, not the batch KL) into the router, and remove the router's self-invented uncertainty heads and the undocumented `0.01 * mean(uncertainty)` loss term. Keep `L = L_cls + λ_KL · L_KL`, and add a load-balancing term only if justified and reported separately (plan §7).
10. **Modality-level fusion:** produce `z_log, z_trace, z_metric`, learn `w_log, w_trace, w_metric` (conditioned on uncertainty), then `z_f = Σ w_m z_m` (plan Step 12). Log these weights.
11. **Diagnose the metric-only behaviour (§3.5):** per-modality probes, gradient and feature statistics for traces/logs, checks for near-constant trace/log inputs. Add modality dropout during training if needed (`src/fusion/fusion.py` already has `ModalityDropout`).
12. Remove the hard-coded 12/24 assumptions so TT (27 services) can run.
13. Re-run Plan §12 steps 4–10 in order as quick single-seed checks on the server: simple MLP, then GAT, then plain MoE, then +graph routing, then +uncertainty, then +UG routing, then +fusion. Confirm each addition changes behaviour.

13b. **Size sweep (§3.10):** shrink experts and dims to the smallest config with negligible validation loss. Single seed; keep the full size if the loss isn't negligible.

### Phase 3: Freeze, then run the experiments (plan §8)
14. Freeze the model (tag a commit). Rewrite ablations to exactly match §8B, each flag removing exactly one component.
15. Baseline comparison (A) under the shared protocol, 5 seeds.
16. Modality ablation (C), trained on each of the 7 subsets.
17. Missing modality (D): 10/20/40/60% random masking plus single-modality drop. Noise (E): single and combined modalities, with relative F1 degradation.
18. Routing analysis (F) per anomaly type (`culprit` 1/4/5) as heatmaps, with entropy. Uncertainty/fusion analysis (G): clean vs. noisy.
19. Sensitivity (H): number of experts, embedding dim, window size, λ_KL, routing temperature, lr.
20. Cross-dataset: TT from the same Zenodo record.

### Phase 4: Reproducibility and paper
21. A single `evaluate.py` / `make_tables.py` that regenerates all tables and plots from saved checkpoints and result JSONs.
22. Only then write the paper (the opencode draft was deleted), using numbers from this pipeline only.

---

## 9. Open questions for the supervisor
1. Is `recovered_artifact` (the MUAD authors' pre-built SN chunks) acceptable as the primary dataset, or must the data be regenerated from raw SN with our own pipeline? Our pipeline currently produces `event_num=14` and a different number of chunks.
2. Given that 83% of windows are anomalous and F1 is around 0.99 everywhere, should we move to a harder setting (e.g. a time-ordered split, normal-class F1, or the TT dataset as the main benchmark) so that differences between models mean something?
3. Should routing be per sample (graph-level) or per node (service-level)?
4. ~~Is HDFS in scope?~~ Removed as out of plan scope (§7); confirm this is acceptable.

---

## 10. Repository clean-up backlog (plan for a clean research codebase)
- ~~Untrack `.pyc`; remove `project/project/`, `run2`, `.swp`, opencode-era `.md` files~~ (done 2026-11-17).
- Stray empty `C:Usersas999OneDriveDesktoptrashimpl/` folder at the repo root: untracked, delete by hand.
- ~~HDFS code, `project/runs/`, `UG-MoE-Research-Paper-Draft.md`~~ (removed 2026-11-17). The raw `hdfs_dataset/` (1.6 GB, untracked, gitignored) is still on the local disk; delete it by hand if no longer needed.
- Old-protocol results in `project/experiments/runs/*_202611{11,12}_*`: keep as the "before" reference until the new-protocol runs replace them, then delete.
- Replace hard-coded Windows paths (`convert_sn_dataset.py`, `gate01`) with CLI arguments.
- Fix `pyproject.toml` dependencies (TOML table → array) so `pip install -e .` works on the server.
- Replace the ad-hoc test scripts with a small `pytest` suite (shape and forward-pass smoke tests on a handful of chunks) that can run locally on CPU.
