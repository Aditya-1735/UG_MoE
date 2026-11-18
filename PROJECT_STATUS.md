# UG-MoE — Project Status vs. Implementation Plan

**Last updated:** 2026-11-17 · **Branch:** `hdfs` · **Plan:** `7. UG-MoE_Implementation_Guide.md` · **Baseline paper:** MUAD (`7. Uncertainty-Aware_Multimodal_Anomaly_Detection_..._Active_Learning.pdf`)

> **How this was assessed.** Only the Python code, the data files, and the result JSONs in `project/experiments/runs/` were treated as evidence. Claims in the older `.md` files (opencode-era plans, `UG-MoE-Research-Paper-Draft.md`, status tables marked "Complete") were **not** taken at face value. Every finding below points to the code that supports it. Line numbers refer to the working copy on 2026-11-17.

---

## 0. Progress log

| Date | Change | Server-verified? |
|---|---|---|
| 2026-11-17 | **Protocol fix (model selection and TT split), within the plan** (§6 S14: "best checkpoint based on validation F1 or PR-AUC"). (1) `common._selection_key` selects on **val macro-F1** (tie: PR-AUC) instead of anomaly-class F1, which the trivial predictor can win on anomaly-heavy validation sets. (2) When a dataset has fewer than 3 fault-free runs (TT: 2), each one is cut chronologically into 60 / 20 / 20% train / val / test segments (`segment_normal_runs`, windows straddling a cut dropped: 8 on TT), so every split contains normal operation. New TT split: train 8,397 / val 3,228 / test 3,226 windows, val and test ~68% anomalous (trivial macro-F1 0.40). SN keeps whole-run assignment (unchanged). **Invalidates** round 5 TT. On SN the two rules select almost the same epochs, so round 5/6 SN stand. Known split caveats, deferred to the final paper runs (time): fault-free test segments come from the same runs as fault-free train segments (TT only); no gap at the cuts (adjacent windows share no seconds); fault-free runs were recorded after the fault runs; Drain3 templates are mined on all runs (label-free). Planned hardening: 60 s gap at cuts, templates mined on train runs only. | ❌ pending (server rerun) |
| 2026-11-18 | **FINAL 5-seed main results + ablations** (§4): v2 full vs MUAD **+4.7 (SN, 0.972 vs 0.925) / +9.7 (TT, 0.723 vs 0.626)** macro-F1; vs GAT +4.1 / +6.3. Full ≈ plain on SN, +2.5 on TT (noisy). Adaptive fusion is the only component that helps on both datasets (−2.4 / −3.3 when removed). Most single-seed ablation effects were seed noise. Robustness over 5 seeds pending. | ✅ |
| 2026-11-18 | **Rounds 16b / 16c (sensitivity, SN):** KL weight, routing temperature, learning rate and window all stay within 3.4 macro-F1 of the default, and the default is best in each (§4). Plan 8H complete on one seed. **Final multi-seed runs started** (seeds 42, 123, 456, 789, 999; `final_*` run directories; `make_report.py` pools seeds per (model, seed) and reuses the seed-42 runs that already used the final protocol). | ✅ |
| 2026-11-18 | **Reporting:** `experiments/make_report.py` regenerates every table (`tables.md`) and figure (robustness curves, sensitivity panels, fusion weights with trace/log zoom, uncertainty change vs clean, routing heatmaps) from the saved runs listed in its `MANIFEST` (plan §10). Smoke-tested locally on synthetic run files. **Decision (user): modality dropout deferred.** It is not in the implementation guide; revisit only if time allows. The "fails without metrics" result is reported as a limitation. | n/a |
| 2026-11-17 | **Rounds 12, 15b, 16a** (§4): modality ablation, every modality alone ≥ 0.83 on SN (traces 0.935), all three 0.987 (+2.8 over the best pair). TT ablations disagree with SN (TT needs graph / dynamic routing, SN needs MoE / UG routing / fusion); only the probabilistic representation matters on both; TT single-seed ablations unreliable. Sensitivity: ≥ 2 experts needed; hidden 64 best (32 −0.9, 128 −1.4), so keep 64. | ✅ |
| 2026-11-17 | **Rounds 11, 14, 15a** (§4): w/o probabilistic representation loses 12.8 / 14.9 F1 at 20 / 40% masking on SN (same clean accuracy), so that component causes the robustness. TT: baselines miss network faults (recall 0.30–0.65 vs 0.88–0.90 for v2, at FPR 0.30 vs 0.13–0.17). TT robustness: v2 best up to 40% masking but the margin vanishes; uncertainty is overconfident under log noise and doesn't flag errors on TT. | ✅ |
| 2026-11-17 | **Round 9 (v2 ablations, SN)** and **round 10 (MUAD at paper hyperparameters, SN)**, server, seed 42 (§4): removing MoE / adaptive fusion / uncertainty routing costs 7.8 / 4.1 / 3.9 macro-F1; graph, dynamic routing and probabilistic representation are within noise on clean SN. MUAD with lr 1e-3 / 120 epochs: 0.793 vs 0.922 with our settings, so our settings favour MUAD. | ✅ |
| 2026-11-17 | **Round 7 (TT, v2 steps)**, server, seed 42 (§4): **full v2 0.823 macro-F1 vs GAT 0.730, MUAD 0.647, v2 plain 0.667**; recall 0.92 vs ≤ 0.70 for every baseline. Uncertainty-guided routing is the decisive step (ug_routing 0.840). First clean-accuracy gain from the UG components. | ✅ |
| 2026-11-17 | **Exp. F + G** (`experiments/run_analysis.py`, round 13, SN): uncertainty rises and the fusion weight falls for a corrupted modality (works in direction), but metrics keep ≥ 88% of the weight even when missing (too weak to fall back). Uncertainty is higher on wrong predictions. Metric routing specializes by fault type (normal → GAT expert 0.87, cpu → cross-modal 0.95, network → temporal). | ✅ |
| 2026-11-17 | **Round 8 (robustness, SN)**, server, seed 42 (§4): UG-MoE v2 beats MUAD under masking and noise (p = 0.4 masking: 0.856 vs 0.716 F1; metric noise σ = 1: 0.964 vs 0.856), with about half the relative drop. Both collapse to trivial without metrics. **Attribution over the §12 steps:** full ≈ plain on clean data, but **full beats plain by +10.7 / +13.4 F1 at 20 / 40% masking**, and plain is no better than MUAD there. The robustness comes from the probabilistic representation plus adaptive fusion. | ✅ |
| 2026-11-17 | **Round 5 (rebuilt SN / TT)**, server, seed 42 (§4): SN macro-F1 legacy UG-MoE 0.984 > Graph-MoE 0.973 > plain MoE 0.969 > GAT 0.965 > MUAD 0.922. TT: 4 of 5 models collapsed to "all anomalous" because of the selection bug above; results void. | ✅ (TT void) |
| 2026-11-17 | **Phase 0.1:** uncertainty blocks (`src/uncertainty/uncertainty.py`) now sample with reparameterization only in `train()` mode and return deterministic μ in `eval()`. Variance / log-variance clamps restored. Working-copy `model.py` semantic-expert fix (`expert(graph.ndata["logs"], B)`) kept. Fixes §3.6. | ❌ pending |
| 2026-11-17 | **Phase 1 (part):** new shared protocol `project/experiments/common.py`: stratified val split from `chunk_train.pkl` (fixed `split_seed=2026`, 15%, ids saved to `split.json`), selection on val F1 (tie: PR-AUC), early stopping on val, test evaluated once, per-seed checkpoints, metrics incl. macro-F1 / normal-class F1 / trivial baseline. `run_comparison.py` rewritten on top of it, with Track A using MUAD's real objective. Fixes §3.4 and §3.8 **for the comparison script only**. | ❌ pending |
| 2026-11-17 | **Decision:** because of time constraints, all experiments run with a **single seed (42)** for now. Multi-seed runs (5 seeds plus paired t-test) are deferred to the final paper runs (Phase 3–4). Single-seed differences below about 1 F1 point are not evidence of improvement. | n/a |
| 2026-11-17 | **Phase 1 (cont.):** `experiments/registry.py` (single definition of every model / ablation + forward adapter + `run_suite`). GAT and Graph-MoE model classes moved to `tracks/baselines/` (node count now read from the graph, not hard-coded). New scripts on the shared protocol: `run_baselines.py` (Exp. A), `run_ablations.py` (rewritten; always trains the `ug_moe` reference too), `run_robustness.py` (Exp. D + E as specified: single-modality drop, random 10/20/40/60% masking, noise scaled to train std on all 7 modality combinations, relative F1 drop). `run_comparison.py` now builds models through the registry. **Removed** (superseded): `run_gat_baseline.py`, `run_plain_moe.py`, `run_graph_moe.py`, `run_track_b_full.py`, `run_missing_modality.py`, `run_noise_robustness.py`. Not migrated yet: routing / uncertainty-fusion analysis (they depend on Phase 2), `run_track_a_al.py`, `run_hdfs.py`. | ❌ pending |
| 2026-11-17 | **First new-protocol result** (server): Track A vs Track B, seed 42. Test F1 0.9882 vs 0.9879, macro-F1 0.9654 vs 0.9645, so there is **no advantage** for UG-MoE once selection moves off the test set (§4). This is the "before" baseline for the Phase 2 fixes. | ✅ |
| 2026-11-17 | **Phase 2: UG-MoE v2** (`tracks/proposed_model/ug_moe_v2.py`), a rewrite following the guide's pseudocode: per-modality routers fed by graph context and per-expert uncertainty, the routed mixture feeds the probabilistic head and the classifier (fixes §3.1), real graph-aware (§3.2) and uncertainty-guided (§3.3) routing, modality-level uncertainty-aware fusion, loss = CE + λ_KL·KL with no hidden terms. 166k parameters (vs 4.4M, §3.10). Plan §8B ablations and §12 incremental steps are registry entries (`abl2:*`, `ug_moe2:*`). Local checks: all variants run on SN and TT shapes, eval is deterministic, every parameter gets a gradient, and perturbing a router or an expert's uncertainty changes the logits. The legacy `ug_moe` is kept only for reproducing old runs. | ❌ pending (server run) |
| 2026-11-17 | **Round 4 (leak-free run split, `recovered_artifact`)**, server, seed 42: UG-MoE 0.9332 / MUAD 0.9324 / plain MoE 0.9124 / GAT 0.5530 / Graph-MoE 0.4961 test macro-F1. **GAT and Graph-MoE collapse to near-trivial** once memorization is impossible; UG-MoE ≈ MUAD > plain MoE. Traces-only MUAD reaches val macro-F1 ≈ 0.77, so traces do carry signal (§4). | ✅ (modality runs still in progress) |
| 2026-11-17 | **New data pipeline** (`project/data_pipeline/`): fetches SN + TT from Zenodo (MD5-verified, both match), rebuilds windowed chunks from raw with per-run clock-offset detection, raw metrics, span latency, Drain3 log counts and run/time metadata per chunk, plus chronological run-level splits (`common.assign_pipeline_runs`). Built and checked locally (data only): **SN** 7 runs → 9,907 chunks (53% normal, now including the 3 fault-free runs), trivial test F1 0.58; **TT** 11 runs → 14,859 chunks (stride 5), trivial test F1 0.685, but **TT traces cover only 2–4% of seconds**. Models made size-agnostic (no hard-coded 12 nodes / 24 edges); all 5 models run on TT; SN parameter counts unchanged. **Deleted** the old SN pipelines and data (`preprocessing/`, `parsed_data/`, `validation/`, Eadro chunk copies, `convert_sn_dataset.py`, `DATASET_VERIFICATION_REPORT.md`). | ❌ pending (server run) |
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
- **Confirmed from the MUAD paper itself** (§IV-A, for the paper's related-work discussion): "We randomly selected 60% of the data … for training and used the remaining 40% for testing, ensuring that the ratio of normal to abnormal samples remains consistent", "Sliding window length = 10, stride = 1". Dataset C = Eadro SN with "15 log templates" (= our `event_num` 15); the artifact's 3,366 / 2,244 is exactly 60 / 40, so `recovered_artifact` *is* the paper's split. No validation set or epoch-selection rule is described (the vendored code selects on test). Reported MUAD F1 on C: 0.9928; ours under the same split: 0.988 (reproduced). The paper's Table III "Only Log" on C (recall 1.000, precision 0.8289, F1 0.9064) is the trivial all-anomalous predictor, and "Only Metric" reaches 0.9903, so on C almost all of the reported F1 comes from class imbalance + leakage + metrics.
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

#### Round 5: rebuilt Eadro SN / TT, chronological run split (seed 42, 2026-11-17)

`runs/round5_sn/`, `runs/round5_tt/`. Selection was still on **val anomaly-F1** (the old rule). Test macro-F1:

| Model | SN F1 | SN macro-F1 | SN PR-AUC | TT F1 | TT F1 normal | TT macro-F1 | TT PR-AUC | TT ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| Trivial "all anomalous" | 0.5787 | 0.2893 | 0.4071 | 0.6852 | 0 | 0.3426 | 0.5212 | 0.5 |
| UG-MoE (legacy) | **0.9813** | **0.9843** | **0.9980** | 0.6852 | 0 | 0.3426 (= trivial) | 0.8306 | 0.7626 |
| Graph-MoE | 0.9681 | 0.9733 | 0.9936 | 0.6852 | 0 | 0.3426 (= trivial) | 0.8448 | 0.7891 |
| Plain MoE | 0.9624 | 0.9690 | 0.9955 | 0.6852 | 0 | 0.3426 (= trivial) | 0.8315 | 0.7561 |
| GAT | 0.9579 | 0.9650 | 0.9940 | 0.6853 | 0.001 | 0.3432 (≈ trivial) | **0.9167** | **0.8998** |
| MUAD | 0.9041 | 0.9219 | 0.9691 | **0.7091** | **0.7471** | **0.7281** | 0.8453 | 0.7979 |

**SN:** every model learns, so the "GAT / Graph-MoE collapse" of round 4 does not reproduce on the rebuilt data. The legacy UG-MoE is best (+6.2 macro-F1 over MUAD, +1.1 over Graph-MoE), and MUAD is last. SN is easy (test = 1 fault + 1 fault-free run).

**TT is invalid: a protocol bug, not a model result.** Four of five models output "anomalous" for every test window (F1 identical to the trivial predictor), yet they rank well (PR-AUC 0.83–0.92, GAT's ROC-AUC 0.90). Cause: the TT validation split had **no fault-free run** (2 fault-free runs, round-robin train / test), so validation was **84.8% anomalous** and the trivial predictor scored **val F1 0.918**. Selecting on anomaly-F1 therefore picked degenerate all-anomalous epochs. MUAD escaped only because its objective never passes through that state. Fixed (log entry below); TT must be rerun. The same weakness probably affected round 4 (validation run 2 is ~83% anomalous; GAT and Graph-MoE were selected at epochs 4 and 2), so the "memorization" reading of round 4 is uncertain until it is rerun with the new rule.

#### Round 6: UG-MoE v2 incremental steps (plan §12) on rebuilt SN (seed 42, 2026-11-17)

`runs/round6_v2_sn/`, selection on val anomaly-F1. On SN that rule picks the same epochs as macro-F1 for every step except `ug_routing` (epoch 20 instead of 12, a tie on val macro-F1 at 0.9775), so these results are valid.

| Step (adds) | Params | Test F1 | Recall | F1 normal | Macro-F1 | PR-AUC |
|---|---|---|---|---|---|---|
| plain (experts + router) | — | 0.9834 | 0.9740 | 0.9888 | 0.9861 | 0.9972 |
| + graph (GAT context, dependency expert) | — | 0.9603 | 0.9307 | 0.9741 | 0.9672 | 0.9956 |
| + prob (μ, σ, KL) | — | 0.9363 | 0.9030 | 0.9588 | 0.9475 | 0.9929 |
| + ug_routing | — | 0.9340 | 0.8944 | 0.9578 | 0.9459 | 0.9936 |
| **full (+ adaptive fusion)** | 166k | **0.9842** | 0.9732 | **0.9894** | **0.9868** | **0.9984** |
| *ref: legacy UG-MoE (round 5)* | 4.4M | 0.9813 | 0.9775 | 0.9873 | 0.9843 | 0.9980 |
| *ref: MUAD (round 5)* | 0.24M | 0.9041 | 0.8571 | 0.9397 | 0.9219 | 0.9691 |

**Reading:** the full v2 model is the best model on SN (macro-F1 0.987, +6.5 over MUAD, level with the legacy UG-MoE at 1/25 of its size), but **it is not better than `plain`** (+0.07, noise). The intermediate steps lose 2–4 points, all through recall, and validation macro-F1 is ~0.98 for every step, so this is a val→test generalization gap rather than worse fitting. Adaptive fusion recovers all of it. Components only pay off together, so the §12 "each step adds value" story does not hold on SN. SN is near ceiling. TT (harder, 2 test fault runs) and the robustness experiments (Exp. D/E, where uncertainty-aware fusion should matter) are the decisive tests.

#### Round 7: rebuilt TT with the fixed protocol (seed 42, 2026-11-17)

`runs/round7_tt/baselines_20261117_164202` (val macro-F1 selection, fault-free runs segmented). Test is
67.9% anomalous.

| Model | Test F1 | Precision | Recall | F1 normal | Macro-F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|---|
| Trivial | 0.8087 | 0.6789 | 1.0000 | 0 | 0.4044 | 0.6789 | 0.5 |
| **GAT** | **0.7854** | 0.8983 | **0.6977** | **0.6740** | **0.7297** | **0.9416** | **0.8934** |
| Plain MoE (legacy) | 0.6810 | 0.9845 | 0.5205 | 0.6559 | 0.6685 | 0.9089 | 0.7951 |
| MUAD | 0.6801 | 0.8975 | 0.5475 | 0.6145 | 0.6473 | 0.9169 | 0.8411 |
| Graph-MoE | 0.6494 | 0.9340 | 0.4977 | 0.6197 | 0.6345 | 0.9346 | 0.8742 |
| UG-MoE (legacy) | 0.8085 | 0.6788 | 0.9995 | 0 | 0.4042 (= trivial) | 0.8333 | 0.6466 |

**Reading:** TT is much harder than SN. Every model except the legacy UG-MoE (which never left the
all-anomalous state; don't build on it) is **precise but misses 30–50% of anomalous windows**, so
anomaly-F1 falls *below* the trivial predictor while macro-F1 is far above it. Ranking is good (PR-AUC
0.91–0.94), but validation macro-F1 at the selected epoch (~0.83) is 10–19 points above test, and the
validation scores swing strongly between epochs (e.g. v2 plain: 0.60 ↔ 0.84). Whether some fault types
are simply invisible on the test runs is now measured: `evaluate` reports per-fault-type recall
(`recall_type<c>`) and `fpr_normal`, printed by `run_robustness.py` on the clean test set.

#### Round 7 (v2): UG-MoE v2 incremental steps on TT (seed 42, 2026-11-17)

`runs/round7_v2_tt/baselines_20261117_164211`, same split and protocol as the TT baselines above.

| Model | Test F1 | Precision | Recall | F1 normal | Macro-F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|---|
| Trivial | 0.8087 | 0.6789 | 1.0000 | 0 | 0.4044 | 0.6789 | 0.5 |
| v2 plain | 0.6949 | 0.9310 | 0.5543 | 0.6396 | 0.6673 | 0.9243 | 0.8443 |
| + graph | 0.7131 | 0.9577 | 0.5680 | 0.6622 | 0.6876 | 0.9330 | 0.8582 |
| + prob | 0.5676 | 0.9360 | 0.4073 | 0.5893 | 0.5785 | 0.8720 | 0.7193 |
| **+ ug_routing** | **0.9163** | 0.8478 | **0.9968** | **0.7635** | **0.8399** | 0.9505 | 0.9053 |
| **full** | 0.8943 | 0.8668 | 0.9237 | 0.7521 | 0.8232 | **0.9527** | **0.9101** |
| *ref: GAT (best baseline)* | 0.7854 | 0.8983 | 0.6977 | 0.6740 | 0.7297 | 0.9416 | 0.8934 |
| *ref: MUAD* | 0.6801 | 0.8975 | 0.5475 | 0.6145 | 0.6473 | 0.9169 | 0.8411 |

**Reading:**
- **On TT the full model is clearly best:** macro-F1 0.823 vs GAT 0.730 (+9.4), MUAD 0.647 (+17.6) and
  v2 plain 0.667 (+15.6). It is the only model family whose anomaly-F1 beats the trivial predictor
  (0.894 vs 0.809), because it **recovers the missed anomalies** (recall 0.92 vs 0.55–0.70 for every
  baseline) at a moderate precision cost. It also has the best ranking (PR-AUC 0.953, ROC-AUC 0.910).
- **Uncertainty-guided routing is the decisive step on TT** (+26 macro-F1 over `prob`, +17 over `plain`).
  Adding the probabilistic representation alone hurts (0.579). Full vs ug_routing (−1.7) is close to
  single-seed noise, given TT's unstable validation curves.
- Together with SN: on the easy dataset (SN, near ceiling) the uncertainty components add robustness but
  no clean accuracy; on the harder dataset (TT) they add **both**. Single seed, and TT's validation
  scores swing strongly between epochs, so this needs the 5-seed final runs before it is a claim.

#### Round 8: robustness (Exp. D/E) on rebuilt SN, UG-MoE v2 full vs MUAD (seed 42, 2026-11-17)

`runs/round8_rob_sn_v2/`, `runs/round8_rob_sn_muad/` (round 6 / round 5 checkpoints; clean scores reproduce exactly: v2 0.9842 F1, MUAD 0.9037, MUAD varies by ±0.0005 because it samples at test time). Test F1, with relative drop in brackets.

| Perturbation | UG-MoE v2 full | MUAD |
|---|---|---|
| Clean | 0.9842 (macro 0.9868) | 0.9037 (macro 0.9216) |
| Drop metrics | 0.5787 = trivial (−41.2%) | 0.5787 = trivial (−36.0%) |
| Drop traces | 0.9793 (−0.5%) | 0.9009 (−0.3%) |
| Drop logs | 0.9811 (−0.3%) | 0.8968 (−0.8%) |
| Random mask p = 0.1 / 0.2 / 0.4 / 0.6 | 0.962 / 0.931 / 0.856 / 0.748 (−2.3 / −5.5 / −13.0 / −24.0%) | 0.857 / 0.809 / 0.716 / 0.662 (−5.2 / −10.5 / −20.8 / −26.8%) |
| Macro-F1 at p = 0.6 | 0.763 | 0.584 |
| Noise on metrics σ = 1 / 2 × std | 0.964 / 0.872 (−2.1 / −11.4%) | 0.856 / 0.742 (−5.3 / −17.9%) |
| Noise on all modalities σ = 1 / 2 | 0.960 / 0.790 (−2.5 / −19.8%) | 0.841 / 0.699 (−7.0 / −22.7%) |
| Noise on traces / logs only, σ = 2 | 0.977 / 0.983 | 0.895 / 0.900 |

**Reading:**
- **UG-MoE v2 is more robust than MUAD under every perturbation except dropping a whole modality:** higher absolute F1 everywhere, and a relative drop about half MUAD's under masking (p ≤ 0.4) and metric noise (σ ≤ 1).
- **Both models fail completely without metrics** (trivial predictor). Traces and logs are now used a little (dropping them costs 0.3–0.8 points; before normalization exactly 0), but neither model can fall back on them. So uncertainty-aware fusion does not give graceful degradation for a missing *dominant* modality. That is the plan's Exp. D claim, and it is not met by either model. The standard remedy is modality dropout during training, which would be a deviation (rule 7), so it needs a proposal first.
- **Attribution (all §12 steps, same checkpoints as round 6, `runs/round8_rob_sn_<step>/`).** The masks
  and noise come from the same condition seeds for every model, so the comparison is paired. Test F1
  (relative drop):

| Model | Clean | Mask 10% | Mask 20% | Mask 40% | Mask 60% | Drop logs | Noise all σ=1 | Noise all σ=2 |
|---|---|---|---|---|---|---|---|---|
| MUAD | 0.904 | 0.857 (5.2%) | 0.809 (10.5%) | 0.716 (20.8%) | 0.662 (26.8%) | 0.897 | 0.841 | 0.699 |
| v2 plain | 0.983 | 0.905 (7.9%) | 0.824 (16.2%) | 0.722 (26.6%) | 0.669 (32.0%) | 0.939 (4.6%) | 0.942 | **0.835** |
| + graph | 0.960 | 0.907 (5.5%) | 0.826 (14.0%) | 0.709 (26.2%) | 0.647 (32.6%) | 0.891 | 0.839 | 0.635 |
| + prob | 0.936 | 0.915 (2.3%) | 0.875 (6.6%) | 0.767 (18.1%) | 0.677 (27.7%) | 0.922 | 0.920 | 0.801 |
| + ug_routing | 0.934 | 0.860 (8.0%) | 0.815 (12.8%) | 0.734 (21.4%) | 0.680 (27.2%) | 0.874 | 0.916 | 0.812 |
| **full** | **0.984** | **0.962 (2.3%)** | **0.931 (5.5%)** | **0.856 (13.0%)** | **0.748 (24.0%)** | **0.981 (0.3%)** | **0.960** | 0.790 |

  **This is the evidence for the "UG" components.** On clean data full ≈ plain (0.984 vs 0.983). But
  with 20% / 40% of observations missing, **full beats plain by +10.7 / +13.4 F1 points**, and plain is no
  better than MUAD there. The **probabilistic representation** is the step that first makes the model
  robust (relative drop at 20% masking: 16.2% → 6.6%). **Adaptive fusion** then restores clean accuracy and
  gives the best absolute robustness. Full also barely notices missing logs (−0.3% vs −4.6% for plain).
  Exception: under extreme noise on all modalities (σ = 2 × std), plain is better (0.835 vs 0.790), so
  claim robustness to *missing* data and moderate noise, not to extreme noise. Graph-aware routing on
  its own makes the model *more* fragile to noise (σ = 2: 0.635). Single seed; confirm with 5 seeds in
  the final runs.

#### Round 9: v2 component ablations (plan §8B) on SN (seed 42, 2026-11-17)

`runs/round9_abl2_sn/baselines_20261117_181606`. Each row removes exactly one component from the full model
(full: test macro-F1 0.9868, F1 0.9842).

| Ablation | Test F1 | Recall | Macro-F1 | Δ macro-F1 vs full |
|---|---|---|---|---|
| w/o graph | 0.9829 | 0.9723 | 0.9857 | −0.1 (noise) |
| **w/o MoE** (single temporal expert) | 0.8845 | 0.7957 | 0.9089 | **−7.8** |
| w/o dynamic routing (uniform weights) | 0.9752 | 0.9550 | 0.9794 | −0.7 (noise) |
| **w/o uncertainty routing** | 0.9362 | 0.8892 | 0.9480 | **−3.9** |
| w/o probabilistic representation | 0.9794 | 0.9654 | 0.9828 | −0.4 (noise) |
| **w/o adaptive fusion** | 0.9340 | 0.8944 | 0.9459 | **−4.1** |

- Clean-accuracy contributions on SN: **MoE (−7.8), adaptive fusion (−4.1) and uncertainty-guided routing
  (−3.9)**. Graph, dynamic routing and the probabilistic representation are within single-seed noise on
  *clean* SN (the probabilistic representation matters for robustness, round 8).
- **Reproducibility check:** `w/o adaptive fusion` is the same configuration as the `ug_routing`
  incremental step, and it reproduced it exactly (identical epoch log, test 0.9459). Training is
  deterministic for a given seed.

#### Round 10: MUAD with the paper's hyperparameters on SN (seed 42, 2026-11-17)

`runs/round10_muad_paperhp_sn/baselines_20261117_202530`: lr 1e-3, up to 120 epochs (early-stopped at 76,
selected epoch 61). **Test macro-F1 0.7928** (F1 0.7200, recall 0.573), vs **0.9219** with our shared
settings (lr 5e-4, ≤ 50 epochs). Validation macro-F1 was 0.973 at the selected epoch, so the paper's
settings overfit the validation run and generalize worse. **Our shared settings favour MUAD, so the
hyperparameter difference cannot explain our margin over it.** Report MUAD with our settings (its
better result).

#### Round 12: modality ablation (plan §8C), v2 full trained on each subset, SN (2026-11-17)

`runs/round12_modality_sn/baselines_20261117_215903`. Test macro-F1 (trivial 0.289):

| Modalities | Macro-F1 | F1 | PR-AUC |
|---|---|---|---|
| metrics | 0.883 | 0.850 | 0.990 |
| traces | 0.935 | 0.923 | 0.975 |
| logs | 0.832 | 0.779 | 0.835 |
| traces + logs | 0.938 | 0.924 | 0.970 |
| metrics + logs | 0.959 | 0.950 | 0.994 |
| metrics + traces | 0.929 | 0.912 | 0.993 |
| **all three** | **0.987** | **0.984** | **0.998** |

- **Every modality carries real signal on its own** (all ≥ 0.83 vs trivial 0.29). Trained on traces only,
  v2 reaches 0.935, above metrics only (0.883). (Partly a traffic signal: in SN fault runs only 58–74% of
  seconds have spans vs 93–95% in fault-free runs.)
- **Fusion pays:** all three beat the best pair by +2.8 and the best single modality by +5.2.
- **Consequence for robustness:** the full model collapses when metrics are removed at test time
  (round 8), yet a model trained without metrics reaches 0.938. The information is there; the full model
  never learns to fall back on it because it never sees a window without metrics in training. This is
  the case for modality dropout (proposal pending).

#### Round 15b: v2 component ablations on TT (2026-11-17)

`runs/round15b_abl2_tt/baselines_20261117_215852` (full: 0.8232 macro-F1).

| Ablation | Macro-F1 | Δ vs full | (SN Δ, round 9) |
|---|---|---|---|
| w/o graph | 0.636 | **−18.8** | −0.1 |
| w/o MoE | 0.825 | +0.2 | **−7.8** |
| w/o dynamic routing | 0.667 | **−15.6** | −0.7 |
| w/o uncertainty routing | 0.821 | −0.3 | **−3.9** |
| w/o probabilistic | 0.760 | **−6.3** | −0.4 (but −13 to −15 under masking) |
| w/o adaptive fusion | 0.840 | +1.7 | **−4.1** |

- `w/o adaptive fusion` reproduced the `ug_routing` step exactly again (0.8399): training is deterministic.
- **The pattern differs between datasets.** TT needs graph and dynamic routing (−19 / −16), SN needs MoE,
  uncertainty routing and adaptive fusion. **The only component that matters on both is the
  probabilistic representation** (TT clean −6.3; SN robustness −13 to −15).
- **Caution on TT:** a single temporal expert (w/o MoE) matches the full model, while a 4-expert mixture
  without graph context or with fixed weights collapses to 0.64–0.67 (like `plain` / `graph`, round 7).
  With TT's strongly fluctuating validation curves, single-seed TT ablations are not reliable enough to
  rank components. They must be settled by the 5-seed final runs.

#### Round 16a: sensitivity, number of experts and embedding size (plan §8H, SN, 2026-11-17)

`runs/round16a_sens_sn/baselines_20261117_215920` (full: 4 experts, hidden 64, 166k params, 0.9868).

| Variant | Macro-F1 | Δ |
|---|---|---|
| 1 expert (`abl2:no_moe`, round 9) | 0.909 | −7.8 |
| 2 experts (temporal + dependency) | 0.988 | +0.2 |
| 3 experts (+ cross-modal) | 0.984 | −0.3 |
| 4 experts (full) | 0.987 | — |
| hidden 16 | 0.980 | −0.7 |
| hidden 32 (43k params) | 0.978 | −0.9 |
| hidden 128 (652k params) | 0.973 | −1.4 |

- At least two experts are needed (1 → 2: +7.9); beyond that, SN is flat within noise.
- Embedding size: 64 is best; every other size is within 1.4 points. Size sweep (step 13b): 32 loses
  0.9 test points, more than the 0.5-point threshold, so **keep hidden 64**. The 4× larger model is
  slightly worse, so capacity is not what drives the results.

#### FINAL: 5-seed main results and ablations (seeds 42, 123, 456, 789, 999; 2026-11-18)

`runs/final_sn_main_{a,b}`, `runs/final_tt_main_{a,b}` (+ TT seed 42 from round 7), `runs/final_{sn,tt}_abl_{a,b}`
(+ seed 42 from rounds 9 / 15b). The means below were pooled by hand from the per-run means; the exact
mean ± std comes from `make_report.py`. **These numbers supersede the single-seed tables above.**

Test macro-F1, 5-seed mean:

| Model | SN | TT |
|---|---|---|
| MUAD | 0.925 | 0.626 |
| GAT | 0.931 | 0.661 |
| v2 plain (= plain MoE) | 0.971 | 0.698 |
| v2 graph (= Graph-MoE) | 0.960 | 0.667 |
| **UG-MoE v2 full** | **0.972** | **0.723** |

- **UG-MoE v2 beats MUAD on both datasets: +4.7 (SN) and +9.7 (TT) macro-F1**, and GAT by +4.1 / +6.3.
- **Full vs plain: +0.1 (SN), +2.5 (TT).** Not established as a clean-accuracy gain. TT seeds vary a
  lot (full ranges 0.65–0.82 across seeds; plain reached 0.84 on seed 123).
- **The single-seed TT margin (+17.6 over MUAD) was optimistic; the 5-seed margin is +9.7.**

Component ablations, 5-seed mean, Δ vs full:

| Ablation | SN | Δ SN | TT | Δ TT |
|---|---|---|---|---|
| w/o graph | 0.978 | +0.6 | 0.741 | +1.8 |
| w/o MoE | 0.965 | −0.6 | 0.724 | +0.1 |
| w/o dynamic routing | 0.977 | +0.5 | 0.673 | **−5.0** |
| w/o uncertainty routing | 0.974 | +0.2 | 0.698 | −2.6 |
| w/o probabilistic | 0.980 | +0.8 | 0.704 | −1.9 |
| **w/o adaptive fusion** | 0.948 | **−2.4** | 0.690 | **−3.3** |

- **Adaptive (uncertainty-aware) fusion is the one component that helps clean accuracy on both
  datasets** (−2.4 / −3.3 when removed). On SN it is the only consistent effect: every seed shows the
  drop (0.946–0.950 vs ~0.97).
- On TT, dynamic routing (−5.0) and uncertainty routing (−2.6) point the right way, but TT's per-seed
  spread is large, so check the stds from `make_report.py` before claiming them.
- **Several single-seed effects disappeared:** w/o MoE on SN (−7.8 → −0.6), w/o graph on TT
  (−18.8 → +1.8), w/o uncertainty routing on SN (−3.9 → +0.2). The single-seed ablations were dominated
  by seed noise, which is exactly why the final runs use 5 seeds.
- The clean-accuracy role of the probabilistic representation is not significant. Its robustness role
  (round 15a) is being re-checked over 5 seeds (final robustness runs).

#### Rounds 16b and 16c: sensitivity, KL weight, routing temperature, learning rate, window (SN, 2026-11-18)

`runs/round16b_sens_sn/`, `runs/round16c_sens_sn_{lr1e-4,lr1e-3,w5,w20}/`. Test macro-F1 (default marked *):

| Parameter | Values → macro-F1 |
|---|---|
| KL weight λ | 0.001 → 0.979 · **0.01* → 0.987** · 0.1 → 0.953 · 1.0 → 0.972 |
| Routing temperature | 0.5 → 0.969 · **1.0* → 0.987** · 2.0 → 0.983 |
| Learning rate | 1e-4 → 0.982 · **5e-4* → 0.987** · 1e-3 → 0.985 |
| Window (s) | 5 → 0.956 · **10* → 0.987** · 20 → 0.973 |

Every setting stays within 3.4 points of the default (the worst is λ = 0.1). The defaults were fixed
before these runs, not tuned on test, so no test-set tuning bias. The window runs use rebuilt datasets
(`SN_w5_s1`, `SN_w20_s1`) with slightly different class balance (trivial macro-F1 0.281 / 0.305), so
compare them with that in mind. Single seed; the KL-weight curve is non-monotonic (0.1 below 1.0),
which is probably seed noise.

#### Round 15a: robustness of `abl2:no_probabilistic` on SN (2026-11-17)

`runs/round15a_rob_sn_noprob/`. Full model minus the probabilistic representation (and therefore minus
uncertainty routing), everything else equal.

| Model | Clean F1 | Mask 10% | Mask 20% | Mask 40% | Mask 60% | Noise all σ=1 |
|---|---|---|---|---|---|---|
| full | 0.984 | 0.962 | **0.931** | **0.856** | **0.748** | 0.960 |
| w/o probabilistic | 0.979 | 0.895 | 0.803 | 0.707 | 0.646 | 0.902 |

**Direct evidence:** the same architecture without the probabilistic representation has the same clean
accuracy (−0.5) but loses **12.8 / 14.9 F1 points** at 20 / 40% masking. The probabilistic representation
is what makes the model robust to missing data.

#### Rounds 11 and 14: TT robustness (MUAD, GAT, v2 full) and TT routing / uncertainty analysis (2026-11-17)

`runs/round11_rob_tt_{muad,gat}/`, `runs/round14_rob_tt_v2/`, `runs/round14_analysis_tt/`.

**Per-fault-type recall on the clean TT test set (new metric):**

| Model | cpu_load | network_delay | network_loss | FPR on normal windows | Macro-F1 |
|---|---|---|---|---|---|
| MUAD | 0.949 | 0.389 | 0.302 | 0.131 | 0.647 |
| GAT | 0.963 | 0.645 | 0.486 | 0.167 | 0.730 |
| **v2 full** | **1.000** | **0.895** | **0.877** | 0.300 | **0.823** |

The anomalies the baselines miss on TT are the **network faults**: CPU load is easy for every model.
UG-MoE v2 detects 88–90% of network faults vs 30–65%, at the cost of more false alarms (30% of normal
windows vs 13–17%).

**Robustness on TT: read macro-F1, not anomaly-F1.** MUAD and GAT are conservative (low recall), so
perturbations push them towards "anomalous" and their anomaly-F1 *rises* towards the trivial 0.809,
while macro-F1 falls. Macro-F1 under masking:

| Model | Clean | Mask 10% | Mask 20% | Mask 40% | Mask 60% | Drop metrics |
|---|---|---|---|---|---|---|
| MUAD | 0.647 | 0.636 | 0.616 | 0.577 | 0.559 | 0.243 |
| GAT | 0.730 | 0.652 | 0.615 | 0.540 | 0.471 | 0.404 (trivial) |
| v2 full | **0.823** | **0.734** | **0.672** | **0.581** | 0.547 | 0.404 (trivial) |

v2 stays best up to 40% masking, but its margin shrinks from +9 / +18 points (vs GAT / MUAD) to about
level at 40–60%. **The robustness advantage seen on SN is not reproduced on TT.**

**Analysis on TT (Exp. F / G):**
- Fusion is more balanced on TT than SN (clean: metric 0.61, trace 0.28, log 0.11).
- **When metrics are dropped the mechanism works strongly:** metric log-variance −0.28 → +0.80 and its
  fusion weight 0.61 → 0.13, with traces taking 0.79. The model still predicts trivially, because TT
  traces cover only 2–4% of seconds: the modality it falls back on carries almost no information.
- **Failure case: noise on logs.** Log uncertainty *decreases* (−0.03 → −0.14) and the log fusion
  weight *rises* to 0.65 (F1 0.894 → 0.817). The learned variance is overconfident on out-of-distribution
  noisy inputs, a known limitation of learned (aleatoric) variance heads. Same under noise on all
  modalities (log weight 0.43).
- On TT the uncertainty does **not** flag errors (log-variance −0.22 correct vs −0.25 wrong), unlike SN.
- Routing: CPU-load windows again get a dedicated metric expert (semantic, 0.996). Traces route to the
  cross-modal expert (0.87), logs to the semantic expert (0.79).

#### Round 13: routing and uncertainty analysis (Exp. F + G), UG-MoE v2 full, SN test (2026-11-17)

`runs/round13_analysis_sn/analysis_ug_moe2_full_20261117_191831` (`run_analysis.py`, round 6 checkpoint).

**G, fusion and uncertainty.** Mean fusion weight / mean predicted log-variance per modality:

| Condition | F1 | w metric | w trace | w log | logvar metric | logvar trace | logvar log |
|---|---|---|---|---|---|---|---|
| clean | 0.984 | 0.959 | 0.029 | 0.013 | −0.22 | −0.03 | +0.03 |
| drop metrics | 0.579 | 0.880 | 0.093 | 0.027 | **−0.02** | −0.09 | +0.04 |
| drop traces | 0.979 | 0.972 | **0.015** | 0.013 | −0.23 | **+0.08** | +0.03 |
| drop logs | 0.981 | 0.964 | 0.021 | 0.015 | −0.22 | +0.02 | **+0.15** |
| mask 20% | 0.934 | 0.932 | 0.044 | 0.024 | −0.04 | −0.02 | +0.00 |
| mask 40% | 0.846 | **0.889** | 0.068 | 0.043 | **+0.09** | +0.00 | −0.01 |
| noise metrics σ=1 | 0.971 | **0.910** | 0.062 | 0.028 | **−0.07** | −0.03 | +0.03 |
| noise all σ=1 | 0.960 | 0.874 | 0.094 | 0.032 | −0.07 | +0.03 | +0.06 |

- **The mechanism works in the intended direction.** A perturbed modality's predicted uncertainty
  rises (metrics −0.22 → −0.07 under noise, → +0.09 at 40% masking; traces −0.03 → +0.08 when dropped;
  logs +0.03 → +0.15 when dropped), and its fusion weight falls while the others gain (metric weight
  0.96 → 0.91 under metric noise, → 0.89 at 40% masking).
- **But the shift is small.** On clean data metrics get 96% of the fusion weight. Even with metrics
  dropped they keep 88%, which is why the model collapses without metrics: the uncertainty signal is
  right but not strong enough to hand the decision to traces and logs. (Modality dropout in training
  would address this; it is a deviation and needs a proposal.)
- **Uncertainty flags errors:** fusion-weighted log-variance is −0.227 on correct predictions vs −0.042
  on the 36 wrong ones.

**F, expert routing (clean test).** Routing entropy (0 = a single expert, 1 = uniform): metric 0.41,
trace 0.65, log 0.97.
- **Metric routing specializes by fault type:** normal → dependency (GAT) expert 0.87; cpu_load →
  cross-modal expert 0.95; network delay / loss → mainly temporal (0.52 / 0.49, entropy 0.83).
- Trace routing favours the cross-modal expert (0.67), consistent with sparse traces drawing on the
  other modalities. Log routing is near-uniform, matching the logs' tiny fusion weight (they are barely
  used). The semantic expert is rarely chosen (0.03–0.19).

#### Round 4: leak-free run-level split on `recovered_artifact` (seed 42, 2026-11-17)

`runs/round4_runsplit/` (normalized inputs, `--split run`: train runs 0–1 = 2,804 / val run 2 = 1,404 / test run 3 = 1,402 windows; test 83% anomalous). Δ = change in test macro-F1 vs round 2 (same models, leaky random split).

| Model | Best epoch (val) | Test F1 | F1 normal | Macro-F1 | PR-AUC | ROC-AUC | Δ macro-F1 vs leaky |
|---|---|---|---|---|---|---|---|
| Trivial "all anomalous" | — | 0.9077 | 0.0000 | 0.4538 | 0.8310 | 0.5000 | — |
| **UG-MoE (Track B)** | 19 | **0.9767** | **0.8898** | **0.9332** | **0.9984** | 0.9924 | −2.6 |
| **MUAD (Track A)** | 32 | **0.9767** | 0.8880 | 0.9324 | 0.9934 | 0.9747 | −3.3 |
| Plain MoE | 11 | 0.9688 | 0.8560 | 0.9124 | 0.9957 | 0.9811 | −6.0 |
| Graph-MoE | 2 | 0.9112 | 0.0810 | 0.4961 (≈ trivial) | 0.9819 | 0.9389 | **−47.4** |
| GAT | 4 | 0.8738 | 0.2323 | 0.5530 | 0.9584 | 0.8170 | **−40.8** |
| MUAD, traces only | 40 | 0.9034 | 0.2500 | 0.5767 | 0.8555 | 0.6134 | — |
| MUAD, logs only | 26 | 0.9077 | 0.0000 | 0.4538 (= trivial) | 0.9238 | 0.6793 | — |
| MUAD, traces + logs | 37 | 0.8978 | 0.5223 | 0.7101 | 0.9613 | 0.8428 | — |

**Modality ablation (Exp. C, MUAD, leak-free):** metrics carry almost all the signal (all three modalities 0.932 macro-F1). **Traces alone** reach val macro-F1 0.767 but only 0.577 on the test run (weak cross-run generalization). **Logs alone** never beat the trivial predictor at the default threshold, though they still rank somewhat (ROC-AUC 0.68). **Traces + logs together reach 0.710 macro-F1 and ROC-AUC 0.84, far more than either alone**, so the two are complementary. So on this artifact traces and logs are *useful but secondary*: they matter most as a fallback when metrics are missing or noisy (Exp. D/E), which is where uncertainty-aware fusion should show its value. Caveat: these are MUAD's Hawkes log features; the rebuilt data uses template counts and may behave differently.

**Readings:**
- **The leakage was inflating everything, and it hid real differences.** GAT and Graph-MoE looked competitive on the leaky split (macro-F1 0.96–0.97) but **collapse to near-trivial on an unseen run**. They were memorizing near-duplicate windows (their training loss keeps falling while validation loss rises from epoch 2–4). MUAD and UG-MoE lose only ~3 points, so they generalize.
- **The ranking flips relative to the leaky split:** UG-MoE is now level with MUAD (macro-F1 0.9332 vs 0.9324, +0.08, within noise) and **above plain MoE by +2.1 macro-F1** (above the ~1-point single-seed threshold). On the leaky split plain MoE was best. A plausible reading is that the uncertainty blocks plus KL act as a regularizer that helps cross-run generalization. That is a hypothesis to test in the ablations, not a claim.
- **Validation metrics are very noisy epoch to epoch** (e.g. GAT's val F1 swings 0.62–0.92), so single-epoch selection is fragile. Consider selecting on val PR-AUC or val loss, or smoothing, once multi-seed runs are possible.
- **Traces carry signal on their own:** MUAD trained on traces only reaches val macro-F1 ≈ 0.77 (trivial 0.45) while still improving. The models were taking a metrics shortcut, but traces aren't empty.
- Caveat: one test run (3 faulted services), one seed. The rebuilt SN/TT data (balanced, more runs) is the stronger benchmark.

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
12. ~~Remove the hard-coded 12/24 assumptions so TT (27 services) can run.~~ Done 2026-11-17 (`num_edges` from metadata; experts infer N from the graph).
13. Re-run Plan §12 steps 4–10 in order as quick single-seed checks on the server: simple MLP, then GAT, then plain MoE, then +graph routing, then +uncertainty, then +UG routing, then +fusion. Confirm each addition changes behaviour.

13b. **Size sweep (§3.10):** shrink experts and dims to the smallest config with negligible validation loss. Single seed; keep the full size if the loss isn't negligible.

### Phase 3: Freeze, then run the experiments (plan §8)
14. Freeze the model (tag a commit). Rewrite ablations to exactly match §8B, each flag removing exactly one component.
15. Baseline comparison (A) under the shared protocol, 5 seeds.
16. Modality ablation (C), trained on each of the 7 subsets.
17. Missing modality (D): 10/20/40/60% random masking plus single-modality drop. Noise (E): single and combined modalities, with relative F1 degradation.
18. Routing analysis (F) per anomaly type (`culprit` 1/4/5) as heatmaps, with entropy. Uncertainty/fusion analysis (G): clean vs. noisy.
19. Sensitivity (H): number of experts, embedding dim, window size, λ_KL, routing temperature, lr.
20. Cross-dataset: TT from the same Zenodo record. The pipeline exists (`data_pipeline/`); note that TT traces are nearly absent (2–4% of seconds).

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
