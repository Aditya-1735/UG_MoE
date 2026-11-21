# UG-MoE for Multimodal Microservice Anomaly Detection

## 1. Project overview

I built **UG-MoE v2**, an uncertainty-guided, graph-aware mixture of experts that classifies each 10-second window of metrics, traces, and logs as **normal or anomalous**. I also audited the MUAD baseline and rebuilt the evaluation from raw recordings. Under the shared protocol, UG-MoE v2 improved test macro-F1 over MUAD by **4.6 points on SocialNetwork** and **9.7 points on TrainTicket**.

## 2. MUAD audit and evaluation changes

| Finding | Project response |
|---|---|
| In the recovered SocialNetwork split, **83.87% of test windows** (1,882 of 2,244) had a training neighbor sharing nine of ten seconds of metrics. | Keep fault recordings in separate sets. For TrainTicket's fault-free recordings, use nonoverlapping time blocks and discard windows that cross the cuts. |
| The released MUAD trainer chooses its checkpoint using test F1. | Use validation macro-F1 for checkpoint selection in **both models**. Use the test set for final scoring. |
| On the recovered test set, an always-anomalous predictor gets **0.9064 anomaly F1** and **0.4532 macro-F1**. | Lead with macro-F1, which averages normal-class and anomaly-class F1. Report the constant baseline. |
| MUAD's released model mixes up variance and log-variance in its uncertainty penalty and samples at prediction time. | UG-MoE v2 uses the correct Gaussian KL formula and uses the latent mean in evaluation. The MUAD baseline retains its published implementation's sampling and KL behavior. |

In the local audit, MUAD scored **0.9882 anomaly F1** on the recovered random split and **0.9767** on a whole-recording split of the recovered data. The second run also changed normalization and which windows were tested, so the score drop does not isolate the impact of overlap. The 83.87% measurement applies to the recovered files. The paper does not specify how it selected the checkpoint for its published score [1, 3].

## 3. Implementation and experimental setup

I aligned the Eadro recordings to a one-second grid and labeled a window anomalous when it overlapped an injected fault by at least one second. The inputs are seven container metrics, span latency, and Drain3 log-template counts per service. Each window uses the benchmark's service dependency graph. Normalization was fitted on training data only [2, 4].

| Benchmark | Fault / fault-free recordings | Services / edges | Window / stride | Train / validation / test windows |
|---|---:|---:|---:|---:|
| SocialNetwork (SN) | 4 / 3 | 12 / 24 | 10 s / 1 s | 4,236 / 2,834 / 2,837 |
| TrainTicket (TT) | 9 / 2 | 27 / 44 | 10 s / 5 s | 8,397 / 3,228 / 3,226 |

UG-MoE v2 encodes each modality and applies four shared experts. A GRU models **changes over time**, self-attention models **relationships across services**, GATv2 uses **service dependencies**, and cross-modal attention combines **signals from the same service**. A router weights the experts using features, graph context, and estimated uncertainty. A second set of learned weights combines the three modalities for binary classification [4].

## 4. Final results

The main comparison uses the same data and evaluation rules across models. Scores are **mean ± standard deviation across five training seeds** on fixed test sets. The constant predictor has no training seed [5].

| Model | SN macro-F1 | SN anomaly F1 | TT macro-F1 | TT anomaly F1 |
|---|---:|---:|---:|---:|
| Always predict anomaly | 0.2893 | 0.5787 | 0.4044 | 0.8087 |
| MUAD | 0.9251 ± 0.0049 | 0.9075 ± 0.0065 | 0.6262 ± 0.0316 | 0.6456 ± 0.0491 |
| GAT | 0.9309 ± 0.0347 | 0.9136 ± 0.0451 | 0.6609 ± 0.0440 | 0.6917 ± 0.0606 |
| Plain MoE | 0.9709 ± 0.0170 | 0.9646 ± 0.0213 | 0.6981 ± 0.0727 | 0.7353 ± 0.0920 |
| Graph MoE | 0.9598 ± 0.0177 | 0.9509 ± 0.0222 | 0.6666 ± 0.0628 | 0.6883 ± 0.0875 |
| **UG-MoE v2** | **0.9715 ± 0.0128** | **0.9656 ± 0.0158** | **0.7234 ± 0.0899** | **0.7656 ± 0.1169** |

Compared with MUAD, the macro-F1 gains had paired p-values of **0.0021 (SN)** and **0.0420 (TT)**. UG-MoE used about **31% fewer parameters**. On SN, CPU-load recall rose from **65.8% to 98.4%**. On TT, recall rose from **33.1% to 59.4%** for network delay and from **27.3% to 50.8%** for network loss. TT false alarms rose from **10.9% to 18.3%**, and PR-AUC was close to MUAD's (**0.9167 vs. 0.9143**) [5].

**Additional test metrics for MUAD and UG-MoE v2 (five-seed means)** [5].

| Dataset and model | Precision | Recall | PR-AUC | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| SN, MUAD | 0.9699 | 0.8530 | 0.9679 | 0.9649 |
| SN, UG-MoE v2 | 0.9896 | 0.9430 | 0.9963 | 0.9977 |
| TT, MUAD | 0.9082 | 0.5041 | 0.9143 | 0.8372 |
| TT, UG-MoE v2 | 0.9030 | 0.6997 | 0.9167 | 0.8283 |

### Component ablations

Five-seed means. Macro-F1 Δ is the change in points from full UG-MoE v2. Anomaly F1 is the variant's test score [5].

| Component removed | SN macro-F1 Δ | SN anomaly F1 | TT macro-F1 Δ | TT anomaly F1 |
| --- | ---: | ---: | ---: | ---: |
| Full model (reference) | — | 0.9656 | — | 0.7656 |
| Graph context and dependency expert | +0.6 | 0.9729 | +1.8 | 0.7901 |
| Mixture of experts | −0.6 | 0.9573 | +0.1 | 0.7790 |
| Dynamic routing | +0.5 | 0.9719 | −5.0 | 0.7047 |
| Uncertainty input to router | +0.2 | 0.9685 | −2.6 | 0.7312 |
| Probabilistic representation | +0.8 | 0.9757 | −1.9 | 0.7483 |
| Adaptive fusion | −2.4 | 0.9359 | −3.3 | 0.7221 |

Adaptive fusion helped on average on both datasets. Every component test had p > 0.05, and Plain MoE scored close to the full model [5].

### Modality subsets

Each model was trained and tested with the listed inputs on SocialNetwork (seed 42). All three inputs performed best on both scores [5].

| Training inputs | SN macro-F1 | SN anomaly F1 |
| --- | ---: | ---: |
| Logs | 0.8318 | 0.7788 |
| Traces | 0.9353 | 0.9232 |
| Metrics | 0.8829 | 0.8500 |
| Traces + logs | 0.9379 | 0.9237 |
| Metrics + logs | 0.9587 | 0.9499 |
| Metrics + traces | 0.9292 | 0.9120 |
| All three | 0.9868 | 0.9842 |

### Missing and noisy inputs

Five-seed mean test scores. Each cell is **macro-F1 (anomaly F1)**. Masking zeros each service–modality observation with the stated probability. Strong Gaussian noise uses twice the training standard deviation per channel. The saved sweep covers all seven nonempty modality combinations at five noise levels [5].

| Test condition | SN UG-MoE | SN MUAD | TT UG-MoE | TT MUAD |
| --- | ---: | ---: | ---: | ---: |
| 10% random masking | 0.942 (0.931) | 0.898 (0.878) | 0.616 (0.812) | 0.622 (0.676) |
| 20% random masking | 0.903 (0.889) | 0.850 (0.831) | 0.573 (0.804) | 0.613 (0.706) |
| 40% random masking | 0.811 (0.797) | 0.710 (0.721) | 0.506 (0.785) | 0.568 (0.694) |
| 60% random masking | 0.696 (0.708) | 0.543 (0.645) | 0.471 (0.777) | 0.526 (0.616) |
| Metrics removed | 0.274 (0.545) | 0.289 (0.579) | 0.406 (0.706) | 0.243 (0.000) |
| Traces removed | 0.973 (0.968) | 0.915 (0.895) | 0.711 (0.759) | 0.628 (0.647) |
| Logs removed | 0.967 (0.960) | 0.922 (0.904) | 0.676 (0.720) | 0.654 (0.690) |
| Strong noise on metrics | 0.856 (0.834) | 0.784 (0.734) | 0.575 (0.744) | 0.548 (0.727) |
| Strong noise on traces | 0.941 (0.937) | 0.910 (0.890) | 0.651 (0.801) | 0.626 (0.645) |
| Strong noise on logs | 0.971 (0.965) | 0.923 (0.905) | 0.584 (0.827) | 0.659 (0.697) |
| Strong noise on all three | 0.808 (0.802) | 0.735 (0.687) | 0.470 (0.783) | 0.537 (0.749) |

At 40% masking, the **mean relative anomaly-F1 drop** from clean SN was **17.6% for UG-MoE** and **20.5% for MUAD**. On TT, anomaly F1 rose by **4.9%** and **7.5%**, respectively, while **macro-F1 fell by 30.1% and 9.2%**. These are averages of the per-seed relative changes. The TT example shows why both classes matter under corruption [5].

Removing metrics brought UG-MoE close to the constant baseline on both datasets. Under strong all-stream noise on SN, the variant without probabilistic modeling scored **0.810 (0.796)**, close to the full model's **0.808 (0.802)**, in macro-F1 (anomaly F1). The source of the robustness gain remains unresolved [5].

### Expert routing and uncertainty

Seed-42 diagnostics describe what the model learned in one run [4, 5].

| Diagnostic | Observation |
|---|---|
| **Routing by fault type** | On SN, metrics favored the dependency expert for normal windows (0.87 mean weight), the cross-modal expert for CPU load (0.95), and the temporal expert for network faults (about 0.50). On TT, CPU-load metrics favored the semantic expert (0.996). |
| **Normalized routing entropy** | SN metrics were more concentrated than SN logs (0.41 vs. 0.97). A value of 0 means one expert receives all the weight, while 1 means uniform weights. |
| **Uncertainty and fusion** | With metric noise on SN, metric log-variance rose from −0.22 to −0.07 and fusion weight fell from 0.96 to 0.91. On TT, log noise reduced log-variance from −0.03 to −0.14 and increased the log weight from 0.11 to 0.65. Anomaly F1 fell from 0.894 to 0.817. |

### Hyperparameter sensitivity

One parameter was varied at a time on SocialNetwork (seed 42). **Bold** marks the default [5].

| Hyperparameter | Setting → macro-F1 (anomaly F1) |
| --- | --- |
| Number of experts | 1 → 0.9089 (0.8845) · 2 → 0.9883 (0.9861) · 3 → 0.9835 (0.9804) · **4 → 0.9868 (0.9842)** |
| Embedding dimension | 16 → 0.9798 (0.9758) · 32 → 0.9783 (0.9739) · **64 → 0.9868 (0.9842)** · 128 → 0.9728 (0.9672) |
| Window size (seconds) | 5 → 0.9563 (0.9464) · **10 → 0.9868 (0.9842)** · 20 → 0.9732 (0.9692) |
| Uncertainty-loss weight (KL) | 0.001 → 0.9787 (0.9746) · **0.01 → 0.9868 (0.9842)** · 0.1 → 0.9525 (0.9418) · 1 → 0.9724 (0.9668) |
| Routing temperature | 0.5 → 0.9690 (0.9626) · **1 → 0.9868 (0.9842)** · 2 → 0.9828 (0.9793) |
| Learning rate | 0.0001 → 0.9817 (0.9782) · **0.0005 → 0.9868 (0.9842)** · 0.001 → 0.9850 (0.9822) |

The largest observed macro-F1 drops were with one expert (**−7.8 points**) and KL weight 0.1 (**−3.4 points**). Two experts slightly exceeded four at this seed. The default settings were set before the sweep, and these results are exploratory [4, 5].

## 5. Conclusion and limitations

The evidence supports a higher clean macro-F1 than MUAD on both rebuilt benchmarks and better robustness on SocialNetwork. Individual component gains remain inconclusive. TrainTicket results vary across seeds and trade higher fault recall for more false alarms. The number of held-out recordings is small, the TrainTicket normal sets share source recordings without gaps at the cuts, and the label-free log vocabulary uses all runs. The performance of a separately tuned MUAD with a corrected KL term remains open [3–5].

## Saved outputs

The repository contains processed data and graph metadata, baseline and UG-MoE checkpoints, and the guide's performance tables, ablations, robustness plots, routing and fusion plots, and sensitivity plot. The script `make_report.py` regenerates the tables and plots from saved results [4–6].

## Sources

1. H. Dou et al. *Uncertainty-Aware Multimodal Anomaly Detection for Microservice Systems With Active Learning*. IEEE Transactions on Services Computing, 2026. DOI 10.1109/TSC.2026.3672587. MUAD reference implementation.
2. C. Lee et al. *Eadro: An End-to-End Troubleshooting Framework for Microservices on Multi-Source Data*. ICSE, 2023. Benchmark dataset, Zenodo record 7615394.
3. *MUAD-gaps.md*. Baseline audit and reproducible checks.
4. *UG-MoE_Implementation_Report.md*. Data, model, evaluation protocol, and code references.
5. *UG-MoE consolidated experiment results*. 18 November 2026, 20:16.
6. *UG-MoE Implementation Guide*, sections 8–10. Experiment plan, evaluation metrics, and required outputs.
