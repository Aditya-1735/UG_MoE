# MUAD vs. UG-MoE: what is wrong with MUAD's numbers, and what our model actually shows

> **Final 5-seed results (2026-11-18) supersede the single-seed numbers in §4 below.** Test macro-F1,
> mean over seeds 42 / 123 / 456 / 789 / 999:
>
> | Model | SN | TT |
> |---|---|---|
> | MUAD | 0.925 | 0.626 |
> | GAT | 0.931 | 0.661 |
> | Plain MoE (v2 plain) | 0.971 | 0.698 |
> | Graph-MoE (v2 graph) | 0.960 | 0.667 |
> | **UG-MoE v2** | **0.972** | **0.723** |
>
> - UG-MoE v2 beats MUAD by **+4.7 (SN) / +9.7 (TT)** and GAT by +4.1 / +6.3.
> - Against its own plain variant it gains +0.1 (SN) and +2.5 (TT). That is not an established
>   clean-accuracy gain, because TT varies a lot between seeds.
> - The single-seed TT margin over MUAD (+17.6, §4.4b) was optimistic.
> - In the 5-seed ablations, **uncertainty-aware adaptive fusion is the only component whose removal
>   hurts on both datasets (−2.4 SN, −3.3 TT)**. Most single-seed ablation effects (§4.4a) turned out to
>   be seed noise.
> - The robustness results (§4.5–4.6) are being re-run over 5 seeds.

Status: 2026-11-17. All results are **single seed (42)**. Differences below about 1 macro-F1 point are
treated as noise. Every claim below points to the paper text, a line of code, or a run directory, so it
can be checked. Things not yet shown are listed in §6.

---

## 0. Summary

1. **MUAD's reported F1 ≈ 0.99 on SN comes from its evaluation protocol, not from the detection
   problem.** The paper splits stride-1 windows (neighbours share 9 of 10 seconds) **at random** into
   train and test. In the shipped data, **83.9% of test windows have a near-identical training window.**
   The test set is also **83% anomalous**, so predicting "anomalous" for everything already gets F1 0.906.
2. **MUAD's code has defects that matter for a fair evaluation:** it picks the best epoch **on the test
   set**, it **samples random noise at test time** (non-deterministic predictions), and its **KL term is
   mathematically wrong** (it does not regularize towards the prior it claims). Its shipped log features
   are ~6·10⁻¹⁰ in size, so **logs have no effect** on MUAD at all.
3. **We reproduce MUAD's number under its own protocol** (0.988 vs. the paper's 0.993). So the gap to
   the paper is caused by the protocol, not by a broken reproduction.
4. **On a leak-free protocol**, which we apply identically to every model:
   - On MUAD's own data: UG-MoE and MUAD **tie** (macro-F1 0.933 vs 0.932).
   - On SN rebuilt from the raw Eadro data (harder, includes fault-free runs): **UG-MoE v2 0.987
     vs. MUAD 0.922 macro-F1 (+6.5 points)**, with recall 0.973 vs 0.857. It uses 166k parameters
     (MUAD: 240k).
   - On TrainTicket (TT, much harder): **UG-MoE v2 0.823 vs. MUAD 0.647 macro-F1 (+17.6)**, and +9.4
     over the best baseline (GAT). Recall 0.92 vs 0.55: the baselines miss about half the anomalies (§4.4b).
5. **Where the components matter:** on SN (near ceiling) the full UG-MoE v2 is level with its own
   "plain" variant on clean data (0.987 vs 0.986). On TT the uncertainty components are decisive
   (plain 0.667 → full 0.823; uncertainty-guided routing is the key step). The uncertainty components show their value under
   **missing data**: UG-MoE v2 degrades about half as much as MUAD (40% masking: 0.856 vs 0.716 F1), and
   **the plain variant does not** (0.722, no better than MUAD). So the robustness comes from the
   probabilistic representation plus uncertainty-aware fusion (§4.6). Both models still fail completely
   without metrics. TT (round 7) is running.

---

## 1. Where MUAD's "F1 > 0.99" comes from: the paper's own protocol

### 1.1 What the paper says (MUAD paper, §IV-A "Experimental Setup")

> "We randomly selected 60% of the data from each of the three datasets for training and used the
> remaining 40% for testing, ensuring that the ratio of normal to abnormal samples remains consistent
> between the training and testing sets."

> "Sliding window length = **10**, stride = **1**."

> "Dataset C is the SN dataset from Eadro [11] … 7 metrics sampled every 1 s, 126,384 traces, and
> **15 log templates**."

The setup describes **no validation set** and **no rule for choosing the reported epoch** (only "120
epochs per training stage").

### 1.2 Our `recovered_artifact` is exactly that split

- `project/recovered_artifact/` (from the MUAD repository) has `event_num = 15` (= "15 log
  templates"), and **3,366 train / 2,244 test windows = exactly 60 / 40**.
- The windows were produced by Eadro's preprocessing (`project/vendor/eadro/codes/preprocess/align.py`):
  `intervals = [(s, s+9) for s in range(start, end-9)]` (stride 1), then `np.random.shuffle(chunk_idx)`
  (random split).

### 1.3 Why a random split of stride-1 windows leaks

Window *t* covers seconds *t … t+9* and window *t+1* covers *t+1 … t+10*: they share 9 of 10 seconds.
Randomly assigning windows to train and test therefore puts near-copies of most test windows into training.

**Measured** (`project/experiments/common.py`, `reconstruct_runs`). Consecutive windows can be chained
because window *t+1*'s first 9 metric steps equal window *t*'s last 9. This recovers the original
**4 fault-injection runs** (1,402 / 1,402 / 1,404 / 1,402 windows; every chunk covered, no ambiguity).
Then: **1,882 of 2,244 test windows (83.9%) have a training window as their immediate, 1-second-shifted
neighbour.** The model is mostly tested on data it has already seen.

### 1.4 The test set is imbalanced, so F1 alone says little

Test labels: 384 normal, 1,860 anomalous (**83% anomalous**). A constant "anomalous" predictor gets
precision 0.83, recall 1.0, **F1 = 0.906**, without looking at the data. The paper reports only the
anomaly-class F1, precision and recall (§IV-A-3).

**The paper's own ablation shows this** (Table III, Dataset C):

| Variant | F1 | Recall | Precision |
|---|---|---|---|
| Only Log | 0.9064 | **1.000** | **0.8289** |
| Only Metric | 0.9903 | 0.9925 | 0.9882 |
| MUAD (all three) | 0.9928 | 0.9941 | 0.9914 |

"Only Log" has recall exactly 1.0 and precision equal to the anomalous share of the test set: it
predicts "anomalous" for every window. "Only Metric" alone gives 0.9903. So of MUAD's 0.9928 on SN,
about 0.906 is free from the imbalance, almost everything else comes from the metrics, and logs and
traces add 0.0025.

---

## 2. Defects in MUAD's code

We use MUAD's code **unchanged** (`project/vendor/muad`), so these defects affect our MUAD numbers
too. We did not fix them: Track A is a faithful reproduction.

### 2.1 The best epoch is chosen on the test set
`project/vendor/muad/training/base_model.py`, lines 86–91:
```python
if epoch % evaluation_epoch == 0 and test_loader:
    test_results = self.evaluate(test_loader)
    if test_results["F1"] > best_f1:
        best_f1 = test_results["F1"]
        best_state = copy.deepcopy(self.model.state_dict())
```
The reported F1 is the **maximum test F1 over epochs**, which is optimistic by construction. In our earlier
ablation runs, which used the same rule, the last-epoch F1 was 0.3–1.1 points below the "best" F1 chosen this way. Early stopping
(lines 76–80) watches the **training** loss, so there is no held-out signal at all.

### 2.2 Predictions are random at test time
`project/vendor/muad/models/main_model.py`, lines 28 and 31–34:
```python
new_sample = self.reparametrize(mu_attn, var_attn)   # called in forward(), also in eval()
def reparametrize(self, mu, var):
    std = var.sqrt(); eps = torch.randn_like(std)
    return mu + eps * std
```
There is no `if self.training` check, so every evaluation draws fresh noise. **Evidence:** the same MUAD
checkpoint gave test F1 0.9884 in one evaluation and 0.9882 in another (`runs/robustness_muad_20261117_113500`
vs. the comparison run). In a probabilistic encoder, test-time predictions should use the mean μ.

### 2.3 The KL term is the wrong formula
`main_model.py`, lines 17–18 and 36–37:
```python
self.fc_var = nn.Sequential(nn.Linear(128, 64), nn.Sigmoid())       # var in (0, 1): a variance
def kl_loss(self, mu, var):
    return -0.5 * torch.mean(torch.sum(1 + var - mu ** 2 - var.exp(), dim=-1))
```
- The standard KL to N(0, I) is −½ Σ(1 + **log σ²** − μ² − **σ²**). It needs the *log*-variance in the
  2nd term and the variance in the last.
- MUAD passes the **variance** where the log-variance belongs (the sampler, `var.sqrt()`, confirms that
  `var` is a variance). The term becomes −½ Σ(1 + σ² − μ² − e^{σ²}).
- With f(v) = 1 + v − eᵛ: f(0) = 0 and f′(v) = 1 − eᵛ < 0 for v > 0. So this "KL" is ≥ 0 and is
  **minimized at σ² = 0, μ = 0**: it pushes the variance to zero. The true KL is minimized at **σ² = 1**.
- So MUAD's "uncertainty" regularizer does not do what the paper describes. It pulls the model towards
  zero variance, the opposite of the intended prior.

### 2.4 The noise is large compared with the signal it is added to (derived from the code)
Both μ and σ² are a sigmoid output (in (0, 1)), multiplied element-wise by a softmax over 64 dimensions
(`SimpleAttention`, `graph_model.py` lines 41–44: `return x * softmax(x W + b)`). The softmax weights sum
to 1, so with spread-out attention each μ entry is about 0.5/64 ≈ 0.008. σ² is about the same size,
which gives a standard deviation of √0.008 ≈ 0.09. **In that regime the noise is about ten times the
mean**, and §2.2 shows this noise is also added at test time. This is a back-of-envelope estimate from the
code, not a measurement on a trained checkpoint. Measuring it on the round 5 checkpoint takes one short
script (we can add it if the supervisor wants it).

### 2.5 The shipped log features are numerically invisible
Descriptive statistics of `recovered_artifact/chunk_train.pkl`: **every log value lies in
[0, 6.06·10⁻¹⁰]** (mean 5.9·10⁻¹⁰, std 9.4·10⁻¹¹). A linear layer with weights around 0.1 turns this into
about 10⁻¹¹, effectively zero. MUAD's code does not normalize it.

**Evidence that MUAD ignores logs and traces** (MUAD's own trained checkpoint, test set, perturbed
at test time; `runs/robustness_muad_20261117_113500`):

| Perturbation at test time | MUAD F1 | MUAD macro-F1 |
|---|---|---|
| Clean | 0.9884 | 0.9663 |
| Remove logs | 0.9884 (identical) | 0.9663 |
| Remove traces | 0.9884 (identical) | 0.9663 |
| Noise on logs / traces, up to 2× std | unchanged (±0.0003) | unchanged |
| Remove metrics | 0.9064 (= all-anomalous) | 0.4532 |

So MUAD, as published, is a **metrics-only detector** on SN. Its confidence-aware fusion has nothing
to fuse. (The old UG-MoE behaved the same before our fixes, so this is a data problem as much as a
model problem. That's why we added normalization for every model, §3.)

### 2.6 Other differences to know (not defects, but relevant)
- MUAD's loss is `CE + 0.6·confidence_loss + KL` (`main_model.py:111`). We train MUAD with exactly this
  loss.
- **Hyperparameters, checked:** the paper uses lr 1e-3 and 120 epochs. Our shared protocol uses lr 5e-4,
  at most 50 epochs, early stopping (patience 15), the same for every model. We trained MUAD once more
  with the paper's settings on rebuilt SN (`runs/round10_muad_paperhp_sn/`). It scored **test macro-F1
  0.793**, against **0.922** with our settings: the paper's settings fit the validation run better (0.973)
  but generalize worse. **Our settings favour MUAD**, so they cannot explain our margin over it. We report
  MUAD's better result.

---

## 3. What we changed in the evaluation (identical for every model)

| Change | Why | Where |
|---|---|---|
| **Run-level split**: whole fault-injection runs go to train / val / test | Removes the 83.9% near-duplicate leak (§1.3). Plan §13: "do not randomly split rows belonging to the same time window". | `common.load_splits(split="run")`, `assign_pipeline_runs` |
| **Validation split, select on val macro-F1, test evaluated once** | Removes test-set selection (§2.1). Plan §6 S14: "best checkpoint based on validation F1 or PR-AUC". Macro-F1 because the trivial predictor wins anomaly-F1 on anomaly-heavy validation sets (§6). | `common.train_and_select`, `_selection_key` |
| **Z-score normalization fitted on the training split only** | Makes the 10⁻¹⁰ log features visible (§2.5). Plan Step 5. | `common.fit_normalizer` |
| **Report macro-F1, normal-class F1, PR-AUC, ROC-AUC and a trivial baseline row** | F1 alone hides imbalance (§1.4). | `common.compute_metrics`, `trivial_baseline_metrics` |
| **Rebuilt SN and TT from the raw Zenodo data** | Adds the fault-free runs MUAD's data left out (realistic class balance), records run and time for every window, MD5-checked downloads. | `project/data_pipeline/` |
| **Deterministic evaluation** for our models | §2.2 | `ug_moe_v2.py` line 208 (`... if self.training else mu`) |

MUAD itself is **not modified**. It only receives the same data, split and selection rule as every
other model.

---

## 4. Results

### 4.1 Reproducing MUAD's protocol (leaky random split, `recovered_artifact`)

| | MUAD test F1 |
|---|---|
| MUAD paper, Dataset C | 0.9928 |
| Ours, same split, epoch chosen on validation (`runs/comparison_20261117_104222`) | **0.9882** (macro-F1 0.9654) |

Our MUAD reproduces the paper (−0.5 points, consistent with choosing on validation instead of test).

### 4.2 Same data, leak-free run split (round 4, `runs/round4_runsplit/`)
Train runs 0–1, val run 2, test run 3. Test 83% anomalous, so the trivial predictor gets F1 0.908.

| Model | Params | Test F1 | F1 normal | Macro-F1 | PR-AUC |
|---|---|---|---|---|---|
| Trivial "all anomalous" | — | 0.9077 | 0 | 0.4538 | 0.8310 |
| UG-MoE (legacy) | 4.4M | 0.9767 | 0.8898 | **0.9332** | **0.9984** |
| MUAD | 0.24M | 0.9767 | 0.8880 | 0.9324 | 0.9934 |
| Plain MoE (legacy) | 4.2M | 0.9688 | 0.8560 | 0.9124 | 0.9957 |

Removing the leak costs MUAD only 1.2 F1 points. Its normal-class F1 of 0.888, however, shows that it
misclassifies about 1 in 9 normal windows, which F1 hides. UG-MoE and MUAD tie.

### 4.3 SN rebuilt from raw (rounds 5 and 6, `runs/round5_sn/`, `runs/round6_v2_sn/`)
7 runs (4 fault + 3 fault-free). Train = 2 fault runs + 1 fault-free; val and test = 1 fault + 1
fault-free run each (about 59% normal). Trivial F1 0.579 / macro-F1 0.289.

| Model | Params | Test F1 | Recall | F1 normal | **Macro-F1** | PR-AUC |
|---|---|---|---|---|---|---|
| Trivial | — | 0.5787 | 1.0000 | 0 | 0.2893 | 0.4071 |
| **UG-MoE v2 (full)** | 166k | **0.9842** | 0.9732 | **0.9894** | **0.9868** | **0.9984** |
| UG-MoE v2 plain (experts + router only) | — | 0.9834 | 0.9740 | 0.9888 | 0.9861 | 0.9972 |
| UG-MoE (legacy) | 4.4M | 0.9813 | 0.9775 | 0.9873 | 0.9843 | 0.9980 |
| Graph-MoE | 0.32M | 0.9681 | 0.9576 | 0.9785 | 0.9733 | 0.9936 |
| Plain MoE (legacy) | 4.2M | 0.9624 | 0.9307 | 0.9756 | 0.9690 | 0.9955 |
| GAT | 0.30M | 0.9579 | 0.9351 | 0.9722 | 0.9650 | 0.9940 |
| **MUAD** | 0.24M | 0.9041 | **0.8571** | 0.9397 | **0.9219** | 0.9691 |

**MUAD is the weakest of all trained models here.** Its main weakness is recall: it misses 14% of
anomalous windows, against 2.7% for UG-MoE v2. On this data the logs are Drain3 template counts, not
MUAD's Hawkes intensities, so MUAD is not on its home features. We think the gap comes from MUAD's
noisy, mis-regularized probabilistic encoder (§2.2–2.4) combined with the harder test run. That is a
hypothesis, not a measured cause.

### 4.4 UG-MoE v2, adding one component at a time (plan §12; round 6, SN)

| Step | Adds | Test macro-F1 | Recall |
|---|---|---|---|
| plain | experts + learned router | 0.9861 | 0.9740 |
| graph | + graph-aware routing, dependency (GAT) expert | 0.9672 | 0.9307 |
| prob | + probabilistic representation (μ, σ², KL) | 0.9475 | 0.9030 |
| ug_routing | + uncertainty-guided routing | 0.9459 | 0.8944 |
| **full** | + uncertainty-aware adaptive fusion | **0.9868** | 0.9732 |

Validation macro-F1 is about 0.98 for **every** step. The middle steps fit the data as well as the
others, but generalize worse to the unseen test run. Adaptive fusion recovers the loss completely. On SN
the components only help **together**, and the full model is not better than "plain" on clean data.

### 4.4a Removing one component at a time (plan §8B ablations; round 9, SN, `runs/round9_abl2_sn/`)

| Model | Test macro-F1 | Δ vs full |
|---|---|---|
| **Full UG-MoE v2** | **0.9868** | — |
| w/o graph | 0.9857 | −0.1 |
| w/o MoE (one expert only) | 0.9089 | **−7.8** |
| w/o dynamic routing (fixed equal expert weights) | 0.9794 | −0.7 |
| w/o uncertainty-guided routing | 0.9480 | **−3.9** |
| w/o probabilistic representation | 0.9828 | −0.4 |
| w/o adaptive fusion | 0.9459 | **−4.1** |

The mixture of experts, uncertainty-guided routing and uncertainty-aware fusion each cost 4–8 points when
removed. Graph, dynamic routing and the probabilistic representation make no measurable difference on
clean SN. The probabilistic representation's contribution is robustness (§4.6). One check on
reproducibility: "w/o adaptive fusion" is the same configuration as the incremental step "+ uncertainty
routing" (§4.4), and it reproduced that result exactly (0.9459). Training is deterministic for a fixed
seed.

**The same ablations on TT (round 15b) give a different picture:** removing the graph (−18.8) or dynamic
routing (−15.6) hurts most, removing the probabilistic representation costs 6.3, and removing the MoE,
uncertainty routing or adaptive fusion makes no difference. The only component that matters on both
datasets is the **probabilistic representation** (TT clean accuracy; SN robustness, §4.6). TT results
fluctuate strongly between training epochs, so with one seed the TT ablation is not reliable enough to
rank components. The 5-seed final runs must settle it.

**Which modalities carry signal (plan Exp. C; round 12, SN):** v2 trained on each subset scores
metrics 0.883, traces 0.935, logs 0.832, traces + logs 0.938, metrics + logs 0.959, metrics + traces 0.929,
and **all three 0.987**. Every modality carries real signal, and combining all three adds +2.8 over the
best pair.

### 4.4b TrainTicket (TT), the harder dataset (rounds 7 and 7-v2, `runs/round7_tt/`, `runs/round7_v2_tt/`)
11 runs (9 fault, 2 fault-free), 27 services. Train = 5 fault runs + the first 60% of both fault-free
runs; val / test = 2 fault runs each + later 20% segments of the fault-free runs. Test 68% anomalous
(trivial F1 0.809, macro-F1 0.404).

| Model | Test F1 | Recall | F1 normal | **Macro-F1** | PR-AUC |
|---|---|---|---|---|---|
| Trivial | 0.8087 | 1.0000 | 0 | 0.4044 | 0.6789 |
| **UG-MoE v2 full** | **0.8943** | **0.9237** | **0.7521** | **0.8232** | **0.9527** |
| UG-MoE v2 up to uncertainty-guided routing | 0.9163 | 0.9968 | 0.7635 | 0.8399 | 0.9505 |
| UG-MoE v2 plain | 0.6949 | 0.5543 | 0.6396 | 0.6673 | 0.9243 |
| GAT | 0.7854 | 0.6977 | 0.6740 | 0.7297 | 0.9416 |
| Plain MoE (legacy) | 0.6810 | 0.5205 | 0.6559 | 0.6685 | 0.9089 |
| **MUAD** | 0.6801 | 0.5475 | 0.6145 | **0.6473** | 0.9169 |
| Graph-MoE | 0.6494 | 0.4977 | 0.6197 | 0.6345 | 0.9346 |

- **UG-MoE v2 beats MUAD by +17.6 macro-F1 and the best baseline (GAT) by +9.4.**
- Every baseline is precise but **misses 30–50% of the anomalies**. Its anomaly-F1 is *below* the
  trivial predictor, which is another reason to report macro-F1. UG-MoE v2 detects 92% of them.
- **Here the uncertainty components improve clean accuracy:** uncertainty-guided routing lifts macro-F1
  from 0.58 to 0.84. On SN, which is near its ceiling, they only improved robustness (§4.6).
- **Which anomalies:** recall by fault type on the TT test set:

  | Model | CPU load | Network delay | Network loss | False alarms on normal windows |
  |---|---|---|---|---|
  | MUAD | 0.949 | 0.389 | 0.302 | 13.1% |
  | GAT | 0.963 | 0.645 | 0.486 | 16.7% |
  | **UG-MoE v2** | **1.000** | **0.895** | **0.877** | 30.0% |

  CPU faults are easy for every model. The baselines miss most **network faults**, which UG-MoE v2
  detects. The price is more false alarms (30% of normal windows flagged vs 13–17%), which should be
  stated in the paper.
- Caveat: TT's validation scores fluctuate strongly between epochs, so the 5-seed final runs are
  needed to fix the size of this gain.

### 4.5 Robustness on rebuilt SN (plan Exp. D/E; round 8, `runs/round8_rob_sn_v2/`, `runs/round8_rob_sn_muad/`)
Saved checkpoints from §4.3, perturbed only at test time. The clean scores reproduce §4.3 exactly, which
confirms the checkpoints loaded correctly. Masking = random (window, service, modality) entries set to
zero; noise = Gaussian with σ = level × the training standard deviation per channel. Test F1 (relative drop):

| Perturbation | UG-MoE v2 full | MUAD |
|---|---|---|
| Clean | 0.9842 | 0.9037 |
| Random mask 10% | 0.962 (−2.3%) | 0.857 (−5.2%) |
| Random mask 20% | 0.931 (−5.5%) | 0.809 (−10.5%) |
| Random mask 40% | 0.856 (−13.0%) | 0.716 (−20.8%) |
| Random mask 60% | 0.748 (−24.0%); macro-F1 0.763 | 0.662 (−26.8%); macro-F1 0.584 |
| Noise on metrics, σ = 1 × std | 0.964 (−2.1%) | 0.856 (−5.3%) |
| Noise on metrics, σ = 2 × std | 0.872 (−11.4%) | 0.742 (−17.9%) |
| Noise on all three modalities, σ = 2 | 0.790 (−19.8%) | 0.699 (−22.7%) |
| Drop traces / drop logs | 0.979 / 0.981 | 0.901 / 0.897 |
| **Drop metrics** | **0.579 = trivial** | **0.579 = trivial** |

- UG-MoE v2 is **more robust than MUAD in every masking and noise setting**: higher absolute F1, and
  about **half the relative drop** at moderate perturbation (masking ≤ 40%, noise σ ≤ 1).
- **Both models fail without metrics.** Traces and logs now contribute a little (they had no effect
  before normalization, §2.5), but neither model can detect anomalies from them alone. Graceful
  degradation when the dominant modality is missing is **not achieved by either model**.
### 4.6 Where the robustness comes from: the same test on every incremental step (round 8)
Same checkpoints as §4.4. The masks and noise are drawn from the same random seeds for every model, so
each column compares the models on identical perturbations. Test F1 (relative drop):

| Model | Clean | Mask 20% | Mask 40% | Mask 60% | Drop logs |
|---|---|---|---|---|---|
| MUAD | 0.904 | 0.809 (10.5%) | 0.716 (20.8%) | 0.662 (26.8%) | 0.897 |
| v2 plain (experts + router) | 0.983 | 0.824 (16.2%) | 0.722 (26.6%) | 0.669 (32.0%) | 0.939 (4.6%) |
| + graph | 0.960 | 0.826 (14.0%) | 0.709 (26.2%) | 0.647 (32.6%) | 0.891 |
| + probabilistic representation | 0.936 | 0.875 (6.6%) | 0.767 (18.1%) | 0.677 (27.7%) | 0.922 |
| + uncertainty-guided routing | 0.934 | 0.815 (12.8%) | 0.734 (21.4%) | 0.680 (27.2%) | 0.874 |
| **full (+ adaptive fusion)** | **0.984** | **0.931 (5.5%)** | **0.856 (13.0%)** | **0.748 (24.0%)** | **0.981 (0.3%)** |

- **Direct test** (round 15a): the full model *without* the probabilistic representation, everything
  else equal, has the same clean F1 (0.979) but drops to 0.803 / 0.707 at 20% / 40% masking (full:
  0.931 / 0.856). Removing that one component costs 12.8 / 14.9 points under missing data.
- **On clean data full ≈ plain, but with 20% / 40% of observations missing, full is +10.7 / +13.4 F1
  points better.** The plain MoE is no more robust than MUAD. So the robustness gain over MUAD comes from
  the **uncertainty components**, not from having experts.
- The **probabilistic representation** is the step that first makes the model robust (relative drop at
  20% masking falls from 16.2% to 6.6%). **Uncertainty-aware adaptive fusion** then restores the clean
  accuracy and gives the best absolute robustness. With it the model also barely notices missing logs.
- Limit: under extreme noise on *all* modalities (σ = 2 × std), plain is better (0.835 vs 0.790). The
  claim is robustness to **missing data and moderate noise**.

### 4.7 Does the uncertainty do what it is meant to? (plan Exp. F/G; `runs/round13_analysis_sn/`)
We evaluated the trained full v2 model under perturbations and recorded, per modality, its predicted
uncertainty (log-variance) and its fusion weight.

| Condition | F1 | Fusion weight metric / trace / log | Uncertainty of the perturbed modality |
|---|---|---|---|
| clean | 0.984 | 0.959 / 0.029 / 0.013 | (metric −0.22, trace −0.03, log +0.03) |
| noise on metrics (σ = 1) | 0.971 | **0.910** / 0.062 / 0.028 | metric −0.22 → **−0.07** |
| 40% masking | 0.846 | **0.889** / 0.068 / 0.043 | metric −0.22 → **+0.09** |
| traces dropped | 0.979 | 0.972 / **0.015** / 0.013 | trace −0.03 → **+0.08** |
| logs dropped | 0.981 | 0.964 / 0.021 / 0.015 | log +0.03 → **+0.15** |
| metrics dropped | 0.579 | 0.880 / 0.093 / 0.027 | metric −0.22 → −0.02 |

- **Yes, in direction:** when a modality is corrupted, the model's uncertainty for that modality goes
  up, and its fusion weight goes down in favour of the others. This is the behaviour the method claims,
  and MUAD's code cannot produce it reliably (§2.2–2.3).
- **Not in size:** metrics carry 96% of the weight on clean data and still 88% when they are missing,
  so the shift is too small to hand the decision to traces and logs. This is why both models fail
  without metrics (§4.5). The planned remedy is modality dropout during training.
- **Uncertainty flags errors:** the uncertainty is higher on wrong predictions (−0.04) than on
  correct ones (−0.23).
- **The experts specialize by fault type** (metric routing): normal windows go mostly to the dependency
  (GAT) expert (0.87), CPU-load faults to the cross-modal expert (0.95), network delay / loss mainly to the
  temporal expert (~0.5). So the router learned fault-specific paths, as intended (plan Exp. F).

### 4.8 Robustness and uncertainty on TT (rounds 11 and 14): where the claims stop holding
On TT, anomaly-F1 is misleading under perturbation. MUAD and GAT miss many anomalies, so corrupting
their input makes them flag more windows, and their anomaly-F1 *rises* towards the trivial 0.809. We
therefore compare macro-F1:

| Model | Clean | 10% masked | 20% masked | 40% masked | 60% masked |
|---|---|---|---|---|---|
| MUAD | 0.647 | 0.636 | 0.616 | 0.577 | 0.559 |
| GAT | 0.730 | 0.652 | 0.615 | 0.540 | 0.471 |
| **UG-MoE v2** | **0.823** | **0.734** | **0.672** | **0.581** | 0.547 |

- UG-MoE v2 stays best up to 40% missing data, but its lead shrinks from +9 / +18 points to about level.
  **The large robustness advantage seen on SN does not carry over to TT.**
- **The uncertainty mechanism works strongly on TT when metrics are missing:** metric uncertainty jumps
  (−0.28 → +0.80) and its fusion weight falls from 0.61 to 0.13, with traces taking over (0.79). Detection
  still fails, because TT traces cover only 2–4% of seconds: the modality the model falls back on is
  nearly empty. That is a limit of the data, not of the mechanism.
- **Failure case:** with noise on the logs, the model becomes *more* confident in the logs and raises
  their weight from 0.11 to 0.65, which costs 8 F1 points. Learned variance can be overconfident on
  inputs unlike the training data, a known limitation that the paper should state. On TT the uncertainty
  also does not separate correct from wrong predictions (unlike SN).

---

## 5. Why UG-MoE is better than MUAD, and how far the evidence goes

**Supported by the results:**
1. **Higher accuracy on realistic data:** +6.5 macro-F1 over MUAD on rebuilt SN (§4.3), mostly
   from recall (0.973 vs 0.857). With one seed, but far above the 1-point noise threshold.
2. **Never worse than MUAD** on any leak-free setting so far (tie on MUAD's own data, §4.2).
3. **More robust to missing and noisy data** (§4.5), and this comes from the uncertainty components
   (§4.6: +10.7 / +13.4 F1 over the plain variant at 20 / 40% masking).
4. **Smaller:** 166k parameters vs 240k for MUAD, and 25× smaller than our earlier 4.4M model with
   the same accuracy.

**Supported by the design (verified in the code, `project/tracks/proposed_model/ug_moe_v2.py`):**
1. **Correct probabilistic head.** Experts predict a mean and a **log**-variance. The KL is the
   standard formula `0.5·(μ² + σ² − 1 − log σ²)` (line 209), unlike MUAD's (§2.3).
2. **Deterministic evaluation:** sampling only in training, μ at test time (line 208), unlike MUAD (§2.2).
3. **Specialized experts** (temporal GRU, cross-service attention, dependency GAT, cross-modal attention),
   shared by all modalities, with **one router per modality**. The router sees the modality's features, a
   graph-context vector and each expert's predicted uncertainty. MUAD has one fixed encoder per modality.
4. **Fusion weighted by predicted uncertainty** at modality level, which is what plan §6 asks for.
5. All models get normalized inputs, so logs and traces are actually visible (§2.5).

**Not yet supported, so don't claim it yet:**
- That the *uncertainty* components (probabilistic representation, uncertainty routing) improve accuracy:
  on SN the full model equals "plain" (§4.4).
- Graceful degradation when metrics are missing: neither model has it (§4.5).
- Accuracy gains from the uncertainty components on *clean* data: none on SN (§4.4), large on TT
  (§4.4b). Single seed, so the TT gain needs the 5-seed runs.
- Robustness on TT beyond 20% missing data (§4.8), and reliable uncertainty under noise the model never
  saw in training (the log-noise failure case, §4.8).

---

## 6. Caveats and corrections we made along the way

- **Single seed.** No standard deviation and no significance test yet. Final paper runs will use 5 seeds.
- **TT round 5 was void.** TT has only 2 fault-free runs, and the first split put none in validation.
  Validation was then 84.8% anomalous, the trivial predictor scored **0.918 validation F1**, and
  selecting on anomaly-F1 picked degenerate "all anomalous" epochs for 4 of 5 models. Fixed by selecting
  on macro-F1 and cutting each TT fault-free run chronologically into train / val / test segments.
  Rerun = round 7.
- **The "GAT / Graph-MoE collapse" seen in round 4** may partly be the same selection effect
  (validation run 83% anomalous, selected at epochs 4 and 2). On rebuilt SN both work well (§4.3).
- **Split caveats (TT only, deferred):** fault-free test segments come from the same runs as fault-free
  training segments; there is no gap at the cuts (adjacent windows share no seconds). For SN and TT:
  fault-free runs were recorded after the fault runs; Drain3 templates are mined over all runs
  (label-free). Planned before the final runs: a 60-second gap at cuts and templates mined on training
  runs only.
- **Residual artifact caveat:** Eadro z-scored metrics and traces per run over the whole run before
  windowing (a mild, label-free normalization leak in `recovered_artifact` that we can't undo). The
  rebuilt data does not have it: raw values, normalized with training statistics only.

---

## 7. How to check any number here

All runs are on the GPU server under `project/experiments/runs/<name>/`. Each has `summary.json`,
`split.json` (exact window ids per split), `normalization.json` and one `<model>_seed42.json` with the
full per-epoch validation history.

```bash
cd project
# leaky reproduction of the paper's protocol
python experiments/run_comparison.py --seeds 42 --device cuda --split random
# leak-free, MUAD's data
python experiments/run_baselines.py --device cuda --models ug_moe,muad,plain_moe
# rebuilt SN
python -m data_pipeline.fetch_eadro && python -m data_pipeline.build_eadro --system SN
python experiments/run_baselines.py --device cuda --data-dir datasets/eadro/processed/SN_w10_s1 \
  --models muad,ug_moe2:plain,ug_moe2:full
```

Code references: leakage measurement and split, `project/experiments/common.py`; MUAD,
`project/vendor/muad/models/main_model.py` and `training/base_model.py`; UG-MoE v2,
`project/tracks/proposed_model/ug_moe_v2.py`; detailed log of every round, `PROJECT_STATUS.md` §4.
