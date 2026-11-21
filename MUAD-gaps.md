# MUAD verification on SocialNetwork

Audit performed on 21 November 2026. The recovered SocialNetwork (SN) data contain overlapping windows across training and test, and MUAD's public training code selects checkpoints using test F1. In the public probabilistic encoder, the sampler treats its output as variance while the KL term treats the same output as log-variance. A fresh run of the saved seed-42 checkpoint confirmed stochastic inference and found that removing traces changed logits while preserving every class prediction.

The reference is Dou et al., *Uncertainty-Aware Multimodal Anomaly Detection for Microservice Systems With Active Learning*, IEEE Transactions on Services Computing, DOI 10.1109/TSC.2026.3672587. This audit checks the claims in the earlier comparison note against the paper, the public implementation, and the files available in this repository. [Reference PDF](7.%20Uncertainty-Aware_Multimodal_Anomaly_Detection_for_Microservice_Systems_With_Active_Learning.pdf), [public MUAD repository](https://github.com/slg-frank/muad), [earlier comparison note](MUAD_vs_UG-MoE.md)

## 1. Overlapping windows cross the train/test split

**Confirmed on the recovered SN artifact.** Section IV-A of the paper says that 60% of each dataset was randomly selected for training and the remaining 40% for testing, with similar class proportions. It specifies a 10-step window and stride 1, and identifies Dataset C as Eadro's SocialNetwork data. In the recovered files there are exactly **3,366 training windows and 2,244 test windows**, and every metrics window has 10 steps. The recovered train and test pickles are byte-identical to their respective copies in the vendored MUAD dataset directory. [Paper, §IV-A, pp. 2567–2568](7.%20Uncertainty-Aware_Multimodal_Anomaly_Detection_for_Microservice_Systems_With_Active_Learning.pdf#page=6), [recovered training pickle](project/recovered_artifact/chunk_train.pkl), [recovered test pickle](project/recovered_artifact/chunk_test.pkl), [vendored dataset copies](project/vendor/muad/Dataset/chunk)

**Proof.** I matched each window's last nine metric steps against every window's first nine steps, across all services and metric channels. Each nine-step prefix and suffix had a unique match key. The matches form four chains of **1,402, 1,402, 1,402, and 1,404** windows, covering all 5,610 windows. **1,882 of 2,244 test windows (83.87%)** have at least one immediate neighbor in the original training pickle. Because each such pair shares nine seconds of identical metrics, this split tests heavily on telemetry already present in training. The measurement applies to the recovered split. The paper gives a matching split ratio and window settings, and the reported class proportions are consistent with the artifact. Verifying exact split identity would require the paper's window IDs or original run files. [Repository's run reconstruction](project/experiments/common.py#L99-L136), [Eadro window generation](project/vendor/eadro/codes/preprocess/align.py#L16-L35), [Eadro shuffle and split](project/vendor/eadro/codes/preprocess/align.py#L127-L147), [reproduction below](#reproduce-the-checks)

## 2. Anomaly-class F1 is high for a constant predictor

**Confirmed on the recovered test set.** There are **1,860 anomalous and 384 normal windows**, so the anomaly prevalence is **82.8877%**. Predicting anomaly on every window gives TP = 1,860, FP = 384, FN = 0, and TN = 0. Its precision is 0.8289, recall is 1.000, anomaly-class F1 is `2 × 1860 / (2 × 1860 + 384) = 0.9064`, and macro-F1 is **0.4532** because normal-class F1 is zero. [Saved test label distribution and trivial baseline](project/experiments/runs/comparison_20261117_104222/summary.json), [metric implementation](project/experiments/common.py#L350-L373)

Table III reports **0.9064 F1, 1.000 recall, and 0.8289 precision** for Dataset C's **Only Log** variant. The match to the constant predictor at the reported precision is consistent with constant predictions. An exact statement about the paper's per-window predictions requires its prediction files. The same table reports **0.9903** F1 for Only Metric, **0.9470** for Only Trace, and **0.9928** for full MUAD. The **0.0025** difference compares the trained variants' overall F1 scores. [Paper, §IV-A.3 and Table III, pp. 2568 and 2570](7.%20Uncertainty-Aware_Multimodal_Anomaly_Detection_for_Microservice_Systems_With_Active_Learning.pdf#page=9), [local Table III transcription](7.%20Uncertainty-Aware_Multimodal_Anomaly_Detection_for_Microservice_Systems_With_Active_Learning.md#c-rq2-ablation-study)

## 3. The public trainer selects checkpoints on test F1

**Confirmed in both the public and vendored source.** The public full-label entry point passes `test_loader` into `fit()`. At each evaluation interval, `fit()` evaluates the test set, saves weights when test F1 improves, and restores the best weights. The entry point evaluates the restored checkpoint on the same test set again. The training-loss early-stopping rule is a separate mechanism. This means the test set influences checkpoint choice even though its examples never enter gradient updates. Stochastic inference can make the final re-evaluation score differ from the score that selected the checkpoint. [Public full-label entry point](https://github.com/slg-frank/muad/blob/master/main.py#L38-L69), [public training loop](https://github.com/slg-frank/muad/blob/master/training/base_model.py#L52-L93), [vendored training loop](project/vendor/muad/training/base_model.py#L54-L97)

The public active-learning branch also passes in the test loader to select checkpoints and uses test precision to decide whether to stop the outer loop. Its variable named `val_loader` supplies the candidate pool for active learning. The paper describes a train/test split and training duration, with no explicit rule for selecting the reported epoch. The code establishes how the released implementation behaves. The exact checkpoint procedure used internally for the published tables remains unspecified. [Public active-learning branch](https://github.com/slg-frank/muad/blob/master/main.py#L70-L136), [paper, §IV-A](7.%20Uncertainty-Aware_Multimodal_Anomaly_Detection_for_Microservice_Systems_With_Active_Learning.pdf#page=7)

The final UG-MoE comparison invokes MUAD's model through the shared experiment trainer. That trainer selects on **validation macro-F1**, breaks ties by validation PR-AUC, restores the best weights, and then evaluates test. The vendored model includes compatibility changes such as label-device handling and the ability to pass the log feature dimension from metadata. Its sampling and KL behavior match the public implementation. [Model registry](project/experiments/registry.py#L90-L102), [shared selection and training](project/experiments/common.py#L417-L500), [vendored model](project/vendor/muad/models/main_model.py#L40-L84), [public model](https://github.com/slg-frank/muad/blob/master/models/main_model.py#L40-L84)

## 4. Inference depends on the random seed

**Confirmed in code and by rerunning the checkpoint.** `UncertainBlock.forward()` calls `reparametrize()` in both training and evaluation. The sampler draws fresh standard-normal noise with `torch.randn_like()`. PyTorch's `eval()` changes module behavior such as dropout, while this explicit sampling call still runs. With the same checkpoint, unchanged inputs, the saved test order, and CPU evaluation, changing only the RNG seed from 42 to 123 changed **one of 2,244 labels** and moved anomaly F1 from **0.9884378 to 0.9881720**. The largest absolute logit change was **0.335833**. A probabilistic detector can sample at inference and average multiple draws. This implementation uses one fresh draw per forward pass. [Public uncertainty block](https://github.com/slg-frank/muad/blob/master/models/main_model.py#L21-L37), [saved checkpoint](project/experiments/runs/comparison_20261117_104222/track_a_seed42_best.pth), [saved test order](project/experiments/runs/comparison_20261117_104222/split.json), [reproduction below](#reproduce-the-checks)

## 5. The KL implementation uses variance as log-variance

**Confirmed by comparing the paper's Eq. (5) with the sampler and KL implementation.** Let $v = \sigma^2 > 0$ be a single latent coordinate's variance. The sampler multiplies standard-normal noise by `sqrt(var)`, so `var` is used as $v$. The KL expression expects log-variance, and `forward()` passes the same variance to it. [Paper, Eq. (5), p. 2566](7.%20Uncertainty-Aware_Multimodal_Anomaly_Detection_for_Microservice_Systems_With_Active_Learning.pdf#page=5), [public implementation](https://github.com/slg-frank/muad/blob/master/models/main_model.py#L17-L37), [vendored implementation](project/vendor/muad/models/main_model.py#L17-L37)

| Per-coordinate penalty | Formula | Derivative in $v$ | Where the variance term is minimized |
|---|---|---|---|
| Gaussian KL to the standard normal | $\tfrac12(\mu^2 + v - 1 - \log v)$ | $\tfrac12(1 - 1/v)$ | At $v = 1$ |
| MUAD code with the supplied variance | $\tfrac12(\mu^2 + e^v - 1 - v)$ | $\tfrac12(e^v - 1)$ | As $v \to 0^+$ |

For example, at `mu = 0, v = 0.01`, the intended penalty is **1.807585** and the implemented penalty is about **0.000025**. At `mu = 0, v = 1`, the intended penalty is **0** and the implemented penalty is **0.359141**. This proves the regularizer gives the opposite incentive for small positive variances. Its effect on final accuracy would require retraining under a controlled comparison. The paper's printed Eq. (4) also multiplies the noise by the quantity it labels variance. For sampling from the Gaussian defined in Eq. (3), the multiplier should be its square root, as used in the code. [Paper, Eqs. (3)–(5)](7.%20Uncertainty-Aware_Multimodal_Anomaly_Detection_for_Microservice_Systems_With_Active_Learning.pdf#page=5), [local transcription](7.%20Uncertainty-Aware_Multimodal_Anomaly_Detection_for_Microservice_Systems_With_Active_Learning.md#c-graph-based-probabilistic-encoder-gpe)

## 6. Logs, traces, and the magnitude of sampling noise

**The recovered log features are extremely small.** Across the raw training pickle, their minimum is **0**, maximum **6.0600 × 10⁻¹⁰**, mean **5.9096 × 10⁻¹⁰**, and standard deviation **9.4259 × 10⁻¹¹**. The public dataset loader converts them to float tensors and feeds them to the log encoder without rescaling. [Training artifact](project/recovered_artifact/chunk_train.pkl), [public loader](https://github.com/slg-frank/muad/blob/master/data/dataset.py#L7-L21), [public log encoder](https://github.com/slg-frank/muad/blob/master/models/encoders.py#L28-L36)

**Paired ablation.** I zeroed one modality at a time for the saved seed-42 MUAD checkpoint on the 2,244 raw test windows. For each pass I used the same batch order and reset the RNG to seed 42, so the same random draws were used. Changes are relative to clean inference. [Checkpoint and saved test order](project/experiments/runs/comparison_20261117_104222), [earlier robustness run](project/experiments/runs/robustness_muad_20261117_113500/summary.json#L33-L86), [reproduction below](#reproduce-the-checks)

| Input condition | Anomaly F1 | Macro-F1 | Predictions changed | Largest absolute logit change |
|---|---:|---:|---:|---:|
| Clean | 0.988438 | 0.966260 | 0 | 0 |
| Logs zeroed | 0.988438 | 0.966260 | 0 | 0 |
| Traces zeroed | 0.988438 | 0.966260 | 0 | 0.524549 |
| Metrics zeroed | 0.906433 | 0.453216 | 385 | 5.300615 |

Logs had no measurable effect on this checkpoint's logits. Traces affected the scores while leaving the binary decisions unchanged. Removing metrics made all 2,244 windows predict as anomalous. These results establish metrics dominance for the **binary decisions of this checkpoint on raw SN**. Table III's trace-only result comes from a separately trained model. The saved robustness JSON also shows small changes in macro-F1 under some log/trace noise conditions, so the earlier note's “unchanged” wording is too broad. [Earlier wording](MUAD_vs_UG-MoE.md#25-the-shipped-log-features-are-numerically-invisible), [saved perturbation results](project/experiments/runs/robustness_muad_20261117_113500/summary.json)

**Measured noise scale.** The earlier `~10×` estimate assumes that both sigmoid outputs are near 0.5 and that attention is nearly uniform over 64 dimensions. I collected the actual post-attention means and variances from the checkpoint. The RMS noise scale is `sqrt(mean(variance))`, which is the expected noise RMS across latent coordinates and windows. The RMS latent mean is `sqrt(mean(mean_vector ** 2))`, over the same coordinates and windows. [Earlier estimate](MUAD_vs_UG-MoE.md#24-the-noise-is-large-compared-with-the-signal-it-is-added-to-derived-from-the-code), [attention implementation](project/vendor/muad/models/graph_model.py#L34-L48), [reproduction below](#reproduce-the-checks)

| Modality | RMS latent mean | RMS noise scale | Noise / mean |
|---|---:|---:|---:|
| Metrics | 0.057655 | 0.020834 | 0.361× |
| Traces | 0.032581 | 0.022953 | 0.705× |
| Logs | 0.019944 | 0.024471 | 1.227× |

## 7. How to interpret the reported scores

The paper's Dataset C score is **0.9928 anomaly F1**. The saved local original-split run gives **0.9882 anomaly F1 and 0.9654 macro-F1**, using a validation subset for checkpoint selection. The saved run-level comparison on recovered data gives **0.9767 anomaly F1 and 0.9324 macro-F1**. That run also changes normalization and which windows are tested. These runs show that the score is sensitive to the evaluation setup. They leave the separate causal effects of overlap, checkpoint selection, and normalization unmeasured. The random stride-one protocol cannot establish generalization to unseen runs. [Paper Table III](7.%20Uncertainty-Aware_Multimodal_Anomaly_Detection_for_Microservice_Systems_With_Active_Learning.pdf#page=9), [original-split score and validation history](project/experiments/runs/comparison_20261117_104222/track_a_seed42.json#L1-L25), [run-level score](project/experiments/runs/round4_runsplit/baselines_20261117_125419/muad_seed42.json#L1-L25), [run-level configuration](project/experiments/runs/round4_runsplit/baselines_20261117_125419/summary.json)

The paper specifies a learning rate of **0.001** and **120 epochs per training stage**. On rebuilt SN, the saved seed-42 run using those two settings under the shared trainer reaches **0.7928 test macro-F1**, versus **0.9219** for the shared defaults in the earlier SN run. That checks one seed and one pair of settings. It gives no estimate of MUAD's best attainable score under a broader tuning sweep. [Paper, §IV-A.4](7.%20Uncertainty-Aware_Multimodal_Anomaly_Detection_for_Microservice_Systems_With_Active_Learning.pdf#page=7), [paper-settings run](project/experiments/runs/round10_muad_paperhp_sn/baselines_20261117_202530/summary.json), [shared-settings run](project/experiments/runs/round5_sn/baselines_20261117_135658/summary.json)

I recomputed the means and paired two-sided t-tests from the saved per-seed macro-F1 results. All five seeds use the same data split within each benchmark. The p-values measure consistency across these training seeds. The results quantify performance against the released MUAD model as adapted for the shared protocol. They leave open how a corrected or differently tuned MUAD would compare. [Saved five-seed tables](project/experiments/runs/report_20261118_201618/tables.md), [per-seed data source and pairing code](project/experiments/make_report.py#L101-L153)

| Benchmark | MUAD macro-F1 | Full UG-MoE v2 macro-F1 | Difference in points | Paired p-value |
|---|---:|---:|---:|---:|
| SocialNetwork | 0.9251 | 0.9715 | +4.65 | 0.002071 |
| TrainTicket | 0.6262 | 0.7234 | +9.72 | 0.042000 |

## Reproduce the checks

Run the first block from the repository root with an environment that can import NumPy, PyTorch, DGL, and scikit-learn. It reads the local recovered artifact and the saved checkpoint, and runs inference on CPU. The reference run used Python 3.11, NumPy 1.26.4, PyTorch 2.2.1+cu118, and DGL 2.2.1+cu118. The checkpoint and run outputs live in the local experiment workspace. [Environment setup](activate_conda.sh), [saved evaluation inputs](project/experiments/runs/comparison_20261117_104222)

```bash
python -B - <<'PY'
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from sklearn.metrics import f1_score
import torch

project = Path("project").resolve()
artifact = project / "recovered_artifact"
run = project / "experiments/runs/comparison_20261117_104222"
vendor = project / "vendor/muad"
sys.path.insert(0, str(vendor))
from data.dataset import ChunkDataset
from data.utils import create_dataloader
from models.main_model import MainModel

def load_chunks(split):
    name = f"chunk_{split}.pkl"
    path = artifact / name
    assert path.read_bytes() == (vendor / "Dataset/chunk" / name).read_bytes()
    with path.open("rb") as file:
        return pickle.load(file)

train = load_chunks("train")
test = load_chunks("test")
windows = {**train, **test}
prefixes = {c["metrics"][:, :-1].tobytes(): key for key, c in windows.items()}
suffixes = {c["metrics"][:, 1:].tobytes(): key for key, c in windows.items()}
assert len(prefixes) == len(suffixes) == len(windows) == len(train) + len(test)
successors = {key: prefixes[end] for end, key in suffixes.items() if end in prefixes}
predecessors = {right: left for left, right in successors.items()}

visited = set()
chain_lengths = []
for key in windows.keys() - predecessors.keys():
    length = 0
    while key is not None:
        assert key not in visited
        visited.add(key)
        length += 1
        key = successors.get(key)
    chain_lengths.append(length)
assert len(visited) == len(windows)

overlap = sum(predecessors.get(k) in train or successors.get(k) in train for k in test)
positives = sum(c["culprit"] >= 1 for c in test.values())
logs = np.stack([c["logs"] for c in train.values()]).astype(np.float64)
print("Counts and chain lengths", len(train), len(test), sorted(chain_lengths))
print("Test windows with training neighbors", overlap, overlap / len(test))
print("Anomalous and normal", positives, len(test) - positives)
print("Always anomalous precision, F1, macro-F1",
      positives / len(test), 2 * positives / (len(test) + positives),
      positives / (len(test) + positives))
print("Training log min, max, mean, std", logs.min(), logs.max(), logs.mean(), logs.std())

torch.set_num_threads(2)
metadata = json.loads((artifact / "metadata.json").read_text())
keys = json.loads((run / "split.json").read_text())["test_ids"]
dataset = ChunkDataset(test, keys, metadata["node_num"], tuple(metadata["edges"]))
loader = create_dataloader(dataset, batch_size=50, shuffle=False)
model = MainModel(torch.device("cpu"), hidden_dim=[64, 64], event_num=metadata["event_num"])
state = torch.load(run / "track_a_seed42_best.pth", map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval()

@torch.no_grad()
def evaluate(seed, removed=None):
    torch.manual_seed(seed)
    logits, labels = [], []
    for graph, target in loader:
        if removed is not None:
            graph.ndata[removed] = torch.zeros_like(graph.ndata[removed])
        logits.append(model(graph, target)["MMlogit"])
        labels.append(target >= 1)
    return torch.cat(logits), torch.cat(labels)

def collect_into(target):
    def hook(module, inputs, output):
        target.append(output.detach())
    return hook

modalities = ("metric", "trace", "log")
latents, hooks = {}, []
for modality in modalities:
    block = getattr(model, f"{modality}_uncertain_block")
    for field in ("mu", "var"):
        latents[modality, field] = []
        layer = getattr(block, f"attention_{field}")
        hooks.append(layer.register_forward_hook(collect_into(latents[modality, field])))

clean, labels = evaluate(42)
for hook in hooks:
    hook.remove()

def report(name, logits):
    pred = logits.argmax(dim=1)
    print(name, "F1", f1_score(labels, pred), "macro-F1",
          f1_score(labels, pred, average="macro"), "predicted anomalies", pred.sum().item(),
          "changed labels", (pred != clean.argmax(dim=1)).sum().item(),
          "max abs logit change", (logits - clean).abs().max().item())

report("clean", clean)
report("evaluation seed 123", evaluate(123)[0])
for removed in ("logs", "traces", "metrics"):
    report(f"drop {removed}", evaluate(42, removed)[0])

for modality in modalities:
    mu = torch.cat(latents[modality, "mu"])
    var = torch.cat(latents[modality, "var"])
    rms_mu = mu.square().mean().sqrt().item()
    rms_noise = var.mean().sqrt().item()
    print(modality, "RMS latent mean, RMS noise, ratio", rms_mu, rms_noise, rms_noise / rms_mu)
PY
```

Run the second block from the same directory to recompute the paired comparisons from the run JSONs. It uses the report generator's manifest to locate the per-seed files and uses seeds **42, 123, 456, 789, and 999** for both models. [Report manifest](project/experiments/make_report.py#L32-L43), [saved results](project/experiments/runs/report_20261118_201618/tables.md)

```bash
python -B - <<'PY'
from pathlib import Path
import sys

import numpy as np
from scipy.stats import ttest_rel

experiments = Path("project/experiments").resolve()
sys.path.insert(0, str(experiments))
from make_report import MANIFEST, merged_models

seeds = [42, 123, 456, 789, 999]
runs = experiments / "runs"
for benchmark in ("sn", "tt"):
    patterns = [pattern for pattern in MANIFEST[f"main_{benchmark}"] if any(runs.glob(pattern))]
    results, _ = merged_models(runs, patterns)
    muad = [results["muad"]["by_seed"][seed]["macro_f1"] for seed in seeds]
    full = [results["ug_moe2:full"]["by_seed"][seed]["macro_f1"] for seed in seeds]
    print(benchmark.upper(), "MUAD", np.mean(muad), "UG-MoE v2", np.mean(full),
          "p", ttest_rel(full, muad).pvalue)
PY
```
