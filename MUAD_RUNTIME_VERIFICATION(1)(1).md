# MUAD Runtime Verification

## Objective

Determine, by actually executing it, whether the official MUAD repository (`github.com/slg-frank/muad`) runs end-to-end against the recovered Dataset C artifact, exactly as released. This document records only what was empirically observed by running unmodified MUAD source code. No source files in the repository were edited. All environment changes are documented in full.

This document resolves — with direct execution evidence — the primary open question left unresolved in `MUAD_MODEL_FORENSICS_COMPLETE.md` §10/§24: whether the `hidden_dim`/`MMClasifier`/192-dim-feature shape situation actually causes a runtime failure, and why the recovered `training.log` nonetheless showed successful-looking runs.

---

## Phase A — Environment Forensics

| Dependency | Required (per current `requirements.txt`, HEAD `58923ca`) | Required per commit `6031ccf`'s `requirements.txt` | Installed (this verification) | Compatibility Status |
|---|---|---|---|---|
| Python | Not pinned (Windows conda env implied by file paths) | Not pinned | 3.12.3 | Deviation (repo's original env is Windows/older Python; exact original Python version UNKNOWN) |
| PyTorch (`torch`) | `1.13.1` | Not specified — file was the single-line comment `# Required packages` only | `2.2.1` | **Deviation — 1.13.1 not installable from this container's PyPI mirror (only ≥2.2.0 available)** |
| DGL (`dgl`) | `2.0.0` | Not specified | `2.1.0` | **Deviation — 2.0.0 not available from this mirror (only 0.1.x and 2.1.0 available)** |
| torchdata | `0.5.1` | Not specified | `0.7.1` | **Deviation — 0.5.1 not offered by this mirror's version list; 0.7.1 chosen because it is the newest version that still retains the `torchdata.datapipes` module DGL's `graphbolt` submodule imports** |
| NumPy | Windows conda build, exact version unpinned (`file:///C:/ci/numpy_and_numpy_base_1653574840943/work`) | Not specified | `1.26.4` | Deviation — downgraded from a pre-installed `2.4.4` because `torch==2.2.1`'s compiled extensions were built against the NumPy 1.x ABI |
| SciPy | `1.7.3` | Not specified | `1.17.1` | Deviation — never downgraded; not observed to block any import in this verification |
| pandas | `1.3.5` | Not specified | `3.0.2` (pre-installed, untouched) | Deviation — not observed to block any import in this verification (MUAD's own code does not import pandas directly in any file inspected in Step 2) |
| pydantic | Not listed in `requirements.txt` at all | Not specified | `2.13.5` (installed during this verification) | **Added — required transitively by installed `dgl==2.1.0`'s `graphbolt` submodule (`ondisk_metadata.py`), not by MUAD's own code** |
| tick, drain3, Eadro-side deps | Listed (`tick==0.7.0.1`, `drain3==0.9.11`, etc.) | Not specified | Not installed | Not needed — these are Eadro-side preprocessing dependencies (per `DATASET_FORENSICS_COMPLETE.md`), not imported anywhere in MUAD's own `main.py`/`models/`/`training/`/`active_learning/`/`data/`/`utils/` (confirmed by the complete import list traced in Phase B) |

**Key finding: `requirements.txt` itself is historically inconsistent with the commit that produced the recovered logs.** At commit `6031ccf` (source of the recovered `training.log`/`scores.txt`/chunk artifacts), `requirements.txt` contained only the single comment line `# Required packages` — **no dependency versions were specified at all at that point in history.** The detailed, version-pinned `requirements.txt` seen at current HEAD (`58923ca`), including the Windows-conda-specific file paths and the large set of unrelated packages (`spherogram`, `knot-floer-homology`, `zhusuan`, `tensorkit`, etc.), was added only in the final "Refactor" commit — VERIFIED via `git diff 6031ccf 58923ca -- requirements.txt`. This means the pinned versions in the current `requirements.txt` cannot be assumed to be the exact environment that produced the recovered logs; that environment's exact dependency versions are **UNKNOWN**.

---

## Phase B — Installation

All commands below were run against this container's PyPI mirror. No MUAD source file was edited at any point. Order matters (later steps were needed to fix failures surfaced by earlier steps):

```
1. pip install torch --break-system-packages     (initial state: torch 2.14.0, pre-installed)
2. pip install dgl --break-system-packages       (initial attempt: dgl 2.1.0)
   -> import dgl FAILS: ModuleNotFoundError: No module named 'torchdata.datapipes'
3. pip install torchdata --break-system-packages (installed: torchdata 0.11.0)
   -> import dgl STILL FAILS: same error (0.11.0 no longer ships `datapipes`)
4. pip install "torchdata==0.7.1" --break-system-packages
   -> torchdata.datapipes now importable
   -> import dgl FAILS differently: ModuleNotFoundError: No module named 'pydantic'
5. pip install pydantic --break-system-packages
   -> import dgl FAILS differently again:
      FileNotFoundError: Cannot find DGL C++ graphbolt library at
      .../dgl/graphbolt/libgraphbolt_pytorch_2.14.0.so
      (dgl==2.1.0 ships precompiled graphbolt .so files only for specific torch versions;
       inspected directory listing showed .so files for torch 2.0.0, 2.0.1, 2.1.0, 2.1.1, 2.1.2, 2.2.0, 2.2.1 only)
6. pip uninstall torch + all nvidia-*/triton CUDA packages --break-system-packages
   (freed disk space; this container's disk was nearly full and blocked further installs)
7. pip install "torch==2.2.1" --break-system-packages
   -> import torch FAILS: UserWarning "Failed to initialize NumPy: _ARRAY_API not found"
      (torch==2.2.1's compiled C extensions were built against NumPy 1.x ABI;
       container had numpy 2.4.4 pre-installed)
8. pip install "numpy<2" --break-system-packages  (installed: numpy 1.26.4)
   -> import torch: OK
   -> import dgl: OK
   -> from dgl.nn.pytorch import GATv2Conv, GlobalAttentionPooling: OK
```

**Minimum environment change required to import the unmodified MUAD source successfully, relative to the container's original state:** downgrade `torch` from `2.14.0` to `2.2.1` (the newest version for which `dgl==2.1.0` ships a precompiled `graphbolt` binary), downgrade `numpy` to `<2` (ABI compatibility with `torch==2.2.1`), pin `torchdata==0.7.1` (retains the `datapipes` submodule that this `dgl` version's `graphbolt` imports), and add `pydantic` (a transitive dependency of `dgl`'s `graphbolt`, absent from MUAD's own `requirements.txt` entirely). **No MUAD source file was modified to achieve this.**

**No dependency conflict was found that required modifying MUAD's own code to resolve** — every failure encountered in Phase B was resolved purely through package version changes external to the repository.

---

## Phase C — Data Loading Smoke Test

Using the official, unmodified `data/utils.py::load_data` and `data/dataset.py::ChunkDataset`/`create_dataloader`, with the recovered Dataset C artifact placed unmodified at `Dataset/chunk/{chunk_train.pkl,chunk_test.pkl,metadata.json}` (MD5 checksums verified identical to the files documented in `DATASET_FORENSICS_COMPLETE.md` §12 before any test was run):

```
train_data, node_num, edges = load_data("Dataset/chunk/chunk_train.pkl", "Dataset/chunk/metadata.json")
-> train_data chunks: 3366
-> test_data chunks: 2244
-> node_num: 12
-> edges: ([1,1,1,1,1,1,1,1,10,10,10,2,0,0,7,7,7,5,3,11,11,11,11,11],
           [1,10,6,2,7,8,5,3,10,2,0,2,0,5,7,4,9,5,3,1,10,11,0,5])
-> edges identical whether loaded via chunk_train.pkl+metadata.json or chunk_test.pkl+metadata.json: CONFIRMED

Sample chunk (key "MDgXKHl5"):
  traces:  shape=(12, 10, 1), dtype=float64
  metrics: shape=(12, 10, 7), dtype=float64
  logs:    shape=(12, 15), dtype=float64
  culprit: 1 (python int)
```

`ChunkDataset`/`create_dataloader` (batch_size=5 test):
```
Single graph: Graph(num_nodes=12, num_edges=24,
    ndata_schemes={'metrics': Scheme(shape=(10,7), dtype=torch.float32),
                    'traces':  Scheme(shape=(10,1), dtype=torch.float32),
                    'logs':    Scheme(shape=(15,), dtype=torch.float32)})
Single label: 1

Batched graph (batch_size=5): Graph(num_nodes=60, num_edges=120, ...)
Batched labels: tensor([1, 1, 4, 0, 5]), shape [5]
batch_graph.ndata["metrics"].shape: [60, 10, 7]
batch_graph.ndata["traces"].shape:  [60, 10, 1]
batch_graph.ndata["logs"].shape:    [60, 15]
batch_graph.batch_size: 5
```

**Result: the recovered Dataset C artifact loads, constructs DGL graphs, and batches exactly as documented in `MUAD_MODEL_FORENSICS_COMPLETE.md` §4 and §19.** No discrepancy found between the predicted and actual shapes at this stage.

---

## Phase D — Model Construction

`BaseModel` was instantiated exactly as current-HEAD `main.py` does it (i.e., **without** an explicit `hidden_dim` kwarg, since current `main.py` does not pass one):

```python
model = BaseModel(
    device, lr=params["lr"], epochs=params["epochs"], patience=params["patience"],
    result_dir=params["result_dir"], hash_id="phase_d_test",
)
```

**Actual printed model structure (`model.model`, i.e., the real `MainModel` instance), relevant excerpt:**
```
MMClasifier: Sequential(
  (0): LinearLayer(
    (clf): Sequential(
      (0): Linear(in_features=64, out_features=2, bias=True)
    )
  )
)
```
Full encoder/uncertainty-block/TCP-layer structure matches `MUAD_MODEL_FORENSICS_COMPLETE.md` §6.1–§6.8 exactly (GRU(7,64), GRU(1,64), Linear(15,64), GATv2Conv with `fc_src`/`fc_dst` both `Linear(64,256)` — i.e. `4 heads × 64 = 256`, confirming `attn_head=4`/`out_feats=64` — `MaxPool1d(kernel_size=4)`, `GlobalAttentionPooling(gate_nn=Linear(64,1))`, three `UncertainBlock`s each with `Linear(64,128)→LayerNorm→Tanh`, `Linear(128,64)→Sigmoid` ×2, `SimpleAttention` ×2, six per-modality `LinearLayer`s, one `TCPConfidenceLayer(192,1)`).

**Confirmed empirically: `hidden_dim` actually received by `MainModel.__init__` under current-HEAD `main.py`'s exact call is `[64]`** (the class default), **not** `params.json`'s `[64,64]` — directly observed from the printed module structure, not inferred.

---

## Phase E — Single Forward Pass

Executed exactly `model(graph, labels)` (the identical call `training/base_model.py`'s `fit()`/`evaluate()` make), using the current-HEAD, as-constructed-in-Phase-D model and one real batch from the recovered Dataset C training data.

**Exact, unmodified, uncaught exception:**
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5x192 and 64x2)
```
**Exact location (from the real traceback, not reconstructed):**
```
File ".../muad_repo/models/main_model.py", line 107, in forward
    MMlogit = self.MMClasifier(feature)
File ".../torch/nn/modules/container.py", line 217, in forward
    input = module(input)
File ".../muad_repo/models/layers.py", line 40, in forward
    return self.clf(x)
File ".../torch/nn/modules/linear.py", line 116, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5x192 and 64x2)
```
**Actual constructor arguments in force at failure time:** `hidden_dim=[64]` (Phase D), `num_class=2`, `aloss=0.4`, `b_loss=0.01` (all class defaults, since `main.py` passes none of them).

**This is a real, directly-observed, unmodified-code failure — not inferred, not hypothesized.** `MUAD_RUNTIME_VERIFICATION` treats this as the primary empirical result for "does current-HEAD MUAD run as released against Dataset C": **it does not.**

### E.1 — Root-cause investigation (git history, not source modification)

Since Step 2's static analysis flagged a direct contradiction with the recovered `training.log`'s apparent success, git history of `main.py` and `configs/params.json` was inspected (read-only, `git show`/`git diff`, no working-tree edits) to explain the contradiction:

- **Commit `5c8dc79` ("Initial commit")** and **`5b011bd` ("Update README.md and code comments...")**: `main.py`'s `BaseModel(...)` call **explicitly includes `hidden_dim = [64, 64]`** as a keyword argument. VERIFIED by `git show <commit>:main.py`.
- **Commit `6031ccf`** (the exact commit whose `Dataset/chunk/*.pkl` and `result/training.log`/`scores.txt` were recovered and used throughout this project) — its `main.py` **still contains `hidden_dim = [64, 64]`** in the `BaseModel(...)` call. VERIFIED directly.
- **Commit `58923ca` ("Refactor code structure for improved readability and maintainability")** — the same commit already known (`DATASET_FORENSICS_COMPLETE.md` §12) to have deleted the `Dataset/chunk/*` and `result/*` artifacts — **restructured `main.py` into the `full_label`/`active_learning` branches and, in doing so, silently dropped the `hidden_dim = [64, 64]` keyword argument from the `BaseModel(...)` call**, while leaving `hidden_dim: [64,64]` untouched inside `configs/params.json`. VERIFIED via `git diff 6031ccf 58923ca -- main.py`, which shows the `hidden_dim = [64, 64]` line present on the "before" side and absent on the "after" side.

**Empirical confirmation, by direct execution (using only unmodified MUAD classes, explicitly passing the historically-attested `hidden_dim=[64,64]` kwarg — i.e., reproducing what commit `6031ccf`'s `main.py` literally did, not a source-code fix):**
```python
model = BaseModel(device, lr=..., epochs=..., patience=..., result_dir=..., hash_id=...,
                   hidden_dim=[64, 64])   # <- exactly what commit 6031ccf's main.py passed
print(model.model.MMClasifier)
```
```
MMClasifier: Sequential(
  (0): LinearLayer(Linear(in_features=192, out_features=64, bias=True))
  (1): ReLU()
  (2): LinearLayer(Linear(in_features=64, out_features=2, bias=True))
)
```
Running the identical forward pass with this configuration **succeeded**:
```
MMlogit: shape=[5,2]
loss: shape=[] (scalar)
y_pred: [1, 1, 0, 0, 0]
feture: shape=[5,192]
TCPConfidence_sig: shape=[5,1]
```

**Conclusion, stated precisely:** the shape-mismatch failure is real and reproducible on **current HEAD** (`58923ca`), caused by a **silent parameter-passing regression introduced in that exact commit**, which also — per `DATASET_FORENSICS_COMPLETE.md` — is the same commit that deleted the recovered chunk/log artifacts. The commit that produced the recovered, apparently-successful `training.log` (`6031ccf`) did **not** have this bug, because it still passed `hidden_dim=[64,64]` explicitly. **Both facts are now fully reconciled with direct evidence; no contradiction remains unresolved.**

---

## Phase F — Single Training Batch

Performed using the **historically-attested configuration** (`hidden_dim=[64,64]`, matching commit `6031ccf`), since this is the only configuration under which a forward pass (and therefore `loss.backward()`) is possible at all. This is explicitly a separate, labeled track from "current HEAD as released" — see Phase K's classification.

```python
model.train()
optimizer.zero_grad()
res = model(graph, labels)
loss = res["loss"]
loss.backward()
optimizer.step()
```
**Observed:**
- `loss.item()` = `2.002124786376953`, `torch.isfinite(loss)` = `True`.
- At least one parameter's gradient is non-`None` and has nonzero absolute sum — confirmed directly (`any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters())` → `True`).
- `optimizer.step()` completed without error.

**Result: training batch succeeds, under the historically-attested `hidden_dim=[64,64]` configuration.** Not tested under current-HEAD's default configuration, since Phase E already shows the forward pass itself fails there.

---

## Phase G — One Evaluation Batch (Stochasticity Test)

Using `BaseModel.evaluate()` exactly as released, historically-attested configuration, same batch run twice:

```
Evaluation run 1: {'F1': 0.6923076923076923, 'Rec': 0.5625, 'Pre': 0.9}
Evaluation run 2: {'F1': 0.6428571428571429, 'Rec': 0.5625, 'Pre': 0.75}
Identical results across two calls: False
```

Direct logit/prediction comparison on the identical batch, `model.eval()`, `torch.no_grad()`:
```
MMlogit run1 (first 3 rows): [[ 0.0546,-0.0037],[-0.0747, 0.0947],[ 0.0227,-0.0196]]
MMlogit run2 (first 3 rows): [[-0.0205,-0.0228],[ 0.0256, 0.0085],[ 0.0185, 0.0135]]
y_pred run1: [0,1,0,1,1,1,1,1,1,1,0,1,0,1,0,1,1,1,1,0]
y_pred run2: [0,0,0,1,1,1,1,1,0,0,1,1,1,1,0,1,1,0,1,1]
Logits identical (torch.allclose): False
```

**Result: evaluation is empirically, directly confirmed stochastic.** Repeated calls to `evaluate()` on the exact same batch, exact same model weights, produce different logits, different predictions, and different F1/Precision/Recall — confirming `MUAD_MODEL_FORENSICS_COMPLETE.md` §7's static claim ("no `if self.training` branch exists anywhere in `UncertainBlock`") with direct runtime evidence, not just code inspection.

---

## Phase H — Checkpoint Behavior

A minimal real `fit()` run (12 epochs, `evaluation_epoch=3`, historically-attested `hidden_dim=[64,64]`, small train/test subsets for speed) was executed via the unmodified `BaseModel.fit()`:

```
Epoch 1/12, Loss: 1.91011
Epoch 2/12, Loss: 1.86163
Epoch 3/12, Loss: 1.82992
Test -- F1: 0.8571, Rec: 1.0000, Pre: 0.7500
Epoch 4/12, Loss: 1.74155
Epoch 5/12, Loss: 1.74885
Epoch 6/12, Loss: 1.68246
Test -- F1: 0.8571, Rec: 1.0000, Pre: 0.7500
Epoch 7/12, Loss: 1.64383
Epoch 8/12, Loss: 1.58561
Epoch 9/12, Loss: 1.52966
Test -- F1: 0.8571, Rec: 1.0000, Pre: 0.7500
Epoch 10/12, Loss: 1.53279
Epoch 11/12, Loss: 1.52347
Epoch 12/12, Loss: 1.48648
Test -- F1: 0.8571, Rec: 1.0000, Pre: 0.7500
Best F1 0.8571 at epoch 3

best_f1: 0.8571428571428571   best_epoch: 3
Final evaluate() after fit(): {'F1': 0.8571428571428571, 'Rec': 1.0, 'Pre': 0.75}
```

**Determined experimentally, exactly as the unmodified code executes:**
- Test evaluation occurs exactly every `evaluation_epoch` epochs (here, epochs 3, 6, 9, 12) — confirmed by the interleaved `"Test -- F1:..."` log lines appearing precisely at those epoch boundaries.
- `best_state` is selected by the **first** epoch whose `test_results["F1"]` exceeds the running best — in this specific run, epoch 3's F1 (0.8571) was never exceeded by later evaluations (which, due to the stochastic `UncertainBlock`, also happened to land on 0.8571 exactly in this run rather than higher — an artifact of the specific random draws in this short run, not a general guarantee), so `best_epoch` stayed at 3 through to the end.
- `test_results["F1"] > best_f1` (strict `>`) is confirmed as the literal comparison used — a tie does not update `best_state`, directly observed here (epochs 6, 9, 12 all produced F1=0.8571, equal to but not greater than the existing best, and none of them changed `best_epoch` from 3).
- `best_state` loads successfully: the final `model.evaluate(test_loader)` call after `fit()` returns reproduces `best_f1` exactly (`0.8571428571428571` both times), confirming `load_state_dict(best_state)` inside `fit()` took effect.
- Final evaluation runs and is reported correctly.

---

## Phase I — Active Learning Smoke Test

Using the unmodified `active_learning/strategies.py` functions and a minimal subset of the recovered training data (300 chunks), following `main.py`'s `active_learning` branch logic exactly, with the historically-attested `hidden_dim=[64,64]` (required, per Phase E, for any forward pass to succeed at all).

**Initial split:** `split_data(train_data_subset, train_ratio=0.1)` → 30 train keys, 270 val keys — confirmed directly, matching the documented 10% rule.

**Entropy-based selection:** `entropy_based_selection(model, val_loader, device, n_samples=5)` scored all 270 pool samples (confirmed: returned list length 270, one entropy value per pool sample, computed via `n_samples=5` repeated stochastic forward passes as coded — function signature default, unmodified).

**Confidence-based selection:** at `confidence_threshold=0.9` (the unmodified default), `confidence_based_selection` returned **0** high-confidence samples on this small, minimally-trained smoke-test model — an expected outcome for a barely-trained model, not itself evidence of a bug.

**The core non-propagation finding — isolated and directly confirmed:**
```
train_loader.dataset id BEFORE selection/combination logic: 139698065374816
train_loader.dataset id AFTER  selection/combination logic: 139698065374816   (IDENTICAL — same Python object)
train_loader.dataset length BEFORE: 30
train_loader.dataset length AFTER:  30                                        (UNCHANGED)
A correctly-constructed CombinedDataset (mirroring exactly what main.py builds) has length: 40
train_keys (local variable) grew to: 40
```
**Directly confirmed: `train_loader` continues to iterate over the original 30-sample dataset object even after `train_keys` grows to 40 and a properly-combined 40-sample `CombinedDataset` object exists in memory.** This is not an inference from reading the code — it was measured by comparing Python object identities (`id()`) and dataset lengths before and after running the exact sequence of operations `main.py`'s `active_learning` branch performs. This directly confirms `MUAD_MODEL_FORENSICS_COMPLETE.md` §14's static finding with runtime evidence.

**MC/entropy forward passes:** confirmed as 5 per batch (function default `n_samples=5`, unmodified). **`n_select`:** confirmed as the value passed by the caller (`main.py` passes `200`; this smoke test used `10` for speed, and the function respected that parameter correctly — top-10 of the sorted-descending entropy list). **Confidence threshold:** confirmed as `0.9` (unmodified default). **Pseudo-label generation:** confirmed to draw from `MMlogit.argmax(dim=1)` at the high-confidence indices (no pseudo-labels were generated in this specific smoke-test run since 0 samples cleared the 0.9 threshold, but the code path itself was exercised and returned an empty list correctly rather than erroring).

---

## Phase J — 50 vs 100 Epoch Discrepancy

**Fully resolved via git history, no invented explanation needed.**

```
Current HEAD (58923ca) configs/params.json:  "epochs": 100, "max_iter": 1,  (no "loss1"/"loss2"/"trainType" at 6031ccf)
Commit 6031ccf         configs/params.json:  "epochs": 50,  "max_iter": 30
```
**VERIFIED via `git show 6031ccf:configs/params.json`:** the commit that produced the recovered `training.log` (which shows `Epoch X/50` throughout) used a config file with **`"epochs": 50`** — an exact match, no discrepancy at all once the correct historical config is consulted. The apparent "50 vs 100" contradiction was an artifact of comparing the recovered log against the *current* `params.json` rather than the config that was actually in effect when that log was produced.

**Exact diff, `git diff 6031ccf 58923ca -- configs/params.json`:**
```diff
-    "max_iter": 30,
+    "max_iter": 1,
-    "epochs": 50,
-     "hidden_dim": [64, 64]
+    "epochs": 100,
+     "hidden_dim": [64, 64],
+    "trainType": "full_label",
+    "loss1": [0.001,0.01,0.1,1],
+    "loss2": [0.4,0.6,0.8,1]
```
**This single commit (`58923ca`) is now established as the source of every major discrepancy found across this investigation:** the epochs change (50→100), the `max_iter` change (30→1, meaning the originally-intended ~30-iteration active-learning loop was reduced to a single iteration), the introduction of the `full_label`/`active_learning` branching alongside the silent loss of the `hidden_dim` kwarg wiring (Phase E), the addition of never-consumed `trainType`/`loss1`/`loss2`-adjacent dead config surface, and (per `DATASET_FORENSICS_COMPLETE.md` §12) the deletion of the `Dataset/chunk/*` and `result/*` artifacts from the repository.

**Not resolved by this phase:** what the exact dependency versions were in the environment that produced the `6031ccf` logs (Phase A), and whether `max_iter=30` was ever actually exercised for a full 30-iteration run anywhere (no such log was recovered — the recovered `training.log` shows only `"Starting active learning iteration 1"` messages, never iteration 2 or beyond, across all the runs it contains) — this remains **UNKNOWN**.

---

## Phase K — Runtime Conclusion

| Question | Runtime Result | Status | Evidence |
|---|---|---|---|
| 1. Can the official repository import successfully? | Yes, all of `main.py`'s imports succeed | VERIFIED | Phase B/C direct execution |
| 2. Can Dataset C load? | Yes, exactly as documented in Step 1/2 | VERIFIED | Phase C |
| 3. Can DGL graphs be constructed? | Yes — 12 nodes, 24 edges per graph; batches correctly | VERIFIED | Phase C |
| 4. Can `MainModel` instantiate? | Yes, under both current-HEAD's implicit default and the historically-attested explicit `hidden_dim=[64,64]` | VERIFIED | Phase D |
| 5. What `hidden_dim` actually executes (current HEAD)? | `[64]` (class default; `main.py` passes no such kwarg) | VERIFIED | Phase D, direct module printout |
| 6. What is the fused feature dimension? | `192` (`3×64`, `torch.cat` of trace/metric/log) | VERIFIED | Phase E traceback (`5x192`), Phase F/G tensor shapes |
| 7. What does `MMClasifier` actually expect (current HEAD)? | First layer: `Linear(in_features=64, out_features=2)` — a **mismatch** against the 192-dim feature | VERIFIED | Phase D printout, Phase E traceback |
| 8. Does the forward pass succeed (current HEAD, default config)? | **No — `RuntimeError: mat1 and mat2 shapes cannot be multiplied (5x192 and 64x2)`** | VERIFIED (failure) | Phase E, exact traceback captured |
| 8b. Does the forward pass succeed (historically-attested `hidden_dim=[64,64]`)? | Yes | VERIFIED | Phase E.1, Phase F |
| 9. Does backward succeed? | Yes, under the historically-attested configuration (not testable under current-HEAD default, since forward already fails) | VERIFIED (conditional) | Phase F |
| 10. Does evaluation succeed? | Yes, under the historically-attested configuration | VERIFIED (conditional) | Phase G, H |
| 11. Is evaluation stochastic? | Yes — confirmed by two `evaluate()` calls on the identical batch producing different F1/Rec/Pre and different predictions | VERIFIED | Phase G |
| 12. Does test-set checkpoint selection execute? | Yes — `best_state` is updated exactly when `test_results["F1"] > best_f1` at each `evaluation_epoch`-th epoch, confirmed by direct trace of a real `fit()` run | VERIFIED | Phase H |
| 13. Does active-learning data actually propagate to the next training round? | **No — `train_loader`'s underlying dataset object and length are unchanged after the selection/combination logic runs, despite `train_keys` growing and a correctly-combined larger dataset existing in memory** | VERIFIED (non-propagation confirmed) | Phase I, `id()`/length comparison |
| 14. Can the 50 vs 100 epoch discrepancy be explained? | **Yes, fully — commit `6031ccf`'s `params.json` has `epochs:50`, matching the recovered log exactly; current HEAD's `params.json` was changed to `epochs:100` in the later `58923ca` refactor** | VERIFIED | Phase J, `git show`/`git diff` |

---

## 1. Environment Specification

```text
Python:      3.12.3
PyTorch:     2.2.1+cu121   (repo requires 1.13.1; not installable in this container — closest available version compatible with dgl==2.1.0's precompiled graphbolt binaries)
DGL:         2.1.0         (repo requires 2.0.0; not available in this container's mirror — 2.1.0 is the closest available)
torchdata:   0.7.1         (repo requires 0.5.1; not available in this container's mirror — 0.7.1 chosen as the newest version retaining the `datapipes` submodule dgl's graphbolt needs)
NumPy:       1.26.4        (downgraded from a pre-installed 2.4.4 for torch==2.2.1 ABI compatibility; repo's own pin is an unpinned Windows-conda build)
SciPy:       1.17.1        (repo requires 1.7.3; left at container default, not observed to block anything)
pydantic:    2.13.5        (not in repo's requirements.txt at all; required transitively by dgl==2.1.0's graphbolt submodule)
Operating system: Linux (container); original repo environment inferred to be Windows, based on requirements.txt's C:\ file paths and pywin32 pin
```

## 2. Exact Runtime Model Structure

See Phase D for the full printed `MainModel` (current-HEAD-default configuration) — reproduced there verbatim from a real `print(model.model)` call, not reconstructed.

## 3. Exact Runtime Shape Ledger (observed, not inferred)

| Stage | Tensor | Observed Shape |
|---|---|---|
| Single graph | `ndata["metrics"]` | `[12,10,7]` |
| Single graph | `ndata["traces"]` | `[12,10,1]` |
| Single graph | `ndata["logs"]` | `[12,15]` |
| Batch (B=5) | batched graph | `num_nodes=60, num_edges=120` |
| Batch (B=5) | `labels` | `[5]` |
| Forward (historically-attested config) | `feature` (post-fusion) | `[5,192]` |
| Forward (historically-attested config) | `MMlogit` | `[5,2]` |
| Forward (historically-attested config) | `TCPConfidence_sig` | `[5,1]` |
| Forward (current-HEAD default config) | attempted `MMClasifier` input | `[5,192]` fed into a layer expecting `[*,64]` → **RuntimeError, no output tensor produced** |

## 4. Exceptions / Failures

**Exact exception (current-HEAD default configuration, Phase E):**
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5x192 and 64x2)
```
at `models/main_model.py:107` (`MMlogit = self.MMClasifier(feature)`), propagating from `models/layers.py:40` (`LinearLayer.forward`), propagating from PyTorch's `torch/nn/modules/linear.py:116` (`F.linear`).

No other uncaught exception occurred anywhere else in Phases C, D, F, G, H, I under their respective tested configurations.

## 5. Resolved Contradictions

- **`MUAD_MODEL_FORENSICS_COMPLETE.md` §10's central open question is now fully resolved:** the static shape-mismatch analysis was correct for current HEAD; the recovered logs' apparent success is explained by a **git-history-confirmed silent regression** (loss of the `hidden_dim=[64,64]` kwarg) introduced in commit `58923ca`, which post-dates the commit (`6031ccf`) that produced those logs.
- **The 50-vs-100-epochs discrepancy (§15/§24 of the model forensics doc) is fully resolved:** commit `6031ccf`'s own `params.json` specifies `epochs:50`, exactly matching the recovered log; current HEAD's `params.json` was changed to `100` in the same later refactor commit.
- **The AL non-propagation hypothesis (§14 of the model forensics doc, previously a static-code-tracing claim) is now confirmed by direct runtime measurement**, not just code reading.
- **The evaluation-stochasticity claim (§7) is now confirmed by direct runtime measurement** (two different F1 scores from one fixed batch and one fixed set of weights).
- **The checkpoint-selection mechanism (§12/§17) is now confirmed by direct runtime trace** of a real `fit()` call, including the strict-`>` tie-breaking behavior.

## 6. Remaining Contradictions / Unknowns

- The exact dependency versions of the environment that originally produced the `6031ccf` logs remain **UNKNOWN** (that commit's `requirements.txt` was empty of version pins).
- Whether `max_iter=30` (the value in effect at commit `6031ccf`) was ever actually exercised for more than 1 logged iteration anywhere is **UNKNOWN** — no recovered log shows a second active-learning iteration.
- Whether the paper's published Dataset-C numbers (F1=0.9928) were produced under the historically-attested `hidden_dim=[64,64]`/`epochs=50`/`max_iter=30` configuration, the current-HEAD (broken) configuration, or some other configuration entirely, remains **UNKNOWN** — this runtime verification did not have access to whatever exact environment/config produced the paper's own numbers, only to the recovered git-history artifacts.
- Whether the `confidence_based_selection` global-index-arithmetic edge case (flagged in the model forensics doc §14) actually misbehaves on a non-round pool size was not triggered in this smoke test (pool size 270, batch size 50 → last batch of 20; this specific run produced 0 high-confidence selections, so the edge case's consequences were not observable here either way) — remains **UNKNOWN**.

## 7. Reproduction Decision

**Classification: (B) "Official code runs only with an environment compatibility change" — for the historically-attested configuration (`hidden_dim=[64,64]`, matching commit `6031ccf`).**

**Classification: (C) "Official source cannot run as released because of a source-level contradiction" — for current HEAD (`58923ca`) specifically**, independent of any environment/dependency-version change: the shape-mismatch failure in Phase E occurs purely from `main.py`'s own Python-level argument-passing logic (a missing keyword argument), not from any DGL/PyTorch/environment incompatibility. No environment change of any kind would resolve it, since the bug is internal to the repository's own current-HEAD source code.

Both classifications are recorded because they apply to two different, precisely-identified states of the same repository (current HEAD vs. the historically-attested commit `6031ccf` configuration), and the task's evidence supports different answers for each. Neither classification is chosen without the direct execution evidence documented in Phases B and E above.
