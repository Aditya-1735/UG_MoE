# MUAD — Complete Model, Mathematics, Training, and Implementation Forensics

## 1. Scope and Baseline

- **Step 1 is frozen.** `DATASET_FORENSICS_COMPLETE.md` is the authoritative baseline for all Dataset C / Eadro SocialNetwork facts (raw archive structure, fault ground truth, tensor shapes `metrics(12,10,7)`, `traces(12,10,1)`, `logs(12,15)`, culprit semantics, split 60/40 = 3366/2244, `metadata.json`). This document does not re-derive those facts; it cites them where relevant.
- **This document covers:** the MUAD model architecture, mathematics, loss functions, training loop, active-learning algorithm, evaluation protocol, and all implementation-level behavior of the official `github.com/slg-frank/muad` repository, as it pertains to Dataset C.
- **Implementation is intentionally NOT performed in this document.** No new code is written here beyond small isolated PyTorch snippets used strictly to empirically test shape/behavior claims about the *existing* MUAD code (documented in §6.4/§10 as such).
- Dataset A/B facts are not used anywhere in this document except where the MUAD paper itself explicitly distinguishes them.

---

## 2. MUAD Source Inventory

| Source | Path / Identifier | What It Establishes | Status |
|---|---|---|---|
| MUAD paper | IEEE TSC Vol.19 No.3, 2026, DOI 10.1109/TSC.2026.3672587 | Intended method (GPE, CFM, active learning, loss terms, hyperparameters as documented intent) | Read in full |
| Official MUAD repository (current HEAD) | `github.com/slg-frank/muad`, commit `58923ca` (latest) | Current-state code for `main.py`, `data/`, `models/`, `training/`, `active_learning/`, `utils/`, `configs/params.json` | Fully read, file-by-file |
| Deleted-but-recovered commit | `6031ccf` (pre-refactor, parent of `58923ca`) | Confirms `models/main_model.py`, `training/base_model.py` logic identical to current HEAD at the time the recovered training/scores logs were generated; also the source of the recovered `Dataset/chunk/*` and `result/*` artifacts (see Step 1 doc §12) | Read via `git show 6031ccf:<path>` |
| `training/trainer.py` | `training/trainer.py` | Confirmed to be a genuinely empty (0-byte) file at commit `6031ccf` and current HEAD — not a second training entrypoint | Verified via `git cat-file -p`, blob hash matches git's empty-blob hash |
| `configs/params.json` | `configs/params.json` | All configuration keys and their values (see §15) | Read directly |
| Recovered `result/training.log`, `result/scores.txt` | From commit `6031ccf` | Empirical evidence of actual runtime behavior (epoch counts, no crash observed) — used in §10 to cross-check a code-derived shape-mismatch hypothesis | Read directly (already used in Step 1 doc) |
| Small isolated PyTorch scripts (this investigation's own) | Not part of MUAD | Used only to empirically verify shape-transform claims (`GATv2Conv`→maxpool→`GlobalAttentionPooling` shape trace; `MMClasifier` input/output shape test) | Explicitly marked RECONSTRUCTED / this-investigation-only in §6.6 and §10; never presented as MUAD code |

No unofficial reimplementations of MUAD were consulted. No general knowledge about GATv2, GPE/CFM-style architectures, or active learning was used to fill any gap — every architectural claim below traces to a specific line of the official repository or a specific sentence of the paper, with the two kept explicitly distinguished throughout.

---

## 3. End-to-End MUAD Pipeline

```
Dataset C chunks (chunk_train.pkl / chunk_test.pkl, per Step 1 §12)
   │  keys: "metrics"(12,10,7), "traces"(12,10,1), "logs"(12,15), "culprit"(int)
   ▼
data/utils.py::load_data()                         — pickle.load + metadata.json read
   ▼
data/dataset.py::ChunkDataset.__init__()            — EAGERLY builds one dgl.graph per chunk,
                                                        attaches ndata["metrics"/"traces"/"logs"]
   ▼
data/utils.py::create_dataloader()                   — torch DataLoader, batch_size=50, shuffle=True(train)/False(test/val),
                                                        collate_fn=collate → dgl.batch(graphs), torch.tensor(labels)
   ▼
models/main_model.py::MainModel.forward(graph, fault_indexs)
   │
   ├─ models/encoders.py::MetricEncoder  → models/layers.py::GRUEncoder(in=7)  → models/graph_model.py::GraphModel1  → [B,64]
   ├─ models/encoders.py::TraceEncoder   → models/layers.py::GRUEncoder(in=1)  → models/graph_model.py::GraphModel1  → [B,64]
   ├─ models/encoders.py::LogEncoder     → nn.Linear(15,64)                    → models/graph_model.py::GraphModel1  → [B,64]
   ▼
models/main_model.py::UncertainBlock ×3 (one per modality)                     — μ,σ² (Sigmoid-bounded), reparameterization → [B,64] each
   ▼
per-modality TCP classifier + confidence layers (models/layers.py::LinearLayer)  — per-modality CE + MSE confidence loss
   ▼
Confidence-weighted concatenation: cat(trace*σ(conf), metric*σ(conf), log*σ(conf)) → [B,192]
   ▼
models/main_model.py::MMClasifier (nn.Sequential of LinearLayer)                — binary anomaly logits [B,2]
   ▼
y_anomaly = (fault_indexs >= 1).long()                                          — binary target, derived from Dataset C's `culprit`
   ▼
total_loss = MMLoss + 0.6*confidence_loss + mean_kl_loss                        — see §11
   ▼
training/base_model.py::BaseModel.fit()                                         — Adam, loss-plateau early stop, TEST-SET-driven checkpoint selection
   ▼
[if trainType=="active_learning"] active_learning/strategies.py::hybrid_selection — entropy-based + confidence-based querying (see §14)
   ▼
training/base_model.py::BaseModel.evaluate()                                     — binary TP/FP/FN/TN via label==0 vs label!=0, F1/Rec/Pre
```

**No explicit root-cause/culprit prediction head exists anywhere in the code.** The multi-class `culprit` value is consumed only to derive the binary `y_anomaly`; no layer of `MainModel` outputs a per-service culprit probability. This is a VERIFIED FROM CODE fact, contrasted with §16.

---

## 4. Input Specification

Per Step 1 §12 (VERIFIED), a single Dataset C chunk has:
- `metrics`: `(12, 10, 7)`
- `traces`: `(12, 10, 1)`
- `logs`: `(12, 15)`
- `culprit`: Python `int` (observed values in the recovered artifact: `{0,1,4,5}` only — see Step 1 §12)

**VERIFIED FROM CODE** (`data/dataset.py::ChunkDataset.__init__`): each chunk becomes one `dgl.graph(edges, num_nodes=12)` with:
```
graph.ndata["metrics"] = FloatTensor(12, 10, 7)
graph.ndata["traces"]  = FloatTensor(12, 10, 1)
graph.ndata["logs"]    = FloatTensor(12, 15)
```
and the chunk's `culprit` (or a supplied pseudo-label) is kept as the paired label, **not** stored as node/graph data.

**VERIFIED FROM CODE** (`data/utils.py::collate`): a batch of `B` such graphs is merged via `dgl.batch(graphs)` into one `DGLGraph` with `B×12` total nodes; `labels` becomes a `torch.tensor` of shape `[B]`.

```text
Per-graph node feature shapes (pre-batch), each node = one of 12 services:
  ndata["metrics"]: [12, 10, 7]
  ndata["traces"]:  [12, 10, 1]
  ndata["logs"]:    [12, 15]

After dgl.batch of B graphs:
  batched_graph.ndata["metrics"]: [B*12, 10, 7]
  batched_graph.ndata["traces"]:  [B*12, 10, 1]
  batched_graph.ndata["logs"]:    [B*12, 15]
  labels: [B]          (culprit ints, or fault_indexs used directly as-is)
```
Default `batch_size=50` (VERIFIED, `data/utils.py::create_dataloader`).

---

## 5. Graph Construction

**Graph type:** `dgl.graph(edges, num_nodes=node_num)` — **VERIFIED FROM CODE**, `data/dataset.py` line 14. `dgl.graph` from an edge-tuple `(src_tensor, dst_tensor)` constructs a **directed** graph by DGL's default semantics; no explicit `to_bidirected` or symmetrization call exists anywhere in the MUAD codebase.

**Node count:** `node_num=12` (from Step 1 §12's `metadata.json`, `node_num: 12`), read via `data/utils.py::load_data` and passed straight through to every `ChunkDataset` construction in `main.py`.

**Exact edge list (VERIFIED, from Step 1 §12's recovered `metadata.json`, reused here as the graph MUAD actually loads):**
```
edges = ([1,1,1,1,1,1,1,1,10,10,10,2,0,0,7,7,7,5,3,11,11,11,11,11],
          [1,10,6,2,7,8,5,3,10,2,0,2,0,5,7,4,9,5,3,1,10,11,0,5])
```
24 directed edges, `(src_list, dst_list)` tuple format — this is exactly the format `dgl.graph()` expects for edge construction from two parallel index lists.

**Self-loops:** the edge list above includes `(1→1)` and `(11→11)` among its 24 entries (row 0 has repeated `1`s for multiple targets including itself at position where src=1,dst=1) — **VERIFIED by direct inspection of the edge tuple**: `edges[0][0]=1, edges[1][0]=1` is a self-loop on node 1. No other explicit `dgl.add_self_loop()` call exists in the code. Whether every node has a self-loop is not established — only that at least node 1 and node 11 do, per the raw edge list; a full self-loop audit was not performed in this document.

**`allow_zero_in_degree=True`** is passed to every `GATv2Conv` layer (`models/graph_model.py`, VERIFIED) — this is DGL's required flag to permit nodes with no incoming edges (which would otherwise raise a DGL error), implying the authors expected/allowed some nodes to have zero in-degree under this fixed 12-node, 24-edge graph.

**Node features:** `metrics`, `traces`, `logs` (raw, pre-encoder) attached as `ndata` per §4; after encoding, per-modality embeddings (`[N,64]` pre-pooling, i.e. per-node) are the features that GAT layers actually operate on (see §6.6).

**Edge features:** **none.** No `edata` is ever set anywhere in `data/dataset.py` or elsewhere. `GATv2Conv` in this codebase operates on node features only, with attention weights learned purely from node-pair representations (standard GATv2 mechanism), not from any edge attribute.

**Graph batching:** `dgl.batch(graphs)` (VERIFIED, `data/utils.py::collate`) — DGL's standard mechanism that merges `B` graphs into one disjoint-union graph, preserving per-graph node/edge membership internally for later per-graph pooling operations (used by `GlobalAttentionPooling`, §6.6).

**Message passing direction:** determined by DGL/`GATv2Conv`'s default convention, which propagates information from `src` to `dst` along each `(src,dst)` edge as listed in the `edges` tuple above — i.e., strictly along the direction given by `(edges[0][i], edges[1][i])` for each `i`, since the graph is directed and no reverse edges or bidirection call is added.

**Relation to the Eadro graph:** per Step 1 §12/§6, this edge list and node count are the **same 12-node, 24-edge dependency graph produced by Eadro's own preprocessing** (`util.py`'s `edge_info`, materialized into the shared `metadata.json`). MUAD does not construct or modify this graph in any way — it is loaded verbatim from `metadata.json` (VERIFIED, `data/utils.py::load_data`).

**Graph normalization:** **none observed.** No adjacency normalization (e.g., symmetric degree normalization, as in GCN) is applied anywhere; `GATv2Conv`'s own internal attention-based weighting is the only form of edge/neighbor weighting present.

---

## 6. Complete Architecture

### 6.1 Metric Encoder
**File:** `models/encoders.py::MetricEncoder`. **VERIFIED FROM CODE.**
```python
self.metric_model = GRUEncoder(in_size=7, out_dim=64)
self.status_model = GraphModel1(in_dim=out_dim, device=device, **kwargs)   # out_dim default = 64

def forward(self, graph):
    metric_embedding = self.metric_model(graph.ndata["metrics"])   # [N,10,7] -> [N,64]
    return {'metric_embedding1': self.status_model(graph, metric_embedding)}  # -> [B,64] (graph-pooled)
```
- **Input dimension:** 7 (matches Step 1's `metric_num=7`).
- **GRU parameters:** `nn.GRU(input_size=7, hidden_size=64, batch_first=True, bidirectional=False)` — single layer (PyTorch default `num_layers=1`, not overridden).
- **Sequence dimension:** 10 (the `chunk_lenth`/window length from Step 1).
- **Output representation:** last hidden state (`hidden.squeeze(0)`), i.e. `[N, 64]` per node (N = total nodes in the batched graph).
- **Graph interaction:** the `[N,64]` per-node embedding is passed into `GraphModel1` (§6.6), which runs `GATv2Conv` + pooling and returns a **graph-level** `[B,64]` vector (one per sample in the batch, not per node).
- **Exact tensor shapes:** `[N,10,7] → GRU → [N,64] → GraphModel1 → [B,64]`.

### 6.2 Trace Encoder
**File:** `models/encoders.py::TraceEncoder`. **VERIFIED FROM CODE.**
```python
self.trace_model = GRUEncoder(in_size=1, out_dim=64)
self.status_model = GraphModel1(in_dim=out_dim, device=device, **kwargs)

def forward(self, graph):
    trace_embedding = self.trace_model(graph.ndata["traces"])   # [N,10,1] -> [N,64]
    return {"trace_embedding1": self.status_model(graph, trace_embedding)}  # -> [B,64]
```
- **Input dimension:** 1 — matches the recovered Dataset C chunk's `traces` shape `(12,10,1)` (Step 1 §9/§12) exactly, **not** Eadro's own 2-channel allocation. This is the single-channel-trimmed artifact Step 1 already flagged as a RECONSTRUCTED, undocumented compatibility step outside MUAD's own code.
- Same GRU/GraphModel1 structure as §6.1, only `in_size` differs.
- **Exact tensor shapes:** `[N,10,1] → GRU → [N,64] → GraphModel1 → [B,64]`.

### 6.3 Log Encoder
**File:** `models/encoders.py::LogEncoder`. **VERIFIED FROM CODE.**
```python
self.log_embedder = nn.Linear(15, out_dim)   # out_dim default = 64
self.status_model = GraphModel1(in_dim=out_dim, device=device, **kwargs)

def forward(self, graph):
    log_emb = self.log_embedder(graph.ndata["logs"])   # [N,15] -> [N,64]
    return {'log_embedding': self.status_model(graph, log_emb)}  # -> [B,64]
```
- **Input dimension:** 15 (matches Step 1's `event_num=15`).
- **No GRU** — logs have no time-window dimension in Dataset C's chunk format (`(12,15)`, not `(12,10,15)`), so a single `nn.Linear(15,64)` maps each node's 15-dim log-intensity vector directly to 64-dim. No activation function is applied within `LogEncoder` itself (the raw linear output goes straight into `GraphModel1`).
- **Exact tensor shapes:** `[N,15] → Linear → [N,64] → GraphModel1 → [B,64]`.

### 6.4 GPE (Graph-based Probabilistic Encoder)

**Paper's stated purpose (VERIFIED FROM PAPER):** model each modality's features as a Gaussian distribution `N(μ,σ²)` via GAT + MLP, to capture intra-modal uncertainty, with a KL term regularizing toward `N(0,I)`.

**Code implementation — the class actually named `UncertainBlock`, not `GPE`; the paper never gives it this exact identifier either, referring to it descriptively (VERIFIED FROM CODE, `models/main_model.py`):**
```python
class UncertainBlock(nn.Module):
    def __init__(self, in_dim=64, out_dim=128):
        self.encoder = nn.Sequential(nn.Linear(in_dim,128), nn.LayerNorm(128), nn.Tanh())
        self.fc_mu  = nn.Sequential(nn.Linear(128,64), nn.Sigmoid())
        self.fc_var = nn.Sequential(nn.Linear(128,64), nn.Sigmoid())
        self.attention_mu  = SimpleAttention(64)
        self.attention_var = SimpleAttention(64)

    def forward(self, x):                      # x: [B,64], graph-pooled modality embedding
        encoded = self.encoder(x)               # [B,64] -> [B,128], LayerNorm, Tanh
        mu  = self.fc_mu(encoded)                # [B,128] -> [B,64], Sigmoid  (bounded (0,1))
        var = self.fc_var(encoded)               # [B,128] -> [B,64], Sigmoid  (bounded (0,1))
        mu_attn  = self.attention_mu(mu)          # SimpleAttention: elementwise-gated by softmax(x·W+b)
        var_attn = self.attention_var(var)
        new_sample = self.reparametrize(mu_attn, var_attn)
        return new_sample, self.kl_loss(mu_attn, var_attn)

    def reparametrize(self, mu, var):
        std = var.sqrt()                          # treats `var` as VARIANCE (must be >=0; Sigmoid guarantees this)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_loss(self, mu, var):
        return -0.5 * torch.mean(torch.sum(1 + var - mu**2 - var.exp(), dim=-1))  # treats `var` as LOG-VARIANCE
```
- **Inputs:** the graph-pooled, per-graph `[B,64]` modality embedding from `GraphModel1` (i.e., operates on the *already graph-attended, already-pooled* representation — this is a **graph-level**, not node-level, uncertainty model, matching the paper's stated intent to model "graph-level features for each modality" rather than per-service embeddings).
- **Outputs:** `new_sample` (`[B,64]`, the reparameterized stochastic embedding actually used downstream) and a scalar `kl_loss` for that modality.
- **Learnable parameters:** `encoder` (Linear+LayerNorm), `fc_mu`, `fc_var` (each their own Linear+activation), and `SimpleAttention`'s `attention_weights`/`bias` (separate instances for μ and σ² — 4 learnable matrices total: `attention_mu.attention_weights [64,64]`, `attention_var.attention_weights [64,64]`, plus their biases).
- **Uncertainty representation:** `fc_mu`/`fc_var` outputs, both passed through **Sigmoid**, bounding both to `(0,1)`.
- **Graph operations:** none inside `UncertainBlock` itself — all GAT/graph-attention operations happen upstream in `GraphModel1` (§6.6); `UncertainBlock` operates purely on the already-pooled `[B,64]` vector.
- **PRECISELY DOCUMENTED, UNRESOLVED CODE-LEVEL INCONSISTENCY (VERIFIED FROM CODE, not inferred):** the identical tensor `var_attn` (output of `fc_var`→Sigmoid→`SimpleAttention`, bounded to `(0,1)`) is used two different ways in the same class:
  - In `reparametrize`, it is treated as **raw variance**: `std = var.sqrt()`.
  - In `kl_loss`, it is treated as **log-variance**: the formula `1 + var - mu**2 - var.exp()` is the standard closed-form KL divergence between `N(mu, exp(var))` and `N(0,I)` *only if* `var` represents `log(σ²)`.
  - These are mathematically incompatible interpretations of the same tensor. This is stated as an observed code fact; **no attempt is made here to determine which interpretation the authors intended, or to silently correct it.**

### 6.5 CFM (Confidence-aware Fusion Mechanism)

**Paper's stated purpose (VERIFIED FROM PAPER):** a modality-specific classifier produces a softmax output; the True Classification Probability (TCP) — the softmax probability at the ground-truth label — is used as a training target for a separate confidence-prediction network, whose sigmoid output becomes that modality's fusion weight.

**Code implementation (VERIFIED FROM CODE, `models/main_model.py::MainModel._compute_confidence_loss` and `forward`):**
```python
def _compute_confidence_loss(self, x, y, classifier, confidence_layer, criterion):
    logit = classifier(x)                               # [B,64] -> [B,2]  (LinearLayer, Xavier-init)
    prob = F.softmax(logit, dim=1)                       # [B,2]
    p_target = torch.gather(prob, 1, y.unsqueeze(1)).squeeze()   # TCP: prob at the true label y
    confidence = torch.sigmoid(confidence_layer(x)).squeeze()    # [B,64] -> [B,1] -> sigmoid -> [B]
    return F.mse_loss(confidence, p_target) + criterion(logit, y)   # MSE(confidence, TCP) + CrossEntropy(logit, y)
```
Called once per modality: `confidence_loss_trace/_metric/_log = _compute_confidence_loss(new_<modality>, y_anomaly, TCPClassifierLayer_<modality>, TCPConfidenceLayer_<modality>, criterion)`, each combining a per-modality classification loss with a per-modality confidence-regression loss against that modality's own TCP.

**Fusion weight application (VERIFIED FROM CODE):**
```python
TCPConfidence_trace_sig  = torch.sigmoid(self.TCPConfidenceLayer_trace(new_trace))    # [B,1]
TCPConfidence_metric_sig = torch.sigmoid(self.TCPConfidenceLayer_metric(new_metric))  # [B,1]
TCPConfidence_log_sig    = torch.sigmoid(self.TCPConfidenceLayer_log(new_log))        # [B,1]

feature = torch.cat((
    new_trace  * TCPConfidence_trace_sig,     # [B,64] * [B,1] broadcast -> [B,64]
    new_metric * TCPConfidence_metric_sig,
    new_log    * TCPConfidence_log_sig
), dim=-1)                                     # -> [B,192]
```
- **Equations, exactly as implemented:** `ω_v = sigmoid(Linear_v(z_v))` for `v ∈ {trace, metric, log}`; fused representation `h = concat(z_trace·ω_trace, z_metric·ω_metric, z_log·ω_log)`. This matches the paper's Eq. 7 (`ωv = Sigmoid(MLP(zv))`) and the general shape of Eq. "h = Σωv·zv" — **except the code concatenates the three weighted vectors rather than summing them**, which is a difference between the paper's stated `h = Σωvzv` (implying same-dimensional weighted sum) and the code's `torch.cat(...)` (producing a 3×-wider vector). **This is a PAPER VS CODE DISCREPANCY, stated exactly, not reconciled** — see §16.
- **Code parameterization:** per-modality `TCPClassifierLayer_<v>` (`LinearLayer(64,2)`) and `TCPConfidenceLayer_<v>` (`LinearLayer(64,1)`), both Xavier-normal initialized, zero bias (from `LinearLayer._xavier_init`).
- There is also a fourth, separate confidence layer, **`TCPConfidenceLayer = LinearLayer(192,1)`**, applied to the *final fused* 192-dim `feature` (not per-modality), producing `TCPConfidence_sig` in the returned dict — this is used only at inference/active-learning-query time (§14), **not** in the loss computation (`total_loss` never references `TCPConfidenceLayer`'s output).

### 6.6 GAT / Graph Layers

**File:** `models/graph_model.py::GraphModel1`. **VERIFIED FROM CODE.**
```python
class GraphModel1(nn.Module):
    def __init__(self, in_dim, graph_hiddens=[64], device='cpu', attn_head=4, activation=0.2, **kwargs):
        layers = []
        for i, hidden in enumerate(graph_hiddens):
            in_feats = graph_hiddens[i-1] if i>0 else in_dim
            dropout = kwargs.get("attn_drop", 0)
            layers.append(GATv2Conv(in_feats, out_feats=hidden, num_heads=attn_head,
                                     attn_drop=dropout, negative_slope=activation,
                                     allow_zero_in_degree=True))
        self.net = nn.Sequential(*layers)
        self.out_dim = graph_hiddens[-1]
        self.pooling = GlobalAttentionPooling(nn.Linear(self.out_dim, 1))
        self.maxpool = nn.MaxPool1d(attn_head)

    def forward(self, graph, x):
        out = x
        for layer in self.net:
            out = layer(graph, out)                                       # [N,in] -> [N, num_heads, out_feats]
            out = self.maxpool(out.permute(0,2,1)).permute(0,2,1).squeeze()  # -> [N, out_feats]  (max over heads)
        return self.pooling(graph, out)                                    # DGL GlobalAttentionPooling -> [B, out_feats]
```
- **Layer count:** `graph_hiddens=[64]` (default, and — per §10 — this default is what actually executes, since `params.json`'s `hidden_dim` is never wired to `GraphModel1`'s `graph_hiddens` argument at all; they are different parameters in different classes and neither config value reaches here). This means **exactly one `GATv2Conv` layer** runs.
- **Heads:** `attn_head=4` (default, unconfigured elsewhere).
- **Hidden dimension:** `out_feats=64` (from `graph_hiddens=[64]`).
- **Attention computation:** DGL's built-in `GATv2Conv` (dynamic/GATv2-style attention, as opposed to GATv1's static attention) — the exact attention-score formula is DGL library-internal code, not MUAD's own code, and is not reproduced here since it is external library behavior, not part of MUAD's own implementation to forensically audit.
- **Aggregation across heads:** **not** DGL's built-in head-averaging/head-concatenation — MUAD applies its **own custom max-pooling across the head dimension**: `nn.MaxPool1d(attn_head)` after a `permute(0,2,1)`, taking the elementwise-max feature value across the 4 attention heads (VERIFIED via the shape trace: `[N,4,64] → permute → [N,64,4] → MaxPool1d(kernel=4) → [N,64,1] → permute back → squeeze → [N,64]`, empirically confirmed with a synthetic tensor in this investigation).
- **Residuals:** none observed — no skip/residual connection is added anywhere around the `GATv2Conv` call or the maxpool step.
- **Activations:** `negative_slope=0.2` for `GATv2Conv`'s internal LeakyReLU (standard GAT attention nonlinearity, passed as the `activation` constructor arg, default `0.2`); no additional activation is applied to `GraphModel1`'s own output.
- **Dropout:** `attn_drop = kwargs.get("attn_drop", 0)` — **defaults to 0** (no attention dropout) since `attn_drop` is never passed as a kwarg anywhere in the call chain from `main.py`/`BaseModel`/`MainModel`/`MetricEncoder`/etc. (VERIFIED by tracing every `**kwargs` pass-through — no code sets `attn_drop`).
- **DGL APIs used:** `dgl.nn.pytorch.GATv2Conv`, `dgl.nn.pytorch.GlobalAttentionPooling`.
- **Graph-level pooling:** `GlobalAttentionPooling(nn.Linear(64,1))` — a learned per-node gate (`Linear(64→1)`, then implicitly softmax-normalized within-graph by DGL's internal implementation) used to compute a weighted sum of node features **per graph** in the batch, yielding `[B,64]`. This is a **graph-level readout**, producing one vector per sample, not one per service/node — confirmed by tracing the return shape and consistent with the paper's stated intent ("global attention pooling... to obtain the graph-level features").

### 6.7 Fusion
Already fully specified in §6.5: **concatenation**, not summation or attention-based mixing at the top level. Exact code: `torch.cat((new_trace*ω_trace, new_metric*ω_metric, new_log*ω_log), dim=-1)`. Order in the concatenation is **trace, metric, log** (VERIFIED from the exact argument order in the `torch.cat` call) — note this is **not** alphabetical and not the same order the encoders are constructed in `MainModel.__init__` (`metric_encoder`, `trace_encoder`, `log_encoder`) — a minor but exact, worth-recording ordering detail. Resulting dimension: `192 = 3×64`. Subsequent layer: `MMClasifier` (§6.8).

### 6.8 Classifiers / Prediction Heads

**VERIFIED FROM CODE, complete enumeration — every `nn.Linear`-based head in `MainModel`:**

| Head | Constructor | Input dim | Output dim | Purpose |
|---|---|---:|---:|---|
| `TCPClassifierLayer_trace` | `LinearLayer(hidden_dim[0], num_class)` | 64 | 2 | Per-modality (trace) binary classifier, used only to compute TCP and its own CE loss |
| `TCPConfidenceLayer_trace` | `LinearLayer(hidden_dim[0], 1)` | 64 | 1 | Per-modality (trace) confidence predictor |
| `TCPClassifierLayer_metric` | same | 64 | 2 | Per-modality (metric) binary classifier |
| `TCPConfidenceLayer_metric` | same | 64 | 1 | Per-modality (metric) confidence predictor |
| `TCPClassifierLayer_log` | same | 64 | 2 | Per-modality (log) binary classifier |
| `TCPConfidenceLayer_log` | same | 64 | 1 | Per-modality (log) confidence predictor |
| `TCPConfidenceLayer` | `LinearLayer(192, 1)` | 192 | 1 | Fused-feature confidence, used for active-learning querying only (§14), not in `total_loss` |
| `MMClasifier` | `nn.Sequential(...)`, see §10 for exact resulting structure given actual runtime `hidden_dim` | see §10 | 2 | Final fused binary anomaly classifier — this is the head whose output (`MMlogit`) determines `y_pred` |

- **Anomaly prediction:** `MMClasifier`'s output, `MMlogit`, `argmax`-ed and thresholded `>=1` in `inference()` (below).
- **Culprit/root-cause prediction:** **none** — no head exists anywhere that outputs a per-service probability distribution over the 12 nodes. The paper's Eq. 11 (`ŷ=argmax[Softmax(Wh+b)]`, `ŷ∈{0,1}`) matches this: MUAD, as released, is architecturally binary-only for Dataset C.
- **Per-modality predictions:** the three `TCPClassifierLayer_<v>` outputs exist and are trained (via `criterion(logit,y)` inside `_compute_confidence_loss`), but are **never read or used anywhere in `forward()`'s returned dict or in `total_loss` beyond that per-modality CE term** — i.e., their softmax probabilities/logits are not separately surfaced to the caller.
- **Fusion prediction:** `MMlogit` from `MMClasifier(feature)`.
- **Confidence calculation:** two distinct code paths — (a) per-modality `ω_v = sigmoid(TCPConfidenceLayer_v(z_v))`, used for fusion-weighting (§6.5/6.7); (b) fused-level `TCPConfidence_sig = sigmoid(TCPConfidenceLayer(feature))`, used only downstream by `active_learning/strategies.py::confidence_based_selection` (§14), never inside `MainModel`'s own loss.

**`inference()` — VERIFIED FROM CODE:**
```python
def inference(self, MMlogit):
    dect_logit = MMlogit.detach().argmax(dim=1)
    return (dect_logit >= 1).long().tolist()
```
Since `num_class=2` (argmax over `{0,1}`), `dect_logit>=1` is equivalent to `dect_logit==1` — this `>=1` phrasing exactly mirrors the `y_anomaly=(fault_indexs>=1)` construction earlier, applied here to the *predicted* class index rather than the raw label. With only 2 classes, this is a plain binary argmax; the `>=1` form only matters if `num_class` were ever `>2`, which it is not, anywhere in the executed path for Dataset C.

---

## 7. Uncertainty Modeling

**Mean μ:** `fc_mu` output, `Sigmoid`-bounded to `(0,1)`, then passed through `SimpleAttention` (a learned elementwise gate: `x * softmax(x·W+b)`) — VERIFIED FROM CODE (§6.4).

**Variance σ² / σ / log-variance:** as documented in §6.4, the single tensor `var`/`var_attn` (also `Sigmoid`-bounded to `(0,1)`) is used **inconsistently**: as raw variance in `reparametrize` (`std=var.sqrt()`) and as log-variance in `kl_loss` (`var.exp()`). **This is not resolved in this document — both usages are recorded as fact.**

**Reparameterization trick:** `new_sample = mu_attn + eps * std`, `eps ~ N(0,I)` via `torch.randn_like(std)` — VERIFIED FROM CODE, standard reparameterization form, matching the paper's Eq. 4 (`zv = μv + ξσv²`, note the paper's own equation also writes `ξσv²`, i.e. multiplying noise by *variance* rather than *standard deviation* — the code instead multiplies `eps` by `std=var.sqrt()`, i.e. **standard deviation**, not variance. **This is a second PAPER VS CODE DISCREPANCY**: the paper's stated equation (Eq. 4, `zv = μv+ξσv²`) does not match the code's actual `mu + eps*std` (which mathematically requires `eps*σ`, not `eps*σ²`, to be a valid reparameterization of `N(μ,σ²)`. The code's version is the mathematically standard/correct reparameterization form; the paper's printed equation, taken completely literally, is not. Both are recorded here as-is; §16 logs the discrepancy formally.**

**Number of samples:** **one** stochastic sample per forward call during ordinary training/evaluation (`reparametrize` is called once per `UncertainBlock.forward`). During active-learning's `entropy_based_selection` (§14), the model's `forward()` is called **5 times** per batch (`n_samples=5` default) specifically to average entropy over repeated stochastic samples — this is the only place multiple forward passes ("MC-style" sampling) are used, and it is for **active-learning query scoring only**, not for training or ordinary evaluation.

**Deterministic vs. stochastic behavior:** `UncertainBlock.forward` always samples via `reparametrize` regardless of `self.training` state — **VERIFIED FROM CODE, no `if self.training` branch exists anywhere in `UncertainBlock` or `MainModel`.** This means **`BaseModel.evaluate()` (test-time) also uses a freshly-sampled stochastic embedding each call**, not a deterministic `mu`-only pass — i.e., test-set predictions are not fully deterministic given fixed weights, since `torch.randn_like` draws new noise every call (even though `torch.no_grad()` is used in `evaluate()`, this does not disable the random sampling itself, only gradient tracking).

**How uncertainty becomes confidence:** indirectly — `UncertainBlock`'s `mu`/`var` do not themselves become "confidence"; instead, the **sampled** `new_<modality>` embedding is fed into `TCPConfidenceLayer_<v>`, whose sigmoid output is what the paper/code calls "confidence" (`ω_v`, §6.5). The KL loss (`mean_kl_loss`) is a separate, additive loss term (§11) that regularizes the *distributional* parameters toward `N(0,I)`, but does not itself feed into the confidence/fusion-weight computation.

| Paper Equation | Code Expression | Source | Match? |
|---|---|---|---|
| Eq.2: `μv,σv² = MLP(gv)` | `mu=fc_mu(encoded); var=fc_var(encoded)` | `main_model.py::UncertainBlock.forward` | Structurally yes (both MLP-derived); code additionally applies `SimpleAttention` gating not mentioned in this specific paper equation |
| Eq.3: `p(zv\|x)=N(μv,σv²)` | implicit in `reparametrize` | `main_model.py::UncertainBlock.reparametrize` | Consistent framing |
| Eq.4: `zv=μv+ξσv²`, `ξ~N(0,I)` | `mu + eps*std` where `std=var.sqrt()` | `main_model.py::UncertainBlock.reparametrize` | **Does NOT match literally** — code uses `σ` (`std`), paper's printed equation uses `σ²`; code is the standard/mathematically-valid form |
| Eq.5: KL(N(μv,σv²)‖N(0,I)) = ½Σ(μv²+σv²−log σv²−1) | `-0.5*mean(sum(1+var-mu**2-var.exp(), dim=-1))` | `main_model.py::UncertainBlock.kl_loss` | Algebraically equivalent **only if** code's `var` = paper's `log σv²` — but the same `var` is used as raw `σv²` in `reparametrize` (see above); **internally inconsistent within the code itself**, and the paper's Eq. 5 (written in terms of `σv²` directly, with an explicit `log σv²` term) is consistent with the code's KL formula only under that log-variance reading, not under the `reparametrize` reading |
| Eq.6/§III-D TCP definition | `p_target = gather(softmax(logit),1,y)` | `main_model.py::_compute_confidence_loss` | Matches |
| Eq.7: `ωv=Sigmoid(MLP(zv))` | `sigmoid(TCPConfidenceLayer_v(new_v))` | `main_model.py::forward` | Matches (single Linear, not multi-layer MLP, but Sigmoid(Linear(z)) is a valid, minimal reading of "MLP") |
| "h=Σωvzv" (paper prose, §III-D) | `torch.cat((...), dim=-1)` | `main_model.py::forward` | **Does NOT match** — concatenation, not summation (§6.5/§16) |

---

## 8. Modality-Specific Classifiers and Confidence

Fully specified in §6.5/§6.8. Summary of dimension flow, per modality `v ∈ {trace, metric, log}`:
```
z_v [B,64] (sampled, from UncertainBlock)
  → TCPClassifierLayer_v: Linear(64,2) → logit_v [B,2]
  → softmax(logit_v) → prob_v [B,2]
  → TCP_v = prob_v[range(B), y_anomaly]           # gathered at true label
  → TCPConfidenceLayer_v: Linear(64,1) → sigmoid → ω_v [B,1]
  → per-modality loss: MSE(ω_v, TCP_v) + CrossEntropy(logit_v, y_anomaly)
```
No entropy is computed per-modality anywhere in `MainModel` (entropy only appears in the separate active-learning module, §14, computed on the **fused** `MMlogit`, not per-modality logits). Modality outputs are combined only via the confidence-weighted concatenation described in §6.5/§6.7 — there is no separate "modality voting" or per-modality-prediction-averaging step.

---

## 9. Fusion and Final Prediction — Full Tensor-Shape Trace

```
Raw chunk (per graph):
  metrics [12,10,7]   traces [12,10,1]   logs [12,15]
        │                   │                  │
   GRUEncoder(7,64)   GRUEncoder(1,64)    Linear(15,64)
        │                   │                  │
     [12,64]             [12,64]            [12,64]     (per-node, N=12 within this one graph; batched: [B*12,64])
        │                   │                  │
   GraphModel1          GraphModel1        GraphModel1     (GATv2Conv×1, 4 heads, maxpool-over-heads, GlobalAttentionPooling)
        │                   │                  │
      [B,64]              [B,64]             [B,64]        (graph-level, one vector per sample)
        │                   │                  │
   UncertainBlock      UncertainBlock     UncertainBlock    (Linear→LayerNorm→Tanh→[fc_mu,fc_var]→SimpleAttention→reparam)
        │                   │                  │
   new_metric[B,64]   new_trace[B,64]     new_log[B,64]     (stochastic samples)
        │                   │                  │
   ×sigmoid(conf_metric) ×sigmoid(conf_trace) ×sigmoid(conf_log)   [B,64] each, elementwise-scaled
        │                   │                  │
        └─────────── torch.cat(dim=-1), order=(trace,metric,log) ──────────┘
                              │
                          feature [B,192]
                              │
                        MMClasifier(feature)
                              │
                        MMlogit [B,2]
                              │
                   inference(): argmax(dim=1) >=1 → y_pred [B] (0/1 list)
```

---

## 10. Mathematical Specification

**Notation.** Let `B` = batch size, `N=12` = node count, `T=10` = window length, `D=64` = hidden dim. For modality `v∈{m,t,l}` (metric, trace, log): `X_v` = raw per-node modality tensor; `g_v ∈ ℝ^{B×64}` = graph-pooled embedding; `(μ_v,σ_v) ∈ ℝ^{B×64}` = distributional parameters; `z_v ∈ ℝ^{B×64}` = sampled embedding; `ω_v ∈ ℝ^{B×1}` = fusion/confidence weight; `h ∈ ℝ^{B×192}` = fused representation; `y ∈ {0,1}^B` = binary anomaly target.

| Mathematical Component | Paper Equation / Definition | Code Implementation | Exact Match? | Notes |
|---|---|---|---|---|
| Metric encoding | `fm = GRU({M_1..M_T})` (Eq.1) | `GRUEncoder(7,64)` on `ndata["metrics"]` | Yes | `models/layers.py::GRUEncoder` |
| Trace encoding | prose only, no numbered equation | `GRUEncoder(1,64)` on `ndata["traces"]` | Structurally yes | Input dim 1 is a dataset-compatibility fact (Step 1 §9), not from paper |
| Log encoding | prose only | `Linear(15,64)` on `ndata["logs"]` | Structurally yes; paper describes a Hawkes+FC pipeline whose *output* is this 15-dim vector — that pipeline itself lives in Dataset C preprocessing (Step 1), not in this repo |
| Graph attention | "GAT to learn salient info... pooling to obtain graph-level features" (§III-C prose) | `GATv2Conv`(1 layer, 4 heads) + custom max-over-heads + `GlobalAttentionPooling` | Consistent in spirit; GATv2 (not vanilla GAT) and the max-pool-over-heads step are code-level specifics with no paper equation given | Paper does not specify head-count, layer-count, or the max-pool-over-heads mechanism |
| Distributional params | Eq.2: `μv,σv²=MLP(gv)` | `fc_mu`,`fc_var` (each `Linear(128,64)`+Sigmoid) preceded by `Linear(64,128)+LayerNorm+Tanh`, followed by `SimpleAttention` gating | Structurally yes, with the `SimpleAttention` step and Sigmoid-bounding being code-level additions the paper's Eq.2 does not mention |
| Sampling | Eq.3: `p(zv\|x)=N(μv,σv²)` | `reparametrize` | Consistent framing |
| Reparameterization | Eq.4: `zv=μv+ξσv²` | `mu+eps*std`, `std=var.sqrt()` | **NO** — paper writes `σv²`, code uses `σv` (`std`); see §7 |
| KL regularization | Eq.5: `KL=½Σ(μv²+σv²−logσv²−1)` | `-0.5*mean(sum(1+var-mu**2-var.exp(),dim=-1))` | Consistent **only** under a log-variance reading of `var`, which contradicts `reparametrize`'s use of the same `var` as raw variance — internally inconsistent, see §7 |
| TCP | Eq.6: `TCPv=y·p(y\|zv)` | `gather(softmax(logit),1,y)` | Yes |
| Confidence network | Eq.7: `ωv=Sigmoid(MLP(zv))` | `sigmoid(Linear(zv))` | Yes, with "MLP" realized as a single Linear layer |
| Per-modality losses | Eq.8 `Lp^v` (CE), Eq.9 `Lq^v` (MSE), Eq.10 `Lcon=Σ(Lp^v+Lq^v)` | `criterion(logit,y) + F.mse_loss(confidence,p_target)`, summed over 3 modalities | Yes, matches Eq.8–10 exactly in structure |
| Fusion | prose "h=Σωvzv" | `torch.cat((z_t*ω_t,z_m*ω_m,z_l*ω_l),dim=-1)` | **NO** — concatenation, not weighted sum; see §6.5/§16 |
| Final classification | Eq.11: `ŷ=argmax[Softmax(Wh+b)]` | `MMClasifier(feature).argmax(dim=1)`, `MMClasifier` is `Linear`-based (exact structure depends on `hidden_dim`, see below) | Structurally yes |
| Total training loss | Eq.13: `L=Lano+λ1·Lkl+λ2·Lconf` | `total_loss = MMLoss + 0.6*confidence_loss + mean_kl_loss` | **Partially** — see §11 for the exact λ-coefficient discrepancy |

**`MMClasifier`'s exact structure is itself a documented open question, not a settled fact:**
Given the code's actual default `hidden_dim=[64]` (see §15's finding that `params.json`'s `hidden_dim:[64,64]` is never passed through from `main.py`), `MMClasifier` as constructed by the `for i in range(1,len(hidden_dim))` loop (which does not execute when `len(hidden_dim)==1`) reduces to a **single `LinearLayer(64,2)`** — VERIFIED by direct construction of this exact code in isolation (§10 below has the empirical test). This directly **conflicts** with the `feature` tensor's actual dimension of `192` (`3×64`, from the concatenation in §6.7/§6.9), which was also empirically confirmed by constructing the identical `UncertainBlock`+concatenation code in isolation. Feeding a `[B,192]` tensor into a `Linear(64,2)` layer raises a PyTorch shape-mismatch error (`mat1 and mat2 shapes cannot be multiplied`), **empirically reproduced in this investigation using isolated PyTorch code copied verbatim from the relevant MUAD classes.**

**This directly contradicts the recovered `result/training.log` (Step 1 §12; also this document §2), which shows 100 completed epochs with no traceback across multiple runs**, meaning the actual executed code, at the commit that produced those logs, evidently did **not** hit this shape mismatch. **This tension is explicitly unresolved.** Two facts are both true and are recorded as such, without one overriding the other:
1. **VERIFIED (this investigation, isolated code test):** `MainModel.__init__`'s literal, unmodified logic — with the default `hidden_dim=[64]` that `main.py` actually supplies (since it never passes `hidden_dim` as a kwarg) — constructs an `MMClasifier` whose first (and only) layer expects a 64-dim input, while `forward()`'s `feature` tensor is 192-dim.
2. **VERIFIED (recovered `training.log`):** actual logged training runs at the commit containing this identical code completed 100 epochs without a traceback.
**No explanation for this contradiction is asserted.** Possible resolutions (untested, listed only as open hypotheses, not conclusions): a DGL/PyTorch version-specific broadcasting behavior not reproduced in this investigation's plain-PyTorch isolated test; a different effective `hidden_dim` value at actual runtime not visible in the `.py` source (e.g. supplied via an untracked local config or shell environment variable not committed to git); or an error in this investigation's shape trace that further, more exhaustive testing (e.g., installing the exact pinned `dgl==2.0.0`/`torch==?` versions and running the real `main.py` end-to-end) would reveal. This is logged as **UNKNOWN / UNVERIFIABLE** pending an actual full-stack run of the real repository.

---

## 11. Loss Functions

**VERIFIED FROM CODE, `models/main_model.py::MainModel.forward`, exhaustive enumeration:**

| Loss | Formula | Code | Inputs | Reduction | Weight in total_loss |
|---|---|---|---|---|---|
| Per-modality CE (×3, trace/metric/log) | `CrossEntropy(logit_v, y)` | `criterion(logit, y)` inside `_compute_confidence_loss`, `criterion=nn.CrossEntropyLoss()` | `logit_v [B,2]`, `y=y_anomaly [B]` | mean (CE default) | summed into `confidence_loss`, then ×0.6 |
| Per-modality confidence MSE (×3) | `MSE(ω_v, TCP_v)` | `F.mse_loss(confidence, p_target)` | `confidence [B]`, `p_target [B]` | mean (MSE default) | summed into `confidence_loss`, then ×0.6 |
| `confidence_loss` (total) | `Σ_v (CE_v + MSE_v)` for v∈{trace,metric,log} | `confidence_loss_trace + confidence_loss_metric + confidence_loss_log` | — | sum of 6 sub-terms | **× 0.6** (hardcoded literal, see below) |
| KL divergence (×3, averaged) | `mean_v KL(N(μv,σv)‖N(0,I))` | `mean_kl_loss = (m_kl_loss + t_kl_loss + l_kl_loss) / 3` | `mu_attn,var_attn` per modality | mean over 3 modalities, and `kl_loss` itself is `torch.mean(...)` over batch | **× 1** (added directly, no extra coefficient) |
| Final anomaly loss (`MMLoss`) | `CrossEntropy(MMlogit, y)` | `MMLoss = criterion(MMlogit, y_anomaly)` | `MMlogit [B,2]`, `y_anomaly [B]` | mean | **× 1** |
| **`total_loss`** | — | `MMLoss + 0.6*confidence_loss + mean_kl_loss` | — | — | — |

**Exact total-loss equation, matching the code verbatim:**
```
total_loss = MMLoss + 0.6 * (CE_trace+MSE_trace + CE_metric+MSE_metric + CE_log+MSE_log) + (KL_trace+KL_metric+KL_log)/3
```

**What `loss1`/`loss2`/`aloss`/`b_loss` actually represent — determined precisely, not guessed:**
- `MainModel.__init__(self, ..., aloss=0.4, b_loss=0.01, **kwargs)` stores `self.k1=aloss`, `self.k2=b_loss` — **but neither `self.k1` nor `self.k2` is referenced anywhere in `forward()` or anywhere else in the class.** VERIFIED by exhaustive `grep` of `self.k1`/`self.k2` in `models/main_model.py`: they are assigned and never read. **These are dead attributes.**
- `params.json`'s `"loss1": [0.001,0.01,0.1,1]` and `"loss2": [0.4,0.6,0.8,1]` are **never read by `main.py`** at all (VERIFIED — `main.py` only ever accesses `params["random_seed"]`, `params["gpu"]`, `params["lr"]`, `params["patience"]`, `params["result_dir"]`, `params["evaluation_epoch"]`, `params["epochs"]`, `params["trainType"]`, `params["max_iter"]`, `params["precicison"]` — an exhaustive list, confirmed by grep). **`loss1` and `loss2` are dead configuration entries, present in the config file but never consumed by any executed code path.**
- The paper's Eq.13, `L_MUAD = L_ano + λ1·L_kl + λ2·L_conf`, describes `λ1`,`λ2` as tunable weighting coefficients — the paper's §IV-F even reports "optimal values of λ1... 1.0 and 0.1" and "optimal value for... λ2... 0.6" from a sensitivity sweep (Fig.7). **The code's actual, hardcoded coefficients are `λ1=1` (implicit, KL added with no multiplier) and `λ2=0.6` (hardcoded literal)** — the `0.6` **does** match one of `loss2`'s grid values and the paper's stated "optimal" λ2, but it is **hardcoded directly in `forward()`, not read from `params.json`'s `loss2` list or from any sweep/selection mechanism** — VERIFIED, no code anywhere iterates over `params["loss1"]`/`params["loss2"]` to run the sensitivity sweep the paper's Fig.7 describes. **The sweep described in the paper (Fig. 7, §IV-F) has no corresponding executed code in this repository.**
- **Whether applied per-modality or after fusion:** the CE+MSE components are computed per-modality (3× each) and summed before the single ×0.6 multiplication; `MMLoss` (the main anomaly CE) is computed once, after fusion, and is unweighted (implicit ×1).
- **Whether applied during full_label vs active_learning:** identical — `total_loss` is computed identically inside `MainModel.forward()` regardless of `params["trainType"]`; the training-mode branching in `main.py` only affects *which data* is fed to `.fit()`, not the loss formula itself.

---

## 12. Training Procedure

**VERIFIED FROM CODE, `training/base_model.py::BaseModel.fit`, exact execution order:**
```
optimizer = Adam(model.parameters(), lr=self.lr)          # lr from params["lr"]=0.001
best_f1, best_state, best_epoch = -1, None, 0
prev_loss, worse_count = inf, 0

for epoch in 1..self.epochs:                                # self.epochs = params["epochs"]=100
    model.train()
    total_loss = 0
    for graph, labels in train_loader:                       # batch_size=50, shuffle=True
        graph = graph.to(device)
        optimizer.zero_grad()
        res = model(graph, labels)                            # labels used directly as fault_indexs
        loss = res["loss"]
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    log("Epoch {epoch}/{self.epochs}, Loss: {avg_loss}")

    if avg_loss > prev_loss:                                  # loss-increase-based early stopping
        worse_count += 1
        if worse_count >= self.patience:                       # patience = params["patience"]=15
            break
    else:
        worse_count = 0
    prev_loss = avg_loss

    if epoch % evaluation_epoch == 0 and test_loader:           # evaluation_epoch = params["evaluation_epoch"]=5
        test_results = evaluate(test_loader)
        if test_results["F1"] > best_f1:
            best_f1 = test_results["F1"]
            best_state = deepcopy(model.state_dict())            # TEST-SET-DRIVEN CHECKPOINT SELECTION
            best_epoch = epoch

if best_state: model.load_state_dict(best_state)
return best_f1, best_epoch
```
- **Dataset loading:** per §4, via `ChunkDataset`+`create_dataloader`.
- **Train/test/validation:** in `full_label` mode, only train+test exist (no validation split at all — VERIFIED, `main.py`'s `full_label` branch never calls `split_data`). In `active_learning` mode, train/val/test all exist (§14).
- **Batch size:** 50 (default, `create_dataloader`, never overridden anywhere).
- **Shuffle:** `True` for train, `False` for test/val (VERIFIED, exact `shuffle=` args at each `create_dataloader` call site in `main.py`).
- **Optimizer:** `torch.optim.Adam`, **no weight decay** (default `weight_decay=0`, never set), no scheduler of any kind (no `lr_scheduler` import or usage anywhere in the codebase).
- **Learning rate:** `0.001` (from `params.json`), constant throughout training (no decay/schedule).
- **Epochs:** `100` (config default) — **but recovered `training.log` shows actual runs logging `Epoch X/50`** (§10/§15 — an unreconciled discrepancy, not explained here).
- **Random seed:** `42`, set once at the very start of `main()` via `seed_everything` (§13's utils reference), before any data loading or model construction.
- **GPU/CPU:** `torch.device("cuda" if params["gpu"] and cuda.is_available() else "cpu")` — `params["gpu"]=false` in the committed config, so **CPU is the default/actual configured device.**
- **Gradient handling:** standard `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()`, once per batch, no gradient accumulation, no gradient clipping anywhere in the code.
- **Early stopping condition:** based on **training loss increasing** for `patience` consecutive epochs (`avg_loss > prev_loss`) — this is **not** validation/test-loss-based; it is a simple training-loss-plateau/increase heuristic.
- **Checkpoint saving:** `best_state` is an in-memory `deepcopy` of `state_dict()`, saved to disk only once, at the very end of the `full_label` branch of `main.py` (`torch.save(model.model.state_dict(), ".../best_model.pth")`) — **not** saved incrementally per-epoch, and **not** saved at all inside the `active_learning` branch of `main.py` (no `torch.save` call exists in that branch — VERIFIED by grep).
- **Logging:** Python `logging` module, configured by `utils/logging_utils.py::setup_logging` (file+stream handler, `INFO` level), invoked once in `main()`.

---

## 13. Full-Label Training Mode

**VERIFIED FROM CODE, `main.py`, `if params['trainType']=="full_label":` branch:**
- **Samples used:** the **entire** `chunk_train.pkl` (all keys, `train_keys=list(train_data.keys())`) — i.e., all 3,366 recovered train chunks (Step 1 §12), no subsampling.
- **Labels available:** every training sample's true `culprit` value (used directly as `fault_indexs` in `model(graph,labels)`) — full supervision, no pseudo-labeling logic invoked in this branch.
- **Losses:** the full `total_loss` formula (§11), unchanged from the active-learning branch.
- **Checkpoint selection:** `model.fit(train_loader, test_loader, params["evaluation_epoch"])` — **test set is passed directly into `.fit()`**, and per §12's trace, `best_state` is chosen by test-set F1 during training.
- **Stopping behavior:** training-loss-plateau early stopping (§12), independent of any test-set metric; **additionally**, after `.fit()` returns, the code does **one more** `model.evaluate(test_loader)` call (line 68) — this final evaluation, whose results are dumped via `dump_scores`, is performed on the model **already loaded with the test-set-best checkpoint** (since `.fit()` calls `self.model.load_state_dict(best_state)` before returning) — so this "final evaluation" is not independent of the test set used to select the checkpoint; it is evaluating the checkpoint that was *selected because* it scored best on this same test set.
- **Test-set access during training:** confirmed direct and repeated — every `evaluation_epoch`-th epoch (default every 5 epochs) throughout the full `100`-epoch (or `50`, per recovered logs — unreconciled) training run.
- **Final evaluation:** `test_results = model.evaluate(test_loader)`, saved via `dump_scores(test_results, params["result_dir"])` — this is the number that would appear in `result/scores.txt` for a `full_label` run (the recovered `scores.txt`, per Step 1, in fact only contains entries from `active_learning`-branch calls to `dump_scores(..., iteration)`, not a `full_label` run — no `full_label`-mode empirical log was recovered).

---

## 14. Active Learning

**VERIFIED FROM CODE, `main.py`, `elif params["trainType"]=="active_learning":` branch, and `active_learning/strategies.py`.**

### Initialization
```python
train_keys, val_keys = split_data(train_data, train_ratio=0.1)   # utils/general_utils.py
```
- `split_data`: `n_train = int(len(keys)*0.1)`; `train_keys = random.sample(keys, n_train)` (module-level `random`, seeded via `seed_everything`); `val_keys` = all remaining keys **not** in `train_keys` (list-membership check, `O(n²)` but functionally correct).
- **Initial labeled pool:** 10% of `chunk_train.pkl`'s keys (≈337 of 3,366, per Step 1's recovered counts) — matches the paper's "randomly labeling 10% of the data."
- **Remaining pool (`val_keys`):** the other ≈90% (≈3,029), used as the unlabeled pool the active-learning loop queries from. **Note:** this pool is drawn entirely from `chunk_train.pkl` — the paper/code's "validation" terminology here refers to the *unlabeled query pool*, not a held-out model-selection validation set distinct from train; `val_loader` (built from `val_keys`) is passed to `hybrid_selection`, never to `.fit()` as a third, separate validation split for early stopping.

### Training (per iteration, `for iteration in range(params["max_iter"])`, `max_iter=1` in the committed config)
- A **new** `BaseModel` (hence a freshly-initialized `MainModel`, new random weights subject to the global seed's PyTorch RNG state at that point) is constructed **every iteration** — VERIFIED, `model = BaseModel(...)` is inside the `for iteration` loop, not created once outside it.
- `model.fit(train_loader, test_loader, params["evaluation_epoch"])` is called — **same `train_loader`/`test_loader` objects across all iterations** (constructed once, before the loop) — see the critical finding below.
- **CRITICAL, VERIFIED-BY-CODE-TRACING FINDING:** although `train_keys` is extended (`.extend(new_train_keys)`, `.extend(high_conf_keys)`) and `train_dataset` is reassigned to a `CombinedDataset` including newly pseudo-labeled samples (lines 121–129 of `main.py`), **`train_loader` itself is never reconstructed from this updated `train_dataset`** — VERIFIED by exhaustive grep: `train_loader = create_dataloader(...)` appears exactly once in the entire file, at line 84, before the loop. Since `ChunkDataset`/`CombinedDataset` (§6, data/dataset.py) build their `.data` list **eagerly at construction time**, and `DataLoader` holds a reference to the `Dataset` object it was built from (not to the *variable name* `train_dataset`), reassigning the Python variable `train_dataset` to a new `CombinedDataset` object does **not** change what `train_loader` iterates over. **This means, as released, the active-learning loop's queried/pseudo-labeled samples are computed and tracked (`train_keys` grows, logged in `"Iteration {n} complete. Train size: ..."`) but are never actually used to retrain the model in the next iteration's `.fit()` call**, since that call still receives the original, unexpanded `train_loader`. This is stated as a directly-traced code fact, not a hypothesis.

### Uncertainty estimation (`active_learning/strategies.py::entropy_based_selection`)
- **Entropy formula:** `H(x) = -Σ_c p(c|x)·log(p(c|x)+1e-8)`, computed from `softmax(MMlogit, dim=1)` — matches paper Eq.12 exactly in form (`H(xi)=-Σp(c|xi)logp(c|xi)`), with the `+1e-8` numerical-stability epsilon being a code-only addition not in the paper's equation.
- **MC-style repeated sampling:** `n_samples=5` (default) forward passes per batch — **VERIFIED**, `for _ in range(n_samples): res = model(graph, labels)`, entropy computed each time and averaged (`torch.stack(entropy_batch).mean(dim=0)`). This genuinely produces different results each call because `UncertainBlock`'s `reparametrize` samples fresh noise every forward pass regardless of train/eval mode (§7).
- **Probability aggregation:** simple arithmetic mean of the 5 per-call entropy values (not of the probabilities themselves — entropy is computed per-sample-call, then averaged).
- **Uncertainty ranking:** `sorted_indices = np.argsort(uncertainties)[::-1]` — **descending** sort (highest entropy first).

### Query
- `n_select=200` (default, `hybrid_selection`'s signature; the actual call in `main.py` passes `n_select=200` explicitly).
- `selected_uncertain = uncertain_indices[:n_select]` — top-200 highest-entropy samples from the **entire** `val_loader` pool (not per-batch top-k; `entropy_based_selection` collects entropies across all batches first via `.extend`, then the caller slices the globally-sorted list).
- **Tie handling:** `np.argsort` uses a stable or default sort algorithm without special tie-breaking logic (no explicit tie-handling code exists) — ties are resolved by whatever `np.argsort`'s default (quicksort) does, effectively arbitrary/implementation-defined ordering among exactly-equal entropy values.

### Pseudo-labeling (`confidence_based_selection`)
- **Confidence threshold:** `confidence_threshold=0.9` (default, matches paper's stated θ=0.9).
- **Confidence source:** `res["TCPConfidence_sig"]` — the **fused-feature** confidence (`sigmoid(TCPConfidenceLayer(feature))`, 192→1), **not** any per-modality TCP/confidence value, and **not** literally the paper's per-modality `TCPv` (§III-D) — this is a distinct, additional confidence head (§6.8) applied to the post-fusion representation.
- **Which samples are pseudo-labeled:** every sample in `val_loader` whose `TCPConfidence_sig > 0.9`; `predicted_labels` (used as pseudo-labels) are `MMlogit.argmax(dim=1)` restricted to those high-confidence indices — i.e., the **model's own predicted class**, not the true `culprit`/`fault_indexs`.
- **Global-index arithmetic caveat (VERIFIED code pattern, potential-bug flagged, not asserted as definitely triggered):** `global_indices = [batch_idx*len(labels) + i for i in batch_indices]` assumes every batch (including the last) has exactly `len(labels)` elements equal to the configured `batch_size=50`; if the pool size is not an exact multiple of 50, the final batch is smaller, and this index-arithmetic would compute incorrect global indices for that batch. Whether the actual pool sizes involved ever hit this edge case was not empirically tested in this investigation.
- **Original labels remain fixed:** yes for the initial `train_keys`/`val_keys` split — pseudo-labels only ever apply to `high_conf_keys`, which are packaged into a **separate** `ChunkDataset(..., pseudo_labels)` (VERIFIED, `data/dataset.py::ChunkDataset.__init__`'s `labels` parameter overrides `chunk["culprit"]` when provided, exactly for this purpose) — true labels are never overwritten in-place.

### Iteration
- **Loop count:** `params["max_iter"]=1` in the committed config — so, as configured, the active-learning loop runs **exactly once** regardless of the paper's iterative framing ("we repeat the above iterative process until the query budget b is exhausted").
- **Stopping condition:** `if test_results["Pre"] >= params["precicison"]` (`precicison=0.9999`) → `break`. Given `max_iter=1`, this condition is only ever checked once and, if unmet (near-certain given a 0.9999 threshold), the loop simply ends because `range(1)` is exhausted — not because the precision condition triggered.
- **Test-set involvement:** `test_results = model.evaluate(test_loader)` is called every iteration, both to log/dump scores (`dump_scores(test_results, result_dir, iteration)`) and to check the precision-based stopping condition — confirmed, direct test-set use in the control flow of the active-learning loop, in addition to the test-set-driven checkpoint selection already occurring inside `.fit()` (§12).
- **Pool updates:** `del val_keys[idx]` for each selected index (both uncertain and high-confidence), removing them from the pool — this executes regardless of whether `train_loader` is actually updated to reflect the growing `train_keys` (per the critical finding above).

### Pseudocode (matching the actual implementation exactly, including its apparent non-propagation bug)
```
seed_everything(42)
train_data, node_num, edges = load(chunk_train.pkl, metadata.json)
test_data, _, _ = load(chunk_test.pkl, metadata.json)

train_keys, val_keys = split_data(train_data, train_ratio=0.1)     # ~337 / ~3029
train_loader = DataLoader(ChunkDataset(train_data, train_keys, ...), batch_size=50, shuffle=True)
val_loader   = DataLoader(ChunkDataset(train_data, val_keys, ...),   batch_size=50, shuffle=False)
test_loader  = DataLoader(ChunkDataset(test_data, all_test_keys, ...), batch_size=50, shuffle=False)

for iteration in range(1):                          # max_iter=1
    model = BaseModel(...)                            # fresh weights each iteration
    best_f1, best_epoch = model.fit(train_loader, test_loader, evaluation_epoch=5)
        # -> trains up to `epochs` epochs (config:100 / observed-log:50) on the ORIGINAL 10%-labeled train_loader
        # -> test-set-F1-driven checkpoint selection inside .fit()

    test_results = model.evaluate(test_loader)
    dump_scores(test_results, result_dir, iteration)

    uncertain_idx, high_conf_idx, pseudo_labels = hybrid_selection(model.model, val_loader, device, n_select=200)
        # entropy_based_selection: 5 stochastic forward passes/batch, sort desc, take top 200
        # confidence_based_selection: TCPConfidence_sig > 0.9 -> pseudo-label = argmax(MMlogit)

    new_train_keys = [val_keys[i] for i in uncertain_idx]
    train_keys.extend(new_train_keys)                  # tracked, but...
    if high_conf_idx:
        pseudo_dataset = ChunkDataset(train_data, [val_keys[i] for i in high_conf_idx], ..., pseudo_labels)
        train_dataset = CombinedDataset(train_dataset, pseudo_dataset)   # ...train_loader is NEVER rebuilt from this
        train_keys.extend([val_keys[i] for i in high_conf_idx])

    for idx in sorted(uncertain_idx + high_conf_idx, reverse=True):
        del val_keys[idx]

    log(f"Train size: {len(train_keys)}, Val size: {len(val_keys)}")   # grows, but train_loader unaffected

    if test_results["Pre"] >= 0.9999:
        break
# loop ends after 1 iteration (max_iter=1); no torch.save() call in this branch
```

---

## 15. Hyperparameter Specification

| Parameter | Value | Source | Code Location | Notes |
|---|---|---|---|---|
| `random_seed` | 42 | `params.json` | `main.py`→`seed_everything` | Consumed |
| `gpu` | false | `params.json` | `main.py` device selection | Consumed; CPU is actual default |
| `lr` | 0.001 | `params.json` | `BaseModel.fit`, `optim.Adam` | Consumed |
| `patience` | 15 | `params.json` | `BaseModel.fit` early-stop counter | Consumed |
| `result_dir` | "./result/" | `params.json` | `main.py`, `BaseModel` | Consumed |
| `evaluation_epoch` | 5 | `params.json` | `BaseModel.fit` | Consumed |
| `max_iter` | 1 | `params.json` | `main.py` active_learning loop bound | Consumed; caps AL to 1 iteration despite paper's "iterative" framing |
| `precicison` | 0.9999 | `params.json` | `main.py` AL early-stop check | Consumed (sic — misspelled key, used as-is) |
| `epochs` | 100 | `params.json` | `BaseModel.__init__`→`self.epochs` | **Consumed, but recovered training.log shows `Epoch X/50`, not `/100` — unreconciled** |
| `hidden_dim` | [64, 64] | `params.json` | **Never read by `main.py`** | **Dead config entry** — `MainModel.__init__` always uses its own default `[64]` since `main.py` never passes `hidden_dim` as a kwarg to `BaseModel`/`MainModel` |
| `trainType` | "full_label" | `params.json` | `main.py` branch selector | Consumed; default is full_label, not active_learning |
| `loss1` | [0.001,0.01,0.1,1] | `params.json` | **Never read anywhere** | **Dead config entry** |
| `loss2` | [0.4,0.6,0.8,1] | `params.json` | **Never read anywhere** | **Dead config entry**; the code's hardcoded `0.6` multiplier on `confidence_loss` happens to equal one value from this list but is not sourced from it |
| `aloss` (`self.k1`) | 0.4 (class default) | `models/main_model.py::MainModel.__init__` signature default | Assigned to `self.k1`, **never read in `forward()`** | Dead attribute |
| `b_loss` (`self.k2`) | 0.01 (class default) | same | Assigned to `self.k2`, **never read** | Dead attribute |
| `num_class` | 2 (class default) | `MainModel.__init__` | Used throughout (`TCPClassifierLayer_*`, `MMClasifier` final layer) | Consumed |
| `attn_head` | 4 (class default) | `GraphModel1.__init__` | `GATv2Conv(num_heads=4)`, `nn.MaxPool1d(4)` | Consumed |
| `activation` (GAT neg. slope) | 0.2 (class default) | `GraphModel1.__init__` | `GATv2Conv(negative_slope=0.2)` | Consumed |
| `attn_drop` | 0 (kwargs.get default) | `GraphModel1.__init__` | `GATv2Conv(attn_drop=...)` | Consumed; effectively always 0 since never overridden |
| Confidence threshold θ | 0.9 (function default) | `active_learning/strategies.py::hybrid_selection`/`confidence_based_selection` | Consumed; matches paper's stated θ |
| `n_select` | 200 (explicit call arg in `main.py`) | `main.py` line 116 | Consumed |
| MC samples (entropy) | 5 (`entropy_based_selection` default `n_samples`) | `active_learning/strategies.py` | Consumed |
| `train_ratio` (AL initial split) | 0.1 (explicit call arg in `main.py`) | `main.py` line 76 / `split_data` | Consumed; matches paper's 10% claim |
| Batch size | 50 (function default) | `data/utils.py::create_dataloader` | Consumed, never overridden |
| Weight decay | 0 (Adam default, unset) | `BaseModel.fit` | Consumed implicitly |
| LR scheduler | none | — | No scheduler code exists anywhere | N/A |

**Conflicts, shown without collapsing:**

| Parameter | Paper | Config (`params.json`) | Runtime/Recovered Artifact | Status |
|---|---|---|---|---|
| Epochs | Not numerically specified in paper text for Dataset C | 100 | Recovered `training.log`: `Epoch X/50` | **Unreconciled** |
| `hidden_dim` | Not specified as a list in paper | `[64,64]` | Effectively `[64]` at runtime (never wired through) | **Config value never takes effect; code default silently used instead** |
| `loss1`/`loss2` (λ1/λ2 sweep) | Paper reports a sensitivity sweep, "optimal" λ1=1.0/0.1, λ2=0.6 (Fig.7) | Grids present in config | **No code executes a sweep over these grids; `forward()` hardcodes λ1=1 (implicit), λ2=0.6 (literal)** | **Paper describes an experiment (Fig.7) with no corresponding code in this repository** |
| Train mode | Paper's headline contribution is active learning | Default `"full_label"` | Recovered logs show only `active_learning`-branch runs were actually executed and logged | Config default and what-was-actually-run both established, but neither is `"the" canonical mode` without further evidence |
| Train/test split | 60%/40% (both papers) | N/A (pre-split file, no split code in MUAD) | Recovered artifact: exactly 60.0%/40.0% (Step 1 §12) | Matches papers; split itself performed upstream in Eadro's `align.py`, not MUAD |
| `precicison` threshold | Not stated | 0.9999 | With `max_iter=1`, this condition can only be checked once, making it effectively moot for stopping-early purposes as configured | Config value present but functionally near-inert given `max_iter=1` |
| MC/entropy samples | Paper does not specify a sample count for entropy estimation | N/A | Code default: 5 | Code-only detail, no paper equation specifies this |

---

## 16. Paper vs Code Discrepancies

| Topic | Paper Says | Code Does | Artifact Shows | Status | Reproduction Decision |
|---|---|---|---|---|---|
| Reparameterization | `zv=μv+ξσv²` (Eq.4, literal σ² multiplier) | `mu+eps*std`, `std=σ=var.sqrt()` (multiplies by σ, not σ²) | N/A | Paper/code mismatch; code is the mathematically standard form | Faithful reproduction: follow the code (§22) |
| KL-loss input semantics | Eq.5 written in terms of `σv²` with explicit `logσv²` term | Same tensor `var` used as raw variance in `reparametrize` and as log-variance in `kl_loss` — internally inconsistent within the code itself | N/A | Genuine internal code inconsistency, not merely a paper/code gap | Faithful reproduction: replicate the code exactly as-is, including this inconsistency (§22) |
| Fusion mechanism | Prose: `h=Σωvzv` (implies weighted sum, same dimensionality as each `zv`) | `torch.cat(...)`, tripling the dimension to 192 | Would need actual model run to confirm end-to-end (blocked, §10) | Paper/code mismatch | Faithful reproduction: concatenation, per code |
| `MMClasifier` structure / `hidden_dim` wiring | Not specified numerically | `params.json`'s `hidden_dim:[64,64]` never reaches `MainModel`; effective default `[64]` produces a `Linear(64,2)` `MMClasifier`, apparently shape-incompatible with the 192-dim fused feature | Recovered `training.log` shows successful, crash-free 100-epoch-labeled runs | **Unresolved contradiction, not explained** (§10) | Flag prominently; do not silently "fix" the dimension before attempting a real end-to-end run to see what actually happens |
| λ1/λ2 sensitivity sweep | Fig.7, §IV-F: sweep over λ1∈{...}, λ2∈{...}, report optimal values | No sweep code exists; values hardcoded (`0.6`) or unused (`self.k1/k2`) | N/A | Paper describes an experiment absent from the released repository | Cannot faithfully reproduce Fig.7 from this code alone |
| Active learning "iterative... until query budget exhausted" | Paper frames AL as a multi-round iterative process | `max_iter=1` in committed config — loop runs once | Recovered logs show only single-iteration runs (`"Starting active learning iteration 1"`, never "iteration 2") | Config caps what the paper frames as iterative to one round, as released | Faithful reproduction: `max_iter=1` per config, unless the person explicitly wants a modified run |
| AL sample-selection propagation | Paper implies each iteration's newly-labeled data trains the next round's model | Newly-selected `train_keys`/`train_dataset` are computed and logged but `train_loader` is never rebuilt from them — traced exactly in §14 | N/A (moot at `max_iter=1`, but would matter if `max_iter>1`) | Code-level non-propagation bug, verified by direct tracing | Faithful reproduction: replicate exactly, including non-propagation, if attempting a `max_iter>1` run |
| Root-cause/culprit prediction | Paper's Dataset C experiments (RQ1 text) mention a "preliminary experiment... extended MUAD with a simple classification head for service-level localization" achieving HitRate@1 99.24% | No such extended classification head exists anywhere in this repository — `MainModel` is binary-only | N/A | Paper describes an extension not present in the released code | Out of scope for faithful reproduction of the released repo; would need separate implementation, explicitly labeled as an extension, not part of MUAD's base released code |
| Confidence source for AL pseudo-labeling | Paper's TCP/confidence formulation (Eq.6/7) is presented per-modality | `confidence_based_selection` uses the **fused**-feature `TCPConfidence_sig` (from the separate `TCPConfidenceLayer(192,1)`), not any per-modality TCP | N/A | Paper's per-modality framing vs. code's fused-level confidence for this specific use (AL querying) | Faithful reproduction: use `TCPConfidence_sig` (fused) exactly as coded |
| Test-set-driven checkpoint selection | Not addressed by paper | `BaseModel.fit()` selects `best_state` via test-set F1, both modes | N/A | Confirmed code-level fact (already flagged in Step 1 doc for data/eval purposes; reconfirmed here at the model-training level) | Faithful reproduction: replicate as-is (§22); clean-eval alternative in §23 |

---

## 17. Evaluation Protocol

**Metrics:** Precision, Recall, F1 — **VERIFIED FROM CODE**, `BaseModel.evaluate()`:
```python
for i, pred in enumerate(predictions):
    label = labels[i].item()
    if label == 0:
        TN += 1 if pred==0 else 0
        FP += 1 if pred!=0 else 0
    else:
        TP += 1 if pred!=0 else 0
        FN += 1 if pred==0 else 0
precision = TP/(TP+FP) if (TP+FP)>0 else 0
recall    = TP/(TP+FN) if (TP+FN)>0 else 0
f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0
```
- **Anomaly labels:** the raw `labels` tensor from the dataloader — i.e., the **multi-class `culprit` value itself** (not a pre-computed `y_anomaly`) is compared as `label==0` vs `label!=0` directly in `evaluate()`. This is evaluated independently of, but consistent with, `MainModel.forward()`'s own `y_anomaly=(fault_indexs>=1)` construction: **note the asymmetry** — `evaluate()`'s ground-truth check is `label==0` (negative) vs `label!=0` (positive), which is equivalent to `label>=1` only for non-negative labels; if a `culprit==-1` sample were ever present (none are, in the recovered Dataset C artifact per Step 1 §12), `evaluate()`'s `label!=0` would classify it as **positive/anomalous** (since `-1 != 0`), whereas `forward()`'s `fault_indexs>=1` would classify the *same* `-1` sample as **negative** (since `-1 < 1`). **These two binarization rules are not equivalent for negative label values, though they happen to agree for all label values actually observed in the recovered artifact (`{0,1,4,5}`, none negative).** This is a precise, code-derived latent inconsistency, flagged exactly as found.
- **Root-cause labels:** not evaluated anywhere in this codebase (no HR@k/NDCG@k computation exists in the released repo, despite the paper reporting such metrics for a "preliminary" root-cause extension — see §16).
- **Averaging:** none needed — this is a single binary confusion matrix per `evaluate()` call, no per-class or per-node breakdown exists in the code.
- **Thresholding:** implicit in `argmax` (§6.8's `inference()`), not a tunable probability threshold.
- **Test-set usage:** (a) periodic evaluation every `evaluation_epoch` epochs during `.fit()`, whose F1 drives checkpoint selection (§12); (b) one additional post-`.fit()` evaluation call in `full_label` mode (§13); (c) per-iteration evaluation in `active_learning` mode, feeding both `dump_scores` logging and the precision-based stopping check (§14).
- **Checkpoint selection / early stopping:** already fully detailed in §12 — training-loss-plateau early stopping (patience-based) is independent of the test set; **checkpoint selection (which weights end up in the returned model) is test-set-F1-driven**, confirmed exactly.

**Distinguishing code-level fact from consequence from provable impact (as required):**
- **Code-level fact (VERIFIED):** `best_state` is chosen by comparing `test_results["F1"]` across evaluation checkpoints, both training modes.
- **Methodological consequence (reasonable inference, not independently measured):** the reported final F1/Precision/Recall for any given run is a best-of-N-checkpoints-on-test value, which will tend to be optimistic relative to a protocol that selects checkpoints without test-set access.
- **Provable impact on published numbers:** **not established.** The recovered `training.log`/`scores.txt` (Step 1 §12, this document §2) show empirical F1≈0.92–0.93 for the logged `active_learning`-mode runs, well below the paper's published 0.9928 for Dataset C — this gap is recorded as a fact; **no causal link between the test-set-driven checkpoint selection and this specific numeric gap is asserted or provable from available evidence.**

---

## 18. Leakage / Reproduction Integrity

**Verified leakage-like behavior (code-level fact, directly observed):**
- `BaseModel.fit()` uses `test_loader` results to select `best_state`, in both `full_label` and `active_learning` modes (§12, re-confirmed here at the model-code level).
- `active_learning`-mode's per-iteration stopping check reads `test_results["Pre"]` (§14).
- Window-level train/test split leakage risk is a Dataset-C-level (Eadro preprocessing) fact, already fully documented in Step 1 §11/§14/§15 — re-cited here, not re-derived, since it is unchanged by anything in the MUAD model code itself.

**Potential methodological risk (follows logically from the above, not independently measured):**
- Reported test metrics likely overstate true generalization performance relative to a checkpoint-selection protocol without test-set access, for the reasons given in §17.
- If the AL loop's non-propagation bug (§14) is real and general (not just an artifact of `max_iter=1`), then any future `max_iter>1` faithful-reproduction run would need to explicitly decide whether to replicate this non-propagation (true faithful reproduction) or fix it (a "clean" variant, to be kept separate per §23).

**Unproven impact (cannot be established from available artifacts):**
- Whether the specific published Table I/II Dataset-C numbers (F1=0.9928) were produced under `full_label` or `active_learning` mode, at how many actual epochs, with which `hidden_dim` effectively in force, or whether the `MMClasifier` shape question (§10) was ever actually encountered/resolved in whatever run produced those numbers.
- Whether the test-set-driven checkpoint selection, specifically, is responsible for any particular fraction of the gap between the recovered log's ≈0.92–0.93 F1 and the paper's 0.9928.
- No claim is made here that the published results are invalid, and no motivation is attributed to the paper's authors for any of the discrepancies recorded in this document.

---

## 19. Tensor Shape Ledger

| Stage | Tensor | Shape | Source |
|---|---|---|---|
| Raw chunk (single) | `metrics` | `(12,10,7)` | Step 1 §12 (recovered artifact) |
| Raw chunk (single) | `traces` | `(12,10,1)` | Step 1 §12 |
| Raw chunk (single) | `logs` | `(12,15)` | Step 1 §12 |
| Batched graph | `ndata["metrics"]` | `[B*12,10,7]` | `data/dataset.py`, `data/utils.py::collate` |
| Batched graph | `ndata["traces"]` | `[B*12,10,1]` | same |
| Batched graph | `ndata["logs"]` | `[B*12,15]` | same |
| Metric GRU output | per-node metric embedding | `[B*12,64]` | `models/layers.py::GRUEncoder` |
| Trace GRU output | per-node trace embedding | `[B*12,64]` | same |
| Log Linear output | per-node log embedding | `[B*12,64]` | `models/encoders.py::LogEncoder` |
| `GraphModel1`, post-GATv2Conv | per modality | `[B*12,4,64]` (heads not yet pooled) | `models/graph_model.py`, DGL `GATv2Conv` documented output convention |
| `GraphModel1`, post-maxpool-over-heads | per modality | `[B*12,64]` | `models/graph_model.py::GraphModel1.forward` (empirically shape-traced in this investigation) |
| `GraphModel1`, post-`GlobalAttentionPooling` | per modality, graph-level | `[B,64]` | same |
| `UncertainBlock`, `encoded` | per modality | `[B,128]` | `models/main_model.py::UncertainBlock.forward` |
| `UncertainBlock`, `mu`/`var` | per modality | `[B,64]` each | same |
| `UncertainBlock`, sampled `new_<modality>` | per modality | `[B,64]` | same |
| Per-modality classifier logit | `logit_v` | `[B,2]` | `_compute_confidence_loss` |
| Per-modality confidence | `ω_v` | `[B,1]` | `main_model.py::forward` |
| Weighted per-modality embedding | `z_v * ω_v` | `[B,64]` (broadcast) | same |
| Fused feature | `feature` | `[B,192]` | `torch.cat(dim=-1)` |
| Fused-level confidence | `TCPConfidence_sig` | `[B,1]` | `sigmoid(TCPConfidenceLayer(feature))` |
| Final classifier input | `feature` fed to `MMClasifier` | `[B,192]` expected; **`[64,2]`-shaped first layer per §10's finding, unresolved shape mismatch** | `models/main_model.py::MainModel.__init__`/`forward` |
| Final logits | `MMlogit` | `[B,2]` | `MMClasifier(feature)` |
| Prediction | `y_pred` | `[B]` list, values `{0,1}` | `inference()` |
| Loss (all components) | `total_loss` | scalar | `MainModel.forward` |

---

## 20. File-by-File Implementation Map

| Repository File | Classes / Functions | Responsibility | Reproduction Importance |
|---|---|---|---|
| `main.py` | `main()` | Top-level orchestration: config load, seeding, data loading, mode branching (`full_label`/`active_learning`), training/eval calls, checkpoint save | Critical — defines actual executed control flow, including the AL non-propagation issue (§14) |
| `data/dataset.py` | `ChunkDataset`, `CombinedDataset` | Wrap chunk dicts into per-chunk DGL graphs; eager `.data` list construction | Critical — determines exactly what tensors reach the model |
| `data/utils.py` | `read_json`, `collate`, `load_data`, `create_dataloader` | Metadata/chunk loading, DGL batching, DataLoader construction (batch_size=50) | Critical |
| `models/layers.py` | `GRUEncoder`, `FullyConnected` (unused by `MainModel`), `LinearLayer` | Reusable building blocks: GRU wrapper, Xavier-initialized linear layer | High |
| `models/graph_model.py` | `GraphModel1`, `SimpleAttention` | Single-layer GATv2 + custom max-over-heads + `GlobalAttentionPooling` graph readout; learned attention gate used inside `UncertainBlock` | Critical |
| `models/encoders.py` | `MetricEncoder`, `TraceEncoder`, `LogEncoder` | Per-modality raw-feature → graph-pooled `[B,64]` embedding pipelines | Critical |
| `models/main_model.py` | `UncertainBlock`, `MainModel` | GPE-equivalent uncertainty modeling, CFM-equivalent confidence-weighted fusion, all losses, `inference()` | Critical — the core of the architecture and its documented inconsistencies |
| `models/__init__.py` | exports `MainModel` | Package export only | Low |
| `training/base_model.py` | `BaseModel` (`fit`, `evaluate`) | Training loop, Adam optimizer, loss-plateau early stopping, test-set-driven checkpoint selection, binary confusion-matrix evaluation | Critical |
| `training/trainer.py` | (empty file) | None — confirmed 0 bytes | None |
| `active_learning/strategies.py` | `entropy_based_selection`, `confidence_based_selection`, `hybrid_selection` | MC-style entropy querying, fused-confidence pseudo-labeling, combined selection | Critical for AL mode |
| `utils/general_utils.py` | `seed_everything`, `dump_params`, `dump_scores`, `get_device`, `split_data` | Seeding, logging/dumping helpers, 10%/90% initial AL split | High |
| `utils/logging_utils.py` | `setup_logging` | Logging configuration | Low |
| `configs/params.json` | — | All configuration values, several of which are dead (§15) | Critical to read correctly, including which keys are actually consumed |

---

## 21. Code-to-Paper Mapping

| Paper Concept | Repository File | Class/Function | Implementation Detail |
|---|---|---|---|
| Unimodal Feature Representation | `models/encoders.py` | `MetricEncoder`,`TraceEncoder`,`LogEncoder` | GRU(7→64)/GRU(1→64)/Linear(15→64), each followed by `GraphModel1` |
| Graph-based Probabilistic Encoder (GPE) | `models/main_model.py`, `models/graph_model.py` | `UncertainBlock` (distribution/sampling) + `GraphModel1` (the "graph-based" part, applied upstream per-modality) | Paper treats GPE as one unit combining graph attention and probabilistic modeling; code splits this across two classes, with `GraphModel1` producing the graph-level embedding that `UncertainBlock` then treats probabilistically |
| Confidence-aware Fusion Mechanism (CFM) | `models/main_model.py` | `_compute_confidence_loss`, the `TCPConfidenceLayer_*` heads, and the `torch.cat` fusion line in `forward` | See §6.5 for full equation/code correspondence and the sum-vs-concat discrepancy |
| Graph Attention Network usage | `models/graph_model.py` | `GraphModel1` using `dgl.nn.pytorch.GATv2Conv` | GATv2, not vanilla GAT; single layer; custom max-over-heads aggregation not described in paper |
| Anomaly Detector | `models/main_model.py` | `MMClasifier`, `inference()` | Binary-only; exact structure dependent on the `hidden_dim` wiring issue (§10) |
| Root-cause / culprit identity use | `models/main_model.py::forward` | `y_anomaly = (fault_indexs >= 1).long()` | Culprit collapsed to binary; no separate root-cause head exists in released code (§6.8, §16) |
| Active Training / Active Learning | `main.py`, `active_learning/strategies.py` | the `active_learning` branch of `main()`, `hybrid_selection` and its sub-functions | §14 gives the complete, exact algorithm including the non-propagation finding |
| Losses (Eq.8–10, Eq.13) | `models/main_model.py::forward`, `_compute_confidence_loss` | `total_loss = MMLoss + 0.6*confidence_loss + mean_kl_loss` | §11 gives the exact formula and the dead-coefficient findings |
| Evaluation (Precision/Recall/F1) | `training/base_model.py::evaluate` | manual TP/FP/FN/TN accumulation | §17 |

---

## 22. Faithful Reproduction Specification

To reproduce the released system as faithfully as possible, the following must be preserved **exactly as observed**, without correction:
- `hidden_dim` effectively `[64]` (not `[64,64]`) at runtime, since `main.py` never passes it through — reproduce this exact non-wiring, including whatever the actual `MMClasifier`/`feature`-dimension resolution turns out to be once empirically run end-to-end (§10's open question must be resolved by an actual run before implementation, not assumed either way).
- `loss1`, `loss2`, `aloss`/`self.k1`, `b_loss`/`self.k2` as **dead, unused values** — do not wire them in as if they were meant to be used, unless a real end-to-end run reveals they are in fact consumed somewhere not yet found.
- The exact `total_loss = MMLoss + 0.6*confidence_loss + mean_kl_loss` formula, including the hardcoded `0.6` and the implicit `×1` on both `MMLoss` and `mean_kl_loss`.
- The `reparametrize`/`kl_loss` internal `var` inconsistency (§6.4/§7) — implement both usages exactly as coded, without picking one consistent interpretation.
- Concatenation-based fusion (`torch.cat`, order trace/metric/log), not summation, even though paper prose says "Σ".
- The **test-set-driven checkpoint selection** inside `BaseModel.fit()` — explicitly included, not silently replaced with a validation-based rule.
- The **active-learning stopping/precision-check behavior**, including `max_iter=1`'s effect of running the "iterative" process exactly once as configured.
- The **AL sample-selection non-propagation** (§14) — if a faithful reproduction attempts `max_iter>1`, this exact non-propagation behavior must be preserved (i.e., `train_loader` must likewise not be rebuilt from the updated `train_dataset`/`train_keys`), unless and until further evidence (e.g. an actual successful multi-iteration run from the original authors) shows this reading is wrong.
- All observed tensor transformations (§19), including the single-channel trace input (a Dataset-C-level, not MUAD-level, compatibility fact per Step 1).
- Dataset-C-specific compatibility steps (trace-channel trim, per Step 1 §9) must be marked as reconstructed dataset-preparation steps, external to MUAD's own code, not as part of "faithful MUAD reproduction" proper.
- Do not silently improve, fix, or rationalize any of the above — including the apparent `MMClasifier` shape issue — until an actual, real end-to-end run of the unmodified repository (with correctly matched `dgl`/`torch` versions) either confirms or refutes the shape-mismatch reading in §10.

---

## 23. Clean Evaluation Specification

Kept strictly separate from §22; **not** a replacement for faithful reproduction.

Potential, explicitly separated changes for a leakage-controlled evaluation track:
- **Validation-based checkpoint selection:** carve an explicit validation split from `train_keys` (distinct from both the AL query pool and the test set) and select `best_state` using validation F1, never `test_loader`.
- **No test-set access during training:** remove the `test_loader` argument from `.fit()` entirely for the duration of training; evaluate on `test_loader` exactly once, after all training/selection decisions are finalized.
- **Grouped/episode-aware split:** at the Dataset-C level (Step 1 §11/§21), regenerate `chunk_train.pkl`/`chunk_test.pkl` with train/test assignment grouped by original fault episode rather than Eadro's flat per-chunk random shuffle — this is a Dataset-C-level change, cited here only because it materially affects what "clean evaluation" means for anything built on top of MUAD's training code.
- **Independent final test evaluation:** a single `evaluate(test_loader)` call, performed once, after the model (selected via validation) is completely finalized — no further checkpoint changes after this call.
- **Active-learning propagation fix (if desired for a clean multi-iteration variant):** rebuild `train_loader` from the updated `train_dataset` each iteration — explicitly a **modification** to the released behavior (§14's non-propagation), not a faithful-reproduction detail, and must be labeled as such if implemented.

---

## 24. Unknowns and Unverifiable Details

| Question | Why It Matters | Evidence Available | Status |
|---|---|---|---|
| Does the released code actually run end-to-end without a shape error, given the `hidden_dim`/`MMClasifier`/192-dim-feature situation (§10)? | Determines whether §22's "preserve as-is" instruction is even executable without modification | Code-level shape trace says mismatch; recovered `training.log` shows successful runs; environment in this investigation could not run the real `dgl`-dependent code end-to-end due to a `torchdata`/`dgl` version incompatibility | **UNKNOWN / UNVERIFIABLE** without a successful real run |
| What `epochs` value actually executed for the runs that produced the recovered `scores.txt`/`training.log` — is `params.json`'s `100` or the logged `/50` authoritative? | Determines exact faithful-reproduction epoch count | Both directly observed, contradictory | **UNKNOWN** |
| Which training mode (`full_label` vs `active_learning`) and which exact run produced the paper's published 0.9928 F1 for Dataset C? | Determines which exact code path "the" faithful reproduction target is | No versioning/hash link between the recovered artifact/logs and the paper's published numbers | **UNKNOWN / UNVERIFIABLE** |
| Does the `confidence_based_selection` global-index arithmetic bug (§14) actually trigger, given the real pool sizes involved? | Determines whether AL pseudo-labeling assigns correct sample identities in practice | Code pattern identified; not empirically tested against real pool sizes | **UNKNOWN**, flagged as a code-level risk only |
| Is the paper's Fig.7 λ1/λ2 sensitivity sweep backed by any code, anywhere, that was simply not included in this repository? | Determines whether `loss1`/`loss2` are meant to be wired in via an external/untracked script | No sweep code found anywhere in the repository's history | **UNKNOWN**, absence is verified, cause of absence is not |
| Does the paper's "preliminary experiment... extended MUAD with a simple classification head" for HitRate@1 root-cause localization correspond to any code anywhere (including deleted commits)? | Determines whether this is fully out-of-repo or partially present | Not found in current or recovered-deleted files; not exhaustively searched beyond the files already inventoried in §2 | **UNKNOWN / UNVERIFIABLE from files examined** |

---

## 25. Reproduction Readiness

### VERIFIED AND IMPLEMENTABLE
- Full data pipeline (§3–§5), all encoder architectures (§6.1–6.3), `GraphModel1`'s structure (§6.6), `UncertainBlock`'s exact (if internally inconsistent) computation (§6.4/§7), the confidence/fusion mechanism's exact code (§6.5/§6.7), all loss formulas as literally coded (§11), the full training loop (§12), full-label mode (§13), the active-learning algorithm as coded including its non-propagation behavior (§14), the evaluation protocol (§17), and the complete tensor shape ledger (§19) except for the one open item below.

### REQUIRES RECONSTRUCTION
- Nothing at the MUAD-model level itself requires reconstruction (unlike Step 1's raw-to-parsed_data gap) — every component of the model, training, and AL code is directly present and readable in the official repository.

### REQUIRES EXPERIMENTAL VERIFICATION
- The `hidden_dim`/`MMClasifier`/192-dim-feature shape question (§10) — requires an actual successful run of the real, unmodified repository with correctly matched `dgl`/`torch` versions to determine what genuinely happens at runtime, since this investigation's isolated-code test and the recovered training logs give contradictory signals.
- The `epochs=100` vs. logged `/50` discrepancy — requires checking whether a different, untracked `params.json` was used for the specific runs that produced the recovered logs.
- The `confidence_based_selection` global-index-arithmetic edge case (§14) — requires testing against real, non-round pool sizes.

### CURRENTLY UNPROVABLE
- Which exact run/configuration produced the paper's published Dataset-C numbers.
- Whether the paper's Fig.7 sweep or the "preliminary" root-cause-localization extension have any corresponding code at all, anywhere.
- Whether the test-set-driven checkpoint selection specifically, as opposed to other factors, explains the gap between the recovered logs' ≈0.92–0.93 F1 and the paper's 0.9928.

---

## 26. Exact Step 2 Deliverable

**What is now fully understood, source-grounded and ready to hand to an implementer:**
- Dataset forensic specification — `DATASET_FORENSICS_COMPLETE.md` (Step 1, frozen).
- MUAD model forensic specification — this document (§2–§21).
- Paper/code discrepancy register — §16 (model-level) + Step 1's own discrepancy tables (data-level).
- Tensor shape ledger — §19, with one open item (§10/§24) explicitly flagged rather than guessed.
- Code-to-math mapping — §10 (equation table), §21 (concept table).
- Faithful reproduction specification — §22.
- Clean evaluation specification — §23.

**What remains before coding begins (from §24/§25's REQUIRES EXPERIMENTAL VERIFICATION list):**
1. Obtain a working `dgl`+`torch` environment matching (or close to) the repository's pinned `requirements.txt` versions, and run the real, unmodified `main.py` end-to-end on the recovered (or a freshly regenerated, per Step 1) Dataset C chunk files.
2. Use that real run to resolve the `hidden_dim`/`MMClasifier` shape question definitively — either it runs cleanly (revealing a gap in this document's static analysis) or it fails with the predicted shape error (confirming the static analysis and requiring a decision on how "faithful reproduction" handles a repository that cannot run as-released).
3. Confirm the actual `epochs` value and training mode used for any run whose numbers are meant to be matched.

Once these three items are resolved, the combination of `DATASET_FORENSICS_COMPLETE.md` + `MUAD_MODEL_FORENSICS_COMPLETE.md` constitutes:

**Dataset forensic specification + MUAD model forensic specification + paper/code discrepancy register + tensor shape ledger + code-to-math mapping + faithful reproduction specification + clean evaluation specification = READY TO IMPLEMENT**, contingent on closing the three experimental-verification items above.
