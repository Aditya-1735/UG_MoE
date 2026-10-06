# Implementation Blueprint
## Eadro SocialNetwork → MUAD Dataset C → MUAD Model (B.Tech Final-Year Project)

**Status:** PLANNING ONLY. No code, no files, no installs, no training, no MUAD modification has occurred as part of this document.

**Inputs to this document (frozen baselines, not re-derived here):**
- `DATASET_FORENSICS_COMPLETE.md` — raw archive, fault ground truth (36 records), 12-node graph, tensor schemas, recovered chunk artifact.
- `MUAD_MODEL_FORENSICS_COMPLETE.md` — full model architecture, math, losses, training loop, active learning, discrepancy register.
- `MUAD_RUNTIME_VERIFICATION.md` — empirical proof that commit `6031ccf` runs end-to-end (with `hidden_dim=[64,64]`, `epochs=50`, `max_iter=30`) and current HEAD `58923ca` crashes (`RuntimeError: mat1 and mat2 shapes cannot be multiplied (5x192 and 64x2)`).

Any point below marked **[RE-VERIFY]** is the only kind of item that may require going back to the forensic documents; everything else is treated as settled.

---

# PART 1 — FINAL SYSTEM ARCHITECTURE

```
RAW EADRO SN ARCHIVE  (SN Dataset.zip, 7 experiments: 4 fault + 3 no-fault)
        │  Responsibility: ground truth for faults, metrics, logs, traces
        │  Source of truth: DATASET_FORENSICS_COMPLETE.md §4–§9
        ▼
[L2] RAW → parsed_data   (RECONSTRUCTED — no official converter exists)
        │  Input:  metrics/*.csv, logs.json, spans.json, SN.fault-*.json  (per experiment)
        │  Output: parsed_data/SN/{records<idx>.json, metrics<idx>/<service>.csv,
        │           logs<idx>.csv, templates.json, traces<idx>.json}
        │  Source of truth: DATASET_FORENSICS_COMPLETE.md §10 (spec only, no code existed)
        │  Status: RECONSTRUCT (we write this; label every line as reconstructed)
        ▼
[L3] EADRO PREPROCESSING   (FAITHFUL REPRODUCTION of vendored, unmodified Eadro code)
        │  codes/preprocess/{util,single_process,align}.py, pinned commit
        │  Input:  parsed_data/SN/*
        │  Output: chunks.pkl → chunk_train.pkl / chunk_test.pkl / metadata.json
        │          (metrics[12,10,7], traces[12,10,2], logs[12,15], culprit:int)
        │  Source of truth: DATASET_FORENSICS_COMPLETE.md §11
        │  Status: REUSE (vendor Eadro's own code verbatim, do not rewrite)
        ▼
[L3.5] MUAD-COMPATIBILITY TRIM   (RECONSTRUCTED, explicitly separate from Eadro's own code)
        │  traces[...,:1]  → traces[12,10,1]   (drops Eadro's always-zero 2nd channel)
        │  Source of truth: DATASET_FORENSICS_COMPLETE.md §9/§12 (undocumented, inferred from recovered artifact)
        │  Status: RECONSTRUCT, isolated in its own module, never merged into L3
        ▼
[Gate] DATASET VALIDATION   (see PART 6)
        │  Compares regenerated chunks against recovered chunk_train.pkl/chunk_test.pkl/metadata.json
        ▼
[L5] MUAD DATALOADER   (FAITHFUL REPRODUCTION, vendored from commit 6031ccf)
        │  data/dataset.py::ChunkDataset/CombinedDataset, data/utils.py::load_data/create_dataloader
        │  Input:  chunk_train.pkl / chunk_test.pkl / metadata.json
        │  Output: batched dgl.DGLGraph, labels tensor [B]
        │  Source of truth: MUAD_MODEL_FORENSICS_COMPLETE.md §4, §19
        │  Status: REUSE (vendor verbatim)
        ▼
[L7] MODALITY ENCODERS   (FAITHFUL REPRODUCTION)
        │  MetricEncoder: GRU(7,64) → [N,64]         Source: encoders.py, layers.py::GRUEncoder
        │  TraceEncoder:  GRU(1,64) → [N,64]         Source: encoders.py, layers.py::GRUEncoder
        │  LogEncoder:    Linear(15,64) → [N,64]     Source: encoders.py
        ▼
[L6] GRAPH PROCESSING (GraphModel1)   (FAITHFUL REPRODUCTION)
        │  GATv2Conv(4 heads,64) → max-pool-over-heads → GlobalAttentionPooling
        │  Input:  [N,64] per-node embedding + fixed 12-node/24-edge graph (from metadata.json)
        │  Output: [B,64] graph-level embedding, per modality
        │  Source of truth: MUAD_MODEL_FORENSICS_COMPLETE.md §6.6
        ▼
[L8] UNCERTAINTY / GPE (UncertainBlock)   (FAITHFUL REPRODUCTION, including internal μ/σ² inconsistency)
        │  Input:  [B,64] graph embedding per modality
        │  Output: [B,64] sampled embedding + scalar KL loss, per modality
        │  Source of truth: MUAD_MODEL_FORENSICS_COMPLETE.md §6.4, §7
        ▼
[L9] CONFIDENCE / CFM   (FAITHFUL REPRODUCTION)
        │  Per-modality TCP classifier + confidence layer → ω_v ∈ [B,1]
        │  Fusion: concat(z_trace·ω_trace, z_metric·ω_metric, z_log·ω_log) → [B,192]
        │  Source of truth: MUAD_MODEL_FORENSICS_COMPLETE.md §6.5, §6.7
        ▼
[L10] ANOMALY CLASSIFIER (MMClasifier)   (FAITHFUL REPRODUCTION — commit 6031ccf's hidden_dim=[64,64] wiring)
        │  Input:  [B,192]    Output: MMlogit [B,2]
        │  Source of truth: MUAD_RUNTIME_VERIFICATION.md Phase E.1 (empirically confirmed working config)
        ▼
[Prediction]  y_pred = (argmax(MMlogit) >= 1)   [B]  {0,1}
        ▼
[Evaluation]  BaseModel.evaluate(): TP/FP/FN/TN via label==0 vs label!=0 → Precision/Recall/F1
```

**Active-learning path (separate, branches off after [L5]):**
```
[L5] MUAD DATALOADER → split_data(train_ratio=0.1) → train_keys(10%), val_keys(90% pool)
        ▼
[L11] TRAINING LOOP (BaseModel.fit) on the 10% labeled train set
        │  Test-set-driven checkpoint selection (test F1 > best_f1) — FAITHFULLY REPRODUCED, not fixed
        ▼
[L12] ACTIVE LEARNING QUERY (hybrid_selection)
        │  entropy_based_selection: 5 stochastic forward passes/batch over val pool → top-n_select=200
        │  confidence_based_selection: TCPConfidence_sig > 0.9 → pseudo-labels
        ▼
[L12] POOL UPDATE
        │  train_keys.extend(...), CombinedDataset built — BUT train_loader NOT rebuilt (confirmed bug, faithfully reproduced)
        ▼
[loop] back to [L11] for next of max_iter=30 iterations (per commit 6031ccf's config)
        ▼
[Evaluation] same as above, each iteration
```

---

# PART 2 — IMPLEMENTATION LAYERS

| Layer | Name | Responsibility | Reuse / Reconstruct / Implement |
|---|---|---|---|
| L1 | Raw dataset ingestion | Read `SN Dataset.zip` contents into memory-addressable paths, no transformation | IMPLEMENT (thin, ours) |
| L2 | Raw → parsed_data reconstruction | Metrics copy, log UTC+8 correction + Drain parsing, span→trace-interval conversion, fault JSON reformatting | RECONSTRUCT |
| L3 | Eadro preprocessing | Windowing, normalization, chunking, split | REUSE (vendored, unmodified) |
| L3.5 | MUAD compatibility trim | `traces[...,:1]` | RECONSTRUCT (isolated) |
| L4 | Dataset validation | Gates 1–10 (PART 6) | IMPLEMENT (ours) |
| L5 | MUAD dataset adapter | `ChunkDataset`/`CombinedDataset`/`DataLoader` | REUSE (vendored) |
| L6 | Graph construction | `GraphModel1` (GATv2 + pooling) | REUSE (vendored) |
| L7 | Modality encoders | `MetricEncoder`/`TraceEncoder`/`LogEncoder`/`GRUEncoder` | REUSE (vendored) |
| L8 | GPE / uncertainty | `UncertainBlock` | REUSE (vendored, including its internal inconsistency) |
| L9 | CFM / confidence | TCP classifiers, confidence layers, fusion concat | REUSE (vendored) |
| L10 | Multimodal classifier | `MMClasifier`, `inference()` | REUSE (vendored, with `hidden_dim=[64,64]` wiring restored per commit `6031ccf`) |
| L11 | Training | `BaseModel.fit`/`evaluate` | REUSE (vendored, including test-set checkpoint selection) |
| L12 | Active learning | `hybrid_selection`, `main.py`'s AL branch | REUSE (vendored, including non-propagation bug) |
| L13 | Evaluation | Precision/Recall/F1, dataset-vs-artifact regression comparison | IMPLEMENT (ours, wraps L11's evaluate + comparison logic) |
| L14 | Experiment management | Config, seeds, naming, logging, checkpoints | IMPLEMENT (ours) |
| L15 | Visualization / demo | Plots, confusion matrices, uncertainty visualizations, a small demo script | IMPLEMENT (ours, optional/extension) |

This decomposition matches the original 15-layer suggestion closely, with one addition (**L3.5**, isolated because Step 1/2 established it as a real but undocumented transformation that must never be silently folded into "faithful Eadro reproduction" at L3) and L13 widened to explicitly include the dataset-artifact regression comparison (PART 7), since that is as important as model evaluation for this project's goals.

---

# PART 3 — PROJECT DIRECTORY STRUCTURE

```
project/
├── README.md                          # ours — project overview, how to run each track
├── docs/
│   ├── DATASET_FORENSICS_COMPLETE.md  # carried over, frozen reference
│   ├── MUAD_MODEL_FORENSICS_COMPLETE.md
│   ├── MUAD_RUNTIME_VERIFICATION.md
│   └── IMPLEMENTATION_BLUEPRINT.md    # this document
│
├── vendor/
│   ├── eadro/                         # git submodule or pinned-commit copy of BEbillionaireUSD/Eadro
│   │   └── codes/preprocess/{util,single_process,align}.py   # UNMODIFIED, original reproduction code
│   └── muad/                          # git submodule or pinned-commit copy of slg-frank/muad @ 6031ccf
│       ├── main.py, data/, models/, training/, active_learning/, utils/, configs/   # UNMODIFIED, original reproduction code
│
├── raw_data/
│   └── SN Dataset/                    # extracted raw archive, read-only, never modified
│
├── preprocessing/                     # ours — L2, L3.5
│   ├── raw_to_parsed/
│   │   ├── convert_metrics.py         # RECONSTRUCTED
│   │   ├── convert_logs.py            # RECONSTRUCTED (UTC+8 fix + Drain)
│   │   ├── convert_traces.py          # RECONSTRUCTED (Jaeger span → interval)
│   │   ├── convert_faults.py          # RECONSTRUCTED (fault JSON → records format)
│   │   └── run_conversion.py          # ours — orchestrates the above
│   ├── run_eadro_preprocessing.sh     # ours — invokes vendor/eadro/codes/preprocess/align.py unmodified, with --test_ratio 0.4
│   └── muad_compat/
│       └── trim_trace_channel.py      # RECONSTRUCTED, isolated, clearly labeled non-original
│
├── validation/                        # ours — L4, PART 6/7
│   ├── gates/
│   │   ├── gate01_raw_integrity.py
│   │   ├── gate02_fault_manifest.py
│   │   ├── gate03_parsed_data_schema.py
│   │   ├── gate04_chunk_schema.py
│   │   ├── gate05_tensor_dims.py
│   │   ├── gate06_labels.py
│   │   ├── gate07_graph_structure.py
│   │   ├── gate08_split.py
│   │   ├── gate09_artifact_comparison.py
│   │   └── gate10_dataloader_compat.py
│   └── run_all_gates.py
│
├── experiments/                       # ours — L14
│   ├── configs/
│   │   ├── faithful_full_label.json
│   │   ├── faithful_active_learning.json
│   │   ├── clean_evaluation.json
│   │   ├── smoke_test.json
│   │   └── ablation_*.json
│   ├── runs/                          # output dir, gitignored — checkpoints, logs, metrics per run
│   └── run_experiment.py              # ours — thin CLI wrapper choosing a track + config
│
├── tracks/
│   ├── faithful_reproduction/         # ours — orchestration only, imports vendor/muad unmodified
│   │   ├── run_full_label.py
│   │   └── run_active_learning.py
│   └── clean_evaluation/              # ours — a deliberately separate copy/subclass, PART 12 changes only, never merged into faithful_reproduction
│       ├── validation_checkpoint.py
│       └── run_clean.py
│
├── ablations/                         # ours — L15's scientific variants, PART 15
│   ├── metric_only.py
│   ├── trace_only.py
│   ├── log_only.py
│   ├── deterministic_multimodal.py    # no GPE, no CFM
│   ├── multimodal_with_graph.py
│   ├── multimodal_with_uncertainty.py # +GPE, no CFM
│   └── multimodal_with_confidence.py  # +GPE +CFM (= full MUAD, path already covered by faithful_reproduction)
│
├── tests/                             # ours — PART 13
│   ├── unit/
│   ├── integration/
│   ├── regression/
│   └── e2e/
│
├── scripts/                           # ours — small utility CLIs
│   ├── inspect_chunk.py
│   ├── compare_to_recovered_artifact.py
│   └── env_report.py
│
├── notebooks/                         # ours — exploratory only, not part of any pipeline
│
├── requirements-repro.txt             # ours — the empirically-verified compatible environment
│                                       #        (torch==2.2.1, dgl==2.1.0, torchdata==0.7.1, numpy<2, pydantic)
│                                       #        per MUAD_RUNTIME_VERIFICATION.md Phase A/B
└── environment_notes.md               # ours — full record of every deviation from vendor/muad's own requirements.txt
```

**Per-item classification, condensed:**
- **REUSE (original reproduction code, unmodified):** everything under `vendor/`.
- **RECONSTRUCTED code (clearly labeled, non-original):** everything under `preprocessing/raw_to_parsed/`, `preprocessing/muad_compat/`.
- **Project-specific code (ours):** `validation/`, `experiments/`, `tracks/`, `ablations/`, `scripts/`, `tests/`, `README.md`, `environment_notes.md`.
- **Configuration:** `experiments/configs/*.json`, `requirements-repro.txt`.
- **Test code:** `tests/**`.

**DO NOT CREATE THESE FILES YET** — this is the target structure only.

---

# PART 4 — RAW → parsed_data IMPLEMENTATION PLAN

All four sub-plans below are **RECONSTRUCTED**, per `DATASET_FORENSICS_COMPLETE.md` §10 — no official Eadro code for this stage exists anywhere.

### Metrics
- **INPUT:** `data/SN.<ts>/metrics/<service>.csv` and `no fault/SN.<ts>/metrics/<service>.csv`, header `timestamp,cpu_usage_system,cpu_usage_total,cpu_usage_user,memory_usage,memory_working_set,rx_bytes,tx_bytes`, 1 row/second, Unix epoch integer timestamps. (§7 of dataset doc)
- **OUTPUT:** `parsed_data/SN/metrics<idx>/<service>.csv` — same schema, same 12 service filenames (with the `nginx-web-server` name, already consistent between raw metrics filename and Eadro's node list — §6).
- **TRANSFORMATION:** copy/rename only; no numeric change (raw schema already matches what `single_process.py::deal_metrics` expects to read).
- **VALIDATION:** row count per file ≈ experiment duration in seconds (±2, allowing for header/edge rounding); all 12 expected service filenames present per experiment; no missing timestamps beyond a documented tolerance.

### Logs
- **INPUT:** `data/SN.<ts>/logs.json` — JSON object keyed by 12 service names, each a list of raw strings `[YYYY-Mon-DD HH:MM:SS.ffffff] <level>: (file:line:func) message`, **local time UTC+8** (§8).
- **OUTPUT:** `parsed_data/SN/templates.json` (event template list) + `parsed_data/SN/logs<idx>.csv` (rows: `timestamp,service,event`), matching what `single_process.py::deal_logs` reads.
- **TRANSFORMATION:**
  1. Parse the bracketed timestamp from each raw string.
  2. Convert from local UTC+8 to Unix epoch UTC (confirmed exact 8-hour offset, §8) — **mandatory step, absent from any released code.**
  3. Run Drain (or an equivalent template-mining parser) on the message portion to assign each line an event-template id, building `templates.json` incrementally.
  4. Emit one row per log line: `(epoch_timestamp, service_name, event_id)`.
- **VALIDATION:** every service key in `logs.json` maps to one of the 12 known node names (no unmapped services); converted timestamps fall within the experiment's `[start,end]` window (from the matching `SN.fault-*.json`); template count is stable/reasonable (cross-checked against the recovered artifact's `event_num=15` as a target, not a hard requirement, since our own Drain run may not reproduce exactly 15 templates — see PART 7's "expected difference" category).

### Traces
- **INPUT:** `data/SN.<ts>/spans.json` — list of Jaeger trace objects, each with `spans[]` (microsecond `startTime`/`duration`, per-trace-local `processID`) and a `processes` map resolving `processID→serviceName` (§9).
- **OUTPUT:** `parsed_data/SN/traces<idx>.json` — dict keyed by absolute-integer-second timestamp string, valued by `{edge_key: [latencies]}`, matching `single_process.py::deal_traces`'s expected input.
- **TRANSFORMATION:**
  1. For each trace, resolve every span's `processID → serviceName` via that trace's own `processes` dict, applying the `nginx-thrift↔nginx-web-server` alias explicitly (§6/§9).
  2. Convert `startTime` (microseconds) to an integer-second bucket (floor division by 1,000,000).
  3. Reconstruct caller→callee edges from each span's `references` (`CHILD_OF`) field, mapping to `edge_key` strings consistent with the fixed 12-node graph's edge naming.
  4. Aggregate per-second-bucket, per-edge latency lists (append `duration` in whatever unit `deal_traces` expects — milliseconds, per its own internal aggregation of "average delay"; confirm unit during implementation against `single_process.py`'s exact arithmetic, not assumed here).
- **VALIDATION:** every span's resolved service name is one of the 12 known nodes; every second-bucket key falls inside the experiment window; edge keys correspond to edges actually present in the fixed 24-edge graph (spans resolving to an edge outside this list are a signal to investigate, not silently drop).

### Faults
- **INPUT:** `data/SN.fault-*.json` / `no fault/SN.fault-*.json` — `{"start":..., "faults":[{"name":..., "fault":..., "start":..., "duration":...}], "end":...}` (§5).
- **OUTPUT:** `parsed_data/SN/records<idx>.json` — matching `align.py::get_basic`'s expected `{"faults":[{"s":..., "e":..., "service":...}], ...}` schema.
- **TRANSFORMATION:** `s = fault["start"]`; `e = fault["start"] + fault["duration"]`; `service` = the fault's `name` field with the `socialnetwork-` prefix / `-1` suffix stripped and the `nginx-thrift→nginx-web-server` alias applied.
- **VALIDATION:** every one of the 36 raw fault records (§5.1) is accounted for exactly once in the converted output — a direct count check against the known-good total established in the dataset forensic doc; every resolved `service` is one of the 12 known node names; `no fault/` experiments produce a `records<idx>.json` with an empty `faults` list, preserved as genuinely fault-free (not silently dropped, per the unresolved §12/§17 tension in the dataset doc — this conversion must not itself introduce or erase that tension).

**Nothing above may be described as original Eadro behavior anywhere in code comments, docstrings, or logs — every module in `preprocessing/raw_to_parsed/` must carry an explicit "RECONSTRUCTED, not part of the official Eadro release" header.**

---

# PART 5 — EADRO PREPROCESSING IMPLEMENTATION PLAN

| Step | Exact original behavior | Source (vendored) | Our module | Validation test |
|---|---|---|---|---|
| Node ordering | Fixed 12-node list, `util.py`'s `Info` class | `vendor/eadro/codes/preprocess/util.py` | none — reused as-is | Gate 7: loaded node list == documented 12-node table |
| Metric normalization | z-score, per-service per-metric, over that service's whole experiment file | `single_process.py::deal_metrics` | none — reused as-is | Gate 5: output `[*,12,10,7]`; spot-check mean≈0/std≈1 per service per metric across a full experiment |
| Trace processing | Per-node mean latency → channel 0 only of `[*,12,10,2]`; channel 1 always zero; per-node z-score over experiment | `single_process.py::deal_traces` | none — reused as-is | Gate 5: channel 1 is exactly zero everywhere (confirming we haven't "fixed" it) |
| Log processing | Drain-parsed events → per-window HawkesADM4(decay=3) fit → intensity vector | `single_process.py::deal_logs` | none — reused as-is (depends on our L2's `templates.json`/`logs<idx>.csv`) | Gate 5: output `[*,12,event_num]` |
| Window length | `chunk_lenth=10` (seconds) | `align.py` default arg | none — invoke with default | Gate 5: window covers exactly 10 seconds |
| Stride | 1 second, via `range(start, end-chunk_lenth+1)` | `align.py::get_basic` | none — reused as-is | Gate 5: consecutive chunk start-times differ by 1s (only checkable if we retain this info ourselves during generation, before Eadro's own randomized chunk-ID step discards it — see PART 6 Gate 4 note) |
| Label / fault overlap rule | `threshold=1` — 1s overlap suffices to label the window anomalous, attributed to that fault's service | `align.py::get_basic` default | none — invoke with default | Gate 6: spot-check a handful of known fault intervals produce the expected anomalous-window count |
| Chunk creation | Random 8-char alphanumeric chunk ID (`get_chunkid()`), 4-key dict (`metrics/traces/logs/culprit`) | `align.py::get_chunks` | none — reused as-is | Gate 4: schema matches recovered artifact's key set exactly |
| Graph metadata | `node_num`, `edges`, `event_num`, `metric_num`, `chunk_lenth`, `chunk_num` written to `metadata.json` | `align.py::get_all_chunks` | none — reused as-is | Gate 4: matches recovered `metadata.json` field-for-field where not dataset-instance-dependent (`node_num=12`, `metric_num=7` must match exactly; `event_num`, `chunk_num` may differ — see PART 7) |
| Train/test split | `np.random.shuffle` over flat chunk-index list, **default `test_ratio=0.3`**, overridable via CLI | `align.py::split_chunks` | our `run_eadro_preprocessing.sh` invokes with **`--test_ratio 0.4`** explicitly (matches both papers and the recovered artifact's empirical 60/40 split — §11/§12 of dataset doc) | Gate 8: resulting split ≈60/40 |
| Randomization | Uses whatever RNG state is active at invocation time (no explicit seed call inside `align.py` itself, confirmed by reading the file — **[RE-VERIFY if not already covered]**: `DATASET_FORENSICS_COMPLETE.md` did not explicitly confirm the presence/absence of a seed call inside `align.py`; this must be checked against the vendored source directly during implementation, not assumed here) | `align.py` | our `run_eadro_preprocessing.sh` sets `PYTHONHASHSEED`/`numpy.random.seed`/`random.seed` explicitly before invocation, documented as an addition for reproducibility, not a behavior change | Gate 8 note: two runs with the same explicit seed produce the same split; this is a reproducibility aid we add, not a claim about the original's exact historical randomness |
| Serialization | `pickle.dump` of the chunk dict, `json.dump` of metadata | `align.py` | none — reused as-is | Gate 4: files load via plain `pickle.load`/`json.load` |

---

# PART 6 — DATASET VALIDATION GATES

| Gate | Checks | Method | Expected Result | Failure Action | Blocks Later Phases? |
|---|---|---|---|---|---|
| 1. Raw dataset integrity | Archive extracts to the documented 7-experiment structure (4 fault + 3 no-fault); `SN-all.tgz` redundancy holds | File-listing diff against `DATASET_FORENSICS_COMPLETE.md` §4's documented tree | Exact structural match | STOP — do not proceed to L2 | Yes, blocks everything |
| 2. Fault manifest correctness | Raw fault record count == 36; type/service distribution matches §5's table | Direct count from our L2 fault converter's input parsing | 36 records, 12/12/12 type split, 3/service | STOP — investigate before generating any chunks | Yes, blocks L3 |
| 3. `parsed_data` schema correctness | Our reconstructed `records*.json`/`metrics<idx>/*.csv`/`logs<idx>.csv`/`templates.json`/`traces<idx>.json` match the field names/types `single_process.py`/`align.py` actually read (verified by static inspection of the vendored code, not assumption) | Schema-diff script against the vendored Eadro source's actual field accesses | No `KeyError`/schema mismatch when Eadro's own code reads our output | STOP — fix L2 before running L3 | Yes, blocks L3 |
| 4. Eadro chunk schema correctness | Our generated `chunk_train.pkl`/`chunk_test.pkl` have exactly the 4 keys (`metrics`,`traces`,`logs`,`culprit`) with correct dtypes | Direct `pickle.load` + key/dtype assertion | Matches recovered artifact's schema exactly | STOP | Yes, blocks L3.5/L5 |
| 5. Tensor dimensions | `metrics[12,10,7]`, `traces[12,10,2]` (pre-trim, Eadro-native) | Shape assertion on a sample of generated chunks | Exact match | STOP | Yes |
| 6. Labels | `culprit` ∈ valid range (`-1` or `0..11`); every fault-derived label traces back to one of the 36 known fault records | Cross-reference generated chunk labels against the fault manifest (Gate 2's output) | No orphan/unexplained label values | STOP — investigate before proceeding | Yes |
| 7. Graph structure | `node_num=12`, `edges` == the documented 24-edge list, deterministic ordering | Direct comparison against `DATASET_FORENSICS_COMPLETE.md` §6/§12's edge table | Exact match | STOP | Yes |
| 8. Train/test split | Split ratio ≈60/40 (matching recovered artifact within reasonable sampling variance, since exact chunk membership will differ due to randomization) | Count assertion | ~60/40, not exactly 3366/2244 (those exact counts are dataset-instance-specific, see PART 7) | WARN if far from 60/40 (e.g. >5pp off with a fixed seed) — investigate, do not silently proceed | Yes, but with a documented tolerance |
| 9. Comparison against recovered MUAD artifact | Full PART 7 comparison suite | See PART 7 | Category-classified match/difference | WARN and log all differences; STOP only for "Investigation required" category items | Partially — "exact match" failures block, "expected difference" items do not |
| 10. DataLoader compatibility | Vendored `ChunkDataset`/`create_dataloader` (unmodified) successfully loads our generated chunks and produces correctly-shaped batches | Run the actual vendored `data/utils.py::load_data` + `ChunkDataset` + `create_dataloader` against our output, exactly as `MUAD_RUNTIME_VERIFICATION.md` Phase C did against the recovered artifact | Batched graph/labels shapes match Phase C's documented output exactly | STOP | Yes, blocks L6 onward |

**Rule, stated explicitly per the task's instruction:** if any gate marked "Blocks Later Phases? Yes" fails, no later-numbered milestone (PART 18) may begin, and no model-facing code (L6 onward) may run against the affected dataset. Gate 9's sub-items are the only ones permitted to be "expected differences" rather than hard blockers, and only for the specific categories enumerated in PART 7.

---

# PART 7 — RECOVERED DATASET COMPARISON

**Comparison target:** the recovered `chunk_train.pkl`/`chunk_test.pkl`/`metadata.json` documented in `DATASET_FORENSICS_COMPLETE.md` §12.

| Property | Category | Rationale |
|---|---|---|
| `node_num` (=12) | **Exact match required** | Fixed by the raw archive's own instrumentation scope (§6); not dataset-generation-dependent |
| `edges` (24-edge list) | **Exact match required** | Fixed graph topology from Eadro's own hardcoded `util.py` |
| `metric_num` (=7) | **Exact match required** | Fixed by raw metrics CSV schema |
| `metrics` tensor shape `(12,10,7)` | **Exact match required** | Structural, not value-dependent |
| `traces` tensor shape, Eadro-native `(12,10,2)` pre-trim | **Exact match required** (structural) | Fixed by `single_process.py::deal_traces`'s allocation |
| `traces` channel 1 == 0 everywhere | **Exact match required** | Direct consequence of Eadro's own code never writing that channel — a real regression here would indicate our L3 reproduction is wrong |
| `logs` tensor shape `(12,15)` | **Expected difference** (dimension itself may differ) | `15` is this specific dataset's Drain-derived template count in the recovered artifact; our own Drain run may legitimately produce a different template count depending on parser configuration/version — the *shape rule* (`[12, event_num]`) is exact-match; the *value* of `event_num` is an expected difference unless we deliberately match parser settings |
| `culprit` value range (`{-1,0..11}` in principle) | **Investigation required** if our generation ever produces values outside the fault-manifest-derived set (Gate 6); **expected difference** vs. the recovered artifact's specific observed subset `{0,1,4,5}`, since that subset reflects one prior partial/experimental run, not a documented invariant (§12/§17's own unresolved tension) |
| `chunk_num`, exact 3366/2244 split counts | **Expected difference** | Depends on `chunk_lenth`/`threshold`/RNG state at generation time; only the ~60/40 *ratio* is exact-match-required (Gate 8) |
| Presence of `culprit==-1` chunks | **Investigation required, not silently resolved either way** | The recovered artifact has none, despite genuine no-fault raw recordings existing (§12/§17). If our regeneration *does* produce `-1` chunks, this is not treated as "our reproduction is wrong" — it is treated as new evidence bearing on the unresolved §17 tension, to be reported, not silently reconciled in either direction |
| Individual tensor *values* (e.g., a specific chunk's metric readings) | **Expected difference** | No two independent runs of Eadro's random-shuffle-based `split_chunks` will assign identical chunk IDs to identical windows; per-window numeric values should be *distributionally* similar (same normalization procedure) but will not be bit-identical to the recovered artifact's specific instance |
| Normalization behavior (z-score, per-service/per-node) | **Exact match required** (procedure, not resulting values) | This is a code-behavior check, not a data-identity check — verified by confirming our vendored L3 code path is bit-identical to the historical Eadro source, not by comparing numbers to the recovered pickle |

**"Dataset regression test" concept:** a single script (`tests/regression/test_dataset_regression.py`) that:
1. Loads both the recovered artifact and our freshly-generated chunks.
2. Asserts every "Exact match required" property.
3. Logs (does not fail on) every "Expected difference" property, printing both values for human review.
4. Raises a loud, explicit, non-silent warning for every "Investigation required" property, halting any downstream milestone until a human has reviewed and recorded a decision in `docs/` (not auto-resolved by the test itself).

---

# PART 8 — MUAD IMPLEMENTATION ORDER

The forensic findings (specifically `MUAD_RUNTIME_VERIFICATION.md`'s empirical proof that the model only runs end-to-end under commit `6031ccf`'s exact configuration) dictate that **we vendor, not reimplement, every model-internal component** — there is no scientific value in hand-rewriting `GraphModel1`/`UncertainBlock`/`MainModel` from scratch when the original source is fully available and its exact runtime behavior has already been empirically nailed down. The "implementation order" below is therefore an **integration and verification order**, not a from-scratch build order — each phase's job is to prove, incrementally, that our environment + our regenerated dataset correctly drive the vendored, unmodified code, before trusting a full run.

| Phase | Module(s) | Source being reproduced | Input | Output | Shape | Objective | Unit test | Integration test | Exit criterion |
|---|---|---|---|---|---|---|---|---|---|
| A | `data/utils.py`, `data/dataset.py` | Vendored, unmodified | Our regenerated `chunk_train.pkl` etc. (post-Gate-10) | Batched `DGLGraph`, `labels` | `[B*12,10,7]` etc. | Confirm our dataset drives the official loader identically to how it drove the recovered artifact | — | Reruns `MUAD_RUNTIME_VERIFICATION.md` Phase C exactly, against our own generated data | Shapes match Phase C's documented output |
| B | `models/encoders.py::MetricEncoder` (isolated) | Vendored | `ndata["metrics"]` | `[B,64]` | `[N,10,7]→[N,64]→[B,64]` | Confirm metric-only path runs in isolation | Forward-pass shape assertion | — | No exception, correct shape |
| C | `models/encoders.py::TraceEncoder` (isolated) | Vendored | `ndata["traces"]` (post-L3.5 trim, `(*,10,1)`) | `[B,64]` | same pattern | Confirm trace-only path runs with the trimmed channel | Forward-pass shape assertion | — | No exception, correct shape |
| D | `models/encoders.py::LogEncoder` (isolated) | Vendored | `ndata["logs"]` | `[B,64]` | `[N,15]→[N,64]→[B,64]` | Confirm log-only path runs | Forward-pass shape assertion | — | No exception, correct shape |
| E | `models/graph_model.py::GraphModel1` (already exercised inside B/C/D, but tested standalone too) | Vendored | `[N,64]` + fixed graph | `[B,64]` | as documented in §6.6 | Confirm GATv2+maxpool+GlobalAttentionPooling shape trace matches `MUAD_MODEL_FORENSICS_COMPLETE.md` §19 exactly, using our own environment's dgl version | Shape assertion at each internal step (heads dim, post-maxpool, post-pooling) | — | Matches documented shape ledger |
| F | `models/main_model.py::UncertainBlock` (isolated) | Vendored | `[B,64]` | `[B,64]` sample + scalar KL | as in §6.4 | Confirm reparameterization + KL run without error; confirm stochasticity (two calls differ) | Shape + stochasticity assertion | — | Two forward calls on identical input produce different output (mirrors Runtime doc Phase G) |
| G | Per-modality `TCPClassifierLayer_*`/`TCPConfidenceLayer_*` (isolated) | Vendored | `[B,64]` | logit `[B,2]`, confidence `[B,1]` | as in §6.8 | Confirm per-modality classifier/confidence heads run | Shape assertion | — | No exception |
| H | Fusion (`torch.cat` + `TCPConfidenceLayer`) | Vendored | 3×`[B,64]` weighted | `[B,192]` | as in §6.5/6.7 | Confirm concatenation order (trace,metric,log) and dimension | Shape + order assertion | — | Matches §6.7's documented order exactly |
| I | Full `MainModel` — **constructed with the historically-attested `hidden_dim=[64,64]` kwarg**, per Runtime doc Phase E.1 | Vendored `models/main_model.py`, instantiated via our own wrapper that explicitly restores the dropped kwarg | Batched graph + labels | `MMlogit[B,2]`, `loss` (scalar), `y_pred[B]` | full §19 ledger | Reproduce Runtime doc Phase E's success case end-to-end, on our own regenerated dataset | Full forward-pass shape + finiteness assertion | Reruns Runtime doc Phase E/F/G exactly | Forward+backward succeed, loss finite, gradients nonzero |
| J | `training/base_model.py::BaseModel.fit`/`evaluate` | Vendored, unmodified | `train_loader`,`test_loader` | `best_f1`,`best_epoch`, checkpoint | — | Reproduce Runtime doc Phase H's checkpoint-selection trace on our own data | — | Reruns Phase H's minimal-epoch trace | Test-set-F1-driven `best_state` selection observed exactly as documented |
| K | `active_learning/strategies.py`, `main.py`'s AL branch | Vendored, unmodified | `val_loader` | selected indices, pseudo-labels | — | Reproduce Runtime doc Phase I's non-propagation finding on our own data | — | Reruns Phase I's `id()`/length check | Same non-propagation behavior confirmed (or, if our environment somehow differs, this becomes a **new** forensic finding requiring investigation, not a silent fix) |

**Critical constraint carried through every phase:** `MainModel`/`BaseModel` must always be constructed via **our own thin wrapper** that explicitly passes `hidden_dim=[64,64]` (documented as "restoring commit `6031ccf`'s historically-attested configuration," never as "fixing a bug"). Constructing via current-HEAD `58923ca`'s exact call signature (no `hidden_dim` kwarg) remains available as a **separate, clearly labeled negative-control test** confirming the crash still reproduces in our environment too — this is valuable evidence, not something to avoid.

---

# PART 9 — MUAD MODEL IMPLEMENTATION DETAILS (COMPONENT CHECKLIST)

Every row: **REUSE (vendored, unmodified)** — none of these are redesigned. "Implementation location in our project" means *the wrapper/test that exercises it*, not a reimplementation.

| Component | Exact mathematical behavior | Tensor dimensions | Original source | Our exercising location | Test required |
|---|---|---|---|---|---|
| `GRUEncoder` | `nn.GRU(in,64,batch_first=True,bidirectional=False)`, return final hidden state | `[N,10,in]→[N,64]` | `models/layers.py` | Phase B/C | shape + determinism-given-fixed-seed |
| `MetricEncoder` | GRU(7,64) → `GraphModel1` | `[N,10,7]→[N,64]→[B,64]` | `models/encoders.py` | Phase B | shape |
| `TraceEncoder` | GRU(1,64) → `GraphModel1` | `[N,10,1]→[N,64]→[B,64]` | `models/encoders.py` | Phase C | shape (post-L3.5 trim only) |
| `LogEncoder` | Linear(15,64), no activation → `GraphModel1` | `[N,15]→[N,64]→[B,64]` | `models/encoders.py` | Phase D | shape |
| `GraphModel1` | 1× `GATv2Conv`(4 heads,64,neg_slope=0.2,attn_drop=0,allow_zero_in_degree=True) → custom max-pool over heads → `GlobalAttentionPooling(Linear(64,1))` | `[N,64]→[N,4,64]→[N,64]→[B,64]` | `models/graph_model.py` | Phase E | shape at every internal step |
| GATv2 layer specifics | 4 heads, `negative_slope=0.2`, `attn_drop=0` (never overridden anywhere in the call chain) | `fc_src`/`fc_dst`: `Linear(64,256)` each (4×64) | `models/graph_model.py` (dgl library internals not our code) | Phase E, Phase D of Runtime doc (module printout) | printed-module comparison against Runtime doc's Phase D output |
| Global pooling | `GlobalAttentionPooling`, learned gate `Linear(64,1)`, per-graph weighted sum | `[N,64]→[B,64]` | `models/graph_model.py` (dgl library) | Phase E | shape |
| `UncertainBlock` | `Linear(64,128)→LayerNorm→Tanh` → `[fc_mu,fc_var]` each `Linear(128,64)→Sigmoid` → `SimpleAttention`×2 → reparameterize | `[B,64]→[B,128]→[B,64]×2→[B,64]` | `models/main_model.py` | Phase F | shape + stochasticity (two calls differ) |
| Reparameterization | `mu + eps*std`, `std=var.sqrt()`, `eps~N(0,I)` | `[B,64]` | `models/main_model.py::UncertainBlock.reparametrize` | Phase F | statistical sanity (sampled values vary run-to-run under fixed mu/var, fixed seed only fixes the eps draw sequence) |
| KL divergence | `-0.5*mean(sum(1+var-mu²-var.exp(),dim=-1))` — note: uses the *same* `var` as `reparametrize`'s `std=var.sqrt()`, an internal inconsistency, reproduced exactly, not corrected | scalar | `models/main_model.py::UncertainBlock.kl_loss` | Phase F | numeric formula unit test (feed known mu/var, assert exact formula output) |
| TCP | `gather(softmax(logit),1,y)` | `[B,2]→[B]` | `models/main_model.py::_compute_confidence_loss` | Phase G | unit test against hand-computed softmax+gather on a toy example |
| Confidence layers (per-modality) | `sigmoid(Linear(64,1)(z_v))` | `[B,64]→[B,1]` | `models/main_model.py` | Phase G | shape |
| Weighted fusion | `cat(z_trace*ω_trace, z_metric*ω_metric, z_log*ω_log, dim=-1)` — **concatenation, order trace/metric/log** | 3×`[B,64]→[B,192]` | `models/main_model.py::forward` | Phase H | shape + order assertion (perturb one modality, confirm which 64-column block changes) |
| `MMClasifier` | With `hidden_dim=[64,64]`: `Linear(192,64)→ReLU→Linear(64,2)` (historically-attested, Runtime doc Phase E.1) | `[B,192]→[B,64]→[B,2]` | `models/main_model.py` | Phase I | shape; **also** a negative-control test confirming the default (no `hidden_dim` kwarg) reproduces the exact crash from Runtime doc Phase E |
| Total loss | `MMLoss + 0.6*confidence_loss + mean_kl_loss` where `confidence_loss=Σ_v(CE_v+MSE_v)`, `mean_kl_loss=(KL_t+KL_m+KL_l)/3` | scalar | `models/main_model.py::forward` | Phase I | unit test: hand-compute the total from returned intermediate values (if exposed) or via a controlled toy forward pass, assert exact numeric match to the formula |

---

# PART 10 — KNOWN SOURCE-LEVEL ISSUES: IMPLEMENTATION DECISION TABLE

| Issue | Original Behavior | Faithful Reproduction Decision | Clean-Evaluation Decision | Patch Needed? | Patch Location | Documentation Requirement |
|---|---|---|---|---|---|---|
| 1. Current HEAD `58923ca` `hidden_dim`/`MMClasifier` crash | `main.py` never passes `hidden_dim`; `MMClasifier` becomes `Linear(64,2)`, incompatible with the 192-dim fused feature → `RuntimeError` | **Not used as the reproduction baseline at all.** Reproduced only as a documented negative-control test (Part 9's `MMClasifier` row) | N/A — not a methodological issue, a broken commit | No patch to `58923ca` itself; we simply don't build on it | — | Test explicitly documents "this reproduces the known HEAD regression, confirming our environment matches the one used in `MUAD_RUNTIME_VERIFICATION.md`" |
| 2. Historical `6031ccf` succeeds | `hidden_dim=[64,64]` explicitly passed → `MMClasifier`=`Linear(192,64)→ReLU→Linear(64,2)`, shape-correct | **This is our faithful-reproduction baseline**, restored via our own thin wrapper that passes this kwarg explicitly (never edited into `vendor/muad`'s files) | Same baseline, further modified per Part 12 | Yes — but it is a **wrapper-level constructor-argument restoration**, not a source edit | `tracks/faithful_reproduction/` (our orchestration code, imports vendor unmodified) | Every run log/config must state explicitly: "hidden_dim=[64,64] restored via wrapper, matching commit 6031ccf; vendor/muad source files are unmodified" |
| 3. `UncertainBlock` variance interpretation mismatch (`var` used as raw variance in `reparametrize`, as log-variance in `kl_loss`) | Internally inconsistent, both readings coexist in the same class | **Reproduce exactly, both readings, unmodified** | **Not changed even in clean_evaluation** — this is a modeling-math issue, not an evaluation-protocol issue; correcting it would be a Part-15-style ablation/extension, not "clean evaluation" | No | N/A (vendored as-is) | Every doc referencing model math must footnote this exact inconsistency, per Part 9's KL row |
| 4. Paper fusion formula (`Σωvzv`) vs. code concatenation | Code concatenates, paper prose implies weighted sum | **Reproduce code behavior (concatenation)** | Unchanged — this is a paper-vs-code discrepancy, not a leakage/evaluation-protocol issue | No | N/A | Documented in Part 9's fusion row and in `tracks/faithful_reproduction/README` |
| 5. Active-learning `train_loader` not rebuilt after dataset update | Newly-selected/pseudo-labeled samples tracked in `train_keys`/`CombinedDataset` but never fed to the next `.fit()` call | **Reproduce exactly** — `tracks/faithful_reproduction/run_active_learning.py` must exhibit the identical non-propagation, verified against Runtime doc Phase I | **This is exactly the kind of thing `clean_evaluation` may fix** — Part 12 lists "propagation fix" as an explicit, separately-implemented, separately-labeled variant | Yes, but **only in `tracks/clean_evaluation/`**, never in `tracks/faithful_reproduction/` | `tracks/clean_evaluation/` only | Clean-eval variant's README must state exactly what changed and why, citing this issue by number |
| 6. Test-set F1 used for checkpoint selection | `BaseModel.fit()` selects `best_state` via `test_results["F1"]` | **Reproduce exactly** | **This is the headline clean_evaluation change** — Part 12's "validation-based checkpoint selection" | Yes, only in `tracks/clean_evaluation/` (a modified `fit()` call path, not an edit to `vendor/muad/training/base_model.py`) | `tracks/clean_evaluation/validation_checkpoint.py` | Must explicitly log both the faithful (test-driven) and clean (validation-driven) checkpoint's metrics side-by-side in any comparison report |
| 7. Test precision used in active-learning stopping | `main.py`'s AL branch checks `test_results["Pre"] >= precicison` to decide early stop | **Reproduce exactly** | Clean variant may replace with a validation-based or iteration-count-based stopping rule, explicitly listed in Part 12 | Yes, clean_evaluation only | `tracks/clean_evaluation/run_clean.py` | Same as above |
| 8. Paper/code hyperparameter discrepancies (epochs 50 vs 100, `max_iter` 30 vs 1, dead `loss1`/`loss2`/`hidden_dim`-in-config, λ-sweep absent from code) | Config drifted across commits; several config keys are dead | **Faithful baseline uses commit `6031ccf`'s exact config values** (`epochs=50`, `max_iter=30`, `hidden_dim=[64,64]` restored via wrapper) — **not** current HEAD's `params.json` | Clean evaluation may use either config explicitly, stated up front, never silently mixed | No patch — this is a config-selection decision, not a code patch | `experiments/configs/faithful_full_label.json` / `faithful_active_learning.json` pin the `6031ccf` values explicitly | Every experiment config file header comment states which historical config it reproduces and why |

**No issue in this table is resolved silently anywhere in `tracks/faithful_reproduction/`.**

---

# PART 11 — FAITHFUL REPRODUCTION TRACK

**Track name:** `faithful_reproduction`
**Objective:** "Reproduce what the historical MUAD implementation (commit `6031ccf`) actually does, on our own regenerated Dataset C, in a modern-but-compatible environment."

- **Commit baseline:** `slg-frank/muad @ 6031ccf` (vendored unmodified into `vendor/muad/`).
- **Dataset baseline:** our own regenerated Dataset C (post-Gate-10), *not* the recovered pickle itself (which is retained only as a comparison target, per PART 7) — though a secondary faithful-reproduction run using the recovered pickle directly (bypassing L1–L4 entirely) is explicitly permitted as an additional, clearly-labeled variant, since it lets us isolate "does our dataset regeneration matter" from "does the model code behave as documented."
- **Config baseline:** commit `6031ccf`'s `configs/params.json` values — `epochs=50`, `max_iter=30`, `evaluation_epoch=5`, `patience=15`, `lr=0.001`, `random_seed=42`, `gpu=false`, `precicison=0.9999`, **`hidden_dim=[64,64]` explicitly restored via our wrapper** (since current-HEAD's `main.py` doesn't wire it, but `6031ccf`'s did).
- **Hyperparameters:** exactly as in the table above; `loss1`/`loss2`/`aloss`/`b_loss` remain dead (never wired in), reproduced as dead, not activated.
- **Training behavior:** `full_label`-equivalent full-dataset training was **not** how `6031ccf`'s `main.py` operated (it had no `trainType` branching at all — it ran the active-learning-style loop unconditionally, per `MUAD_RUNTIME_VERIFICATION.md` Phase E.1's git-history finding). **Correction to the original task framing:** the faithful-reproduction track's "full label" variant must therefore be understood as *our own construction*, modeled on current-HEAD's `full_label` branch logic but using `6031ccf`'s model-construction kwargs — this must be documented as a **hybrid**, explicitly: "training-loop control flow from `58923ca`'s `full_label` branch (since `6031ccf` had no such branch), model-construction arguments from `6031ccf` (since `58923ca`'s are broken)." This is not a silent merge — it is the only way to exercise "full label" mode at all with a working model, and must be labeled as such everywhere it appears.
- **Checkpoint behavior:** test-set F1-driven `best_state` selection, exactly as coded, reproduced without alteration.
- **Evaluation behavior:** `BaseModel.evaluate()`'s exact TP/FP/FN/TN logic (`label==0` vs `label!=0`), unmodified.
- **Active-learning behavior:** `main.py`'s AL branch logic (from `58923ca`, since `6031ccf` itself had no branching — same hybrid-labeling requirement as above applies), with `hidden_dim=[64,64]` restored, `max_iter=30` (not `1`), `n_select=200`, `confidence_threshold=0.9`, entropy `n_samples=5` — **including the confirmed train_loader non-propagation bug, unfixed.**
- **Known quirks that must be preserved, explicitly enumerated:** (1) evaluation stochasticity (no eval-mode gating on `UncertainBlock`'s sampling), (2) test-set-driven checkpoint selection, (3) AL non-propagation, (4) `var`'s dual raw-variance/log-variance interpretation, (5) fusion-by-concatenation, (6) dead `loss1`/`loss2`/`aloss`/`b_loss`, (7) the `TCPConfidence_sig` (fused-level) rather than per-modality confidence being used for AL pseudo-labeling.

**This track must not include any of the Part 12 changes.**

---

# PART 12 — CLEAN EVALUATION TRACK

**Track name:** `clean_evaluation`
**Objective:** "Correct specific, enumerated methodological issues, only after `faithful_reproduction` exists and has been run at least once for comparison."

**Explicitly listed changes (nothing beyond this list is in scope for `clean_evaluation`):**
1. **Validation-based checkpoint selection** — carve a validation split from `train_keys` (distinct from the AL query pool and from the test set); select `best_state` by validation F1, never touching `test_loader` during training.
2. **No test-set access during training** — `test_loader` is not passed to `.fit()` at all in the clean track; a single `evaluate(test_loader)` call happens once, after all model-selection decisions are final.
3. **Clean stopping criteria** — replace the test-precision-based AL early-stop with a validation-based or fixed-iteration-count rule, explicitly stated in `tracks/clean_evaluation/run_clean.py`'s config.
4. **Episode/group-aware splitting, if justified** — this operates at the *dataset* level (our L3/Gate 8), regenerating `chunk_train.pkl`/`chunk_test.pkl` with fault-episode-grouped rather than flat-random assignment. Explicitly marked optional/conditional: only pursued if a future decision (not made in this document) determines it's warranted given Part 7's leakage-risk discussion in the dataset forensic doc.
5. **Deterministic evaluation** — an optional flag to disable `UncertainBlock`'s sampling at eval time (e.g., using `mu_attn` directly instead of the reparameterized sample) for a deterministic-inference variant, explicitly labeled as a `clean_evaluation`-only inference mode, never used to characterize "the" MUAD model's behavior without this caveat stated.

**Also folds in Part 10's Issue 5 fix (AL propagation) as an explicit, separately-toggleable change within this track**, not bundled silently with the other four.

**This track is never merged into `faithful_reproduction`.** Every `clean_evaluation` run's output must be labeled as such and never presented as "the" MUAD reproduction result without the faithful-reproduction number alongside it for comparison.

---

# PART 13 — TEST STRATEGY

### Unit tests
| Test | Expected behavior |
|---|---|
| Metric normalization | z-score computed per-service-per-metric over one experiment; mean≈0, std≈1 on synthetic input with known statistics |
| Timestamp conversion (UTC+8→UTC) | A known local-time string converts to the exact expected Unix epoch (using the documented 8-hour offset) |
| Service alias mapping | `nginx-thrift`/`socialnetwork-nginx-thrift-1` both resolve to node `nginx-web-server` / node id 11; all other 11 services map identity |
| Fault interval handling | A synthetic fault record with known start/duration produces the exact expected `(s,e,service)` triple in `records<idx>.json` format |
| Window overlap | A synthetic 10s window and a synthetic fault interval with a known 1s overlap is labeled anomalous; a window with zero overlap is not |
| Label generation | `culprit` assignment matches the expected node id for a synthetic fault-overlapping window |
| Trace extraction | A synthetic Jaeger span with known `processID`/`startTime`/`duration` produces the expected `(second_bucket, edge_key, latency)` entry |
| Log conversion | A synthetic raw log line produces the expected `(epoch_timestamp, service, event_id)` row after Drain parsing |
| `UncertainBlock` | Given fixed `mu`/`var` inputs (bypassing the upstream network), `reparametrize`/`kl_loss` produce exactly the values the documented formulas predict |
| Fusion dimensions | Three synthetic `[B,64]` tensors concatenate to `[B,192]` in the documented trace/metric/log order |
| Loss calculation | Given fixed logits/labels/confidences, `total_loss` matches the hand-computed formula from Part 9's table exactly |

### Integration tests
| Test | Expected behavior |
|---|---|
| raw → parsed_data | Running the full L2 pipeline on the real raw archive (or a small fixture subset) produces schema-valid `parsed_data/SN/*` files, passing Gate 3 |
| parsed_data → chunk | Running vendored `align.py` on our L2 output produces schema-valid chunks, passing Gate 4 |
| chunk → DataLoader | Vendored `ChunkDataset`/`create_dataloader` load our chunks without error, passing Gate 10 |
| DataLoader → model | A real batch from our DataLoader drives a full `MainModel` forward pass (with `hidden_dim=[64,64]` wrapper) without shape errors |
| model → loss | The forward pass's returned `loss` is a finite scalar |
| loss → backward | `loss.backward()` succeeds; at least one parameter has a nonzero gradient |

### Regression tests
| Test | Expected behavior |
|---|---|
| Recovered Dataset C shape | Our freshly-generated chunks match every "Exact match required" property from Part 7's table against the recovered artifact |
| Recovered chunk counts | Our split ratio is within documented tolerance of 60/40 |
| Known model dimensions | A freshly-constructed `MainModel` (via our wrapper) prints a module structure matching `MUAD_RUNTIME_VERIFICATION.md` Phase D's documented output exactly (module names, layer types, in/out feature counts) |
| Historical runtime behavior | Re-running `MUAD_RUNTIME_VERIFICATION.md`'s Phase E/F/G/H/I procedures against our own environment and our own regenerated dataset reproduces the same qualitative results (forward succeeds with `hidden_dim=[64,64]`, fails without it; evaluation is stochastic; checkpoint selection is test-F1-driven; AL non-propagation is observed) |

### End-to-end tests
| Test | Expected behavior |
|---|---|
| Minimal complete training/evaluation run | A `faithful_reproduction` run with a small epoch count and a small data subset completes without error, produces a saved checkpoint, and produces a Precision/Recall/F1 report — this is Experiment 2 in Part 14, promoted to a standing e2e test that must pass before any full-scale run is attempted |

---

# PART 14 — REPRODUCTION EXPERIMENTS

| # | Name | Goal | Inputs | Configuration | Output | Metric | Comparison Target | Success Criteria |
|---|---|---|---|---|---|---|---|---|
| 0 | Dataset reproduction | Regenerate Dataset C from raw and pass all validation gates | Raw archive | `experiments/configs/dataset_gen.json` (test_ratio=0.4, chunk_lenth=10, threshold=1) | Regenerated `chunk_train.pkl`/`chunk_test.pkl`/`metadata.json` | Gate pass/fail counts, Part 7 comparison table | Recovered artifact (Part 7 categories) | All "Exact match required" properties pass; all "Investigation required" properties are explicitly reported, not silently ignored |
| 1 | Minimal MUAD forward pass | Confirm one batch flows through the full model without error, using the `hidden_dim=[64,64]` wrapper | A handful of chunks (real or synthetic) | Phase I of Part 8 | `MMlogit`, `loss`, `y_pred` | Shape/finiteness only | Runtime doc Phase E's success case | Matches Phase E.1's documented shapes exactly |
| 2 | Minimal training run | Confirm `.fit()` executes a handful of epochs, checkpoints correctly | Small data subset | 3–12 epochs, `evaluation_epoch=1` or `3` | Trained checkpoint, logged per-epoch test F1 | Loss decreasing trend (qualitative, not a target number), checkpoint-selection trace | Runtime doc Phase H's documented trace | Test evaluation fires at the expected epochs; `best_state` selection follows the documented `>` comparison exactly |
| 3 | Faithful full-label training | Full-scale training on the full regenerated Dataset C, faithful_reproduction track | Full dataset (post-Gate-10) | `experiments/configs/faithful_full_label.json` (epochs=50, per `6031ccf`) | Trained checkpoint, F1/Rec/Pre | F1/Rec/Pre (test-set-selected checkpoint, as coded) | Recovered `scores.txt`/`training.log` numbers (§12 of dataset doc, ≈0.92–0.93 F1 for the AL-mode runs actually logged there) | Report the achieved numbers alongside the historical ones; do **not** require matching a specific score — success is "the faithful pipeline runs to completion and produces a directly comparable report," not hitting a target F1 |
| 4 | Faithful active learning | Full AL loop, `max_iter=30`, non-propagation bug preserved | Full dataset | `experiments/configs/faithful_active_learning.json` | Per-iteration checkpoints/scores | Per-iteration F1/Rec/Pre, `train_keys`/`val_keys` size trace | Recovered `training.log`'s single-iteration pattern (since no multi-iteration historical log was ever recovered — §Phase J of runtime doc) | Loop runs the configured number of iterations without error; the non-propagation behavior is directly observed and logged, not assumed |
| 5 | Comparison with recovered results | Systematic side-by-side of Experiment 3/4's numbers against every empirically-recovered number in the three baseline docs | Experiment 3/4 outputs + recovered artifacts | — | A comparison report/table | — | All three baseline docs | Every comparison is reported with its "Exact match / Expected difference / Investigation required" classification (reusing Part 7's framework, extended to model-output numbers) |
| 6 | Clean evaluation | Run the `clean_evaluation` track (Part 12) on the same dataset | Full dataset | `experiments/configs/clean_evaluation.json` | Trained checkpoint, F1/Rec/Pre under validation-based selection | Same metrics, clean protocol | Experiment 3's faithful numbers | Report both side-by-side; success is a valid, completed, clean-protocol run with a transparent comparison — not a claim that clean numbers are "better" or "worse" in any predetermined sense |
| 7 | Ablations | Run each Part 15 baseline variant | Full dataset | one config per variant | metrics per variant | F1/Rec/Pre per variant | Each other variant, and the full model | Each variant completes and produces a report; differences are described, not judged against an unstated target |
| 8 | Robustness experiments | (Optional/extension, Part D of the project objective) — e.g. modality-missing or noise-injection tests, if pursued, modeled on the MUAD paper's RQ3 but **not** assumed to reproduce its exact numbers | Full dataset + perturbation config | To be specified at the time this experiment is actually planned in detail (not fully specified in this blueprint — flagged in Part 20 as a remaining planning item for D-track extensions only) | — | — | — | — |

**No experiment in this table has an arbitrary target performance score as its success criterion**, per the task's explicit instruction — every "success criteria" cell above is about completion, correctness of protocol, and transparent comparison, not about hitting a number.

---

# PART 15 — BASELINES AND ABLATIONS

| Order | Variant | Implementation | Scientific question answered |
|---|---|---|---|
| 1 | Metric-only | `MetricEncoder`→`GraphModel1`→`UncertainBlock`→single-modality classifier (paper's "Only Metric" ablation, model's `TCPClassifierLayer_metric` reused directly as the final head, bypassing fusion) | How much does the metric modality alone contribute? |
| 2 | Trace-only | Same pattern with `TraceEncoder` | How much does the trace modality alone contribute? |
| 3 | Log-only | Same pattern with `LogEncoder` | How much does the log modality alone contribute? |
| 4 | Deterministic multimodal | All three encoders + `GraphModel1`, fusion via plain concatenation (no `UncertainBlock`, no confidence weighting — i.e. `ω_v=1` for all v) | Does uncertainty/confidence modeling add anything over naive fusion? (maps to paper's "w/o GPE" + "w/o CFM" combined) |
| 5 | Multimodal with graph | Same as 4, explicitly confirming `GraphModel1`'s contribution is present (this is really the same as 4, listed separately only to isolate "graph vs. no graph" if a no-graph variant is later added — flagged as a possible future addition, not committed to in this blueprint) | Isolates the graph-attention contribution, if a no-graph control is added later |
| 6 | Multimodal with uncertainty | Adds `UncertainBlock` back (GPE), still no confidence-weighted fusion (`ω_v=1`) | Maps to paper's "w/o CFM" ablation |
| 7 | Multimodal with confidence-aware fusion | Adds CFM's confidence weighting back, no `UncertainBlock` (deterministic `z_v = mu` only, no sampling/KL) | Maps to paper's "w/o GPE" ablation |
| 8 | Full MUAD | Everything, as vendored (= `faithful_reproduction` track's Experiment 3) | The complete, released model |
| 9 | Full MUAD + active learning | Everything, plus the AL loop (= `faithful_reproduction` track's Experiment 4) | Does active learning reduce labeling cost while maintaining performance, as claimed? |

Each baseline is implemented as a thin variant configuration/wrapper around the same vendored components (per Part 8/9's "reuse, don't rewrite" principle) — no baseline requires writing a new encoder or graph layer from scratch; only the *composition* changes.

---

# PART 16 — EXPERIMENT MANAGEMENT

- **Configuration files:** one JSON per experiment under `experiments/configs/`, each with a mandatory header comment block stating: which track (`faithful_reproduction`/`clean_evaluation`/ablation name), which historical config baseline it derives from (`6031ccf` vs. current-HEAD vs. neither), and which Part-10 issues are reproduced-as-is vs. patched.
- **Random seeds:** `42` by default (matching `6031ccf`'s `params.json`), explicitly set via the vendored `seed_everything()` for all `faithful_reproduction` runs; `clean_evaluation`/ablation runs may use a documented seed sweep (e.g., 3–5 seeds) since they are not bound to reproducing one historical run bit-for-bit.
- **Experiment naming:** `{track}_{variant}_{dataset_version}_{seed}_{timestamp}` (e.g., `faithful_full_label_datasetv1_seed42_20260921T1200`).
- **Output directories:** `experiments/runs/{experiment_name}/` containing `config.json` (copy of the exact config used), `checkpoints/`, `training.log`, `scores.txt`, `env_report.json` (see below).
- **Model checkpoints:** saved exactly as the vendored code does (`best_model.pth` for `full_label`-style runs; per-iteration for AL runs), never overwritten across experiment names.
- **Logs:** the vendored `logging_utils.py::setup_logging` output, retained per-run, never appended across separate experiments.
- **Metrics:** `scores.txt` (vendored `dump_scores` format) plus a structured `metrics.json` (ours) for easier downstream comparison/plotting.
- **Plots:** generated post-hoc by `scripts/` (Part 15's comparison plots, loss curves, confusion matrices) — never generated inside the vendored training loop itself.
- **Environment metadata:** `scripts/env_report.py` captures Python/torch/dgl/torchdata/numpy versions (mirroring `MUAD_RUNTIME_VERIFICATION.md` Phase A's table) into every run's `env_report.json`.
- **Git commit tracking:** every run's `config.json` also records (a) our own project repo's commit hash, (b) `vendor/eadro`'s pinned commit hash, (c) `vendor/muad`'s pinned commit hash (`6031ccf`), and (d) the dataset version identifier (hash of the generated `chunk_train.pkl`, so any future re-generation is distinguishable from this one).

**Every experiment must record:** dataset version (chunk-file hash), code commit (all three repos), config (full JSON copy), seed, hardware (CPU/GPU, core count, RAM — captured automatically), software environment (`env_report.json`).

---

# PART 17 — COMPUTATIONAL PLAN

| Mode | Purpose | Data scope | Epochs | Expected duration (CPU, per `params.json`'s `gpu:false` default) |
|---|---|---|---|---|
| Development mode | Writing/debugging L2/L3.5/validation code | Single small fixture experiment (a few hundred windows) | N/A (no training) | Seconds–minutes per iteration |
| Debug mode | Stepping through model forward/backward | A handful of real chunks (10–50) | 1–3 | Seconds |
| Small-data smoke-test mode | End-to-end pipeline sanity (Experiment 1/2, e2e test) | Small subset (hundreds of chunks) | 3–12 | Minutes |
| Full Dataset C mode | Dataset regeneration + validation only, no training | Full 5,610-chunk-scale dataset | N/A | Minutes (dataset generation is not the bottleneck; raw archive is small, ~79MB) |
| Full reproduction mode | Experiment 3/4 (faithful, full-scale) | Full dataset | 50 (full_label-style), up to 30 iterations × per-iteration epochs (AL) | Hours, CPU-bound — realistic for a B.Tech project's available hardware, since the original repo's own `params.json` specifies `gpu:false` and the dataset is small (thousands of 12-node graphs, not millions) |
| Clean-evaluation mode | Experiment 6 | Full dataset | Same order of magnitude as full reproduction | Hours |

**No step in this plan assumes GPU access is required** — the vendored code's own default config (`"gpu": false`) and the dataset's modest scale (12-node graphs, batch size 50, thousands of chunks) make CPU-only execution realistic on typical B.Tech-available hardware, consistent with what `MUAD_RUNTIME_VERIFICATION.md` already used throughout Phases C–J.

---

# PART 18 — MILESTONES

### Milestone 1 — Raw archive ingestion + Gate 1
1. **Files to create:** `preprocessing/raw_to_parsed/run_conversion.py` (stub orchestrator only, no conversion logic yet), `validation/gates/gate01_raw_integrity.py`.
2. **Inputs required:** the raw `SN Dataset.zip`, extracted to `raw_data/`.
3. **Expected outputs:** a printed/logged confirmation that the extracted tree matches `DATASET_FORENSICS_COMPLETE.md` §4's documented structure exactly (7 experiments, `SN-all.tgz` redundancy confirmed).
4. **Tests:** `tests/integration/test_gate01.py`.
5. **Completion criterion:** Gate 1 passes on the real archive.

### Milestone 2 — Fault manifest conversion + Gate 2
1. **Files:** `preprocessing/raw_to_parsed/convert_faults.py`, `validation/gates/gate02_fault_manifest.py`.
2. **Inputs:** the 4 `data/SN.fault-*.json` + 3 `no fault/SN.fault-*.json` files.
3. **Outputs:** `parsed_data/SN/records<idx>.json` per experiment (7 files).
4. **Tests:** unit tests (fault interval transform, alias mapping) + `tests/integration/test_gate02.py`.
5. **Completion criterion:** Gate 2 passes — 36 total fault records accounted for exactly, correctly distributed across services/types.

### Milestone 3 — Metrics conversion + partial Gate 3
1. **Files:** `preprocessing/raw_to_parsed/convert_metrics.py`.
2. **Inputs:** `metrics/*.csv` per experiment.
3. **Outputs:** `parsed_data/SN/metrics<idx>/<service>.csv`.
4. **Tests:** unit test (schema passthrough) + row-count validation.
5. **Completion criterion:** all 7×12 metric files converted, schema-valid.

### Milestone 4 — Logs conversion (UTC+8 fix + Drain) + partial Gate 3
1. **Files:** `preprocessing/raw_to_parsed/convert_logs.py`.
2. **Inputs:** `logs.json` per experiment.
3. **Outputs:** `parsed_data/SN/templates.json`, `logs<idx>.csv` per experiment.
4. **Tests:** unit tests (timezone conversion, template assignment stability) + integration test reading the output back.
5. **Completion criterion:** all log lines converted with the UTC+8 offset applied and verified against a known fault-JSON timestamp for the same experiment.

### Milestone 5 — Traces conversion (Jaeger parsing) + full Gate 3
1. **Files:** `preprocessing/raw_to_parsed/convert_traces.py`.
2. **Inputs:** `spans.json` per experiment.
3. **Outputs:** `parsed_data/SN/traces<idx>.json`.
4. **Tests:** unit tests (processID resolution + alias, timestamp bucketing) + `tests/integration/test_gate03_full.py`.
5. **Completion criterion:** Gate 3 fully passes — every reconstructed `parsed_data/SN/*` file is schema-valid against the vendored Eadro code's actual field accesses.

### Milestone 6 — Vendor Eadro + run preprocessing + Gate 4/5/6/7/8
1. **Files:** `vendor/eadro/` (pinned checkout), `preprocessing/run_eadro_preprocessing.sh`.
2. **Inputs:** Milestone 2–5's `parsed_data/SN/*`.
3. **Outputs:** `chunks.pkl`, `chunk_train.pkl`, `chunk_test.pkl`, `metadata.json`.
4. **Tests:** `validation/gates/gate04..08_*.py`.
5. **Completion criterion:** Gates 4–8 all pass.

### Milestone 7 — MUAD compatibility trim + Gate 9
1. **Files:** `preprocessing/muad_compat/trim_trace_channel.py`.
2. **Inputs:** Milestone 6's chunk files.
3. **Outputs:** trimmed chunk files (`traces[...,:1]`).
4. **Tests:** the full `tests/regression/test_dataset_regression.py` (Part 7).
5. **Completion criterion:** Gate 9's "Exact match required" items pass; every "Expected difference"/"Investigation required" item is logged and reviewed (not silently passed).

### Milestone 8 — Vendor MUAD + DataLoader smoke test + Gate 10
1. **Files:** `vendor/muad/` (pinned at `6031ccf`), `requirements-repro.txt`, `environment_notes.md`.
2. **Inputs:** Milestone 7's trimmed chunks.
3. **Outputs:** confirmed batched-graph shapes.
4. **Tests:** Phase A of Part 8, reruns `MUAD_RUNTIME_VERIFICATION.md` Phase C against our own data.
5. **Completion criterion:** Gate 10 passes.

### Milestone 9 — Model wrapper + Phases B–I of Part 8
1. **Files:** `tracks/faithful_reproduction/` wrapper module constructing `BaseModel`/`MainModel` with `hidden_dim=[64,64]` explicit.
2. **Inputs:** Milestone 8's DataLoader.
3. **Outputs:** confirmed forward/backward success (Experiment 1).
4. **Tests:** Part 9's full component checklist, Part 13's unit+integration tests.
5. **Completion criterion:** Experiment 1 succeeds; negative-control test (no `hidden_dim` kwarg) reproduces the documented crash.

### Milestone 10 — Minimal training run (Experiment 2) + Phase J
1. **Files:** `experiments/run_experiment.py`, `experiments/configs/smoke_test.json`.
2. **Inputs:** Milestone 9's working model + a small data subset.
3. **Outputs:** a completed short training run with checkpoint.
4. **Tests:** the standing e2e test (Part 13).
5. **Completion criterion:** e2e test passes; checkpoint-selection behavior matches Runtime doc Phase H.

### Milestone 11 — Faithful full-scale training (Experiment 3)
1. **Files:** `experiments/configs/faithful_full_label.json`.
2. **Inputs:** full Milestone 8 dataset.
3. **Outputs:** trained checkpoint, F1/Rec/Pre report.
4. **Tests:** regression comparison against recovered `scores.txt`.
5. **Completion criterion:** run completes; report produced (no target score required).

### Milestone 12 — Faithful active learning (Experiment 4) + Phase K
1. **Files:** `tracks/faithful_reproduction/run_active_learning.py`, `experiments/configs/faithful_active_learning.json`.
2. **Inputs:** full dataset.
3. **Outputs:** per-iteration checkpoints/scores.
4. **Tests:** reruns Runtime doc Phase I's `id()`/length check at full scale.
5. **Completion criterion:** loop completes `max_iter=30` iterations (or documents exactly where/why it doesn't); non-propagation behavior directly confirmed at scale.

### Milestone 13 — Comparison report (Experiment 5)
1. **Files:** `scripts/compare_to_recovered_artifact.py`, extended to model outputs.
2. **Inputs:** Milestone 11/12 outputs + the three baseline docs.
3. **Outputs:** a structured comparison report.
4. **Tests:** none new — this milestone *is* a test/report.
5. **Completion criterion:** report produced, every comparison classified.

### Milestone 14 — Clean evaluation track (Experiment 6)
1. **Files:** `tracks/clean_evaluation/*`.
2. **Inputs:** full dataset.
3. **Outputs:** clean-protocol checkpoint + report.
4. **Tests:** confirms none of the faithful-reproduction quirks are silently present (validation-based selection verified, no test-loader passed during training).
5. **Completion criterion:** run completes; side-by-side comparison with Milestone 11 produced.

### Milestone 15 — Ablations (Experiment 7)
1. **Files:** `ablations/*.py`.
2. **Inputs:** full dataset.
3. **Outputs:** one report per variant.
4. **Tests:** each variant's own shape/completion test.
5. **Completion criterion:** all Part 15 variants run and report.

### Milestone 16 — Visualization/demo (optional, D-track extension)
1. **Files:** `scripts/`, possibly a small `notebooks/` or lightweight web demo.
2. **Inputs:** any completed experiment's outputs.
3. **Outputs:** plots/demo artifacts.
4. **Tests:** none required (presentation layer).
5. **Completion criterion:** at the project team's discretion — not blocking for the core reproduction goal.

---

# PART 19 — DECISIONS WE MUST NOT REVISIT

1. Dataset C = Eadro SocialNetwork, Zenodo DOI `10.5281/zenodo.7615393`.
2. The raw archive (`SN Dataset.zip`) is the source for dataset reconstruction; it contains exactly 7 experiments (4 fault + 3 no-fault), and `SN-all.tgz` is a redundant copy, not additional data.
3. The raw archive contains exactly 36 fault records; the Eadro paper's "72" figure remains an unresolved, unexplained discrepancy — never silently substituted either direction.
4. Eadro's raw→parsed_data stage is a **reconstructed** stage, because no official converter exists in the Eadro repository (confirmed via full commit-history inspection).
5. Historical MUAD commit `6031ccf` is the faithful-reproduction baseline, because it is the exact commit that (a) produced the recovered chunk/log artifacts and (b) was empirically confirmed to run end-to-end in `MUAD_RUNTIME_VERIFICATION.md`.
6. Current HEAD `58923ca` is **not** the primary reproduction baseline, because runtime verification found a source-level regression (dropped `hidden_dim` kwarg) that makes it crash on the documented forward pass.
7. `faithful_reproduction` and `clean_evaluation` remain two permanently separate tracks; no change from Part 12's list may ever be merged into `faithful_reproduction`.
8. No silent fixes anywhere in vendored code; every deviation from a vendored file's literal behavior happens only in our own wrapper/orchestration code, and is documented as a deviation, never presented as "how MUAD works."
9. No synthetic anomalies are added to Dataset C at any stage.
10. No unverified paper claim (e.g., the λ1/λ2 sensitivity sweep, the "preliminary" root-cause-localization extension, the "72 injections" figure) is treated as an implementation fact unless independently confirmed by code or raw data, per the three baseline documents.
11. The `traces[...,:1]` channel trim is a **reconstructed, MUAD-specific compatibility step**, not part of faithful Eadro reproduction — it lives only in L3.5/`preprocessing/muad_compat/`, never inside the vendored Eadro preprocessing code path.
12. The recovered chunk/log artifacts are **evidence for comparison**, not a definitionally-correct target our own regeneration must bit-for-bit match (see Part 7's exact-match/expected-difference/investigation-required framework).

---

# PART 20 — FINAL READINESS CHECKLIST

| Item | Status | Notes |
|---|---|---|
| Dataset reconstruction plan | ✅ Specified | Part 4 |
| Eadro preprocessing plan | ✅ Specified | Part 5 |
| Dataset validation gates | ✅ Specified | Part 6 |
| Recovered artifact comparison | ✅ Specified | Part 7 |
| Model implementation order | ✅ Specified | Part 8 |
| Tensor shape ledger | ✅ Specified (inherited from `MUAD_MODEL_FORENSICS_COMPLETE.md` §19 + empirically confirmed in `MUAD_RUNTIME_VERIFICATION.md` §3, reused directly by Part 8/9) | — |
| Mathematical implementation mapping | ✅ Specified | Part 9, inherits `MUAD_MODEL_FORENSICS_COMPLETE.md` §10 |
| Training plan | ✅ Specified | Part 11 |
| Active-learning plan | ✅ Specified | Part 11, Part 8 Phase K |
| Experiment plan | ✅ Specified | Part 14 |
| Test strategy | ✅ Specified | Part 13 |
| Faithful reproduction track | ✅ Specified | Part 11 |
| Clean evaluation track | ✅ Specified | Part 12 |
| Project directory structure | ✅ Specified | Part 3 |
| First implementation milestone | ✅ Specified | Part 18, Milestone 1 |

**Known open items not fully specified in this blueprint (flagged, not blocking Milestone 1):**
- The exact Drain-parser configuration/version to use for log-template mining in Milestone 4 is not pinned — any reasonable, documented choice is acceptable, since template count is an "expected difference" category (Part 7), but the specific choice must be recorded in `preprocessing/raw_to_parsed/convert_logs.py`'s documentation when written.
- Part 14's Experiment 8 (robustness experiments) is explicitly left under-specified, as a D-track (optional extension) item, per the task's own framing of extensions as separate from the core reproduction goal — this does not block readiness for Milestones 1–15.
- `align.py`'s internal seeding behavior (whether it calls any RNG-seeding function itself) is flagged **[RE-VERIFY]** in Part 5 — this must be confirmed by reading the actual vendored file at Milestone 6, not assumed; it does not block Milestones 1–5.

None of the three open items above concern Milestone 1, which depends only on the raw archive already in hand and Gate 1's structural check.

**READY TO CODE = YES**, for Milestones 1 through 15 as specified. The three flagged open items are minor, non-blocking, and each has an explicit point in the milestone sequence where it must be resolved before that specific milestone (not before Milestone 1) proceeds.
