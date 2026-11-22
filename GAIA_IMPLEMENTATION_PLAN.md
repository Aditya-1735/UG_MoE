# GAIA MicroSS implementation and experiment log

Updated 2026-11-22 on `hdfs` · GPU server · Conda environment `muad`

**Injection records audited and telemetry storage measured.** The pinned telemetry requires about 3.81 GiB compressed and 29.41 GiB fully extracted. The acquisition plan now keeps the telemetry as split ZIPs and streams members during preprocessing. Bulk download, clock alignment, final label rules, implementation, and GAIA experiments are pending.

This file is the working record for the third dataset. Update the checklist and append a dated journal entry after each implementation step or server run. Record what changed, how it was verified, and where the evidence lives. Keep proposed settings marked as such until the data audit supports them. Use `muad` for Python commands. Codex runs short checks and records verified checkpoints locally. Publishing to `origin/hdfs` uses the project user's own GitHub identity with write access. The user launches long downloads, data builds, and GPU experiments with `nohup`.

## 1. Objective and benchmark scope

Train UG-MoE v2 on GAIA MicroSS and evaluate normal-versus-anomalous classification on a later GAIA period. Each sample is one time window containing a graph of monitored services or service instances. The main comparison follows the SN and TT final runs: MUAD, GAT, v2 plain, v2 graph, and v2 full, all trained with the same GAIA inputs and partitions. Report this as supervised evaluation on a third system.

Use the **July 2021 MicroSS data** from CloudWise's `release-v1.0`, at `5f1d772a7845dd93e4ab04dfc0ee725bf0ca933a`. Use `metric/`, `trace/`, and `business/` as feature sources and `run/` as the source for injection records. CloudWise's August update omits August traces. The benchmark period will be the audited intersection of available July telemetry and label coverage. [Pinned release](https://github.com/CloudWise-OpenSource/GAIA-DataSet/releases/tag/release-v1.0) · [Release notes](https://github.com/CloudWise-OpenSource/GAIA-DataSet/releases) · [Upstream schema](https://github.com/CloudWise-OpenSource/GAIA-DataSet#micross)

AnoFusion used GAIA for instance-level failure detection under its own split and evaluation protocol. Cite it as related work. Any numerical comparison would require running both methods on a shared prediction target, split, and scoring protocol. [AnoFusion, §5–6](KDD23-AnoFusion.pdf)

### Lessons carried over from SN and TT

| Prior finding | GAIA decision |
| --- | --- |
| Randomly split, overlapping SN windows inflated scores. | Split chronological blocks before generating windows. Keep entire incidents in one partition. |
| Early TT validation was anomaly-heavy and anomaly-F1 favored an all-anomalous checkpoint. | Verify both classes in every split. Select by validation macro-F1, with validation PR-AUC as the tiebreaker. |
| Clock offsets and sparse TT traces affected how much those inputs could contribute. | Audit timestamps and coverage for each modality before setting the sampling grid or interpreting the model's results. |
| SN and TT log templates were mined across all runs. | Fit GAIA's template vocabulary, feature statistics, and any trace-derived topology on training data only. |
| Several seed-42 effects disappeared in the final results. | Use seed 42 for integration and exploration. Use paired seeds for the final comparative results. |

[Provided guide](<7. UG-MoE_Implementation_Guide.md>) · [Implementation report](UG-MoE_Implementation_Report.md) · [Project status](PROJECT_STATUS.md) · [Project explained](PROJECT_EXPLAINED.md) · [Run history](EXPERIMENT_RUN_HISTORY.md)

## 2. Milestones and completion evidence

| Status | Milestone | Evidence to record here |
| --- | --- | --- |
| Done | Plan the integration and confirm upstream release and current runner interfaces. | Release link, exact commit, code references, and initial journal entry. |
| Done | Fetch, test, and extract the injection-record archive. | Pinned commit verified. `7z t` passed. The archive extracted one 3,643,229-byte CSV. [Retry log](project/experiments/runs/gaia_fetch_run_v1_retry_20261122_145518.log) |
| Done | Measure the archive and extraction footprint and verify split-ZIP streaming. | [Size log](project/experiments/runs/gaia_telemetry_sizes_20261122_145949.log), three terminal-volume indexes, and a local streaming probe. |
| Pending | Fetch and verify the remaining compressed MicroSS telemetry. | All 81 files match the pinned Git blobs. Each archive passes `7z t`. |
| Pending | Audit telemetry, topology, and fault records. | Common date range, time conversions, cadence, node mapping, coverage, parsed and unresolved incident counts. |
| Pending | Freeze the GAIA feature schema, label rules, and chronological split. | Selected window and stride, cut dates and gaps, coverage policy, topology provenance, split manifest. |
| Pending | Implement the adapter and verify shared loader and baseline compatibility. | Commit, edited files, compilation result, user-run build log, and data and loader checks. |
| Pending | Complete the full-v2 training run at seed 42. | Launch command, log, checkpoint, `summary.json`, selected epoch, metrics, and skipped-batch count. |
| Pending | Complete the seed-42 comparison and check checkpoint evaluation. | Per-model results on the same split and a checkpoint-based clean-score check. |
| Pending | Complete final paired runs and regenerate the GAIA results. | Exact run directories, five seed results per model, tables, and the final findings and limitations. |

## 3. Acquisition and raw-data audit

The project filesystem showed 9.8 GiB free on 2026-11-22. This accommodates the compressed archives. The telemetry CSVs require about 29.41 GiB if fully extracted. Keep the archive parts together under `project/datasets/gaia/archives/micross_v1/`. Download the parts directly from the pinned revision so the compressed files occupy a single copy outside the sparse checkout. Stream individual CSV members using `7z x -so` and retain only compact summaries or processed features. Estimate the footprint of processed windows and temporary state before a full build. The commands are in [Server commands](#7-server-commands). [CloudWise MicroSS directory](https://github.com/CloudWise-OpenSource/GAIA-DataSet/tree/main/MicroSS)

| Modality | Archive files | Compressed | Extracted |
| --- | ---: | ---: | ---: |
| Metric | 17 | 0.78 GiB | 4.46 GiB |
| Trace | 32 | 1.50 GiB | 7.77 GiB |
| Business | 32 | 1.53 GiB | 17.18 GiB |
| Total | 81 | 3.81 GiB | 29.41 GiB |

Compressed sizes came from the pinned Git tree. Extracted sizes came from `unzip -Z -t` on the three terminal `.zip` volumes. The terminal volumes are already cached in the archive directories and their Git blob hashes match the pinned tree. Local p7zip 16.02 passed a byte-for-byte streaming check on a small multipart ZIP. The full GAIA archives still need integrity checks after the numbered parts arrive.

| Audit item | What to verify and record |
| --- | --- |
| Metrics | Parse metric identities and source entities from filenames. Check epoch-millisecond conversion, observed cadence and gaps, duplicates, and which CPU, memory, and network channels are usable across nodes. Record whether the selected values are gauges, counters, or derived rates. |
| Traces | Verify `start_time`, `end_time`, and the record timestamp against the metric clock. Count invalid durations and unmatched parents. Join parent and child spans within the same trace. Measure per-node and per-day trace coverage. |
| Business logs | Compare the outer `datetime` field with the event timestamp embedded in the message. Record the chosen timestamp and offset, parse rate, node mapping coverage, and genuine periods of low activity. Mask variable tokens such as request IDs before template mining. |
| Injection records | Parse explicit start, end or duration, type, and affected entity when available. Distinguish message emission time from injection start. Deduplicate repeated records and audit ambiguous or conflicting intervals. Keep a count of excluded intervals and the reason for exclusion. |
| Entity alignment | Verify the mapping among metric source IDs, trace `service_name` and `host_ip`, and business-log services. Prefer instances when all sources can be mapped reliably. Record the chosen node unit, any host metrics shared across nodes, unresolved identities, and the order of the nodes. |

The audit must establish a usable common July period and enough labeled incidents and monitored background in each partition. Record missing telemetry as missing in the audit even when the final feature encoding uses a fixed fill value. Treat monitored background outside documented injections as the negative class under the selected GAIA protocol. Record any periods whose coverage or incident labels are too uncertain to use.

### Injection records observed on 2026-11-22

`run_table_2021-07.csv` contains **17,155 rows** across ten service IDs and 30 calendar dates. The `datetime` field contains the date only, and 19 July is absent. This file establishes where injection records are available. Establish coverage for negative labels independently from the aligned telemetry and record any ambiguous periods. [Extracted CSV](project/datasets/gaia/raw/micross_v1/run/run/run_table_2021-07.csv)

| Record family | Count | What the records provide |
| --- | --- | --- |
| Memory injection | 652 | Unique service/start/duration tuples. All declare 600 seconds. Messages are emitted about 601 seconds after the stated start. |
| File-moving injection | 43 | Unique service/start/duration tuples. All declare 600 seconds. Messages are emitted about 600 seconds after the stated start. |
| CPU injection | 15 | Twelve declare about 3 seconds. Three declare about 19–23 days and extend past July. Flag the three intervals for verification before using their durations. |
| Access permission denied injection | 15 | The message declares one hour and supplies an emission timestamp. An explicit start field is absent. Verify the start interpretation against telemetry before using these intervals. |
| Failed injection / normal-memory markers | 10 / 17 | Count separately from successful fault injections. |

The 695 explicit 600-second records are an initial candidate set, based on the stated start in the message. Verify their clock alignment and merge overlapping candidates into system-level time intervals before counting independent episodes. CPU outliers, uncertain permission start times, and 19 July coverage remain open. Avoid treating unresolved affected periods as confirmed normal time.

### Preliminary metric cadence

Four Docker metric CSVs for `mobservice1` and `logservice2` were readable from the metric archive's terminal volume. Their extracted bytes matched the ZIP entry sizes and CRCs. CPU, disk I/O, and network samples all had a modal interval of **30 seconds**, with duplicates, gaps, or out-of-order rows in some files. The 10-second Eadro setting is unsuitable for these series at their native cadence. Audit the selected channels before freezing a GAIA grid and a window that spans multiple real observations.

## 4. Data contract and preprocessing

### Chronological partition and window labels

1. Start from approximately **60% train, 20% validation, 20% test** of usable time. Place cuts at incident-free boundaries around those targets and record the exact timestamps. Leave a gap of at least **60 seconds** at each cut, extended if the feature lookback requires it. Check that each partition has normal windows and distinct fault episodes.
2. Evaluate a **30-second grid** for GAIA. The initial metric sample is slower than the 10-second Eadro window. Choose window length and stride from the measured cadence, coverage, and incident durations. Record the missing-value policy and qualify what the chosen resolution can show about short CPU incidents.
3. Generate windows within each partition. Prevent rolling features, metric fills, incidents, or windows from crossing cuts. Base positive labels on overlap with a confirmed injection interval. Use the Eadro convention of at least one second of overlap if the selected timestamps support that resolution. Record how overlapping injections are handled for per-type counts. The classifier remains binary.
4. Select metric channels and fit the log-template vocabulary using training data. Keep an unknown-template bucket for later logs. Fit normalization on training windows and apply it identically to validation and test. Infer a fixed graph from training parent–child calls and retain a documented node-to-index map. Preserve label metadata for diagnostics.

### Outputs and integration points

The proposed processed root is `project/datasets/gaia/processed/micross_v1/`.

| Output | Contents |
| --- | --- |
| `chunks.pkl` | Window ID, start time, aligned metric `[N, T, M]`, trace `[N, T, R]`, and log `[N, E]` arrays, and incident metadata. `culprit=0` denotes normal operation and positive IDs denote fault types. `N` is the fixed node count and `T` the number of time steps. |
| `metadata.json` | `format=gaia_micross_v1`, source revision, system, node IDs, directed `edges`, `node_num`, `metric_num`, `trace_dim`, `event_num`, label map, sampling and feature definitions. |
| `quality.json` | Date range, timestamp conversions, per-day and per-node coverage, field parse failures, unmapped entities, graph checks, and counts of windows and independent incidents by split and type. |
| `split_manifest.json` | Exact split bounds, gaps, window IDs, incident assignments, source revision, and a stable manifest identifier or hash for matching runs. |

- **Adapter.** Add `project/data_pipeline/build_gaia.py` to read the audited archives, align the telemetry, and write the contract above. Read selected metric files and stream trace and business CSVs in bounded batches. Estimate processed size and graph-loader memory from a sample before building the whole benchmark.
- **Shared loader.** `common.py` currently sends any directory with `chunks.pkl` through the Eadro run-assignment code. Dispatch by dataset format and read GAIA's explicit manifest. Reuse the existing DGL dataset, training-only normalizer, training loop, and metrics. Verify that SN and TT still resolve their existing splits. [Shared loader](project/experiments/common.py#L189)
- **Model dimensions.** Full v2 already takes metric, trace, and event dimensions from metadata. MUAD assumes seven metric channels and one trace channel, and the GAT/Graph-MoE builder assumes one trace channel. If GAIA needs different dimensions, pass them through where needed and preserve the SN/TT defaults. All GAIA models use the same representations. [Registry](project/experiments/registry.py#L79) · [MUAD encoders](project/vendor/muad/models/encoders.py#L6)
- **Verification.** Compile changed Python files. For the user-run data check, verify the manifest is complete and disjoint, every window uses only its own partition's observations, every incident belongs to at most one partition, feature arrays are finite with the declared shapes, edges resolve to valid nodes, and each split has both classes. Verify that learned preprocessing uses training data and that the quality report explains drops or missing modalities. Record split and normalization artifacts from the first runner invocation.

## 5. Experiment sequence

| Stage | Runs | Completion check |
| --- | --- | --- |
| Single-model integration | `ug_moe2:full`, seed **42**, using the audited GAIA dataset. Use the shared defaults for the first comparable run. | Training logs the expected split, metrics are finite, the epoch history shows zero skipped batches, checkpoint and per-seed result exist, and `errors` is empty. Investigate unexpected skips before the final runs. |
| Same-seed comparison | `muad`, `gat`, `ug_moe2:plain`, and `ug_moe2:graph` at seed **42** with the same frozen inputs as full v2. | All five models have results for the same split, feature schema, and settings. Compare with all-normal and all-anomalous predictors and investigate any collapse. |
| Checkpoint evaluation | Reload the full-v2 checkpoint on the same GAIA artifacts. Run clean evaluation and, if covering robustness, the existing missing-modality, noise, and routing/fusion diagnostics. | Clean full-v2 metrics match training's held-out evaluation within numerical tolerance. Record model key, checkpoint, split, normalization, and seed for every evaluation. |
| Final comparison | **42, 123, 456, 789, 999** for all five models, after the seed-42 workflow is verified. | Report per-seed values, mean and standard deviation, paired deltas, and seed counts. Reuse a seed-42 result only if its code, data, split, preprocessing, and settings match the final protocol. |

Use validation macro-F1 for checkpoint selection and validation PR-AUC for ties. Record test macro-F1, anomaly precision, recall and F1, normal-class F1, PR-AUC, ROC-AUC, normal-window false-positive rate, and per-type recall where labels support it. Make protocol and hyperparameter decisions from the training and validation data.

The first comparable run uses the historical defaults of Adam, learning rate `5e-4`, batch size `50`, weight decay `1e-5`, a `50` epoch cap, and patience `15`. Record any changes that become necessary for GAIA and apply comparable settings to the models being compared. Keep legacy variants out of the main table by passing the model names explicitly. [Runner defaults](project/experiments/registry.py#L213)

### Additional analyses and reporting

The core GAIA deliverable is the clean main comparison. If the GAIA results will be used to claim robustness or explain a component, run the corresponding guide experiment after the seed-42 gates:

- **Component ablation (8B).** Use the six `abl2:*` configurations on the same split. Pair seed-level deltas against full v2 for any final component claim.
- **Modality subset (8C).** Train with the existing `@metrics`, `@traces`, `@logs`, and pair suffixes. Keep these trained-subset results distinct from test-time masking in the report.
- **Missing modalities and noise (8D–E).** Reuse `run_robustness.py` with one trained checkpoint per invocation. Start with full v2, MUAD, and GAT. Use the existing single-modality drops, 10/20/40/60% random masking, and training-standard-deviation-scaled noise. Repeat across final paired seeds for cross-dataset robustness claims.
- **Routing and uncertainty (8F–G).** Use `run_analysis.py` on the full-v2 checkpoint. Label seed-42 heatmaps and fusion results as exploratory and derive class names from GAIA metadata.
- **Sensitivity (8H).** Defer a GAIA sweep until there is a reason to include one. Keep the frozen GAIA configuration as the reference when reporting any later sweep.

After verification, add GAIA paths and the GAIA label map to the report workflow. `make_report.py` currently iterates over SN and TT. Verify the regenerated GAIA table against the per-seed JSON files and record the exact report directory. [Report implementation](project/experiments/make_report.py#L35)

## 6. Decisions to finalize after the audit

| Decision | Current proposal | Recorded outcome |
| --- | --- | --- |
| Exact dates and time zone | Common July period after aligning all sources. | Pending audit. |
| Node granularity and topology | Use verified service instances where possible. Infer the fixed graph from training call spans. | Pending entity and trace audit. |
| Metric channels, cadence, and windows | Start from the Eadro feature definitions where GAIA has equivalent channels. Evaluate 30-second bins. | Four sampled series show 30-second cadence. Final channel set, window, and stride are pending the broader audit. |
| Label and exclusion rules | Use explicit injection intervals, with an audit of ambiguous cases and overlap rules. | Pending review of `run/` records. |
| Split boundaries and background coverage | Approximately 60/20/20 chronological with a gap at each cut. | Pending incident and coverage counts. |
| Final GAIA experiment scope | Clean five-model comparison, with further analyses tied to claims about GAIA. | Pending seed-42 results. |

## 7. Server commands

### Fetch the injection records first

From the GPU server. The job clones without populating the worktree, verifies the release commit, configures sparse checkout, then checks out `MicroSS/run`. This sequence avoids the URL-handling bug in `git clone --sparse` on the server's Git 2.25.1. It then tests and extracts `run.zip` for the label audit. [Upstream Git fix](https://github.com/git/git/commit/47dbf10d8a5ce8c9b441a16b7698c7d70585dff0)

```bash
cd "$HOME/AHA_BTP/UG_MoE/project"
df -h .
mkdir -p datasets/gaia/raw experiments/runs

GAIA_RELEASE="$PWD/datasets/gaia/raw/release-v1.0"
GAIA_RAW="$PWD/datasets/gaia/raw/micross_v1"
GAIA_LOG="experiments/runs/gaia_fetch_run_v1_retry_$(date +%Y%m%d_%H%M%S).log"

nohup bash -c '
  set -euo pipefail
  if [[ ! -d "$1/.git" ]]; then
    git -c protocol.version=2 clone --depth 1 --filter=blob:none --no-checkout --branch release-v1.0 \
      https://github.com/CloudWise-OpenSource/GAIA-DataSet.git "$1"
  fi
  REV=$(git -C "$1" rev-parse HEAD)
  printf "GAIA release commit %s\n" "$REV"
  test "$REV" = 5f1d772a7845dd93e4ab04dfc0ee725bf0ca933a
  git -C "$1" sparse-checkout init --cone
  git -C "$1" sparse-checkout set MicroSS/run
  git -C "$1" checkout --detach "$REV"
  archive="$1/MicroSS/run/run.zip"
  7z t "$archive"
  7z l "$archive"
  7z x -aoa "$archive" "-o$2/run"
  printf "Injection records extracted to %s\n" "$2/run"
' _ "$GAIA_RELEASE" "$GAIA_RAW" > "$GAIA_LOG" 2>&1 < /dev/null &

echo "PID=$!  Log=$GAIA_LOG"
tail -f "$GAIA_LOG"
```

### Inventory the telemetry archive sizes

This reads GitHub's metadata for the `MicroSS` tree at the verified revision and reports compressed sizes by modality. The 2026-11-22 results are recorded above. [GitHub tree API](https://docs.github.com/en/rest/git/trees#get-a-tree)

```bash
cd "$HOME/AHA_BTP/UG_MoE/project"
GAIA_RELEASE="$PWD/datasets/gaia/raw/release-v1.0"
GAIA_LOG="experiments/runs/gaia_telemetry_sizes_$(date +%Y%m%d_%H%M%S).log"

nohup bash -c '
  set -euo pipefail
  TREE=$(git -C "$1" rev-parse HEAD:MicroSS)
  curl -fsSL --retry 3 "https://api.github.com/repos/CloudWise-OpenSource/GAIA-DataSet/git/trees/$TREE?recursive=1" |
    "$2" -c "
import json
import sys

tree = json.load(sys.stdin)
if tree[\"truncated\"]:
    raise SystemExit(\"GitHub returned an incomplete archive inventory\")
for modality in (\"metric\", \"trace\", \"business\"):
    parts = [item for item in tree[\"tree\"]
             if item[\"type\"] == \"blob\" and item[\"path\"].startswith(modality + \"/\")]
    if not parts:
        raise SystemExit(f\"Missing archive parts for {modality}\")
    size = sum(item[\"size\"] for item in parts)
    print(f\"{modality}: {len(parts)} files, {size / 1024**3:.2f} GiB compressed\")
"
  df -h "$1"
' _ "$GAIA_RELEASE" "$HOME/miniconda3/envs/muad/bin/python" > "$GAIA_LOG" 2>&1 < /dev/null &

echo "PID=$!  Log=$GAIA_LOG"
tail -f "$GAIA_LOG"
```

### Fetch and verify the compressed telemetry

The [archive downloader](scripts/fetch_gaia_archives.sh) downloads the numbered volumes directly into the archive directory. It reads expected file paths and Git blob hashes from the pinned release. It reuses existing verified files and renames a temporary download only after its hash matches. Once all parts are present, it checks each archive without extracting its CSVs. Allow time for about 3.7 GiB of remaining downloads and the integrity checks.

```bash
cd "$HOME/AHA_BTP/UG_MoE/project"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate muad
GAIA_LOG="experiments/runs/gaia_fetch_telemetry_v1_$(date +%Y%m%d_%H%M%S).log"

nohup bash ../scripts/fetch_gaia_archives.sh > "$GAIA_LOG" 2>&1 < /dev/null &

echo "PID=$!  Log=$GAIA_LOG"
tail -f "$GAIA_LOG"
```

### First seed-42 training job

Run after the GAIA adapter and shared loader are implemented and the processed data and split pass the checks above. The example uses the proposed processed root. Replace it with the recorded path if the build uses a different root.

```bash
cd "$HOME/AHA_BTP/UG_MoE/project"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate muad

export DGLBACKEND=pytorch
CUDA_LIBS="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="$CUDA_LIBS/cuda_nvrtc/lib:$CUDA_LIBS/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

nvidia-smi
read -r -p 'Available GPU index: ' GAIA_GPU

GAIA_DATA=datasets/gaia/processed/micross_v1
GAIA_RUN="experiments/runs/gaia_full_s42_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES="${GAIA_GPU:?Enter an available GPU index}" \
  nohup "$CONDA_PREFIX/bin/python" -u experiments/run_baselines.py \
  --device cuda \
  --data-dir "$GAIA_DATA" \
  --seeds 42 \
  --models ug_moe2:full \
  --output-dir "$GAIA_RUN" > "$GAIA_RUN.log" 2>&1 < /dev/null &

echo "PID=$!  Log=$GAIA_RUN.log"
tail -f "$GAIA_RUN.log"
```

The `Saved to:` line identifies the timestamped results directory. Check `models["ug_moe2:full"].errors` in `summary.json`, the epoch history in `ug_moe2_full_seed42.json`, and the corresponding `_best.pth` file. For the rest of the seed-42 comparison, use the same launch with a new output root and `--models muad,gat,ug_moe2:plain,ug_moe2:graph`. For checkpoint diagnostics, use `run_robustness.py` or `run_analysis.py` with the same `--data-dir` and `--seeds 42`, plus `--model ug_moe2:full --checkpoint <saved checkpoint>` and a new output root. Add the actual build, comparison, checkpoint-evaluation, and final launch commands to the journal as those stages are reached.

## 8. Implementation journal

| Date | Milestone | What was done and how | Evidence and next step |
| --- | --- | --- | --- |
| 2026-11-22 | Plan drafted | Reviewed CloudWise's release and schemas, the SN/TT run history, and the current dataset loader and model registry on `hdfs` at `19be29b`. Documented the proposed GAIA data contract, split and evaluation sequence. | Release and code references are linked above. Next step is acquisition and the raw-data audit. |
| 2026-11-22 | Acquisition prepared | Checked the server workspace. No GAIA data or run logs were present. `df -h .` showed 9.9 GiB free and `7z` was available. Split acquisition into an injection-record job and a telemetry job. | Injection-record launch is ready for the user. Confirm archive integrity and extraction from its log, then audit the label files. Confirm storage before fetching telemetry. |
| 2026-11-22 | First acquisition attempt failed | The user launched the injection-record job using `git clone --filter=blob:none --sparse`. Git 2.25.1 tried to use the HTTPS URL as its working directory during sparse-checkout initialization. The workspace check found no resulting checkout. Replaced the launch with `--no-checkout`, followed by sparse-checkout setup and a checkout of the verified revision. | [Failed launch log](project/experiments/runs/gaia_fetch_run_v1_20261122_145300.log) · [Git bug fix](https://github.com/git/git/commit/47dbf10d8a5ce8c9b441a16b7698c7d70585dff0). Retry and verify `run.zip`. |
| 2026-11-22 | Injection records acquired | The user reran the revised launch. The checkout resolved to the pinned commit, `7z t` passed, and one CSV was extracted. Read-only inspection counted records by message family and checked explicit starts and durations against message emission times. | [Retry log](project/experiments/runs/gaia_fetch_run_v1_retry_20261122_145518.log) · [Label audit](#injection-records-observed-on-2026-11-22). Size the telemetry archives next. |
| 2026-11-22 | Telemetry storage measured | The user ran the GitHub inventory. Codex fetched the three terminal ZIP volumes, matched their Git blob hashes to the pinned release, and used their indexes to measure extraction sizes. A small split-ZIP probe verified `7z x -so` against the original member bytes. | [Size log](project/experiments/runs/gaia_telemetry_sizes_20261122_145949.log). Cache the remaining compressed parts and test the complete archives. |
| 2026-11-22 | Metric sampling and downloader prepared | CRC-verified four metric samples and measured 30-second cadence. Added a resumable downloader that verifies every archive part against the Git tree. A temporary three-archive fixture passed download, cached-part reuse, `7z t`, and retry checks. | [Preliminary cadence](#preliminary-metric-cadence) · [Downloader](scripts/fetch_gaia_archives.sh). Launch the full archive download in `muad`. |
| 2026-11-22 | GitHub publication blocked | Committed the plan and storage audit as `f7541b5` and the downloader as `2ce7707`. HTTPS required GitHub authentication. The shared server identity lacked write access to `Aditya-1735/UG_MoE`. The project user directed that future publishing use their own identity. | The local `hdfs` checkpoints are ready for publication through the project user's authorized GitHub access. |
| 2026-11-22 | User published the first GAIA checkpoints | The user pushed `hdfs`. A read-only GitHub check confirmed `2ce7707` on the remote, with the plan and downloader committed. The revised publishing note was still a working-tree edit at the time of that push. | Publish the revised note in a follow-up commit on `hdfs`. |

For a code or data update, record the date, commit, affected files, decision and rationale, exact artifacts, verification performed, and next unresolved item. For any failed attempt, keep the command and log path and add the cause and recovery once confirmed.

### Run log

Both injection-archive attempts are recorded in the journal. GAIA training runs are pending. Add one entry per launch, including retries or superseded runs.

```text
Date and purpose:
Status:
Code commit / data release / split manifest ID:
GPU / Conda environment:
Exact command:
Log path / output directory / checkpoint:
Model(s) / seed(s) / deviations from defaults:
Split sizes and class counts:
Selected epoch / skipped batches / runner errors:
Validation and test results:
Interpretation and follow-up:
```
