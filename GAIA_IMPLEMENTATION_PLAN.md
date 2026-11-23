# GAIA MicroSS implementation and experiment log

Updated 2026-11-23 on `hdfs` · GPU server · Conda environment `muad`

**Matched GAIA comparison complete.** At seed 42, full UG-MoE v2 achieved anomaly F1 **0.6571** and adapted AnoFusion achieved **0.1284** on the same 1,194 held-out system windows. [Matched results](#matched-seed-42-results) · [Matched protocol](#matched-gaia-evaluation)

This file is the working record for the third dataset. Update the checklist and append a dated journal entry after each implementation step or server run. Record what changed, how it was verified, and where the evidence lives. Keep proposed settings marked as such until the data audit supports them. Use `muad` for Python commands. Codex runs short checks and records verified checkpoints locally. Publishing to `origin/hdfs` uses the project user's own GitHub identity with write access. The user launches long downloads, data builds, and GPU experiments with `nohup`.

## 1. Objective and benchmark scope

Train `ug_moe2:full` on GAIA MicroSS at **seed 42** and evaluate normal-versus-anomalous classification on a later GAIA period. Each sample is one time window containing a graph of monitored service instances. Report this as supervised evaluation on a third system. The matched comparison reuses this checkpoint and fits AnoFusion once at seed 42 for each monitored instance.

Use the **July 2021 MicroSS data** from CloudWise's `release-v1.0`, at `5f1d772a7845dd93e4ab04dfc0ee725bf0ca933a`. Use `metric/`, `trace/`, and `business/` as feature sources and `run/` as the source for injection records. CloudWise's August update omits August traces. The benchmark period will be the audited intersection of available July telemetry and label coverage. [Pinned release](https://github.com/CloudWise-OpenSource/GAIA-DataSet/releases/tag/release-v1.0) · [Release notes](https://github.com/CloudWise-OpenSource/GAIA-DataSet/releases) · [Upstream schema](https://github.com/CloudWise-OpenSource/GAIA-DataSet#micross)

**AnoFusion (KDD 2023) is the published GAIA reference.** The paper used CloudWise GAIA MicroSS as D1. Report its Table 3 result alongside our result with both evaluation protocols identified. [AnoFusion paper, §5–6](KDD23-AnoFusion.pdf)

### Lessons carried over from SN and TT

| Prior finding | GAIA decision |
| --- | --- |
| Randomly split, overlapping SN windows inflated scores. | Split chronological blocks before generating windows. Keep entire incidents in one partition. |
| Early TT validation was anomaly-heavy and anomaly-F1 favored an all-anomalous checkpoint. | Verify both classes in every split. Select by validation macro-F1, with validation PR-AUC as the tiebreaker. |
| Clock offsets and sparse TT traces affected how much those inputs could contribute. | Audit timestamps and coverage for each modality before setting the sampling grid or interpreting the model's results. |
| SN and TT log templates were mined across all runs. | Use fixed log severity counts for the GAIA run. Fit feature normalization and infer trace-derived topology on training data only. |
| Several seed-42 effects disappeared in the final results. | Label the GAIA result as single-seed. Consider additional UG-MoE seeds only after the seed-42 workflow passes and the schedule allows. |

[Provided guide](<7. UG-MoE_Implementation_Guide.md>) · [Implementation report](UG-MoE_Implementation_Report.md) · [Project status](PROJECT_STATUS.md) · [Project explained](PROJECT_EXPLAINED.md) · [Run history](EXPERIMENT_RUN_HISTORY.md)

## 2. Milestones and completion evidence

| Status | Milestone | Evidence to record here |
| --- | --- | --- |
| Done | Plan the integration and confirm upstream release and current runner interfaces. | Release link, exact commit, code references, and initial journal entry. |
| Done | Fetch, test, and extract the injection-record archive. | Pinned commit verified. `7z t` passed. The archive extracted one 3,643,229-byte CSV. [Retry log](project/experiments/runs/gaia_fetch_run_v1_retry_20261122_145518.log) |
| Done | Measure the archive and extraction footprint and verify split-ZIP streaming. | [Size log](project/experiments/runs/gaia_telemetry_sizes_20261122_145949.log), three terminal-volume indexes, and a local streaming probe. |
| Done | Fetch and verify the remaining compressed MicroSS telemetry. | All 81 files present. The downloader checked blob hashes, all three `7z t` checks passed, and the completion marker was printed. [Bulk download log](project/experiments/runs/gaia_fetch_telemetry_v1_git_20261122_153224.log) |
| Done | Audit telemetry and fault records. | [Coverage JSON](project/experiments/runs/gaia_coverage_20261122_155845/coverage.json) and [audit log](project/experiments/runs/gaia_coverage_20261122_155845.log). |
| Done | Freeze the GAIA feature schema, label rules, and chronological split. | [Data contract](#4-data-contract-and-preprocessing) and [decision record](#6-decisions). |
| Done | Implement the streaming adapter and shared loader. | [Builder](project/data_pipeline/build_gaia.py) and [loader](project/experiments/common.py#L189). |
| Done | Build and check the full GAIA artifacts. | [Build log](project/experiments/runs/gaia_build_20261122_170933.log) · [Quality report](project/datasets/gaia/processed/micross_v1/quality.json) · [Split manifest](project/datasets/gaia/processed/micross_v1/split_manifest.json). |
| Done | Complete the full-v2 training run at seed 42. | [Run record](#run-log) · [Runner summary](project/experiments/runs/gaia_full_s42_20261122_175126/baselines_20261122_175136/summary.json). |
| Done | Report the seed-42 result alongside the published AnoFusion reference. | [Both results and scoring protocols](#anofusion-comparison). |
| Done | Export and verify the aligned AnoFusion inputs. | [Export log](project/experiments/runs/gaia_matched_20261122_183655/export.log) · [Observations](project/experiments/runs/gaia_matched_20261122_183655/observations.npz). |
| Done | Train and score AnoFusion and compare both models on the frozen windows. | [AnoFusion log](project/experiments/runs/gaia_matched_20261122_183655/anofusion.log) · [Matched results](#matched-seed-42-results). |

## 3. Acquisition and raw-data audit

The project filesystem showed 9.2 GiB free at the first bulk download. This accommodates the compressed archives. The telemetry CSVs require about 29.41 GiB if fully extracted. Keep the archive parts together under `project/datasets/gaia/archives/micross_v1/`. The downloader fetches blobs from the pinned revision over Git HTTPS and puts its temporary Git object store on `/tmp`, which had 16 GiB free when checked. Stream individual CSV members using `7z x -so` and retain only compact summaries or processed features. Estimate the footprint of processed windows and temporary state before a full build. The commands are in [Server commands](#7-server-commands). [CloudWise MicroSS directory](https://github.com/CloudWise-OpenSource/GAIA-DataSet/tree/main/MicroSS)

| Modality | Archive files | Compressed | Extracted |
| --- | ---: | ---: | ---: |
| Metric | 17 | 0.78 GiB | 4.46 GiB |
| Trace | 32 | 1.50 GiB | 7.77 GiB |
| Business | 32 | 1.53 GiB | 17.18 GiB |
| Total | 81 | 3.81 GiB | 29.41 GiB |

Compressed sizes came from the pinned Git tree. Extracted sizes came from `unzip -Z -t` on the three terminal `.zip` volumes. Local p7zip 16.02 passed a byte-for-byte streaming check on a small multipart ZIP. All 81 archive parts matched their pinned Git blob hashes and all three archives passed `7z t`.

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

The 695 unique, explicit 600-second memory and file-moving injections form the positive-label source, using the stated starts. These are service-level records and can overlap in time. The masking and date exclusion policy below covers the unresolved intervals.

### Preliminary metric cadence

Four Docker metric CSVs for `mobservice1` and `logservice2` were readable from the metric archive's terminal volume. Their extracted bytes matched the ZIP entry sizes and CRCs. CPU, disk I/O, and network samples all had a modal interval of **30 seconds**, with duplicates, gaps, or out-of-order rows in some files. The 10-second Eadro setting is unsuitable for these series at their native cadence. Audit the selected channels before freezing a GAIA grid and a window that spans multiple real observations.

### Archive schemas and clock evidence

- The archives contain 6,640 metric CSVs, ten trace CSVs, and ten business-log CSVs. Metric filenames, trace `service_name`, business `service`, and injection records agree on ten application service IDs. Metric filenames map each of these IDs to one host. The coverage scan selects seven Docker channels available for all ten services across both metric date ranges.
- Metrics use Unix epoch milliseconds. Use `start_time` for trace events and the timestamp embedded in the message for business logs. The top-level business `datetime` field is date-only in the observed files.
- For early July memory injections, interpreting the `run/` timestamps as UTC+8 gave 49 rises above 0.5 GiB in 50 events with enough surrounding metric data. Interpreting them as UTC gave zero such rises in 56 events with enough metric data. The aligned median rise under UTC+8 was 1.002 GiB.
- In the business-log sample, all 880 parseable embedded epoch timestamps agreed within one second with the message clock interpreted as UTC+8. The trace and business event clocks also agreed within one second for 19,230 of 19,330 sampled trace-ID matches within the same service. Use UTC+8 for the local calendar and retain the metric epoch values as absolute times.
- The coverage audit records each member's row range, daily occupied 30-second bins, invalid timestamps, and service mismatches. The full scan covered **5,230,960** metric rows in 140 selected members, **28,681,438** trace rows in ten members, and **87,974,871** business-log rows in ten members. It flagged 294 invalid business-log timestamps and out-of-order rows in all three modalities. The adapter aggregates by timestamps to handle ordering. [Audit code](project/data_pipeline/audit_gaia.py) · [Coverage report](project/experiments/runs/gaia_coverage_20261122_155845/coverage.json)

## 4. Data contract and preprocessing

### Chronological partition and window labels

An eligible UTC+8 date must appear in the injection log and have observations for every metric series and all ten instances in both trace and business logs. July 19 lacks injection records. The first potentially unbounded CPU interval starts July 26, so July 26 onward is excluded. The 11 remaining days are split chronologically using the 60/20/20 target.

| Split | Assigned July 2021 dates (UTC+8) | Normal | Memory | File move | Total |
| --- | --- | ---: | ---: | ---: | ---: |
| Train | 1, 2, 3, 8, 10, 17, 21 | 2,614 | 539 | 32 | 3,185 |
| Validation | 22, 23 | 923 | 200 | 18 | 1,141 |
| Test | 24, 25 | 1,007 | 176 | 11 | 1,194 |

The full build matched the preflight counts. July 3 passed the daily coverage check and had zero retained windows after window-level filtering. The manifest records assigned dates and retained window IDs. Its SHA-256 is `81a385ae4ace3b9480dead6d209b9f50955f05f992bf0dd934bf34e0f8a7bc26`. [Quality report](project/datasets/gaia/processed/micross_v1/quality.json) · [Split manifest](project/datasets/gaia/processed/micross_v1/split_manifest.json)

- Use 30-second bins and non-overlapping 120-second windows within each day. Require all seven metric channels for all ten instances in all four bins. Drop windows with missing metric bins. The preflight dropped 2,239 windows for this reason.
- Leave a 120-second margin on both sides of the two midnight partition cuts on July 22 and 24. Mask confirmed injections that cross a cut. The preflight found one such injection.
- Exclude windows that overlap any of the 27 bounded ambiguous intervals. These cover 12 short CPU injections and 15 permission exceptions. Each permission exception is masked within one hour on either side of its message timestamp. The preflight excluded 157 windows for masked intervals and four for cut margins.
- Mark a window anomalous when it overlaps a confirmed memory or file-moving injection by at least one second. The type with greatest overlap supplies the per-type count if events overlap. Other eligible windows supply the normal class. The classifier maps both fault types to the same binary target.

### Input features and topology

- Metrics contain the bin mean for each of the three Docker CPU, two memory, and two network series selected by the audit. This uses the supplied metric values.
- Traces contain `log1p` of the mean nonnegative span latency in milliseconds per instance and 30-second bin. A bin with no usable spans is encoded as zero.
- Business logs contain `log1p` of per-window counts in five fixed severity buckets per instance. The buckets are INFO, WARNING, ERROR, DEBUG, and OTHER. The fixed encoding avoids a separate template-mining pass. A bucket with no events is encoded as zero.
- The graph uses directed parent-to-child calls joined by trace and span IDs in the first common hour of training traces on July 1, starting at 14:57:53.049982 UTC+8. The full build confirmed 30 directed cross-service links and one self-loop per node. The quality report records the links and their supporting counts.
- The runner fits normalization on training windows and applies the stored statistics to all splits. Feature dimensions and service order are stored in `metadata.json`.

### Outputs and integration points

The processed root is `project/datasets/gaia/processed/micross_v1/`. The four artifacts occupy 8.6 MiB on the server.

| Output | Contents |
| --- | --- |
| `chunks.pkl` | Window ID, epoch start time, metric `[10, 4, 7]`, trace `[10, 4, 1]`, and log `[10, 5]` arrays. `culprit` is 0 for normal, 1 for memory, and 2 for file moving. |
| `metadata.json` | `format=gaia_micross_v1`, source revision, system, node IDs, directed `edges`, `node_num`, `metric_num`, `trace_dim`, `event_num`, label map, sampling and feature definitions. |
| `quality.json` | Audit and label paths, record and exclusion counts, retained windows by split and class, dropped windows by reason, graph sample timestamps, and directed call counts. |
| `split_manifest.json` | Exact split dates, UTC+8 offset, margins, window IDs, and source revision. |

- The adapter streams selected CSV columns in batches and publishes the processed directory only after all four artifacts are ready. [Builder](project/data_pipeline/build_gaia.py)
- The shared loader dispatches on the GAIA format marker, checks that the manifest assigns each window exactly once, and uses the existing dataset, train-only normalizer, model registry, and training loop. [Shared loader](project/experiments/common.py#L189) · [Model registry](project/experiments/registry.py#L79)
- After the full build, check the manifest, shapes, finite values, graph indices, and class counts before launching seed 42. Record the saved split and normalization artifacts from the runner.

## 5. Experiment sequence

| Stage | Runs | Completion check |
| --- | --- | --- |
| Data pilot | Stream a bounded sample to check the adapter and loader before the full build. | Verify finite tensors, labels, graph indices, disjoint split boundaries, and storage estimates. |
| Single GAIA training run | `ug_moe2:full`, seed **42**, using the frozen GAIA data and shared defaults. | Training logs the expected split, metrics are finite, the epoch history shows zero skipped batches, the checkpoint and result JSON exist, and `errors` is empty. |
| Published comparison | Place our seed-42 test precision, recall, and anomaly F1 alongside the AnoFusion paper result. | Record both evaluation protocols and the scope of the comparison. |

Use validation macro-F1 for checkpoint selection and validation PR-AUC for ties. Record test macro-F1, anomaly precision, recall and F1, normal-class F1, PR-AUC, ROC-AUC, normal-window false-positive rate, and per-type recall where labels support it. Make protocol and hyperparameter decisions from the training and validation data.

The first run uses the historical defaults of Adam, learning rate `5e-4`, batch size `50`, weight decay `1e-5`, a `50` epoch cap, and patience `15`. Pass `--models ug_moe2:full` explicitly. Additional seeds and GAIA ablations are deferred. [Runner defaults](project/experiments/registry.py#L213)

### Seed-42 result

The run selected epoch **17** by validation macro-F1 with PR-AUC as the tiebreaker. Early stopping ended training at epoch 32. The recorded history contains zero skipped batches and the runner reported no errors. The saved split matches the frozen manifest, and the checkpoint loaded on CPU with finite tensors. [Training log](project/experiments/runs/gaia_full_s42_20261122_175126.log) · [Per-seed result](project/experiments/runs/gaia_full_s42_20261122_175126/baselines_20261122_175136/ug_moe2_full_seed42.json)

| Split | Anomaly precision | Anomaly recall | Anomaly F1 | Normal F1 | Macro-F1 | PR-AUC | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.9381 | 0.9037 | 0.9206 | 0.9817 | 0.9511 | 0.9310 | 0.9479 |
| Test | 0.9892 | 0.4920 | 0.6571 | 0.9545 | 0.8058 | 0.6357 | 0.6996 |

On the July 24–25 test split, the model detected 92 of 187 anomalous windows and falsely flagged one of 1,007 normal windows. It detected 81 of 176 memory-injection windows and all 11 file-moving windows. All 95 misses occurred during memory injections. The runner's all-anomalous reference scored 0.2708 anomaly F1 on the same test windows. [Test metrics and class counts](project/experiments/runs/gaia_full_s42_20261122_175126/baselines_20261122_175136/summary.json)

### AnoFusion comparison

Table 3 reports **0.795 precision, 0.945 recall, and 0.857 F1** for AnoFusion on GAIA D1. The paper evaluates unsupervised instance-level detection, uses the first 60% of each instance's timeline for training and the remaining 40% for testing, and describes one-minute observations and continuous failure-segment scoring. Our complete trace scan counted 28,681,438 rows across ten instances, which matches the D1 trace count in Table 2. The score from Table 3 is a published reference under the paper's protocol. [AnoFusion paper, Table 2, Table 3, §5.1 and §6.2](KDD23-AnoFusion.pdf) · [GAIA audit log](project/experiments/runs/gaia_coverage_20261122_155845.log)

The published `mobservice2` label file has one binary label per second across July in UTC+8, with 123,479 positive seconds. The 66 explicit 600-second memory injections for that instance span 39,600 second bins using the recorded start rounded down to a second. All these bins are positive in the published labels. This corroborates the memory labels and establishes that the published demo covers additional positive periods. [Published `mobservice2` labels](https://github.com/AIOps-Lab-NKU/AnoFusion/blob/ae3c4ea98509008844a3eecf7c0bd1529faaba67/labeled_service/mobservice2.csv)

| Source | Prediction unit and scoring | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |
| UG-MoE v2 full, seed 42 | Supervised binary detection on two-minute system windows. Chronological train/validation/test split, with July 24–25 held out. | 0.9892 | 0.4920 | 0.6571 |
| AnoFusion, published GAIA D1 | Unsupervised instance-level detection with one-minute observations and continuous failure-segment scoring. Chronological 60/40 split for each instance. | 0.795 | 0.945 | 0.857 |

The prediction units, label coverage, splits, and scoring protocols differ. The table places the single-seed UG-MoE result alongside the published reference. [UG-MoE runner summary](project/experiments/runs/gaia_full_s42_20261122_175126/baselines_20261122_175136/summary.json) · [AnoFusion paper, Table 3](KDD23-AnoFusion.pdf)

### Matched GAIA evaluation

Evaluate both models on the frozen system-window manifest and its injection-derived labels. Use the same 1,141 validation and 1,194 test windows. Each two-minute window receives one binary prediction per model. Report the adapted AnoFusion result with its features and thresholds, and use Table 3 as the published reference under its original protocol.

| Script | Role |
| --- | --- |
| `data_pipeline/export_gaia_anofusion.py` | Reuse the frozen metrics and trace arrays. Stream the compressed business members to compute log severity counts per 30-second bin. Verify that the reconstructed counts reproduce the saved two-minute log features, then export 13 channels for each instance in each bin. |
| `experiments/run_gaia_anofusion.py` | Train the pinned AnoFusion network on the retained training bins per instance. Export forecast scores with timestamps and manifest hashes. |
| `experiments/compare_gaia.py` | Regenerate UG-MoE scores from the seed-42 checkpoint. Verify the split and normalization, align AnoFusion scores to the identical window IDs, and save shared metrics and per-window predictions. |

The AnoFusion model source is pinned at `ae3c4ea98509008844a3eecf7c0bd1529faaba67`. Use the exported GAIA features, the network's 20-bin history, and its six modality-pair graph relations. Fit the max-absolute scaling and the mutual-information graph for each instance on training bins. Estimate mutual information after discretizing the channels with training deciles. Initialize the upstream graph aggregation weights from a seeded uniform distribution and select the forecasting checkpoint by validation MSE, with a 100-epoch cap and patience 10. Set the device from the runner instead of the demo's fixed CUDA index. [AnoFusion network](https://github.com/AIOps-Lab-NKU/AnoFusion/blob/ae3c4ea98509008844a3eecf7c0bd1529faaba67/model/AnoFusion.py) · [Demo runner](https://github.com/AIOps-Lab-NKU/AnoFusion/blob/ae3c4ea98509008844a3eecf7c0bd1529faaba67/utils/main.py)

Restart forecast history at timestamp gaps and split cuts. Pad short histories with their first observed bin. Every retained window has at least three forecast targets after this reset rule. Convert the demo's normalized forecast error to a per-instance percentile using training errors. Take the maximum percentile across all scored bins and all ten instances in each window. This yields one system score per window. Require score coverage from every instance for every validation and test window.

Choose each model's highest threshold among those that maximize validation macro-F1. Report precision, recall, anomaly F1, normal F1, macro-F1, PR-AUC, and ROC-AUC on the frozen test windows. The original seed-42 result above uses the checkpoint's argmax decisions. UG-MoE trains with binary window labels. AnoFusion trains on telemetry forecasts and uses the same validation labels for its final threshold.

#### Matched seed-42 results

Both pipelines were evaluated on the same 1,194 two-minute system windows from July 24–25 and used the same 1,141 validation windows for their decision thresholds. The chosen thresholds were `0.5266876221` for UG-MoE v2 and `0.9965999842` for AnoFusion's aggregated percentile score. Their validation macro-F1 scores were 0.9511 and 0.5273, respectively. [Evaluation summary](project/experiments/runs/gaia_matched_20261122_183655/comparison/summary.json)

| Seed-42 model | Test precision | Test recall | Anomaly F1 | Normal F1 | Macro-F1 | PR-AUC | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| UG-MoE v2 full | 0.9892 | 0.4920 | 0.6571 | 0.9545 | 0.8058 | 0.6357 | 0.6996 |
| AnoFusion (adapted) | 0.1500 | 0.1123 | 0.1284 | 0.8617 | 0.4951 | 0.1506 | 0.4844 |

UG-MoE detected 92 of 187 anomalous windows and falsely flagged one of 1,007 normal windows. Adapted AnoFusion detected 21 of 187 and falsely flagged 119 normal windows. Its detections included 19 of 176 memory-injection windows and two of 11 file-moving windows. The anomaly F1 difference was 0.5287 under this single-seed protocol. The published 0.857 remains the reference for AnoFusion's original instance-level D1 evaluation. [Window predictions](project/experiments/runs/gaia_matched_20261122_183655/comparison/predictions.csv) · [AnoFusion paper, Table 3](KDD23-AnoFusion.pdf)

#### Why the adapted AnoFusion score differs from Table 3

The paper uses Drain-based log template clusters, multiple trace response-time statistics, and one-minute observations. Our matched run uses five severity counts and mean span latency on 30-second bins, along with seven metric channels. The paper's online detector divides the centered forecast errors by their interquartile range and applies a dynamic threshold based on extreme value theory. The public demo divides by the 25th percentile and exports timestamped scores. Our adapter follows that error formula with a numerical floor, then takes train-relative percentiles and their maximum over a system window. Its decision threshold is fixed using validation labels. The paper scores instance-level failure segments on a 60/40 chronological split. The shared evaluator scores individual system windows on the frozen GAIA split. The pinned public release includes preprocessed example inputs and labels for `mobservice2`. [Paper, §§4.2, 4.6, 5.1 and 6.2](KDD23-AnoFusion.pdf) · [Demo scoring code](https://github.com/AIOps-Lab-NKU/AnoFusion/blob/ae3c4ea98509008844a3eecf7c0bd1529faaba67/utils/main.py#L99)

An audit of the saved scores reproduced every system-window maximum and confirmed that the union of instance injection intervals recreates the binary labels. Adapted AnoFusion's validation and test ROC-AUC were 0.5665 and 0.4844. Test PR-AUC was 0.1506 at an anomaly prevalence of 0.1566. Using each instance's own injection windows and its raw within-window maximum, the median ROC-AUC across instances fell from 0.6222 on validation to 0.5127 on test. The anomaly ranking was weak for a typical instance before percentile calibration and cross-instance aggregation. Some instances retained signal. For example, `dbservice2` had a raw test ROC-AUC of 0.8877 and zero windows above the shared percentile cutoff. These diagnostics leave the effect of each preprocessing and scoring choice unmeasured. [Window predictions](project/experiments/runs/gaia_matched_20261122_183655/comparison/predictions.csv) · [Evaluation summary](project/experiments/runs/gaia_matched_20261122_183655/comparison/summary.json)

## 6. Decisions

| Decision | Recorded outcome |
| --- | --- |
| Dates and time zone | 11 assigned July dates in UTC+8, with retained windows on ten. Validation starts July 22 and test starts July 24. |
| Node granularity and topology | Ten service instances with directed parent-to-child calls from the first common training trace hour and one self-loop per instance. |
| Features and windows | Seven metric channels, mean trace span latency, and five fixed business-log severity counts. Thirty-second bins in non-overlapping two-minute windows. |
| Label and exclusion rules | Confirmed memory and file-moving intervals supply positives. Mask bounded CPU and permission intervals, exclude July 26 onward, and leave two-minute margins around both cuts. |
| AnoFusion benchmark | Use the published D1 scores as a reference and run the pinned network under the same system-window scoring as UG-MoE. |
| GAIA experiment scope | Full UG-MoE v2 at seed 42 and one adapted AnoFusion run at seed 42 across the ten instances. |

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

The [archive downloader](scripts/fetch_gaia_archives.sh) reads expected file paths and blob hashes from the pinned release. It fetches missing parts through a temporary Git partial clone over public HTTPS with credentials disabled. It reuses existing verified files and renames a temporary download only after its hash matches. Once all parts are present, it checks each archive without extracting its CSVs. Allow time for about 3.7 GiB of remaining downloads and the integrity checks. [Git partial clone](https://git-scm.com/docs/partial-clone)

```bash
cd "$HOME/AHA_BTP/UG_MoE/project"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate muad
GAIA_LOG="experiments/runs/gaia_fetch_telemetry_v1_git_$(date +%Y%m%d_%H%M%S).log"

nohup bash ../scripts/fetch_gaia_archives.sh > "$GAIA_LOG" 2>&1 < /dev/null &

echo "PID=$!  Log=$GAIA_LOG"
tail -f "$GAIA_LOG"
```

### Run the streaming coverage audit

Run after `All three telemetry archives verified` appears in the download log. The scan reads the selected metric members and the trace and business-log members without writing expanded CSVs. Use the report to choose the chronological split and identify days with insufficient coverage.

```bash
cd "$HOME/AHA_BTP/UG_MoE/project"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate muad
GAIA_AUDIT="experiments/runs/gaia_coverage_$(date +%Y%m%d_%H%M%S)"

nohup "$CONDA_PREFIX/bin/python" -u -m data_pipeline.audit_gaia \
  --output "$GAIA_AUDIT/coverage.json" > "$GAIA_AUDIT.log" 2>&1 < /dev/null &

echo "PID=$!  Log=$GAIA_AUDIT.log"
tail -f "$GAIA_AUDIT.log"
```

The job finishes with `GAIA coverage audit saved to ...`. Record the JSON and log paths. Use the full scan for split decisions. The `--rows-per-member` option is intended for bounded checks.

### Build the GAIA dataset

Use the completed coverage report. The builder reads the verified archives in place and writes the compact processed features to `datasets/gaia/processed/micross_v1`.

```bash
cd "$HOME/AHA_BTP/UG_MoE/project"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate muad
GAIA_BUILD_LOG="experiments/runs/gaia_build_$(date +%Y%m%d_%H%M%S).log"

nohup "$CONDA_PREFIX/bin/python" -u -m data_pipeline.build_gaia \
  --coverage experiments/runs/gaia_coverage_20261122_155845/coverage.json \
  > "$GAIA_BUILD_LOG" 2>&1 < /dev/null &

echo "PID=$!  Log=$GAIA_BUILD_LOG"
tail -f "$GAIA_BUILD_LOG"
```

The completion marker is `GAIA dataset saved to ...`. Verify the four artifacts and record the class counts and graph from `quality.json` before running seed 42.

### First seed-42 training job

Run after the processed data and split pass the checks above.

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

The `Saved to:` line identifies the timestamped results directory. Check `models["ug_moe2:full"].errors` in `summary.json`, the epoch history in `ug_moe2_full_seed42.json`, and the corresponding `_best.pth` file. Record the executed launch and test metrics in the journal, then report precision, recall, and anomaly F1 alongside the published reference.

### Run the matched GAIA comparison

Run from `project/` with `muad` active. The input export reads the frozen dataset and streams the business archive. Retain the `GAIA_MATCH` value for the later commands.

```bash
cd "$HOME/AHA_BTP/UG_MoE/project"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate muad

GAIA_MATCH="experiments/runs/gaia_matched_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$GAIA_MATCH"

nohup "$CONDA_PREFIX/bin/python" -u -m data_pipeline.export_gaia_anofusion \
  --output "$GAIA_MATCH/observations.npz" \
  > "$GAIA_MATCH/export.log" 2>&1 < /dev/null &
echo "PID=$!  Log=$GAIA_MATCH/export.log"
tail -f "$GAIA_MATCH/export.log"
```

After `Exported 5520 GAIA windows` appears, use the available GPU and the pinned checkout at `experiments/runs/gaia_anofusion_source`. The runner verifies the checkout revision and tracks the completed scores for all ten instances.

To recreate that checkout in a fresh workspace, fetch the pinned public source and materialize only its model directory.

```bash
(
  set -euo pipefail
  export GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=/bin/false
  ANO_SOURCE=experiments/runs/gaia_anofusion_source
  ANO_REV=ae3c4ea98509008844a3eecf7c0bd1529faaba67
  git -c credential.helper= -c protocol.version=2 clone --depth=1 --filter=blob:none --no-checkout \
    https://github.com/AIOps-Lab-NKU/AnoFusion.git "$ANO_SOURCE"
  git -C "$ANO_SOURCE" -c credential.helper= -c protocol.version=2 fetch --depth=1 --filter=blob:none origin "$ANO_REV"
  git -C "$ANO_SOURCE" sparse-checkout init --cone
  git -C "$ANO_SOURCE" sparse-checkout set model
  git -C "$ANO_SOURCE" -c credential.helper= -c protocol.version=2 checkout --detach "$ANO_REV"
)
```

```bash
nvidia-smi
read -r -p 'Available GPU index: ' GAIA_GPU

CUDA_VISIBLE_DEVICES="${GAIA_GPU:?Enter an available GPU index}" \
  nohup "$CONDA_PREFIX/bin/python" -u -m experiments.run_gaia_anofusion \
  --source experiments/runs/gaia_anofusion_source \
  --observations "$GAIA_MATCH/observations.npz" \
  --output-dir "$GAIA_MATCH/anofusion" \
  --device cuda --seed 42 \
  > "$GAIA_MATCH/anofusion.log" 2>&1 < /dev/null &
echo "PID=$!  Log=$GAIA_MATCH/anofusion.log"
tail -f "$GAIA_MATCH/anofusion.log"
```

After `AnoFusion scores saved to` appears, run the matched evaluator on CPU.

```bash
export DGLBACKEND=pytorch
CUDA_LIBS="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="$CUDA_LIBS/cuda_nvrtc/lib:$CUDA_LIBS/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
UG_RUN=experiments/runs/gaia_full_s42_20261122_175126/baselines_20261122_175136

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  nohup "$CONDA_PREFIX/bin/python" -u experiments/compare_gaia.py \
  --ug-run "$UG_RUN" \
  --ano-run "$GAIA_MATCH/anofusion" \
  --output-dir "$GAIA_MATCH/comparison" \
  > "$GAIA_MATCH/comparison.log" 2>&1 < /dev/null &
echo "PID=$!  Log=$GAIA_MATCH/comparison.log"
tail -f "$GAIA_MATCH/comparison.log"
```

The evaluator writes `comparison/summary.json` and `comparison/predictions.csv`. Record the final run paths and the shared-window results here.

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
| 2026-11-22 | First telemetry download timed out | The raw GitHub route stalled on `business_split.z01` at connection setup. An independent Git HTTPS probe downloaded that 51,380,224-byte volume and matched its pinned Git blob hash. Cached the verified part and switched the downloader to Git HTTPS with its temporary object store on `/tmp`. A small split-archive fixture verified retry, reuse, integrity checks, and scratch cleanup. | [Failed download log](project/experiments/runs/gaia_fetch_telemetry_v1_20261122_152540.log) · [Retry command](#fetch-and-verify-the-compressed-telemetry). Retry and verify all three archives. |
| 2026-11-22 | Freed space for GAIA telemetry | At the user's request, removed the 2.1 GiB extracted Eadro `raw/` directory after verifying both pinned source archive checksums and checking the processed SN and TT files. | `/DATA` showed 8.9 GiB free after removal. The Eadro directory now uses 443 MiB for archives and processed data. |
| 2026-11-22 | All telemetry archives verified | The user ran the revised downloader. All 81 parts were present and all three archives passed `7z t`. `/DATA` showed 7.5 GiB free after completion. | [Bulk download log](project/experiments/runs/gaia_fetch_telemetry_v1_git_20261122_153224.log). Keep the archives compressed. |
| 2026-11-22 | Clock and coverage checks | Confirmed the ten common application service IDs and gathered evidence for UTC+8 from memory injections, embedded business-log times, and trace-to-log matches. Added a streaming coverage audit and checked 128 rows from 160 real archive members plus a complete metric member in `muad`. | [Clock evidence](#archive-schemas-and-clock-evidence) · [Coverage command](#run-the-streaming-coverage-audit). Run the full coverage audit before freezing the split. |
| 2026-11-22 | AnoFusion baseline added | The user selected the KDD 2023 GAIA benchmark paper for comparison. Verified its Table 3 scores, checked the public source tree at `ae3c4ea98509008844a3eecf7c0bd1529faaba67`, and confirmed that our trace count matches its D1 count. Compared the released `mobservice2` labels with the raw memory injection intervals in `muad`. Specified a matched system-window rerun and tracked the published instance-level result. | [AnoFusion comparison](#anofusion-comparison). Freeze shared windows and labels after the coverage scan. |
| 2026-11-22 | GAIA schedule narrowed | The user confirmed a single full UG-MoE v2 training run at seed 42. The comparison will use AnoFusion's published D1 metrics. The prior rerun plan was superseded. | [Experiment sequence](#5-experiment-sequence). Finish coverage, build the pilot and full dataset, then train seed 42. |
| 2026-11-22 | Full coverage audit completed | The user completed the streaming scan. Combined its per-service coverage with the injection records to select 11 eligible dates. A preflight on all selected real metric members and the real labels yielded 5,520 usable windows with both classes in every split. | [Coverage JSON](project/experiments/runs/gaia_coverage_20261122_155845/coverage.json) · [Split and window counts](#chronological-partition-and-window-labels). |
| 2026-11-22 | GAIA adapter implemented and checked | Added archive streaming, time-bin aggregation, a training trace graph, and an explicit split manifest. A bounded real trace probe found 30 directed cross-service links. In `muad`, the changed Python files compiled, and a six-window compressed fixture using the real audit and labels passed feature-shape, class, and split checks. The shared loader wrote its training-only normalization statistics, and full v2 produced finite CPU logits and loss for one batch from each split. | [Builder](project/data_pipeline/build_gaia.py) · [Full build command](#build-the-gaia-dataset). The full telemetry build and GPU training are pending. |
| 2026-11-22 | Full GAIA build completed and verified | The user ran the build on `hdfs` at `4caf3f9` in `muad`. It completed in about 24 minutes and saved 5,520 windows. Checked every saved feature array for correct shape and finite values, verified all window IDs against the disjoint chronological split and cut margins, and confirmed 30 cross-service arcs and ten self-loops. Checked the saved normalization against the training data and ran full UG-MoE v2 on one real CPU batch of 50 from each split with finite outputs and loss. | [Build log](project/experiments/runs/gaia_build_20261122_170933.log) · [Quality report](project/datasets/gaia/processed/micross_v1/quality.json) · [Seed-42 launch](#first-seed-42-training-job). GPU training is pending. |
| 2026-11-22 | Seed-42 launches recorded | The initial launch stopped at CUDA initialization. The retry completed and its artifacts passed the checkpoint, split, history, and metric checks. | [Run log](#run-log). |
| 2026-11-22 | Published comparison recorded | Paired the verified seed-42 scores with AnoFusion's reported GAIA D1 scores and documented the evaluation protocols. | [AnoFusion comparison](#anofusion-comparison). |
| 2026-11-22 | Matched comparison implemented | Added the input exporter, pinned AnoFusion runner, and shared window scorer. The three scripts compiled in `muad`. A small compressed fixture exercised the exporter and one CPU training epoch for each of the ten instances using the public network. Mock AnoFusion scores on the real manifest exercised the complete window join and reproduced the saved UG-MoE F1 at its original threshold. Checked the validation threshold search against an exhaustive sweep. | The pinned source checkout is available in the ignored `project/experiments/runs/gaia_anofusion_source` directory. Start the full export and baseline training with the [matched run commands](#run-the-matched-gaia-comparison). |
| 2026-11-22 | Matched input export completed | The user ran the export in `muad` at `3717663`. It took about 17 minutes and wrote 5.8 MiB for 5,520 windows, four bins, ten instances, and 13 channels. Verified that every value was finite, the manifest and window order matched, and every window had three or four forecast targets. Per instance there are 12,646 train, 4,430 validation, and 4,582 test targets. A short CPU check with the real `mobservice2` data passed with one batch per split. | [Export log](project/experiments/runs/gaia_matched_20261122_183655/export.log). Observation SHA-256 is `50d643f77ab5741e48e3a083081a53709e84231711eeea64bac841f2f90dcbb5`. Launch AnoFusion training at seed 42. |
| 2026-11-22 23:01 IST | AnoFusion training in progress | The seed-42 CUDA job had saved checkpoints and scores for `dbservice1`, `dbservice2`, `logservice1`, and `logservice2`, with best validation-forecast epochs 43, 100, 59, and 46 respectively. Verified the four score files against the frozen manifest and checked their 4,430 validation and 4,582 test scores per instance. `mobservice1` was at epoch 56. | [Training log](project/experiments/runs/gaia_matched_20261122_183655/anofusion.log). Run the shared evaluator after all ten score files and the runner summary are saved. |
| 2026-11-22 23:12 IST | Continued on a single GPU | The spare launch for `redisservice2`, `webservice1`, and `webservice2` stopped at CUDA initialization with `No CUDA GPUs are available`. The original seed-42 process on GPU 1 remains active with five completed instances and `mobservice2` training. The user confirmed that it will train all ten instances sequentially. | [Spare launch log](project/experiments/runs/gaia_matched_20261122_183655/anofusion_tail.log) · [Original training log](project/experiments/runs/gaia_matched_20261122_183655/anofusion.log). Evaluate the original `anofusion` directory once all ten score files and its summary are saved. |
| 2026-11-23 03:32 IST | AnoFusion training completed | The original seed-42 job saved all ten checkpoints and score files on GPU 1. Verified finite checkpoint parameters, the best validation forecast epoch for every instance, the frozen manifest hash, and matching timestamps for all 4,430 validation and 4,582 test forecasts per instance. | [Training log](project/experiments/runs/gaia_matched_20261122_183655/anofusion.log) · [AnoFusion run summary](project/experiments/runs/gaia_matched_20261122_183655/anofusion/summary.json). |
| 2026-11-23 | Matched evaluation completed | Ran the shared evaluator on CPU in `muad` with the original AnoFusion output and the saved UG-MoE checkpoint. Verified all 2,335 prediction rows against the frozen split and labels, checked each validation threshold by exhaustive sweep, and recomputed the reported metrics from the CSV. | [Matched results](#matched-seed-42-results) · [Evaluator log](project/experiments/runs/gaia_matched_20261122_183655/comparison.log). |
| 2026-11-23 | Audited the gap to published AnoFusion | Compared the scoring implementation with the pinned demo and paper. Recomputed the system-window score aggregation and checked the raw anomaly ranking against per-instance injection labels. | [Protocol differences and diagnostic results](#why-the-adapted-anofusion-score-differs-from-table-3). |

For a code or data update, record the date, commit, affected files, decision and rationale, exact artifacts, verification performed, and next unresolved item. For any failed attempt, keep the command and log path and add the cause and recovery once confirmed.

### Run log

#### 2026-11-22 17:47 IST, CUDA initialization failure

The runner requested CUDA for `ug_moe2:full` at seed 42 and exited the model run before epoch 1 with `No CUDA GPUs are available`. The launch produced no model checkpoint or result. The user retried in a separate run. [First launch log](project/experiments/runs/gaia_full_s42_20261122_174745.log)

#### 2026-11-22 17:51–18:06 IST, seed-42 retry

The user ran in `muad` on `hdfs` with the code at `ac89cb2`, the pinned GAIA release, and the manifest recorded above. The runner saved the selected epoch, checkpoint, split, and training statistics. The GPU index was not captured in the artifacts. The runner arguments below are from the saved config. [Launch log](project/experiments/runs/gaia_full_s42_20261122_175126.log) · [Runner summary](project/experiments/runs/gaia_full_s42_20261122_175126/baselines_20261122_175136/summary.json)

```bash
# From project/ with muad active and CUDA visibility set as above.
nohup "$CONDA_PREFIX/bin/python" -u experiments/run_baselines.py \
  --device cuda \
  --data-dir datasets/gaia/processed/micross_v1 \
  --models ug_moe2:full \
  --seeds 42 \
  --output-dir experiments/runs/gaia_full_s42_20261122_175126 \
  > experiments/runs/gaia_full_s42_20261122_175126.log 2>&1 < /dev/null &
```

Artifacts are in `project/experiments/runs/gaia_full_s42_20261122_175126/baselines_20261122_175136/`. [Checkpoint](project/experiments/runs/gaia_full_s42_20261122_175126/baselines_20261122_175136/ug_moe2_full_seed42_best.pth) · [Saved split](project/experiments/runs/gaia_full_s42_20261122_175126/baselines_20261122_175136/split.json) · [Normalization](project/experiments/runs/gaia_full_s42_20261122_175126/baselines_20261122_175136/normalization.json)
