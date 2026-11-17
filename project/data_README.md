# Data

Two kinds of data feed the experiments. Model dimensions (`node_num`, `edges`, `event_num`,
`metric_num`) are always read from the directory's `metadata.json`, and every script takes
`--data-dir`. **Never compare models trained on different datasets or splits.**

## 1. `recovered_artifact/`: MUAD's processed SN chunks (default `--data-dir`)

- Source: MUAD repository (commit 6031ccf), built with Eadro's preprocessing.
- Files: `chunk_train.pkl` (3,366), `chunk_test.pkl` (2,244), `metadata.json`; `event_num` 15,
  12 services, 24 edges. `vendor/muad/Dataset/chunk/` is a byte-identical copy used by vendored MUAD.
- Labels: `culprit` = fault type: 0 normal, 1 cpu_load, 4 network_delay, 5 network_loss (decoded from
  the raw SN fault files); ≥1 = anomalous; ~83% of windows are anomalous.
- **Structure:** stride-1, 10-second windows from the 4 SN fault-injection runs only. The artifact's own
  train/test split is random and therefore **leaky** (83.9% of test windows have a 1-second-shifted
  training neighbour). `experiments/common.py` re-chains the chunks into their 4 runs and splits by run
  (`--split run`, default: train runs 0–1, val run 2, test run 3). See `PROJECT_STATUS.md` §3.12.
- Logs are Hawkes intensities (~6e-10); the loader's train-split z-score makes them usable.
- Kept for continuity with the MUAD paper. For new results prefer the rebuilt data below.

## 2. Eadro SN / TT rebuilt from raw (`data_pipeline/`)

Raw data: Zenodo record [10.5281/zenodo.7615394](https://zenodo.org/records/7615394)

| Archive | Size | MD5 | Runs |
|---|---|---|---|
| `SN Dataset.zip` | 78.8 MB | `4c523faf64770b1e9a709cdd610e07c4` | 4 fault (~23.5 min) + 3 fault-free |
| `TT Dataset.zip` | 48.9 MB | `2d77861e4a025781e598d51866fd6f4e` | 9 fault (~1.8 h) + 2 fault-free |

```bash
cd project
python -m data_pipeline.fetch_eadro                          # -> datasets/eadro/raw/{SN,TT}  (MD5-verified)
python -m data_pipeline.build_eadro --system SN              # -> datasets/eadro/processed/SN_w10_s1/
python -m data_pipeline.build_eadro --system TT --stride 5   # -> datasets/eadro/processed/TT_w10_s5/
python experiments/run_baselines.py --data-dir datasets/eadro/processed/SN_w10_s1 --device cuda
```
`project/datasets/` is gitignored. Each processed directory holds `chunks.pkl`, `metadata.json`
(services, edges, templates, runs, label map) and `quality.json` (per-run alignment diagnostics).

What the builder does (details in `data_pipeline/build_eadro.py`):
- **Windows:** length 10 s (`--window`), stride `--stride`; anomalous if overlapping a fault by ≥1 s
  (Eadro's rule). `culprit`: 0 normal, 1 cpu_load, 2 network_delay, 3 network_loss. Every chunk also
  stores `run`, `run_index`, `run_kind`, `t0`, `fault_type`, `culprit_service` and `trace_frac`.
- **Metrics:** 7 container metrics per service per second, **raw** (normalization happens in the loader
  with train-split statistics only).
- **Traces:** log1p of the mean span latency (ms) per service per second. Span clocks are offset by whole
  hours (+8 h for SN); the offset is detected per run and out-of-window spans are dropped. In SN fault
  runs only 58–74% of seconds have any span, versus 93–95% in fault-free runs: faults suppress traffic.
  **TT's span files only cover about the first hour of each 1.8-hour run**, so later windows have no
  trace data. `trace_frac` records the coverage per window.
- **Logs:** Drain3 templates (numbers / IPs / ids masked first), per-window per-service counts of the
  top-30 templates plus an "other" bucket, log1p (SN: 12 templates, so `event_num` = 13). These are
  counts, not MUAD's Hawkes intensities. Templates are mined over all runs, which is unsupervised and
  label-free.
- **Graph:** Eadro's service dependency graph (SN: 12 services / 24 edges, identical to
  `recovered_artifact`; TT: 27 services / 44 edges).

**Split** (`experiments/common.assign_pipeline_runs`, automatic for these directories): fault runs in
chronological order, with the last ⌈20%⌉ as test, the ⌈20%⌉ before them as validation and the rest as
training. Fault-free runs: with at least 3 (SN) they are assigned whole, round-robin train, test, val;
with fewer (TT has 2) each one is cut chronologically into 60 / 20 / 20% train / val / test segments
(windows straddling a cut are dropped), so every split contains normal operation. No window crosses splits.
- **SN** (stride 1): train = 2 fault runs + 1 normal run (4,236 windows), val = 1 fault + 1 normal,
  test = 1 fault + 1 normal (~2,835 each); about 53% normal. A trivial "all anomalous" predictor gets
  test F1 ≈ 0.58 / macro-F1 ≈ 0.29.
- **TT** (stride 5): train = 5 fault runs + the first 60% of both fault-free runs (8,397), val = 2 fault
  runs + 60–80% segments (3,228), test = 2 fault runs + last 20% segments (3,226); val / test ~68%
  anomalous, trivial test F1 0.809 / macro-F1 0.404. **TT traces cover only 2–4% of seconds** (sparse
  spans), so TT is effectively a metrics + logs dataset. The two TT fault-free runs have identical
  request counts but different files (MD5), consistent with a replayed load generator, not duplicates.
