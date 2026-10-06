# Dataset Verification Report

## 1. Dataset Source
- **Official Source**: Zenodo DOI 10.5281/zenodo.7615394 (Eadro SocialNetwork Dataset)
- **Archive**: SN Dataset.zip (78,753,809 bytes, MD5: 4c523faf64770b1e9a709cdd610e07c4)
- **Download Date**: 2026-10-02
- **Referenced Repositories**:
  - Eadro: github.com/BEbillionaireUSD/Eadro (commit 82ff9c9 - latest)
  - MUAD: github.com/slg-frank/muad (commit 6031ccf - for recovered artifacts)

## 2. Version/Commit Information
| Component | Version/Commit | Source |
|---|---|---|
| Raw Dataset | Zenodo record 7615394 | Downloaded directly |
| Eadro Preprocessing Code | commit 82ff9c9 | github.com/BEbillionaireUSD/Eadro |
| MUAD Model Code (baseline) | commit 6031ccf | github.com/slg-frank/muad |
| Recovered Chunk Artifacts | commit 6031ccf | MUAD repo Dataset/chunk/ |

## 3. Directory Structure Verification

### Raw Archive Structure (SN Dataset.zip)
```
SN Dataset/
├── SN-all.tgz                    (28,003,635 bytes - REDUNDANT)
├── data/                         (4 fault experiments)
│   ├── SN.fault-2022-04-17T181245D2022-04-17T183616.json
│   ├── SN.fault-2022-04-17T183729D2022-04-17T190100.json
│   ├── SN.fault-2022-04-17T190213D2022-04-17T192544.json
│   ├── SN.fault-2022-04-17T192658D2022-04-17T195031.json
│   └── SN.<same-4-timestamps>/   each containing: logs.json, spans.json, metrics/ (12 CSVs)
└── no fault/                     (3 fault-free experiments)
    ├── SN.fault-2022-04-20T182405D2022-04-20T184806.json   (73 bytes, empty faults)
    ├── SN.fault-2022-04-21T105249D2022-04-21T111651.json   (76 bytes, empty faults)
    ├── SN.fault-2022-04-21T153302D2022-04-21T155703.json   (76 bytes, empty faults)
    └── SN.<same-3-timestamps>.tar.xz   each extracting to: logs.json, spans.json, metrics/ (12 CSVs)
```

**Verification**: ✅ MATCHES forensic document §4 exactly. SN-all.tgz confirmed redundant (contains same 4 fault experiments).

## 4. Number of Services
**Expected**: 12 services (per Eadro util.py and raw archive)
**Verified**: 12 services per experiment across all 7 experiments

| Node ID | Service Name | Metrics CSV | Fault Target Name | Jaeger serviceName |
|---|---|---|---|---|
| 0 | social-graph-service | social-graph-service.csv | socialnetwork-social-graph-service-1 | social-graph-service |
| 1 | compose-post-service | compose-post-service.csv | socialnetwork-compose-post-service-1 | compose-post-service |
| 2 | post-storage-service | post-storage-service.csv | socialnetwork-post-storage-service-1 | post-storage-service |
| 3 | user-timeline-service | user-timeline-service.csv | socialnetwork-user-timeline-service-1 | user-timeline-service |
| 4 | url-shorten-service | url-shorten-service.csv | socialnetwork-url-shorten-service-1 | url-shorten-service |
| 5 | user-service | user-service.csv | socialnetwork-user-service-1 | user-service |
| 6 | media-service | media-service.csv | socialnetwork-media-service-1 | media-service |
| 7 | text-service | text-service.csv | socialnetwork-text-service-1 | text-service |
| 8 | unique-id-service | unique-id-service.csv | socialnetwork-unique-id-service-1 | unique-id-service |
| 9 | user-mention-service | user-mention-service.csv | socialnetwork-user-mention-service-1 | user-mention-service |
| 10 | home-timeline-service | home-timeline-service.csv | socialnetwork-home-timeline-service-1 | home-timeline-service |
| 11 | nginx-web-server | nginx-web-server.csv | socialnetwork-nginx-thrift-1 | nginx-web-server |

**Alias Note**: Node 11 has 3-way naming: `nginx-web-server` (metrics/Jaeger serviceName), `nginx-thrift` (ChaosBlade fault target, Jaeger hostname tag)

**Verification**: ✅ 12 services consistent across all experiments

## 5. Experiments
**Total**: 7 distinct experiment recordings
- **Fault experiments**: 4 (each ~1410-1413 seconds, ~23.5 min)
- **No-fault experiments**: 3 (each ~1441 seconds, ~24.02 min)
- **Total no-fault duration**: ~1.2 hours (matches Eadro paper claim)

## 6. Fault Scenarios
**Total fault records**: 36 (verified by direct JSON parsing)

| Experiment | Duration | Faults | Services Targeted | Fault Types |
|---|---|---|---|---|
| SN.fault-2022-04-17T181245D2022-04-17T183616.json | 1410.3s | 9 | text-service, home-timeline-service, media-service | cpu_load, network_delay, network_loss |
| SN.fault-2022-04-17T183729D2022-04-17T190100.json | 1410.3s | 9 | post-storage-service, social-graph-service, url-shorten-service | cpu_load, network_delay, network_loss |
| SN.fault-2022-04-17T190213D2022-04-17T192544.json | 1411.1s | 9 | nginx-thrift, unique-id-service, user-service | cpu_load, network_delay, network_loss |
| SN.fault-2022-04-17T192658D2022-04-17T195031.json | 1413.0s | 9 | compose-post-service, user-timeline-service, user-mention-service | cpu_load, network_delay, network_loss |

**Fault-type distribution**: cpu_load: 12, network_delay: 12, network_loss: 12
**Per-service distribution**: Each of 12 services receives exactly 3 faults (one per type)
**Fault duration**: All 36 records have duration = 120 seconds (2 minutes)
**Timing gaps**: ~33-43 seconds between consecutive faults (not exactly 30s as paper states)

**No-fault experiments**: 3 experiments with empty `faults: []` arrays, ~1.2hr total

**Discrepancy Note**: Eadro paper claims 72 fault injections for SN; raw archive contains 36. Cause UNKNOWN.

## 7. Available Modalities
All 3 modalities present in every experiment:

| Modality | Raw Format | Verified Schema |
|---|---|---|
| **Metrics** | CSV per service (12 files/exp) | `timestamp,cpu_usage_system,cpu_usage_total,cpu_usage_user,memory_usage,memory_working_set,rx_bytes,tx_bytes` — 7 metrics, 1 row/sec, Unix epoch integer timestamps |
| **Logs** | JSON keyed by service (12 keys) | Raw strings: `[YYYY-Mon-DD HH:MM:SS.ffffff] <level>: (file:line:func) message` — **UTC+8 local time** (8-hour offset from Unix epoch) |
| **Traces** | spans.json (Jaeger format) | List of trace objects with `spans[]` (microsecond startTime/duration), `processes` map (processID→serviceName) |

## 8. Recovered MUAD Chunk Artifact Verification

### metadata.json
```json
{
    "chunk_lenth": 10,
    "chunk_num": 5610,
    "edges": [[1,1,1,1,1,1,1,1,10,10,10,2,0,0,7,7,7,5,3,11,11,11,11,11],
              [1,10,6,2,7,8,5,3,10,2,0,2,0,5,7,4,9,5,3,1,10,11,0,5]],
    "event_num": 15,
    "metric_num": 7,
    "node_num": 12
}
```

**Note on `event_num`**: Recovered artifact has `event_num: 15` but our Drain reconstruction produced 13 templates. This is an **expected difference** (per forensic doc §12/§17) since Drain parser configuration/version affects template count. The shape rule `[12, event_num]` is exact-match; the value of `event_num` may differ.

### Chunk Counts
- chunk_train.pkl: 3,366 chunks
- chunk_test.pkl: 2,244 chunks
- Total: 5,610 (matches chunk_num)
- Split ratio: 60.0% / 40.0% (matches paper, NOT Eadro code default of 30/70)

### Per-Chunk Tensor Shapes
- metrics: (12, 10, 7) ✅
- traces: (12, 10, 1) ✅ (NOTE: Eadro code allocates 2 channels, channel 1 always zero — trimmed for MUAD)
- logs: (12, 15) ✅
- culprit: int

### Culprit Label Distribution (all 5,610 chunks)
| culprit | Service | Count |
|---|---|---|
| 0 | social-graph-service | 953 |
| 1 | compose-post-service | 1,537 |
| 4 | url-shorten-service | 1,560 |
| 5 | user-service | 1,560 |
| -1 (normal) | — | 0 |

**Critical Finding**: Zero chunks with `culprit == -1` despite 3 genuine no-fault experiments (~1.2hr) in raw data. This is an **unresolved discrepancy** (forensic doc §12/§17).

### Binary Label Transformation (MUAD code)
```python
y_anomaly = (fault_indexs >= 1).long()
```
Result: culprit 0 → label 0 ("normal"); culprit {1,4,5} → label 1 ("anomaly")
Distribution: 4,657 anomaly (83.0%) / 953 normal (17.0%)

## 9. Graph Structure
- **Nodes**: 12 (fixed)
- **Edges**: 24 directed edges (from metadata.json, matches Eadro util.py edge_info)
- **Self-loops**: Present on nodes 1 and 11 at minimum
- **allow_zero_in_degree**: True (required by GATv2Conv)

## 10. Verification Summary

| Check | Status | Notes |
|---|---|---|
| Raw archive structure | ✅ PASS | Matches forensic doc §4 exactly |
| Fault record count | ✅ PASS | 36 records verified |
| Service consistency | ✅ PASS | 12 services across all 7 experiments |
| Modality completeness | ✅ PASS | Metrics, logs, traces all present |
| Recovered artifact shapes | ✅ PASS | Matches forensic doc §12 |
| Train/test split ratio | ✅ PASS | 60/40 (requires --test_ratio 0.4) |
| Graph structure | ✅ PASS | 12 nodes, 24 edges match |
| No-fault label discrepancy | ⚠️ FLAGGED | 0 chunks with culprit=-1 despite raw no-fault data |
| 72 vs 36 fault discrepancy | ⚠️ FLAGGED | Paper claims 72, raw data has 36 |

## 11. Next Steps
1. **L2 Raw → parsed_data reconstruction** (4 converters needed)
2. **L3 Eadro preprocessing** (run vendored code with --test_ratio 0.4)
3. **L3.5 MUAD compatibility trim** (traces[...,:1])
4. **Validation Gates 1-10** (compare regenerated chunks to recovered artifact)

---

**Report Generated**: 2026-10-02  
**Status**: Dataset acquisition and verification COMPLETE  
**Next Phase**: Raw dataset processing (Section 11 of master plan)