# UG-MoE — Uncertainty-Guided Graph-Aware Mixture-of-Experts for Anomaly Detection

Multimodal (metrics + traces + logs) anomaly detection for microservice systems. Predicts
Normal vs. Anomalous per time window. Extends the MUAD baseline (vendored in `project/vendor/muad`)
by replacing its GAT-based graph representation with a Graph-Aware Mixture-of-Experts, adding
uncertainty-guided routing and uncertainty-aware adaptive fusion.

- Implementation plan: `7. UG-MoE_Implementation_Guide.md`
- Baseline paper (MUAD): `7. Uncertainty-Aware_Multimodal_Anomaly_Detection_for_Microservice_Systems_With_Active_Learning.pdf`
- **Current status, known issues and roadmap: `PROJECT_STATUS.md`**

## Quick start

```bash
cd project
python experiments/run_comparison.py --device cuda       # Track A (MUAD) vs Track B (UG-MoE)
python experiments/run_baselines.py  --device cuda       # GAT, plain MoE, Graph-MoE, MUAD, UG-MoE
python experiments/run_ablations.py  --device cuda       # UG-MoE + component ablations
python experiments/run_robustness.py --device cuda       # missing-modality + noise robustness
```

All scripts default to a single seed (42); pass `--seeds 42,123,...` for multi-seed runs.
Every run uses the same protocol (`project/experiments/common.py`): a validation split carved from
the training chunks, checkpoint selection on validation F1, and a single test evaluation. Outputs
go to `project/experiments/runs/<experiment>_<timestamp>/`.

## Layout

| Path | Purpose |
|---|---|
| `project/tracks/proposed_model/` | `GraphAwareMoE` (the proposed model) |
| `project/src/` | Model components: experts, routing, uncertainty, fusion |
| `project/tracks/baselines/` | GAT and Graph-MoE baselines |
| `project/experiments/` | Shared protocol (`common.py`), model registry (`registry.py`), experiment scripts |
| `project/recovered_artifact/` | Dataset used by all experiments (MUAD chunks: 12 services, 24 edges, `event_num=15`) |
| `project/preprocessing/` | Data pipelines (raw SN → parsed, MUAD-compatible chunks, HDFS) |
| `project/validation/` | Data validation gates (`run_all_gates.py`) |
| `project/vendor/` | Third-party code: MUAD baseline, Eadro |

See `project/data_README.md` for dataset provenance and why datasets must not be mixed.
