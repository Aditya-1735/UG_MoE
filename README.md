# UG-MoE — Graph-Aware Mixture-of-Experts for Anomaly Detection

Graph-aware Mixture-of-Experts (MoE) model for multimodal anomaly detection in
microservice systems. Extends MUAD (Multimodal Uncertainty-Aware Anomaly
Detection) with specialized experts, graph-aware + uncertainty-guided routing,
per-expert uncertainty blocks, and adaptive multimodal fusion.

## Quick Start

```bash
# Install
pip install -e .

# Single-seed smoke test (comparison Track A vs Track B)
python experiments/run_comparison.py --seeds 42 --device cuda

# Full ablation study (5 seeds × 6 configs)
python experiments/run_ablations.py --seeds 42,123,456,789,999 --device cuda

# Full Track A vs Track B comparison
python experiments/run_comparison.py --seeds 42,123,456,789,999 --device cuda
```

## Directory Layout

| Directory | Purpose |
|-----------|---------|
| `src/` | Original source: experts, routing, fusion, uncertainty |
| `tracks/proposed_model/` | GraphAwareMoE main model |
| `experiments/` | Training scripts (Track A, Track B, ablations, comparison, baselines) |
| `configs/` | Experiment configs |
| `vendor/` | Third-party code (MUAD baseline, Eadro) |
| `preprocessing/` | Data pipelines (HDFS, MUAD-compat, raw-to-parsed) |
| `project/recovered_artifact/` | Original MUAD dataset (event_num=15) |
| `project/checkpoints/` | Saved model checkpoints |
| `project/experiments/runs/` | Experiment outputs (gitignored) |

## Key Experiment Scripts

| Script | Purpose |
|--------|---------|
| `run_track_a_full.py` | Track A: MUAD faithful reproduction |
| `run_track_b_full.py` | Track B: Graph-Aware MoE proposed model |
| `run_track_a_al.py` | Track A: Active learning |
| `run_ablations.py` | 6 ablation configs × seeds |
| `run_comparison.py` | Track A vs Track B (apples-to-apples, paired t-test) |
| `run_gat_baseline.py` | GAT baseline |
| `run_plain_moe.py` | Plain MoE baseline |
| `run_graph_moe.py` | Graph-MoE baseline |
| `run_missing_modality.py` | Missing-modality robustness |
| `run_noise_robustness.py` | Noise injection robustness |
| `run_routing_analysis.py` | Expert routing weight analysis |
| `run_uncertainty_fusion_analysis.py` | Uncertainty/fusion weight analysis |
| `run_hdfs.py` | HDFS log-only mode |

## Dataset

The original MUAD dataset lives in `project/recovered_artifact/` (event_num=15,
12 service nodes, 24 dependency edges). See `project/data_README.md` for full
provenance and mixing warnings.

## Citation

If you use this code, please cite: [paper citation TBD]
