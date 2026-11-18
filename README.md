# UG-MoE: Uncertainty-Guided Graph-Aware Mixture-of-Experts for Anomaly Detection

Multimodal (metrics + traces + logs) anomaly detection for microservice systems: every 10-second window
is classified as normal or anomalous. UG-MoE extends the MUAD baseline (vendored in `project/vendor/muad`)
with shared specialised experts, graph- and uncertainty-guided routing per modality, a probabilistic
representation and uncertainty-aware fusion over the modalities.

- Implementation plan: `7. UG-MoE_Implementation_Guide.md`
- Baseline paper (MUAD): `7. Uncertainty-Aware_Multimodal_Anomaly_Detection_for_Microservice_Systems_With_Active_Learning.pdf`
- **What we built, how and why, with all results: `UG-MoE_Implementation_Report.md`**
- Status log, every experiment round, open issues: `PROJECT_STATUS.md`
- Evidence brief on MUAD vs UG-MoE (for the supervisor): `MUAD_vs_UG-MoE.md`

## Quick start (from `project/`, on a GPU machine)

```bash
cd project
# 1. data: download Eadro SN + TT from Zenodo (MD5-checked) and build windowed chunks
python -m data_pipeline.fetch_eadro
python -m data_pipeline.build_eadro --system SN
python -m data_pipeline.build_eadro --system TT --stride 5
SN=datasets/eadro/processed/SN_w10_s1; TT=datasets/eadro/processed/TT_w10_s5

# 2. main comparison (plan 8A), 5 seeds
python experiments/run_baselines.py --device cuda --data-dir $SN --seeds 42,123,456,789,999 \
    --models muad,gat,ug_moe2:plain,ug_moe2:graph,ug_moe2:full
# 3. component ablations (8B), modality ablation (8C), sensitivity (8H): same script, other --models, e.g.
python experiments/run_baselines.py --device cuda --data-dir $SN --models abl2:no_adaptive_fusion
python experiments/run_baselines.py --device cuda --data-dir $SN --models ug_moe2:full@traces+logs
python experiments/run_baselines.py --device cuda --data-dir $SN --models "ug_moe2:full#hidden_dim=32"
# 4. robustness (8D/8E) and routing / uncertainty analysis (8F/8G) from a saved checkpoint
python experiments/run_robustness.py --device cuda --data-dir $SN --model ug_moe2:full --checkpoint <best.pth>
python experiments/run_analysis.py   --device cuda --data-dir $SN --model ug_moe2:full --checkpoint <best.pth>
# 5. every table and figure (no GPU)
python experiments/make_report.py
```

Every script uses the same protocol (`project/experiments/common.py`):
- leak-free run-level train / validation / test split;
- normalization fitted on the training split only;
- checkpoint selected on validation macro-F1;
- the test set evaluated once.

Outputs go to `project/experiments/runs/<name>/<experiment>_<timestamp>/`.

## Layout

| Path | Purpose |
|---|---|
| `project/data_pipeline/` | Zenodo download + raw-to-chunk builder for Eadro SN / TT (`systems.py`, `fetch_eadro.py`, `build_eadro.py`) |
| `project/tracks/proposed_model/ug_moe_v2.py` | **UG-MoE v2**, the plan-faithful model (incremental steps + ablation switches) |
| `project/tracks/proposed_model/model.py`, `project/src/` | Legacy UG-MoE (`GraphAwareMoE`) and its components, kept for reproducing earlier results |
| `project/tracks/baselines/` | GAT and Graph-MoE baselines |
| `project/experiments/` | Shared protocol, model registry, experiment scripts, report generator |
| `project/recovered_artifact/` | MUAD's original SN chunks (kept for reproducing the MUAD paper's protocol) |
| `project/vendor/` | Third-party code: MUAD baseline, Eadro |

See `project/data_README.md` for dataset provenance, the split rules and why datasets must not be mixed.
