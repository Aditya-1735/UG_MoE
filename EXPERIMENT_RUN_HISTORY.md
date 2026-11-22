# UG-MoE experiment run history

This guide follows the GPU-server runs from the recovered artifact through the final five-seed report. Commands come from the archived launch instructions and shell excerpts. A recorded result directory comes from server output. Launch wrappers and GPU assignments are only known where the shell was pasted. In the examples, `GPU` stands for a slot chosen at launch. Historical defaults changed between rounds, so the same command on a newer revision can produce a different split or selected epoch.

## Working directory and artifact naming

Local changes were pushed to branch `hdfs`. Training and checkpoint evaluation ran on the server in the `muad` environment. Paths below are relative to the server’s `project/` directory.

```bash
cd ~/AHA_BTP/UG_MoE && git pull && cd project
```

| Artifact | Where to look |
|---|---|
| Console output | The file following `>` in the launch command. A chained job can write multiple experiments to one log. |
| Baseline or ablation training | `<output-dir>/baselines_<timestamp>/` holds `summary.json`, per-model/seed JSON and `<model_tag>_seed<seed>_best.pth`. For example, `ug_moe2:full` becomes `ug_moe2_full` in filenames. |
| Robustness evaluation | `<output-dir>/robustness_<model_tag>_<timestamp>/summary.json`. The input weights are the path passed to `--checkpoint`. |
| Routing and fusion analysis | `<output-dir>/analysis_<model_tag>_<timestamp>/analysis.json` and the routing, fusion and uncertainty plots. |
| Final report | `experiments/runs/report_<timestamp>/tables.md` and figures. Each invocation creates a fresh report directory. |

## Round 1 — recovered-artifact comparison and baselines

These early runs used the historical window split. The archived comparison used Track A for MUAD and Track B for legacy UG-MoE. The foreground commands below have no recorded redirected log.

```bash
python experiments/run_comparison.py --seeds 42 --device cuda
python experiments/run_baselines.py --models gat,plain_moe,graph_moe --device cuda
python experiments/run_robustness.py --model ug_moe --checkpoint experiments/runs/comparison_20261117_104222/track_b_seed42_best.pth --device cuda
python experiments/run_robustness.py --model muad --checkpoint experiments/runs/comparison_20261117_104222/track_a_seed42_best.pth --device cuda
```

The training summaries were `experiments/runs/comparison_20261117_104222/summary.json` and `experiments/runs/baselines_20261117_105856/summary.json`. A separate earlier attempt on `preprocessing/sn` failed with a matrix-shape error and was logged in `experiments/runs/sn_comparison.log`.

## Round 2 — normalized artifact baselines and robustness

Two slots ran in parallel. The first trained both tracks and evaluated their checkpoints. The second trained the other three baselines. These are historical window-split results.

```bash
CUDA_VISIBLE_DEVICES=2 nohup bash -c '
python experiments/run_baselines.py --models muad,ug_moe --device cuda --output-dir experiments/runs/round2_tracks &&
D=$(ls -d experiments/runs/round2_tracks/baselines_* | tail -1) &&
echo "=== robustness on $D" &&
python experiments/run_robustness.py --model ug_moe --checkpoint "$D/ug_moe_seed42_best.pth" --device cuda --output-dir experiments/runs/round2_tracks &&
python experiments/run_robustness.py --model muad --checkpoint "$D/muad_seed42_best.pth" --device cuda --output-dir experiments/runs/round2_tracks
' > experiments/runs/round2_tracks.log 2>&1 &
```

```bash
CUDA_VISIBLE_DEVICES=0 nohup python experiments/run_baselines.py --models gat,plain_moe,graph_moe --device cuda \
  --output-dir experiments/runs/round2_baselines > experiments/runs/round2_baselines.log 2>&1 &
```

Recorded result directories under `experiments/runs/`

- `round2_baselines/baselines_20261117_115645`
- `round2_tracks/baselines_20261117_115635`

## Round 3 — modality diagnostic on the old split

The original six-subset proposal was shortened to the three non-metric subsets below. These are issued commands. Completion of these reduced jobs is unverified. Round 4 superseded this comparison when overlapping train/test windows were identified.

```bash
cd ~/AHA_BTP/UG_MoE && git pull && cd project
CUDA_VISIBLE_DEVICES=2 nohup python experiments/run_baselines.py --device cuda --epochs 30 --patience 8 \
  --output-dir experiments/runs/round3_modality --models muad@traces,muad@logs,muad@traces+logs \
  > experiments/runs/round3_muad.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python experiments/run_baselines.py --device cuda --epochs 30 --patience 8 \
  --output-dir experiments/runs/round3_modality --models gat@traces,gat@logs,gat@traces+logs \
  > experiments/runs/round3_gat.log 2>&1 &
```

## Round 4 — run-level split on the recovered artifact

Two jobs shared the output root. Their separate logs identify which models each trained.

```bash
# Slot 1, GPU 2: the two large models
CUDA_VISIBLE_DEVICES=2 nohup python experiments/run_baselines.py --device cuda --output-dir experiments/runs/round4_runsplit \
  --models ug_moe,plain_moe > experiments/runs/round4_big.log 2>&1 &
# Slot 2, GPU 0: small baselines + the modality diagnostic
CUDA_VISIBLE_DEVICES=0 nohup python experiments/run_baselines.py --device cuda --output-dir experiments/runs/round4_runsplit \
  --models muad,gat,graph_moe,muad@traces,muad@logs,muad@traces+logs > experiments/runs/round4_small.log 2>&1 &
```

Recorded result directories under `experiments/runs/`

- `round4_runsplit/baselines_20261117_125410`

## Data build for rounds 5–16

The pipeline fetched Eadro, then wrote chunks.pkl, metadata.json and quality.json beneath the processed directory for each system. SN used 10-second windows at stride 1 and TT used 10-second windows at stride 5.

```bash
python -m data_pipeline.fetch_eadro
python -m data_pipeline.build_eadro --system SN
python -m data_pipeline.build_eadro --system TT --stride 5
```

## Round 5 — baselines on rebuilt SN and TT

The seed-42 SN checkpoint supplied the MUAD baseline for round 8. Round 5 TT had no normal windows in validation and was superseded by round 7. Its table is historical.

```bash
cd ~/AHA_BTP/UG_MoE/project
# SN rebuilt (balanced, 7 runs)
CUDA_VISIBLE_DEVICES=2 nohup python experiments/run_baselines.py --device cuda --data-dir datasets/eadro/processed/SN_w10_s1 \
  --output-dir experiments/runs/round5_sn --models ug_moe,plain_moe,muad,gat,graph_moe > experiments/runs/round5_sn.log 2>&1 &
# TT (27 services, metrics + logs effectively)
CUDA_VISIBLE_DEVICES=0 nohup python experiments/run_baselines.py --device cuda --data-dir datasets/eadro/processed/TT_w10_s5 \
  --output-dir experiments/runs/round5_tt --models ug_moe,plain_moe,muad,gat,graph_moe > experiments/runs/round5_tt.log 2>&1 &
```

Confirmed SN training directory: `experiments/runs/round5_sn/baselines_20261117_135658/`. TT training output follows `round5_tt/baselines_<timestamp>/` under the same runs root.

## Round 6 — incremental v2 training on SN

Five configurations were trained with seed 42 and the earlier anomaly-F1 selection rule. Their checkpoints supplied rounds 8 and 13.

```bash
CUDA_VISIBLE_DEVICES=1 nohup python experiments/run_baselines.py --device cuda \
  --data-dir datasets/eadro/processed/SN_w10_s1 --output-dir experiments/runs/round6_v2_sn \
  --models ug_moe2:plain,ug_moe2:graph,ug_moe2:prob,ug_moe2:ug_routing,ug_moe2:full \
  > experiments/runs/round6_v2_sn.log 2>&1 &
```

Confirmed training directory: `experiments/runs/round6_v2_sn/baselines_20261117_145646/`.

## Round 7 — TT baselines and incremental v2

The revised protocol segmented the two normal TT runs and selected checkpoints on validation macro-F1. The logged train/validation/test sizes were 8,397 / 3,228 / 3,226. The baseline command used the runner’s default model list: MUAD, GAT, plain MoE, Graph-MoE and legacy UG-MoE.

```bash
# slot A: TT baselines
CUDA_VISIBLE_DEVICES="${GPU:?set GPU to an available device index}" nohup python experiments/run_baselines.py --device cuda \
  --data-dir datasets/eadro/processed/TT_w10_s5 --output-dir experiments/runs/round7_tt \
  > experiments/runs/round7_tt.log 2>&1 &

# slot B: v2 incremental steps on TT
CUDA_VISIBLE_DEVICES="${GPU:?set GPU to an available device index}" nohup python experiments/run_baselines.py --device cuda \
  --data-dir datasets/eadro/processed/TT_w10_s5 --output-dir experiments/runs/round7_v2_tt \
  --models ug_moe2:plain,ug_moe2:graph,ug_moe2:prob,ug_moe2:ug_routing,ug_moe2:full \
  > experiments/runs/round7_v2_tt.log 2>&1 &
```

Recorded result directories under `experiments/runs/`

- `round7_tt/baselines_20261117_164202`
- `round7_v2_tt/baselines_20261117_164211`

## Round 8 — SN checkpoint robustness

The successful full-v2/MUAD evaluations used the round 6 and round 5 checkpoints. The initial v2 attempt failed during CUDA initialization. The retry instructions used the following launch with an explicit GPU and chained success check.

```bash
D=datasets/eadro/processed/SN_w10_s1
CK_V2=$(find experiments/runs/round6_v2_sn -name 'ug_moe2_full_seed42_best.pth' | head -1)
CK_MU=$(find experiments/runs/round5_sn -name 'muad_seed42_best.pth' | head -1)
CUDA_VISIBLE_DEVICES=1 nohup sh -c "
python experiments/run_robustness.py --device cuda --data-dir $D --model ug_moe2:full --checkpoint '$CK_V2' --output-dir experiments/runs/round8_rob_sn_v2 &&
python experiments/run_robustness.py --device cuda --data-dir $D --model muad --checkpoint '$CK_MU' --output-dir experiments/runs/round8_rob_sn_muad" \
  > experiments/runs/round8_rob_sn.log 2>&1 &
```

Recorded result directories under `experiments/runs/`

- `round8_rob_sn_v2/robustness_ug_moe2_full_20261117_172202`
- `round8_rob_sn_muad/robustness_muad_20261117_172957`

## Rounds 8–10 — shared SN queue

The second round 8 job evaluated the other four v2 checkpoints. Round 9 trained six component ablations. Round 10 trained MUAD with a learning rate of 0.001 and a 120-epoch cap. All three stages wrote to `round8_10_sn.log`.

```bash
cd ~/AHA_BTP/UG_MoE/project
D=datasets/eadro/processed/SN_w10_s1
R6=$(dirname $(find experiments/runs/round6_v2_sn -name 'ug_moe2_full_seed42_best.pth' | head -1)); echo "$R6"
CUDA_VISIBLE_DEVICES=1 nohup sh -c "
for s in plain graph prob ug_routing; do
  python experiments/run_robustness.py --device cuda --data-dir $D --model ug_moe2:\$s \
    --checkpoint $R6/ug_moe2_\${s}_seed42_best.pth --output-dir experiments/runs/round8_rob_sn_\$s;
done;
python experiments/run_baselines.py --device cuda --data-dir $D --output-dir experiments/runs/round9_abl2_sn \
  --models abl2:no_graph,abl2:no_moe,abl2:no_dynamic_routing,abl2:no_uncertainty_routing,abl2:no_probabilistic,abl2:no_adaptive_fusion;
python experiments/run_baselines.py --device cuda --data-dir $D --output-dir experiments/runs/round10_muad_paperhp_sn \
  --models muad --lr 1e-3 --epochs 120" \
  > experiments/runs/round8_10_sn.log 2>&1 &
```

Recorded result directories under `experiments/runs/`

- `round8_rob_sn_plain/robustness_ug_moe2_plain_20261117_174416`
- `round8_rob_sn_graph/robustness_ug_moe2_graph_20261117_175047`
- `round8_rob_sn_prob/robustness_ug_moe2_prob_20261117_175839`
- `round8_rob_sn_ug_routing/robustness_ug_moe2_ug_routing_20261117_180656`
- `round9_abl2_sn/baselines_20261117_181606`
- `round10_muad_paperhp_sn/baselines_20261117_202530`

## Round 11 — TT robustness for MUAD and GAT

```bash
T=datasets/eadro/processed/TT_w10_s5; R7=experiments/runs/round7_tt/baselines_20261117_164202
CUDA_VISIBLE_DEVICES="${GPU:?set GPU to an available device index}" nohup sh -c "
for m in muad gat; do python experiments/run_robustness.py --device cuda --data-dir $T --model \$m --checkpoint $R7/\${m}_seed42_best.pth --output-dir experiments/runs/round11_rob_tt_\$m; done" \
  > experiments/runs/round11.log 2>&1 &
```

Recorded result directories under `experiments/runs/`

- `round11_rob_tt_muad/robustness_muad_20261117_214022`
- `round11_rob_tt_gat/robustness_gat_20261117_214802`

## Round 12 — SN modality-subset training

```bash
CUDA_VISIBLE_DEVICES="${GPU:?set GPU to an available device index}" nohup python experiments/run_baselines.py --device cuda --data-dir datasets/eadro/processed/SN_w10_s1 \
  --output-dir experiments/runs/round12_modality_sn \
  --models ug_moe2:full@metrics,ug_moe2:full@traces,ug_moe2:full@logs,ug_moe2:full@traces+logs,ug_moe2:full@metrics+logs,ug_moe2:full@metrics+traces \
  > experiments/runs/round12.log 2>&1 &
```

Recorded result directories under `experiments/runs/`

- `round12_modality_sn/baselines_20261117_215903`

## Round 13 — SN routing, fusion and uncertainty

```bash
cd ~/AHA_BTP/UG_MoE && git pull && cd project
CK_V2=$(find experiments/runs/round6_v2_sn -name 'ug_moe2_full_seed42_best.pth' | head -1); echo "$CK_V2"
CUDA_VISIBLE_DEVICES="${GPU:?set GPU to an available device index}" nohup python experiments/run_analysis.py --device cuda --model ug_moe2:full \
  --data-dir datasets/eadro/processed/SN_w10_s1 --checkpoint "$CK_V2" \
  --output-dir experiments/runs/round13_analysis_sn > experiments/runs/round13_analysis_sn.log 2>&1 &
```

Recorded result directories under `experiments/runs/`

- `round13_analysis_sn/analysis_ug_moe2_full_20261117_191831`

## Round 14 — TT full-v2 robustness and analysis

```bash
T=datasets/eadro/processed/TT_w10_s5; CK=experiments/runs/round7_v2_tt/baselines_20261117_164211/ug_moe2_full_seed42_best.pth
CUDA_VISIBLE_DEVICES="${GPU:?set GPU to an available device index}" nohup sh -c "
python experiments/run_robustness.py --device cuda --data-dir $T --model ug_moe2:full --checkpoint $CK --output-dir experiments/runs/round14_rob_tt_v2 &&
python experiments/run_analysis.py --device cuda --data-dir $T --model ug_moe2:full --checkpoint $CK --output-dir experiments/runs/round14_analysis_tt" \
  > experiments/runs/round14.log 2>&1 &
```

Recorded result directories under `experiments/runs/`

- `round14_rob_tt_v2/robustness_ug_moe2_full_20261117_214144`
- `round14_analysis_tt/analysis_ug_moe2_full_20261117_215001`

## Round 15a — SN robustness without the probabilistic component

```bash
CK_NP=$(find experiments/runs/round9_abl2_sn -name 'abl2_no_probabilistic_seed42_best.pth' | head -1); echo "$CK_NP"
CUDA_VISIBLE_DEVICES="${GPU:?set GPU to an available device index}" nohup python experiments/run_robustness.py --device cuda --data-dir datasets/eadro/processed/SN_w10_s1 \
  --model abl2:no_probabilistic --checkpoint "$CK_NP" --output-dir experiments/runs/round15a_rob_sn_noprob \
  > experiments/runs/round15a.log 2>&1 &
```

Recorded result directories under `experiments/runs/`

- `round15a_rob_sn_noprob/robustness_abl2_no_probabilistic_20261117_214233`

## Round 15b — TT component ablations

```bash
CUDA_VISIBLE_DEVICES="${GPU:?set GPU to an available device index}" nohup python experiments/run_baselines.py --device cuda --data-dir datasets/eadro/processed/TT_w10_s5 \
  --output-dir experiments/runs/round15b_abl2_tt \
  --models abl2:no_graph,abl2:no_moe,abl2:no_dynamic_routing,abl2:no_uncertainty_routing,abl2:no_probabilistic,abl2:no_adaptive_fusion \
  > experiments/runs/round15b.log 2>&1 &
```

Recorded result directories under `experiments/runs/`

- `round15b_abl2_tt/baselines_20261117_215852`

## Round 16a — SN expert subsets and hidden dimensions

```bash
CUDA_VISIBLE_DEVICES="${GPU:?set GPU to an available device index}" nohup python experiments/run_baselines.py --device cuda --data-dir datasets/eadro/processed/SN_w10_s1 \
  --output-dir experiments/runs/round16a_sens_sn \
  --models "ug_moe2:full#experts=temporal-dependency,ug_moe2:full#experts=temporal-dependency-cross_modal,ug_moe2:full#hidden_dim=16,ug_moe2:full#hidden_dim=32,ug_moe2:full#hidden_dim=128" \
  > experiments/runs/round16a.log 2>&1 &
```

Recorded result directories under `experiments/runs/`

- `round16a_sens_sn/baselines_20261117_215920`

## Round 16b — SN KL weights and routing temperatures

```bash
CUDA_VISIBLE_DEVICES="${GPU:?set GPU to an available device index}" nohup python experiments/run_baselines.py --device cuda --data-dir datasets/eadro/processed/SN_w10_s1 \
  --output-dir experiments/runs/round16b_sens_sn \
  --models "ug_moe2:full#lambda_kl=0.001,ug_moe2:full#lambda_kl=0.1,ug_moe2:full#lambda_kl=1.0,ug_moe2:full#routing_temperature=0.5,ug_moe2:full#routing_temperature=2.0" \
  > experiments/runs/round16b.log 2>&1 &
```

Recorded result directories under `experiments/runs/`

- `round16b_sens_sn/baselines_20261118_003547`

## Round 16c — SN learning rates and window lengths

```bash
D=datasets/eadro/processed/SN_w10_s1
CUDA_VISIBLE_DEVICES="${GPU:?set GPU to an available device index}" nohup sh -c "
for lr in 1e-4 1e-3; do python experiments/run_baselines.py --device cuda --data-dir $D --output-dir experiments/runs/round16c_sens_sn_lr\$lr --models ug_moe2:full --lr \$lr; done;
for w in 5 20; do python -m data_pipeline.build_eadro --system SN --window \$w &&
  python experiments/run_baselines.py --device cuda --data-dir datasets/eadro/processed/SN_w\${w}_s1 --output-dir experiments/runs/round16c_sens_sn_w\$w --models ug_moe2:full; done" \
  > experiments/runs/round16c.log 2>&1 &
```

Recorded result directories under `experiments/runs/`

- `round16c_sens_sn_lr1e-4/baselines_20261118_003631`
- `round16c_sens_sn_lr1e-3/baselines_20261118_010247`
- `round16c_sens_sn_w5/baselines_20261118_013215`
- `round16c_sens_sn_w20/baselines_20261118_015218`

## Final training — five paired seeds

The final seed set was 42, 123, 456, 789 and 999. SN main training repeated all five under the revised protocol. TT main training reused seed 42 from round 7. The SN and TT component ablations reused seed 42 from rounds 9 and 15b, respectively. Each dataset combined its two new shards with any reused seed during reporting.

### Main comparisons

The middle launch wrote SN shard B and TT shard A to the same `final_main_b.log`. The other two launches had their own logs.

```bash
M=muad,gat,ug_moe2:plain,ug_moe2:graph,ug_moe2:full
CUDA_VISIBLE_DEVICES="${GPU:?set GPU to an available device index}" nohup python experiments/run_baselines.py --device cuda --data-dir datasets/eadro/processed/SN_w10_s1 \
  --seeds 42,123,456 --models $M --output-dir experiments/runs/final_sn_main_a > experiments/runs/final_sn_main_a.log 2>&1 &
```

```bash
M=muad,gat,ug_moe2:plain,ug_moe2:graph,ug_moe2:full
CUDA_VISIBLE_DEVICES="${GPU:?set GPU to an available device index}" nohup sh -c "
python experiments/run_baselines.py --device cuda --data-dir datasets/eadro/processed/SN_w10_s1 --seeds 789,999 --models $M --output-dir experiments/runs/final_sn_main_b;
python experiments/run_baselines.py --device cuda --data-dir datasets/eadro/processed/TT_w10_s5 --seeds 123 --models $M --output-dir experiments/runs/final_tt_main_a" \
  > experiments/runs/final_main_b.log 2>&1 &
```

```bash
M=muad,gat,ug_moe2:plain,ug_moe2:graph,ug_moe2:full
CUDA_VISIBLE_DEVICES="${GPU:?set GPU to an available device index}" nohup python experiments/run_baselines.py --device cuda --data-dir datasets/eadro/processed/TT_w10_s5 \
  --seeds 456,789,999 --models $M --output-dir experiments/runs/final_tt_main_b > experiments/runs/final_tt_main_b.log 2>&1 &
```

Recorded result directories under `experiments/runs/`

- `final_sn_main_a/baselines_20261118_022049`
- `final_sn_main_b/baselines_20261118_022159`
- `final_tt_main_a/baselines_20261118_054819`
- `final_tt_main_b/baselines_20261118_022257`

### Component ablations

```bash
A=abl2:no_graph,abl2:no_moe,abl2:no_dynamic_routing,abl2:no_uncertainty_routing,abl2:no_probabilistic,abl2:no_adaptive_fusion
# 2a: TT, the most needed (its single-seed ablation was unreliable)
CUDA_VISIBLE_DEVICES="${GPU:?set GPU to an available device index}" nohup python experiments/run_baselines.py --device cuda --data-dir datasets/eadro/processed/TT_w10_s5 --seeds 123,456 --models $A --output-dir experiments/runs/final_tt_abl_a > experiments/runs/final_tt_abl_a.log 2>&1 &
# 2b
CUDA_VISIBLE_DEVICES="${GPU:?set GPU to an available device index}" nohup python experiments/run_baselines.py --device cuda --data-dir datasets/eadro/processed/TT_w10_s5 --seeds 789,999 --models $A --output-dir experiments/runs/final_tt_abl_b > experiments/runs/final_tt_abl_b.log 2>&1 &
# 2c: SN
CUDA_VISIBLE_DEVICES="${GPU:?set GPU to an available device index}" nohup python experiments/run_baselines.py --device cuda --data-dir datasets/eadro/processed/SN_w10_s1 --seeds 123,456 --models $A --output-dir experiments/runs/final_sn_abl_a > experiments/runs/final_sn_abl_a.log 2>&1 &
# 2d
CUDA_VISIBLE_DEVICES="${GPU:?set GPU to an available device index}" nohup python experiments/run_baselines.py --device cuda --data-dir datasets/eadro/processed/SN_w10_s1 --seeds 789,999 --models $A --output-dir experiments/runs/final_sn_abl_b > experiments/runs/final_sn_abl_b.log 2>&1 &
```

Recorded result directories under `experiments/runs/`

- `final_tt_abl_a/baselines_20261118_071505`
- `final_tt_abl_b/baselines_20261118_072416`
- `final_sn_abl_a/baselines_20261118_094510`
- `final_sn_abl_b/baselines_20261118_103931`

## Final robustness — 35 checkpoint evaluations

The saved launcher was `/tmp/final_rob.sh`. It ran four SN and three TT configurations at each of five seeds. Its shared log was `experiments/runs/final_rob.log`. The commands show which training roots supplied the weights; the log records the exact checkpoint selected for each evaluation.

```bash
cat > /tmp/final_rob.sh <<'EOF'
cd ~/AHA_BTP/UG_MoE/project; R=experiments/runs
rob() {  # rob <data-dir> <model> <ckpt-tag> <out-name> <search dirs...>
  D=$1; M=$2; TAG=$3; OUT=$4; shift 4
  for s in 42 123 456 789 999; do
    CK=$(find "$@" -name "${TAG}_seed${s}_best.pth" 2>/dev/null | sort | tail -1)
    [ -z "$CK" ] && { echo "missing $TAG seed $s"; continue; }
    python experiments/run_robustness.py --device cuda --data-dir $D --model $M --seeds $s --checkpoint $CK --output-dir $R/final_rob_${OUT}_s$s
  done
}
SN=datasets/eadro/processed/SN_w10_s1; TT=datasets/eadro/processed/TT_w10_s5
rob $SN ug_moe2:full ug_moe2_full sn_full $R/final_sn_main_*
rob $SN ug_moe2:plain ug_moe2_plain sn_plain $R/final_sn_main_*
rob $SN muad muad sn_muad $R/final_sn_main_*
rob $SN abl2:no_probabilistic abl2_no_probabilistic sn_noprob $R/round9_abl2_sn $R/final_sn_abl_*
rob $TT ug_moe2:full ug_moe2_full tt_full $R/round7_v2_tt $R/final_tt_main_*
rob $TT muad muad tt_muad $R/round7_tt $R/final_tt_main_*
rob $TT gat gat tt_gat $R/round7_tt $R/final_tt_main_*
EOF
CUDA_VISIBLE_DEVICES="${GPU:?set GPU to an available device index}" nohup sh /tmp/final_rob.sh > experiments/runs/final_rob.log 2>&1 &
```

Recorded result directories under `experiments/runs/`

- `final_rob_sn_full_s42/robustness_ug_moe2_full_20261118_142358`
- `final_rob_sn_full_s123/robustness_ug_moe2_full_20261118_143202`
- `final_rob_sn_full_s456/robustness_ug_moe2_full_20261118_143934`
- `final_rob_sn_full_s789/robustness_ug_moe2_full_20261118_144605`
- `final_rob_sn_full_s999/robustness_ug_moe2_full_20261118_144851`
- `final_rob_sn_plain_s42/robustness_ug_moe2_plain_20261118_145620`
- `final_rob_sn_plain_s123/robustness_ug_moe2_plain_20261118_150230`
- `final_rob_sn_plain_s456/robustness_ug_moe2_plain_20261118_150841`
- `final_rob_sn_plain_s789/robustness_ug_moe2_plain_20261118_151454`
- `final_rob_sn_plain_s999/robustness_ug_moe2_plain_20261118_152111`
- `final_rob_sn_muad_s42/robustness_muad_20261118_152729`
- `final_rob_sn_muad_s123/robustness_muad_20261118_153431`
- `final_rob_sn_muad_s456/robustness_muad_20261118_154123`
- `final_rob_sn_muad_s789/robustness_muad_20261118_154818`
- `final_rob_sn_muad_s999/robustness_muad_20261118_155511`
- `final_rob_sn_noprob_s42/robustness_abl2_no_probabilistic_20261118_160205`
- `final_rob_sn_noprob_s123/robustness_abl2_no_probabilistic_20261118_160932`
- `final_rob_sn_noprob_s456/robustness_abl2_no_probabilistic_20261118_161454`
- `final_rob_sn_noprob_s789/robustness_abl2_no_probabilistic_20261118_162219`
- `final_rob_sn_noprob_s999/robustness_abl2_no_probabilistic_20261118_162950`
- `final_rob_tt_full_s42/robustness_ug_moe2_full_20261118_163715`
- `final_rob_tt_full_s123/robustness_ug_moe2_full_20261118_164631`
- `final_rob_tt_full_s456/robustness_ug_moe2_full_20261118_165552`
- `final_rob_tt_full_s789/robustness_ug_moe2_full_20261118_170518`
- `final_rob_tt_full_s999/robustness_ug_moe2_full_20261118_171448`
- `final_rob_tt_muad_s42/robustness_muad_20261118_172423`
- `final_rob_tt_muad_s123/robustness_muad_20261118_173307`
- `final_rob_tt_muad_s456/robustness_muad_20261118_173916`
- `final_rob_tt_muad_s789/robustness_muad_20261118_174517`
- `final_rob_tt_muad_s999/robustness_muad_20261118_175359`
- `final_rob_tt_gat_s42/robustness_gat_20261118_180242`
- `final_rob_tt_gat_s123/robustness_gat_20261118_181011`
- `final_rob_tt_gat_s456/robustness_gat_20261118_181734`
- `final_rob_tt_gat_s789/robustness_gat_20261118_182457`
- `final_rob_tt_gat_s999/robustness_gat_20261118_183223`

## Final report and result lookup

The corrected report paired ablation deltas on matching seeds and used seed 42 for the sensitivity sweep reference. The latest confirmed output in this history is `experiments/runs/report_20261118_201618/tables.md` and 11 PNG figures in that directory. Older report directories are earlier renders.

```bash
python experiments/make_report.py
R=experiments/runs/report_20261118_201618
ls "$R"
grep -A14 "Modality ablation" "$R/tables.md"
grep -aE "Loaded checkpoint|Saved to:" experiments/runs/final_rob.log
```

The messages for `final_sn_steps*/baselines_*` and `final_tt_steps*/baselines_*` mean those extra directories were absent. The incremental probabilistic and uncertainty-routing rows therefore retained their exploratory one-seed results. The final main and component comparisons used the five seeds listed above.
