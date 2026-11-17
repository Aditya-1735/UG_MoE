# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workflow and ground rules (from the user)

1. **Branch**: the initialised git repo is currently on the `hdfs` branch.
2. **Local vs GPU server**: work happens on the user's local (Windows) machine, where we only verify that files compile/import (e.g. `python -m py_compile <file>`) — do not try to run training or experiments locally. Changes are pushed to the GitHub repo; the user then manually runs `git pull` on the GPU server (repo root `~/AHA_BTP/UG_MoE`) and runs the commands we give them there. So when suggesting commands, write them for the server (Linux, from `project/`), and say what output/files to bring back.
3. **Don't trust the docs, trust the code**: the project was previously developed with opencode, whose documentation (paper draft, plans, forensics) was found unreliable and deleted on 2026-11-17. The baseline paper is `7. Uncertainty-Aware_Multimodal_Anomaly_Detection_for_Microservice_Systems_With_Active_Learning.pdf` (MUAD), and the supervisor's plan is `7. UG-MoE_Implementation_Guide.md`. It is unknown how much of the implementation actually follows them. Do not assume anything claimed in the `.md` files (status tables, reported results, "complete" markers) is true unless the Python code confirms it; the Python files are the primary source of truth. When code and plan disagree, flag it rather than silently picking one.
4. **Existing results**: everything in `project/experiments/runs/` (logs and `summary.json` files) was produced on the `project/recovered_artifact` dataset (`event_num=15`).
5. **Status tracking**: `PROJECT_STATUS.md` (repo root) is the code-verified record of what is done, broken, and next relative to the implementation guide. Read it before starting work and update it when a phase/step changes.
6. **Single seed for now** (time constraint, set 2026-11-17): every experiment runs with **seed 42 only**. Give the user single-seed commands (`--seeds 42`), make new scripts default to one seed, and don't propose multi-seed runs. Keep scripts *able* to take several seeds so final paper runs can use 5 seeds later. Single-seed results have no std or significance test, so compare them with that caveat and don't call sub-1-point F1 differences an improvement.
7. **Deviating from the plan is allowed, with justification first**: `7. UG-MoE_Implementation_Guide.md` is the default. If you are confident that a change would make the codebase, model, project or research better, we may deviate slightly from it, but **before implementing** present (a) what the plan says, (b) the proposed deviation, (c) the reasoning and evidence (code, results, literature), (d) risks and what it costs in comparability or time, and (e) a concrete implementation and validation plan. Wait for the user's go-ahead. Record each approved deviation, with its rationale, in `PROJECT_STATUS.md` so the paper can justify it.
8. **Keep it clean**: this is meant to become a research-paper codebase that works properly and follows the plan. Keep the directory tidy (no stray artifacts, scratch files or duplicate output dirs) and keep committing/pushing only what we are confident about.

## What this repo is

Research code for **UG-MoE**: a Graph-Aware Mixture-of-Experts (MoE) for multimodal anomaly detection in microservice systems (metrics + traces + logs). It extends the vendored **MUAD** baseline (`project/vendor/muad`) and is compared against it. The two models are called **Track A** (MUAD, faithful reproduction) and **Track B** (`GraphAwareMoE`, the proposed model). Eadro (`project/vendor/eadro`) is vendored for the SN (SocialNetwork) dataset format.

The repo root holds only the reference documents (`7. UG-MoE_Implementation_Guide.md`, the MUAD paper as PDF + markdown, the proposal PDF), `README.md`, `PROJECT_STATUS.md` and this file. The opencode-era planning/forensics markdown files and paper draft were deleted on 2026-11-17 (they are in git history before that date); don't recreate parallel planning docs, update `PROJECT_STATUS.md` instead. **All code lives under `project/`.**

## Running things

Everything is run **from `project/`** (scripts `sys.path.insert` the project root, `vendor/muad` and `src`, and use relative data paths):

```bash
cd project
# Environment: conda env "muad" (see ../activate_conda.sh: sets DGLBACKEND=pytorch and CUDA_VISIBLE_DEVICES=2).
# Needs torch, dgl, numpy, scipy, scikit-learn, matplotlib. There is also a project/.venv on the author's machine.

python experiments/run_comparison.py --seeds 42 --device cuda            # Track A vs Track B (current standard: single seed)
python experiments/run_comparison.py --seeds 42,123,456,789,999          # multi-seed + paired t-test (only for final paper runs)
python experiments/run_baselines.py --device cuda                        # Exp. A: muad, gat, plain_moe, graph_moe, ug_moe (--models to subset)
python experiments/run_ablations.py --device cuda                        # ug_moe + all ablations (--ablations no_graph,no_fusion to subset)
python experiments/run_robustness.py --device cuda                       # Exp. D+E on ug_moe (--model, --checkpoint <best.pth> to skip training)
```

All migrated scripts share the same CLI (`registry.add_common_args`: `--seeds` (default 42), `--epochs`, `--lr`, `--patience`, `--val-ratio`, `--split-seed`, `--data-dir`, `--device`) and write to `project/experiments/runs/<name>_<YYYYMMDD_HHMMSS>/`: `summary.json`, `split.json`, one `<model>_seed<N>.json` per run (with val history) and `<model>_seed<N>_best.pth`. Not yet migrated (they still select on test): `run_routing_analysis.py`, `run_uncertainty_fusion_analysis.py`, `run_track_a_al.py`. These are run on the GPU server (see workflow above), not locally.

- **Tests**: there is no pytest suite wired up. `tracks/proposed_model/test_components.py` is an ad-hoc component shape check (`python tracks/proposed_model/test_components.py`, CPU, random tensors).
- **Data validation**: `python validation/run_all_gates.py` runs 10 sequential gate scripts (raw integrity → parsed schema → chunk schema → tensor dims → labels → graph → split → artifact comparison → dataloader). Gate 1 hardcodes an absolute `raw_data/SN Dataset` path from the author's machine.
- **Lint/format config** (`pyproject.toml`): black/flake8 with line length 100. Note `pyproject.toml`'s `[project.dependencies]` is written as a TOML table rather than an array, so `pip install -e .` may fail; don't rely on it.

## Architecture

### Model (`project/tracks/proposed_model/model.py`, `GraphAwareMoE`)

Forward pass, in order: `encode_modalities` → per-expert outputs → (optional) per-expert uncertainty block → router → fusion → classifier.

- **Components live in `project/src/`**: `experts/` (Temporal, Semantic/log, Dependency, CrossModal), `routing/router.py` (`create_router`: `base` / `graph_aware` / `uncertainty_guided`), `uncertainty/` (`UncertaintyBlock`, `VariationalUncertaintyBlock` — returns `(sample, kl_loss)`), `fusion/` (`AdaptiveFusion` with attention/weighted_concat/gated/confidence, and `DynamicFusion`). `model.py` imports them as `src.*`, so `project/` must be on `sys.path`.
- Inputs are batched DGL graphs with `ndata["metrics"] [B*12,10,7]`, `["traces"] [B*12,10,1]`, `["logs"] [B*12,event_num]`. Experts: temporal, semantic, dependency, cross_modal. Router input is `concat(metric, trace, log)` embeddings. Multimodal only: the HDFS log-only mode was removed on 2026-11-17 because the plan is multimodal-only (it survives in git history and on the `backup-hdfs` branch).
- **Routing + fusion both run.** Routing weights scale the expert outputs into `combined`, but the logits come from `self.fusion(expert_outputs)` — `combined` is not what the classifier sees. Routing influences the loss only via its load-balancing aux loss. Keep this in mind when changing routing behaviour.
- **Loss** = CE + `0.1 * mean(KL from uncertainty blocks)` + `0.01 * router aux loss`. Labels are culprit IDs binarized as `labels >= 1`.
- Router receives `graph_info` = flattened edge list per sample, hardcoded for **24 edges (48 values)**; the 12-node / 24-edge MUAD topology is assumed in several places (`num_nodes=12`, `.view(B, 12, -1)`). Other datasets (SN, via `preprocessing/sn` and `experiments/convert_sn_dataset.py`) needed per-node feature expansion/padding to fit — recent commits (`log_node` padded to `log_dim`, router output shape handling) are fixes in this area.
- The forward pass raises `RuntimeError("NaN/Inf ...")` at each stage on non-finite values. Experiment scripts also disable flash/mem-efficient SDPA (`torch.backends.cuda.enable_*_sdp(False)`) because of NaN gradients in Transformer attention — keep that when writing new training scripts.
- Configs: `DEFAULT_CONFIG` in `model.py` (note `log_dim=15` matches the recovered dataset's `event_num=15`; the constructor default is 14). `create_graph_aware_moe(config, muad_baseline=None)` can reuse a MUAD baseline's encoders.
- Ablation configs (`registry.ABLATIONS`): `no_graph`, `no_uncertainty`, `no_fusion`, `single_expert`, `plain_moe`, `graph_transformer`.
- Size: `ug_moe` has ~4.4M parameters vs ~0.24M (MUAD) and ~0.3M (GAT). The user doesn't mind large experts, but shrinking is welcome when the loss is negligible (validation F1 / macro-F1 within ~0.5 points). This is planned as a size sweep after the Phase 2 fixes (`PROJECT_STATUS.md` §3.10, step 13b), not before.

### Shared experiment protocol (`project/experiments/common.py`)

New and migrated experiment scripts must use this module rather than writing their own loop: `load_splits` (val = stratified 15% of `chunk_train.pkl`, fixed `split_seed`, shared across models and seeds; per-channel z-score of all modalities fitted on the train split only, `--normalize train_zscore` by default, because raw log features are ~6e-10 and invisible to the networks. Checkpoints trained before 2026-11-17 evening used raw inputs, so evaluate them with `--normalize none`), `train_and_select` (select on **val** F1, tie-break PR-AUC, early stopping, test evaluated once, per-seed checkpoint), `compute_metrics` (anomaly F1 / P / R plus macro-F1, normal-class F1, PR-AUC, ROC-AUC) and `trivial_baseline_metrics`. Scripts import it as `from common import ...` (the script directory is on `sys.path`).

Models are defined once in `project/experiments/registry.py`: `MODELS` (`muad`, `ug_moe`, `plain_moe`, `gat`, `graph_moe`) plus `abl:<name>` for `ABLATIONS`, each with its forward adapter (MUAD/UG-MoE take raw culprit labels and return `MMlogit`; the baselines in `project/tracks/baselines/` take binary labels and return `logits`). `run_suite` trains a list of models × seeds. To add a model or ablation, register it there; don't build models inside scripts.

### Track A / vendor code

`run_comparison.py` is the reference for how experiments are wired: Track A is `vendor.muad.models.main_model.MainModel` trained on its own returned `loss` (MUAD objective), Track B is `GraphAwareMoE`, and both go through `common.train_and_select` with identical data, split and seeds. Data loading underneath is `vendor.muad.data.utils.load_data` (returns `train_data, node_num, edges, metadata`) → `ChunkDataset` → DGL dataloaders. Treat `project/vendor/` as third-party; avoid editing it unless a fix is required for both tracks.

### Datasets (see `project/data_README.md`; **do not mix them**)

- `project/recovered_artifact/` — original MUAD dataset (`chunk_{train,test}.pkl` + `metadata.json`, `event_num=15`, 12 nodes, 24 edges). **Use this for Track A vs Track B**; `vendor/muad/Dataset/chunk/` is a byte-identical copy.
- `project/preprocessing/muad_compat/output/` — regenerated data with `event_num=14`; using it for one track and not the other invalidates the comparison.
- SN (Eadro SocialNetwork): `preprocessing/raw_to_parsed/` (raw → `parsed_data/SN`); `preprocessing/sn/` holds `parse_sn.py` / `create_sequences.py` (unverified). `experiments/convert_sn_dataset.py` is **broken** (its output had wrong shapes and a single label, so it was deleted; see `PROJECT_STATUS.md` §3.11); chunks for Eadro in `vendor/eadro/codes/chunks/SN/`. Raw data (`raw_data/`, `SN_dataset/`, `hdfs_dataset/`) is gitignored.

## Repo quirks

- Large artifacts are committed to git: `.pkl` chunk files (some modified locally, so `git status` is noisy) and `experiments/runs/*` outputs. Stage files explicitly by path; never `git add -A`. `.pyc` files were untracked on 2026-11-17 (`__pycache__/` is gitignored).
- The only output location is `project/experiments/runs/`. A stray, empty `C:Usersas999OneDriveDesktoptrashimpl/` folder at the repo root is a path-mangling artifact (untracked; delete it by hand, because the tool's safety check blocks removing it).
- Main branch is `main`; other branches: `run_track_b_full`, `prog`, `backup-hdfs`.
- `.kilo/` (gitignored) contains planning notes from another AI tool, not part of the codebase.
