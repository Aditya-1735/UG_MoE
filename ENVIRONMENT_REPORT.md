# Environment Report

## Detected Python version
**3.11.14** (local Windows .venv)

Target environments (Colab/Kaggle) typically provide Python 3.10 or 3.11. Compatible.

## Detected ML framework
**PyTorch 2.2.1** with **DGL (Deep Graph Library) 2.2.1**

- PyTorch is the primary tensor computation framework
- DGL is used for graph neural network operations (GATv2Conv, GlobalAttentionPooling, graph batching)
- No TensorFlow, JAX, or other frameworks detected in source code

## Detected GPU requirements
**GPU: OPTIONAL**

The codebase supports both CPU and GPU execution:
- Device selection via `--device` argument (default: "cpu")
- `torch.device(args.device)` pattern used throughout
- GPU detection: `torch.cuda.is_available()` checks in vendor code
- No hardcoded `.cuda()` calls - uses `.to(device)` pattern
- Mixed precision (fp16/bf16) NOT used
- Distributed training (DDP, DataParallel) NOT used
- Multi-GPU NOT assumed

**Expected GPU VRAM:** Minimal. Model size ~few MB. Batch size default 50. A GPU with 8GB+ VRAM is more than sufficient.

**CUDA Requirements:** 
- PyTorch 2.2.1 supports CUDA 11.8, 12.1
- DGL 2.2.1 provides prebuilt wheels for CUDA 11.8, 12.1, 12.2
- Colab/Kaggle provide their own CUDA runtime (typically 11.8 or 12.x)
- **Do NOT install CUDA manually** - use platform-provided version

## Detected dependencies

### Core (required)
| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.2.1 | Primary ML framework |
| dgl | 2.2.1 | Graph neural networks |
| numpy | 1.26.4 | Numerical computation |
| pandas | 3.0.6 | Data processing |
| scipy | 1.17.1 | Scientific computing |
| scikit-learn | 1.9.1 | Metrics, utilities |
| tqdm | 4.70.1 | Progress bars |
| pyyaml | 6.0.3 | Config parsing |
| networkx | 3.6.1 | Graph utilities (vendor/eadro) |

### Preprocessing (required for data pipeline)
| Package | Version | Purpose |
|---------|---------|---------|
| drain3 | 0.9.11 | Log template mining |
| tick | 0.8.0.2 | Hawkes process for log features |

### Standard Library (no install needed)
`json`, `os`, `sys`, `pathlib`, `datetime`, `random`, `hashlib`, `collections`, `re`, `typing`, `math`, `argparse`, `subprocess`, `pickle`, `logging`

### Optional / Not Required
- `matplotlib`, `pillow` - only if visualization needed
- `wandb`, `mlflow` - not used in codebase
- `torchvision`, `torchaudio` - not used
- `transformers`, `accelerate`, `peft` - not used
- `bitsandbytes`, `xformers`, `flash-attn` - not used
- `hydra`, `omegaconf` - not used (uses argparse + json)

## Local environment
- OS: Windows 10/11
- Python: 3.11.14
- PyTorch: 2.2.1 (CPU or CUDA)
- DGL: 2.2.1 (CPU or CUDA)
- .venv location: `project/.venv/`

## Kaggle compatibility
**COMPATIBLE with minor adjustments**

| Aspect | Status | Notes |
|--------|--------|-------|
| Python version | ✅ | Kaggle supports 3.10/3.11 |
| PyTorch + CUDA | ✅ | Pre-installed, use `--no-deps` for torch/dgl |
| DGL | ⚠️ | Must install with correct CUDA variant |
| Disk space | ✅ | ~2GB for deps + dataset |
| Internet access | ✅ | For pip install |
| Dataset paths | ⚠️ | Must adjust Windows paths to `/kaggle/input/...` |
| Writable output | ✅ | `/kaggle/working/` |

**Known issues:**
1. Hardcoded Windows paths in preprocessing scripts (`C:/Users/...`)
2. `tick` package may need compilation on Linux
3. DGL must match Kaggle's CUDA version

## Colab compatibility
**COMPATIBLE with minor adjustments**

| Aspect | Status | Notes |
|--------|--------|-------|
| Python version | ✅ | Colab supports 3.10/3.11 |
| PyTorch + CUDA | ✅ | Pre-installed |
| DGL | ⚠️ | Must install with correct CUDA variant |
| GPU access | ✅ | T4, V100, A100 available |
| Disk space | ⚠️ | ~70GB but ephemeral |
| Internet access | ✅ | For pip install |
| Dataset paths | ⚠️ | Must adjust to `/content/...` or Google Drive |
| Persistence | ⚠️ | Runtime resets lose files |

**Known issues:**
1. Hardcoded Windows paths in preprocessing scripts
2. `tick` package may need compilation
3. DGL must match Colab's CUDA version
4. Large raw dataset (78MB zip) must be uploaded or downloaded each session

## Known compatibility risks

1. **Hardcoded Windows paths** - Multiple preprocessing scripts use `C:/Users/as999/OneDrive/Desktop/trash/impl/...` which will fail on Linux
2. **tick package** - Requires C++ compilation, may fail on some environments
3. **DGL CUDA variant** - Must install `dgl-cu118` or `dgl-cu121` matching platform CUDA
4. **Raw data location** - 78MB `SN_Dataset.zip` in `raw_data/` - must be accessible
5. **Recovered artifact** - Empty `recovered_artifact/` directory (Track A needs this)
6. **Eadro preprocessing** - Complex pipeline requiring raw data → parsed_data → chunks → MUAD compat

## System packages required
- `build-essential` / `gcc` / `g++` - for compiling `tick` and DGL extensions
- `python3-dev` - headers for C extensions
- On Ubuntu/Debian: `apt-get install -y build-essential python3-dev`

## Environment variables required
None required by code. Optional:
- `PYTHONHASHSEED` - set by `seed_everything()` for reproducibility
- `CUDA_VISIBLE_DEVICES` - for GPU selection (standard)

## External downloads required

### 1. Raw Dataset
- **Source:** `raw_data/SN_Dataset/SN_Dataset.zip` (78.7 MB) or extracted `raw_data/SN Dataset/`
- **Required for:** Full preprocessing pipeline
- **Alternative:** Use pre-generated `project/preprocessing/muad_compat/output/` (chunks.pkl, metadata.json, chunk_train.pkl, chunk_test.pkl)

### 2. Recovered Artifact (Track A only)
- **Expected:** `project/recovered_artifact/` with `chunk_train.pkl`, `chunk_test.pkl`, `metadata.json`
- **Status:** Directory exists but empty
- **Action:** Must be populated from original MUAD commit 6031ccf

### 3. MUAD Baseline Checkpoint (Track B optional)
- **For:** Encoder reuse in `tracks/proposed_model/train.py --muad-checkpoint`
- **Status:** Not provided

## Unknowns that require manual confirmation

1. **Recovered artifact contents** - The `project/recovered_artifact/` directory is empty. Track A experiments require the original MUAD data from commit 6031ccf.

2. **Exact DGL CUDA variant** - Need to verify Colab/Kaggle CUDA version at runtime and install matching `dgl-cuXXX` package.

3. **tick installation on Linux** - May require `libboost-dev` and `cmake`. Test before full run.

4. **Dataset preprocessing time** - Full raw→parsed→chunks→MUAD-compat pipeline takes significant time. Prefer using pre-generated `muad_compat/output/`.

5. **Track A vs Track B data difference** - Track A uses `event_num=15` (recovered artifact), Track B uses `event_num=14` (regenerated data). Ensure correct config.

6. **Eadro vendor code** - The `vendor/eadro/` contains original Eadro implementation. Only used for preprocessing reference. Not needed for Track B training.