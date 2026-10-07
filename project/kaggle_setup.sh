#!/bin/bash
# kaggle_setup.sh - Kaggle environment setup for Graph-Aware MoE project
# Run this in a Kaggle notebook cell: !bash kaggle_setup.sh
# NOTE: Run `git clone` and `git lfs pull` in a separate notebook cell FIRST

set -e  # Exit on error

echo "=========================================="
echo "Kaggle Setup: Graph-Aware MoE Project"
echo "=========================================="

# 1. Install system dependencies
echo "[1/5] Checking system dependencies..."
pip install --upgrade pip -q

# 2. Install Python dependencies with GPU support
echo "[2/5] Installing Python dependencies..."
pip install --upgrade pip -q

# Detect CUDA version for DGL wheel
CUDA_VERSION=$(python -c "import torch; print('cu' + ''.join(torch.version.cuda.split('.')[:2]))" 2>/dev/null || echo "cpu")
echo "Detected CUDA: $CUDA_VERSION"

if [ "$CUDA_VERSION" != "cpu" ]; then
    echo "Installing DGL for CUDA $CUDA_VERSION..."
    pip install dgl -f https://data.dgl.ai/wheels/torch-2.5/${CUDA_VERSION}/repo.html -q
else
    echo "CPU-only environment, installing CPU DGL"
    pip install dgl==2.2.1 -q
fi

pip install torch==2.5.1 torchdata==0.7.1 "numpy<2" pydantic tick==0.8.0.2 drain3 scikit-learn pandas scipy matplotlib tqdm -q

# 3. Verify critical imports and GPU
echo "[2/5] Verifying imports and GPU..."
python -c "
import torch, dgl, torchdata, numpy, pandas, sklearn
print(f'PyTorch: {torch.__version__}')
print(f'DGL: {__import__(\"dgl\").__version__}')
print(f'TorchData: {torchdata.__version__}')
print(f'NumPy: {numpy.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
"

# 4. Check tick compatibility (optional - may not be needed on Kaggle)
echo "[3/5] Checking tick compatibility..."
python -c "
import tick
print(f'tick version: {tick.__version__}')
try:
    from tick.hawkes import HawkesADM4
    import numpy as np
    model = HawkesADM4(decay=3)
    model.fit([np.array([0.1, 0.5])], end_time=2.0, baseline_start=np.ones(1)*0.2)
    print('[OK] HawkesADM4 works')
except Exception as e:
    print(f'Tick issue (may not be needed on Kaggle): {e}')
    print('If preprocessing is already done, tick is not needed for training')
"

# 4. Verify model loads
echo '[4/5] Verifying model loads...'
python -c "
import sys
sys.path.insert(0, '.')
import torch
from tracks.proposed_model import GraphAwareMoE, DEFAULT_CONFIG
config = {
    'num_nodes': 12,
    'metric_dim': 7,
    'trace_dim': 1,
    'log_dim': 14,
    'num_classes': 2,
    'hidden_dim': 64,
    'expert_hidden_dim': 128,
    'fusion_dim': 128,
    'router_hidden_dim': 128,
    'num_experts': 4,
    'expert_types': ['temporal', 'semantic', 'dependency', 'cross_modal'],
    'router_type': 'uncertainty_guided',
    'top_k': 2,
    'use_graph_routing': True,
    'use_uncertainty_routing': True,
    'fusion_type': 'confidence',
    'use_uncertainty': True,
    'uncertainty_type': 'variational',
    'dropout': 0.1,
    'log_dim': 14
}
model = GraphAwareMoE(**config)
print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')
print('[OK] Model loads successfully')
"

# 5. Quick training test (1 epoch)
echo "[5/5] Quick training test (1 epoch)..."
python -c "
import sys
sys.path.insert(0, '.')
import torch
from vendor.muad.data.utils import load_data, create_dataloader
from vendor.muad.data.dataset import ChunkDataset
from tracks.proposed_model import GraphAwareMoE, DEFAULT_CONFIG
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Check if data exists
import os
if os.path.exists('preprocessing/muad_compat/output/chunk_train.pkl'):
    print('Using local data')
    data_dir = 'preprocessing/muad_compat/output'
else:
    print('Data not found locally - check git lfs pull or Kaggle Dataset mount')
    exit(0)

from vendor.muad.data.utils import load_data, create_dataloader
from vendor.muad.data.dataset import ChunkDataset
from tracks.proposed_model import GraphAwareMoE, DEFAULT_CONFIG
import torch.nn as nn

train_data, node_num, edges = load_data(
    'preprocessing/muad_compat/output/chunk_train.pkl',
    'preprocessing/muad_compat/output/metadata.json'
)
test_data, _, _ = load_data(
    'preprocessing/muad_compat/output/chunk_test.pkl',
    'preprocessing/muad_compat/output/metadata.json'
)

train_dataset = ChunkDataset(train_data, list(train_data.keys())[:100], node_num, edges)
test_dataset = ChunkDataset(test_data, list(test_data.keys())[:50], node_num, edges)

train_loader = create_dataloader(train_dataset, batch_size=8, shuffle=True)
test_loader = create_dataloader(test_dataset, batch_size=8, shuffle=False)

config = DEFAULT_CONFIG.copy()
config['log_dim'] = 14
model = GraphAwareMoE(**{k:v for k,v in config.items() if k!='epochs'}).to('cuda' if torch.cuda.is_available() else 'cpu')

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

model.train()
for graph, labels in train_loader:
    graph, labels = graph.to(device), labels.to(device)
    binary_labels = (labels >= 1).long()
    outputs = model(graph, binary_labels)
    loss = outputs['loss']
    loss.backward()
    print(f'Loss: {loss.item():.4f}')
    break
print('Training step works')
"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run ablation studies:  python kaggle_train.py --experiment ablation --seeds 42,123 --epochs 30"
echo "  2. Run full Track B:      python kaggle_train.py --experiment track_b_full --seeds 42,123,456 --epochs 50"
echo "  3. Or run Track A:        python kaggle_train.py --experiment track_a_full --seeds 42 --epochs 50"
echo ""
echo "For ablation studies (2 seeds x 30 epochs = ~2 hrs):"
echo "  python kaggle_train.py --experiment ablation --seeds 42,123 --epochs 30"
echo ""
echo "For full Track B training (5 seeds x 50 epochs):"
echo "  python kaggle_train.py --experiment track_b_full --seeds 42,123,456,789,999 --epochs 50"
echo ""
echo "To run with Kaggle Save Version (survives disconnect):"
echo "  1. Click 'Save Version' -> 'Save & Run All'"
echo "  2. Output goes to /kaggle/working and is saved with version"
echo ""
echo "Done!"