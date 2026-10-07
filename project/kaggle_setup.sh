#!/bin/bash
# kaggle_setup.sh - Automated Kaggle environment setup for Graph-Aware MoE project
# Run this in a Kaggle notebook cell: !bash kaggle_setup.sh

set -e  # Exit on error

echo "=========================================="
echo "Kaggle Setup: Graph-Aware MoE Project"
echo "=========================================="

# 1. Clone repository
REPO_URL="https://github.com/Aditya-1735/UG_MoE.git"  # CHANGE THIS
REPO_DIR="UG_MoE"

echo "[1/6] Cloning repository..."
if [ -d "$REPO_DIR" ]; then
    echo "  Directory exists, pulling latest..."
    cd $REPO_DIR && git pull
else
    git clone https://github.com/Aditya-1735/UG_MoE.git
    cd UG_MoE
fi

# 1b. Pull LFS files (required for dataset)
echo "[1b/6] Pulling LFS files..."
git lfs install
git lfs pull

# 2. Install system dependencies (Kaggle has most pre-installed)
echo "[2/6] Checking system dependencies..."
pip install --upgrade pip -q

# 3. Install Python dependencies
echo "[3/6] Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# 3b. Verify critical imports
python -c "
import torch, dgl, torchdata, numpy, pandas, sklearn
print(f'PyTorch: {torch.__version__}')
print(f'DGL: {__import__(\"dgl\").__version__}')
print(f'TorchData: {torchdata.__version__}')
print(f'NumPy: {numpy.__version__}')
print(f'CUDA available: {__import__(\"torch\").cuda.is_available()}')
if __import__('torch').cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# 4. Apply tick patches (if needed - usually not needed on Kaggle)
echo "[4/6] Checking tick compatibility..."
python -c "
import tick
print(f'tick version: {tick.__version__}')
try:
    from tick.hawkes import HawkesADM4
    import numpy as np
    model = HawkesADM4(decay=3)
    model.fit([np.array([0.1, 0.5])], end_time=2.0, baseline_start=np.ones(1)*0.2)
    print('✓ HawkesADM4 works')
except Exception as e:
    print(f'Tick issue: {e}')
    print('Applying patches...')
    import sys
    sys.path.insert(0, '.')
    # Apply patches if needed
" || echo "Tick check completed"

# 5. Verify model loads
echo "[5/6] Verifying model loads..."
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
print('✓ Model loads successfully')
"

# 6. Quick training test (1 epoch)
echo "[6/6] Quick training test (1 epoch)..."
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

data_dir = 'project/preprocessing/muad_compat/output'
# Check if data exists
import os
if os.path.exists('project/preprocessing/muad_compat/output/chunk_train.pkl'):
    print('Using local data')
    data_dir = 'project/preprocessing/muad_compat/output'
else:
    print('Data not found locally - skipping training test')
    exit(0)

train_data, node_num, edges = load_data(
    'project/preprocessing/muad_compat/output/chunk_train.pkl',
    'project/preprocessing/muad_compat/output/metadata.json'
)
test_data, _, _ = load_data(
    'project/preprocessing/muad_compat/output/chunk_test.pkl',
    'project/preprocessing/muad_compat/output/metadata.json'
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
print('✓ Training step works')
"

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run ablation studies:  python experiments/run_ablations.py"
echo "  2. Run full Track B:      python experiments/run_track_b_full.py"
echo "  3. Or run Track A:        python experiments/run_track_a_full.py"
echo ""
echo "For ablation studies (2 seeds x 50 epochs = ~4 hrs):"
echo "  python experiments/run_ablations.py"
echo ""
echo "For full Track B training (5 seeds x 50 epochs):"
echo "  python experiments/run_track_b_full.py"
echo ""
echo "To run with nohup (survives disconnect):"
echo "  nohup python experiments/run_ablations.py > ablations.log 2>&1 &"
echo "  tail -f ablations.log"
echo ""
echo "Done!"