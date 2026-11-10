#!/usr/bin/env python3
"""
GAT Baseline - Graph Attention Network baseline for anomaly detection.
Uses DGL GATv2Conv on the service dependency graph with multimodal node features,
followed by a classifier head.
Saves checkpoint to checkpoints/baseline_gat.pth
"""
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "vendor/muad"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vendor.muad.data.utils import load_data, create_dataloader
from vendor.muad.data.dataset import ChunkDataset
from vendor.muad.utils.general_utils import seed_everything
from tracks.proposed_model import DEFAULT_CONFIG

# Disable Flash SDPA for stability
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

try:
    import dgl
    import dgl.nn as dglnn
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False


class GATBaseline(nn.Module):
    """
    GAT Baseline: Graph Attention Network on service dependency graph
    with multimodal node features + MLP classifier.
    """
    
    def __init__(self, metric_dim=7, trace_dim=1, log_dim=14, hidden_dim=64,
                 num_heads=4, num_layers=2, dropout=0.1, num_classes=2):
        super().__init__()
        if not DGL_AVAILABLE:
            raise RuntimeError("DGL is required for GAT baseline")
        
        # Modality encoders (per-node)
        self.metric_encoder = nn.Linear(metric_dim, hidden_dim)
        self.trace_encoder = nn.Linear(trace_dim, hidden_dim)
        self.log_encoder = nn.Linear(log_dim, hidden_dim)
        
        # Combined feature dimension
        combined_dim = hidden_dim * 3
        
        # GAT layers on dependency graph
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_feats = combined_dim if i == 0 else hidden_dim * num_heads
            out_feats = hidden_dim
            self.gat_layers.append(
                dglnn.GATv2Conv(
                    in_feats=in_feats,
                    out_feats=out_feats,
                    num_heads=num_heads,
                    feat_drop=dropout,
                    attn_drop=dropout,
                    negative_slope=0.2,
                    residual=True,
                    allow_zero_in_degree=True,
                )
            )
            if i < num_layers - 1:
                self.norms.append(nn.LayerNorm(hidden_dim * num_heads))
        
        self.output_proj = nn.Linear(hidden_dim * num_heads, hidden_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, graph, labels=None):
        """
        Args:
            graph: Batched DGL graph with ndata['metrics'], 'traces', 'logs'
            labels: [B] binary labels (optional)
        Returns:
            dict with logits, loss (if labels provided), y_pred
        """
        B = graph.batch_size
        N = 12  # nodes per graph
        
        # Get per-node modality features
        metrics = graph.ndata["metrics"]  # [B*12, 10, 7]
        traces = graph.ndata["traces"]    # [B*12, 10, 1]
        logs = graph.ndata["logs"]        # [B*12, 15]
        
        # Encode each modality per node (mean over time for metrics/traces)
        metric_feat = self.metric_encoder(metrics.mean(dim=1))  # [B*12, hidden]
        trace_feat = self.trace_encoder(traces.mean(dim=1))     # [B*12, hidden]
        log_feat = self.log_encoder(logs)                        # [B*12, hidden]
        
        # Combine modalities per node
        combined = torch.cat([metric_feat, trace_feat, log_feat], dim=-1)  # [B*12, 3*hidden]
        
        # GAT on dependency graph
        h = combined
        for i, gat_layer in enumerate(self.gat_layers):
            h = gat_layer(graph, h)  # [B*12, num_heads, out_feats]
            if i < len(self.gat_layers) - 1:
                h = h.flatten(1)
                h = self.norms[i](h)
                h = F.elu(h)
            else:
                h = h.flatten(1)
        
        h = self.output_proj(h)  # [B*12, hidden_dim]
        
        # Graph-level pooling (mean over nodes)
        h = h.view(B, N, -1).mean(dim=1)  # [B, hidden_dim]
        
        # Classify
        logits = self.classifier(h)  # [B, num_classes]
        y_pred = logits.argmax(dim=-1)
        
        result = {"logits": logits, "y_pred": y_pred}
        
        if labels is not None:
            loss = self.criterion(logits, labels)
            result["loss"] = loss
        
        return result


def run_single_experiment(run_id: int, seed: int, output_base: Path):
    """Run single GAT baseline experiment with given seed."""
    print(f"\n{'='*60}")
    print(f"GAT BASELINE - RUN {run_id}/5 - Seed: {seed}")
    print(f"{'='*60}")
    
    seed_everything(seed)
    
    # Load recovered artifact (same as Track A/B for apples-to-apples)
    data_dir = Path(__file__).parent.parent / "recovered_artifact"
    train_data, node_num, edges = load_data(
        str(data_dir / "chunk_train.pkl"),
        str(data_dir / "metadata.json")
    )
    test_data, _, _ = load_data(
        str(data_dir / "chunk_test.pkl"),
        str(data_dir / "metadata.json")
    )
    
    train_dataset = ChunkDataset(train_data, list(train_data.keys()), node_num, edges)
    test_dataset = ChunkDataset(test_data, list(test_data.keys()), node_num, edges)
    
    train_loader = create_dataloader(train_dataset, batch_size=50, shuffle=True)
    test_loader = create_dataloader(test_dataset, batch_size=50, shuffle=False)
    
    device = torch.device("cpu")
    model = GATBaseline(
        metric_dim=7, trace_dim=1, log_dim=14,
        hidden_dim=64, num_heads=4, num_layers=2,
        dropout=0.1, num_classes=2,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    best_f1 = 0
    best_epoch = 0
    
    for epoch in range(1, 51):
        model.train()
        train_loss = 0
        for graph, labels in train_loader:
            graph, labels = graph.to(device), labels.to(device)
            binary_labels = (labels >= 1).long()
            
            optimizer.zero_grad()
            outputs = model(graph, binary_labels)
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            train_loss += loss.item()
        
        if epoch % 5 == 0:
            model.eval()
            all_preds = []
            all_labels = []
            all_probs = []
            test_loss = 0
            with torch.no_grad():
                for graph, labels in test_loader:
                    graph, labels = graph.to(device), labels.to(device)
                    binary_labels = (labels >= 1).long()
                    outputs = model(graph, binary_labels)
                    loss = outputs["loss"]
                    test_loss += loss.item()
                    logits = outputs["logits"]
                    probs = torch.softmax(logits, dim=-1)[:, 1]
                    preds = outputs["y_pred"]
                    
                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(binary_labels.cpu().tolist())
                    all_probs.extend(probs.cpu().tolist())
            
            from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
            f1 = f1_score(all_labels, all_preds, average='binary')
            precision = precision_score(all_labels, all_preds, average='binary')
            recall = recall_score(all_labels, all_preds, average='binary')
            roc_auc = roc_auc_score(all_labels, all_probs)
            pr_auc = average_precision_score(all_labels, all_probs)
            
            print(f"  Epoch {epoch:2d}: Train Loss={train_loss/len(train_loader):.4f}, "
                  f"Test Loss={test_loss/len(test_loader):.4f}, "
                  f"F1={f1:.4f}, Pre={precision:.4f}, Rec={recall:.4f}, "
                  f"ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
    
    # Final evaluation
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for graph, labels in test_loader:
            graph, labels = graph.to(device), labels.to(device)
            binary_labels = (labels >= 1).long()
            outputs = model(graph, binary_labels)
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = outputs["y_pred"]
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(binary_labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
    
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
    final_f1 = f1_score(all_labels, all_preds, average='binary')
    final_pre = precision_score(all_labels, all_preds, average='binary')
    final_rec = recall_score(all_labels, all_preds, average='binary')
    final_roc_auc = roc_auc_score(all_labels, all_probs)
    final_pr_auc = average_precision_score(all_labels, all_probs)
    
    # Save checkpoint
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "baseline_gat.pth")
    
    results = {
        "run_id": run_id,
        "seed": seed,
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "final_f1": final_f1,
        "final_precision": final_pre,
        "final_recall": final_rec,
        "final_roc_auc": final_roc_auc,
        "final_pr_auc": final_pr_auc,
        "checkpoint": str(checkpoint_dir / "baseline_gat.pth"),
    }
    
    with open(output_base / f"run_{run_id}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"  Best F1: {best_f1:.4f} (epoch {best_epoch})")
    print(f"  Final F1: {final_f1:.4f}, Pre: {final_pre:.4f}, Rec: {final_rec:.4f}, "
          f"ROC-AUC: {final_roc_auc:.4f}, PR-AUC: {final_pr_auc:.4f}")
    print(f"  Checkpoint saved to: {checkpoint_dir / 'baseline_gat.pth'}")
    
    return results


def main():
    print("="*60)
    print("GAT BASELINE - FULL EXPERIMENTS")
    print("="*60)
    print("Config: hidden_dim=64, epochs=50, lr=0.001, batch=50, weight_decay=1e-5")
    print("Data: recovered_artifact (event_num=15)")
    print("5 runs with seeds: 42, 123, 456, 789, 999")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(__file__).parent.parent / "experiments" / "runs" / f"gat_baseline_{timestamp}"
    output_base.mkdir(parents=True, exist_ok=True)
    
    seeds = [42, 123]
    all_results = []
    
    for i, seed in enumerate(seeds, 1):
        try:
            results = run_single_experiment(i, seed, output_base)
            all_results.append(results)
        except Exception as e:
            print(f"  ERROR in run {i}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"run_id": i, "seed": seed, "error": str(e)})
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY - GAT BASELINE")
    print(f"{'='*60}")
    
    successful = [r for r in all_results if "error" not in r]
    if successful:
        f1s = [r["best_f1"] for r in successful]
        print(f"Successful runs: {len(successful)}/5")
        print(f"Best F1s: {[f'{f:.4f}' for f in f1s]}")
        print(f"Mean F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    
    summary = {
        "experiment": "gat_baseline",
        "timestamp": timestamp,
        "config": {
            "hidden_dim": 64,
            "epochs": 50,
            "lr": 0.001,
            "batch_size": 50,
            "weight_decay": 1e-5,
            "event_num": 15,
            "seeds": seeds,
        },
        "results": all_results,
    }
    
    with open(output_base / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_base}")


if __name__ == "__main__":
    main()
