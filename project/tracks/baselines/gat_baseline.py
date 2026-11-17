"""GAT baseline (plan step 7): GATv2 over the service graph with per-node multimodal
features (time-averaged metrics/traces + log vector), mean-pooled, MLP classifier.
Expects BINARY labels."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn


class GATBaseline(nn.Module):
    """
    GAT Baseline: Graph Attention Network on service dependency graph
    with multimodal node features + MLP classifier.
    """
    
    def __init__(self, metric_dim=7, trace_dim=1, log_dim=15, hidden_dim=64,
                 num_heads=4, num_layers=2, dropout=0.1, num_classes=2):
        super().__init__()
        
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
        N = graph.num_nodes() // B  # nodes per graph
        
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
