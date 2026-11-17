"""Graph-MoE baseline (plan section 8A): GATv2 node representations, per-node top-k routing
over expert classification heads, mean-pooled logits. Expects BINARY labels."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn


class GraphMoE(nn.Module):
    """
    Graph-MoE: Graph-centric Mixture-of-Experts.
    
    Uses GAT on the dependency graph as the primary expert router,
    with modality-specific experts contributing per-node features.
    """
    
    def __init__(self, metric_dim=7, trace_dim=1, log_dim=15, hidden_dim=64,
                 num_experts=4, num_heads=4, num_layers=2, dropout=0.1,
                 num_classes=2, top_k=2):
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_dim = hidden_dim
        
        # Modality encoders (per-node)
        self.metric_encoder = nn.Linear(metric_dim, hidden_dim)
        self.trace_encoder = nn.Linear(trace_dim, hidden_dim)
        self.log_encoder = nn.Linear(log_dim, hidden_dim)
        
        # GAT for graph-aware node representations
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        combined_dim = hidden_dim * 3
        
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
        
        self.graph_proj = nn.Linear(hidden_dim * num_heads, hidden_dim)
        
        # Expert-specific linear heads
        self.expert_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            ) for _ in range(num_experts)
        ])
        
        # Graph-aware router: per-node routing based on graph structure
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )
        
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, graph, labels=None):
        """
        Args:
            graph: Batched DGL graph with ndata['metrics'], 'traces', 'logs'
            labels: [B] binary labels (optional)
        Returns:
            dict with logits, loss (if labels provided), y_pred, routing_weights
        """
        B = graph.batch_size
        N = graph.num_nodes() // B  # nodes per graph
        
        # Get per-node modality features
        metrics = graph.ndata["metrics"]  # [B*12, 10, 7]
        traces = graph.ndata["traces"]    # [B*12, 10, 1]
        logs = graph.ndata["logs"]        # [B*12, 15]
        
        # Encode each modality per node
        metric_feat = self.metric_encoder(metrics.mean(dim=1))  # [B*12, hidden]
        trace_feat = self.trace_encoder(traces.mean(dim=1))     # [B*12, hidden]
        log_feat = self.log_encoder(logs)                        # [B*12, hidden]
        
        # Combine modalities per node
        combined = torch.cat([metric_feat, trace_feat, log_feat], dim=-1)  # [B*12, 3*hidden]
        
        # GAT on dependency graph
        h = combined
        for i, gat_layer in enumerate(self.gat_layers):
            h = gat_layer(graph, h)
            if i < len(self.gat_layers) - 1:
                h = h.flatten(1)
                h = self.norms[i](h)
                h = F.elu(h)
            else:
                h = h.flatten(1)
        
        h = self.graph_proj(h)  # [B*12, hidden_dim]
        
        # Per-expert predictions
        expert_logits = []
        for expert_head in self.expert_heads:
            expert_logits.append(expert_head(h))  # [B*12, num_classes]
        expert_logits = torch.stack(expert_logits, dim=1)  # [B*12, num_experts, num_classes]
        
        # Graph-aware routing weights per node
        routing_logits = self.router(h)  # [B*12, num_experts]
        routing_weights = torch.softmax(routing_logits, dim=-1)  # [B*12, num_experts]
        
        # Top-k routing
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights = torch.softmax(top_k_weights, dim=-1)
        sparse_weights = torch.zeros_like(routing_weights)
        sparse_weights.scatter_(-1, top_k_indices, top_k_weights)
        
        # Weighted combination of expert predictions
        # expert_logits: [B*12, num_experts, num_classes]
        # sparse_weights: [B*12, num_experts]
        weighted_logits = (expert_logits * sparse_weights.unsqueeze(-1)).sum(dim=1)  # [B*12, num_classes]
        
        # Graph-level pooling
        weighted_logits = weighted_logits.view(B, N, -1).mean(dim=1)  # [B, num_classes]
        routing_weights = routing_weights.view(B, N, -1).mean(dim=1)  # [B, num_experts]
        
        y_pred = weighted_logits.argmax(dim=-1)
        
        result = {"logits": weighted_logits, "y_pred": y_pred, "routing_weights": routing_weights}
        
        if labels is not None:
            loss = self.criterion(weighted_logits, labels)
            result["loss"] = loss
        
        return result
