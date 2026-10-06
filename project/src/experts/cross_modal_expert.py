"""
Cross-modal Expert - Learns relationships between logs, traces, and metrics.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention between two modalities.
    """
    
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, embed_dim)
        self.key_proj = nn.Linear(key_dim, embed_dim)
        self.value_proj = nn.Linear(key_dim, embed_dim)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: [batch, seq_len_q, query_dim]
            key_value: [batch, seq_len_kv, key_dim]
            key_padding_mask: [batch, seq_len_kv] - True for padded positions
        Returns:
            [batch, seq_len_q, embed_dim]
        """
        q = self.query_proj(query)
        k = self.key_proj(key_value)
        v = self.value_proj(key_value)
        
        # Cross-attention
        attn_out, _ = self.cross_attn(
            query=q, key=k, value=v,
            key_padding_mask=key_padding_mask,
        )
        
        # Residual + norm
        x = self.norm(q + self.dropout(attn_out))
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + self.dropout(ffn_out))
        
        return x


class CrossModalExpert(nn.Module):
    """
    Cross-modal Expert for learning relationships between:
    - Logs, Traces, Metrics
    
    Uses bidirectional cross-attention between all modality pairs.
    """
    
    def __init__(
        self,
        metric_dim: int = 64,
        trace_dim: int = 64,
        log_dim: int = 64,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        fusion_type: str = "cross_attention",  # "cross_attention", "concat_mlp"
    ):
        super().__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == "cross_attention":
            # Three modalities, pairwise cross-attention
            # Project inputs to embed_dim first
            self.input_projections = nn.ModuleDict({
                "metric": nn.Linear(metric_dim, embed_dim),
                "trace": nn.Linear(trace_dim, embed_dim),
                "log": nn.Linear(log_dim, embed_dim),
            })
            
            self.layers = nn.ModuleList()
            
            for _ in range(num_layers):
                layer = nn.ModuleDict({
                    "metric_to_trace": CrossAttentionBlock(embed_dim, embed_dim, embed_dim, num_heads, dropout),
                    "trace_to_metric": CrossAttentionBlock(embed_dim, embed_dim, embed_dim, num_heads, dropout),
                    "metric_to_log": CrossAttentionBlock(embed_dim, embed_dim, embed_dim, num_heads, dropout),
                    "log_to_metric": CrossAttentionBlock(embed_dim, embed_dim, embed_dim, num_heads, dropout),
                    "trace_to_log": CrossAttentionBlock(embed_dim, embed_dim, embed_dim, num_heads, dropout),
                    "log_to_trace": CrossAttentionBlock(embed_dim, embed_dim, embed_dim, num_heads, dropout),
                })
                self.layers.append(layer)
            
            # Output projections
            self.metric_out = nn.Linear(embed_dim, metric_dim)
            self.trace_out = nn.Linear(embed_dim, trace_dim)
            self.log_out = nn.Linear(embed_dim, log_dim)
            
        elif fusion_type == "concat_mlp":
            # Simple concatenation + MLP
            self.fusion = nn.Sequential(
                nn.Linear(metric_dim + trace_dim + log_dim, embed_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.metric_out = nn.Linear(embed_dim, metric_dim)
            self.trace_out = nn.Linear(embed_dim, trace_dim)
            self.log_out = nn.Linear(embed_dim, log_dim)
            
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
    
    def forward(
        self,
        metric: torch.Tensor,  # [B, metric_dim] or [B, N, metric_dim]
        trace: torch.Tensor,   # [B, trace_dim] or [B, N, trace_dim]
        log: torch.Tensor,     # [B, log_dim] or [B, N, log_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            metric: Metric embeddings
            trace: Trace embeddings
            log: Log embeddings
        Returns:
            Dict with enhanced embeddings for each modality
        """
        # Ensure 3D: [B, 1, D] if 2D
        if metric.dim() == 2:
            metric = metric.unsqueeze(1)
            trace = trace.unsqueeze(1)
            log = log.unsqueeze(1)
        
        B, N, _ = metric.shape
        
        if self.fusion_type == "cross_attention":
            # Project inputs to embed_dim
            metric_enhanced = self.input_projections["metric"](metric)
            trace_enhanced = self.input_projections["trace"](trace)
            log_enhanced = self.input_projections["log"](log)
            
            for layer in self.layers:
                # Cross-attention between all pairs
                # Note: We treat each modality as a "token" (N=1) or multiple nodes
                
                # Metric attends to Trace
                m2t = layer["metric_to_trace"](metric_enhanced, trace_enhanced)
                # Trace attends to Metric
                t2m = layer["trace_to_metric"](trace_enhanced, metric_enhanced)
                # Metric attends to Log
                m2l = layer["metric_to_log"](metric_enhanced, log_enhanced)
                # Log attends to Metric
                l2m = layer["log_to_metric"](log_enhanced, metric_enhanced)
                # Trace attends to Log
                t2l = layer["trace_to_log"](trace_enhanced, log_enhanced)
                # Log attends to Trace
                l2t = layer["log_to_trace"](log_enhanced, trace_enhanced)
                
                # Combine cross-attention outputs (residual)
                metric_enhanced = metric_enhanced + m2t + m2l
                trace_enhanced = trace_enhanced + t2m + t2l
                log_enhanced = log_enhanced + l2m + l2t
            
            return {
                "metric": self.metric_out(metric_enhanced).squeeze(1) if N == 1 else self.metric_out(metric_enhanced),
                "trace": self.trace_out(trace_enhanced).squeeze(1) if N == 1 else self.trace_out(trace_enhanced),
                "log": self.log_out(log_enhanced).squeeze(1) if N == 1 else self.log_out(log_enhanced),
            }
            
        elif self.fusion_type == "concat_mlp":
            # Concatenate and fuse
            combined = torch.cat([metric, trace, log], dim=-1)  # [B, N, 3*D]
            fused = self.fusion(combined)  # [B, N, embed_dim]
            
            return {
                "metric": self.metric_out(fused).squeeze(1) if N == 1 else self.metric_out(fused),
                "trace": self.trace_out(fused).squeeze(1) if N == 1 else self.trace_out(fused),
                "log": self.log_out(fused).squeeze(1) if N == 1 else self.log_out(fused),
            }


class CrossModalExpertGraph(nn.Module):
    """
    Cross-modal Expert that works with batched DGL graphs.
    Fuses multi-modal node features using cross-attention.
    """
    
    def __init__(
        self,
        metric_dim: int = 64,
        trace_dim: int = 64,
        log_dim: int = 64,
        embed_dim: int = 128,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Per-node cross-modal fusion
        self.node_fusion = CrossModalExpert(
            metric_dim=metric_dim,
            trace_dim=trace_dim,
            log_dim=log_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            fusion_type="cross_attention",
        )
        
        # Calculate fused dimension (3 * metric_dim for cross_attention output)
        fused_dim = metric_dim * 3
        
        # Graph-level pooling
        self.graph_pool = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Output projection (from fused_dim to hidden_dim)
        fused_dim = metric_dim * 3
        self.output_proj = nn.Linear(fused_dim, hidden_dim)
    
    def forward(self, graph) -> Dict[str, torch.Tensor]:
        """
        Args:
            graph: Batched DGL graph with ndata['metrics'], 'traces', 'logs' embeddings
        Returns:
            Dict with graph-level fused embeddings per modality
        """
        B = graph.batch_size
        N = 12  # nodes per graph
        
        # Get node embeddings from graph
        metric_emb = graph.ndata.get("metric_emb", None)  # [B*12, D]
        trace_emb = graph.ndata.get("trace_emb", None)
        log_emb = graph.ndata.get("log_emb", None)
        
        if metric_emb is None:
            raise ValueError("Graph must have 'metric_emb' in ndata")
        
        # Reshape to [B, 12, D]
        metric_emb = metric_emb.view(B, N, -1)
        trace_emb = trace_emb.view(B, N, -1) if trace_emb is not None else torch.zeros_like(metric_emb)
        log_emb = log_emb.view(B, N, -1) if log_emb is not None else torch.zeros_like(metric_emb)
        
        # Cross-modal fusion per node
        fused = self.node_fusion(metric_emb, trace_emb, log_emb)
        
        # Pool over nodes to get graph-level
        # Use attention pooling
        all_fused = torch.stack([fused["metric"], fused["trace"], fused["log"]], dim=2)  # [B, 12, 3, D]
        all_fused = all_fused.view(B, N, -1)  # [B, 12, 3*D]
        
        # Attention pooling
        attn_weights = torch.softmax(self.graph_pool(all_fused), dim=1)  # [B, 12, 1]
        graph_emb = (all_fused * attn_weights).sum(dim=1)  # [B, 3*D]
        
        # Project to output dim
        graph_emb = self.output_proj(graph_emb)
        
        return {
            "fused_embedding": graph_emb,
            "metric": fused["metric"].mean(dim=1),  # [B, D] - average over nodes
            "trace": fused["trace"].mean(dim=1),
            "log": fused["log"].mean(dim=1),
        }