"""
Temporal Expert - Models temporal behavior from metric/trace sequences.
Uses GRU/LSTM/Transformer to capture temporal dependencies.
"""
import torch
import torch.nn as nn
from typing import Optional


class TemporalExpert(nn.Module):
    """
    Temporal Expert for sequence modeling.
    Supports GRU, LSTM, or Transformer backbones.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        backbone: str = "gru",
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.backbone_type = backbone.lower()
        
        if self.backbone_type == "gru":
            self.backbone = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
            )
            self.output_dim = hidden_dim * (2 if bidirectional else 1)
            
        elif self.backbone_type == "lstm":
            self.backbone = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
            )
            self.output_dim = hidden_dim * (2 if bidirectional else 1)
            
        elif self.backbone_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_dim = input_dim
            
        else:
            raise ValueError(f"Unknown backbone: {backbone}. Choose from: gru, lstm, transformer")
        
        # Projection to standard hidden_dim if needed
        if self.output_dim != hidden_dim:
            self.projection = nn.Linear(self.output_dim, hidden_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim] or [num_nodes, seq_len, input_dim]
        Returns:
            [batch, hidden_dim] - last hidden state (GRU/LSTM) or pooled (Transformer)
        """
        if self.backbone_type in ("gru", "lstm"):
            # RNN: output shape [batch, seq_len, hidden_dim * num_directions]
            # hidden shape [num_layers * num_directions, batch, hidden_dim]
            output, hidden = self.backbone(x)
            
            if self.backbone_type == "lstm":
                hidden = hidden[0]  # LSTM returns (h_n, c_n)
            
            # Take last layer's hidden state
            if self.backbone.bidirectional:
                # Concatenate forward and backward
                last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
            else:
                last_hidden = hidden[-1]
                
        elif self.backbone_type == "transformer":
            # Transformer: output shape [batch, seq_len, input_dim]
            output = self.backbone(x)
            # Pool over sequence dimension (mean pooling)
            last_hidden = output.mean(dim=1)
        
        # Project to standard hidden_dim
        return self.projection(last_hidden)


class TemporalExpertGraph(nn.Module):
    """
    Temporal Expert that works with batched DGL graphs.
    Extracts temporal features from node data.
    """
    
    def __init__(
        self,
        metric_dim: int = 7,
        trace_dim: int = 1,
        hidden_dim: int = 64,
        backbone: str = "gru",
    ):
        super().__init__()
        self.metric_encoder = TemporalExpert(metric_dim, hidden_dim, backbone=backbone)
        self.trace_encoder = TemporalExpert(trace_dim, hidden_dim, backbone=backbone)
        
    def forward(self, graph) -> dict:
        """
        Args:
            graph: Batched DGL graph with ndata['metrics'] [N, 10, 7] and ndata['traces'] [N, 10, 1]
        Returns:
            dict with 'metric_embedding', 'trace_embedding' each [B, 64]
        """
        # graph.batch_size gives B, graph.num_nodes() = B * 12
        B = graph.batch_size
        N_per_graph = 12
        
        # Metrics: [B*12, 10, 7] -> reshape to [B, 12, 10, 7] -> [B*12, 10, 7]
        metrics = graph.ndata["metrics"]  # [B*12, 10, 7]
        traces = graph.ndata["traces"]    # [B*12, 10, 1]
        
        # Encode each node's temporal sequence
        metric_emb = self.metric_encoder(metrics)  # [B*12, 64]
        trace_emb = self.trace_encoder(traces)      # [B*12, 64]
        
        # Pool over nodes per graph to get graph-level embeddings
        metric_emb = metric_emb.view(B, N_per_graph, -1).mean(dim=1)  # [B, 64]
        trace_emb = trace_emb.view(B, N_per_graph, -1).mean(dim=1)    # [B, 64]
        
        return {
            "metric_embedding": metric_emb,
            "trace_embedding": trace_emb,
        }