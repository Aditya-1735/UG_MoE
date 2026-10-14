"""
Dependency Expert - Models service relationships using GAT/Graph Transformer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    import dgl
    import dgl.nn as dglnn
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False


class GATDependencyExpert(nn.Module):
    """
    Dependency Expert using GAT (Graph Attention Network) for service relationship modeling.
    Works with the fixed 12-node, 24-edge dependency graph.
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
        residual: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.residual = residual
        
        if not DGL_AVAILABLE:
            raise RuntimeError("DGL is required for GATDependencyExpert")
        
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_feats = in_dim if i == 0 else hidden_dim * num_heads
            # Use hidden_dim for all layers to maintain consistent dimensions
            out_feats = hidden_dim
            
            self.gat_layers.append(
                dglnn.GATv2Conv(
                    in_feats=in_feats,
                    out_feats=out_feats,
                    num_heads=num_heads,
                    feat_drop=dropout,
                    attn_drop=dropout,
                    negative_slope=negative_slope,
                    residual=residual and i > 0,
                    allow_zero_in_degree=True,
                )
            )
            
            if i < num_layers - 1:
                self.norms.append(nn.LayerNorm(hidden_dim * num_heads))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim * num_heads, out_dim)
    
    def forward(self, graph, node_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            graph: DGL graph (batched) with 12 nodes per subgraph
            node_features: [B*12, in_dim] - features per node
        Returns:
            [B*12, out_dim] - updated node embeddings
        """
        h = node_features
        
        for i, gat_layer in enumerate(self.gat_layers):
            h = gat_layer(graph, h)  # [B*12, num_heads, out_feats]
            
            if i < self.num_layers - 1:
                # Merge heads
                h = h.flatten(1)  # [B*12, num_heads * out_feats]
                h = self.norms[i](h)
                h = F.elu(h)
            else:
                # Last layer: flatten heads for output projection
                h = h.flatten(1)  # [B*12, num_heads * out_feats]
        
        return self.output_proj(h)


class GraphTransformerExpert(nn.Module):
    """
    Dependency Expert using Graph Transformer (Transformer on graph structure).
    Alternative to GAT, uses full attention with graph structure bias.
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Linear(hidden_dim, out_dim)
    
    def forward(
        self, 
        node_features: torch.Tensor, 
        adjacency: Optional[torch.Tensor] = None,
        batch_size: int = 1
    ) -> torch.Tensor:
        """
        Args:
            node_features: [B*12, in_dim] or [B, 12, in_dim]
            adjacency: [12, 12] or [B, 12, 12] - adjacency matrix for structure bias
            batch_size: number of graphs
        Returns:
            [B*12, out_dim] or [B, 12, out_dim]
        """
        # Handle input shape
        if node_features.dim() == 2:
            # [B*12, in_dim] -> [B, 12, in_dim]
            node_features = node_features.view(batch_size, 12, -1)
        
        B, N, _ = node_features.shape
        
        # Project to hidden dim
        x = self.input_proj(node_features)  # [B, 12, hidden_dim]
        
        # Add structure bias if adjacency provided
        if adjacency is not None:
            if adjacency.dim() == 2:
                adjacency = adjacency.unsqueeze(0).expand(B, -1, -1)
            # Convert adjacency to attention bias (0 for connected, -inf for not)
            attn_bias = torch.where(adjacency > 0, 0.0, float('-inf'))
        else:
            attn_bias = None
        
        # Flatten for Transformer: [B, 12, hidden_dim] -> [B*12, hidden_dim]
        # Actually keep as [B, 12, hidden_dim] for batch_first=True
        x = self.encoder(x, mask=attn_bias)  # [B, 12, hidden_dim]
        
        # Project output
        x = self.output_proj(x)  # [B, 12, out_dim]
        
        return x


class DependencyExpert(nn.Module):
    """
    Unified Dependency Expert supporting both GAT and Graph Transformer.
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 64,
        expert_type: str = "gat",  # "gat" or "graph_transformer"
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.expert_type = expert_type
        
        if expert_type == "gat":
            if not DGL_AVAILABLE:
                raise RuntimeError("DGL required for GAT expert. Install dgl or use 'graph_transformer'.")
            self.expert = GATDependencyExpert(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                **kwargs
            )
        elif expert_type == "graph_transformer":
            self.expert = GraphTransformerExpert(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown expert_type: {expert_type}")
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.expert(*args, **kwargs)