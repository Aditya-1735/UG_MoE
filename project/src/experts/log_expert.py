"""
Log-Only Experts for Log-Only MoE (HDFS, etc.)

These experts replace the multimodal experts when log_only_mode=True:
- LogExpert: Template sequence encoding (replaces LogSemanticExpert)
- LogGraphExpert: Log co-occurrence graph processing (replaces DependencyExpert)
- LogContextExpert: Intra-sequence attention (replaces CrossModalExpert)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    import dgl
    import dgl.nn as dglnn
    import dgl.nn.pytorch as dglnn_pytorch
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False
    dgl = None
    dglnn = None
    dglnn_pytorch = None


class LogExpert(nn.Module):
    """
    Log Expert for template sequence encoding.
    Replaces LogSemanticExpert for log-only datasets.
    
    Processes variable-length template ID sequences using Transformer encoder.
    """
    
    def __init__(
        self,
        vocab_size: int = 50,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 100,
        use_transformer: bool = True,
        pooling: str = "attention",  # "attention", "mean", "cls"
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.use_transformer = use_transformer
        self.pooling = pooling
        
        # Event embedding (0 = padding)
        self.event_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_dim = embed_dim
        else:
            # Simple feed-forward on embeddings
            self.encoder = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embed_dim),
            )
            self.output_dim = embed_dim
        
        # Pooling
        if pooling == "attention":
            self.pool_projection = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, 1),
            )
        elif pooling == "cls":
            # Learnable CLS token
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        else:
            # mean pooling - no extra params
            pass
        
        # Output projection to hidden_dim
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        template_ids: torch.Tensor,  # [B, seq_len]
        attention_mask: Optional[torch.Tensor] = None,  # [B, seq_len], 1=valid, 0=pad
    ) -> torch.Tensor:
        """
        Args:
            template_ids: [B, seq_len] - event template IDs (0 = padding)
            attention_mask: [B, seq_len] - 1 for valid, 0 for padding
        Returns:
            [B, embed_dim] - pooled sequence representation
        """
        B, seq_len = template_ids.shape
        
        # Truncate if needed
        if seq_len > self.max_seq_len:
            template_ids = template_ids[:, :self.max_seq_len]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_seq_len]
            seq_len = self.max_seq_len
        
        # Embeddings
        x = self.event_embedding(template_ids)  # [B, seq_len, embed_dim]
        
        # Position embeddings
        positions = torch.arange(seq_len, device=template_ids.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embedding(positions)
        x = self.dropout(x)
        
        if self.use_transformer:
            if attention_mask is not None:
                # Transformer expects True for masked positions
                attn_mask = (attention_mask == 0)
            else:
                attn_mask = None
            x = self.encoder(x, src_key_padding_mask=attn_mask)  # [B, seq_len, embed_dim]
        else:
            x = self.encoder(x)  # [B, seq_len, embed_dim]
        
        x = self.norm(x)
        
        # Pooling
        if self.pooling == "attention":
            # Attention pooling
            attn_weights = self.pool_projection(x)  # [B, seq_len, 1]
            if attention_mask is not None:
                attn_weights = attn_weights.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)
            attn_weights = torch.softmax(attn_weights, dim=1)
            x = (x * attn_weights).sum(dim=1)  # [B, embed_dim]
        elif self.pooling == "cls":
            # Use CLS token (first position after prepending)
            cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
            x = torch.cat([cls_tokens, x], dim=1)  # [B, seq_len+1, embed_dim]
            x = x[:, 0]  # [B, embed_dim]
        else:
            # Mean pooling with mask
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                x = x.mean(dim=1)
        
        return self.output_projection(x)


class LogGraphExpert(nn.Module):
    """
    Log Graph Expert for log co-occurrence graph processing.
    Replaces DependencyExpert for log-only mode.
    
    Supports GAT, GraphTransformer, and simple GCN.
    """
    
    def __init__(
        self,
        in_dim: int = 64,
        hidden_dim: int = 64,
        out_dim: int = 64,
        expert_type: str = "gat",  # "gat", "graph_transformer", "gcn"
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        residual: bool = True,
    ):
        super().__init__()
        self.expert_type = expert_type
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        if expert_type == "gat":
            if not DGL_AVAILABLE:
                raise RuntimeError("DGL is required for GAT. Install dgl or use 'graph_transformer'.")
            self.expert = self._build_gat(in_dim, hidden_dim, out_dim, num_heads, num_layers, dropout, residual)
        elif expert_type == "graph_transformer":
            self.expert = self._build_graph_transformer(in_dim, hidden_dim, out_dim, num_layers, num_heads, dropout)
        elif expert_type == "gcn":
            if not DGL_AVAILABLE:
                raise RuntimeError("DGL is required for GCN.")
            self.expert = self._build_gcn(in_dim, hidden_dim, out_dim, num_layers, dropout)
        else:
            raise ValueError(f"Unknown expert_type: {expert_type}")
    
    def _build_gat(self, in_dim, hidden_dim, out_dim, num_heads, num_layers, dropout, residual):
        class GATExpert(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_layers = num_layers
                self.gat_layers = nn.ModuleList()
                self.norms = nn.ModuleList()
                
                for i in range(num_layers):
                    in_feats = in_dim if i == 0 else hidden_dim * num_heads
                    if DGL_AVAILABLE:
                        self.gat_layers.append(
                            dglnn_pytorch.GATv2Conv(
                                in_feats=in_feats,
                                out_feats=hidden_dim,
                                num_heads=num_heads,
                                feat_drop=dropout,
                                attn_drop=dropout,
                                negative_slope=0.2,
                                residual=residual and i > 0,
                                allow_zero_in_degree=True,
                            )
                        )
                    else:
                        # Placeholder for when DGL is not available
                        self.gat_layers.append(None)
                    if i < num_layers - 1:
                        self.norms.append(nn.LayerNorm(hidden_dim * num_heads))
                
                self.output_proj = nn.Linear(hidden_dim * num_heads, out_dim)
            
            def forward(self, graph, node_features):
                if not DGL_AVAILABLE:
                    raise RuntimeError("DGL is required for GAT. Install dgl or use 'graph_transformer'.")
                h = node_features
                for i, gat_layer in enumerate(self.gat_layers):
                    h = gat_layer(graph, h)  # [N, num_heads, out_feats]
                    if i < self.num_layers - 1:
                        h = h.flatten(1)  # [N, num_heads * out_feats]
                        h = self.norms[i](h)
                        h = F.elu(h)
                    else:
                        h = h.flatten(1)
                return self.output_proj(h)
        return GATExpert()
    
    def _build_graph_transformer(self, in_dim, hidden_dim, out_dim, num_layers, num_heads, dropout):
        class GraphTransformerExpert(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_proj = nn.Linear(in_dim, hidden_dim)
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.output_proj = nn.Linear(hidden_dim, out_dim)
            
            def forward(self, graph, node_features):
                # node_features: [N, in_dim] or [B, N, in_dim]
                if node_features.dim() == 2:
                    # Single graph: [N, in_dim] -> add batch dim
                    x = node_features.unsqueeze(0)  # [1, N, in_dim]
                    squeeze_output = True
                else:
                    x = node_features
                    squeeze_output = False
                
                B, N, _ = x.shape
                
                # Project to hidden dim
                x = self.input_proj(x)  # [B, N, hidden_dim]
                
                # Build attention bias from graph structure
                if hasattr(graph, 'adjacency_matrix'):
                    adj = graph.adjacency_matrix().to_dense()
                    if adj.dim() == 2:
                        adj = adj.unsqueeze(0).expand(x.shape[0], -1, -1)
                    attn_bias = torch.where(adj > 0, 0.0, float('-inf'))
                else:
                    # Build from edges
                    adj = torch.zeros(N, N, device=node_features.device)
                    src, dst = graph.edges()
                    adj[src, dst] = 1
                    adj = adj.unsqueeze(0).expand(x.shape[0], -1, -1)
                    attn_bias = torch.where(adj > 0, 0.0, float('-inf'))
                
                x = self.encoder(x, mask=attn_bias)  # [B, N, hidden_dim]
                
                # Graph-level pooling (mean over nodes)
                x = x.mean(dim=1)  # [B, hidden_dim]
                
                x = self.output_proj(x)
                
                if squeeze_output:
                    x = x.squeeze(0)
                return x
        return GraphTransformerExpert()
    
    def _build_gcn(self, in_dim, hidden_dim, out_dim, num_layers, dropout):
        class GCNExpert(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList()
                self.norms = nn.ModuleList()
                
                for i in range(num_layers):
                    in_feats = in_dim if i == 0 else hidden_dim
                    if DGL_AVAILABLE:
                        self.layers.append(
                            dglnn_pytorch.GraphConv(
                                in_feats=in_feats,
                                out_feats=hidden_dim,
                                activation=F.relu if i < num_layers - 1 else None,
                                dropout=dropout,
                                allow_zero_in_degree=True,
                            )
                        )
                    else:
                        self.layers.append(None)
                    if i < num_layers - 1:
                        self.norms.append(nn.LayerNorm(hidden_dim))
                
                self.output_proj = nn.Linear(hidden_dim, out_dim)
            
            def forward(self, graph, node_features):
                if not DGL_AVAILABLE:
                    raise RuntimeError("DGL is required for GCN. Install dgl or use 'graph_transformer'.")
                h = node_features
                for i, layer in enumerate(self.layers):
                    h = layer(graph, h)
                    if i < num_layers - 1:
                        h = self.norms[i](h)
                return self.output_proj(h)
        return GCNExpert()
    
    def forward(self, graph, node_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            graph: DGL graph with log template co-occurrence structure
            node_features: [N, in_dim] or [B, N, in_dim]
        Returns:
            [N, out_dim] or [B, out_dim] - node/graph embeddings
        """
        return self.expert(graph, node_features)


class LogContextExpert(nn.Module):
    """
    Log Context Expert for intra-sequence attention.
    Replaces CrossModalExpert for log-only mode.
    
    Performs self-attention within the log sequence to capture
    contextual relationships between templates.
    """
    
    def __init__(
        self,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 100,
        use_positional: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.use_positional = use_positional
        
        if use_positional:
            self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Pooling
        self.pool_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )
    
    def forward(
        self,
        template_ids: torch.Tensor,  # [B, seq_len]
        attention_mask: Optional[torch.Tensor] = None,  # [B, seq_len]
    ) -> torch.Tensor:
        """
        Args:
            template_ids: [B, seq_len] - template IDs (already embedded externally)
            attention_mask: [B, seq_len] - 1 for valid, 0 for padding
        Returns:
            [B, embed_dim] - contextual representation
        """
        # This expert expects pre-embedded features
        # The input should be [B, seq_len, embed_dim]
        if template_ids.dim() == 2:
            raise ValueError("LogContextExpert expects pre-embedded input [B, seq_len, embed_dim]")
        
        x = template_ids  # [B, seq_len, embed_dim]
        B, seq_len, _ = x.shape
        
        if self.use_positional:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.shape[0], -1)
            x = x + self.pos_embedding(positions)
        
        x = self.dropout(x)
        
        if attention_mask is not None:
            attn_mask = (attention_mask == 0)
        else:
            attn_mask = None
        
        x = self.encoder(x, src_key_padding_mask=attn_mask)  # [B, seq_len, embed_dim]
        x = self.norm(x)
        
        # Attention pooling
        attn_weights = self.pool_projection(x)  # [B, seq_len, 1]
        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)
        attn_weights = torch.softmax(attn_weights, dim=1)
        x = (x * attn_weights).sum(dim=1)  # [B, embed_dim]
        
        return self.output_projection(x)


def create_log_experts(
    vocab_size: int = 50,
    embed_dim: int = 64,
    hidden_dim: int = 128,
    expert_hidden_dim: int = 128,
    max_seq_len: int = 100,
    dropout: float = 0.1,
    log_graph_type: str = "gat",
    num_heads: int = 4,
    num_layers: int = 2,
) -> nn.ModuleDict:
    """
    Factory function to create all log-only experts.
    
    Returns:
        ModuleDict with keys: "log", "log_graph", "log_context"
    """
    experts = nn.ModuleDict()
    
    # Log Expert: template sequence encoder
    experts["log"] = LogExpert(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=expert_hidden_dim,
        num_layers=2,
        num_heads=4,
        dropout=dropout,
        max_seq_len=max_seq_len,
        pooling="attention",
    )
    
    # Log Graph Expert: log co-occurrence graph processor
    experts["log_graph"] = LogGraphExpert(
        in_dim=embed_dim,
        hidden_dim=expert_hidden_dim,
        out_dim=embed_dim,
        expert_type=log_graph_type,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )
    
    # Log Context Expert: intra-sequence attention
    experts["log_context"] = LogContextExpert(
        embed_dim=embed_dim,
        hidden_dim=expert_hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        max_seq_len=100,
    )
    
    return experts