"""
Semantic Expert - Models log/event semantics using Transformer or embedding.
"""
import torch
import torch.nn as nn
from typing import Optional


class SemanticExpert(nn.Module):
    """
    Semantic Expert for log/event semantics.
    Uses Transformer encoder or embedding-based approach.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        use_transformer: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.use_transformer = use_transformer
        
        # Event embedding
        self.event_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_dim = embed_dim
        else:
            # Simple feed-forward on embeddings
            self.encoder = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embed_dim),
            )
            self.output_dim = embed_dim
        
        # Pooling projection
        self.pool_projection = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, event_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            event_ids: [batch, seq_len] - event template IDs (0 = padding/unseen)
            attention_mask: [batch, seq_len] - 1 for valid, 0 for padding
        Returns:
            [batch, embed_dim] - pooled semantic representation
        """
        batch_size, seq_len = event_ids.shape
        
        # Embeddings
        x = self.event_embedding(event_ids)  # [batch, seq_len, embed_dim]
        
        # Position embeddings
        positions = torch.arange(seq_len, device=event_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_embedding(positions)
        
        if self.use_transformer:
            if attention_mask is not None:
                # Transformer expects True for masked positions
                attn_mask = (attention_mask == 0)
            else:
                attn_mask = None
            x = self.encoder(x, src_key_padding_mask=attn_mask)  # [batch, seq_len, embed_dim]
        else:
            x = self.encoder(x)  # [batch, seq_len, embed_dim]
        
        # Pool over sequence (masked mean pooling)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        
        return self.pool_projection(x)


class LogSemanticExpert(nn.Module):
    """
    Semantic Expert specialized for log data from the dataset.
    Handles the log format: [batch, num_nodes, event_dim] where event_dim is the 
    Hawkes intensity vector or event count vector.
    """
    
    def __init__(
        self,
        input_dim: int = 14,  # event_num from metadata
        hidden_dim: int = 64,
        embed_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Since logs in dataset are already Hawkes intensity vectors [12, 14] per graph
        # We treat each node's log vector as a "token" and use cross-node attention
        self.node_projection = nn.Linear(input_dim, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Graph-level pooling
        self.graph_pool = nn.Linear(embed_dim, 1)  # Attention weights for nodes
        self.output_projection = nn.Linear(embed_dim, hidden_dim)
    
    def forward(self, log_features: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Args:
            log_features: [batch * 12, event_dim] - log features per node
            batch_size: number of graphs in batch
        Returns:
            [batch, hidden_dim] - graph-level semantic embedding
        """
        num_nodes = 12
        
        # Project node features
        x = self.node_projection(log_features)  # [batch*12, embed_dim]
        
        # Reshape to [batch, 12, embed_dim]
        x = x.view(batch_size, num_nodes, -1)
        
        # Cross-node attention
        x = self.encoder(x)  # [batch, 12, embed_dim]
        
        # Attention pooling over nodes
        attn_weights = torch.softmax(self.graph_pool(x), dim=1)  # [batch, 12, 1]
        pooled = (x * attn_weights).sum(dim=1)  # [batch, embed_dim]
        
        return self.output_projection(pooled)