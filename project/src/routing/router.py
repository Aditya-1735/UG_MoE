"""
Expert Router - Graph-aware and uncertainty-guided expert routing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class ExpertRouter(nn.Module):
    """
    Graph-aware and uncertainty-guided expert router.
    
    Routes input to specialized experts based on:
    - Modality representations
    - Graph structure information
    - Uncertainty estimates
    """
    
    def __init__(
        self,
        num_experts: int = 4,
        input_dim: int = 64,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        temperature: float = 1.0,
        use_graph: bool = True,
        use_uncertainty: bool = True,
        top_k: int = 2,  # Number of experts to route to
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.top_k = top_k
        self.use_graph = use_graph
        self.use_uncertainty = use_uncertainty
        
# Expert names
        self.expert_names = ["temporal", "semantic", "dependency", "cross_modal"]
        assert len(self.expert_names) == num_experts
        
        # Input projection - project to hidden_dim (will be concatenated with graph and uncertainty features)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Graph encoder (if using graph structure)
        if use_graph:
            self.graph_encoder = nn.Sequential(
                nn.Linear(24 * 2, hidden_dim),  # 24 edges * 2 (src, dst)
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
        
        # Uncertainty encoder (if using uncertainty)
        if use_uncertainty:
            self.uncertainty_encoder = nn.Sequential(
                nn.Linear(num_experts, hidden_dim),  # per-expert uncertainty
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
        
        # Routing network - always takes hidden_dim * 3 (metric + trace + log or graph + uncertainty)
        self.router = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )
        
        # Load balancing loss coefficient
        self.register_buffer("load_balancing_weight", torch.tensor(0.01))
    
    def forward(
        self,
        features: torch.Tensor,  # [B, input_dim]
        graph_info: Optional[torch.Tensor] = None,   # [B, 24*2] edge list flattened
        expert_uncertainties: Optional[torch.Tensor] = None,  # [B, num_experts]
        return_uncertainty: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            features: [B, input_dim] - combined modality features (already concatenated metric+trace+log)
            graph_info: Graph structure encoding [B, edge_features]
            expert_uncertainties: Per-expert uncertainty estimates [B, num_experts]
            return_uncertainty: Whether to return per-expert uncertainty
            
        Returns:
            routing_weights: [B, num_experts]
            uncertainties: [B, num_experts] (optional)
        """
        B = features.shape[0]
        
        # Project to hidden_dim
        x = self.input_proj(features)  # [B, hidden_dim]
        
        # Add graph info (if enabled and provided)
        if self.use_graph:
            if graph_info is not None:
                graph_feat = self.graph_encoder(graph_info)  # [B, hidden_dim]
            else:
                graph_feat = torch.zeros(B, self.hidden_dim, device=features.device, dtype=features.dtype)
            x = torch.cat([x, graph_feat], dim=-1)
        else:
            # Pad with zeros to maintain hidden_dim * 3 structure
            x = torch.cat([x, torch.zeros(B, self.hidden_dim, device=features.device, dtype=features.dtype)], dim=-1)
        
        # Add uncertainty info
        if self.use_uncertainty:
            if expert_uncertainties is not None:
                unc_feat = self.uncertainty_encoder(expert_uncertainties)  # [B, hidden_dim]
            else:
                unc_feat = torch.zeros(B, self.hidden_dim, device=features.device, dtype=features.dtype)
            x = torch.cat([x, unc_feat], dim=-1)
        else:
            # Pad with zeros to maintain hidden_dim * 3 structure
            x = torch.cat([x, torch.zeros(B, self.hidden_dim, device=features.device, dtype=features.dtype)], dim=-1)
        
        # Compute routing logits
        logits = self.router(x) / self.temperature  # [B, num_experts]
        
        # Soft routing weights
        routing_weights = F.softmax(logits, dim=-1)  # [B, num_experts]
        
        # Top-k routing
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Renormalize top-k
        top_k_weights = F.softmax(top_k_weights, dim=-1)
        
        # Create sparse routing matrix
        sparse_weights = torch.zeros_like(routing_weights)
        sparse_weights.scatter_(-1, top_k_indices, top_k_weights)
        
        # Load balancing loss
        aux_loss = None
        if self.training:
            expert_fraction = sparse_weights.mean(dim=0)
            target_fraction = 1.0 / self.num_experts
            loss = self.num_experts * torch.sum(expert_fraction ** 2) - 1.0
            aux_loss = self.load_balancing_weight * loss
        
        if return_uncertainty:
            # Return dummy uncertainties for compatibility
            return sparse_weights, torch.zeros_like(sparse_weights)
        
        return sparse_weights, None
    
    def _compute_load_balancing_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """
        Load balancing loss to encourage uniform expert utilization.
        Based on Switch Transformer / GShard load balancing.
        """
        # routing_weights: [B, num_experts]
        expert_fraction = routing_weights.mean(dim=0)  # [num_experts]
        target_fraction = 1.0 / self.num_experts
        
        # Coefficient of variation squared
        loss = self.num_experts * torch.sum(expert_fraction ** 2) - 1.0
        return self.load_balancing_weight * loss


class GraphAwareRouter(nn.Module):
    """
    Graph-aware router that uses the dependency graph structure
    to make routing decisions based on service relationships.
    """
    
    def __init__(
        self,
        num_experts: int = 4,
        node_feat_dim: int = 64,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Graph attention for routing
        self.graph_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Per-node routing heads
        self.routing_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_experts),
            ) for _ in range(12)  # 12 services
        ])
        
        # Global routing head
        self.global_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )
    
    def forward(
        self,
        features: torch.Tensor,  # [B, input_dim]
        return_uncertainty: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            features: [B, input_dim] - combined modality features
            return_uncertainty: Whether to return per-expert uncertainty
            
        Returns:
            routing_weights: [B, num_experts]
            uncertainties: [B, num_experts] (optional)
        """
        B = features.shape[0]
        
        # Encode features
        h = self.node_encoder(features)  # [B, hidden_dim]
        
        # Add dummy dimension for graph attention (treat as single node)
        h = h.unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Graph attention (services attend to each other)
        # We'll use a simple approach: self-attention
        h_attn, _ = self.graph_attention(h, h, h)  # [B, 1, hidden_dim]
        h = h + h_attn  # Residual
        h = h.squeeze(1)  # [B, hidden_dim]
        
        # Global routing
        global_routing = F.softmax(self.global_head(h), dim=-1)  # [B, num_experts]
        
        if return_uncertainty:
            # Return dummy uncertainties for compatibility
            return global_routing, torch.zeros_like(global_routing)
        
        return global_routing, None


class UncertaintyGuidedRouter(nn.Module):
    """
    Uncertainty-guided router that uses expert uncertainty estimates
    to make more robust routing decisions.
    """
    
    def __init__(
        self,
        num_experts: int = 4,
        input_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        uncertainty_weight: float = 1.0,
        top_k: int = 2,  # Accept for compatibility, not used in this router
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.uncertainty_weight = uncertainty_weight
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Uncertainty processor
        self.uncertainty_processor = nn.Sequential(
            nn.Linear(num_experts, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Routing head
        self.router = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )
        
        # Uncertainty estimator for each expert
        self.expert_uncertainty = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
                nn.Softplus(),  # Ensure positive
            ) for _ in range(num_experts)
        ])
    
    def forward(
        self,
        features: torch.Tensor,  # [B, input_dim]
        graph_info: Optional[torch.Tensor] = None,  # Ignored, for compatibility
        expert_uncertainties: Optional[torch.Tensor] = None,  # [B, num_experts]
        return_uncertainty: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            features: [B, input_dim] - combined modality features
            graph_info: Ignored, for compatibility
            expert_uncertainties: [B, num_experts] - optional external uncertainties
            return_uncertainty: Whether to return per-expert uncertainty
            
        Returns:
            routing_weights: [B, num_experts]
            uncertainties: [B, num_experts] (optional)
        """
        B = features.shape[0]
        
        # Encode features
        h = self.encoder(features)  # [B, hidden_dim]
        
        # Estimate per-expert uncertainty
        uncertainties = []
        for i in range(self.num_experts):
            unc = self.expert_uncertainty[i](h)  # [B, 1]
            uncertainties.append(unc)
        uncertainties = torch.cat(uncertainties, dim=-1)  # [B, num_experts]
        
        # Process uncertainty
        unc_feat = self.uncertainty_processor(uncertainties)  # [B, hidden_dim]
        
        # Combine with features
        combined = torch.cat([h, unc_feat], dim=-1)  # [B, 2*hidden_dim]
        
        # Routing logits (penalize high uncertainty experts)
        logits = self.router(combined)  # [B, num_experts]
        
        # Apply uncertainty penalty
        logits = logits - self.uncertainty_weight * uncertainties
        
        routing_weights = F.softmax(logits, dim=-1)
        
        if return_uncertainty:
            return routing_weights, uncertainties
        return routing_weights, None


def create_router(
    router_type: str = "base",
    num_experts: int = 4,
    **kwargs
) -> nn.Module:
    """
    Factory function to create router.
    
    Args:
        router_type: "base", "graph_aware", "uncertainty_guided"
        num_experts: Number of experts
        **kwargs: Additional arguments
        
    Returns:
        Router module
    """
    # Filter kwargs based on router type
    if router_type == "base":
        valid_keys = {'input_dim', 'hidden_dim', 'num_heads', 'dropout', 'temperature', 'use_graph', 'use_uncertainty', 'top_k', 'load_balancing_weight'}
    elif router_type == "graph_aware":
        valid_keys = {'node_feat_dim', 'hidden_dim', 'num_heads', 'dropout'}
    elif router_type == "uncertainty_guided":
        valid_keys = {'input_dim', 'hidden_dim', 'dropout', 'uncertainty_weight', 'top_k'}
    else:
        valid_keys = set()
    
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    
    if router_type == "base":
        return ExpertRouter(num_experts=num_experts, **filtered_kwargs)
    elif router_type == "graph_aware":
        return GraphAwareRouter(num_experts=num_experts, **filtered_kwargs)
    elif router_type == "uncertainty_guided":
        return UncertaintyGuidedRouter(num_experts=num_experts, **filtered_kwargs)
    else:
        raise ValueError(f"Unknown router_type: {router_type}")