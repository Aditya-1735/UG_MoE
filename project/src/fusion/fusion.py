"""
Adaptive Multimodal Fusion - Learnable importance of logs, traces, metrics.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class AdaptiveFusion(nn.Module):
    """
    Adaptive Multimodal Fusion.
    
    Learns the importance of each modality (logs, traces, metrics)
    conditioned on the input, using confidence/uncertainty estimates.
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int] = None,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        fusion_type: str = "attention",  # "attention", "weighted_concat", "gated"
    ):
        super().__init__()
        
        if modality_dims is None:
            modality_dims = {"metric": 64, "trace": 64, "log": 64}
        
        self.modality_dims = modality_dims
        self.modalities = list(modality_dims.keys())
        self.fusion_type = fusion_type
        self.hidden_dim = hidden_dim
        
        # Per-modality projections to common dim
        self.projections = nn.ModuleDict({
            mod: nn.Linear(dim, hidden_dim) 
            for mod, dim in modality_dims.items()
        })
        
        if fusion_type == "attention":
            # Cross-modality attention fusion
            self.cross_attention = nn.ModuleDict()
            for mod in self.modalities:
                others = [m for m in self.modalities if m != mod]
                self.cross_attention[mod] = nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
            
            # Output projection
            self.output_proj = nn.Linear(hidden_dim * len(self.modalities), hidden_dim)
            
        elif fusion_type == "weighted_concat":
            # Learnable weights per modality
            self.modality_weights = nn.ParameterDict({
                mod: nn.Parameter(torch.ones(1)) 
                for mod in self.modalities
            })
            self.output_proj = nn.Linear(hidden_dim * len(self.modalities), hidden_dim)
            
        elif fusion_type == "gated":
            # Gated fusion with per-modality gates
            self.gates = nn.ModuleDict({
                mod: nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Sigmoid(),
                ) for mod in self.modalities
            })
            self.output_proj = nn.Linear(hidden_dim * len(self.modalities), hidden_dim)
            
        elif fusion_type == "confidence":
            # Confidence-weighted fusion (like MUAD's CFM)
            self.confidence_nets = nn.ModuleDict({
                mod: nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid(),
                ) for mod in self.modalities
            })
            self.output_proj = nn.Linear(hidden_dim * len(self.modalities), hidden_dim)
            
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        modalities: Dict[str, torch.Tensor],  # {mod: [B, D]}
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Args:
            modalities: Dict mapping modality name to embeddings [B, D]
            return_weights: Whether to return fusion weights
            
        Returns:
            fused: [B, hidden_dim] - fused representation
            weights: Optional dict of modality weights
        """
        B = list(modalities.values())[0].shape[0]
        
        # Project all modalities to hidden_dim
        projected = {}
        for mod in self.modalities:
            if mod in modalities:
                projected[mod] = self.projections[mod](modalities[mod])
            else:
                projected[mod] = torch.zeros(B, self.hidden_dim, device=list(modalities.values())[0].device)
        
        if self.fusion_type == "attention":
            # Cross-attention between modalities
            attended = {}
            for mod in self.modalities:
                # Query: this modality, Key/Value: all modalities
                q = projected[mod].unsqueeze(1)  # [B, 1, hidden_dim]
                kv = torch.stack([projected[m] for m in self.modalities], dim=1)  # [B, num_mod, hidden_dim]
                
                attn_out, attn_weights = self.cross_attention[mod](
                    query=q, key=kv, value=kv
                )
                attended[mod] = attn_out.squeeze(1)  # [B, hidden_dim]
            
            # Concatenate attended features
            fused = torch.cat([attended[m] for m in self.modalities], dim=-1)
            
        elif self.fusion_type == "weighted_concat":
            # Weighted concatenation
            weights = {}
            for mod in self.modalities:
                weights[mod] = F.softplus(self.modality_weights[mod])
            
            # Normalize weights
            total_weight = sum(weights.values())
            norm_weights = {m: w / total_weight for m, w in weights.items()}
            
            fused = torch.cat([
                projected[m] * norm_weights[m] 
                for m in self.modalities
            ], dim=-1)
            
        elif self.fusion_type == "gated":
            # Gated fusion
            gated = {}
            for mod in self.modalities:
                gate = self.gates[mod](projected[mod])  # [B, hidden_dim]
                gated[mod] = projected[mod] * gate
            
            fused = torch.cat([gated[m] for m in self.modalities], dim=-1)
            
        elif self.fusion_type == "confidence":
            # Confidence-weighted fusion (MUAD CFM style)
            confidences = {}
            for mod in self.modalities:
                confidences[mod] = self.confidence_nets[mod](projected[mod])  # [B, 1]
            
            fused = torch.cat([
                projected[m] * confidences[m] 
                for m in self.modalities
            ], dim=-1)
        
        # Final projection
        fused = self.output_proj(fused)  # [B, hidden_dim]
        fused = self.norm(fused)
        fused = self.dropout(fused)
        
        if return_weights:
            if self.fusion_type in ("weighted_concat", "confidence"):
                if self.fusion_type == "weighted_concat":
                    weights_dict = {m: F.softplus(self.modality_weights[m]) for m in self.modalities}
                    total = sum(weights_dict.values())
                    weights_dict = {m: w / total for m, w in weights_dict.items()}
                else:
                    weights_dict = {m: c for m, c in confidences.items()}
                return fused, weights_dict
        
        return fused, None


class HierarchicalFusion(nn.Module):
    """
    Hierarchical Fusion: First fuse within modality (if multi-view), 
    then fuse across modalities.
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 128,
        intra_fusion: str = "attention",  # "attention", "mean", "max"
        inter_fusion: str = "attention",  # "attention", "concat", "gated"
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.modalities = list(modality_dims.keys())
        
        # Intra-modality fusion (if multi-view per modality)
        # For now, assume single view per modality
        
        # Inter-modality fusion
        self.inter_fusion = AdaptiveFusion(
            modality_dims={m: d for m, d in modality_dims.items()},
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            fusion_type=inter_fusion,
        )
    
    def forward(
        self,
        modalities: Dict[str, torch.Tensor],
        return_weights: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            modalities: {mod: [B, D]} or {mod: [B, V, D]} for multi-view
        Returns:
            fused: [B, hidden_dim]
        """
        # Handle multi-view if present
        processed = {}
        for mod, feat in modalities.items():
            if feat.dim() == 3:
                # [B, V, D] - multi-view, apply intra-fusion
                if hasattr(self, 'intra_fusion_type') and self.intra_fusion_type == "attention":
                    # Self-attention over views
                    feat = feat.mean(dim=1)  # Simple mean for now
                else:
                    feat = feat.mean(dim=1)
            processed[mod] = feat
        
        return self.inter_fusion(processed, return_weights=return_weights)


class DynamicFusion(nn.Module):
    """
    Dynamic Fusion that adapts based on input uncertainty/quality.
    
    Uses uncertainty estimates to dynamically weight modalities.
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.modalities = list(modality_dims.keys())
        
        # Projections
        self.projections = nn.ModuleDict({
            mod: nn.Linear(dim, hidden_dim) 
            for mod, dim in modality_dims.items()
        })
        
        # Quality/uncertainty estimator per modality
        self.quality_estimators = nn.ModuleDict({
            mod: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),  # Quality score in [0, 1]
            ) for mod in self.modalities
        })
        
        # Fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim * len(self.modalities), hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        modalities: Dict[str, torch.Tensor],
        return_quality: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Args:
            modalities: {mod: [B, D]}
            return_quality: Whether to return quality scores
            
        Returns:
            fused: [B, hidden_dim]
            quality: {mod: [B, 1]} quality scores
        """
        B = list(modalities.values())[0].shape[0]
        
        projected = {}
        qualities = {}
        
        for mod in self.modalities:
            if mod in modalities:
                proj = self.projections[mod](modalities[mod])  # [B, hidden_dim]
                projected[mod] = proj
                qualities[mod] = self.quality_estimators[mod](proj)  # [B, 1]
            else:
                projected[mod] = torch.zeros(B, self.hidden_dim, device=list(modalities.values())[0].device)
                qualities[mod] = torch.zeros(B, 1, device=list(modalities.values())[0].device)
        
        # Quality-weighted fusion
        quality_weights = torch.cat([qualities[m] for m in self.modalities], dim=-1)  # [B, num_mod]
        quality_weights = F.softmax(quality_weights, dim=-1)  # [B, num_mod]
        
        # Weighted concatenation
        weighted = []
        for i, mod in enumerate(self.modalities):
            weighted.append(projected[mod] * quality_weights[:, i:i+1])  # [B, hidden_dim]
        
        fused = torch.cat(weighted, dim=-1)  # [B, num_mod * hidden_dim]
        fused = self.fusion_net(fused)  # [B, hidden_dim]
        fused = self.norm(fused)
        fused = self.dropout(fused)
        
        if return_quality:
            return fused, qualities
        
        return fused, None


class ModalityDropout(nn.Module):
    """
    Modality Dropout for robust fusion training.
    Randomly drops modalities during training to prevent over-reliance.
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 128,
        dropout_prob: float = 0.1,
        fusion_module: nn.Module = None,
    ):
        super().__init__()
        self.modality_dims = modality_dims
        self.modalities = list(modality_dims.keys())
        self.dropout_prob = dropout_prob
        
        if fusion_module is None:
            self.fusion = AdaptiveFusion(modality_dims, fusion_type="confidence")
        else:
            self.fusion = fusion_module
    
    def forward(
        self,
        modalities: Dict[str, torch.Tensor],
        training: bool = True,
    ) -> torch.Tensor:
        """
        During training, randomly drop modalities.
        """
        if training and self.training:
            # Randomly select modalities to keep
            available = list(modalities.keys())
            num_to_keep = max(1, int(len(available) * (1 - self.dropout_prob)))
            
            # Keep at least one modality
            keep = torch.randperm(len(available))[:num_to_keep]
            kept_modalities = {available[i]: modalities[available[i]] for i in keep}
        else:
            kept_modalities = modalities
        
        # Fill missing with zeros
        for mod in self.modalities:
            if mod not in kept_modalities:
                # Get device from existing modalities
                device = list(kept_modalities.values())[0].device
                kept_modalities[mod] = torch.zeros(
                    1, self.modality_dims[mod], device=device
                )
        
        return self.fusion(kept_modalities)[0]