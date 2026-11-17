"""
Graph-Aware Mixture-of-Experts for Multimodal Microservice Anomaly Detection.

This is the main proposed model (Track B) that extends MUAD with:
- Specialized experts (Temporal, Semantic, Dependency, Cross-modal)
- Graph-aware and uncertainty-guided expert routing
- Uncertainty modeling per expert
- Adaptive multimodal fusion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any

from src.experts import (
    TemporalExpertGraph,
    LogSemanticExpert,
    DependencyExpert,
    CrossModalExpertGraph,
)
from src.routing import ExpertRouter, create_router
from src.uncertainty import UncertaintyBlock, compute_predictive_uncertainty
from src.fusion import AdaptiveFusion, DynamicFusion


class GraphAwareMoE(nn.Module):
    """
    Graph-Aware Mixture-of-Experts for Multimodal Anomaly Detection.
    
    Architecture:
    1. Modality encoders (from MUAD baseline)
    2. Specialized experts (Temporal, Semantic, Dependency, Cross-modal)
    3. Graph-aware + uncertainty-guided expert router
    4. Per-expert uncertainty blocks
    5. Adaptive multimodal fusion
    6. Anomaly classifier
    """
    
    def __init__(
        self,
        # Data dimensions
        num_nodes: int = 12,
        metric_dim: int = 7,
        trace_dim: int = 1,
        log_dim: int = 14,  # event_num from metadata
        num_classes: int = 2,
        
        # Model dimensions
        hidden_dim: int = 64,
        expert_hidden_dim: int = 128,
        fusion_dim: int = 128,
        router_hidden_dim: int = 128,
        
        # Expert configuration
        num_experts: int = 4,
        expert_types: List[str] = None,
        
        # Routing configuration
        router_type: str = "uncertainty_guided",  # "base", "graph_aware", "uncertainty_guided"
        top_k: int = 2,
        use_graph_routing: bool = True,
        use_uncertainty_routing: bool = True,
        
        # Fusion configuration
        fusion_type: str = "confidence",  # "attention", "weighted_concat", "gated", "confidence", "dynamic"
        
        # Uncertainty
        use_uncertainty: bool = True,
        uncertainty_type: str = "variational",  # "standard", "variational"
        
        # Training
        dropout: float = 0.1,
        dependency_expert_type: str = "gat",  # "gat" or "graph_transformer"

        # MUAD baseline compatibility
        muad_metric_encoder: Optional[nn.Module] = None,
        muad_trace_encoder: Optional[nn.Module] = None,
        muad_log_encoder: Optional[nn.Module] = None,
        muad_graph_model: Optional[nn.Module] = None,
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.use_uncertainty = use_uncertainty

        if expert_types is None:
            expert_types = ["temporal", "semantic", "dependency", "cross_modal"]
        self.expert_types = expert_types
        self.num_experts = len(expert_types)
        
        # ============================================================
        # 1. Modality Encoders (reuse MUAD baseline if provided)
        # ============================================================
        class GRUEncoder(nn.Module):
            """GRU encoder that returns last hidden state."""
            def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
                super().__init__()
                self.gru = nn.GRU(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                )
                self.linear = nn.Linear(hidden_size, hidden_size)
            
            def forward(self, x):
                # x: [batch, seq_len, input_size]
                _, hidden = self.gru(x)
                # hidden: [num_layers, batch, hidden_size]
                last_hidden = hidden[-1]  # [batch, hidden_size]
                return self.linear(last_hidden)
        
        if muad_metric_encoder is not None:
            self.metric_encoder = muad_metric_encoder
        else:
            self.metric_encoder = GRUEncoder(metric_dim, hidden_dim)

        if muad_trace_encoder is not None:
            self.trace_encoder = muad_trace_encoder
        else:
            self.trace_encoder = GRUEncoder(trace_dim, hidden_dim)

        if muad_log_encoder is not None:
            self.log_encoder = muad_log_encoder
        else:
            self.log_encoder = nn.Linear(log_dim, hidden_dim)

        # Graph model (GAT) from MUAD; None -> mean pooling over nodes
        self.graph_model = muad_graph_model

        # ============================================================
        # 2. Specialized Experts
        # ============================================================
        self.experts = nn.ModuleDict()

        for exp_type in self.expert_types:
            if exp_type == "temporal":
                self.experts["temporal"] = TemporalExpertGraph(
                    metric_dim=metric_dim,
                    trace_dim=trace_dim,
                    hidden_dim=hidden_dim,
                    backbone="gru",
                )
            elif exp_type == "semantic":
                self.experts["semantic"] = LogSemanticExpert(
                    input_dim=log_dim,
                    hidden_dim=hidden_dim,
                    embed_dim=expert_hidden_dim,
                )
            elif exp_type == "dependency":
                self.experts["dependency"] = DependencyExpert(
                    in_dim=hidden_dim,
                    hidden_dim=expert_hidden_dim,
                    out_dim=hidden_dim,
                    expert_type=dependency_expert_type,
                    num_heads=4,
                    num_layers=2,
                    dropout=dropout,
                )
            elif exp_type == "cross_modal":
                self.experts["cross_modal"] = CrossModalExpertGraph(
                    metric_dim=hidden_dim,
                    trace_dim=hidden_dim,
                    log_dim=hidden_dim,
                    embed_dim=expert_hidden_dim,
                    hidden_dim=hidden_dim,
                    num_heads=4,
                    num_layers=2,
                    dropout=dropout,
                )
            else:
                raise ValueError(f"Unknown expert type: {exp_type}")

        self.dependency_expert_type = dependency_expert_type

        # ============================================================
        # 3. Expert Router (input: concatenated metric + trace + log embeddings)
        # ============================================================
        self.router = create_router(
            router_type=router_type,
            num_experts=self.num_experts,
            input_dim=hidden_dim * 3,
            hidden_dim=router_hidden_dim,
            dropout=dropout,
            top_k=min(top_k, self.num_experts),
            use_graph=use_graph_routing,
            use_uncertainty=use_uncertainty_routing,
        )
        
        # ============================================================
        # 4. Uncertainty Blocks (one per expert)
        # ============================================================
        if use_uncertainty:
            self.uncertainty_blocks = nn.ModuleDict()
            for exp_type in self.expert_types:
                if uncertainty_type == "variational":
                    from src.uncertainty import VariationalUncertaintyBlock
                    self.uncertainty_blocks[exp_type] = VariationalUncertaintyBlock(
                        in_dim=hidden_dim,
                        hidden_dim=expert_hidden_dim,
                        out_dim=hidden_dim,
                        dropout=dropout,
                    )
                else:
                    self.uncertainty_blocks[exp_type] = UncertaintyBlock(
                        in_dim=hidden_dim,
                        hidden_dim=expert_hidden_dim,
                        out_dim=hidden_dim,
                        dropout=dropout,
                    )
        
        # ============================================================
        # 5. Adaptive Multimodal Fusion
        # ============================================================
        modality_dims = {exp: hidden_dim for exp in self.expert_types}
        if fusion_type == "dynamic":
            self.fusion = DynamicFusion(modality_dims, hidden_dim=fusion_dim, dropout=dropout)
        else:
            self.fusion = AdaptiveFusion(
                modality_dims=modality_dims,
                hidden_dim=fusion_dim,
                dropout=dropout,
                fusion_type=fusion_type,
            )
        
        # ============================================================
        # 6. Anomaly Classifier
        # ============================================================
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        
        # Auxiliary losses storage
        self.aux_losses = {}
    
    def encode_modalities(self, graph) -> Dict[str, torch.Tensor]:
        """
        Encode raw modalities using MUAD baseline encoders.
        
        Args:
            graph: Batched DGL graph with ndata['metrics'], 'traces', 'logs'
            
        Returns:
            Graph-level {"metric", "trace", "log"}: [B, D] and per-node {"*_node"}: [B*N, D]
        """
        B = graph.batch_size
        
        metrics = graph.ndata["metrics"]   # [B*N, T, metric_dim]
        traces = graph.ndata["traces"]     # [B*N, T, trace_dim]
        logs = graph.ndata["logs"]         # [B*N, log_dim]
        
        # Encode each modality per node (keep per-node before pooling)
        metric_node = self.metric_encoder(metrics)  # GRU over time -> [B*N, hidden_dim]
        trace_node = self.trace_encoder(traces)     # GRU over time -> [B*N, hidden_dim]
        log_node = self.log_encoder(logs)           # Linear        -> [B*N, hidden_dim]
        
        # Apply graph model (GAT + pooling) if available, else mean-pool over nodes
        if self.graph_model is not None:
            metric_emb = self.graph_model(graph, metric_node)
            trace_emb = self.graph_model(graph, trace_node)
            log_emb = self.graph_model(graph, log_node)
        else:
            metric_emb = metric_node.view(B, self.num_nodes, -1).mean(dim=1)
            trace_emb = trace_node.view(B, self.num_nodes, -1).mean(dim=1)
            log_emb = log_node.view(B, self.num_nodes, -1).mean(dim=1)
        
        return {
            "metric": metric_emb,           # [B, hidden_dim] graph-level
            "trace": trace_emb,             # [B, hidden_dim] graph-level
            "log": log_emb,                 # [B, hidden_dim] graph-level
            "metric_node": metric_node,     # [B*12, hidden_dim] per-node
            "trace_node": trace_node,       # [B*12, hidden_dim] per-node
            "log_node": log_node,           # [B*12, hidden_dim] per-node
        }
    
    def forward(
        self,
        graph,
        labels: Optional[torch.Tensor] = None,
        return_expert_outputs: bool = False,
        return_routing: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass.
        
        Args:
            graph: Batched DGL graph
            labels: [B] ground truth labels (for training)
            return_expert_outputs: Whether to return individual expert outputs
            return_routing: Whether to return routing weights
            
        Returns:
            Dict with keys:
                - MMlogit: [B, num_classes] anomaly logits
                - loss: scalar (if labels provided)
                - y_pred: [B] predictions
                - expert_outputs: per-expert embeddings (optional)
                - routing_weights: [B, num_experts] (optional)
                - aux_losses: dict of auxiliary losses
        """
        B = graph.batch_size
        
        # 1. Encode modalities
        modality_embeddings = self.encode_modalities(graph)
        
        # NaN detection after modality encoding
        for key, val in modality_embeddings.items():
            if torch.is_tensor(val) and not torch.isfinite(val).all():
                print(f"  [NAN SOURCE] encode_modalities '{key}' produced NaN/Inf!")
                raise RuntimeError(f"NaN/Inf in encode_modalities.{key}")
        
        # 2. Run experts
        expert_outputs = {}
        expert_uncertainties = {}
        
        for exp_type in self.expert_types:
            expert = self.experts[exp_type]
            
            if exp_type == "temporal":
                out = expert(graph)
                # temporal returns dict with metric_embedding and trace_embedding, each [B, D]
                if isinstance(out, dict):
                    # Already graph-level, just pick one or average
                    out = out.get("metric_embedding", out.get("trace_embedding", list(out.values())[0]))
                    # out is already [B, D], no reshaping needed
            elif exp_type == "semantic":
                out = expert(graph.ndata["logs"], B)  # [B, D]
            elif exp_type == "dependency":
                # Use actual per-node metric embeddings (not broadcast graph-level)
                node_feats = modality_embeddings["metric_node"]  # [B*12, hidden_dim]
                if self.dependency_expert_type == "graph_transformer":
                    # GraphTransformerExpert expects [B, 12, hidden_dim] input
                    node_feats = node_feats.view(B, self.num_nodes, -1)
                    out = expert(node_feats, batch_size=B)  # [B, 12, hidden_dim]
                    out = out.mean(dim=1)  # Pool to [B, hidden_dim]
                else:
                    out = expert(graph, node_feats)  # GAT expects (graph, node_features)
            elif exp_type == "cross_modal":
                # Use actual per-node embeddings (not broadcast graph-level)
                graph.ndata["metric_emb"] = modality_embeddings["metric_node"]  # [B*12, hidden_dim]
                graph.ndata["trace_emb"] = modality_embeddings["trace_node"]    # [B*12, hidden_dim]
                graph.ndata["log_emb"] = modality_embeddings["log_node"]        # [B*12, hidden_dim]
                out = expert(graph)
                # CrossModalExpertGraph returns dict with fused_embedding [B, D]
                if isinstance(out, dict):
                    out = out.get("fused_embedding", list(out.values())[0])
            
            # Ensure output is [B, hidden_dim]
            if out.dim() == 3:
                out = out.mean(dim=1)
            elif out.dim() == 2 and out.shape[0] == B * self.num_nodes:
                out = out.view(B, self.num_nodes, -1).mean(dim=1)
            
            # NaN detection after each expert
            if not torch.isfinite(out).all():
                print(f"  [NAN SOURCE] Expert '{exp_type}' produced NaN/Inf output!")
                raise RuntimeError(f"NaN/Inf from expert {exp_type}")
            
            expert_outputs[exp_type] = out  # [B, hidden_dim]
            
            # 3. Uncertainty estimation
            if self.use_uncertainty:
                sample, kl_loss = self.uncertainty_blocks[exp_type](expert_outputs[exp_type])
                expert_outputs[exp_type] = sample
                expert_uncertainties[exp_type] = kl_loss
    
        # 4. Expert routing
        # Combine modality features for routing
        routing_features = torch.cat([
            modality_embeddings["metric"],
            modality_embeddings["trace"],
            modality_embeddings["log"],
        ], dim=-1)  # [B, 3*hidden_dim]
        
        # Get routing weights
        # Build graph_info: [B, 24*2] flattened edge list per subgraph
        src, dst = graph.edges()  # [E] global node IDs
        batch_num_edges = graph.batch_num_edges().tolist()
        src_split = torch.split(src, batch_num_edges)
        dst_split = torch.split(dst, batch_num_edges)
        graph_info_list = []
        for s, d in zip(src_split, dst_split):
            edges = torch.stack([s.float(), d.float()], dim=1).flatten()  # [48]
            graph_info_list.append(edges)
        graph_info = torch.stack(graph_info_list, dim=0)  # [B, 48]

        # Build per-expert uncertainties tensor [B, num_experts]
        if self.use_uncertainty and expert_uncertainties:
            unc_list = []
            for exp_type in self.expert_types:
                kl = expert_uncertainties.get(exp_type, 0.0)
                if not isinstance(kl, torch.Tensor):
                    kl = torch.tensor(kl, device=routing_features.device, dtype=routing_features.dtype)
                if kl.dim() == 0:
                    kl = kl.expand(B)
                unc_list.append(kl)
            expert_unc_tensor = torch.stack(unc_list, dim=1)  # [B, num_experts]
        else:
            expert_unc_tensor = None

        routing_weights, routing_aux_loss = self.router(
            routing_features,
            graph_info=graph_info,
            expert_uncertainties=expert_unc_tensor,
            return_uncertainty=True,
        )
        # Handle new return format from router (dict with aux_loss)
        if isinstance(routing_aux_loss, dict):
            routing_aux_loss = routing_aux_loss.get('aux_loss', 0)
        
        # NaN detection after router
        if not torch.isfinite(routing_weights).all():
            print(f"  [NAN SOURCE] Router produced NaN/Inf routing_weights!")
            raise RuntimeError("NaN/Inf from router")
        
        top_k_indices = torch.topk(routing_weights, min(self.router.top_k, self.num_experts), dim=-1).indices
        
        # 5. Apply routing to expert outputs
        # routing_weights: [B, num_experts]
        routed_outputs = []
        for i, exp_type in enumerate(self.expert_types):
            weight = routing_weights[:, i:i+1]  # [B, 1]
            routed_outputs.append(expert_outputs[exp_type] * weight)
        
        # Sum routed outputs
        combined = sum(routed_outputs)  # [B, hidden_dim]
        
        # 5. Adaptive multimodal fusion
        # Prepare expert outputs for fusion
        fusion_inputs = {exp_type: expert_outputs[exp_type] for exp_type in self.expert_types}
        fused, fusion_weights = self.fusion(fusion_inputs, return_weights=True)
        
        # NaN detection after fusion
        if not torch.isfinite(fused).all():
            print(f"  [NAN SOURCE] Fusion produced NaN/Inf output!")
            raise RuntimeError("NaN/Inf from fusion")
        
        # 6. Classification
        MMlogit = self.classifier(fused)
        
        # NaN detection after classifier
        if not torch.isfinite(MMlogit).all():
            print(f"  [NAN SOURCE] Classifier produced NaN/Inf logits!")
            raise RuntimeError("NaN/Inf from classifier")
        
        # Prepare outputs
        outputs = {
            "MMlogit": MMlogit,
            "fused_embedding": fused,
            "expert_outputs": expert_outputs,
            "routing_weights": routing_weights,
            "top_k_indices": top_k_indices,
            "fusion_weights": fusion_weights,
            "expert_uncertainties": expert_uncertainties,
        }
        
        # Compute losses if labels provided
        if labels is not None:
            # Culprit labels -> binary (MUAD convention): culprit 0 = normal, culprit >= 1 = anomalous
            binary_labels = (labels >= 1).long()
            
            # Main classification loss
            criterion = nn.CrossEntropyLoss()
            cls_loss = criterion(MMlogit, binary_labels)
            
            # KL losses from uncertainty blocks
            kl_losses = []
            for exp_type in self.expert_types:
                if self.use_uncertainty and exp_type in expert_uncertainties:
                    kl_losses.append(expert_uncertainties[exp_type])
            
            total_kl = torch.stack(kl_losses).mean() if kl_losses else 0
            
            # Routing load balancing loss
            routing_lb_loss = routing_aux_loss if routing_aux_loss is not None else 0
            # Ensure routing_lb_loss is scalar
            if isinstance(routing_lb_loss, dict):
                routing_lb_loss = routing_lb_loss.get('aux_loss', 0)
            if isinstance(routing_lb_loss, torch.Tensor) and routing_lb_loss.dim() > 0:
                routing_lb_loss = routing_lb_loss.mean()
            
            # Total loss
            total_loss = cls_loss + 0.1 * total_kl + 0.01 * routing_lb_loss
            
            outputs.update({
                "loss": total_loss,
                "cls_loss": cls_loss,
                "kl_loss": total_kl,
                "routing_lb_loss": routing_lb_loss,
            })
        
        # Predictions
        with torch.no_grad():
            y_pred = MMlogit.argmax(dim=-1)
            outputs["y_pred"] = y_pred
        
        if not return_expert_outputs:
            outputs.pop("expert_outputs", None)
        if not return_routing:
            outputs.pop("routing_weights", None)
            outputs.pop("top_k_indices", None)
            outputs.pop("fusion_weights", None)
        
        return outputs


def create_graph_aware_moe(
    config: Dict[str, Any],
    muad_baseline: Optional[nn.Module] = None,
) -> GraphAwareMoE:
    """
    Factory function to create GraphAwareMoE from config.
    
    Args:
        config: Configuration dictionary
        muad_baseline: Optional MUAD baseline model to reuse encoders
        
    Returns:
        GraphAwareMoE model
    """
    # Extract MUAD encoders if baseline provided
    muad_encoders = {}
    if muad_baseline is not None:
        muad_encoders = {
            "muad_metric_encoder": muad_baseline.metric_encoder,
            "muad_trace_encoder": muad_baseline.trace_encoder,
            "muad_log_encoder": muad_baseline.log_encoder,
            "muad_graph_model": muad_baseline.graph_model,
        }
    
    return GraphAwareMoE(
        **config,
        **muad_encoders,
    )


# Default configuration for the proposed model
DEFAULT_CONFIG = {
    "num_nodes": 12,
    "metric_dim": 7,
    "trace_dim": 1,
    "log_dim": 15,
    "num_classes": 2,
    "hidden_dim": 64,
    "expert_hidden_dim": 128,
    "fusion_dim": 128,
    "router_hidden_dim": 128,
    "num_experts": 4,
    "expert_types": ["temporal", "semantic", "dependency", "cross_modal"],
    "router_type": "uncertainty_guided",
    "top_k": 2,
    "use_graph_routing": True,
    "use_uncertainty_routing": True,
    "fusion_type": "confidence",
    "use_uncertainty": True,
    "uncertainty_type": "variational",
    "dropout": 0.1,
    "dependency_expert_type": "gat",
}