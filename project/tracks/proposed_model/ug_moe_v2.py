"""
UG-MoE v2: Uncertainty-Guided Graph-Aware Mixture-of-Experts, implemented to follow the
implementation guide's pseudocode (section 11) step by step.

    for each modality v in {metric, trace, log}:
        h_v                    = Encoder_v(x_v)                               node-level, per time step
        Expert_k(h_v, G)       -> mean mu_{k,v}, log-variance s_{k,v}          k = 1..K shared experts
        g_v                    = Router_v(pool(h_v), graph_ctx(h_v, G), u_v)   u_v[k] = mean(s_{k,v})
        mu_v, var_v            = moments of the mixture  sum_k g_v[k] N(mu_{k,v}, exp(s_{k,v}))
        z_v                    = mu_v + sqrt(var_v) * eps    (training)  |  mu_v  (eval)
    w                          = softmax_v( FusionNet([mu_v, mean log var_v]) )
    z                          = sum_v w_v z_v
    logits                     = Classifier(z)
    loss                       = CE + lambda_kl * mean_v KL(N(mu_v, var_v) || N(0, I))  [+ lb_weight * load balance]

Experts (shared across modalities; a learned modality embedding tells them which one they see):
    temporal    GRU over the time axis of each service
    semantic    self-attention across services (content-based, no graph)
    dependency  GAT over the service dependency graph
    cross_modal attention from this modality's service vector to all three modalities of the same service

Every switch maps to one ablation of the guide's section 8B:
    use_graph=False              w/o graph            (no dependency expert, no graph context in the router)
    use_moe=False                w/o MoE              (single temporal expert, no router)
    dynamic_routing=False        w/o dynamic routing  (uniform expert weights)
    uncertainty_routing=False    w/o uncertainty routing (router does not see expert uncertainty)
    probabilistic=False          w/o probabilistic representation (z_v = mu_v, no variance, no KL)
    adaptive_fusion=False        w/o adaptive fusion  (w_v = 1/3)
"""
from typing import Dict, Optional

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F

MODALITIES = ("metric", "trace", "log")
EXPERTS = ("temporal", "semantic", "dependency", "cross_modal")
LOGVAR_MIN, LOGVAR_MAX = -8.0, 8.0


class TemporalExpert(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gru = nn.GRU(d, d, batch_first=True)

    def forward(self, h_seq, **_):                        # h_seq [BN, T, D]
        _, last = self.gru(h_seq)
        return last[-1]                                   # [BN, D]


class SemanticExpert(nn.Module):
    def __init__(self, d, heads=4, dropout=0.1):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=dropout,
                                                batch_first=True, norm_first=True)

    def forward(self, h_node, B, N, **_):                 # h_node [BN, D]
        return self.layer(h_node.view(B, N, -1)).reshape(B * N, -1)


class DependencyExpert(nn.Module):
    def __init__(self, d, heads=4, dropout=0.1):
        super().__init__()
        self.gat = dglnn.GATv2Conv(d, d // heads, heads, feat_drop=dropout, attn_drop=dropout,
                                   residual=True, allow_zero_in_degree=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, h_node, graph, **_):
        return self.norm(F.elu(self.gat(graph, h_node).flatten(1)))


class CrossModalExpert(nn.Module):
    def __init__(self, d, heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, h_node, all_nodes, **_):            # all_nodes [BN, 3, D]
        out, _ = self.attn(h_node.unsqueeze(1), all_nodes, all_nodes)
        return self.norm(h_node + out.squeeze(1))


class UGMoEv2(nn.Module):
    def __init__(
        self,
        metric_dim: int = 7,
        trace_dim: int = 1,
        log_dim: int = 15,
        num_nodes: int = 12,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.1,
        use_graph: bool = True,
        use_moe: bool = True,
        dynamic_routing: bool = True,
        uncertainty_routing: bool = True,
        probabilistic: bool = True,
        adaptive_fusion: bool = True,
        routing_temperature: float = 1.0,
        lambda_kl: float = 0.01,
        lb_weight: float = 0.0,
        experts: Optional[str] = None,
        **_unused,
    ):
        super().__init__()
        if uncertainty_routing and not probabilistic:
            raise ValueError("uncertainty_routing needs probabilistic experts (they provide the uncertainty)")
        D = hidden_dim
        self.D, self.num_nodes = D, num_nodes
        self.use_graph, self.use_moe = use_graph, use_moe
        self.dynamic_routing = dynamic_routing and use_moe
        self.uncertainty_routing = uncertainty_routing and use_moe
        self.probabilistic, self.adaptive_fusion = probabilistic, adaptive_fusion
        self.temperature, self.lambda_kl, self.lb_weight = routing_temperature, lambda_kl, lb_weight

        # 1. modality encoders (per service, per time step)
        self.enc = nn.ModuleDict({"metric": nn.Linear(metric_dim, D), "trace": nn.Linear(trace_dim, D),
                                  "log": nn.Linear(log_dim, D)})
        self.mod_emb = nn.Parameter(torch.zeros(len(MODALITIES), D))
        nn.init.normal_(self.mod_emb, std=0.02)

        # 2. experts (shared across modalities)
        names = list(EXPERTS) if use_moe else ["temporal"]
        if experts is not None and use_moe:      # plan 8H "number of experts": '-'-joined subset of EXPERTS
            names = [n for n in EXPERTS if n in experts.split("-")]
            if not names:
                raise ValueError(f"experts='{experts}' selects none of {EXPERTS}")
        if not use_graph:
            names = [n for n in names if n != "dependency"]
        self.expert_names = names
        builders = {"temporal": lambda: TemporalExpert(D), "semantic": lambda: SemanticExpert(D, dropout=dropout),
                    "dependency": lambda: DependencyExpert(D, dropout=dropout),
                    "cross_modal": lambda: CrossModalExpert(D, dropout=dropout)}
        self.experts = nn.ModuleDict({n: builders[n]() for n in names})
        K = len(names)
        self.mu_head = nn.ModuleDict({n: nn.Linear(D, D) for n in names})
        if probabilistic:
            self.logvar_head = nn.ModuleDict({n: nn.Linear(D, D) for n in names})

        # 3. routers (one per modality)
        if self.dynamic_routing:
            if use_graph:
                self.graph_ctx = dglnn.GraphConv(D, D, allow_zero_in_degree=True)
            in_dim = D + (D if use_graph else 0) + (K if self.uncertainty_routing else 0)
            self.router = nn.ModuleDict({m: nn.Sequential(nn.Linear(in_dim, D), nn.GELU(), nn.Dropout(dropout),
                                                          nn.Linear(D, K)) for m in MODALITIES})

        # 5. fusion over modalities
        if adaptive_fusion:
            self.fusion = nn.Sequential(nn.Linear(D + (1 if probabilistic else 0), D), nn.GELU(), nn.Linear(D, 1))

        # 6. classifier
        self.classifier = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Dropout(dropout), nn.Linear(D, num_classes))
        self.node_pool = nn.Linear(2 * D, D)

    # ------------------------------------------------------------------ helpers
    def _pool(self, h_node, B):                           # [BN, D] -> [B, D] (mean + max over services)
        h = h_node.view(B, -1, self.D)
        return self.node_pool(torch.cat([h.mean(1), h.max(1).values], -1))

    def _encode(self, graph):
        logs = graph.ndata["logs"]
        # Older datasets store one count vector per node. GAIA also supplies per-minute counts.
        x = {"metric": graph.ndata["metrics"], "trace": graph.ndata["traces"],
             "log": logs.unsqueeze(1) if logs.ndim == 2 else logs}
        seq = {m: self.enc[m](x[m]) + self.mod_emb[i] for i, m in enumerate(MODALITIES)}  # [BN, T, D]
        node = {m: seq[m].mean(1) for m in MODALITIES}     # [BN, D]
        return seq, node

    # ------------------------------------------------------------------ forward
    def forward(self, graph, labels: Optional[torch.Tensor] = None, return_details: bool = False) -> Dict:
        B = graph.batch_size
        N = graph.num_nodes() // B
        seq, node = self._encode(graph)
        all_nodes = torch.stack([node[m] for m in MODALITIES], 1)      # [BN, 3, D]

        z, mu_v, lv_v, routing, unc_routing, kl_terms, lb_terms = {}, {}, {}, {}, {}, [], []
        for mi, m in enumerate(MODALITIES):
            mus, lvs = [], []
            for name in self.expert_names:
                e = self.experts[name](h_seq=seq[m], h_node=node[m], graph=graph, all_nodes=all_nodes, B=B, N=N)
                pooled = self._pool(e, B)                                # [B, D]
                mus.append(self.mu_head[name](pooled))
                if self.probabilistic:
                    lvs.append(self.logvar_head[name](pooled).clamp(LOGVAR_MIN, LOGVAR_MAX))
            mu_k = torch.stack(mus, 1)                                   # [B, K, D]
            K = mu_k.shape[1]

            # routing weights g_v [B, K]
            if self.dynamic_routing:
                parts = [self._pool(node[m], B)]
                if self.use_graph:
                    parts.append(self._pool(F.elu(self.graph_ctx(graph, node[m])), B))
                if self.uncertainty_routing:
                    u = torch.stack(lvs, 1).mean(-1)                     # [B, K] expected log-variance per expert
                    parts.append(u)
                    unc_routing[m] = u
                g = F.softmax(self.router[m](torch.cat(parts, -1)) / self.temperature, -1)
                frac = g.mean(0)
                lb_terms.append(K * (frac ** 2).sum() - 1.0)
            else:
                g = torch.full((B, K), 1.0 / K, device=mu_k.device)
            routing[m] = g

            # mixture moments -> probabilistic modality representation
            gw = g.unsqueeze(-1)
            mu = (gw * mu_k).sum(1)                                     # [B, D]
            if self.probabilistic:
                var_k = torch.stack(lvs, 1).exp()
                var = (gw * (var_k + mu_k ** 2)).sum(1) - mu ** 2
                var = var.clamp_min(1e-6)
                logvar = var.log()
                z[m] = mu + var.sqrt() * torch.randn_like(mu) if self.training else mu
                kl_terms.append(0.5 * (mu ** 2 + var - 1.0 - logvar).mean())
                lv_v[m] = logvar
            else:
                z[m] = mu
            mu_v[m] = mu

        # fusion over modalities
        if self.adaptive_fusion:
            feats = [torch.cat([mu_v[m], lv_v[m].mean(-1, keepdim=True)], -1) if self.probabilistic else mu_v[m]
                     for m in MODALITIES]
            w = F.softmax(torch.cat([self.fusion(f) for f in feats], -1), -1)   # [B, 3]
        else:
            w = torch.full((B, len(MODALITIES)), 1.0 / len(MODALITIES), device=z["metric"].device)
        fused = sum(w[:, i:i + 1] * z[m] for i, m in enumerate(MODALITIES))
        logits = self.classifier(fused)

        out = {"MMlogit": logits, "y_pred": logits.argmax(-1)}
        if labels is not None:
            y = (labels >= 1).long()
            cls = F.cross_entropy(logits, y)
            kl = torch.stack(kl_terms).mean() if kl_terms else logits.new_zeros(())
            lb = torch.stack(lb_terms).mean() if lb_terms else logits.new_zeros(())
            out.update(loss=cls + self.lambda_kl * kl + self.lb_weight * lb, cls_loss=cls, kl_loss=kl, lb_loss=lb)
        if return_details:
            out.update(routing_weights={m: routing[m].detach() for m in MODALITIES},
                       fusion_weights=w.detach(),
                       modality_logvar={m: lv_v[m].mean(-1).detach() for m in lv_v},
                       expert_uncertainty={m: unc_routing[m].detach() for m in unc_routing},
                       expert_names=list(self.expert_names))
        return out


# Plan section 12 incremental order: each step adds one component to the previous one.
INCREMENTAL_STEPS = {
    "plain":        dict(use_graph=False, dynamic_routing=True, uncertainty_routing=False, probabilistic=False, adaptive_fusion=False),
    "graph":        dict(use_graph=True,  dynamic_routing=True, uncertainty_routing=False, probabilistic=False, adaptive_fusion=False),
    "prob":         dict(use_graph=True,  dynamic_routing=True, uncertainty_routing=False, probabilistic=True,  adaptive_fusion=False),
    "ug_routing":   dict(use_graph=True,  dynamic_routing=True, uncertainty_routing=True,  probabilistic=True,  adaptive_fusion=False),
    "full":         dict(use_graph=True,  dynamic_routing=True, uncertainty_routing=True,  probabilistic=True,  adaptive_fusion=True),
}

# Plan section 8B component ablations, each removing exactly one thing from the full model.
ABLATIONS_V2 = {
    "no_graph":               dict(use_graph=False),
    "no_moe":                 dict(use_moe=False, uncertainty_routing=False),
    "no_dynamic_routing":     dict(dynamic_routing=False),
    "no_uncertainty_routing": dict(uncertainty_routing=False),
    "no_probabilistic":       dict(probabilistic=False, uncertainty_routing=False),
    "no_adaptive_fusion":     dict(adaptive_fusion=False),
}
