"""
Model registry: one place that defines how every compared model is built and called.

Each entry maps a name to (builder(metadata) -> nn.Module, forward_fn). Forward functions follow
common.ForwardFn: (model, graph, raw_culprit_labels) -> (logits [B, 2], loss).

Names:
  muad        Track A - MUAD MainModel, trained on its own objective
  ug_moe      Track B - GraphAwareMoE with DEFAULT_CONFIG (full proposed model)
  plain_moe   GraphAwareMoE without graph routing, uncertainty routing, uncertainty blocks;
              weighted_concat fusion
  gat         GAT baseline (tracks/baselines/gat_baseline.py)
  graph_moe   Graph-MoE baseline (tracks/baselines/graph_moe.py)
  abl:<name>  ug_moe with the overrides in ABLATIONS[<name>]
"""
import json
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch

import common  # noqa: F401  (sets up sys.path)
from common import default_forward, seed_everything, summarize, to_binary, train_and_select
from tracks.baselines import GATBaseline, GraphMoE
from tracks.proposed_model import DEFAULT_CONFIG, GraphAwareMoE
from vendor.muad.models.main_model import MainModel

# NOTE (PROJECT_STATUS.md section 3.7): with the current model these overrides do NOT isolate the
# components their names suggest (e.g. graph routing is never active in the full model, so
# 'no_graph' only swaps the router class). They are kept as-is until Phase 2 rewrites the model;
# the ablation set will then be redefined to match implementation guide section 8B.
ABLATIONS: Dict[str, Dict] = {
    "no_graph": {"use_graph_routing": False, "router_type": "base"},
    "no_uncertainty": {"use_uncertainty": False},
    "no_fusion": {"fusion_type": "weighted_concat"},
    "single_expert": {"expert_types": ["temporal"]},
    "plain_moe": {
        "router_type": "base",
        "use_graph_routing": False,
        "use_uncertainty_routing": False,
        "fusion_type": "weighted_concat",
    },
    "graph_transformer": {"dependency_expert_type": "graph_transformer"},
}

PLAIN_MOE_OVERRIDES = {
    "router_type": "base",
    "use_graph_routing": False,
    "use_uncertainty_routing": False,
    "fusion_type": "weighted_concat",
    "use_uncertainty": False,
}


def _ug_moe(metadata, overrides=None):
    config = DEFAULT_CONFIG.copy()
    config.update(
        log_dim=metadata["event_num"], metric_dim=metadata["metric_num"], num_nodes=metadata["node_num"]
    )
    config.update(overrides or {})
    if "expert_types" in (overrides or {}):
        config["num_experts"] = len(config["expert_types"])
    return GraphAwareMoE(**config)


def _binary_logits_forward(model, graph, labels):
    """For baselines that expect binary labels and return {'logits', 'loss'}."""
    out = model(graph, to_binary(labels))
    return out["logits"], out.get("loss")


def _dims(metadata):
    return dict(metric_dim=metadata["metric_num"], trace_dim=1, log_dim=metadata["event_num"])


MODELS: Dict[str, Tuple[Callable, Callable]] = {
    "muad": (lambda md: MainModel(torch.device("cpu"), hidden_dim=[64, 64], event_num=md["event_num"]),
             default_forward),
    "ug_moe": (lambda md: _ug_moe(md), default_forward),
    "plain_moe": (lambda md: _ug_moe(md, PLAIN_MOE_OVERRIDES), default_forward),
    "gat": (lambda md: GATBaseline(**_dims(md)), _binary_logits_forward),
    "graph_moe": (lambda md: GraphMoE(**_dims(md)), _binary_logits_forward),
}


def get_model(name: str):
    """Return (builder, forward_fn) for a registry name, including 'abl:<ablation>'."""
    if name.startswith("abl:"):
        abl = name.split(":", 1)[1]
        if abl not in ABLATIONS:
            raise KeyError(f"Unknown ablation '{abl}'. Options: {sorted(ABLATIONS)}")
        return (lambda md: _ug_moe(md, ABLATIONS[abl])), default_forward
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Options: {sorted(MODELS)} or abl:<{'|'.join(ABLATIONS)}>")
    return MODELS[name]


def run_suite(names: List[str], seeds: List[int], loaders: Dict, device, args, out_dir: Path) -> Dict:
    """Train/select/test every model x seed under the shared protocol; write per-run JSON + checkpoint."""
    results, errors = {n: [] for n in names}, {n: [] for n in names}
    for seed in seeds:
        for name in names:
            builder, fwd = get_model(name)
            tag = name.replace(":", "_")
            print(f"\n--- {name} | seed {seed} ---")
            try:
                seed_everything(seed)
                model = builder(loaders["metadata"])
                res = train_and_select(
                    model, loaders, device,
                    epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                    patience=args.patience if args.patience > 0 else None,
                    ckpt_path=out_dir / f"{tag}_seed{seed}_best.pth",
                    forward_fn=fwd, log_prefix="  ",
                )
                res.update(model=name, seed=seed, n_params=sum(p.numel() for p in model.parameters()))
                results[name].append(res)
                with open(out_dir / f"{tag}_seed{seed}.json", "w") as f:
                    json.dump(res, f, indent=2)
            except Exception as e:
                traceback.print_exc()
                errors[name].append({"seed": seed, "error": repr(e)})
    return {
        n: {
            "test": summarize(results[n], "test"),
            "val": summarize(results[n], "val"),
            "best_epochs": [r["best_epoch"] for r in results[n]],
            "n_params": results[n][0]["n_params"] if results[n] else None,
            "errors": errors[n],
        }
        for n in names
    }


def print_table(suite: Dict, trivial: Dict) -> None:
    cols = ["f1", "precision", "recall", "f1_normal", "macro_f1", "pr_auc", "roc_auc"]
    print(f"\n{'model':<22}" + "".join(f"{c:>11}" for c in cols))
    print(f"{'trivial(all anomalous)':<22}" + "".join(f"{trivial[c]:>11.4f}" for c in cols))
    for name, s in suite.items():
        if not s["test"]:
            print(f"{name:<22}  FAILED: {s['errors']}")
            continue
        print(f"{name:<22}" + "".join(f"{s['test'][c]['mean']:>11.4f}" for c in cols))


def add_common_args(parser):
    from common import DEFAULT_DATA_DIR, PROJECT_ROOT
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated seeds (default: single seed 42)")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "experiments" / "runs"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=15, help="Early-stopping patience on val (<=0 disables)")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda")
    return parser
