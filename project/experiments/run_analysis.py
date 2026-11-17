#!/usr/bin/env python3
"""
Routing and uncertainty analysis of a trained UG-MoE v2 checkpoint (implementation guide section 8F + 8G).
No training: the checkpoint is loaded and the test split is evaluated under a set of conditions.

F. Expert routing
   - mean routing weight per (modality, class, expert): one heatmap per modality
   - routing entropy per modality (normalized by log K, 0 = one expert, 1 = uniform), per class
G. Uncertainty and fusion
   - per condition (clean, one modality dropped, random masking, noise): mean fusion weight and mean
     predicted log-variance of each modality, plus F1. If the uncertainty works as intended, a
     perturbed modality's log-variance goes up and its fusion weight goes down.
   - clean test: uncertainty of correct vs. wrong predictions

Perturbations use run_robustness.py's functions, with fixed seeds per condition.

Usage (from project/):
    python experiments/run_analysis.py --device cuda --model ug_moe2:full \
        --data-dir datasets/eadro/processed/SN_w10_s1 --checkpoint path/to/ug_moe2_full_seed42_best.pth

Writes analysis.json, routing_heatmap.png, fusion_weights.png, modality_uncertainty.png.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from common import compute_metrics, configure_torch, load_splits, seed_everything, to_binary
from registry import add_common_args, get_model, model_tag
from run_robustness import channel_std, drop_modalities, gaussian_noise, random_mask

MODS = ("metric", "trace", "log")                 # model-side names (ug_moe_v2.MODALITIES)
DATA_MODS = ("metrics", "traces", "logs")         # graph.ndata names
COLORS = {"metric": "#2a78d6", "trace": "#eb6834", "log": "#1baf7a"}   # validated categorical slots 1-3
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # sequential ramp


def class_names(metadata):
    if "label_map" in metadata:
        return {int(k): v for k, v in metadata["label_map"].items()}
    return {0: "normal", 1: "cpu_load", 4: "network_delay", 5: "network_loss"}   # recovered_artifact


def conditions(stds):
    """(name, perturb or None). Seeds are fixed per condition, like run_robustness."""
    out = [("clean", None)]
    out += [(f"drop_{m}", drop_modalities([m])) for m in DATA_MODS]
    out += [(f"mask_{p}", random_mask(p, 2000 + i)) for i, p in enumerate((0.2, 0.4))]
    out += [(f"noise_{m}_1.0", gaussian_noise([m], 1.0, stds, 3000 + i)) for i, m in enumerate(DATA_MODS)]
    out += [("noise_all_1.0", gaussian_noise(list(DATA_MODS), 1.0, stds, 3100))]
    return out


@torch.no_grad()
def collect(model, loader, device, perturb):
    model.eval()
    rec = {"culprit": [], "pred": [], "p": [], "fusion": [], "routing": {m: [] for m in MODS},
           "logvar": {m: [] for m in MODS}}
    for graph, labels in loader:
        graph, labels = graph.to(device), labels.to(device)
        if perturb is not None:
            perturb(graph)
        out = model(graph, labels, return_details=True)
        logits = out["MMlogit"]
        rec["culprit"] += labels.long().cpu().tolist()
        rec["pred"] += logits.argmax(-1).cpu().tolist()
        rec["p"] += torch.softmax(logits, -1)[:, 1].cpu().tolist()
        rec["fusion"].append(out["fusion_weights"].cpu())
        for m in MODS:
            rec["routing"][m].append(out["routing_weights"][m].cpu())
            if m in out["modality_logvar"]:
                rec["logvar"][m].append(out["modality_logvar"][m].cpu())
    rec["fusion"] = torch.cat(rec["fusion"]).numpy()
    rec["routing"] = {m: torch.cat(v).numpy() for m, v in rec["routing"].items()}
    rec["logvar"] = {m: torch.cat(v).numpy() for m, v in rec["logvar"].items() if v}
    rec["culprit"], rec["pred"] = np.asarray(rec["culprit"]), np.asarray(rec["pred"])
    return rec, out["expert_names"]


def entropy(g):                                   # g [n, K] -> normalized entropy per sample
    K = g.shape[1]
    if K < 2:
        return np.zeros(len(g))
    return -(g * np.log(np.clip(g, 1e-12, 1))).sum(1) / np.log(K)


def routing_stats(rec, experts, names):
    out = {}
    for m in MODS:
        g = rec["routing"][m]
        per_class = {}
        for c in sorted(set(rec["culprit"].tolist())):
            sel = rec["culprit"] == c
            per_class[names.get(c, str(c))] = {
                "weights": dict(zip(experts, g[sel].mean(0).round(4).tolist())),
                "entropy": float(entropy(g[sel]).mean()), "n": int(sel.sum())}
        out[m] = {"overall_weights": dict(zip(experts, g.mean(0).round(4).tolist())),
                  "entropy": float(entropy(g).mean()), "per_class": per_class}
    return out


def condition_stats(rec):
    y = to_binary(torch.as_tensor(rec["culprit"])).numpy()
    met = compute_metrics(y.tolist(), rec["pred"].tolist(), rec["p"])
    return {"f1": met["f1"], "macro_f1": met["macro_f1"],
            "fusion_weight": {m: float(rec["fusion"][:, i].mean()) for i, m in enumerate(MODS)},
            "logvar": {m: float(v.mean()) for m, v in rec["logvar"].items()}}


def error_uncertainty(rec):
    """Clean test: mean modality log-variance (fusion-weighted) for correct vs. wrong predictions."""
    if not rec["logvar"]:
        return None
    lv = sum(rec["fusion"][:, i] * rec["logvar"][m] for i, m in enumerate(MODS))
    y = (rec["culprit"] >= 1).astype(int)
    ok = rec["pred"] == y
    return {"correct": float(lv[ok].mean()) if ok.any() else None,
            "wrong": float(lv[~ok].mean()) if (~ok).any() else None,
            "n_correct": int(ok.sum()), "n_wrong": int((~ok).sum())}


# ------------------------------------------------------------------ plots
def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)


def plot_routing(stats, experts, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    classes = list(stats[MODS[0]]["per_class"])
    fig, axes = plt.subplots(1, 3, figsize=(4.2 * 3, 0.55 * len(classes) + 1.8), facecolor=SURFACE)
    for ax, m in zip(axes, MODS):
        M = np.array([[stats[m]["per_class"][c]["weights"][e] for e in experts] for c in classes])
        ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if M[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(experts)), [e.replace("_", "-") for e in experts], rotation=20)
        ax.set_yticks(range(len(classes)), classes)
        ent = [stats[m]["per_class"][c]["entropy"] for c in classes]
        ax.set_title(f"{m}  (routing entropy {stats[m]['entropy']:.2f})", color=INK, fontsize=10)
        for i, e in enumerate(ent):
            ax.text(len(experts) - 0.4, i, f"H={e:.2f}", va="center", fontsize=8, color=INK2)
        ax.tick_params(colors=INK2, labelsize=9, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(title, color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_conditions(cond, key, path, title, xlabel, stacked):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(cond)
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(names) + 1.4), facecolor=SURFACE)
    _style(ax)
    y = np.arange(len(names))[::-1]
    if stacked:                                   # fusion weights sum to 1: 100% stacked bars
        left = np.zeros(len(names))
        for m in MODS:
            v = np.array([cond[n][key][m] for n in names])
            ax.barh(y, v, left=left, height=0.6, color=COLORS[m], edgecolor=SURFACE, linewidth=2, label=m)
            for yi, li, vi in zip(y, left, v):
                if vi >= 0.08:
                    ax.text(li + vi / 2, yi, f"{vi:.2f}", ha="center", va="center", fontsize=8, color="white")
            left += v
        ax.set_xlim(0, 1)
    else:                                         # grouped bars: one per modality
        h = 0.26
        for k, m in enumerate(MODS):
            if m not in cond[names[0]][key]:
                continue
            v = np.array([cond[n][key][m] for n in names])
            ax.barh(y + (1 - k) * h, v, height=h - 0.03, color=COLORS[m], label=m)
            for yi, vi in zip(y + (1 - k) * h, v):
                ax.text(vi, yi, f" {vi:.2f}", va="center", ha="left" if vi >= 0 else "right",
                        fontsize=7, color=INK2)
        ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_yticks(y, [f"{n}  (F1 {cond[n]['f1']:.3f})" for n in names])
    ax.set_xlabel(xlabel, color=INK2)
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, pad=24)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = add_common_args(argparse.ArgumentParser(description="Routing + uncertainty analysis (8F, 8G)"))
    parser.add_argument("--model", type=str, default="ug_moe2:full", help="A ug_moe2 registry name")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    configure_torch()
    device = torch.device(args.device)
    seed = int(args.seeds.split(",")[0])
    tag = model_tag(args.model)
    out_dir = Path(args.output_dir) / f"analysis_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    loaders = load_splits(Path(args.data_dir), args.batch_size, args.val_ratio, args.split_seed,
                          split_out=out_dir / "split.json", normalize=args.normalize, split=args.split,
                          test_run=args.test_run, val_run=args.val_run)
    builder, _ = get_model(args.model)
    seed_everything(seed)
    model = builder(loaders["metadata"]).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded checkpoint {args.checkpoint}")
    names = class_names(loaders["metadata"])
    stds = channel_std(loaders["train"])

    cond, routing, err, experts = {}, None, None, None
    for name, perturb in conditions(stds):
        rec, experts = collect(model, loaders["test"], device, perturb)
        cond[name] = condition_stats(rec)
        if name == "clean":
            routing, err = routing_stats(rec, experts, names), error_uncertainty(rec)
        fw = " ".join(f"{m}={cond[name]['fusion_weight'][m]:.3f}" for m in MODS)
        lv = " ".join(f"{m}={v:+.2f}" for m, v in cond[name]["logvar"].items())
        print(f"  {name:<16} F1={cond[name]['f1']:.4f} | fusion {fw} | logvar {lv}")

    print("\nF - routing (clean test), mean expert weight per modality, entropy (0 = one expert, 1 = uniform)")
    for m in MODS:
        w = " ".join(f"{e}={v:.3f}" for e, v in routing[m]["overall_weights"].items())
        print(f"  {m:<7} H={routing[m]['entropy']:.3f} | {w}")
        for c, s in routing[m]["per_class"].items():
            print(f"      {c:<14} H={s['entropy']:.3f} | " + " ".join(f"{v:.3f}" for v in s["weights"].values()))
    if err:
        fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
        print(f"\nG - fusion-weighted log-variance on clean test: correct={fmt(err['correct'])} "
              f"(n={err['n_correct']}) wrong={fmt(err['wrong'])} (n={err['n_wrong']})")

    summary = {"experiment": "routing_uncertainty_analysis", "model": args.model, "checkpoint": args.checkpoint,
               "experts": experts, "class_names": names, "routing_clean": routing, "conditions": cond,
               "uncertainty_correct_vs_wrong": err, "config": vars(args)}
    with open(out_dir / "analysis.json", "w") as f:
        json.dump(summary, f, indent=2)
    system = loaders["metadata"].get("system", "data")
    plot_routing(routing, experts, out_dir / "routing_heatmap.png",
                 f"Expert routing weights by class, {args.model}, {system} test")
    plot_conditions(cond, "fusion_weight", out_dir / "fusion_weights.png",
                    f"Modality fusion weights under perturbation, {system} test", "mean fusion weight", True)
    if cond["clean"]["logvar"]:
        plot_conditions(cond, "logvar", out_dir / "modality_uncertainty.png",
                        f"Predicted modality uncertainty under perturbation, {system} test",
                        "mean log-variance (higher = less certain)", False)
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
