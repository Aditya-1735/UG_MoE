#!/usr/bin/env python3
"""
Regenerate every paper table and figure from saved run outputs (implementation guide section 10:
"a reproducible evaluation script that regenerates all major tables and plots"). No training, no GPU.

Reads the run directories listed in MANIFEST (paths relative to --runs-dir, glob patterns; the newest
match wins) and writes to <runs-dir>/report_<timestamp>/:
    tables.md                       every table (mean ± std when a run has several seeds)
    robustness_masking_<sys>.png    macro-F1 vs share of missing observations
    robustness_noise_<sys>.png      macro-F1 vs noise level (all modalities)
    sensitivity_sn.png              one panel per hyperparameter (plan 8H)
    fusion_<sys>.png                fusion weights per condition, all modalities + zoom on trace / log
    uncertainty_delta_<sys>.png     change of predicted log-variance vs clean, per condition
    routing_<sys>.png               expert-routing heatmaps (plan 8F)
Missing runs are skipped with a warning, so the script can be run at any stage.

Usage (from project/):
    python experiments/make_report.py                    # default --runs-dir experiments/runs
When the final multi-seed runs exist, point MANIFEST at them; nothing else changes.
"""
import argparse
import glob
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------------------- manifest
# Each entry: glob(s) relative to --runs-dir. For baselines-style runs the directory holds summary.json
# with {"models": {name: {"test": {metric: {"mean", "std", "values"}}, "n_params"}}}.
MANIFEST = {
    # single-seed exploration rounds first, final multi-seed runs (final_*) last: same (model, seed) -> final wins
    "main_sn": ["round5_sn/baselines_*", "round6_v2_sn/baselines_*", "final_sn_main*/baselines_*",
                "final_sn_steps*/baselines_*"],
    "main_tt": ["round7_tt/baselines_*", "round7_v2_tt/baselines_*", "final_tt_main*/baselines_*",
                "final_tt_steps*/baselines_*"],
    "ablation_sn": ["round9_abl2_sn/baselines_*", "final_sn_abl*/baselines_*"],
    "ablation_tt": ["round15b_abl2_tt/baselines_*", "final_tt_abl*/baselines_*"],
    "modality_sn": ["round12_modality_sn/baselines_*"],
    "sens_sn": ["round16a_sens_sn/baselines_*", "round16b_sens_sn/baselines_*"],
    # single-model runs whose varied setting is not in the model name: (glob, parameter, value)
    "sens_sn_runs": [("round16c_sens_sn_lr1e-4/baselines_*", "lr", 1e-4),
                     ("round16c_sens_sn_lr1e-3/baselines_*", "lr", 1e-3),
                     ("round16c_sens_sn_w5/baselines_*", "window", 5),
                     ("round16c_sens_sn_w20/baselines_*", "window", 20)],
    # robustness: [single-seed pattern, final pattern]; the final one (all seeds averaged) wins if it exists
    "robust_sn": {"ug_moe2:full": ["round8_rob_sn_v2/robustness_*", "final_rob_sn_full_s*/robustness_*"],
                  "muad": ["round8_rob_sn_muad/robustness_*", "final_rob_sn_muad_s*/robustness_*"],
                  "ug_moe2:plain": ["round8_rob_sn_plain/robustness_*", "final_rob_sn_plain_s*/robustness_*"],
                  "ug_moe2:graph": "round8_rob_sn_graph/robustness_*",
                  "ug_moe2:prob": "round8_rob_sn_prob/robustness_*",
                  "ug_moe2:ug_routing": "round8_rob_sn_ug_routing/robustness_*",
                  "abl2:no_probabilistic": ["round15a_rob_sn_noprob/robustness_*",
                                            "final_rob_sn_noprob_s*/robustness_*"]},
    "robust_tt": {"ug_moe2:full": ["round14_rob_tt_v2/robustness_*", "final_rob_tt_full_s*/robustness_*"],
                  "muad": ["round11_rob_tt_muad/robustness_*", "final_rob_tt_muad_s*/robustness_*"],
                  "gat": ["round11_rob_tt_gat/robustness_*", "final_rob_tt_gat_s*/robustness_*"]},
    "analysis": {"sn": "round13_analysis_sn/analysis_*", "tt": "round14_analysis_tt/analysis_*"},
}
DEFAULTS = {"experts": 4, "hidden_dim": 64, "lambda_kl": 0.01, "routing_temperature": 1.0, "lr": 5e-4, "window": 10}

NAMES = {"trivial": "Trivial (all anomalous)", "muad": "MUAD", "gat": "GAT", "plain_moe": "Plain MoE (legacy)",
         "graph_moe": "Graph-MoE", "ug_moe": "UG-MoE (legacy)", "ug_moe2:plain": "v2 plain (experts + router)",
         "ug_moe2:graph": "+ graph", "ug_moe2:prob": "+ probabilistic", "ug_moe2:ug_routing": "+ uncertainty routing",
         "ug_moe2:full": "UG-MoE v2 (full)", "abl2:no_graph": "w/o graph", "abl2:no_moe": "w/o MoE",
         "abl2:no_dynamic_routing": "w/o dynamic routing", "abl2:no_uncertainty_routing": "w/o uncertainty routing",
         "abl2:no_probabilistic": "w/o probabilistic", "abl2:no_adaptive_fusion": "w/o adaptive fusion"}
METRICS = ("f1", "precision", "recall", "f1_normal", "macro_f1", "pr_auc", "roc_auc")
HEAD = ("F1", "Precision", "Recall", "F1 normal", "Macro-F1", "PR-AUC", "ROC-AUC")

# validated categorical slots 1-5 (dataviz palette); colour follows the model in every figure
SERIES = {"ug_moe2:full": ("#2a78d6", "o"), "muad": ("#eb6834", "s"), "gat": ("#1baf7a", "^"),
          "ug_moe2:plain": ("#eda100", "D"), "abl2:no_probabilistic": ("#e87ba4", "v")}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
MODS = ("metric", "trace", "log")
MOD_COLORS = {"metric": "#2a78d6", "trace": "#eb6834", "log": "#1baf7a"}


def warn(msg):
    print(f"  [skip] {msg}")


def newest(runs_dir, pattern):
    hits = sorted(glob.glob(str(runs_dir / pattern)))
    return Path(hits[-1]) if hits else None


def load_json(path, name):
    p = path / name
    if not p.exists():
        warn(f"{p} missing")
        return None
    with open(p) as f:
        return json.load(f)


def merged_models(runs_dir, patterns):
    """
    {model: entry} pooled over every run directory matching the patterns, one result per (model, seed)
    read from the per-seed files <tag>_seed<N>.json. Later patterns / newer directories override the
    same (model, seed), so a final multi-seed run replaces an earlier seed-42 run of the same model.
    entry = {"test": {metric: {"mean", "std", "values"}}, "n_params", "seeds"}; plus the trivial row.
    """
    per, params, trivial = {}, {}, None
    for pat in patterns:
        hits = sorted(glob.glob(str(runs_dir / pat)))
        if not hits:
            warn(f"no run matches {pat}")
        for d in map(Path, hits):
            if (d / "summary.json").exists():
                s = load_json(d, "summary.json")
                trivial = trivial or (s or {}).get("trivial_always_anomalous_test")
            for f in sorted(d.glob("*_seed*.json")):
                with open(f) as fh:
                    r = json.load(fh)
                if "model" in r and "test" in r:
                    per.setdefault(r["model"], {})[r["seed"]] = r["test"]
                    params[r["model"]] = r.get("n_params")
    models = {}
    for n, by_seed in per.items():
        test = {k: {"mean": float(np.mean([t[k] for t in by_seed.values()])),
                    "std": float(np.std([t[k] for t in by_seed.values()])),
                    "values": [t[k] for t in by_seed.values()]}
                for k in METRICS if all(k in t for t in by_seed.values())}
        models[n] = {"test": test, "n_params": params[n], "seeds": sorted(by_seed), "by_seed": by_seed}
    return models, trivial


def significance_table(models, ref_name, title, metric="macro_f1"):
    """Paired t-test over the seeds both models share: ref (UG-MoE v2 full) vs every other model."""
    from scipy.stats import ttest_rel
    ref = models.get(ref_name)
    if ref is None:
        return []
    out = [f"### {title}", "", f"Paired t-test over shared seeds, test {metric}; Δ = {nice(ref_name)} − model.", "",
           "| Model | Seeds | Δ mean | Δ std | p-value |", "|---|---|---|---|---|"]
    for n, e in models.items():
        if n == ref_name or "by_seed" not in e:
            continue
        common = sorted(set(ref["by_seed"]) & set(e["by_seed"]))
        if len(common) < 2:
            continue
        a = np.array([ref["by_seed"][s][metric] for s in common])
        b = np.array([e["by_seed"][s][metric] for s in common])
        d = a - b
        p = ttest_rel(a, b).pvalue if np.std(d) > 0 else float("nan")
        out.append(f"| {nice(n)} | {len(common)} | {100 * d.mean():+.1f} | {100 * d.std(ddof=1):.1f} | "
                   f"{p:.3f}{' *' if p < 0.05 else ''} |")
    return out + ["", "(* p < 0.05. With 5 seeds the test has little power: treat p as indicative.)", ""]


def robustness_runs(runs_dir, pattern):
    """Average every robustness run matching the pattern (one per seed / checkpoint), cell by cell."""
    runs = {}
    for d in map(Path, sorted(glob.glob(str(runs_dir / pattern)))):
        s = load_json(d, "summary.json")
        if s:
            runs[s.get("seed", d.name)] = s          # one per seed, newest wins

    def avg(*xs):
        if isinstance(xs[0], dict):
            return {k: avg(*[x[k] for x in xs]) for k in xs[0] if all(k in x for x in xs)}
        return float(np.mean(xs)) if isinstance(xs[0], (int, float)) else xs[0]
    if not runs:
        return None
    return {"results": avg(*[r["results"] for r in runs.values()]), "seeds": sorted(map(str, runs))}


def stat(entry, metric):
    d = entry["test"].get(metric)
    if not d:
        return float("nan"), float("nan"), 0
    return d["mean"], d["std"], len(d.get("values", [1]))


def fmt(entry, metric):
    m, s, n = stat(entry, metric)
    if np.isnan(m):
        return "n/a"
    return f"{m:.4f} ± {s:.4f}" if n > 1 else f"{m:.4f}"


def nice(name):
    if "#" in name:
        base, kv = name.split("#", 1)
        return f"{NAMES.get(base, base)} [{kv}]"
    if "@" in name:
        base, mods = name.rsplit("@", 1)
        return f"{mods.replace('+', ' + ')} only" if base == "ug_moe2:full" else f"{NAMES.get(base, base)} @{mods}"
    return NAMES.get(name, name)


# ------------------------------------------------------------------------------------------- tables
def main_table(models, trivial, title, order):
    out = [f"### {title}", "", "Test set, checkpoint selected on validation macro-F1; mean ± std over seeds.", "",
           "| Model | Params | Seeds | " + " | ".join(HEAD) + " |", "|---|---|---|" + "---|" * len(HEAD)]
    if trivial:
        out.append("| " + NAMES["trivial"] + " | — | — | " + " | ".join(f"{trivial[k]:.4f}" for k in METRICS) + " |")
    for n in [n for n in order if n in models] + [n for n in models if n not in order]:
        e = models[n]
        p = e.get("n_params")
        out.append(f"| {nice(n)} | {f'{p / 1000:.0f}k' if p else '—'} | {stat(e, 'f1')[2]} | ")
        out[-1] += " | ".join(fmt(e, k) for k in METRICS) + " |"
    return out + [""]


def ref_on_seeds(ref, seeds, metric="macro_f1"):
    """Mean of `metric` for the reference model over the given seeds only (None if it has none of them).
    Comparisons against a single-seed run must use the reference at that seed, not its multi-seed mean."""
    shared = sorted(set(seeds) & set(ref.get("by_seed", {})))
    if not shared:
        return None
    return float(np.mean([ref["by_seed"][s][metric] for s in shared]))


def delta_table(models, ref, title, order):
    """Δ = model − full, with the full model taken on the seeds each row has (seed-matched)."""
    if ref is None:
        return [f"### {title}", "", "_reference model missing_", ""]
    out = [f"### {title}", "", "Δ = model − full, with the full model evaluated on the same seeds as the row.", "",
           "| Model | Seeds | Macro-F1 | Δ vs full | F1 | Recall |", "|---|---|---|---|---|---|",
           f"| UG-MoE v2 (full) | {len(ref.get('seeds', []))} | {fmt(ref, 'macro_f1')} | — | {fmt(ref, 'f1')} | "
           f"{fmt(ref, 'recall')} |"]
    for n in [n for n in order if n in models] + [n for n in models if n not in order]:
        e = models[n]
        r = ref_on_seeds(ref, e.get("seeds", []))
        d = "—" if r is None else f"{100 * (stat(e, 'macro_f1')[0] - r):+.1f}"
        out.append(f"| {nice(n)} | {len(e.get('seeds', []))} | {fmt(e, 'macro_f1')} | {d} | "
                   f"{fmt(e, 'f1')} | {fmt(e, 'recall')} |")
    return out + [""]


def robustness_table(rob, title):
    out = [f"### {title}", "", "Test macro-F1 (anomaly F1 in brackets) under test-time perturbation.", "",
           "| Model | Clean | −metrics | −traces | −logs | Mask 10% | Mask 20% | Mask 40% | Mask 60% | Noise all σ=1 | Noise all σ=2 |",
           "|---|" + "---|" * 10]
    for n, s in rob.items():
        r = s["results"]
        cells = [r["clean"]] + [r["drop_modality"][m] for m in ("metrics", "traces", "logs")]
        cells += [r["random_mask"][p] for p in ("0.1", "0.2", "0.4", "0.6")]
        cells += [r["noise"]["metrics+traces+logs"][lv] for lv in ("1.0", "2.0")]
        k = len(s.get("seeds", [])) or 1
        out.append(f"| {nice(n)} ({k} seed{'s' if k > 1 else ''}) | "
                   + " | ".join(f"{c['macro_f1']:.3f} ({c['f1']:.3f})" for c in cells) + " |")
    return out + [""]


def fault_type_table(rob, title, names):
    out = [f"### {title}", ""]
    rows = []
    for n, s in rob.items():
        c = s["results"]["clean"]
        types = sorted(k for k in c if k.startswith("recall_type"))
        if types:
            rows.append((n, c, types))
    if not rows:
        return out + ["_per-fault-type recall not recorded in these runs_", ""]
    types = rows[0][2]
    out += ["| Model | " + " | ".join(names.get(int(t[len('recall_type'):]), t) for t in types)
            + " | FPR on normal windows | Macro-F1 |", "|---|" + "---|" * (len(types) + 2)]
    for n, c, _ in rows:
        out.append(f"| {nice(n)} | " + " | ".join(f"{c[t]:.3f}" for t in types)
                   + f" | {c.get('fpr_normal', float('nan')):.3f} | {c['macro_f1']:.3f} |")
    return out + [""]


# ------------------------------------------------------------------------------------------ figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def line_figure(series, xs, xlabel, title, path, xticklabels=None):
    """series: {model: [y...]} plotted against xs; direct labels at line ends + legend (>= 2 series)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    _style(ax)
    for n, ys in series.items():
        col, mk = SERIES.get(n, (INK2, "o"))
        ax.plot(xs, ys, color=col, linewidth=2, marker=mk, markersize=6, label=nice(n),
                markeredgecolor=SURFACE, markeredgewidth=1)
    # direct labels at the line ends, pushed apart so they never overlap (min gap 0.045 in y)
    ends = sorted(((ys[-1], n) for n, ys in series.items()), reverse=True)
    placed = []
    for y, n in ends:
        yl = min(y, placed[-1] - 0.045) if placed else y
        placed.append(yl)
        ax.annotate(f"{nice(n)}  {y:.2f}", (xs[-1], y), xytext=(xs[-1] + 0.03 * (xs[-1] - xs[0]), yl),
                    textcoords="data", va="center", fontsize=8, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8) if abs(yl - y) > 0.01 else None)
    ax.set_xticks(xs, xticklabels or [str(x) for x in xs])
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel("test macro-F1", color=INK2)
    ax.set_ylim(0, 1)
    ax.set_xlim(xs[0] - 0.02 * (xs[-1] - xs[0]), xs[-1] + 0.45 * (xs[-1] - xs[0]))
    if len(series) >= 2:
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title(title, color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def robustness_figures(rob, system, out):
    keep = [n for n in SERIES if n in rob]
    if not keep:
        return
    p = ("0.1", "0.2", "0.4", "0.6")
    mask = {n: [rob[n]["results"]["clean"]["macro_f1"]] + [rob[n]["results"]["random_mask"][q]["macro_f1"] for q in p]
            for n in keep}
    line_figure(mask, [0, 10, 20, 40, 60], "share of missing observations (%)",
                f"Missing data: macro-F1 under random masking, {system} test", out / f"robustness_masking_{system.lower()}.png")
    lv = ("0.1", "0.25", "0.5", "1.0", "2.0")
    noise = {n: [rob[n]["results"]["clean"]["macro_f1"]]
             + [rob[n]["results"]["noise"]["metrics+traces+logs"][x]["macro_f1"] for x in lv] for n in keep}
    line_figure(noise, list(range(6)), "Gaussian noise on all modalities (σ × train std)",
                f"Noise: macro-F1 under noise on all modalities, {system} test",
                out / f"robustness_noise_{system.lower()}.png", ["0", *lv])


def sensitivity_points(models, single_runs, full_entry, no_moe_entry):
    """
    {param: [(value, macro_f1)]} including the default (full model) point. The sweep runs are single-seed,
    so the default and 1-expert points use the same seeds as the sweep (not the 5-seed mean of the final
    runs), otherwise the default would be compared at a different seed from every other point.
    """
    sweep_seeds = {s for e in models.values() for s in e.get("seeds", [])}
    sweep_seeds |= {s for _, _, e in single_runs for s in e.get("seeds", [])}

    def seed_matched(e):
        r = ref_on_seeds(e, sweep_seeds) if sweep_seeds else None
        return stat(e, "macro_f1")[0] if r is None else r

    pts = {k: [] for k in DEFAULTS}
    for n, e in models.items():
        if "#" not in n:
            continue
        for pair in n.split("#", 1)[1].split(","):
            k, v = pair.split("=", 1)
            if k == "experts":
                pts["experts"].append((len(v.split("-")), stat(e, "macro_f1")[0]))
            elif k in pts:
                pts[k].append((float(v), stat(e, "macro_f1")[0]))
    for k, v, e in single_runs:
        pts[k].append((v, stat(e, "macro_f1")[0]))
    if full_entry is not None:
        for k, v in DEFAULTS.items():
            pts[k].append((v, seed_matched(full_entry)))
    if no_moe_entry is not None:
        pts["experts"].append((1, seed_matched(no_moe_entry)))
    return {k: sorted(v) for k, v in pts.items() if len(v) > 1}


def sensitivity_figure(pts, path):
    if not pts:
        return
    plt = _plt()
    labels = {"experts": "number of experts", "hidden_dim": "embedding dimension", "lambda_kl": "KL weight λ",
              "routing_temperature": "routing temperature", "lr": "learning rate", "window": "window (s)"}
    keys = list(pts)
    cols = 3
    rows = int(np.ceil(len(keys) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.8 * rows), facecolor=SURFACE, squeeze=False)
    lo = min(y for v in pts.values() for _, y in v)
    for ax, k in zip(axes.flat, keys):
        _style(ax)
        xs = [x for x, _ in pts[k]]
        ys = [y for _, y in pts[k]]
        pos = list(range(len(xs)))                  # categorical x: values are not evenly spaced
        ax.plot(pos, ys, color=SERIES["ug_moe2:full"][0], linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1)
        d = DEFAULTS[k]
        for i, (x, y) in enumerate(zip(xs, ys)):
            ax.annotate(f"{y:.3f}", (i, y), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7,
                        color=INK if x == d else INK2, fontweight="bold" if x == d else "normal")
        ax.set_xticks(pos, [f"{x:g}" + (" *" if x == d else "") for x in xs])
        ax.set_ylim(max(0, lo - 0.03), 1.0)
        ax.set_title(labels.get(k, k), color=INK, fontsize=10)
    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")
    fig.suptitle("Sensitivity of UG-MoE v2 (SN test macro-F1; * = default)", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def analysis_figures(a, system, out):
    plt = _plt()
    cond = a["conditions"]
    names = list(cond)
    y = np.arange(len(names))[::-1]
    # fusion weights: all modalities + zoom on trace / log
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 0.42 * len(names) + 1.6), facecolor=SURFACE,
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    for ax in (ax1, ax2):
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
    left = np.zeros(len(names))
    for m in MODS:
        v = np.array([cond[n]["fusion_weight"][m] for n in names])
        ax1.barh(y, v, left=left, height=0.6, color=MOD_COLORS[m], edgecolor=SURFACE, linewidth=2, label=m)
        for yi, li, vi in zip(y, left, v):
            if vi >= 0.08:
                ax1.text(li + vi / 2, yi, f"{vi:.2f}", ha="center", va="center", fontsize=8, color="white")
        left += v
    ax1.set_xlim(0, 1)
    ax1.set_yticks(y, [f"{n} (F1 {cond[n]['f1']:.3f})" for n in names])
    ax1.set_xlabel("fusion weight (sums to 1)", color=INK2)
    ax1.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9)
    h = 0.36
    for k, m in enumerate(("trace", "log")):
        v = np.array([cond[n]["fusion_weight"][m] for n in names])
        ax2.barh(y + (0.5 - k) * h, v, height=h - 0.04, color=MOD_COLORS[m], label=m)
        for yi, vi in zip(y + (0.5 - k) * h, v):
            ax2.text(vi, yi, f" {vi:.3f}", va="center", fontsize=7, color=INK2)
    ax2.set_yticks(y, [""] * len(names))
    ax2.set_xlabel("trace / log weight (zoom)", color=INK2)
    ax2.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9)
    fig.suptitle(f"Modality fusion weights under perturbation, {system} test", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"fusion_{system.lower()}.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    # change of predicted log-variance vs clean
    base = cond["clean"]["logvar"]
    if base:
        others = [n for n in names if n != "clean"]
        yo = np.arange(len(others))[::-1]
        fig, ax = plt.subplots(figsize=(8, 0.42 * len(others) + 1.4), facecolor=SURFACE)
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        h = 0.26
        for k, m in enumerate(MODS):
            v = np.array([cond[n]["logvar"][m] - base[m] for n in others])
            ax.barh(yo + (1 - k) * h, v, height=h - 0.03, color=MOD_COLORS[m], label=m)
            for yi, vi in zip(yo + (1 - k) * h, v):
                ax.text(vi, yi, f" {vi:+.2f} " if abs(vi) >= 0.005 else "", va="center",
                        ha="left" if vi >= 0 else "right", fontsize=7, color=INK2)
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.set_yticks(yo, others)
        ax.set_xlabel("change in predicted log-variance vs clean (> 0 = less certain)", color=INK2)
        ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9)
        ax.set_title(f"Does the model become less certain about the corrupted modality? ({system} test)",
                     color=INK, fontsize=11, pad=24)
        fig.tight_layout()
        fig.savefig(out / f"uncertainty_delta_{system.lower()}.png", dpi=200, facecolor=SURFACE)
        plt.close(fig)
    # routing heatmaps (reuse the analysis script's plot)
    from run_analysis import plot_routing
    plot_routing(a["routing_clean"], a["experts"], out / f"routing_{system.lower()}.png",
                 f"Expert routing weights by class, UG-MoE v2, {system} test")


# --------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Regenerate all tables and figures from saved runs")
    ap.add_argument("--runs-dir", default=str(HERE / "runs"))
    args = ap.parse_args()
    runs = Path(args.runs_dir)
    out = runs / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    md = [f"# UG-MoE results report ({datetime.now():%Y-%m-%d %H:%M})", "",
          f"Generated by `experiments/make_report.py` from `{runs}`.", ""]
    order = ["muad", "gat", "plain_moe", "graph_moe", "ug_moe", "ug_moe2:plain", "ug_moe2:graph", "ug_moe2:prob",
             "ug_moe2:ug_routing", "ug_moe2:full"]
    abl_order = ["abl2:no_graph", "abl2:no_moe", "abl2:no_dynamic_routing", "abl2:no_uncertainty_routing",
                 "abl2:no_probabilistic", "abl2:no_adaptive_fusion"]

    print("Main tables")
    full = {}
    for sysname, key in (("SN", "main_sn"), ("TT", "main_tt")):
        models, trivial = merged_models(runs, MANIFEST[key])
        full[sysname] = models.get("ug_moe2:full")
        if models:
            md += main_table(models, trivial, f"Overall performance, {sysname} (plan 8A + §12 steps)", order)
            md += significance_table(models, "ug_moe2:full", f"Significance vs UG-MoE v2, {sysname}")

    print("Ablations")
    no_moe_sn = None
    for sysname, key in (("SN", "ablation_sn"), ("TT", "ablation_tt")):
        models, _ = merged_models(runs, MANIFEST[key])
        if sysname == "SN":
            no_moe_sn = models.get("abl2:no_moe")
        if models:
            md += delta_table(models, full[sysname], f"Component ablation, {sysname} (plan 8B)", abl_order)
            if full[sysname] is not None:
                md += significance_table({**models, "ug_moe2:full": full[sysname]}, "ug_moe2:full",
                                         f"Significance of each ablation, {sysname}")

    print("Modality ablation")
    models, _ = merged_models(runs, MANIFEST["modality_sn"])
    if models:
        md += delta_table(models, full["SN"], "Modality ablation, SN (plan 8C; model trained on the subset)", [])

    print("Sensitivity")
    models, _ = merged_models(runs, MANIFEST["sens_sn"])
    single = []
    for pat, k, v in MANIFEST["sens_sn_runs"]:
        m, _ = merged_models(runs, [pat])
        if m:
            single.append((k, v, next(iter(m.values()))))
    pts = sensitivity_points(models, single, full["SN"], no_moe_sn)
    if pts:
        md += ["### Hyperparameter sensitivity, SN (plan 8H)", "",
               "One-at-a-time sweep from the defaults, single seed; the default (and 1-expert) point is taken "
               "at the same seed, not the 5-seed mean.", "", "| Parameter | Value | Test macro-F1 |", "|---|---|---|"]
        for k, v in pts.items():
            md += [f"| {k} | {x:g}{' (default)' if x == DEFAULTS[k] else ''} | {y:.4f} |" for x, y in v]
        md.append("")
        sensitivity_figure(pts, out / "sensitivity_sn.png")

    print("Robustness")
    for sysname, key in (("SN", "robust_sn"), ("TT", "robust_tt")):
        rob = {}
        for n, pats in MANIFEST[key].items():
            s = None
            for pat in ([pats] if isinstance(pats, str) else pats)[::-1]:   # the last (final) pattern wins
                s = robustness_runs(runs, pat)
                if s:
                    break
            if s:
                rob[n] = s
            else:
                warn(f"no robustness run for {n}")
        if rob:
            md += robustness_table(rob, f"Robustness, {sysname} (plan 8D + 8E)")
            names = {1: "cpu_load", 2: "network_delay", 3: "network_loss"}
            md += fault_type_table(rob, f"Recall per fault type, clean {sysname} test", names)
            robustness_figures(rob, sysname, out)

    print("Routing / uncertainty analysis")
    for sysname, pat in MANIFEST["analysis"].items():
        d = newest(runs, pat)
        a = load_json(d, "analysis.json") if d else None
        if a:
            analysis_figures(a, sysname.upper(), out)
            cond = a["conditions"]
            md += [f"### Fusion weights and uncertainty, {sysname.upper()} (plan 8G)", "",
                   "| Condition | F1 | w metric | w trace | w log | logvar metric | logvar trace | logvar log |",
                   "|---|---|---|---|---|---|---|---|"]
            for n, c in cond.items():
                lv = c["logvar"]
                md.append(f"| {n} | {c['f1']:.3f} | " + " | ".join(f"{c['fusion_weight'][m]:.3f}" for m in MODS)
                          + " | " + " | ".join(f"{lv[m]:+.2f}" if m in lv else "—" for m in MODS) + " |")
            md.append("")
        elif d is None:
            warn(f"no run matches {pat}")

    (out / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {out / 'tables.md'} and {len(list(out.glob('*.png')))} figures to {out}")


if __name__ == "__main__":
    os.chdir(HERE.parent)
    main()
