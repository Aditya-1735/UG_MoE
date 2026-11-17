#!/usr/bin/env python3
"""
Robustness experiments (implementation guide section 8D + 8E), test-time perturbations of a model
trained on clean data under the shared protocol. Replaces run_missing_modality.py and
run_noise_robustness.py.

D. Missing modality
   D1  drop one modality entirely (metrics | traces | logs set to 0 for every service)
   D2  random masking: each (sample, service, modality) observation is zeroed independently
       with probability p in {0.1, 0.2, 0.4, 0.6}
E. Noise
   Gaussian noise with sigma = level * (per-channel std of that modality on the TRAINING split),
   levels {0.1, 0.25, 0.5, 1.0, 2.0}, applied to every non-empty subset of the three modalities.

Every condition reports the full metric set plus relative F1 / macro-F1 degradation vs. clean:
    rel_drop = (clean - perturbed) / clean
Perturbations use a fixed RNG seed per condition, so results are reproducible.

Usage (from project/):
    python experiments/run_robustness.py --device cuda                                   # train ug_moe, seed 42
    python experiments/run_robustness.py --model ug_moe --checkpoint path/to/best.pth --device cuda
"""
import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path

import torch

from common import configure_torch, evaluate, load_splits, seed_everything, train_and_select
from registry import add_common_args, get_model

MODALITIES = ("metrics", "traces", "logs")
MASK_RATES = (0.1, 0.2, 0.4, 0.6)
NOISE_LEVELS = (0.1, 0.25, 0.5, 1.0, 2.0)


@torch.no_grad()
def channel_std(loader):
    """Per-channel std of each modality over the training split (last dim = channels)."""
    sums = {m: None for m in MODALITIES}
    sq = {m: None for m in MODALITIES}
    n = {m: 0 for m in MODALITIES}
    for graph, _ in loader:
        for m in MODALITIES:
            x = graph.ndata[m].float()
            x = x.reshape(-1, x.shape[-1])
            sums[m] = x.sum(0) if sums[m] is None else sums[m] + x.sum(0)
            sq[m] = (x ** 2).sum(0) if sq[m] is None else sq[m] + (x ** 2).sum(0)
            n[m] += x.shape[0]
    return {m: ((sq[m] / n[m]) - (sums[m] / n[m]) ** 2).clamp_min(0).sqrt() for m in MODALITIES}


def make_forward(base_forward, perturb):
    def fwd(model, graph, labels):
        perturb(graph)
        return base_forward(model, graph, labels)
    return fwd


def drop_modalities(mods):
    def perturb(graph):
        for m in mods:
            graph.ndata[m] = torch.zeros_like(graph.ndata[m])
    return perturb


def random_mask(p, seed):
    gen = {}

    def perturb(graph):
        dev = graph.device
        if dev not in gen:
            gen[dev] = torch.Generator(device=dev).manual_seed(seed)
        n = graph.num_nodes()
        for m in MODALITIES:
            keep = (torch.rand(n, generator=gen[dev], device=dev) >= p).float()
            x = graph.ndata[m]
            graph.ndata[m] = x * keep.view(-1, *([1] * (x.dim() - 1)))
    return perturb


def gaussian_noise(mods, level, stds, seed):
    gen = {}

    def perturb(graph):
        dev = graph.device
        if dev not in gen:
            gen[dev] = torch.Generator(device=dev).manual_seed(seed)
        for m in mods:
            x = graph.ndata[m]
            noise = torch.randn(x.shape, generator=gen[dev], device=dev) * stds[m].to(dev) * level
            graph.ndata[m] = x + noise
    return perturb


def with_drop(metrics, clean):
    out = dict(metrics)
    for k in ("f1", "macro_f1"):
        out[f"rel_drop_{k}"] = (clean[k] - metrics[k]) / clean[k] if clean[k] else float("nan")
    return out


def main():
    parser = add_common_args(argparse.ArgumentParser(description="Missing-modality + noise robustness"))
    parser.add_argument("--model", type=str, default="ug_moe", help="Registry name (see experiments/registry.py)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Load this state_dict instead of training (must match --model)")
    args = parser.parse_args()

    configure_torch()
    device = torch.device(args.device)
    seed = int(args.seeds.split(",")[0])  # robustness is run on a single trained model
    tag = args.model.replace(":", "_")

    out_dir = Path(args.output_dir) / f"robustness_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    loaders = load_splits(Path(args.data_dir), args.batch_size, args.val_ratio, args.split_seed,
                          split_out=out_dir / "split.json", normalize=args.normalize, split=args.split, test_run=args.test_run, val_run=args.val_run)

    builder, base_fwd = get_model(args.model)
    seed_everything(seed)
    model = builder(loaders["metadata"]).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        train_info = {"checkpoint": args.checkpoint}
        print(f"Loaded checkpoint {args.checkpoint}")
    else:
        res = train_and_select(
            model, loaders, device, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            patience=args.patience if args.patience > 0 else None,
            ckpt_path=out_dir / f"{tag}_seed{seed}_best.pth", forward_fn=base_fwd, log_prefix="  ",
        )
        train_info = {k: v for k, v in res.items() if k != "history"}

    clean = evaluate(model, loaders["test"], device, base_fwd)
    print(f"\nClean test: F1={clean['f1']:.4f} macroF1={clean['macro_f1']:.4f}")
    stds = channel_std(loaders["train"])

    results = {"clean": clean, "drop_modality": {}, "random_mask": {}, "noise": {}}
    cond_seed = 1000

    print("\nD1 - drop one modality")
    for m in MODALITIES:
        r = with_drop(evaluate(model, loaders["test"], device, make_forward(base_fwd, drop_modalities([m]))), clean)
        results["drop_modality"][m] = r
        print(f"  -{m:<8} F1={r['f1']:.4f} macroF1={r['macro_f1']:.4f} relDropF1={r['rel_drop_f1']:+.3f}")

    print("\nD2 - random masking of (sample, service, modality) observations")
    for p in MASK_RATES:
        cond_seed += 1
        r = with_drop(evaluate(model, loaders["test"], device, make_forward(base_fwd, random_mask(p, cond_seed))), clean)
        results["random_mask"][str(p)] = r
        print(f"  p={p:<4} F1={r['f1']:.4f} macroF1={r['macro_f1']:.4f} relDropF1={r['rel_drop_f1']:+.3f}")

    print("\nE - Gaussian noise (sigma = level x train std per channel)")
    for k in range(1, len(MODALITIES) + 1):
        for mods in itertools.combinations(MODALITIES, k):
            key = "+".join(mods)
            results["noise"][key] = {}
            for level in NOISE_LEVELS:
                cond_seed += 1
                fwd = make_forward(base_fwd, gaussian_noise(mods, level, stds, cond_seed))
                r = with_drop(evaluate(model, loaders["test"], device, fwd), clean)
                results["noise"][key][str(level)] = r
            row = " ".join(f"{lv}:{results['noise'][key][str(lv)]['f1']:.4f}" for lv in NOISE_LEVELS)
            print(f"  {key:<22} F1 by level -> {row}")

    summary = {
        "experiment": "robustness",
        "model": args.model,
        "seed": seed,
        "config": vars(args),
        "mask_rates": MASK_RATES,
        "noise_levels_x_train_std": NOISE_LEVELS,
        "train": train_info,
        "results": results,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
