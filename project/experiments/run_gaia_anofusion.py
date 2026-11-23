"""Fit the pinned AnoFusion network per GAIA instance and save timestamped scores."""

import argparse
from copy import deepcopy
from importlib import import_module
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from sklearn.metrics import mutual_info_score
import torch
from torch import nn

from data_pipeline.export_gaia_anofusion import (
    BIN_SECONDS, CHANNELS, LOG_LEVELS, METRIC_CHANNELS, PARTITIONS, SERVICES, WINDOW_STEPS,
)


ANO_REVISION = "ae3c4ea98509008844a3eecf7c0bd1529faaba67"
HISTORY = 20
EVAL_SPLITS = ("val", "test")


def load_network(source, device):
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != ANO_REVISION:
        raise ValueError(f"Expected AnoFusion revision {ANO_REVISION}, found {head}")
    subprocess.run(["git", "-C", str(source), "diff", "--exit-code", "HEAD", "--", "model"], check=True)
    sys.path.insert(0, str(source.resolve() / "model"))
    try:
        net = import_module("AnoFusion").Net
        # The public demo pins these modules to cuda:1.
        for name in ("GAT", "GATGRU", "GTblock", "GTlayer"):
            import_module(name).device = device
    finally:
        sys.path.pop(0)
    return net, import_module("GTlayer").GTAggregation


def forecast_indices(timestamps, splits):
    """Restart history on a split boundary or gap, padding with the first observed bin."""
    index = np.arange(len(timestamps))
    breaks = np.r_[True, (np.diff(timestamps) != BIN_SECONDS) | (splits[1:] != splits[:-1])]
    first = np.maximum.accumulate(np.where(breaks, index, 0))
    targets = index[~breaks]
    history = np.maximum(targets[:, None] + np.arange(-HISTORY, 0), first[targets, None])
    return targets, history


def relation_graph(train, modality_sizes=(len(METRIC_CHANNELS), len(LOG_LEVELS), 1)):
    """Make the six published modality-pair relations using train-only mutual information."""
    groups = np.repeat((0, 1, 2), modality_sizes)
    if train.shape[1] != len(groups):
        raise ValueError("Expected one modality assignment for every feature")
    edges = np.zeros((len(groups), len(groups), 6), dtype=np.float32)
    cutoffs = np.quantile(train, np.arange(1, 10) / 10, axis=0)
    discrete = [
        np.searchsorted(np.unique(boundaries), column, side="right")
        for boundaries, column in zip(cutoffs.T, train.T, strict=True)
    ]
    relation = {(0, 0): 0, (1, 1): 1, (2, 2): 2, (0, 1): 3, (0, 2): 4, (1, 2): 5}
    for left in range(len(groups)):
        for right in range(left, len(groups)):
            channel = relation[(groups[left], groups[right])]
            weight = mutual_info_score(discrete[left], discrete[right])
            edges[left, right, channel] = edges[right, left, channel] = weight
    return edges


def forecast_batches(model, data, graph, targets, history, rows, batch_size):
    for batch in rows.split(batch_size):
        inputs = data[history[batch]].transpose(1, 2).contiguous()
        adjacency = graph.expand(len(batch), -1, -1, -1)
        yield model(inputs, adjacency), data[targets[batch]]


@torch.no_grad()
def forecast_errors(model, data, graph, targets, history, rows, batch_size):
    model.eval()
    errors = [
        (predicted - observed).abs()
        for predicted, observed in forecast_batches(model, data, graph, targets, history, rows, batch_size)
    ]
    errors = torch.cat(errors).cpu().numpy()
    if not np.isfinite(errors).all():
        raise ValueError("AnoFusion produced a non-finite forecast error")
    return errors


def anomaly_scores(errors):
    median = np.median(errors, axis=1)
    scale = np.maximum(np.quantile(errors, 0.25, axis=1), 1e-6)
    return (errors.max(axis=1) - median) / scale


def fit_service(service, values, splits, targets, history, rows, net, aggregation, args):
    torch.manual_seed(args.seed)
    train = values[splits == "train"]
    scale = np.abs(train).max(axis=0)
    scale[scale == 0] = 1
    scaled = values / scale
    graph = torch.as_tensor(relation_graph(scaled[splits == "train"]), device=args.device)
    data = torch.as_tensor(scaled, device=args.device)
    model = net(node_num=data.shape[1], edge_types=6, window_samples_num=HISTORY, dropout=0.1)
    # The upstream aggregation parameters are allocated with torch.Tensor and lack an initializer.
    for layer in model.modules():
        if isinstance(layer, aggregation):
            nn.init.uniform_(layer.weight, 0, 0.1)
    nn.init.zeros_(model.GTN.weight)
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    best_loss = float("inf")
    history_log = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(rows["train"]), device=args.device)
        batches = forecast_batches(model, data, graph, targets, history, rows["train"][order], args.batch_size)
        for predicted, observed in batches:
            loss = nn.functional.mse_loss(predicted, observed)
            if not torch.isfinite(loss):
                raise ValueError(f"{service} produced a non-finite training loss at epoch {epoch}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        errors = forecast_errors(model, data, graph, targets, history, rows["val"], args.batch_size)
        val_loss = float(np.square(errors, dtype=np.float64).mean())
        scheduler.step(val_loss)
        history_log.append(val_loss)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = deepcopy(model.state_dict())
        print(f"{service} epoch {epoch:3d}: val forecast MSE={val_loss:.7g}", flush=True)
        if epoch - best_epoch >= args.patience:
            break

    model.load_state_dict(best_state)
    train_errors = forecast_errors(model, data, graph, targets, history, rows["train"], args.batch_size)
    reference = np.sort(anomaly_scores(train_errors))
    raw, scores = {}, {}
    for split in EVAL_SPLITS:
        errors = forecast_errors(model, data, graph, targets, history, rows[split], args.batch_size)
        raw[split] = anomaly_scores(errors)
        scores[split] = np.searchsorted(reference, raw[split], side="right") / (len(reference) + 1)
    checkpoint = {
        "state_dict": {name: value.cpu() for name, value in best_state.items()},
        "scale": torch.from_numpy(scale), "graph": graph.cpu(),
        "reference_scores": torch.from_numpy(reference),
        "best_epoch": best_epoch, "validation_mse": history_log,
    }
    return checkpoint, raw, scores


def run(args):
    with np.load(args.observations, allow_pickle=False) as dataset:
        starts, window_splits, data = dataset["starts"], dataset["splits"], dataset["observations"]
        digest = dataset["manifest_sha256"].item()
        if tuple(dataset["services"]) != SERVICES or tuple(dataset["channels"]) != CHANNELS:
            raise ValueError("AnoFusion observations have unexpected GAIA service or channel order")
    if (data.shape != (len(starts), WINDOW_STEPS, len(SERVICES), len(CHANNELS))
            or window_splits.shape != starts.shape or set(window_splits) != set(PARTITIONS)
            or not np.isfinite(data).all()):
        raise ValueError("Invalid AnoFusion observations")
    timestamps = (starts[:, None] + BIN_SECONDS * np.arange(WINDOW_STEPS)).ravel()
    splits = np.repeat(window_splits, WINDOW_STEPS)
    targets, history = forecast_indices(timestamps, splits)
    rows = {split: torch.as_tensor(np.flatnonzero(splits[targets] == split), device=args.device)
            for split in PARTITIONS}
    if any(len(index) == 0 for index in rows.values()):
        raise ValueError("AnoFusion requires forecast targets in every split")
    services = args.services or SERVICES
    if len(set(services)) != len(services):
        raise ValueError("AnoFusion service list contains duplicates")
    net, aggregation = load_network(args.source, args.device)
    args.output_dir.mkdir(parents=True)
    data = data.reshape(-1, len(SERVICES), len(CHANNELS))
    target_tensor = torch.as_tensor(targets, device=args.device)
    history_tensor = torch.as_tensor(history, device=args.device)
    completed = {}
    for service in services:
        checkpoint, raw, scores = fit_service(
            service, data[:, SERVICES.index(service), :], splits, target_tensor, history_tensor,
            rows, net, aggregation, args,
        )
        torch.save(checkpoint, args.output_dir / f"{service}.pth")
        np.savez_compressed(
            args.output_dir / f"{service}.npz", manifest_sha256=digest,
            **{f"{split}_timestamps": timestamps[targets[splits[targets] == split]] for split in EVAL_SPLITS},
            **{f"{split}_scores": scores[split] for split in EVAL_SPLITS},
            **{f"{split}_raw_scores": raw[split] for split in EVAL_SPLITS},
        )
        completed[service] = checkpoint["best_epoch"]
    summary = {
        "source_revision": ANO_REVISION, "manifest_sha256": digest, "services": list(services),
        "observations": str(args.observations), "history_bins": HISTORY, "seed": args.seed,
        "epochs_cap": args.epochs, "patience": args.patience, "batch_size": args.batch_size,
        "device": str(args.device), "best_epochs": completed,
        "forecast_rows": {split: len(index) for split, index in rows.items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"AnoFusion scores saved to {args.output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="Checkout of the pinned public AnoFusion model")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--services", nargs="+", choices=SERVICES, help="Run a bounded instance pilot")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience) < 1:
        parser.error("epochs, batch size, and patience must be positive")
    run(args)


if __name__ == "__main__":
    main()
