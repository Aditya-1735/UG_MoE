"""Run the predefined one-instance GAIA forecasting pilot on an audited minute grid."""

import argparse
from copy import deepcopy
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
import torch
from torch import nn

from data_pipeline.build_gaia_instance import HISTORY, METRIC_CHANNELS, SECONDS, SERVICE, TRACE_CHANNELS
from experiments.audit_gaia_detection import causal_pot
from experiments.run_gaia_anofusion import ANO_REVISION, forecast_batches, forecast_errors, load_network, relation_graph


def segments(times, flags):
    """Inclusive first and last timestamps of each run of consecutive positive minutes."""
    selected = times[flags]
    if len(selected) == 0:
        return []
    return [(int(a[0]), int(a[-1])) for a in np.split(selected, np.flatnonzero(np.diff(selected) != SECONDS) + 1)]


def detection_counts(tp, fp, fn):
    return dict(tp=int(tp), fp=int(fp), fn=int(fn), precision=tp / max(1, tp + fp),
                recall=tp / max(1, tp + fn), f1=2 * tp / max(1, 2 * tp + fp + fn))


def evaluate(times, truth, scores, limits):
    pred = scores > limits
    _, fp, fn, tp = confusion_matrix(truth, pred, labels=[False, True]).ravel().tolist()
    actual, alerts = segments(times, truth), segments(times, pred)
    i = j = matches = 0
    while i < len(actual) and j < len(alerts):
        if actual[i][1] < alerts[j][0]:
            i += 1
        elif alerts[j][1] < actual[i][0]:
            j += 1
        else:
            matches += 1
            i += 1
            j += 1
    return dict(minutes=detection_counts(tp, fp, fn),
                episodes=detection_counts(matches, len(alerts) - matches, len(actual) - matches),
                average_precision=float(average_precision_score(truth, scores)),
                roc_auc=float(roc_auc_score(truth, scores)))


def run(data_dir, source, output, device, epoch_cap=100):
    if output.exists():
        raise FileExistsError(output)
    torch.manual_seed(42)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if (manifest["service"], manifest["source_revision"], manifest["history"]) != (SERVICE, ANO_REVISION, HISTORY):
        raise ValueError("Unexpected pilot data manifest")
    path = data_dir / "minutes.npz"
    with np.load(path, allow_pickle=False) as data:
        times, splits, features = data["timestamps"], data["splits"], data["features"]
        seconds, targets = data["positive_seconds"], data["targets"]
        if (list(data["channels"]) != manifest["channels"] or
                list(data["channels"][:len(METRIC_CHANNELS)]) != list(METRIC_CHANNELS) or
                list(data["channels"][-len(TRACE_CHANNELS):]) != list(TRACE_CHANNELS)):
            raise ValueError("Unexpected feature order")
    expected_rows = np.arange(len(times))
    if np.any(np.diff(times) <= 0) or np.any(times % SECONDS) or not np.isfinite(features).all():
        raise ValueError("Invalid observation times or features")
    breaks = np.r_[True, (np.diff(times) != SECONDS) | (splits[1:] != splits[:-1])]
    first = np.maximum.accumulate(np.where(breaks, expected_rows, 0))
    np.testing.assert_array_equal(targets, expected_rows[expected_rows - first >= HISTORY])
    rows = {}
    for split in ("train", "val", "test"):
        idx = np.flatnonzero(splits[targets] == split)
        if len(idx) != manifest["splits"][split]["targets"] or len(np.unique(seconds[targets[idx]] > 0)) != 2:
            raise ValueError(f"Unexpected {split} target support")
        rows[split] = torch.as_tensor(idx, device=device)
    # Build histories from the previous 60 complete minutes. x[t] is the target.
    history = torch.as_tensor(targets[:, None] + np.arange(-HISTORY, 0), device=device)
    target_tensor = torch.as_tensor(targets, device=device)
    train = features[splits == "train"]
    scale = np.abs(train).max(axis=0)
    scale[scale == 0] = 1
    x = torch.as_tensor(features / scale, device=device)
    width = features.shape[1]
    groups = (len(METRIC_CHANNELS), width - len(METRIC_CHANNELS) - len(TRACE_CHANNELS), len(TRACE_CHANNELS))
    graph = torch.as_tensor(relation_graph(train / scale, groups), device=device)
    net, aggregation = load_network(source, device)
    model = net(width, 6, HISTORY, 0.1)
    # Upstream hard-codes a 20-step GAT. Preserve its architecture at the pilot width.
    model.GAT_GRU.gat_layer = import_module("GAT").GATNet(HISTORY, width, HISTORY, 6)
    for layer in model.modules():
        if isinstance(layer, aggregation):
            nn.init.uniform_(layer.weight, 0, 0.1)
    nn.init.zeros_(model.GTN.weight)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.5)
    losses = []
    best_loss = float("inf")
    batch_size = 32
    for epoch in range(1, epoch_cap + 1):
        model.train()
        order = torch.randperm(len(rows["train"]), device=device)
        batches = forecast_batches(model, x, graph, target_tensor, history, rows["train"][order], batch_size)
        for prediction, observed in batches:
            loss = nn.functional.mse_loss(prediction, observed)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite training loss at epoch {epoch}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        errors = forecast_errors(model, x, graph, target_tensor, history, rows["val"], batch_size)
        val = float(np.mean(np.square(errors, dtype=np.float64)))
        losses.append(val)
        scheduler.step(val)
        if val < best_loss:
            best_loss, best_epoch, state = val, epoch, deepcopy(model.state_dict())
        print(f"Epoch {epoch:3d} validation forecast MSE {val:.7g}", flush=True)
        if epoch - best_epoch >= 10:
            break
    model.load_state_dict(state)
    result = dict(seed=42, input_sha256=sha256(path.read_bytes()).hexdigest(), best_epoch=best_epoch,
                  epoch_cap=epoch_cap, batch_size=batch_size, device=str(device), torch_version=str(torch.__version__),
                  validation_mse=losses, source_revision=ANO_REVISION, service=SERVICE,
                  criterion="IQR residual plus causal peaks over threshold", evaluation={})
    scores = {}
    for split, idx in rows.items():
        errors = forecast_errors(model, x, graph, target_tensor, history, idx, batch_size)
        low, median, high = np.quantile(errors, (0.25, 0.5, 0.75), axis=1)
        scores[split] = (errors.max(axis=1) - median) / np.maximum(high - low, 1e-6)
    frames = []
    for split in ("val", "test"):
        # A fresh deployment for each partition with the same frozen training reference.
        limits = causal_pot(scores["train"], scores[split])
        idx = targets[rows[split].cpu().numpy()]
        truth = seconds[idx] > 0
        result["evaluation"][split] = evaluate(times[idx], truth, scores[split], limits)
        frames.append(pd.DataFrame(dict(split=split, timestamp=times[idx], positive_seconds=seconds[idx],
                                        score=scores[split], limit=limits, alarm=scores[split] > limits)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / "result"
        stage.mkdir()
        torch.save(dict(state_dict={k: v.cpu() for k, v in state.items()}, graph=graph.cpu(),
                        scale=torch.from_numpy(scale), train_scores=torch.from_numpy(scores["train"])),
                   stage / "best.pth")
        pd.concat(frames).to_csv(stage / "predictions.csv", index=False)
        (stage / "summary.json").write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")
        stage.rename(output)
    print(f"Instance pilot saved to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    run(args.data_dir, args.source, args.output, torch.device(args.device))
