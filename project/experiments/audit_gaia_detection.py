"""Audit the frozen window predictions and replay one AnoFusion validation checkpoint."""

import argparse
import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
from scipy.stats import genpareto
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from data_pipeline.build_gaia import (
    BIN_SECONDS, DEFAULT_LABELS, DEFAULT_OUTPUT, GAIA_CLOCK, METRIC_CHANNELS,
    SERVICES, WINDOW_SECONDS, WINDOW_STEPS, read_faults,
)
from data_pipeline.export_gaia_anofusion import CHANNELS, frozen_manifest
from experiments.run_gaia_anofusion import (
    ANO_REVISION, HISTORY, anomaly_scores, forecast_errors, forecast_indices, load_network,
)


def incident_table(frame, chunks, faults):
    """One row per injection beginning on an evaluated date, including zero coverage."""
    starts = np.array([chunks[uid]["t0"] for uid in frame.window_id])
    expected = np.zeros(len(starts), dtype=bool)
    overlaps = np.zeros(len(starts), dtype=int)
    for start, end, _, _ in faults:
        mask = np.minimum(starts + WINDOW_SECONDS, end) - np.maximum(starts, start) >= 1
        expected |= mask
        overlaps += mask
    np.testing.assert_array_equal(expected, frame.label.to_numpy())
    times = (starts[:, None] + np.arange(WINDOW_STEPS) * BIN_SECONDS).ravel()
    metrics = np.concatenate([chunks[uid]["metrics"].transpose(1, 0, 2) for uid in frame.window_id])
    mem = [METRIC_CHANNELS.index(k) for k in ("docker_memory_usage_total", "docker_memory_rss_total")]
    days = {datetime.fromtimestamp(int(t), GAIA_CLOCK).date() for t in starts}
    predictions = frame.ug_prediction.to_numpy()
    rows = []
    for start, end, kind, service in faults:
        day = datetime.fromtimestamp(start, GAIA_CLOCK).date()
        if kind != 1 or day not in days:
            continue
        seconds = np.minimum(starts + WINDOW_SECONDS, end) - np.maximum(starts, start)
        windows = seconds >= 1
        # Only complete 30-second bins. Use monitored background for the baseline.
        before = (times >= start - 300) & (times + BIN_SECONDS <= start) & ~np.repeat(expected, WINDOW_STEPS)
        during = (times >= start) & (times + BIN_SECONDS <= end)
        values = metrics[:, SERVICES.index(service)][:, mem].astype(float) / 2**30
        delta = [None, None]
        if before.any() and during.any():
            delta = np.median(values[during], axis=0) - np.median(values[before], axis=0)
        rows.append(dict(day=str(day), service=service, start=start, end=end,
                         scored_windows=int(windows.sum()), flagged_windows=int(predictions[windows].sum()),
                         partial_windows=int((windows & (seconds < WINDOW_SECONDS)).sum()),
                         shared_windows=int((windows & (overlaps > 1)).sum()),
                         baseline_bins=int(before.sum()), event_bins=int(during.sum()),
                         usage_delta_gib=delta[0], rss_delta_gib=delta[1]))
    return pd.DataFrame(rows)


def causal_pot(train, values):
    """Peaks over threshold. Refit every 256 observations using only prior accepted scores."""
    base = np.quantile(train, 0.95)
    peaks = list(train[train > base] - base)
    n = len(train)
    limits = []
    for i, score in enumerate(values):
        if i % 256 == 0:
            if len(peaks) < 30:
                raise ValueError("Insufficient training tail for a peaks-over-threshold fit")
            shape, _, scale = genpareto.fit(peaks, floc=0)
            tail_probability = min(1.0, 0.005 * n / len(peaks))
            threshold = base + genpareto.isf(tail_probability, shape, scale=scale)
            if not np.isfinite(threshold):
                raise ValueError("Nonfinite peaks-over-threshold estimate")
        limits.append(threshold)
        if score <= threshold:
            n += 1
            if score > base:
                peaks.append(score - base)
    return np.asarray(limits)


def replay(run, source, digest, faults, stage):
    service = "mobservice2"
    directory = run / "anofusion"
    config = json.loads((directory / "summary.json").read_text())
    if (config["manifest_sha256"], config["source_revision"], config["seed"]) != (digest, ANO_REVISION, 42):
        raise ValueError("Unexpected source, seed, or split in checkpoint run")
    with np.load(run / "observations.npz", allow_pickle=False) as data:
        if (data["manifest_sha256"].item() != digest or tuple(data["services"]) != SERVICES
                or tuple(data["channels"]) != CHANNELS):
            raise ValueError("Observation provenance mismatch")
        times = (data["starts"][:, None] + BIN_SECONDS * np.arange(WINDOW_STEPS)).ravel()
        splits = np.repeat(data["splits"], WINDOW_STEPS)
        values = data["observations"][:, :, SERVICES.index(service), :].reshape(-1, len(CHANNELS))
    checkpoint = torch.load(directory / f"{service}.pth", map_location="cpu", weights_only=True)
    net, _ = load_network(source, torch.device("cpu"))
    model = net(len(CHANNELS), 6, HISTORY, 0.1)
    model.load_state_dict(checkpoint["state_dict"])
    targets, histories = forecast_indices(times, splits)
    x = torch.from_numpy(values) / checkpoint["scale"]
    scores = {}
    for split in ("train", "val"):
        rows = np.flatnonzero(splits[targets] == split)
        errors = forecast_errors(model, x, checkpoint["graph"], torch.from_numpy(targets),
                                 torch.from_numpy(histories), torch.from_numpy(rows), config["batch_size"])
        q1, q3 = np.quantile(errors, (0.25, 0.75), axis=1)
        scores[split] = dict(q1=anomaly_scores(errors),
                             iqr=(errors.max(axis=1) - np.median(errors, axis=1)) / np.maximum(q3 - q1, 1e-6))
    val_times = times[targets[splits[targets] == "val"]]
    with np.load(directory / f"{service}.npz") as saved:
        np.testing.assert_array_equal(val_times, saved["val_timestamps"])
        original = saved["val_raw_scores"]
        # A CPU replay of a CUDA checkpoint can differ slightly in float32 reductions.
        relative_error = np.abs(scores["val"]["q1"] - original) / np.maximum(np.abs(original), 1e-6)
        if np.max(relative_error) > 0.005:
            raise ValueError(f"Validation score replay differs by {np.max(relative_error):.4%}")
    truth = np.zeros(len(val_times), dtype=bool)
    for start, end, _, affected in faults:
        if affected == service:
            truth |= np.minimum(val_times + BIN_SECONDS, end) - np.maximum(val_times, start) >= 1
    out = pd.DataFrame(dict(timestamp=val_times, label=truth.astype(int)))
    result = dict(service=service, unit="30-second forecast target", val_bins=len(truth),
                  positive_bins=int(truth.sum()), max_relative_q1_replay_error=float(relative_error.max()))
    for formula, values in scores["val"].items():
        limits = causal_pot(scores["train"][formula], values)
        out[formula] = values
        out[f"{formula}_pot_threshold"] = limits
        result[formula] = dict(roc_auc=float(roc_auc_score(truth, values)),
                               average_precision=float(average_precision_score(truth, values)),
                               pot_confusion_tn_fp_fn_tp=confusion_matrix(truth, values > limits, labels=[0, 1]).ravel().tolist())
    out.to_csv(stage / "mobservice2_validation.csv", index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    manifest, digest = frozen_manifest(args.data_dir)
    pred_path = args.run / "comparison/predictions.csv"
    predictions = pd.read_csv(pred_path)
    with (args.data_dir / "chunks.pkl").open("rb") as stream:
        chunks = pickle.load(stream)
    faults = read_faults(args.labels)[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temp:
        stage = Path(temp) / "result"
        stage.mkdir()
        tables = []
        for split in ("val", "test"):
            frame = predictions.loc[predictions.split == split]
            if frame.window_id.tolist() != manifest[f"{split}_ids"]:
                raise ValueError(f"Unexpected {split} window IDs")
            tables.append(incident_table(frame, chunks, faults).assign(split=split))
        pd.concat(tables).to_csv(stage / "memory_incidents.csv", index=False)
        report = dict(manifest_sha256=digest, predictions_sha256=hashlib.sha256(pred_path.read_bytes()).hexdigest(),
                      validation=replay(args.run, args.source, digest, faults, stage))
        (stage / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        stage.rename(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
