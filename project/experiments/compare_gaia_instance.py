"""Compare seed-42 UG-MoE and saved AnoFusion scores on the frozen one-instance pilot."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import configure_torch, seed_everything, train_and_select
from compare_gaia import validation_threshold
from registry import get_model
import dgl
from data_pipeline.build_gaia_instance import HISTORY, METRIC_CHANNELS, SERVICE, TRACE_CHANNELS
from experiments.run_gaia_anofusion import ANO_REVISION
from experiments.run_gaia_instance import evaluate


def prepare(pilot):
    path = pilot / "data/minutes.npz"
    digest = sha256(path.read_bytes()).hexdigest()
    manifest = json.loads((pilot / "data/manifest.json").read_text())
    ano = json.loads((pilot / "anofusion/summary.json").read_text())
    if (ano["input_sha256"], ano["source_revision"], ano["service"], ano["seed"]) != (digest, ANO_REVISION, SERVICE, 42):
        raise ValueError("AnoFusion results do not match the pilot inputs")
    with np.load(path, allow_pickle=False) as data:
        x, seconds, splits = data["features"], data["positive_seconds"], data["splits"]
        t, targets, channels = data["timestamps"], data["targets"], data["channels"].tolist()
    m, n = len(METRIC_CHANNELS), len(TRACE_CHANNELS)
    if (manifest["channels"] != channels or channels[:m] != list(METRIC_CHANNELS)
            or channels[-n:] != list(TRACE_CHANNELS) or not np.isfinite(x).all()):
        raise ValueError("Unexpected modality order or invalid features")
    breaks = np.r_[True, (np.diff(t) != 60) | (splits[1:] != splits[:-1])]
    pos = np.arange(len(t))
    expected_targets = pos[pos - np.maximum.accumulate(np.where(breaks, pos, 0)) >= HISTORY]
    np.testing.assert_array_equal(targets, expected_targets)
    indices = {s: targets[splits[targets] == s] for s in ("train", "val", "test")}
    saved = pd.read_csv(pilot / "anofusion/predictions.csv", float_precision="round_trip")
    frames = {}
    for split in ("val", "test"):
        frame = saved.loc[saved.split == split].copy()
        idx = indices[split]
        np.testing.assert_array_equal(frame.timestamp, t[idx])
        np.testing.assert_array_equal(frame.positive_seconds, seconds[idx])
        if not np.isfinite(frame[["score", "limit"]]).all().all():
            raise ValueError("Invalid saved AnoFusion scores")
        frames[split] = frame.rename(columns={"score": "ano_score", "limit": "ano_evt_limit", "alarm": "ano_evt_alarm"})
        # Descriptive control for an explicit application error observed in this pilot.
        error_channels = [channels.index(f"drain_{key}") for key, text in manifest["templates"].items()
                          if "information has expired, mobile phone login is invalid" in text]
        frames[split]["expired_login_count"] = x[idx][:, error_channels].sum(axis=1)
    train = x[splits == "train"]
    mean, std = train.mean(0, dtype=np.float64), train.std(0, dtype=np.float64)
    std[std == 0] = 1
    features = torch.from_numpy(((x - mean) / std).astype(np.float32))
    labels = torch.from_numpy((seconds > 0).astype(np.int64))
    node = dgl.graph(([0], [0]), num_nodes=1)
    offsets = torch.arange(-HISTORY, 1)

    def collate(samples):
        idx = torch.as_tensor(samples)
        values = features[idx[:, None] + offsets]
        graph = dgl.batch([node] * len(samples))
        graph.ndata["metrics"] = values[:, :, :m]
        graph.ndata["logs"] = values[:, :, m:-n]
        graph.ndata["traces"] = values[:, :, -n:]
        return graph, labels[idx]

    loaders = {s: DataLoader(idx.tolist(), batch_size=50, shuffle=s == "train", collate_fn=collate)
               for s, idx in indices.items()}
    metadata = dict(metric_num=m, trace_dim=n, event_num=len(channels) - m - n, node_num=1)
    return loaders, frames, metadata, dict(channels=channels, mean=mean.tolist(), std=std.tolist()), digest


@torch.no_grad()
def probabilities(model, loader, device):
    model.eval()
    return np.concatenate([torch.softmax(model(graph.to(device))["MMlogit"], dim=1)[:, 1].cpu().numpy()
                           for graph, _ in loader]).astype(np.float64)


def calibrated_scores(frames, column):
    val = frames["val"]
    threshold = validation_threshold((val.positive_seconds > 0).to_numpy(), val[column].to_numpy())
    result = {"threshold_ge": threshold}
    for split, frame in frames.items():
        scores = frame[column].to_numpy()
        if not np.isfinite(scores).all():
            raise ValueError(f"Invalid {column} in {split}")
        frame[column + "_alarm"] = scores >= threshold
        # The instance evaluator uses strict >. Lower by one float64 ULP to apply >=.
        result[split] = evaluate(frame.timestamp.to_numpy(), (frame.positive_seconds > 0).to_numpy(),
                                 scores, np.nextafter(threshold, -np.inf))
    return result


def run(pilot, output, device):
    if output.exists():
        raise FileExistsError(output)
    configure_torch()
    seed_everything(42)
    loaders, frames, metadata, norm, digest = prepare(pilot)
    builder, forward = get_model("ug_moe2:full")
    model = builder(metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "results"
        stage.mkdir()
        training = train_and_select(model, loaders, device, ckpt_path=stage / "best.pth", forward_fn=forward)
        if any(row["skipped_batches"] for row in training["history"]):
            raise ValueError("Training skipped nonfinite batches")
        training["checkpoint"] = "best.pth"
        for split, frame in frames.items():
            frame["ug_score"] = probabilities(model, loaders[split], device)
        summary = dict(service=SERVICE, seed=42, num_nodes=1, history_minutes=HISTORY,
                       input_sha256=digest, training=training, normalization=norm,
                       threshold_rule="highest threshold maximizing validation macro-F1",
                       ug=calibrated_scores(frames, "ug_score"), ano=calibrated_scores(frames, "ano_score"))
        summary["expired_login_control"] = {
            split: evaluate(frame.timestamp.to_numpy(), (frame.positive_seconds > 0).to_numpy(),
                            frame.expired_login_count.to_numpy(), 0)
            for split, frame in frames.items()
        }
        (stage / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        pd.concat(frames.values()).to_csv(stage / "predictions.csv", index=False)
        stage.rename(output)
    for name in ("ug", "ano"):
        result = summary[name]["test"]
        print(f"{name} test minute F1={result['minutes']['f1']:.4f}, episode F1={result['episodes']['f1']:.4f}")
    print(f"Matched instance comparison saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    run(args.pilot, args.output, torch.device(args.device))
