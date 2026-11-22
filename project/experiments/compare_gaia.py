"""Compare a saved UG-MoE run and AnoFusion scores on identical GAIA windows."""

import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from common import compute_metrics, configure_torch, load_splits, to_binary
from registry import get_model
from data_pipeline.export_gaia_anofusion import (
    BIN_SECONDS, DEFAULT_OUTPUT, PARTITIONS, SERVICES, WINDOW_SECONDS, frozen_manifest,
)
from run_gaia_anofusion import ANO_REVISION, EVAL_SPLITS


def match_ano_windows(ano_dir, manifest, digest):
    summary = json.loads((ano_dir / "summary.json").read_text(encoding="utf-8"))
    if summary["source_revision"] != ANO_REVISION or summary["manifest_sha256"] != digest:
        raise ValueError("AnoFusion scores come from a different upstream revision or GAIA split")
    if len(summary["services"]) != len(SERVICES) or set(summary["services"]) != set(SERVICES):
        raise ValueError("A matched comparison requires AnoFusion scores for all ten GAIA instances")
    starts = {
        split: np.array([int(uid.removeprefix("gaia-")) for uid in manifest[f"{split}_ids"]])
        for split in EVAL_SPLITS
    }
    combined = {split: np.full(len(starts[split]), -np.inf) for split in EVAL_SPLITS}
    expected_times = {}
    for service in SERVICES:
        with np.load(ano_dir / f"{service}.npz", allow_pickle=False) as data:
            if data["manifest_sha256"].item() != digest:
                raise ValueError(f"Manifest mismatch in AnoFusion scores for {service}")
            for split in EVAL_SPLITS:
                timestamps, scores = data[f"{split}_timestamps"], data[f"{split}_scores"]
                if (timestamps.ndim != 1 or timestamps.shape != scores.shape or len(scores) == 0
                        or np.any(np.diff(timestamps) <= 0) or not np.isfinite(scores).all()
                        or np.any((scores < 0) | (scores > 1))):
                    raise ValueError(f"Invalid {split} timestamps or scores for {service}")
                if not np.array_equal(expected_times.setdefault(split, timestamps), timestamps):
                    raise ValueError(f"AnoFusion timestamps differ across services in {split}")
                windows = np.searchsorted(starts[split], timestamps, side="right") - 1
                offset = timestamps - starts[split][windows]
                if np.any((windows < 0) | (offset >= WINDOW_SECONDS) | (offset % BIN_SECONDS != 0)):
                    raise ValueError(f"AnoFusion timestamp outside the frozen {split} windows")
                if len(np.unique(windows)) != len(starts[split]):
                    raise ValueError(f"Missing frozen {split} windows for {service}")
                np.maximum.at(combined[split], windows, scores)
    return combined, summary


@torch.no_grad()
def ug_predictions(data_dir, ug_run, manifest, device, batch_size):
    original_split = json.loads((ug_run / "split.json").read_text(encoding="utf-8"))
    summary = json.loads((ug_run / "summary.json").read_text(encoding="utf-8"))
    config = summary["config"]
    if config["models"] != "ug_moe2:full" or config["seeds"] != "42":
        raise ValueError("Expected the saved full UG-MoE seed-42 run")
    for split in PARTITIONS:
        if original_split[f"{split}_ids"] != manifest[f"{split}_ids"]:
            raise ValueError(f"The UG-MoE checkpoint was trained on a different {split} split")
    with TemporaryDirectory(prefix="gaia-ug-predictions-") as temp:
        split_out = Path(temp) / "split.json"
        loaders = load_splits(data_dir, batch_size=batch_size, split_out=split_out, normalize=config["normalize"])
        current = split_out.with_name("normalization.json")
        previous = ug_run / "normalization.json"
        if json.loads(current.read_text(encoding="utf-8")) != json.loads(previous.read_text(encoding="utf-8")):
            raise ValueError("Current train normalization differs from the saved UG-MoE run")
    builder, forward = get_model("ug_moe2:full")
    model = builder(loaders["metadata"])
    checkpoint = ug_run / "ug_moe2_full_seed42_best.pth"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to(device).eval()
    truth, scores = {}, {}
    for split in EVAL_SPLITS:
        loader = loaders[split]
        ids = [loader.dataset.get_chunk_id(i) for i in range(len(loader.dataset))]
        if ids != manifest[f"{split}_ids"]:
            raise ValueError(f"Unexpected UG-MoE window order in {split}")
        labels, probabilities = [], []
        for graph, culprit in loader:
            logits, _ = forward(model, graph.to(device), culprit.to(device))
            probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            labels.append(to_binary(culprit).numpy())
        truth[split] = np.concatenate(labels)
        scores[split] = np.concatenate(probabilities, dtype=np.float64)
        if not np.isfinite(scores[split]).all():
            raise ValueError(f"UG-MoE returned non-finite {split} scores")
    return truth, scores, checkpoint


def validation_threshold(truth, scores):
    """Maximize validation macro-F1. Prefer the highest threshold on ties."""
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    last_tied = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    positives = truth.sum()
    negatives = len(truth) - positives
    if not positives or not negatives:
        raise ValueError("Threshold selection requires both classes in validation")
    tp = np.r_[0, np.cumsum(truth[order])[last_tied]]
    predicted = np.r_[0, last_tied + 1]
    fp = predicted - tp
    fn, tn = positives - tp, negatives - fp
    macro_f1 = tp / (2 * tp + fp + fn) + tn / (2 * tn + fp + fn)
    thresholds = np.r_[np.nextafter(np.float64(sorted_scores[0]), np.inf), sorted_scores[last_tied]]
    return float(thresholds[np.argmax(macro_f1)])


def compare(args):
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    manifest, digest = frozen_manifest(args.data_dir)
    ano, ano_run = match_ano_windows(args.ano_run, manifest, digest)
    configure_torch()
    truth, ug, checkpoint = ug_predictions(args.data_dir, args.ug_run, manifest, args.device, args.batch_size)
    scores = {"ug_moe2:full": ug, "AnoFusion (adapted)": ano}
    results, predictions = {}, {}
    for model, splits in scores.items():
        threshold = validation_threshold(truth["val"], splits["val"])
        predictions[model] = {split: (splits[split] >= threshold).astype(int) for split in EVAL_SPLITS}
        results[model] = {"threshold": threshold, **{
            split: compute_metrics(truth[split], predictions[model][split], splits[split])
            for split in EVAL_SPLITS
        }}
    summary = {
        "manifest_sha256": digest, "ug_checkpoint": str(checkpoint), "ug_seed": 42,
        "anofusion_run": str(args.ano_run), "anofusion_revision": ANO_REVISION, "anofusion_seed": ano_run["seed"],
        "ano_aggregation": "maximum over instances and scored 30-second bins within each 120-second window",
        "threshold_rule": "highest threshold maximizing validation macro-F1",
        "sizes": {split: len(y) for split, y in truth.items()}, "models": results,
    }
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("split", "window_id", "label", "ug_score", "ug_prediction", "ano_score", "ano_prediction"))
        for split in EVAL_SPLITS:
            rows = zip(manifest[f"{split}_ids"], truth[split], ug[split], predictions["ug_moe2:full"][split],
                       ano[split], predictions["AnoFusion (adapted)"][split], strict=True)
            writer.writerows((split, *row) for row in rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for model, result in results.items():
        test = result["test"]
        print(f"{model}: test precision={test['precision']:.4f} recall={test['recall']:.4f} F1={test['f1']:.4f}")
    print(f"Matched GAIA comparison saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ug-run", type=Path, required=True, help="Saved GAIA seed-42 results directory")
    parser.add_argument("--ano-run", type=Path, required=True, help="GAIA AnoFusion scores directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"))
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    compare(args)


if __name__ == "__main__":
    main()
