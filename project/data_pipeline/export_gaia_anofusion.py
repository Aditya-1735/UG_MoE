"""Export aligned 30-second channels on the frozen GAIA system windows."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory

import numpy as np

from .build_gaia import (
    BIN_SECONDS,
    DEFAULT_ARCHIVES,
    DEFAULT_OUTPUT,
    FORMAT,
    LOG_LEVELS,
    METRIC_CHANNELS,
    PARTITIONS,
    SERVICES,
    WINDOW_SECONDS,
    WINDOW_STEPS,
    local_epoch,
    read_logs,
)


CHANNELS = (*METRIC_CHANNELS, *(f"log_{level}" for level in LOG_LEVELS), "trace_latency")


def frozen_manifest(data_dir):
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    manifest_bytes = (data_dir / "split_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if (metadata["format"], metadata["bin_seconds"], metadata["window_seconds"]) != (
        FORMAT, BIN_SECONDS, WINDOW_SECONDS
    ) or tuple(metadata["services"]) != SERVICES or manifest["split"] != "gaia_chronological":
        raise ValueError("Expected the frozen GAIA MicroSS system-window dataset")
    return manifest, sha256(manifest_bytes).hexdigest()


def export(data_dir, archives, output):
    if output.exists():
        raise FileExistsError(output)
    manifest, digest = frozen_manifest(data_dir)
    with (data_dir / "chunks.pkl").open("rb") as stream:
        chunks = pickle.load(stream)
    ids = [uid for split in PARTITIONS for uid in manifest[f"{split}_ids"]]
    if not ids or len(ids) != len(chunks) or set(ids) != chunks.keys():
        raise ValueError("The manifest must assign each GAIA chunk exactly once")
    windows = [chunks[uid] for uid in ids]
    starts = np.array([window["t0"] for window in windows], dtype=np.int64)
    if (ids != [f"gaia-{start}" for start in starts] or np.any(starts % WINDOW_SECONDS)
            or np.any(np.diff(starts) < WINDOW_SECONDS)):
        raise ValueError("GAIA windows must be aligned, chronological, and disjoint")
    splits = np.repeat(PARTITIONS, [len(manifest[f"{split}_ids"]) for split in PARTITIONS])
    days = {day for split in PARTITIONS for day in manifest["dates"][split]}
    first_day = min(days)
    first_epoch = int(local_epoch(first_day))
    end_epoch = int(local_epoch(max(days))) + 86400
    print("Streaming business logs for 30-second severity counts", flush=True)
    counts = read_logs(archives, days, first_day, (end_epoch - first_epoch) // BIN_SECONDS)
    bins = ((starts - first_epoch) // BIN_SECONDS)[:, None] + np.arange(WINDOW_STEPS)
    counts = counts[bins]
    expected_logs = np.stack([window["logs"] for window in windows])
    if not np.allclose(np.log1p(counts.sum(axis=1)), expected_logs):
        raise ValueError("Re-aggregated GAIA log counts differ from the frozen windows")

    metrics = np.stack([window["metrics"] for window in windows]).transpose(0, 2, 1, 3)
    traces = np.stack([window["traces"] for window in windows]).transpose(0, 2, 1, 3)
    observations = np.concatenate((metrics, np.log1p(counts).astype(np.float32), traces), axis=-1)
    if observations.shape != (len(ids), WINDOW_STEPS, len(SERVICES), len(CHANNELS)):
        raise ValueError("Unexpected dimensions for GAIA observations")
    if not np.isfinite(observations).all():
        raise ValueError("GAIA observations contain a non-finite value")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".gaia-export-", dir=output.parent) as temp:
        stage = Path(temp) / "observations.npz"
        np.savez_compressed(stage, starts=starts, splits=splits, observations=observations,
                            services=SERVICES, channels=CHANNELS, manifest_sha256=digest)
        stage.rename(output)
    print(f"Exported {len(ids)} GAIA windows with {len(CHANNELS)} channels to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--archives", type=Path, default=DEFAULT_ARCHIVES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.data_dir, args.archives, args.output)


if __name__ == "__main__":
    main()
