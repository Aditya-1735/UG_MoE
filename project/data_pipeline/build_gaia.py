"""Build the audited July GAIA system-window benchmark from split ZIP streams."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta
import json
from pathlib import Path
import pickle
import re
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .audit_gaia import (
    BIN_SECONDS,
    DEFAULT_ARCHIVES,
    GAIA_CLOCK,
    METRIC_CHANNELS,
    SERVICES,
    archive_stream,
    select_metrics,
)


GAIA_ROOT = DEFAULT_ARCHIVES.parents[1]
DEFAULT_LABELS = GAIA_ROOT / "raw/micross_v1/run/run/run_table_2021-07.csv"
DEFAULT_OUTPUT = GAIA_ROOT / "processed/micross_v1"
FORMAT = "gaia_micross_v1"
REVISION = "5f1d772a7845dd93e4ab04dfc0ee725bf0ca933a"
WINDOW_SECONDS = 120
WINDOW_STEPS = WINDOW_SECONDS // BIN_SECONDS
DAY_BINS = 86400 // BIN_SECONDS
PARTITIONS = ("train", "val", "test")
LABELS = {0: "normal", 1: "memory", 2: "file_move"}
LOG_LEVELS = ("INFO", "WARNING", "ERROR", "DEBUG", "OTHER")
SERVICE_INDEX = {service: i for i, service in enumerate(SERVICES)}
CHANNEL_INDEX = {channel: i for i, channel in enumerate(METRIC_CHANNELS)}
START_PATTERN = re.compile(r"start\s+(?:at|with)\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)")
DURATION_PATTERN = re.compile(r"lasts?(?:\s+for)?\s+(\d+(?:\.\d+)?)\s+seconds")
LOG_TIMESTAMP = r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:[.,]\d+)?)"


def local_epoch(text):
    return datetime.fromisoformat(text.replace(",", ".")).replace(tzinfo=GAIA_CLOCK).timestamp()


def read_faults(path):
    confirmed, uncertain, unbounded_days, recorded_days = set(), [], [], set()
    july_end = local_epoch("2021-08-01")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            recorded_days.add(row["datetime"])
            message = row["message"]
            if "trigger an access permission denied exception" in message:
                emission = local_epoch(message.partition(" | ")[0])
                uncertain.append((emission - 3600, emission + 3600))
                continue
            if "[memory_anomalies]" in message:
                label = 1
            elif "file moving program" in message:
                label = 2
            elif "[cpu_anomalies]" in message:
                label = None
            else:
                continue
            start_match = START_PATTERN.search(message)
            duration_match = DURATION_PATTERN.search(message)
            if start_match is None or duration_match is None:
                raise ValueError(f"Unrecognized injection interval on {row['datetime']} for {row['service']}")
            start = local_epoch(start_match[1])
            duration = float(duration_match[1])
            if not np.isfinite(duration) or duration <= 0:
                raise ValueError(f"Invalid injection duration on {row['datetime']} for {row['service']}")
            end = start + duration
            if label is not None:
                confirmed.add((start, end, label, row["service"]))
            elif end >= july_end:
                unbounded_days.append(start_match[1][:10])
            else:
                uncertain.append((start, end))
    return sorted(confirmed), uncertain, recorded_days, min(unbounded_days, default="2021-08-01")


def select_partitions(audit, recorded_days, cutoff):
    if audit["rows_per_member_limit"] is not None or audit["bin_seconds"] != BIN_SECONDS:
        raise ValueError("GAIA build requires the full coverage audit on the 30-second grid")
    metric_days = defaultdict(set)
    for row in audit["modalities"]["metric"].values():
        metric_days[row["service"], row["channel"]].update(row["days"])
    expected = {(service, channel) for service in SERVICES for channel in METRIC_CHANNELS}
    coverage = [recorded_days, *(metric_days[key] for key in sorted(expected))]
    for modality in ("trace", "business"):
        by_service = {row["service"]: set(row["days"]) for row in audit["modalities"][modality].values()}
        coverage.extend(by_service[service] for service in SERVICES)
    days = sorted(day for day in set.intersection(*coverage) if "2021-07-01" <= day < cutoff)
    train_end, val_end = round(0.6 * len(days)), round(0.8 * len(days))
    partitions = dict(zip(PARTITIONS, (days[:train_end], days[train_end:val_end], days[val_end:])))
    if any(not dates for dates in partitions.values()):
        raise ValueError("Coverage is insufficient for a chronological train/val/test split")
    return partitions


def read_batches(archive, member, columns, strings=()):
    with archive_stream(archive, member) as stream:
        yield from pd.read_csv(stream, usecols=columns, dtype={col: "string" for col in strings}, chunksize=200_000)


def read_metrics(root, first_epoch, keep_bins):
    archive = root / "metric/metric_split.zip"
    with ZipFile(archive) as listing:
        selected = sorted(select_metrics(listing.namelist()))
    shape = (len(keep_bins), len(SERVICES), len(METRIC_CHANNELS))
    sums = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.int32)
    for member, service, details in selected:
        node, channel = SERVICE_INDEX[service], CHANNEL_INDEX[details["channel"]]
        for batch in read_batches(archive, member, ("timestamp", "value")):
            times = pd.to_numeric(batch["timestamp"], errors="coerce").to_numpy(dtype=np.float64)
            values = pd.to_numeric(batch["value"], errors="coerce").to_numpy(dtype=np.float64)
            valid = np.isfinite(times) & np.isfinite(values)
            bins = times[valid].astype(np.int64) // (BIN_SECONDS * 1000) - first_epoch // BIN_SECONDS
            usable = (bins >= 0) & (bins < len(keep_bins))
            bins, values = bins[usable], values[valid][usable]
            covered = keep_bins[bins]
            np.add.at(sums[:, node, channel], bins[covered], values[covered])
            np.add.at(counts[:, node, channel], bins[covered], 1)
    return np.divide(sums, counts, out=np.full(shape, np.nan, dtype=np.float32), where=counts > 0)


def datetime_bins(times, first_day):
    offsets = (times.to_numpy(dtype="datetime64[s]") - np.datetime64(first_day, "s"))
    return offsets.astype(np.int64) // BIN_SECONDS


def read_traces(root, days, first_day, num_bins, graph_start, graph_end):
    archive = root / "trace/trace_split.zip"
    sums = np.zeros((num_bins, len(SERVICES)), dtype=np.float64)
    counts = np.zeros_like(sums, dtype=np.int32)
    graph_rows = []
    cols = ("start_time", "end_time", "trace_id", "span_id", "parent_id")
    for service, node in SERVICE_INDEX.items():
        member = f"trace/trace_table_{service}_2021-07.csv"
        print(f"  traces {service}", flush=True)
        for batch in read_batches(archive, member, cols, cols):
            selected = batch.loc[batch["start_time"].str.slice(0, 10).isin(days)]
            if selected.empty:
                continue
            starts = pd.to_datetime(selected["start_time"], format="ISO8601", errors="coerce")
            ends = pd.to_datetime(selected["end_time"], format="ISO8601", errors="coerce")
            latency = (ends - starts).dt.total_seconds() * 1000
            valid = latency.notna() & (latency >= 0)
            bins = datetime_bins(starts[valid], first_day)
            np.add.at(sums[:, node], bins, latency[valid].to_numpy(dtype=np.float64))
            np.add.at(counts[:, node], bins, 1)
            in_graph_sample = (starts >= graph_start) & (starts < graph_end)
            spans = selected.loc[in_graph_sample, ["trace_id", "span_id", "parent_id"]].dropna(subset=["trace_id", "span_id"])
            graph_rows.extend((trace, span, parent, node) for trace, span, parent in spans.itertuples(index=False, name=None))
    latent = np.divide(sums, counts, out=np.zeros_like(sums, dtype=np.float32), where=counts > 0)
    return np.log1p(latent)[..., None], graph_rows


def service_edges(graph_rows):
    nodes = {(trace, span): node for trace, span, _, node in graph_rows}
    calls = Counter()
    for trace, _, parent, child in graph_rows:
        caller = nodes.get((trace, parent))
        if caller is not None and caller != child:
            calls[caller, child] += 1
    if not calls:
        raise ValueError("Training trace sample did not contain any matched cross-service parent spans")
    edges = sorted(set(calls) | {(node, node) for node in range(len(SERVICES))})
    return [list(column) for column in zip(*edges)], calls


def read_logs(root, days, first_day, num_bins):
    archive = root / "business/business_split.zip"
    counts = np.zeros((num_bins, len(SERVICES), len(LOG_LEVELS)), dtype=np.int32)
    level_codes = {level: i for i, level in enumerate(LOG_LEVELS)}
    level_codes.update(WARN=level_codes["WARNING"], CRITICAL=level_codes["ERROR"], FATAL=level_codes["ERROR"])
    for service, node in SERVICE_INDEX.items():
        member = f"business/business_table_{service}_2021-07.csv"
        print(f"  logs {service}", flush=True)
        for batch in read_batches(archive, member, ("message",), ("message",)):
            messages = batch["message"]
            messages = messages[messages.str.slice(0, 10).isin(days)].str.slice(0, 80)
            if messages.empty:
                continue
            timestamps = messages.str.extract(LOG_TIMESTAMP, expand=False).str.replace(",", ".", regex=False)
            starts = pd.to_datetime(timestamps, format="ISO8601", errors="coerce")
            valid = starts.notna()
            bins = datetime_bins(starts[valid], first_day)
            levels = messages[valid].str.extract(r"\|\s*([A-Z]+)\s*\|", expand=False)
            codes = levels.map(level_codes).fillna(level_codes["OTHER"]).to_numpy(dtype=np.int64)
            np.add.at(counts[:, node, :], (bins, codes), 1)
    return counts


def assemble_windows(partitions, first_epoch, metrics, traces, logs, confirmed, uncertain):
    boundaries = [local_epoch(partitions[split][0]) for split in PARTITIONS[1:]]
    crosses = [event for event in confirmed if any(event[0] < cut < event[1] for cut in boundaries)]
    excluded = uncertain + [(start, end) for start, end, _, _ in crosses]
    known = [event for event in confirmed if event not in crosses]
    chunks = {}
    ids = {split: [] for split in PARTITIONS}
    dropped = {split: Counter() for split in PARTITIONS}
    class_counts = {split: Counter() for split in PARTITIONS}
    for split, dates in partitions.items():
        for day in dates:
            start_day = int(local_epoch(day))
            day_index = (start_day - first_epoch) // BIN_SECONDS
            day_faults = [event for event in known if event[0] < start_day + 86400 and event[1] > start_day]
            for offset in range(0, DAY_BINS, WINDOW_STEPS):
                start = start_day + offset * BIN_SECONDS
                end = start + WINDOW_SECONDS
                if any(start < cut + WINDOW_SECONDS and end > cut - WINDOW_SECONDS for cut in boundaries):
                    dropped[split]["split_gap"] += 1
                    continue
                if any(start < high and end > low for low, high in excluded):
                    dropped[split]["uncertain_fault"] += 1
                    continue
                bins = slice(day_index + offset, day_index + offset + WINDOW_STEPS)
                if not np.isfinite(metrics[bins]).all():
                    dropped[split]["missing_metrics"] += 1
                    continue
                hit = max(day_faults, default=None, key=lambda event: min(end, event[1]) - max(start, event[0]))
                label = hit[2] if hit is not None and min(end, hit[1]) - max(start, hit[0]) >= 1 else 0
                window_id = f"gaia-{start}"
                chunks[window_id] = {
                    "metrics": metrics[bins].transpose(1, 0, 2).copy(),
                    "traces": traces[bins].transpose(1, 0, 2).copy(),
                    "logs": np.log1p(logs[bins].sum(axis=0)).astype(np.float32),
                    "t0": start,
                    "culprit": label,
                }
                ids[split].append(window_id)
                class_counts[split][LABELS[label]] += 1
        if not class_counts[split]["normal"] or not any(chunks[key]["culprit"] for key in ids[split]):
            raise ValueError(f"GAIA {split} needs both classes after coverage and uncertainty exclusions")
    return chunks, ids, class_counts, dropped, len(crosses)


def build(coverage, archives, labels, output):
    if output.exists():
        raise FileExistsError(f"GAIA output already exists: {output}")
    audit = json.loads(coverage.read_text(encoding="utf-8"))
    confirmed, uncertain, recorded_days, cutoff = read_faults(labels)
    partitions = select_partitions(audit, recorded_days, cutoff)
    print(f"GAIA UTC+8 days by split: {partitions}", flush=True)
    first_day, last_day = partitions["train"][0], partitions["test"][-1]
    first_epoch = int(local_epoch(first_day))
    end_epoch = int(local_epoch(last_day)) + 86400
    num_bins = (end_epoch - first_epoch) // BIN_SECONDS
    days = {day for dates in partitions.values() for day in dates}
    keep_bins = np.zeros(num_bins, dtype=bool)
    for day in days:
        offset = (int(local_epoch(day)) - first_epoch) // BIN_SECONDS
        keep_bins[offset:offset + DAY_BINS] = True

    graph_start = max(datetime.fromisoformat(row["first"]) for row in audit["modalities"]["trace"].values())
    if graph_start.date().isoformat() not in partitions["train"]:
        raise ValueError("The first shared trace hour falls outside the selected training dates")
    graph_end = min(graph_start + timedelta(hours=1), graph_start.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))

    print("Aggregating GAIA metrics", flush=True)
    metrics = read_metrics(archives, first_epoch, keep_bins)
    print("Streaming GAIA traces", flush=True)
    traces, graph_rows = read_traces(archives, days, first_day, num_bins,
                                    graph_start.replace(tzinfo=None), graph_end.replace(tzinfo=None))
    edges, calls = service_edges(graph_rows)
    print("Streaming GAIA business logs", flush=True)
    logs = read_logs(archives, days, first_day, num_bins)
    chunks, ids, class_counts, dropped, crossing = assemble_windows(partitions, first_epoch, metrics, traces, logs,
                                                                     confirmed, uncertain)
    metadata = {
        "format": FORMAT, "system": "GAIA MicroSS", "source_revision": REVISION,
        "services": SERVICES, "node_num": len(SERVICES), "edges": edges,
        "metric_num": len(METRIC_CHANNELS), "metric_channels": METRIC_CHANNELS,
        "trace_dim": 1, "event_num": len(LOG_LEVELS), "log_channels": LOG_LEVELS,
        "label_map": LABELS, "bin_seconds": BIN_SECONDS,
        "window_seconds": WINDOW_SECONDS, "stride_seconds": WINDOW_SECONDS, "chunk_lenth": WINDOW_STEPS,
    }
    manifest = {"split": "gaia_chronological", "dates": partitions, "time_zone": "+08:00",
                "source_revision": REVISION, "split_gap_seconds_each_side": WINDOW_SECONDS,
                **{f"{split}_ids": windows for split, windows in ids.items()}}
    quality = {
        "coverage_audit": str(coverage), "source_labels": str(labels),
        "excluded_dates_from": cutoff, "confirmed_fault_records": len(confirmed),
        "uncertain_fault_intervals": len(uncertain), "cross_split_faults_excluded": crossing,
        "class_counts": class_counts, "dropped_windows": dropped,
        "graph_trace_sample": [graph_start.isoformat(), graph_end.isoformat()],
        "cross_service_edges": {f"{SERVICES[a]}->{SERVICES[b]}": n for (a, b), n in sorted(calls.items())},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".gaia-build-", dir=output.parent) as directory:
        stage = Path(directory) / "complete"
        stage.mkdir()
        with (stage / "chunks.pkl").open("wb") as stream:
            pickle.dump(chunks, stream, protocol=pickle.HIGHEST_PROTOCOL)
        for name, value in (("metadata", metadata), ("quality", quality), ("split_manifest", manifest)):
            (stage / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        stage.rename(output)
    print(f"GAIA windows by split and class: {class_counts}")
    print(f"GAIA dataset saved to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True, help="Full GAIA coverage audit JSON")
    parser.add_argument("--archives", type=Path, default=DEFAULT_ARCHIVES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    csv.field_size_limit(sys.maxsize)
    build(args.coverage, args.archives, args.labels, args.output)


if __name__ == "__main__":
    main()
