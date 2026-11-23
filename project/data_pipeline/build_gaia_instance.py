"""Build the seed-42 mobservice2 pilot inputs on a one-minute grid in UTC+8."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from .audit_gaia import DEFAULT_ARCHIVES, METRIC_CHANNELS, select_metrics
from .build_gaia import LOG_TIMESTAMP, REVISION, local_epoch, read_batches
from experiments.run_gaia_anofusion import ANO_REVISION

SERVICE = "mobservice2"
START = int(local_epoch("2021-07-01"))
MINUTES = 14 * 24 * 60
SECONDS = 60
TRACE_CHANNELS = ("latency_mean_ms", "latency_median_ms", "latency_range_ms", "latency_sd_ms")
HISTORY = 60


def read_minute_metrics(archives):
    archive = archives / "metric/metric_split.zip"
    sums = np.zeros((MINUTES, len(METRIC_CHANNELS)))
    counts = np.zeros(sums.shape, dtype=np.int32)
    with ZipFile(archive) as index:
        members = select_metrics(index.namelist())
    for member, service, info in members:
        if service != SERVICE or not member.endswith("2021-07-01_2021-07-15.csv"):
            continue
        channel = METRIC_CHANNELS.index(info["channel"])
        for frame in read_batches(archive, member, ("timestamp", "value")):
            stamp = pd.to_numeric(frame.timestamp, errors="coerce").to_numpy(dtype=float)
            value = pd.to_numeric(frame.value, errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(stamp) & np.isfinite(value)
            minute = ((stamp[ok] // 1000 - START) // SECONDS).astype(int)
            keep = (minute >= 0) & (minute < MINUTES)
            np.add.at(sums[:, channel], minute[keep], value[ok][keep])
            np.add.at(counts[:, channel], minute[keep], 1)
    return np.divide(sums, counts, where=counts > 0, out=np.full_like(sums, np.nan))


def read_minute_traces(archives):
    archive = archives / "trace/trace_split.zip"
    member = f"trace/trace_table_{SERVICE}_2021-07.csv"
    parts = []
    for frame in read_batches(archive, member, ("start_time", "end_time")):
        start = pd.to_datetime(frame.start_time, errors="coerce", format="ISO8601")
        end = pd.to_datetime(frame.end_time, errors="coerce", format="ISO8601")
        latency = (end - start).dt.total_seconds() * 1000
        minute = (end - pd.Timestamp("2021-07-01")).dt.total_seconds() // SECONDS
        valid = minute.between(0, MINUTES - 1) & latency.ge(0) & np.isfinite(latency)
        parts.append(pd.DataFrame(dict(minute=minute[valid].astype(int), latency=latency[valid])))
    grouped = pd.concat(parts).groupby("minute").latency
    stats = pd.concat([grouped.mean(), grouped.median(), grouped.max() - grouped.min(), grouped.std(ddof=0)], axis=1)
    return stats.reindex(range(MINUTES)).fillna(0).to_numpy(), stats.index.to_numpy()


def log_messages(archives):
    archive = archives / "business/business_split.zip"
    member = f"business/business_table_{SERVICE}_2021-07.csv"
    for frame in read_batches(archive, member, ("message",), ("message",)):
        msg = frame.message
        msg = msg[msg.str.slice(0, 10).between("2021-07-01", "2021-07-14")]
        if msg.empty:
            continue
        clock = msg.str.extract(LOG_TIMESTAMP, expand=False).str.replace(",", ".", regex=False)
        minutes = (pd.to_datetime(clock, format="ISO8601", errors="coerce") - pd.Timestamp("2021-07-01")).dt.total_seconds() // SECONDS
        keep = minutes.between(0, MINUTES - 1)
        yield minutes[keep].astype(int).to_numpy(), msg[keep]


def message_body(message):
    # Remove the volatile metadata prefix, retain the severity and actual message.
    fields = message.rstrip().split(" | ", 6)
    if len(fields) == 7:
        return fields[1] + " " + fields[-1]
    return message.partition(" | ")[2].strip() or message.strip()


def public_labels(source):
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != ANO_REVISION:
        raise ValueError("Unexpected upstream label revision")
    relative = Path("labeled_service") / f"{SERVICE}.csv"
    subprocess.run(["git", "-C", str(source), "diff", "--exit-code", "HEAD", "--", str(relative)], check=True)
    label_path = source / relative
    frame = pd.read_csv(label_path)
    part = frame.loc[frame.timestamp.between(START, START + MINUTES * SECONDS - 1)].sort_values("timestamp")
    np.testing.assert_array_equal(part.timestamp.to_numpy(), START + np.arange(MINUTES * SECONDS))
    if not part.label.isin([0, 1]).all():
        raise ValueError("Unexpected public label value")
    return part.label.to_numpy(dtype=np.uint8).reshape(-1, SECONDS).sum(axis=1), sha256(label_path.read_bytes()).hexdigest()


def build(archives, coverage, source, output):
    if output.exists():
        raise FileExistsError(output)
    labels, labels_sha = public_labels(source)
    print("Aggregating per-instance minute metrics and completed spans", flush=True)
    metrics = read_minute_metrics(archives)
    traces, trace_bins = read_minute_traces(archives)
    audit = json.loads(coverage.read_text())
    if audit["rows_per_member_limit"] is not None:
        raise ValueError("Expected full archive coverage audit")
    record = next(x for x in audit["modalities"]["business"].values() if x["service"] == SERVICE)
    days = pd.date_range("2021-07-01", periods=14)
    trace_days = set(trace_bins // 1440)
    available = [i in trace_days and day.strftime("%Y-%m-%d") in record["days"]
                 for i, day in enumerate(days)]
    eligible = np.isfinite(metrics).all(axis=1) & np.repeat(available, 1440)
    minutes = np.flatnonzero(eligible)
    outer = int(0.6 * len(minutes))
    inner = int(0.8 * outer)
    if min(inner, outer - inner, len(minutes) - outer) <= HISTORY:
        raise ValueError("Insufficient coverage for the split and lookback")
    splits = np.full(MINUTES, "unused", dtype="U6")
    for name, block in zip(("train", "val", "test"), np.split(minutes, [inner, outer]), strict=True):
        splits[block] = name
    # Exclude an entire positive run crossing a partition cut.
    for cut in (minutes[inner], minutes[outer]):
        if labels[cut - 1] and labels[cut]:
            left, right = cut - 1, cut
            while left > 0 and labels[left - 1]:
                left -= 1
            while right + 1 < MINUTES and labels[right + 1]:
                right += 1
            splits[left:right + 1] = "unused"
    minutes = np.flatnonzero(splits != "unused")
    print("Learning Drain templates on training minutes", flush=True)
    config = TemplateMinerConfig()
    config.drain_max_clusters = 4096
    miner = TemplateMiner(config=config)
    for indices, messages in log_messages(archives):
        for message in messages[splits[indices] == "train"]:
            miner.add_log_message(message_body(message))
    clusters = sorted(miner.drain.clusters, key=lambda c: (-c.size, c.cluster_id))[:32]
    if not clusters:
        raise ValueError("No log templates learned")
    columns = {c.cluster_id: i for i, c in enumerate(clusters)}
    log_counts = np.zeros((MINUTES, len(columns) + 1), dtype=np.int32)
    print("Counting frozen Drain templates on the minute grid", flush=True)
    for indices, messages in log_messages(archives):
        use = splits[indices] != "unused"
        for minute, message in zip(indices[use], messages[use], strict=True):
            match = miner.match(message_body(message))
            col = columns.get(match.cluster_id, len(columns)) if match else len(columns)
            log_counts[minute, col] += 1
    times = START + minutes * SECONDS
    starts = np.arange(len(times))
    breaks = np.r_[True, (np.diff(times) != SECONDS) | (splits[minutes][1:] != splits[minutes][:-1])]
    first = np.maximum.accumulate(np.where(breaks, starts, 0))
    target = starts[starts - first >= HISTORY]
    support = {}
    for split in ("train", "val", "test"):
        part = target[splits[minutes[target]] == split]
        positives = int((labels[minutes[part]] > 0).sum())
        if positives == 0 or positives == len(part):
            raise ValueError(f"{split} needs positive and negative targets after gap resets")
        support[split] = dict(minutes=int((splits == split).sum()), targets=len(part), positives=positives)
    channels = [*METRIC_CHANNELS, *(f"drain_{c.cluster_id}" for c in clusters), "other_log", *TRACE_CHANNELS]
    features = np.column_stack((metrics[minutes], log_counts[minutes], traces[minutes])).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite pilot features")
    summary = dict(service=SERVICE, start=START, end_exclusive=START + MINUTES * SECONDS,
                   gaia_revision=REVISION, drain3_version=version("drain3"),
                   drain_max_clusters=config.drain_max_clusters,
                   coverage_sha256=sha256(coverage.read_bytes()).hexdigest(),
                   eligible_minutes_before_purge=int(eligible.sum()),
                   purged_minutes=int(eligible.sum() - len(minutes)),
                   trace_and_log_dates=[str(day.date()) for day, ok in zip(days, available) if ok],
                   source_revision=ANO_REVISION, label_sha256=labels_sha, channels=channels,
                   templates={c.cluster_id: c.get_template() for c in clusters}, splits=support,
                   first_val_timestamp=int(times[splits[minutes] == "val"][0]),
                   first_test_timestamp=int(times[splits[minutes] == "test"][0]), history=HISTORY)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / "data"
        stage.mkdir()
        np.savez_compressed(stage / "minutes.npz", timestamps=times, splits=splits[minutes],
                            features=features, positive_seconds=labels[minutes], targets=target, channels=channels)
        (stage / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
        stage.rename(output)
    print(f"Pilot dataset saved to {output}. Forecast target support: {support}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, default=DEFAULT_ARCHIVES)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    flags = parser.parse_args()
    build(flags.archives, flags.coverage, flags.source, flags.output)
