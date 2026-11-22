"""Audit July GAIA coverage by streaming verified split ZIP members."""

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
from datetime import datetime, timedelta, timezone
from io import TextIOWrapper
from itertools import islice
import json
from pathlib import Path
import re
import subprocess
import sys
from zipfile import ZipFile


DEFAULT_ARCHIVES = Path(__file__).resolve().parents[1] / "datasets/gaia/archives/micross_v1"
GAIA_CLOCK = timezone(timedelta(hours=8))
BIN_SECONDS = 30
SERVICES = tuple(f"{name}service{i}" for name in ("db", "log", "mob", "redis", "web") for i in (1, 2))
METRIC_CHANNELS = (
    "docker_cpu_system_norm_pct",
    "docker_cpu_total_norm_pct",
    "docker_cpu_user_norm_pct",
    "docker_memory_usage_total",
    "docker_memory_rss_total",
    "docker_network_in_bytes",
    "docker_network_out_bytes",
)
METRIC_MEMBER = re.compile(r"metric/([^_]+)_([\d.]+)_(.+)_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.csv")


@contextmanager
def archive_stream(archive, member, partial=False):
    """Open one CSV stream from a multipart archive without expanding it on disk."""
    proc = subprocess.Popen(
        ["7z", "x", "-so", str(archive), member],
        stdout=subprocess.PIPE,
    )
    stream = TextIOWrapper(proc.stdout, encoding="utf-8-sig", newline="")
    try:
        yield stream
    finally:
        stream.close()
        if partial and proc.poll() is None:
            proc.terminate()
        status = proc.wait()
    if not partial and status:
        raise RuntimeError(f"7-Zip failed for {member} in {archive} with exit code {status}")


@contextmanager
def archive_csv(archive, member, row_limit=None):
    with archive_stream(archive, member, partial=row_limit is not None) as stream:
        reader = csv.DictReader(stream)
        yield reader if row_limit is None else islice(reader, row_limit)


def metric_time(row):
    return datetime.fromtimestamp(int(row["timestamp"]) / 1000, GAIA_CLOCK)


def trace_time(row):
    return datetime.fromisoformat(row["start_time"]).replace(tzinfo=GAIA_CLOCK)


def business_time(row):
    stamp = (row["message"] or "").partition(" | ")[0].replace(",", ".")
    return datetime.fromisoformat(stamp).replace(tzinfo=GAIA_CLOCK)


def scan_member(archive, member, service, get_time, service_column, row_limit):
    daily_rows = Counter()
    daily_bins = defaultdict(set)
    issues = Counter()
    first = last = previous = None
    row_count = 0
    with archive_csv(archive, member, row_limit) as rows:
        for row in rows:
            row_count += 1
            if service_column and row[service_column] != service:
                issues["service_mismatch"] += 1
            try:
                stamp = get_time(row)
            except (TypeError, ValueError, OverflowError):
                issues["invalid_timestamp"] += 1
                continue
            day = stamp.date().isoformat()
            seconds = stamp.hour * 3600 + stamp.minute * 60 + stamp.second
            daily_rows[day] += 1
            daily_bins[day].add(seconds // BIN_SECONDS)
            if previous is not None and stamp < previous:
                issues["out_of_order"] += 1
            previous = stamp
            first = stamp if first is None else min(first, stamp)
            last = stamp if last is None else max(last, stamp)
    return {
        "service": service,
        "rows": row_count,
        "first": first.isoformat() if first is not None else None,
        "last": last.isoformat() if last is not None else None,
        "days": {day: {"rows": daily_rows[day], "occupied_bins": len(daily_bins[day])} for day in sorted(daily_rows)},
        "issues": dict(issues),
    }


def select_metrics(members):
    selected = []
    for member in members:
        match = METRIC_MEMBER.fullmatch(member)
        if match is None:
            continue
        service, host, channel, _, _ = match.groups()
        if service in SERVICES and channel in METRIC_CHANNELS:
            selected.append((member, service, {"host": host, "channel": channel}))
    seen = {(service, details["channel"]) for _, service, details in selected}
    missing = set((service, channel) for service in SERVICES for channel in METRIC_CHANNELS) - seen
    if missing:
        raise ValueError(f"Missing metric channels: {sorted(missing)}")
    return selected


def audit(archives, row_limit):
    sources = (("metric", metric_time, None), ("trace", trace_time, "service_name"),
               ("business", business_time, "service"))
    report = {"time_zone": "+08:00", "bin_seconds": BIN_SECONDS, "rows_per_member_limit": row_limit, "modalities": {}}
    for modality, get_time, service_column in sources:
        archive = archives / modality / f"{modality}_split.zip"
        with ZipFile(archive) as listing:
            members = set(listing.namelist())
        if modality == "metric":
            selected = select_metrics(members)
        else:
            selected = [(f"{modality}/{modality}_table_{service}_2021-07.csv", service, {}) for service in SERVICES]
            missing = {member for member, _, _ in selected} - members
            if missing:
                raise ValueError(f"Missing {modality} members: {sorted(missing)}")

        print(f"Auditing {modality}: {len(selected)} members", flush=True)
        results = {}
        for member, service, details in sorted(selected):
            result = scan_member(archive, member, service, get_time, service_column, row_limit)
            results[member] = {**details, **result}
            if modality != "metric":
                print(f"  {service}: {result['rows']:,} rows, {len(result['days'])} days", flush=True)
        report["modalities"][modality] = results
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, default=DEFAULT_ARCHIVES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-per-member", type=int, help="Read only a prefix of each member for a smoke check")
    args = parser.parse_args()
    if args.rows_per_member is not None and args.rows_per_member < 1:
        parser.error("--rows-per-member must be positive")
    csv.field_size_limit(sys.maxsize)
    report = audit(args.archives, args.rows_per_member)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"GAIA coverage audit saved to {args.output}")


if __name__ == "__main__":
    main()
