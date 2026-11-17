#!/usr/bin/env python3
"""
Build windowed multimodal chunks from the raw Eadro runs (after data_pipeline.fetch_eadro).

    python -m data_pipeline.build_eadro --system SN                  # stride 1 (SN is small)
    python -m data_pipeline.build_eadro --system TT --stride 5       # TT runs are ~4.5x longer

Per run (raw/<SYS>/{data,no fault}/<SYS>.<run>/ + <SYS>.fault-<run>.json):
  * time grid   : integer seconds over the span covered by every service's metric CSV
  * metrics     : [N, T, 7] raw container metrics, gaps forward/back-filled. NOT z-scored here; the
                  experiment loader normalizes with train-split statistics only (no leakage).
  * traces      : [N, T] log1p(mean span latency in ms) of the spans served by each service in each
                  second (0 when a service has no span in that second, as in Eadro). Span clocks are
                  offset by whole hours from the metric clock; the offset is detected per run (the
                  shift that puts most spans inside the run window) and out-of-window spans are
                  dropped. Per-window `trace_frac` = fraction of seconds with any span at all
                  (TT's traces only cover part of each run).
  * logs        : Drain3 templates (numbers / ids pre-masked); per-window, per-service counts of the
                  top-K templates (index 0 = all other templates), log1p. Same per-run clock-offset
                  detection as spans.
  * labels      : a window is anomalous if it overlaps an injected fault by >= --threshold seconds
                  (Eadro's rule); culprit = 0 normal, 1 cpu_load, 2 network_delay, 3 network_loss;
                  the faulty service and fault type are kept as well.
Windows have length --window (default 10, as MUAD/Eadro) and stride --stride. Every chunk records
its run and start time, so experiments split by run (experiments/common.py) and windows never
straddle splits.

Output: <root>/processed/<SYS>_w<window>_s<stride>/{chunks.pkl, metadata.json, quality.json}.
Note: Drain3 templates are mined on the logs of all runs (unsupervised, label-free); the per-run
clock offsets are also estimated label-free. Both are documented in quality.json.
"""
import argparse
import json
import pickle
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_pipeline.systems import FAULT_TYPES, METRIC_NAMES, fault_service, system_info  # noqa: E402

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "datasets" / "eadro"
OFFSET_HOURS = range(-14, 15)

# ---------------------------------------------------------------------------- run discovery


def discover_runs(raw: Path, system: str) -> List[Dict]:
    runs = []
    for kind, sub in (("fault", "data"), ("normal", "no fault")):
        for fault_file in sorted((raw / sub).glob(f"{system}.fault-*.json")):
            run_id = fault_file.name[len(f"{system}.fault-"): -len(".json")]
            run_dir = raw / sub / f"{system}.{run_id}"
            if not run_dir.is_dir():
                raise FileNotFoundError(f"{fault_file.name} has no run directory {run_dir}")
            runs.append({"name": run_id, "kind": kind, "dir": run_dir, "fault_file": fault_file})
    if not runs:
        raise FileNotFoundError(f"No runs found under {raw}; run data_pipeline.fetch_eadro first")
    return runs


# ---------------------------------------------------------------------------- metrics


def load_metrics(run_dir: Path, services: List[str]) -> Tuple[int, np.ndarray]:
    tables = {}
    for s in services:
        path = run_dir / "metrics" / f"{s}.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing metric file {path}")
        arr = np.genfromtxt(path, delimiter=",", names=True)
        cols = arr.dtype.names
        if list(cols[1:]) != METRIC_NAMES:
            raise RuntimeError(f"{path}: unexpected columns {cols}")
        tables[s] = (arr["timestamp"].astype(np.int64), np.stack([arr[c] for c in METRIC_NAMES], 1))
    t0 = max(int(ts.min()) for ts, _ in tables.values())
    t1 = min(int(ts.max()) for ts, _ in tables.values())
    T = t1 - t0 + 1
    out = np.full((len(services), T, len(METRIC_NAMES)), np.nan)
    for i, s in enumerate(services):
        ts, vals = tables[s]
        keep = (ts >= t0) & (ts <= t1)
        out[i, ts[keep] - t0] = vals[keep]
    for i in range(len(services)):  # forward fill, then back fill, then 0
        for k in range(out.shape[2]):
            col = out[i, :, k]
            idx = np.where(~np.isnan(col), np.arange(T), 0)
            np.maximum.accumulate(idx, out=idx)
            col = col[idx]
            first = np.argmax(~np.isnan(col)) if (~np.isnan(col)).any() else T
            col[:first] = col[first] if first < T else 0.0
            out[i, :, k] = np.nan_to_num(col)
    return t0, out


# ---------------------------------------------------------------------------- clock offsets


def best_offset(ts: np.ndarray, t0: int, T: int) -> Tuple[int, float]:
    """Whole-hour shift that maximizes the fraction of event times inside [t0, t0 + T)."""
    if len(ts) == 0:
        return 0, 0.0
    best = max(OFFSET_HOURS, key=lambda h: np.mean((ts - h * 3600 >= t0) & (ts - h * 3600 < t0 + T)))
    frac = float(np.mean((ts - best * 3600 >= t0) & (ts - best * 3600 < t0 + T)))
    return best, frac


# ---------------------------------------------------------------------------- traces


def load_traces(run_dir: Path, services: List[str], t0: int, T: int):
    sid = {s: i for i, s in enumerate(services)}
    with open(run_dir / "spans.json") as f:
        traces = json.load(f)
    svc, ts, dur = [], [], []
    unknown = Counter()
    for tr in traces:
        procs = {k: v.get("serviceName") for k, v in (tr.get("processes") or {}).items()}
        for sp in tr.get("spans", []):
            name = procs.get(sp.get("processID"))
            if name not in sid:
                unknown[name] += 1
                continue
            svc.append(sid[name])
            ts.append(sp["startTime"] / 1e6)
            dur.append(sp["duration"] / 1e3)  # microseconds -> ms
    del traces
    svc, ts, dur = np.asarray(svc, int), np.asarray(ts, float), np.asarray(dur, float)
    off, frac = best_offset(ts, t0, T)
    sec = np.floor(ts - off * 3600 - t0).astype(np.int64)
    keep = (sec >= 0) & (sec < T)
    svc, sec, dur = svc[keep], sec[keep], dur[keep]
    flat = svc * T + sec
    n = np.bincount(flat, minlength=len(services) * T).reshape(len(services), T)
    tot = np.bincount(flat, weights=dur, minlength=len(services) * T).reshape(len(services), T)
    mean_ms = np.divide(tot, n, out=np.zeros_like(tot), where=n > 0)
    any_span = n.sum(0) > 0
    info = {"spans_total": int(len(ts)), "spans_kept": int(keep.sum()), "offset_hours": int(off),
            "inside_frac": round(frac, 4), "seconds_with_spans_frac": round(float(any_span.mean()), 4),
            "unknown_services": {str(k): v for k, v in unknown.most_common(5)}}
    return np.log1p(mean_ms), any_span, info


# ---------------------------------------------------------------------------- logs

_TS_PATTERNS = [
    (re.compile(r"^\[(\d{4}-[A-Za-z]{3}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\]\s*"), "%Y-%b-%d %H:%M:%S.%f"),  # SN
    (re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\s*"), "%Y-%m-%d %H:%M:%S.%f"),          # TT
]
_MASKS = [
    (re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"), "<UUID>"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b"), "<IP>"),
    (re.compile(r"\b(?:0x)?[0-9a-fA-F]{12,}\b"), "<HEX>"),
    (re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?"), "<NUM>"),
]


def parse_log_line(line: str):
    for pat, fmt in _TS_PATTERNS:
        m = pat.match(line)
        if m:
            dt = datetime.strptime(m.group(1)[:26], fmt).replace(tzinfo=timezone.utc)
            msg = line[m.end():]
            for p, rep in _MASKS:
                msg = p.sub(rep, msg)
            return dt.timestamp(), msg.strip()
    return None, None


def load_log_events(run_dir: Path, services: List[str]):
    sid = {s: i for i, s in enumerate(services)}
    with open(run_dir / "logs.json") as f:
        logs = json.load(f)
    svc, ts, msgs, unparsed, unknown = [], [], [], 0, Counter()
    for name, lines in logs.items():
        if name not in sid:
            unknown[name] += len(lines)
            continue
        for line in lines:
            t, msg = parse_log_line(line)
            if t is None:
                unparsed += 1
                continue
            svc.append(sid[name]); ts.append(t); msgs.append(msg)
    return np.asarray(svc, int), np.asarray(ts, float), msgs, unparsed, dict(unknown)


def mine_templates(all_msgs: List[str], top_k: int) -> Tuple[List[int], List[str]]:
    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig
    cfg = TemplateMinerConfig()
    cfg.profiling_enabled = False
    miner = TemplateMiner(config=cfg)
    cluster_ids = [miner.add_log_message(m)["cluster_id"] for m in all_msgs]
    templates = {c.cluster_id: c.get_template() for c in miner.drain.clusters}
    top = [cid for cid, _ in Counter(cluster_ids).most_common(top_k)]
    index = {cid: i + 1 for i, cid in enumerate(top)}            # 0 = every other template
    return [index.get(c, 0) for c in cluster_ids], ["<OTHER>"] + [templates[c] for c in top]


# ---------------------------------------------------------------------------- windows + labels


def fault_intervals(fault_json: Dict, services: List[str], t0: int):
    out = []
    for f in fault_json.get("faults", []):
        if f["fault"] not in FAULT_TYPES:
            raise ValueError(f"unknown fault type {f['fault']!r}")
        s = int(np.floor(f["start"] - t0))
        out.append((s, s + int(f["duration"]) - 1, f["fault"], fault_service(f["name"], services)))
    return out


def label_window(s: int, e: int, faults, threshold: int):
    best = None
    for fs, fe, ftype, fsvc in faults:
        ov = min(e, fe) - max(s, fs) + 1
        if ov >= threshold and (best is None or ov > best[0]):
            best = (ov, ftype, fsvc)
    if best is None:
        return 0, None, None
    return FAULT_TYPES.index(best[1]) + 1, best[1], best[2]


# ---------------------------------------------------------------------------- main


def build(system: str, root: Path, window: int, stride: int, threshold: int, top_k: int) -> Path:
    raw = root / "raw" / system
    sysinfo = system_info(system)
    services = sysinfo["services"]
    N = len(services)
    runs = discover_runs(raw, system)
    print(f"[{system}] {len(runs)} runs ({sum(r['kind'] == 'fault' for r in runs)} fault, "
          f"{sum(r['kind'] == 'normal' for r in runs)} normal), {N} services")

    # pass 1: per-run metrics, traces, raw log events
    per_run, all_msgs, msg_owner = [], [], []
    for ri, run in enumerate(runs):
        t0, metrics = load_metrics(run["dir"], services)
        T = metrics.shape[1]
        lat, any_span, trace_info = load_traces(run["dir"], services, t0, T)
        lsvc, lts, lmsg, unparsed, unknown_log = load_log_events(run["dir"], services)
        loff, lfrac = best_offset(lts, t0, T)
        lsec = np.floor(lts - loff * 3600 - t0).astype(np.int64)
        keep = (lsec >= 0) & (lsec < T)
        all_msgs.extend(m for m, k in zip(lmsg, keep) if k)
        msg_owner.append((lsvc[keep], lsec[keep], int(keep.sum())))
        faults = fault_intervals(json.loads(run["fault_file"].read_text()), services, t0)
        per_run.append({"t0": t0, "T": T, "metrics": metrics, "lat": lat, "any_span": any_span,
                        "faults": faults,
                        "quality": {"run": run["name"], "kind": run["kind"], "t0": t0, "seconds": T,
                                    "n_faults": len(faults), "traces": trace_info,
                                    "logs": {"lines_total": int(len(lts) + unparsed), "unparsed": unparsed,
                                             "kept": int(keep.sum()), "offset_hours": int(loff),
                                             "inside_frac": round(lfrac, 4),
                                             "unknown_services": unknown_log}}})
        q = per_run[-1]["quality"]
        print(f"  run {ri:2d} {run['kind']:<6} {run['name']}: {T}s, faults={len(faults)}, "
              f"spans kept {trace_info['spans_kept']}/{trace_info['spans_total']} (offset {trace_info['offset_hours']:+d}h, "
              f"{trace_info['seconds_with_spans_frac']:.0%} of seconds covered), logs kept {q['logs']['kept']}"
              f"/{q['logs']['lines_total']} (offset {loff:+d}h)")

    # pass 2: templates over all runs (label-free), per-second counts per run
    print(f"[{system}] mining log templates over {len(all_msgs)} lines (top {top_k} kept)")
    tmpl_ids, templates = mine_templates(all_msgs, top_k)
    E = len(templates)
    chunks, run_meta, pos = {}, [], 0
    for ri, (run, pr) in enumerate(zip(runs, per_run)):
        lsvc, lsec, n = msg_owner[ri]
        ids = np.asarray(tmpl_ids[pos: pos + n], int)
        pos += n
        T = pr["T"]
        counts = np.zeros((N, T + 1, E))
        np.add.at(counts, (lsvc, lsec + 1, ids), 1)
        cum = np.cumsum(counts, axis=1)                                   # prefix sums over time
        span_cum = np.concatenate([[0], np.cumsum(pr["any_span"])])
        n_anom = 0
        for s in range(0, T - window + 1, stride):
            e = s + window - 1
            culprit, ftype, fsvc = label_window(s, e, pr["faults"], threshold)
            n_anom += culprit > 0
            chunks[f"{system}-{ri:02d}-{s:05d}"] = {
                "metrics": pr["metrics"][:, s:e + 1, :].astype(np.float32),
                "traces": pr["lat"][:, s:e + 1, None].astype(np.float32),
                "logs": np.log1p(cum[:, e + 1, :] - cum[:, s, :]).astype(np.float32),
                "culprit": int(culprit), "fault_type": ftype, "culprit_service": fsvc,
                "run": run["name"], "run_index": ri, "run_kind": run["kind"],
                "t0": int(pr["t0"] + s),
                "trace_frac": float((span_cum[e + 1] - span_cum[s]) / window),
            }
        n_win = len(range(0, T - window + 1, stride))
        run_meta.append({"index": ri, "name": run["name"], "kind": run["kind"], "t0": pr["t0"],
                         "seconds": T, "windows": n_win, "anomalous": int(n_anom)})
        pr["quality"].update(windows=n_win, anomalous=int(n_anom))

    out = root / "processed" / f"{system}_w{window}_s{stride}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f, protocol=pickle.HIGHEST_PROTOCOL)
    metadata = {
        "format": "eadro_runs_v1", "system": system, "node_num": N, "edges": sysinfo["edges"],
        "event_num": E, "metric_num": len(METRIC_NAMES), "trace_dim": 1, "chunk_lenth": window,
        "stride": stride, "threshold": threshold, "chunk_num": len(chunks),
        "services": services, "metric_names": METRIC_NAMES, "templates": templates,
        "label_map": {"0": "normal", **{str(i + 1): t for i, t in enumerate(FAULT_TYPES)}},
        "runs": run_meta,
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open(out / "quality.json", "w") as f:
        json.dump({"runs": [pr["quality"] for pr in per_run], "template_lines": len(all_msgs)}, f, indent=2)
    lab = Counter(c["culprit"] for c in chunks.values())
    print(f"[{system}] {len(chunks)} chunks, event_num={E}, labels={dict(sorted(lab.items()))} -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True, choices=["SN", "TT"])
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--threshold", type=int, default=1, help="min. overlap (s) with a fault to label a window anomalous")
    ap.add_argument("--top-k-templates", type=int, default=30)
    args = ap.parse_args()
    build(args.system, Path(args.root), args.window, args.stride, args.threshold, args.top_k_templates)
    return 0


if __name__ == "__main__":
    sys.exit(main())
