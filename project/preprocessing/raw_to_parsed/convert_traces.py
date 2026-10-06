#!/usr/bin/env python3
"""
RECONSTRUCTED COMPONENT - NOT ORIGINAL SOURCE CODE
Converts raw spans.json to parsed_data format:
- Resolve processID → serviceName per trace (with nginx-thrift ↔ nginx-web-server alias)
- Convert microsecond startTime/duration to integer-second buckets
- Reconstruct caller→callee edges from CHILD_OF references
- Aggregate per-second-bucket, per-edge latency lists
Output: parsed_data/SN/traces<idx>.json (dict keyed by second timestamp string, valued by {edge_key: [latencies]})
"""
import os
import json
from pathlib import Path
from collections import defaultdict

SERVICE_NAMES = [
    'social-graph-service', 'compose-post-service', 'post-storage-service',
    'user-timeline-service', 'url-shorten-service', 'user-service',
    'media-service', 'text-service', 'unique-id-service', 'user-mention-service',
    'home-timeline-service', 'nginx-web-server'
]

SERVICE_TO_NID = {s: i for i, s in enumerate(SERVICE_NAMES)}

# Edge list from Eadro util.py (24 directed edges)
EDGES = [
    (1, 1), (1, 10), (1, 6), (1, 2), (1, 7), (1, 8), (1, 5), (1, 3),
    (10, 10), (10, 2), (10, 0),
    (2, 0), (2, 5),
    (0, 0), (0, 5),
    (7, 7), (7, 4), (7, 9),
    (5, 5), (5, 3),
    (3, 1), (3, 10), (3, 11), (3, 11), (3, 0), (3, 5)
]

EDGE_KEYS = [f"{src}-{dst}" for src, dst in EDGES]

# Alias mapping: raw fault target / Jaeger hostname → Eadro service name
ALIAS_MAP = {
    'socialnetwork-nginx-thrift-1': 'nginx-web-server',
    'nginx-thrift': 'nginx-web-server',
}

RAW_BASE = Path("C:/Users/as999/OneDrive/Desktop/trash/impl/raw_data/SN Dataset")
PARSED_BASE = Path("C:/Users/as999/OneDrive/Desktop/trash/impl/project/parsed_data")


def resolve_service_name(raw_name: str) -> str:
    """Apply alias mapping to resolve service name."""
    # Strip socialnetwork- prefix and -1 suffix if present
    name = raw_name
    if name.startswith('socialnetwork-'):
        name = name[len('socialnetwork-'):]
    if name.endswith('-1'):
        name = name[:-2]
    # Apply alias
    return ALIAS_MAP.get(name, name)


def convert_traces_for_experiment(exp_dir: Path, exp_idx: int, fault_type: str) -> bool:
    """Convert traces for a single experiment."""
    spans_src = exp_dir / "spans.json"
    if not spans_src.exists():
        print(f"  [ERROR] No spans.json in {exp_dir}")
        return False
    
    print(f"  Loading spans.json...")
    with open(spans_src, 'r') as f:
        traces = json.load(f)
    
    print(f"  Processing {len(traces)} traces...")
    
    # Output structure: {second_timestamp_str: {edge_key: [latencies]}}
    traces_output = defaultdict(lambda: defaultdict(list))
    
    for trace_idx, trace in enumerate(traces):
        if trace_idx % 1000 == 0:
            print(f"    Processed {trace_idx}/{len(traces)} traces...")
        
        # Build processID → serviceName mapping for this trace
        process_map = {}
        for pid, pinfo in trace.get('processes', {}).items():
            service_name = pinfo.get('serviceName', '')
            # Apply alias
            resolved = resolve_service_name(service_name)
            process_map[pid] = resolved
        
        # Process spans
        for span in trace.get('spans', []):
            # Get caller service from processID
            caller_pid = span.get('processID')
            if caller_pid not in process_map:
                continue
            caller_service = process_map[caller_pid]
            
            # Find callee from CHILD_OF references
            callee_service = None
            for ref in span.get('references', []):
                if ref.get('refType') == 'CHILD_OF':
                    callee_pid = ref.get('spanID')  # This is spanID, need to find which process it belongs to
                    # Actually, CHILD_OF references point to parent span in same trace
                    # We need to find the parent span's processID
                    parent_span_id = ref.get('spanID')
                    # Find parent span in this trace
                    for parent_span in trace.get('spans', []):
                        if parent_span.get('spanID') == parent_span_id:
                            parent_pid = parent_span.get('processID')
                            if parent_pid in process_map:
                                callee_service = process_map[parent_pid]
                            break
                    break
            
            if not callee_service:
                # No parent found, this might be a root span
                continue
            
            # Convert to edge key
            caller_nid = SERVICE_TO_NID.get(caller_service)
            callee_nid = SERVICE_TO_NID.get(callee_service)
            
            if caller_nid is None or callee_nid is None:
                continue
            
            edge_key = f"{caller_nid}-{callee_nid}"
            
            # Verify edge exists in expected graph
            if edge_key not in EDGE_KEYS:
                print(f"  [WARN] Edge {edge_key} ({caller_service}->{callee_service}) not in expected graph")
                continue
            
            # Convert microsecond timestamp to integer second bucket
            start_time_us = span.get('startTime', 0)
            duration_us = span.get('duration', 0)
            
            second_bucket = start_time_us // 1_000_000  # floor division
            
            # Duration in milliseconds (as deal_traces expects)
            latency_ms = duration_us / 1000.0
            
            traces_output[str(second_bucket)][edge_key].append(latency_ms)
    
    if not traces_output:
        print(f"  [WARN] No valid trace data extracted")
        return False
    
    # Write traces<idx>.json
    traces_json = PARSED_BASE / "SN" / f"traces{exp_idx}.json"
    # Convert defaultdict to regular dict for JSON serialization
    output_dict = {k: dict(v) for k, v in traces_output.items()}
    with open(traces_json, 'w') as f:
        json.dump(output_dict, f)
    
    print(f"  [OK] Converted traces for {fault_type} experiment {exp_idx}: {len(output_dict)} second buckets")
    return True


def main():
    print("=" * 60)
    print("RECONSTRUCTED: convert_traces.py")
    print("Raw spans.json -> traces<idx>.json (per-second-bucket, per-edge latency)")
    print("=" * 60)
    
    PARSED_BASE.mkdir(parents=True, exist_ok=True)
    (PARSED_BASE / "SN").mkdir(parents=True, exist_ok=True)
    
    # Process fault experiments (data/)
    fault_data_dir = RAW_BASE / "data"
    fault_experiments = []
    for item in fault_data_dir.iterdir():
        if item.is_dir() and item.name.startswith("SN.") and not item.name.endswith(".json"):
            fault_experiments.append(item)
    fault_experiments = sorted(fault_experiments)
    
    print(f"\nFound {len(fault_experiments)} fault experiments")
    
    for i, exp_dir in enumerate(fault_experiments):
        print(f"\nProcessing fault experiment {i}: {exp_dir.name}")
        if not convert_traces_for_experiment(exp_dir, i, "fault"):
            print(f"  [FAILED] Experiment {i}")
            return 1
    
    # Process no-fault experiments (no fault/)
    nofault_data_dir = RAW_BASE / "no fault"
    nofault_experiments = []
    for item in nofault_data_dir.iterdir():
        if item.is_dir() and item.name.startswith("SN."):
            nofault_experiments.append(item)
    nofault_experiments = sorted(nofault_experiments)
    
    print(f"\nFound {len(nofault_experiments)} no-fault experiments")
    
    for i, exp_dir in enumerate(nofault_experiments):
        exp_idx = len(fault_experiments) + i
        print(f"\nProcessing no-fault experiment {exp_idx}: {exp_dir.name}")
        if not convert_traces_for_experiment(exp_dir, exp_idx, "no_fault"):
            print(f"  [FAILED] Experiment {exp_idx}")
            return 1
    
    print("\n" + "=" * 60)
    print("ALL TRACES CONVERSION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())