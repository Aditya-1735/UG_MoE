"""
Parse raw SN dataset into structured format.
Converts raw logs, traces, metrics, and faults into structured format.
"""
import sys
import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def parse_logs(logs_path):
    """Parse logs.json into structured format."""
    with open(logs_path, 'r') as f:
        logs_data = json.load(f)
    
    parsed_logs = {}
    for service, logs in logs_data.items():
        parsed_logs[service] = []
        for log_entry in logs_data[service]:
            # Parse log line: [timestamp] <level>: message
            parts = log_entry.split('] ', 1)
            if len(parts) < 2:
                continue
            timestamp_str = parts[0][1:]  # Remove leading '['
            message = parts[1]
            
            # Parse timestamp: "2022-Apr-17 10:12:50.490796"
            try:
                timestamp = pd.to_datetime(timestamp_str).timestamp()
            except:
                timestamp = 0
            
            # Extract level: <error>, <info>, etc.
            level_end = message.find('>:')
            if level_end > 0:
                level = message[1:level_end]  # Remove '<' and '>'
                message = message[level_end + 2:]  # Skip '>: '
            else:
                level = "unknown"
            
            parsed_logs[service].append({
                'timestamp': timestamp,
                'level': level,
                'message': message
            })
    
    return parsed_logs


def parse_spans(spans_path):
    """Parse spans.json into structured traces."""
    with open(spans_path, 'r') as f:
        traces = json.load(f)
    
    parsed_traces = []
    for trace in traces:
        trace_id = trace['traceID']
        spans = trace['spans']
        parsed_spans = []
        for span in spans:
            parsed_spans.append({
                'trace_id': trace_id,
                'span_id': span['spanID'],
                'operation_name': span['operationName'],
                'start_time': span['startTime'] / 1e6,  # Convert microseconds to seconds
                'duration': span['duration'] / 1e6,  # Convert microseconds to seconds
                'service': span.get('processID', 'unknown'),
                'tags': {tag['key']: tag['value'] for tag in span.get('tags', [])}
            })
        parsed_traces.append({
            'trace_id': trace_id,
            'spans': parsed_spans
        })
    return parsed_traces


def parse_metrics(metrics_dir):
    """Parse metrics CSV files."""
    metrics_data = {}
    metrics_dir = Path(metrics_dir)
    for csv_file in metrics_dir.glob("*.csv"):
        service_name = csv_file.stem
        df = pd.read_csv(csv_file)
        # Ensure timestamp is first column
        if 'timestamp' in df.columns:
            df = df.rename(columns={'timestamp': 'timestamp'})
        metrics_data[service_name] = df
    return metrics_data


def parse_faults(fault_file):
    """Parse fault labels."""
    with open(fault_file, 'r') as f:
        fault_data = json.load(f)
    
    faults = []
    for fault in fault_data.get('faults', []):
        faults.append({
            'service': fault['name'],
            'fault_type': fault['fault'],
            'start_time': fault['start'],
            'duration': fault['duration']
        })
    return faults


def parse_sn_dataset(raw_data_dir, output_dir):
    """Parse all SN dataset directories."""
    raw_data_dir = Path(raw_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all experiment directories
    exp_dirs = [d for d in Path(raw_data_dir).iterdir() if d.is_dir() and d.name.startswith('SN.')]
    
    all_sequences = {}
    all_faults = []
    all_services = set()
    
    for exp_dir in tqdm(sorted(exp_dirs), desc="Parsing experiments"):
        exp_name = exp_dir.name
        
        # Parse logs
        logs_path = exp_dir / 'logs.json'
        if logs_path.exists():
            logs = parse_logs(logs_path)
        else:
            logs = {}
        
        # Parse traces
        spans_path = exp_dir / 'spans.json'
        if spans_path.exists():
            traces = parse_spans(spans_path)
        else:
            traces = []
        
        # Parse metrics
        metrics_dir = exp_dir / 'metrics'
        if metrics_dir.exists():
            metrics = parse_metrics(metrics_dir)
        else:
            metrics = {}
        
        # Parse faults
        fault_files = list(exp_dir.glob('SN.fault-*.json'))
        faults = []
        for fault_file in fault_files:
            faults = parse_faults(fault_file)
            all_faults.extend(faults)
        
        # Collect services
        for service in logs:
            all_services.add(service)
        for service in metrics:
            all_services.add(service)
        for trace in traces:
            for span in trace['spans']:
                all_services.add(span.get('service', 'unknown'))
        
        # Store parsed data
        all_sequences[exp_name] = {
            'logs': logs,
            'traces': traces,
            'metrics': metrics,
            'faults': faults
        }
    
    return all_sequences, all_faults, all_services


def parse_logs(logs_path):
    """Parse logs.json into structured format."""
    with open(logs_path, 'r') as f:
        logs_data = json.load(f)
    
    parsed_logs = {}
    for service, logs in logs_data.items():
        parsed_logs[service] = []
        for log_entry in logs_data[service]:
            # Parse log line: [timestamp] <level>: message
            parts = log_entry.split('] ', 1)
            if len(parts) < 2:
                continue
            timestamp_str = parts[0][1:]  # Remove leading '['
            message = parts[1]
            
            # Parse timestamp: "2022-Apr-17 10:12:50.490796"
            try:
                timestamp = pd.to_datetime(timestamp_str).timestamp()
            except:
                timestamp = 0
            
            # Extract level: <error>, <info>, etc.
            level_end = message.find('>:')
            if level_end > 0:
                level = message[1:level_end]  # Remove '<' and '>'
                message = message[level_end + 2:]  # Skip '>: '
            else:
                level = "unknown"
            
            parsed_logs[service].append({
                'timestamp': timestamp,
                'level': level,
                'message': message
            })
    
    return parsed_logs


def parse_spans(spans_path):
    """Parse spans.json into structured traces."""
    with open(spans_path, 'r') as f:
        traces = json.load(f)
    
    parsed_traces = []
    for trace in traces:
        trace_id = trace['traceID']
        spans = trace['spans']
        parsed_spans = []
        for span in spans:
            parsed_spans.append({
                'trace_id': trace_id,
                'span_id': span['spanID'],
                'operation_name': span['operationName'],
                'start_time': span['startTime'] / 1e6,  # Convert microseconds to seconds
                'duration': span['duration'] / 1e6,  # Convert microseconds to seconds
                'service': span.get('processID', 'unknown'),
                'tags': {tag['key']: tag['value'] for tag in span.get('tags', [])}
            })
        parsed_traces.append({
            'trace_id': trace_id,
            'spans': parsed_spans
        })
    return parsed_traces


def parse_metrics(metrics_dir):
    """Parse metrics CSV files."""
    metrics_data = {}
    metrics_dir = Path(metrics_dir)
    for csv_file in metrics_dir.glob("*.csv"):
        service_name = csv_file.stem
        df = pd.read_csv(csv_file)
        # Ensure timestamp is first column
        if 'timestamp' in df.columns:
            df = df.rename(columns={'timestamp': 'timestamp'})
        metrics_data[service_name] = df
    return metrics_data


def parse_faults(fault_file):
    """Parse fault labels."""
    with open(fault_file, 'r') as f:
        fault_data = json.load(f)
    
    faults = []
    for fault in fault_data.get('faults', []):
        faults.append({
            'service': fault['name'],
            'fault_type': fault['fault'],
            'start_time': fault['start'],
            'duration': fault['duration']
        })
    return faults


def parse_sn_dataset(raw_data_dir, output_dir):
    """Parse all SN dataset directories."""
    raw_data_dir = Path(raw_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all experiment directories
    exp_dirs = [d for d in Path(raw_data_dir).iterdir() if d.is_dir() and d.name.startswith('SN.')]
    
    all_sequences = {}
    all_faults = []
    all_services = set()
    
    for exp_dir in tqdm(sorted(exp_dirs), desc="Parsing experiments"):
        exp_name = exp_dir.name
        
        # Parse logs
        logs_path = exp_dir / 'logs.json'
        if logs_path.exists():
            logs = parse_logs(logs_path)
        else:
            logs = {}
        
        # Parse traces
        spans_path = exp_dir / 'spans.json'
        if spans_path.exists():
            traces = parse_spans(spans_path)
        else:
            traces = []
        
        # Parse metrics
        metrics_dir = exp_dir / 'metrics'
        if metrics_dir.exists():
            metrics = parse_metrics(metrics_dir)
        else:
            metrics = {}
        
        # Parse faults
        fault_files = list(exp_dir.glob('SN.fault-*.json'))
        faults = []
        for fault_file in fault_files:
            faults = parse_faults(fault_file)
            all_faults.extend(faults)
        
        # Collect services
        for service in logs:
            all_services.add(service)
        for service in metrics:
            all_services.add(service)
        for trace in traces:
            for span in trace['spans']:
                all_services.add(span.get('service', 'unknown'))
        
        # Store parsed data
        all_sequences[exp_name] = {
            'logs': logs,
            'traces': traces,
            'metrics': metrics,
            'faults': faults
        }
    
    return all_sequences, all_faults, all_services


def main():
    raw_data_dir = "/c/Users/as999/OneDrive/Desktop/trash/impl/SN_dataset/SN Dataset/data"
    output_dir = "/c/Users/as999/OneDrive/Desktop/trash/impl/project/preprocessing/sn/parsed"
    
    print("Parsing SN dataset...")
    sequences, faults, services = parse_sn_dataset(
        "/c/Users/as999/OneDrive/Desktop/trash/impl/SN_dataset/SN Dataset/data",
        "/c/Users/as999/OneDrive/Desktop/trash/impl/project/preprocessing/sn/parsed"
    )
    
    # Save parsed data
    output_dir = Path("/c/Users/as999/OneDrive/Desktop/trash/impl/project/preprocessing/sn/parsed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open("/c/Users/as999/OneDrive/Desktop/trash/impl/project/preprocessing/sn/parsed/parsed_data.pkl", "wb") as f:
        import pickle
        pickle.dump({}, f)  # placeholder
    
    print("Parsing complete!")


if __name__ == "__main__":
    main()
EOF
echo "Created parse_sn.py"