"""
Master HDFS Preprocessing Pipeline.
Orchestrates the full HDFS preprocessing pipeline.
"""
import sys
import os
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from preprocessing.hdfs.parse_hdfs import parse_hdfs_log, TemplateMinerConfig
from preprocessing.hdfs.create_sequences import create_sequences
from preprocessing.hdfs.build_log_graph import build_log_graph
from preprocessing.hdfs.split_data import split_data


def run_full_pipeline(input_log, anomaly_labels, output_base, max_seq_len=100, 
                     similarity=0.4, window_size=5, graph_type="cooccurrence", 
                     train_ratio=0.8, seed=42):
    """
    Run the complete HDFS preprocessing pipeline.
    
    Args:
        input_log: Path to HDFS.log
        anomaly_labels: Path to anomaly_label.csv
        output_base: Base output directory
        max_seq_len: Maximum sequence length
        similarity: Drain3 similarity threshold
        window_size: Window size for co-occurrence graph
        graph_type: Type of log graph to build
        train_ratio: Train/test split ratio
        seed: Random seed
    """
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Stage 1: Parse HDFS.log
    print("=" * 60)
    print("STAGE 1: Parse HDFS.log with Drain3")
    print("=" * 60)
    parse_dir = Path(output_base) / "01_parsed"
    
    # Create Drain3 config
    config = TemplateMinerConfig()
    config.profiling_enabled = False
    config.drain_extra_delimiters = [":", "=", ",", ";", "|", "(", ")", "[", "]", "{", "}"]
    config.mask_prefix = True
    config.mask_numbers = True
    config.mask_variables = True
    config.similarity_threshold = similarity
    config.depth = 4
    config.max_children = 100
    config.max_clusters = 1000
    
    parse_hdfs_log(
        str(input_log), 
        str(parse_dir),
        config=config
    )
    
    # Stage 2: Create sequences
    print("=" * 60)
    print("STAGE 2: Create sequences by block_id")
    print("=" * 60)
    seq_dir = Path(output_base) / "02_sequences"
    create_sequences(
        str(parse_dir / "parsed_lines.pkl"),
        str(anomaly_labels),
        str(seq_dir),
        max_seq_len=max_seq_len
    )
    
    # Stage 3: Build log graph
    print("=" * 60)
    print("STAGE 3: Build log co-occurrence graph")
    print("=" * 60)
    graph_dir = Path(output_base) / "03_log_graph"
    build_log_graph(
        str(Path("preprocessing/hdfs/output_2k") / "sequences.pkl"),
        str(graph_dir),
        graph_type="cooccurrence",
        window_size=5
    )
    
    # Stage 4: Split train/test
    print("=" * 60)
    print("STAGE 4: Split train/test")
    print("=" * 60)
    split_dir = Path(output_base) / "04_split"
    split_data(
        str(Path("preprocessing/hdfs/output_2k") / "sequences.pkl"),
        str(split_dir),
        train_ratio=0.8,
        seed=42
    )
    
    # Create summary
    summary = {
        "input_log": str(input_log),
        "anomaly_labels": str(anomaly_labels),
        "output_base": str(output_base),
        "max_seq_len": max_seq_len,
        "similarity_threshold": 0.4,
        "window_size": 5,
        "graph_type": "cooccurrence",
        "train_ratio": 0.8,
        "stages": {
            "parse": str(parse_dir),
            "sequences": str(seq_dir),
            "log_graph": str(graph_dir),
            "split": str(split_dir)
        }
    }
    
    summary_file = Path(output_base) / "pipeline_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"Output saved to: {output_base}")
    print(f"Summary: {summary_file}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="HDFS Preprocessing Pipeline")
    parser.add_argument("--log", required=True, help="Path to HDFS.log")
    parser.add_argument("--labels", required=True, help="Path to anomaly_label.csv")
    parser.add_argument("--output", required=True, help="Output base directory")
    parser.add_argument("--max-len", type=int, default=100, help="Max sequence length")
    parser.add_argument("--similarity", type=float, default=0.4, help="Drain3 similarity threshold")
    parser.add_argument("--window", type=int, default=5, help="Co-occurrence window size")
    parser.add_argument("--graph-type", choices=["cooccurrence", "sequential", "attention"], 
                       default="cooccurrence", help="Graph type")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stages", nargs="+", choices=["parse", "sequences", "graph", "split", "all"],
                       default=["all"], help="Stages to run")
    args = parser.parse_args()
    
    if "all" in args.stages:
        stages = ["parse", "sequences", "graph", "split"]
    else:
        stages = args.stages
    
    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Stage 1: Parse
    if "parse" in stages:
        print("=" * 60)
        print("STAGE 1: Parse HDFS.log")
        print("=" * 60)
        parse_dir = Path(args.output) / "01_parsed"
        
        config = TemplateMinerConfig()
        config.profiling_enabled = False
        config.drain_extra_delimiters = [":", "=", ",", ";", "|", "(", ")", "[", "]", "{", "}"]
        config.mask_prefix = "*"
        config.mask_suffix = "*"
        config.parametrize_numeric_tokens = True
        config.drain_sim_th = args.similarity
        config.drain_depth = 4
        config.drain_max_children = 100
        config.drain_max_clusters = 1000
        
        parse_hdfs_log(args.log, str(parse_dir), config=config)
    
    # Stage 2: Sequences
    if "sequences" in stages:
        print("=" * 60)
        print("STAGE 2: Create sequences")
        print("=" * 60)
        parse_dir = Path(args.output) / "01_parsed"
        seq_dir = Path(args.output) / "02_sequences"
        create_sequences(
            str(parse_dir / "parsed_lines.pkl"),
            args.labels,
            str(seq_dir),
            max_seq_len=args.max_len
        )
    
    # Stage 3: Graph
    if "graph" in stages:
        print("=" * 60)
        print("STAGE 3: Build log graph")
        print("=" * 60)
        seq_dir = Path(args.output) / "02_sequences"
        graph_dir = Path(args.output) / "03_log_graph"
        build_log_graph(
            str(seq_dir / "sequences.pkl"),
            str(Path(args.output) / "03_log_graph"),
            graph_type=args.graph_type,
            window_size=args.window
        )
    
    # Stage 4: Split
    if "split" in stages:
        print("=" * 60)
        print("STAGE 4: Split train/test")
        print("=" * 60)
        seq_dir = Path(args.output) / "02_sequences"
        split_dir = Path(args.output) / "04_split"
        split_data(
            str(seq_dir / "sequences.pkl"),
            str(split_dir),
            train_ratio=args.train_ratio,
            seed=args.seed
        )
    
    print("=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()