"""
Master script to run full SN preprocessing pipeline.
Orchestrates: parse -> sequences -> graph -> split
"""
import sys
import os
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def run_cmd(cmd, desc):
    """Run command and check result."""
    print(f"\n{'='*60}")
    print(f"STAGE: {desc}")
    print(f"{'='*60}")
    print(f"Running: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {desc} failed!")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    else:
        print(f"SUCCESS: {desc}")
        print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
        return True


def main():
    parser = argparse.ArgumentParser(description="SN Preprocessing Pipeline")
    parser.add_argument("--stages", nargs="+", choices=["parse", "sequences", "graph", "split", "all"], default=["all"], help="Stages to run")
    parser.add_argument("--log", default="/c/Users/as999/OneDrive/Desktop/trash/impl/SN_dataset/SN Dataset/data/SN.2022-04-17T181245D2022-04-17T183616/logs.json", help="Path to logs.json")
    parser.add_argument("--labels", default="/c/Users/as999/OneDrive/Desktop/trash/impl/hdfs_dataset/preprocessed/anomaly_label.csv", help="Path to anomaly labels")
    parser.add_argument("--output", default="preprocessing/sn/output_full", help="Output base directory")
    parser.add_argument("--similarity", type=float, default=0.4, help="Drain3 similarity threshold")
    parser.add_argument("--window", type=int, default=5, help="Co-occurrence window size")
    parser.add_argument("--graph-type", choices=["cooccurrence", "sequential", "attention"], default="cooccurrence")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Determine which stages to run
    stages = args.stages if args.stages != ["all"] else ["parse", "sequences", "graph", "split"]
    
    base_output = Path(args.output)
    base_output.mkdir(parents=True, exist_ok=True)
    
    # Stage 1: Parse
    if "parse" in stages:
        print("\n" + "="*60)
        print("STAGE 1: Parse raw logs")
        print("="*60)
        
        cmd = f'python preprocessing/sn/parse_sn.py --log "{args.log}" --labels "{args.labels}" --output preprocessing/sn/parsed --similarity {args.similarity}'
        if not run_cmd(cmd, "Parse raw logs with Drain3"):
            sys.exit(1)
    
    # Stage 2: Create sequences
    if "sequences" in stages:
        print("\n" + "="*60)
        print("STAGE 2: Create sequences")
        print("="*60)
        
        cmd = "python preprocessing/sn/create_sequences.py"
        if not run_cmd(cmd, "Create sequences"):
            sys.exit(1)
    
    # Stage 3: Build graph
    if "graph" in stages:
        print("\n" + "="*60)
        print("STAGE 3: Build log graph")
        print("="*60)
        
        cmd = f'python preprocessing/sn/build_log_graph.py --sequences preprocessing/sn/output/sequences.pkl --output preprocessing/sn/output/03_log_graph --type {args.graph_type} --window {args.window}'
        if not run_cmd(cmd, "Build log graph"):
            sys.exit(1)
    
    # Stage 4: Split
    if "split" in stages:
        print("\n" + "="*60)
        print("STAGE 4: Split train/test")
        print("="*60)
        
        cmd = f'python preprocessing/sn/split_data.py --sequences preprocessing/sn/output/sequences.pkl --output preprocessing/sn/output/04_split --train-ratio {args.train_ratio} --seed {args.seed}'
        if not run_cmd(cmd, "Split train/test"):
            sys.exit(1)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    import argparse
    import subprocess
    import sys
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="SN Preprocessing Pipeline")
    parser.add_argument("--stages", nargs="+", choices=["parse", "sequences", "graph", "split", "all"], default=["all"], help="Stages to run")
    parser.add_argument("--log", default="/c/Users/as999/OneDrive/Desktop/trash/impl/SN_dataset/SN Dataset/data/SN.2022-04-17T181245D2022-04-17T183616/logs.json", help="Path to logs.json")
    parser.add_argument("--labels", default="/c/Users/as999/OneDrive/Desktop/trash/impl/hdfs_dataset/preprocessed/anomaly_label.csv", help="Path to anomaly labels")
    parser.add_argument("--output", default="preprocessing/sn/output_full", help="Output base directory")
    parser.add_argument("--similarity", type=float, default=0.4, help="Drain3 similarity threshold")
    parser.add_argument("--window", type=int, default=5, help="Co-occurrence window size")
    parser.add_argument("--graph-type", choices=["cooccurrence", "sequential", "attention"], default="cooccurrence")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Determine which stages to run
    stages = args.stages if args.stages != ["all"] else ["parse", "sequences", "graph", "split"]
    
    base_output = Path(args.output)
    base_output.mkdir(parents=True, exist_ok=True)
    
    # Stage 1: Parse
    if "parse" in stages:
        print("\n" + "="*60)
        print("STAGE 1: Parse raw logs")
        print("="*60)
        
        cmd = f'python preprocessing/sn/parse_sn.py --log "{args.log}" --labels "{args.labels}" --output preprocessing/sn/parsed --similarity {args.similarity}'
        if not run_cmd(cmd, "Parse raw logs with Drain3"):
            sys.exit(1)
    
    # Stage 2: Create sequences
    if "sequences" in stages:
        print("\n" + "="*60)
        print("STAGE 2: Create sequences")
        print("="*60)
        
        cmd = "python preprocessing/sn/create_sequences.py"
        if not run_cmd(cmd, "Create sequences"):
            sys.exit(1)
    
    # Stage 3: Build graph
    if "graph" in stages:
        print("\n" + "="*60)
        print("STAGE 3: Build log graph")
        print("="*60)
        
        cmd = f'python preprocessing/sn/build_log_graph.py --sequences preprocessing/sn/output/sequences.pkl --output preprocessing/sn/output/03_log_graph --type {args.graph_type} --window {args.window}'
        if not run_cmd(cmd, "Build log graph"):
            sys.exit(1)
    
    # Stage 4: Split
    if "split" in stages:
        print("\n" + "="*60)
        print("STAGE 4: Split train/test")
        print("="*60)
        
        cmd = f'python preprocessing/sn/split_data.py --sequences preprocessing/sn/output/sequences.pkl --output preprocessing/sn/output/04_split --train-ratio {args.train_ratio} --seed {args.seed}'
        if not run_cmd(cmd, "Split train/test"):
            sys.exit(1)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    import argparse
    import subprocess
    import sys
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="SN Preprocessing Pipeline")
    parser.add_argument("--stages", nargs="+", choices=["parse", "sequences", "graph", "split", "all"], default=["all"], help="Stages to run")
    parser.add_argument("--log", default="/c/Users/as999/OneDrive/Desktop/trash/impl/SN_dataset/SN Dataset/data/SN.2022-04-17T181245D2022-04-17T183616/logs.json", help="Path to logs.json")
    parser.add_argument("--labels", default="/c/Users/as999/OneDrive/Desktop/trash/impl/hdfs_dataset/preprocessed/anomaly_label.csv", help="Path to anomaly labels")
    parser.add_argument("--output", default="preprocessing/sn/output_full", help="Output base directory")
    parser.add_argument("--similarity", type=float, default=0.4, help="Drain3 similarity threshold")
    parser.add_argument("--window", type=int, default=5, help="Co-occurrence window size")
    parser.add_argument("--graph-type", choices=["cooccurrence", "sequential", "attention"], default="cooccurrence")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Determine which stages to run
    stages = args.stages if args.stages != ["all"] else ["parse", "sequences", "graph", "split"]
    
    base_output = Path(args.output)
    base_output.mkdir(parents=True, exist_ok=True)
    
    # Stage 1: Parse
    if "parse" in stages:
        print("\n" + "="*60)
        print("STAGE 1: Parse raw logs")
        print("="*60)
        
        cmd = f'python preprocessing/sn/parse_sn.py --log "{args.log}" --labels "{args.labels}" --output preprocessing/sn/parsed --similarity {args.similarity}'
        if not run_cmd(cmd, "Parse raw logs with Drain3"):
            sys.exit(1)
    
    # Stage 2: Create sequences
    if "sequences" in stages:
        print("\n" + "="*60)
        print("STAGE 2: Create sequences")
        print("="*60)
        
        cmd = "python preprocessing/sn/create_sequences.py"
        if not run_cmd(cmd, "Create sequences"):
            sys.exit(1)
    
    # Stage 3: Build graph
    if "graph" in stages:
        print("\n" + "="*60)
        print("STAGE 3: Build log graph")
        print("="*60)
        
        cmd = f'python preprocessing/sn/build_log_graph.py --sequences preprocessing/sn/output/sequences.pkl --output preprocessing/sn/output/03_log_graph --type {args.graph_type} --window {args.window}'
        if not run_cmd(cmd, "Build log graph"):
            sys.exit(1)
    
    # Stage 4: Split
    if "split" in stages:
        print("\n" + "="*60)
        print("STAGE 4: Split train/test")
        print("="*60)
        
        cmd = f'python preprocessing/sn/split_data.py --sequences preprocessing/sn/output/sequences.pkl --output preprocessing/sn/output/04_split --train-ratio {args.train_ratio} --seed {args.seed}'
        if not run_cmd(cmd, "Split train/test"):
            sys.exit(1)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    import argparse
    import subprocess
    import sys
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="SN Preprocessing Pipeline")
    parser.add_argument("--stages", nargs="+", choices=["parse", "sequences", "graph", "split", "all"], default=["all"], help="Stages to run")
    parser.add_argument("--log", default="/c/Users/as999/OneDrive/Desktop/trash/impl/SN_dataset/SN Dataset/data/SN.2022-04-17T181245D2022-04-17T183616/logs.json", help="Path to logs.json")
    parser.add_argument("--labels", default="/c/Users/as999/OneDrive/Desktop/trash/impl/hdfs_dataset/preprocessed/anomaly_label.csv", help="Path to anomaly labels")
    parser.add_argument("--output", default="preprocessing/sn/output_full", help="Output base directory")
    parser.add_argument("--similarity", type=float, default=0.4, help="Drain3 similarity threshold")
    parser.add_argument("--window", type=int, default=5, help="Co-occurrence window size")
    parser.add_argument("--graph-type", choices=["cooccurrence", "sequential", "attention"], default="cooccurrence")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Determine which stages to run
    stages = args.stages if args.stages != ["all"] else ["parse", "sequences", "graph", "split"]
    
    base_output = Path(args.output)
    base_output.mkdir(parents=True, exist_ok=True)
    
    # Stage 1: Parse
    if "parse" in stages:
        print("\n" + "="*60)
        print("STAGE 1: Parse raw logs")
        print("="*60)
        
        cmd = f'python preprocessing/sn/parse_sn.py --log "{args.log}" --labels "{args.labels}" --output preprocessing/sn/parsed --similarity {args.similarity}'
        if not run_cmd(cmd, "Parse raw logs with Drain3"):
            sys.exit(1)
    
    # Stage 2: Create sequences
    if "sequences" in stages:
        print("\n" + "="*60)
        print("STAGE 2: Create sequences")
        print("="*60)
        
        cmd = "python preprocessing/sn/create_sequences.py"
        if not run_cmd(cmd, "Create sequences"):
            sys.exit(1)
    
    # Stage 3: Build graph
    if "graph" in stages:
        print("\n" + "="*60)
        print("STAGE 3: Build log graph")
        print("="*60)
        
        cmd = f'python preprocessing/sn/build_log_graph.py --sequences preprocessing/sn/output/sequences.pkl --output preprocessing/sn/output/03_log_graph --type {args.graph_type} --window {args.window}'
        if not run_cmd(cmd, "Build log graph"):
            sys.exit(1)
    
    # Stage 4: Split
    if "split" in stages:
        print("\n" + "="*60)
        print("STAGE 4: Split train/test")
        print("="*60)
        
        cmd = f'python preprocessing/sn/split_data.py --sequences preprocessing/sn/output/sequences.pkl --output preprocessing/sn/output/04_split --train-ratio {args.train_ratio} --seed {args.seed}'
        if not run_cmd(cmd, "Split train/test"):
            sys.exit(1)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
EOF
echo "Created run_preprocess.py"