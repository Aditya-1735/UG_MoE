"""
HDFS Log Parser using Drain3.
Parses HDFS.log into structured templates and template IDs.
"""
import sys
import os
import re
import json
import pickle
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig
except ImportError:
    print("drain3 not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "drain3"])
    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig


def parse_hdfs_log(log_file, output_dir, config=None):
    """
    Parse HDFS.log using Drain3 template miner.
    
    Args:
        log_file: Path to HDFS.log
        output_dir: Directory to save parsed results
        config: Optional TemplateMinerConfig
    
    Returns:
        template_miner: Trained TemplateMiner
        parsed_lines: List of (template_id, template_str) per log line
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default Drain3 config for HDFS
    if config is None:
        config = TemplateMinerConfig()
        config.profiling_enabled = False
        config.drain_extra_delimiters = [":", "=", ",", ";", "|", "(", ")", "[", "]", "{", "}"]
        config.mask_prefix = "*"
        config.mask_suffix = "*"
        config.parametrize_numeric_tokens = True
        config.drain_sim_th = 0.4  # Standard for HDFS
        config.drain_depth = 4
        config.drain_max_children = 100
        config.drain_max_clusters = 1000
    
    # No persistence handler (None = no persistence)
    template_miner = TemplateMiner(persistence_handler=None, config=config)
    
    print(f"Parsing {log_file}...")
    parsed_lines = []
    template_counts = defaultdict(int)
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            # Extract content after timestamp and level
            # HDFS format: 2015-10-30 11:25:34,123 INFO [Thread] Class: Message
            match = re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \w+ \[.*?\] .*?: (.*)$', line)
            if match:
                log_message = match.group(1)
            else:
                log_message = line
            
            result = template_miner.add_log_message(log_message)
            template_id = result["cluster_id"]
            template_str = result["template_mined"]
            
            parsed_lines.append({
                "line_num": line_num,
                "template_id": template_id,
                "template": template_str,
                "raw_message": log_message
            })
            template_counts[template_id] += 1
            
            if line_num % 100000 == 0 and line_num > 0:
                print(f"  Processed {line_num} lines, {len(template_miner.drain.clusters)} templates")
    
    print(f"Total lines: {len(parsed_lines)}")
    print(f"Total templates: {len(template_miner.drain.clusters)}")
    print(f"Top 10 templates:")
    for tid, count in sorted(template_counts.items(), key=lambda x: -x[1])[:10]:
        template = template_miner.drain.id_to_cluster[tid].get_template()
        print(f"  [{tid}] ({count}x) {template[:80]}...")
    
    # Save results
    templates_file = output_dir / "templates.json"
    with open(templates_file, 'w') as f:
        templates = {}
        for cluster in template_miner.drain.clusters:
            templates[cluster.cluster_id] = {
                "template": cluster.get_template(),
                "size": cluster.size
            }
        json.dump(templates, f, indent=2)
    
    parsed_file = output_dir / "parsed_lines.pkl"
    with open(parsed_file, 'wb') as f:
        pickle.dump(parsed_lines, f)
    
    print(f"Saved to {output_dir}")
    return template_miner, parsed_lines


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Parse HDFS.log with Drain3")
    parser.add_argument("--input", required=True, help="Path to HDFS.log")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--similarity", type=float, default=0.4, help="Similarity threshold")
    parser.add_argument("--depth", type=int, default=4, help="Drain depth")
    args = parser.parse_args()
    
    config = TemplateMinerConfig()
    config.profiling_enabled = False
    config.drain_extra_delimiters = [":", "=", ",", ";", "|", "(", ")", "[", "]", "{", "}"]
    config.mask_prefix = True
    config.mask_numbers = True
    config.mask_variables = True
    config.similarity_threshold = args.similarity
    config.depth = args.depth
    config.max_children = 100
    config.max_clusters = 1000
    
    parse_hdfs_log(args.input, args.output, config)


if __name__ == "__main__":
    main()