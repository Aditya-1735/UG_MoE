"""
Service lists and dependency graphs of the two Eadro benchmark systems.

Taken from Eadro's own preprocessing (vendor/eadro/codes/preprocess/util.py, class Info) so the
service order and the 24-edge SN graph are identical to the MUAD artifact (recovered_artifact).
"""
import importlib.util
import re
from pathlib import Path
from typing import Dict, List, Tuple

_UTIL = Path(__file__).resolve().parent.parent / "vendor" / "eadro" / "codes" / "preprocess" / "util.py"

METRIC_NAMES = ["cpu_usage_system", "cpu_usage_total", "cpu_usage_user", "memory_usage",
                "memory_working_set", "rx_bytes", "tx_bytes"]
FAULT_TYPES = ["cpu_load", "network_delay", "network_loss"]       # culprit code = index + 1
BENCH = {"SN": "socialnetwork", "TT": "trainticket"}


def _eadro_info(system: str):
    spec = importlib.util.spec_from_file_location("eadro_util", _UTIL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Info(BENCH[system])


def system_info(system: str) -> Dict:
    """{'services': [...], 'edges': [src_ids, dst_ids], 'metric_names': [...]}"""
    if system not in BENCH:
        raise ValueError(f"system must be one of {list(BENCH)}, got {system!r}")
    info = _eadro_info(system)
    if list(info.metric_names) != METRIC_NAMES:
        raise RuntimeError(f"Unexpected Eadro metric names: {info.metric_names}")
    return {"services": list(info.service_names), "edges": [list(e) for e in info.edges],
            "metric_names": list(info.metric_names)}


# Fault-file container names -> service names, e.g.
#   SN: "socialnetwork-home-timeline-service-1"        -> "home-timeline-service"
#       "socialnetwork-nginx-thrift-1"                 -> "nginx-web-server" (container vs service name)
#   TT: "dockercomposemanifests_ts-cancel-service_1"   -> "ts-cancel-service"
_ALIASES = {"nginx-thrift": "nginx-web-server"}


def fault_service(container: str, services: List[str]) -> str:
    name = container
    name = re.sub(r"^(socialnetwork-|dockercomposemanifests_)", "", name)
    name = re.sub(r"[-_]\d+$", "", name)
    name = _ALIASES.get(name, name)
    if name not in services:
        raise KeyError(f"Fault container {container!r} -> {name!r} is not a known service")
    return name


def edge_list(edges) -> List[Tuple[int, int]]:
    return list(zip(edges[0], edges[1]))
