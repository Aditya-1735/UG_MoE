import torch
from torch.utils.data import Dataset
import dgl
import pickle
from dataclasses import dataclass
from typing import Dict


@dataclass
class DatasetConfig:
    num_nodes: int
    metric_dim: int
    trace_dim: int
    log_dim: int
    num_edges: int

    @classmethod
    def from_metadata(cls, metadata: dict):
        return cls(
            num_nodes=metadata["node_num"],
            metric_dim=metadata["metric_num"],
            trace_dim=metadata.get("trace_dim", 1),
            log_dim=metadata["event_num"],
            num_edges=len(metadata["edges"][0]),
        )


class ChunkDataset(Dataset):
    def __init__(self, chunks, chunk_ids, node_num, edges, labels=None, config: DatasetConfig = None):
        self.data = []
        self.idx2id = {}
        for idx, chunk_id in enumerate(chunk_ids):
            self.idx2id[idx] = chunk_id
            chunk = chunks[chunk_id]

            if config is not None:
                # Validate dims against metadata config
                actual_metric_dim = chunk["metrics"].shape[-1]
                actual_trace_dim = chunk["traces"].shape[-1]
                actual_log_dim = chunk["logs"].shape[-1]
                if actual_metric_dim != config.metric_dim:
                    raise ValueError(
                        f"Chunk metric_dim {actual_metric_dim} != config.metric_dim {config.metric_dim}"
                    )
                if actual_trace_dim != config.trace_dim:
                    raise ValueError(
                        f"Chunk trace_dim {actual_trace_dim} != config.trace_dim {config.trace_dim}"
                    )
                if actual_log_dim != config.log_dim:
                    raise ValueError(
                        f"Chunk log_dim {actual_log_dim} != config.log_dim {config.log_dim}"
                    )

            graph = dgl.graph(edges, num_nodes=node_num)
            graph.ndata["metrics"] = torch.FloatTensor(chunk["metrics"])
            graph.ndata["traces"] = torch.FloatTensor(chunk["traces"])
            graph.ndata["logs"] = torch.FloatTensor(chunk["logs"])

            if labels is not None:
                self.data.append((graph, labels[idx]))
            else:
                self.data.append((graph, chunk["culprit"]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_chunk_id(self, idx):
        return self.idx2id[idx]


class CombinedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.data = dataset1.data + dataset2.data
        self.idx2id = {**dataset1.idx2id, **dataset2.idx2id}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_chunk_id(self, idx):
        return self.idx2id[idx]