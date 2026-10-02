from dataclasses import dataclass

import numpy as np
import torch

from mcmf_gtn.config import dataset_spec
from mcmf_gtn.features.skipgram import embed_nodes, fit_embeddings
from mcmf_gtn.features.transforms import fit_transforms, transform_split
from mcmf_gtn.features.walks import generate_walks


@dataclass
class TensorGraph:
    node_ids: np.ndarray
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    y: torch.Tensor
    target_mask: torch.Tensor

    def to(self, device):
        return TensorGraph(self.node_ids, self.x.to(device), self.edge_index.to(device), self.edge_attr.to(device), self.y.to(device), self.target_mask.to(device))


@dataclass
class InputBundle:
    train: TensorGraph
    validation: TensorGraph
    test: TensorGraph
    transforms: dict
    embedding_state: object
    channel_names: tuple


def uses_embeddings(cfg):
    return cfg.variant != "no_embedding"


def channel_selection(cfg):
    channels = dataset_spec(cfg.dataset).channels
    if cfg.variant == "no_amount" and "amount" not in channels:
        raise ValueError("amount-channel removal applies only to ETX Network")
    if cfg.variant == "topology_gtn":
        indices = [0]
    elif cfg.variant == "no_amount":
        indices = [i for i, name in enumerate(channels) if name != "amount"]
    elif cfg.variant == "no_timestamp":
        indices = [i for i, name in enumerate(channels) if name != "timestamp"]
    else:
        indices = list(range(len(channels)))
    return indices, tuple(channels[i] for i in indices)


def build_input(split, cfg, transforms, embedding_state):
    static, attributes = transform_split(split, transforms, cfg.dataset)
    if uses_embeddings(cfg):
        walks = generate_walks(split, cfg, uniform=cfg.variant == "uniform_trw", time_scale=transforms["time_scale"], all_nodes=True)
        embeddings = embed_nodes(split, embedding_state, cfg, walks)
        x = embeddings if cfg.variant == "embedding_only" else np.concatenate([static, embeddings], axis=1)
    else:
        x = static
    indices, _ = channel_selection(cfg)
    return TensorGraph(split.node_ids.copy(), torch.from_numpy(np.asarray(x, dtype=np.float32)), torch.from_numpy(split.edge_index.astype(np.int64)), torch.from_numpy(attributes[:, indices].copy()), torch.from_numpy(split.labels.astype(np.int64)), torch.from_numpy(split.target_mask.copy()))


def prepare_inputs(prepared, cfg):
    _, names = channel_selection(cfg)
    transforms = fit_transforms(prepared.train, cfg)
    state = None
    if uses_embeddings(cfg):
        walks = generate_walks(prepared.train, cfg, uniform=cfg.variant == "uniform_trw", time_scale=transforms["time_scale"])
        state = fit_embeddings(walks, cfg)
    graphs = [build_input(split, cfg, transforms, state) for split in (prepared.train, prepared.validation, prepared.test)]
    return InputBundle(*graphs, transforms, state, names)
