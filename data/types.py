from dataclasses import dataclass, field

import numpy as np


@dataclass
class GraphData:
    node_ids: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    node_times: np.ndarray
    edge_index: np.ndarray
    edge_times: np.ndarray
    edge_amounts: np.ndarray = None

    def validate(self):
        n = len(self.node_ids)
        if self.node_ids.ndim != 1 or len(set(map(str, self.node_ids))) != n:
            raise ValueError("node IDs must be unique")
        if self.features.ndim != 2 or len(self.features) != n or self.features.shape[1] < 1:
            raise ValueError("invalid node feature dimensions")
        if np.isinf(self.features).any():
            raise ValueError("node features contain infinity")
        if self.labels.shape != (n,) or not np.isin(self.labels, [-1, 0, 1]).all():
            raise ValueError("labels must be -1, 0 or 1")
        if self.node_times.shape != (n,) or not np.isfinite(self.node_times).all():
            raise ValueError("node timestamps must be finite")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2 or not np.issubdtype(self.edge_index.dtype, np.integer):
            raise ValueError("edge_index must be a 2xE integer array")
        if self.edge_index.size and (self.edge_index.min() < 0 or self.edge_index.max() >= n):
            raise ValueError("edge endpoints are out of bounds")
        e = self.edge_index.shape[1]
        if self.edge_times.shape != (e,) or not np.isfinite(self.edge_times).all():
            raise ValueError("edge timestamps must be finite")
        if self.edge_amounts is not None:
            if self.edge_amounts.shape != (e,) or not np.isfinite(self.edge_amounts).all() or (self.edge_amounts < 0).any():
                raise ValueError("edge amounts must be finite and nonnegative")


@dataclass
class SplitGraph(GraphData):
    target_mask: np.ndarray = None

    def validate(self):
        super().validate()
        if self.target_mask is None or self.target_mask.shape != self.labels.shape or self.target_mask.dtype != bool:
            raise ValueError("invalid target mask")
        if (self.labels[self.target_mask] < 0).any():
            raise ValueError("unknown nodes cannot be prediction targets")


@dataclass
class PreparedData:
    train: SplitGraph
    validation: SplitGraph
    test: SplitGraph
    stats: dict = field(default_factory=dict)
    eligible_test_ids: np.ndarray = None
