"""Binary node classifiers with directed successor aggregation."""

import torch
from torch import nn
from torch.nn import functional as F

from .adjacency import (
    _coo,
    add_self_loops,
    compose_adjacency,
    row_normalize,
    topology_adjacency,
    validate_edges,
    weighted_sum,
)


VARIANTS = {
    "full", "mlp", "gat", "topology_gcn", "fixed_mean", "direct_ef", "edge_weight",
    "topology_gtn", "no_amount", "no_timestamp", "no_embedding", "embedding_only",
    "uniform_trw", "fixed_product",
}


class DenseBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        if x.shape[0] > 1 or not self.training:
            x = self.batch_norm(x)
        else:
            x = F.batch_norm(x, self.batch_norm.running_mean, self.batch_norm.running_var,
                             self.batch_norm.weight, self.batch_norm.bias, training=False)
        return self.dropout(F.relu(x))


class GraphBlock(DenseBlock):
    def forward(self, x, adjacency):
        return super().forward(torch.sparse.mm(adjacency, x))


class BaseModel(nn.Module):
    def __init__(self, input_dim, num_channels):
        super().__init__()
        self.input_dim = input_dim
        self.num_channels = num_channels
        self.register_buffer("uniform_weights", torch.full((1, num_channels), 1 / num_channels))

    def channel_weights(self):
        return self.uniform_weights

    def _validate(self, x, edge_index, edge_attr):
        if x.ndim != 2 or x.shape[1] != self.input_dim or not x.is_floating_point():
            raise ValueError("x must have shape [N, input_dim] with floating-point values.")
        if not torch.isfinite(x).all():
            raise ValueError("Node features must be finite.")
        validate_edges(edge_index, edge_attr, x.shape[0], self.num_channels, weighted=False)
        if x.device != edge_attr.device or x.device != self.uniform_weights.device:
            raise ValueError("Model, node features and graph tensors must be on the same device.")
        if x.dtype != edge_attr.dtype:
            raise ValueError("Node features and edge-channel values must have the same dtype.")


class SparseGCN(BaseModel):
    def __init__(self, input_dim, num_channels, cfg):
        super().__init__(input_dim, num_channels)
        self.variant = cfg.variant
        self.gt_layers = cfg.gt_layers
        self.top_k = cfg.top_k
        dimensions = (input_dim, *cfg.gcn_hidden)
        self.gcn = nn.ModuleList(GraphBlock(a, b, cfg.dropout) for a, b in zip(dimensions, dimensions[1:]))
        self.head = nn.Linear(dimensions[-1], 2)
        if self.variant == "fixed_product":
            self.register_buffer("channel_logits", torch.zeros(self.gt_layers, num_channels))
        elif self.variant == "fixed_mean":
            self.register_buffer("channel_logits", torch.zeros(1, num_channels))
        elif self.variant not in {"topology_gcn", "direct_ef"}:
            self.channel_logits = nn.Parameter(torch.empty(self.gt_layers, num_channels))
            nn.init.normal_(self.channel_logits, std=0.02)
        if self.variant == "edge_weight":
            self.layer_logits = nn.Parameter(torch.zeros(self.gt_layers))
        if self.variant == "direct_ef":
            hidden = getattr(cfg, "edge_fusion_hidden", 16)
            self.edge_fusion = nn.Sequential(nn.Linear(num_channels, hidden), nn.ReLU(), nn.Linear(hidden, 1), nn.Softplus())

    def channel_weights(self):
        if hasattr(self, "channel_logits"):
            return self.channel_logits.softmax(-1)
        return super().channel_weights()

    def graph_adjacency(self, edge_index, edge_attr, num_nodes):
        validate_edges(edge_index, edge_attr, num_nodes, self.num_channels, weighted=self.variant != "topology_gcn")
        if self.variant == "topology_gcn":
            return topology_adjacency(edge_index, edge_attr, num_nodes)
        if self.variant == "direct_ef":
            weights = self.edge_fusion(edge_attr).squeeze(-1)
            return row_normalize(add_self_loops(_coo(edge_index, weights, num_nodes)))
        if self.variant == "edge_weight":
            factors = [compose_adjacency(edge_index, edge_attr, logits[None], num_nodes, self.top_k)
                       for logits in self.channel_logits]
            return weighted_sum(factors, self.layer_logits.softmax(0))
        return compose_adjacency(edge_index, edge_attr, self.channel_logits, num_nodes, self.top_k)

    def forward(self, x, edge_index, edge_attr):
        self._validate(x, edge_index, edge_attr)
        adjacency = self.graph_adjacency(edge_index, edge_attr, x.shape[0])
        for block in self.gcn:
            x = block(x, adjacency)
        return self.head(x)


class MLP(BaseModel):
    def __init__(self, input_dim, num_channels, cfg):
        super().__init__(input_dim, num_channels)
        dimensions = (input_dim, *cfg.gcn_hidden)
        self.blocks = nn.Sequential(*(DenseBlock(a, b, cfg.dropout) for a, b in zip(dimensions, dimensions[1:])))
        self.head = nn.Linear(dimensions[-1], 2)

    def graph_adjacency(self, edge_index, edge_attr, num_nodes):
        validate_edges(edge_index, edge_attr, num_nodes, self.num_channels, weighted=False)
        diagonal = torch.arange(num_nodes, device=edge_attr.device)
        return _coo(torch.stack((diagonal, diagonal)), torch.ones(num_nodes, device=edge_attr.device, dtype=edge_attr.dtype), num_nodes)

    def forward(self, x, edge_index, edge_attr):
        self._validate(x, edge_index, edge_attr)
        return self.head(self.blocks(x))


class GraphAttention(nn.Module):
    def __init__(self, input_dim, output_dim, heads, dropout, concatenate):
        super().__init__()
        self.output_dim, self.heads = output_dim, heads
        self.concatenate, self.dropout = concatenate, dropout
        self.projection = nn.Linear(input_dim, output_dim * heads, bias=False)
        self.source_attention = nn.Parameter(torch.empty(heads, output_dim))
        self.destination_attention = nn.Parameter(torch.empty(heads, output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim * heads if concatenate else output_dim))
        nn.init.xavier_uniform_(self.source_attention)
        nn.init.xavier_uniform_(self.destination_attention)

    def forward(self, x, indices):
        n = x.shape[0]
        projected = self.projection(x).reshape(n, self.heads, self.output_dim)
        rows, columns = indices
        scores = F.leaky_relu((projected[rows] * self.source_attention).sum(-1)
                              + (projected[columns] * self.destination_attention).sum(-1), 0.2)
        expanded_rows = rows[:, None].expand(-1, self.heads)
        maxima = torch.full((n, self.heads), -torch.inf, device=x.device, dtype=x.dtype)
        maxima.scatter_reduce_(0, expanded_rows, scores, reduce="amax", include_self=True)
        exponentials = (scores - maxima[rows]).exp()
        totals = torch.zeros((n, self.heads), device=x.device, dtype=x.dtype).index_add(0, rows, exponentials)
        attention = F.dropout(exponentials / totals[rows], p=self.dropout, training=self.training)
        messages = projected[columns] * attention[:, :, None]
        output = torch.zeros_like(projected).index_add(0, rows, messages)
        output = output.flatten(1) if self.concatenate else output.mean(1)
        return output + self.bias


class GAT(BaseModel):
    def __init__(self, input_dim, num_channels, cfg):
        super().__init__(input_dim, num_channels)
        hidden = getattr(cfg, "gat_hidden", (64, 32))
        heads = getattr(cfg, "gat_heads", 4)
        self.dropout = getattr(cfg, "gat_dropout", 0.3)
        if len(hidden) != 2 or any(not isinstance(width, int) or width < 1 for width in hidden):
            raise ValueError("gat_hidden must contain two positive integer dimensions.")
        self.attention = nn.ModuleList((GraphAttention(input_dim, hidden[0], heads, self.dropout, True),
                                       GraphAttention(hidden[0] * heads, hidden[1], heads, self.dropout, False)))
        self.head = nn.Linear(hidden[1], 2)

    def graph_adjacency(self, edge_index, edge_attr, num_nodes):
        validate_edges(edge_index, edge_attr, num_nodes, self.num_channels, weighted=False)
        return topology_adjacency(edge_index, edge_attr, num_nodes)

    def forward(self, x, edge_index, edge_attr):
        self._validate(x, edge_index, edge_attr)
        indices = self.graph_adjacency(edge_index, edge_attr, x.shape[0]).indices()
        for layer in self.attention:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(layer(x, indices))
        return self.head(x)


def build_model(input_dim, num_channels, cfg):
    if cfg.variant not in VARIANTS:
        raise ValueError(f"Unknown model variant: {cfg.variant}.")
    if not isinstance(input_dim, int) or input_dim < 1 or not isinstance(num_channels, int) or num_channels < 1:
        raise ValueError("input_dim and num_channels must be positive integers.")
    if cfg.gt_layers < 1 or cfg.top_k < 1:
        raise ValueError("gt_layers and top_k must be positive integers.")
    if not cfg.gcn_hidden or any(width < 1 for width in cfg.gcn_hidden) or not 0 <= cfg.dropout < 1:
        raise ValueError("GCN hidden sizes must be positive and dropout must lie in [0, 1).")
    if cfg.variant == "mlp":
        return MLP(input_dim, num_channels, cfg)
    if cfg.variant == "gat":
        return GAT(input_dim, num_channels, cfg)
    return SparseGCN(input_dim, num_channels, cfg)
