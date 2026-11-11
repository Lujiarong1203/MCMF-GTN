import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SGConv


class FeatureAttention(nn.Module):
    """Global attention in the feature dimension (linear attention).

    M = softmax(Q^T K / sqrt(n)) computes feature-to-feature correlations
    (d x d instead of n x n); the output mixes V through M and blends with
    the previous representation via a learnable scalar.
    """

    def __init__(self, d, heads=1):
        super().__init__()
        assert d % heads == 0
        self.heads = heads
        self.dh = d // heads
        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, d)
        self.wv = nn.Linear(d, d)
        self.wo = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, h, prev):
        n = h.size(0)
        q = self.wq(h).view(n, self.heads, self.dh)
        k = self.wk(h).view(n, self.heads, self.dh)
        v = self.wv(h).view(n, self.heads, self.dh)

        out_heads = []
        for i in range(self.heads):
            qi, ki, vi = q[:, i], k[:, i], v[:, i]
            m = F.softmax(qi.t() @ ki / (n ** 0.5), dim=1)
            out_heads.append(vi @ m)
        z = self.wo(torch.cat(out_heads, dim=1))

        z = self.alpha * z + (1.0 - self.alpha) * prev
        return self.norm(z)


class DUALFormer(nn.Module):
    """Dual graph transformer: local SGC in the node dimension and global
    linear attention in the feature dimension, combined in sequence."""

    def __init__(self, in_dim, hidden=64, num_classes=2, sa_layers=2,
                 gnn_hops=2, heads=1, dropout=0.5):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.sa_layers = nn.ModuleList(
            [FeatureAttention(hidden, heads=heads) for _ in range(sa_layers)])
        self.local = SGConv(hidden, hidden, K=gnn_hops)
        self.norm = nn.LayerNorm(hidden)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index, edge_attr=None):
        h = self.input_proj(x)
        prev = h
        for sa in self.sa_layers:
            h = sa(h, prev)
            prev = h
        h = F.relu(self.local(h, edge_index))
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.norm(h)
        return self.fc(h)
