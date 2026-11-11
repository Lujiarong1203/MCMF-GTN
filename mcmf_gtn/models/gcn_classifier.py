import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0, use_bn=True, residual=False):
        super().__init__()
        self.conv = GCNConv(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim) if use_bn else nn.Identity()
        self.dropout = dropout
        self.residual = residual and (in_dim == out_dim)
        if self.residual:
            self.res_linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, edge_index, edge_weight=None):
        h = self.conv(x, edge_index, edge_weight=edge_weight)
        h = self.bn(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        if self.residual:
            h = h + self.res_linear(x)
        return h


class GCNClassifier(nn.Module):
    def __init__(self, in_dim, hidden=(128, 64, 32), num_classes=2,
                 dropout=0.5, use_residual=True):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_dim = in_dim
        for h_dim in hidden:
            self.layers.append(GCNLayer(prev_dim, h_dim, dropout=dropout,
                                        use_bn=True, residual=use_residual))
            prev_dim = h_dim
        self.fc = nn.Linear(hidden[-1], num_classes)

    def forward(self, x, edge_index, edge_weight=None):
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
        return self.fc(x)
