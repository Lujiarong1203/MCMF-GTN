import torch
import torch.nn as nn
import torch.nn.functional as F

from .gcn_classifier import GCNClassifier


class EFGCN(nn.Module):
    """Edge-feature GCN: maps each edge's channel values to a scalar weight,
    then applies the shared GCN backbone.

    mode "fixed":   arithmetic mean of the channel values
    mode "direct":  small learnable function on the channel vector
    mode "learned": learned scalar coefficients with softmax normalization
    """

    def __init__(self, in_dim, num_channels, mode, gcn_hidden=(128, 64, 32),
                 num_classes=2, dropout=0.5):
        super().__init__()
        self.mode = mode
        if mode == "learned":
            self.channel_weight = nn.Parameter(torch.ones(num_channels))
        elif mode == "direct":
            self.map = nn.Sequential(nn.Linear(num_channels, 1), nn.Softplus())
        elif mode != "fixed":
            raise ValueError(f"Unknown EFGCN mode: {mode}")
        self.gcn = GCNClassifier(in_dim, gcn_hidden, num_classes, dropout)

    def forward(self, x, edge_index, edge_attr):
        if self.mode == "fixed":
            edge_weight = edge_attr.mean(dim=1)
        elif self.mode == "learned":
            w = F.softmax(self.channel_weight, dim=0)
            edge_weight = (edge_attr * w.unsqueeze(0)).sum(dim=1)
        else:
            edge_weight = self.map(edge_attr).squeeze(1)
        return self.gcn(x, edge_index, edge_weight)
