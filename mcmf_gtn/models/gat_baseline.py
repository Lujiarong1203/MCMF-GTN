import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATClassifier(nn.Module):
    def __init__(self, in_dim, hidden=(64, 32), num_classes=2,
                 dropout=0.3, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden[0] // heads, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden[0], hidden[1] // heads, heads=heads, dropout=dropout)
        self.fc = nn.Linear(hidden[1], num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        return self.fc(x)
