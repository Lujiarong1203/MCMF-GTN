import torch.nn as nn

from .gtn import ChannelFusionGenerator
from .gcn_classifier import GCNClassifier


class MCMFGTN(nn.Module):
    """GTN channel fusion followed by a GCN backbone.

    matrix mode: A_P = Q_1 Q_2 ... Q_L (meta-path product, top-K pruned)
    edge mode:   per-edge weights from channel mixing
    """

    def __init__(self, in_dim, num_channels=3, gt_layers=3,
                 gcn_hidden=(128, 64, 32), num_classes=2, dropout=0.5,
                 use_matrix_mult=True, gt_top_k=10):
        super().__init__()
        self.num_channels = num_channels
        self.use_matrix_mult = use_matrix_mult

        self.channel_fusion = ChannelFusionGenerator(
            num_channels, gt_layers,
            use_matrix_mult=use_matrix_mult, top_k=gt_top_k,
        )
        self.gcn = GCNClassifier(in_dim, gcn_hidden, num_classes, dropout)

    def forward(self, x, edge_index_or_adj_channels, edge_attr=None):
        if self.use_matrix_mult:
            A_P = self.channel_fusion(edge_index_or_adj_channels)
            edge_index = A_P.indices()
            edge_weight = A_P.values()
        else:
            edge_index = edge_index_or_adj_channels
            edge_weight = self.channel_fusion(edge_attr)
        return self.gcn(x, edge_index, edge_weight)
