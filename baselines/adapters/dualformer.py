"""DUALFormer adapter.

Implementation following the paper's description (Zhuo et al., ICLR 2025) in
baselines/dualformer/model.py. The model receives the dataset transaction
features without the BTRW embeddings and is trained under the shared protocol
by run_baseline.py.
"""

from dualformer.model import DUALFormer


def build_model(in_dim, num_channels):
    return DUALFormer(in_dim, hidden=64, num_classes=2, sa_layers=2,
                      gnn_hops=2, heads=1, dropout=0.5)


def forward(model, data):
    return model(data.x, data.edge_index)
