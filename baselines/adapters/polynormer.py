"""Polynormer adapter.

The model receives the dataset transaction features without the BTRW
embeddings and is trained under the shared protocol by run_baseline.py.
"""

import os
import sys

_POLY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "polynormer")
if _POLY_DIR not in sys.path:
    sys.path.insert(0, _POLY_DIR)

from model import Polynormer


def build_model(in_dim, num_channels):
    return Polynormer(in_channels=in_dim, hidden_channels=64, out_channels=2,
                      local_layers=3, global_layers=2, dropout=0.5,
                      global_dropout=0.5, heads=1)


def forward(model, data):
    return model(data.x, data.edge_index)
