"""TGTOD adapter (stationary mode, single partition).

The model receives the dataset transaction features without the BTRW
embeddings and is trained under the shared protocol by run_baseline.py.
"""

import os
import sys

_TGTOD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tgtod")
if _TGTOD_DIR not in sys.path:
    sys.path.insert(0, _TGTOD_DIR)

from tgtod import TGTOD


def build_model(in_dim, num_channels):
    return TGTOD(in_channels=in_dim, hidden_channels=32, out_channels=2,
                 time_len=1024, num_parts=1, station=True)


def forward(model, data):
    return model(data.x, [data.edge_index], 0)
