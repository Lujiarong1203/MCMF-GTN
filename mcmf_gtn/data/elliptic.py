import os

import numpy as np
import pandas as pd
import torch

class EllipticDataset:

    FEATURES_HEADER = None          # no header in CSV
    EDGELIST_HEADER = 0             # has header
    CLASSES_HEADER = 0
    NUM_FEATURES = None  # auto-detected in _load()
    NUM_CLASSES = 2

    def __init__(self, data_dir: str, seed: int = 42):
        self.data_dir = data_dir
        self.rng = np.random.default_rng(seed)
        self._load()

    def _load(self):
        base = self.data_dir

        feat = pd.read_csv(
            os.path.join(base, 'elliptic_txs_features.csv'),
            header=None,
        )
        self.NUM_FEATURES = feat.shape[1] - 2
        feat.columns = ['txId', 'Time step'] + [f'f{i}' for i in range(self.NUM_FEATURES)]

        edges = pd.read_csv(
            os.path.join(base, 'elliptic_txs_edgelist.csv'),
            header=0,
        )

        classes = pd.read_csv(
            os.path.join(base, 'elliptic_txs_classes.csv'),
            header=0,
        )

        label_map = {'unknown': -1, '1': 1, '2': 0}
        classes['label'] = classes['class'].astype(str).map(label_map)

        data = feat.merge(classes[['txId', 'label']], on='txId', how='left')
        self.data = data
        self.edges = edges

        self.time_steps = sorted(data['Time step'].unique())
        self.train_cutoff = 34
        self.val_start = 30     # not used; val from training set
        self.test_start = 34

        self._build_masks()

    def _build_masks(self):
        ts = self.data['Time step'].values
        labels = self.data['label'].values

        train_mask = (ts <= self.train_cutoff) & (labels != -1)
        test_mask  = (ts > self.test_start) & (labels != -1)

        train_idx = np.where(train_mask)[0]
        self.rng.shuffle(train_idx)
        val_from_train = int(0.1 * len(train_idx))

        train_mask_adj = train_mask.copy()
        train_mask_adj[train_idx[:val_from_train]] = False
        val_mask_adj = np.zeros(len(labels), dtype=bool)
        val_mask_adj[train_idx[:val_from_train]] = True

        self.train_mask = train_mask_adj
        self.val_mask   = val_mask_adj
        self.test_mask  = test_mask

    @property
    def node_features(self) -> np.ndarray:
        """X_Trans: static transaction features."""
        cols = [f'f{i}' for i in range(self.NUM_FEATURES)]
        return self.data[cols].values.astype(np.float32)

    @property
    def labels(self) -> np.ndarray:
        return self.data['label'].values.astype(np.int64)

    @property
    def tx_ids(self) -> np.ndarray:
        return self.data['txId'].values

    @property
    def time_step(self) -> np.ndarray:
        return self.data['Time step'].values

    @property
    def adjacency_list(self) -> np.ndarray:
        """Return edge list as numpy array (num_edges, 2)."""
        return self.edges[['txId1', 'txId2']].values

    def get_split_indices(self):
        """Return boolean masks for train/val/test."""
        return self.train_mask, self.val_mask, self.test_mask

    def get_labeled_indices(self):
        """All labeled nodes (ignore 'unknown')."""
        return self.data['label'].values != -1

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        train_n = self.train_mask.sum()
        val_n   = self.val_mask.sum()
        test_n  = self.test_mask.sum()
        total_l = train_n + val_n + test_n
        return (f'EllipticDataset({len(self)} nodes, '
                f'{train_n} train, {val_n} val, {test_n} test, '
                f'{total_l} labeled)')
