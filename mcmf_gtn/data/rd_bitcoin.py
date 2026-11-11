import os

import numpy as np
import pandas as pd

class RDBitcoinDataset:

    FEATURES_HEADER = 0           # has header
    EDGELIST_HEADER = 0
    CLASSES_HEADER = 0
    NUM_FEATURES = None  # auto-detected in _load()
    NUM_CLASSES = 2

    SAMPLING_RATIO = 0.1          # default

    def __init__(self, data_dir: str, seed: int = 42, sampling_ratio: float = 0.1):
        self.data_dir = data_dir
        self.rng = np.random.default_rng(seed)
        self.SAMPLING_RATIO = sampling_ratio
        self._load()

    def _load(self):
        base = self.data_dir
        sr = self.SAMPLING_RATIO

        fname = f'rd_features{sr}.csv'
        if sr == int(sr):
            fname = f'rd_features{int(sr)}.csv'
        fpath = os.path.join(base, fname)
        if not os.path.exists(fpath):
            fname = f'rd_features{sr:.2f}.csv'
            fpath = os.path.join(base, fname)

        feat = pd.read_csv(fpath, header=0)
        self.NUM_FEATURES = feat.shape[1] - 2  # auto-detect feature count
        feat = feat.rename(columns={
            feat.columns[0]: 'txId',
            feat.columns[1]: 'Time step',
        })

        edges = pd.read_csv(
            os.path.join(base, 'rd_edgelists.csv'),
            header=0,
        )

        classes = pd.read_csv(
            os.path.join(base, 'rd_classes.csv'),
            header=0,
        )

        label_map = {'unknown': -1, '1': 1, '2': 0, '3': -1, 'illicit': 1, 'licit': 0}
        classes['label'] = classes['class'].astype(str).map(label_map)

        data = feat.merge(classes[['txId', 'label']], on='txId', how='left')
        self.data = data
        self.edges = edges

        self.time_steps = sorted(data['Time step'].unique())
        self.train_cutoff = 34
        self.val_start = 34
        self.test_start = 44

        self._build_masks()

    def _build_masks(self):
        ts = self.data['Time step'].values
        labels = self.data['label'].values

        train_mask = (ts <= self.train_cutoff) & (labels != -1)
        val_mask   = (ts > self.val_start) & (ts <= self.test_start) & (labels != -1)
        test_mask  = (ts > self.test_start) & (labels != -1)

        train_idx = np.where(train_mask)[0]
        self.rng.shuffle(train_idx)
        val_from_train = int(0.1 * len(train_idx))
        val_extra = train_idx[:val_from_train]

        train_mask_adj = train_mask.copy()
        train_mask_adj[val_extra] = False
        val_mask_adj = val_mask.copy()
        val_mask_adj[val_extra] = True

        self.train_mask = train_mask_adj
        self.val_mask   = val_mask_adj
        self.test_mask  = test_mask

    @property
    def node_features(self) -> np.ndarray:
        """X_Trans: static transaction features."""
        feat_cols = self.data.columns[2:2 + getattr(self, 'NUM_FEATURES', 184)]
        return self.data[feat_cols].values.astype(np.float32)

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
        return self.edges.values  # [txId1, txId2]

    def get_split_indices(self):
        return self.train_mask, self.val_mask, self.test_mask

    def get_labeled_indices(self):
        return self.data['label'].values != -1

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        train_n = self.train_mask.sum()
        val_n   = self.val_mask.sum()
        test_n  = self.test_mask.sum()
        return (f'RDBitcoinDataset({len(self)} nodes, '
                f'{train_n} train, {val_n} val, {test_n} test, '
                f'sampling_ratio={self.SAMPLING_RATIO})')
