import pickle
import os
from collections import defaultdict

import networkx as nx
import numpy as np

class ETXDataset:

    NUM_FEATURES = 14   # 14 manually constructed node attributes
    NUM_CLASSES = 2

    def __init__(self, data_dir: str, seed: int = 42):
        self.data_dir = data_dir
        self.rng = np.random.default_rng(seed)
        self._load()

    def _load(self):
        pkl_path = os.path.join(self.data_dir, 'ETX Network.pickle')
        with open(pkl_path, 'rb') as f:
            G = pickle.load(f)  # networkx.MultiDiGraph

        self.G_full = G

        node_labels = {}
        for n, attr in G.nodes(data=True):
            lbl = attr.get('label')
            if lbl is not None:
                node_labels[n] = int(lbl)
        self._node_labels = node_labels

        self._compute_node_stats()

        self._build_masks()

    def _compute_node_stats(self):
        """Compute per-node statistics from raw graph edges."""
        G = self.G_full

        node_first_tx = {}
        node_last_tx = {}
        node_in_deg = defaultdict(int)
        node_out_deg = defaultdict(int)
        node_in_amount = defaultdict(list)
        node_out_amount = defaultdict(list)
        node_counterparties_in = defaultdict(set)
        node_counterparties_out = defaultdict(set)

        for u, v, data in G.edges(data=True):
            ts = data.get('timestamp', 0)
            amt = data.get('amount', 0)

            node_first_tx.setdefault(u, ts)
            node_first_tx[u] = min(node_first_tx[u], ts)
            node_last_tx[u] = max(node_last_tx.get(u, ts), ts)
            node_out_deg[u] += 1
            node_out_amount[u].append(amt)
            node_counterparties_out[u].add(v)

            node_first_tx.setdefault(v, ts)
            node_first_tx[v] = min(node_first_tx[v], ts)
            node_last_tx[v] = max(node_last_tx.get(v, ts), ts)
            node_in_deg[v] += 1
            node_in_amount[v].append(amt)
            node_counterparties_in[v].add(u)

        self._node_first_tx = node_first_tx
        self._node_last_tx = node_last_tx
        self._node_in_deg = node_in_deg
        self._node_out_deg = node_out_deg
        self._node_in_amount = node_in_amount
        self._node_out_amount = node_out_amount
        self._node_counterparties_in = node_counterparties_in
        self._node_counterparties_out = node_counterparties_out

    def _build_masks(self):
        """Time-based split using first transaction timestamp."""
        all_nodes = list(self.G_full.nodes())
        node_first = self._node_first_tx

        timestamps = [node_first.get(n, 0) for n in all_nodes]
        sorted_ts = sorted(timestamps)
        n = len(sorted_ts)
        train_idx_end = int(0.70 * n)   # first 70% of nodes by time
        val_idx_end = int(0.80 * n)     # next 10%

        train_cutoff = sorted_ts[train_idx_end]
        val_cutoff = sorted_ts[val_idx_end]

        train_mask = {}
        val_mask = {}
        test_mask = {}
        for n in all_nodes:
            ts = node_first.get(n, 0)
            has_label = n in self._node_labels
            if ts <= train_cutoff and has_label:
                train_mask[n] = True
            elif ts <= val_cutoff and has_label:
                val_mask[n] = True
            elif has_label:
                test_mask[n] = True
            else:
                train_mask[n] = False
                val_mask[n] = False
                test_mask[n] = False

        self._train_mask = train_mask
        self._val_mask = val_mask
        self._test_mask = test_mask

    @property
    def node_features(self) -> np.ndarray:
        all_nodes = list(self.G_full.nodes())
        node_to_idx = {n: i for i, n in enumerate(all_nodes)}
        n_nodes = len(all_nodes)
        feats = np.zeros((n_nodes, self.NUM_FEATURES), dtype=np.float32)

        for n, i in node_to_idx.items():
            indeg = self._node_in_deg.get(n, 0)
            outdeg = self._node_out_deg.get(n, 0)
            in_amt = np.array(self._node_in_amount.get(n, [0]))
            out_amt = np.array(self._node_out_amount.get(n, [0]))
            first_ts = self._node_first_tx.get(n, 0)
            last_ts = self._node_last_tx.get(n, 0)

            feats[i, 0] = indeg
            feats[i, 1] = outdeg
            feats[i, 2] = np.sum(in_amt)
            feats[i, 3] = np.sum(out_amt)
            feats[i, 4] = np.mean(in_amt) if len(in_amt) > 0 else 0.0
            feats[i, 5] = np.mean(out_amt) if len(out_amt) > 0 else 0.0
            feats[i, 6] = np.max(in_amt) if len(in_amt) > 0 else 0.0
            feats[i, 7] = np.max(out_amt) if len(out_amt) > 0 else 0.0
            feats[i, 8] = len(self._node_counterparties_in.get(n, set()))
            feats[i, 9] = len(self._node_counterparties_out.get(n, set()))
            feats[i, 10] = indeg + outdeg
            feats[i, 11] = last_ts - first_ts
            n_txs = indeg + outdeg
            feats[i, 12] = (last_ts - first_ts) / max(n_txs, 1)
            in_sum = feats[i, 2]
            out_sum = feats[i, 3]
            feats[i, 13] = in_sum / max(out_sum, 1e-8)

        return feats

    @property
    def labels(self) -> np.ndarray:
        all_nodes = list(self.G_full.nodes())
        return np.array([self._node_labels.get(n, -1) for n in all_nodes], dtype=np.int64)

    @property
    def node_ids(self) -> list:
        return list(self.G_full.nodes())

    @property
    def adjacency_list(self) -> np.ndarray:
        """Return edge list (num_edges, 2) with original node ids."""
        edges = []
        for u, v in self.G_full.edges():
            edges.append([u, v])
        return np.array(edges)

    @property
    def edge_attributes(self) -> dict:
        """Return edge attributes: timestamps, amounts."""
        ts_list, amt_list = [], []
        for _, _, data in self.G_full.edges(data=True):
            ts_list.append(data.get('timestamp', 0))
            amt_list.append(data.get('amount', 0))
        return {
            'timestamps': np.array(ts_list, dtype=np.float64),
            'amounts': np.array(amt_list, dtype=np.float64),
        }

    def get_mask_arrays(self) -> tuple:
        """Return boolean masks (train_mask, val_mask, test_mask) as numpy arrays."""
        all_nodes = list(self.G_full.nodes())
        train = np.array([self._train_mask.get(n, False) for n in all_nodes])
        val   = np.array([self._val_mask.get(n, False) for n in all_nodes])
        test  = np.array([self._test_mask.get(n, False) for n in all_nodes])
        return train, val, test

    @property
    def graph(self) -> nx.MultiDiGraph:
        return self.G_full

    def __len__(self):
        return len(self.G_full)

    def __repr__(self):
        train, val, test = self.get_mask_arrays()
        return (f'ETXDataset({len(self)} nodes, '
                f'{train.sum()} train, {val.sum()} val, {test.sum()} test)')
