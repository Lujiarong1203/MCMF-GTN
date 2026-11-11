import os
import numpy as np
import torch

from mcmf_gtn.data.etx import ETXDataset
from mcmf_gtn.data.rd_bitcoin import RDBitcoinDataset
from mcmf_gtn.features.subgraph import construct_subgraph_from_edgelist
from mcmf_gtn.features.to_pyg_data import prepare_pyg_data

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(_PROJ_ROOT, "data")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_etx(seed=42, device=DEVICE):
    e = ETXDataset(DATA_DIR, seed=seed)

    all_nodes = list(e.G_full.nodes())
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}
    n_all = len(all_nodes)

    src_list, dst_list, timestamps, amounts = [], [], [], []
    for u, v, data in e.G_full.edges(data=True):
        src_list.append(node_to_idx[u])
        dst_list.append(node_to_idx[v])
        timestamps.append(data.get("timestamp", 0))
        amounts.append(data.get("amount", 0))

    edgelist = np.column_stack([np.array(src_list, dtype=np.int64),
                                np.array(dst_list, dtype=np.int64)])
    all_node_ids_int = np.arange(n_all, dtype=np.int64)
    timestamps = np.array(timestamps, dtype=np.float64)
    amounts = np.array(amounts, dtype=np.float64)

    rng = np.random.default_rng(seed)
    labels_all = np.array([e._node_labels.get(n, -1) for n in all_nodes], dtype=np.int64)

    node_mask, edge_mask, subgraph_node_ids_int = construct_subgraph_from_edgelist(
        all_node_ids_int, labels_all, edgelist, rng
    )

    si = np.where(node_mask)[0]
    esub = edgelist[edge_mask]
    ts_sub = timestamps[edge_mask]
    amt_sub = amounts[edge_mask]

    if hasattr(e, "node_features"):
        node_feats = e.node_features[node_mask]
    else:
        node_feats = _compute_etx_features(e, all_nodes, node_mask, len(si))

    train_mask, val_mask, test_mask = e.get_mask_arrays()
    train_sub = train_mask[node_mask]
    val_sub = val_mask[node_mask]
    test_sub = test_mask[node_mask]

    if train_sub.sum() == 0:
        n = len(si)
        perm = rng.permutation(n)
        n_train = int(0.7 * n)
        n_val = int(0.1 * n)
        train_sub = np.zeros(n, dtype=bool)
        val_sub = np.zeros(n, dtype=bool)
        test_sub = np.zeros(n, dtype=bool)
        train_sub[perm[:n_train]] = True
        val_sub[perm[n_train:n_train + n_val]] = True
        test_sub[perm[n_train + n_val:]] = True

    data = prepare_pyg_data(
        subgraph_node_ids_int,
        node_feats,
        labels_all[si],
        esub,
        ts_sub, amt_sub,
        train_sub, val_sub, test_sub,
        num_channels=3, include_amount_channel=True,
        device=device,
    )
    data.dataset_name = "ETX"
    data.node_ids = subgraph_node_ids_int
    data.raw_edgelist = esub
    data.raw_timestamps = ts_sub
    data.raw_amounts = amt_sub
    return e, data


def _compute_etx_features(dataset, all_nodes, node_mask, n_sub):
    feats = np.zeros((n_sub, 14), dtype=np.float32)
    sub_nodes = all_nodes[node_mask]
    for i, n in enumerate(sub_nodes):
        indeg = dataset._node_in_deg.get(n, 0)
        outdeg = dataset._node_out_deg.get(n, 0)
        in_amt = np.array(dataset._node_in_amount.get(n, [0]))
        out_amt = np.array(dataset._node_out_amount.get(n, [0]))
        first_ts = dataset._node_first_tx.get(n, 0)
        last_ts = dataset._node_last_tx.get(n, 0)

        feats[i, 0] = indeg
        feats[i, 1] = outdeg
        feats[i, 2] = np.sum(in_amt)
        feats[i, 3] = np.sum(out_amt)
        feats[i, 4] = np.mean(in_amt) if len(in_amt) > 0 else 0
        feats[i, 5] = np.mean(out_amt) if len(out_amt) > 0 else 0
        feats[i, 6] = np.max(in_amt) if len(in_amt) > 0 else 0
        feats[i, 7] = np.max(out_amt) if len(out_amt) > 0 else 0
        feats[i, 8] = len(dataset._node_counterparties_in.get(n, set()))
        feats[i, 9] = len(dataset._node_counterparties_out.get(n, set()))
        feats[i, 10] = indeg + outdeg
        feats[i, 11] = last_ts - first_ts
        feats[i, 12] = (last_ts - first_ts) / max(indeg + outdeg, 1)
        feats[i, 13] = feats[i, 2] / max(feats[i, 3], 1e-8)
    return feats


def load_rd_bitcoin(seed=42, sampling_ratio=0.1, device=DEVICE):
    e = RDBitcoinDataset(os.path.join(DATA_DIR, "RD bitcoin"), seed=seed,
                         sampling_ratio=sampling_ratio)

    rng = np.random.default_rng(seed)
    nm, em, sid = construct_subgraph_from_edgelist(
        e.tx_ids, e.labels, e.adjacency_list, rng
    )
    si = np.where(nm)[0]
    esub = e.adjacency_list[em]
    tx_to_ts = dict(zip(e.tx_ids, e.time_step))
    ts = np.array([tx_to_ts.get(s, 0) for s in esub[:, 0]])
    amt = np.zeros(len(esub), dtype=np.float64)

    data = prepare_pyg_data(
        sid, e.node_features[nm], e.labels[si], esub, ts, amt,
        e.train_mask[si], e.val_mask[si], e.test_mask[si],
        num_channels=2, include_amount_channel=False,
        device=device,
    )
    data.dataset_name = "RD_Bitcoin"
    data.node_ids = sid
    data.raw_edgelist = esub
    data.raw_timestamps = ts
    data.raw_amounts = amt

    x_np = data.x.cpu().numpy()
    nan_mask = np.isnan(x_np)
    if nan_mask.any():
        col_means = np.nanmean(x_np, axis=0)
        col_means = np.nan_to_num(col_means, nan=0.0)
        x_np[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        data.x = torch.tensor(x_np, dtype=torch.float32, device=data.x.device)

    return e, data
