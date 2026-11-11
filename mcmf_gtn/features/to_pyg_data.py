import numpy as np
import torch
from torch_geometric.data import Data


def build_sparse_adj_channels(edge_index, edge_attr, num_nodes, device):
    C = edge_attr.size(1)
    adj_channels = []
    for c in range(C):
        adj = torch.sparse_coo_tensor(
            edge_index,
            edge_attr[:, c],
            size=(num_nodes, num_nodes),
        ).coalesce().to(device)
        adj_channels.append(adj)
    return adj_channels


def prepare_pyg_data(
    subgraph_node_ids,
    node_features,
    labels,
    edgelist,
    timestamps,
    amounts,
    train_mask,
    val_mask,
    test_mask,
    num_channels=3,
    include_amount_channel=False,
    epsilon=1e-8,
    device=None,
):
    """Build a PyG Data object with edge_index, edge_attr, sparse adjacency
    channels, labels, and train/val/test masks."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    node_to_idx = {nid: i for i, nid in enumerate(subgraph_node_ids)}
    n = len(subgraph_node_ids)

    src = np.array([node_to_idx[s] for s, d in edgelist])
    dst = np.array([node_to_idx[d] for s, d in edgelist])
    edge_index_np = np.stack([src, dst], axis=0)

    ts = timestamps.astype(np.float64)
    ts = (ts - ts.min()) / (ts.max() - ts.min() + epsilon)

    if include_amount_channel and num_channels >= 3:
        amt = amounts.astype(np.float64)
        amt = amt / (amt.max() + epsilon)
        edge_attr_np = np.stack([
            np.ones(len(edgelist), dtype=np.float32),
            ts.astype(np.float32),
            amt.astype(np.float32),
        ], axis=1)
    else:
        edge_attr_np = np.stack([
            np.ones(len(edgelist), dtype=np.float32),
            ts.astype(np.float32),
        ], axis=1)

    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_np, dtype=torch.float32)

    C = edge_attr_np.shape[1]
    adj_channels = build_sparse_adj_channels(edge_index, edge_attr, n, device)

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(labels, dtype=torch.long),
    )
    data.adj_channels = adj_channels
    data.num_adj_channels = C

    data.train_mask = torch.tensor(train_mask)
    data.val_mask = torch.tensor(val_mask)
    data.test_mask = torch.tensor(test_mask)

    return data.to(device)
