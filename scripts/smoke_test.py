import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from mcmf_gtn.config import Config
from mcmf_gtn.features.btrw_embeddings import (
    generate_btrw_embeddings, add_btrw_to_features,
)
from mcmf_gtn.features.subgraph import construct_subgraph_from_edgelist
from mcmf_gtn.features.to_pyg_data import prepare_pyg_data
from mcmf_gtn.train_eval.metrics import compute_all_metrics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import build_model, model_forward, evaluate, train_model, set_seed


def make_synthetic_graph(n_nodes=400, n_edges=3000, seed=42):
    rng = np.random.default_rng(seed)
    src = rng.integers(0, n_nodes, n_edges)
    dst = rng.integers(0, n_nodes, n_edges)
    edgelist = np.column_stack([src, dst])
    timestamps = np.sort(rng.uniform(0, 1000, n_edges))
    amounts = rng.uniform(0, 100, n_edges)

    labels = rng.integers(0, 2, n_nodes).astype(np.int64)
    features = rng.normal(size=(n_nodes, 20)).astype(np.float32)

    train = np.zeros(n_nodes, dtype=bool)
    val = np.zeros(n_nodes, dtype=bool)
    test = np.zeros(n_nodes, dtype=bool)
    train[:240] = True
    val[240:300] = True
    test[300:] = True
    return edgelist, timestamps, amounts, labels, features, train, val, test


def build_smoke_data(seed=42, num_channels=3):
    edgelist, timestamps, amounts, labels, features, train, val, test = \
        make_synthetic_graph(seed=seed)
    rng = np.random.default_rng(seed)
    node_ids = np.arange(len(labels))
    nm, em, sid = construct_subgraph_from_edgelist(node_ids, labels, edgelist, rng)
    si = np.where(nm)[0]
    data = prepare_pyg_data(
        sid, features[nm], labels[si], edgelist[em],
        timestamps[em], amounts[em],
        train[si], val[si], test[si],
        num_channels=num_channels,
        include_amount_channel=(num_channels == 3),
        device=torch.device("cpu"),
    )
    data.node_ids = sid
    data.raw_edgelist = edgelist[em]
    data.raw_timestamps = timestamps[em]
    data.raw_amounts = amounts[em]
    return data


def main():
    cfg = Config()
    cfg.epochs = 4
    cfg.seeds = [42]
    cfg.batch_size = 64
    cfg.gcn_hidden = [16, 8]
    cfg.embed_dim = 8
    cfg.walk_length = 6
    cfg.num_walks = 2
    cfg.window_size = 3
    cfg.gt_top_k = 5
    cfg.output_dir = "outputs/smoke"
    device = torch.device("cpu")

    data = build_smoke_data(seed=cfg.seed, num_channels=3)
    print(f"smoke data: {data.x.shape[0]} nodes, {data.edge_index.size(1)} edges, "
          f"{data.num_adj_channels} channels")

    emb = generate_btrw_embeddings(
        data.node_ids, data.raw_edgelist, data.raw_timestamps, data.raw_amounts,
        embed_dim=cfg.embed_dim, walk_length=cfg.walk_length,
        num_walks=cfg.num_walks, alpha=cfg.btrw_alpha, window=cfg.window_size,
        epochs=1, seed=cfg.seed, cache_dir=cfg.output_dir, device="cpu",
        use_amount=(data.num_adj_channels == 3), verbose=False,
    )
    assert emb.shape == (data.x.shape[0], cfg.embed_dim)
    fused = add_btrw_to_features(data.x.cpu().numpy(), emb, mode="concat")
    data.x = torch.tensor(fused, dtype=torch.float32)

    for variant in ["MCMF-GTN", "MCMF-GCN", "EFGCN-FIXED", "EFGCN-DIRECT",
                    "GCN", "MLP", "GAT"]:
        set_seed(cfg.seed)
        model = build_model(cfg, variant, data.x.shape[1],
                            data.num_adj_channels, device)
        info = train_model(model, data, cfg, cfg.seed)
        m = info["metrics"]
        assert set(["f1", "auc", "val_f1", "threshold"]).issubset(m.keys())
        print(f"  {variant:14s} F1={m['f1']:.4f} val_F1={m['val_f1']:.4f} "
              f"th={m['threshold']:.2f}")

    # recent baselines via the shared baseline harness
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baselines"))
    from run_baseline import train_model as baseline_train_model
    from adapters.tgtod import build_model as build_tgtod, forward as tgtod_forward
    from adapters.polynormer import build_model as build_poly, forward as poly_forward
    from adapters.dualformer import build_model as build_dualformer, forward as df_forward

    for name, build, fwd in [("tgtod", build_tgtod, tgtod_forward),
                             ("polynormer", build_poly, poly_forward),
                             ("dualformer", build_dualformer, df_forward)]:
        set_seed(cfg.seed)
        b_model = build(data.x.shape[1], data.num_adj_channels)
        b_info = baseline_train_model(b_model, fwd, data, cfg, cfg.seed)
        print(f"  {name:14s} F1={b_info['metrics']['f1']:.4f} "
              f"val_F1={b_info['metrics']['val_f1']:.4f} "
              f"th={b_info['metrics']['threshold']:.2f}")

    # threshold sweep sanity
    y_true = np.array([0, 0, 1, 1, 1, 1])
    y_prob = np.stack([1 - np.array([.9, .8, .7, .4, .3, .1]),
                       np.array([.9, .8, .7, .4, .3, .1])], axis=1)
    met = compute_all_metrics(y_true, (y_prob[:, 1] > 0.5).astype(int), y_prob)
    assert 0 <= met["f1"] <= 1 and 0 <= met["auc"] <= 1

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
