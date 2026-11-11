import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from mcmf_gtn.config import Config
from mcmf_gtn.data.elliptic import EllipticDataset
from mcmf_gtn.data.loaders import load_etx, load_rd_bitcoin
from mcmf_gtn.features.btrw_embeddings import (
    generate_btrw_embeddings, add_btrw_to_features,
)
from mcmf_gtn.features.subgraph import construct_subgraph_from_edgelist
from mcmf_gtn.features.to_pyg_data import prepare_pyg_data
from mcmf_gtn.models.ef_gcn import EFGCN
from mcmf_gtn.models.gat_baseline import GATClassifier
from mcmf_gtn.models.gcn_classifier import GCNClassifier
from mcmf_gtn.models.mcmf_gtn import MCMFGTN
from mcmf_gtn.train_eval.metrics import compute_all_metrics, count_parameters

VARIANTS = ["MCMF-GTN", "MCMF-GCN", "EFGCN-FIXED", "EFGCN-DIRECT", "GCN", "MLP", "GAT"]


class MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden=(128, 64, 32), num_classes=2, dropout=0.5):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [torch.nn.Linear(prev, h), torch.nn.BatchNorm1d(h),
                       torch.nn.ReLU(), torch.nn.Dropout(dropout)]
            prev = h
        layers.append(torch.nn.Linear(hidden[-1], num_classes))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x, *args, **kwargs):
        return self.net(x)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def build_model(cfg, variant, in_dim, num_channels, device):
    gcn_hidden = tuple(cfg.gcn_hidden)
    if variant == "MCMF-GTN":
        model = MCMFGTN(in_dim, num_channels=num_channels, gt_layers=cfg.gt_layers,
                        gcn_hidden=gcn_hidden, dropout=cfg.dropout,
                        use_matrix_mult=True, gt_top_k=cfg.gt_top_k)
    elif variant == "MCMF-GCN":
        model = MCMFGTN(in_dim, num_channels=num_channels, gt_layers=cfg.gt_layers,
                        gcn_hidden=gcn_hidden, dropout=cfg.dropout,
                        use_matrix_mult=False)
    elif variant == "EFGCN-FIXED":
        model = EFGCN(in_dim, num_channels, "fixed", gcn_hidden=gcn_hidden,
                      num_classes=2, dropout=cfg.dropout)
    elif variant == "EFGCN-DIRECT":
        model = EFGCN(in_dim, num_channels, "direct", gcn_hidden=gcn_hidden,
                      num_classes=2, dropout=cfg.dropout)
    elif variant == "GCN":
        model = GCNClassifier(in_dim, gcn_hidden, 2, cfg.dropout, use_residual=True)
    elif variant == "MLP":
        model = MLP(in_dim, gcn_hidden, 2, cfg.dropout)
    elif variant == "GAT":
        model = GATClassifier(in_dim)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return model.to(device)


def model_forward(model, data):
    if isinstance(model, MCMFGTN):
        if model.use_matrix_mult:
            out = model(data.x, data.adj_channels)
        else:
            out = model(data.x, data.edge_index, data.edge_attr[:, :model.num_channels])
    elif isinstance(model, EFGCN):
        out = model(data.x, data.edge_index, data.edge_attr[:, :data.num_adj_channels])
    elif isinstance(model, (GCNClassifier, GATClassifier)):
        out = model(data.x, data.edge_index)
    else:
        out = model(data.x)
    return torch.nan_to_num(out, nan=0.0)


def evaluate(model, data):
    """Threshold selected on the validation split; metrics on the test split."""
    model.eval()
    with torch.no_grad():
        logits = torch.nan_to_num(model_forward(model, data), nan=0.0)
        probs = torch.softmax(logits, dim=1)

    vp = probs[data.val_mask][:, 1].cpu().numpy()
    vy = data.y[data.val_mask].cpu().numpy()
    best_th, best_f1 = 0.5, -1.0
    for th in np.arange(0.10, 0.95, 0.05):
        pr = (vp > th).astype(int)
        f = f1_score(vy, pr, zero_division=0)
        if f > best_f1:
            best_f1, best_th = f, th

    tp = probs[data.test_mask][:, 1].cpu().numpy()
    ty = data.y[data.test_mask].cpu().numpy()
    tr = (tp > best_th).astype(int)
    pp = np.stack([1 - tp, tp], axis=1)
    m = compute_all_metrics(ty, tr, pp)
    m["threshold"] = float(best_th)
    m["val_f1"] = float(best_f1)
    return m


def train_model(model, data, cfg, seed):
    """100 epochs; the checkpoint with the best validation F1 among the
    evaluation snapshots (epochs 50 and 100) is restored."""
    set_seed(seed)
    nt = data.train_mask.sum().item()
    ti = data.train_mask.nonzero().squeeze()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    best_metrics, best_state, train_times = None, None, []

    for ep in range(cfg.epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(nt, device=data.x.device)
        for bi in range(0, nt, cfg.batch_size):
            bidx = ti[perm[bi:bi + cfg.batch_size]]
            opt.zero_grad()
            logits = model_forward(model, data)
            loss = F.cross_entropy(logits[bidx], data.y[bidx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        train_times.append(time.time() - t0)
        if (ep + 1) % 50 == 0 or ep == cfg.epochs - 1:
            m = evaluate(model, data)
            if best_metrics is None or m["val_f1"] > best_metrics["val_f1"]:
                best_metrics = m
                best_state = {k: v.clone() if torch.is_tensor(v) else v
                              for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)

    info = {"params": count_parameters(model)["trainable"],
            "train_time_per_epoch": float(np.mean(train_times)),
            "memory_mb": float(torch.cuda.max_memory_allocated() / 1e6)
            if torch.cuda.is_available() else 0,
            "metrics": best_metrics}
    return info


def scale_features_train_fit(node_feats, train_mask, val_mask, test_mask):
    """Feature normalization fitted on the training split only."""
    scaler = StandardScaler()
    node_feats[train_mask] = scaler.fit_transform(node_feats[train_mask])
    if val_mask.sum() > 0:
        node_feats[val_mask] = scaler.transform(node_feats[val_mask])
    if test_mask.sum() > 0:
        node_feats[test_mask] = scaler.transform(node_feats[test_mask])
    return node_feats


def load_elliptic(cfg, seed, device):
    e = EllipticDataset(os.path.join(cfg.data_dir, "Elliptic"), seed=seed)
    rng = np.random.default_rng(seed)
    nm, em, sid = construct_subgraph_from_edgelist(
        e.tx_ids, e.labels, e.adjacency_list, rng)
    si = np.where(nm)[0]
    esub = e.adjacency_list[em]
    tx_to_ts = dict(zip(e.tx_ids, e.time_step))
    ts = np.array([tx_to_ts.get(s, 0) for s in esub[:, 0]])
    amt = np.zeros(len(esub), dtype=np.float64)

    node_feats = e.node_features[nm]
    node_feats = scale_features_train_fit(
        node_feats, e.train_mask[si], e.val_mask[si], e.test_mask[si])

    data = prepare_pyg_data(
        sid, node_feats, e.labels[si], esub, ts, amt,
        e.train_mask[si], e.val_mask[si], e.test_mask[si],
        num_channels=2, include_amount_channel=False, device=device)
    data.dataset_name = "Elliptic"
    data.node_ids = sid
    data.raw_edgelist = esub
    data.raw_timestamps = ts
    data.raw_amounts = amt
    return data


def load_dataset(cfg, dataset, seed, device):
    if dataset == "elliptic":
        return load_elliptic(cfg, seed, device)
    if dataset == "etx":
        _, data = load_etx(seed=seed, device=device)
    elif dataset == "rd_bitcoin":
        _, data = load_rd_bitcoin(seed=seed, device=device)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    data.dataset_name = dataset
    x_np = data.x.cpu().numpy()
    data.x = torch.tensor(
        scale_features_train_fit(x_np, data.train_mask.cpu().numpy(),
                                 data.val_mask.cpu().numpy(),
                                 data.test_mask.cpu().numpy()),
        dtype=torch.float32, device=device)
    return data


def apply_btrw(cfg, dataset, data, seed, device):
    """Walks over the complete subgraph; time+amount bias on 3-channel
    datasets, time-only bias on 2-channel datasets."""
    num_channels = data.num_adj_channels
    emb = generate_btrw_embeddings(
        data.node_ids, data.raw_edgelist,
        data.raw_timestamps, data.raw_amounts,
        embed_dim=cfg.embed_dim,
        walk_length=cfg.walk_length,
        num_walks=cfg.num_walks,
        alpha=cfg.btrw_alpha,
        window=cfg.window_size,
        seed=seed,
        cache_dir=os.path.join(cfg.output_dir, dataset),
        device=str(device),
        use_amount=(num_channels == 3),
    )
    return emb


def main():
    parser = argparse.ArgumentParser(description="MCMF-GTN training")
    parser.add_argument("--dataset", choices=["elliptic", "etx", "rd_bitcoin"],
                        default="elliptic")
    parser.add_argument("--variant", choices=VARIANTS, default="MCMF-GTN")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = Config()
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, dataset={args.dataset}, variant={args.variant}, "
          f"epochs={cfg.epochs}, lr={cfg.lr}, batch={cfg.batch_size}")

    data = load_dataset(cfg, args.dataset, cfg.seed, device)

    if args.variant == "MCMF-GTN":
        emb = apply_btrw(cfg, args.dataset, data, cfg.seed, device)
        fused = add_btrw_to_features(data.x.cpu().numpy(), emb, mode="concat")
        data.x = torch.tensor(fused, dtype=torch.float32, device=device)

    in_dim = data.x.shape[1]
    num_channels = data.num_adj_channels
    print(f"Data: {data.x.shape[0]} nodes, {in_dim} features, "
          f"{data.edge_index.size(1)} edges, {num_channels} channels")

    results = {}
    for seed in cfg.seeds:
        set_seed(seed)
        print(f"=== {args.variant} / seed {seed} ===")
        model = build_model(cfg, args.variant, in_dim, num_channels, device)
        info = train_model(model, data, cfg, seed)
        results[str(seed)] = info

        m = info["metrics"]
        print(f"  F1={m['f1']:.4f} AUC={m['auc']:.4f} "
              f"val_F1={m['val_f1']:.4f} threshold={m['threshold']:.2f}")

    print("=" * 60)
    print(f"{args.variant} on {args.dataset}: {len(cfg.seeds)} seeds")
    print("=" * 60)
    for metric in ["f1", "auc", "pr_auc", "mcc", "accuracy", "balanced_acc"]:
        vals = [results[s]["metrics"][metric] for s in results]
        print(f"  {metric:14s} {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(cfg.output_dir, f"{args.dataset}_{args.variant}_{ts}.json")
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
