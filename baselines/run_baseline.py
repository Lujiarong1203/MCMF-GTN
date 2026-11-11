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

_BASE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_BASE, "..")
sys.path.insert(0, _BASE)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, _ROOT)

from mcmf_gtn.config import Config
from mcmf_gtn.train_eval.metrics import compute_all_metrics
from train import load_dataset, set_seed


def load_adapter(name):
    if name == "tgtod":
        from adapters.tgtod import build_model, forward
    elif name == "polynormer":
        from adapters.polynormer import build_model, forward
    elif name == "dualformer":
        from adapters.dualformer import build_model, forward
    else:
        raise ValueError(f"Unknown baseline: {name}")
    return build_model, forward


def evaluate_model(model, forward, data):
    model.eval()
    with torch.no_grad():
        logits = torch.nan_to_num(forward(model, data), nan=0.0)
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


def train_model(model, forward, data, cfg, seed):
    set_seed(seed)
    nt = data.train_mask.sum().item()
    ti = data.train_mask.nonzero().squeeze()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_metrics, best_state, train_times = None, None, []

    for ep in range(cfg.epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(nt, device=data.x.device)
        for bi in range(0, nt, cfg.batch_size):
            bidx = ti[perm[bi:bi + cfg.batch_size]]
            opt.zero_grad()
            logits = torch.nan_to_num(forward(model, data), nan=0.0)
            loss = F.cross_entropy(logits[bidx], data.y[bidx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        train_times.append(time.time() - t0)
        if (ep + 1) % 50 == 0 or ep == cfg.epochs - 1:
            m = evaluate_model(model, forward, data)
            if best_metrics is None or m["val_f1"] > best_metrics["val_f1"]:
                best_metrics = m
                best_state = {k: v.clone() if torch.is_tensor(v) else v
                              for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return {"train_time_per_epoch": float(np.mean(train_times)),
            "metrics": best_metrics}


def main():
    parser = argparse.ArgumentParser(description="Baseline training under the shared protocol")
    parser.add_argument("--dataset", choices=["elliptic", "etx", "rd_bitcoin"],
                        default="elliptic")
    parser.add_argument("--baseline", choices=["tgtod", "polynormer", "dualformer"],
                        required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = Config()
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_dataset(cfg, args.dataset, cfg.seed, device)
    build_model, forward = load_adapter(args.baseline)

    results = {}
    for seed in cfg.seeds:
        set_seed(seed)
        print(f"=== {args.baseline} / seed {seed} ===")
        model = build_model(data.x.shape[1], data.num_adj_channels).to(device)
        info = train_model(model, forward, data, cfg, seed)
        results[str(seed)] = info
        m = info["metrics"]
        print(f"  F1={m['f1']:.4f} AUC={m['auc']:.4f} "
              f"val_F1={m['val_f1']:.4f} threshold={m['threshold']:.2f}")

    print("=" * 60)
    print(f"{args.baseline} on {args.dataset}: {len(cfg.seeds)} seeds")
    print("=" * 60)
    for metric in ["f1", "auc", "pr_auc", "mcc", "accuracy", "balanced_acc"]:
        vals = [results[s]["metrics"][metric] for s in results]
        print(f"  {metric:14s} {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    out = os.path.join(cfg.output_dir,
                       f"{args.dataset}_{args.baseline}_"
                       f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
