import argparse
import json
import os
import sys
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import (
    Config, build_model, evaluate, load_dataset, apply_btrw,
    add_btrw_to_features, set_seed, model_forward, VARIANTS,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--dataset", choices=["elliptic", "etx", "rd_bitcoin"],
                        default="elliptic")
    parser.add_argument("--variant", choices=VARIANTS, default="MCMF-GTN")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = Config()
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(cfg.seed)
    data = load_dataset(cfg, args.dataset, cfg.seed, device)

    if args.variant == "MCMF-GTN":
        emb = apply_btrw(cfg, args.dataset, data, cfg.seed, device)
        fused = add_btrw_to_features(data.x.cpu().numpy(), emb, mode="concat")
        data.x = torch.tensor(fused, dtype=torch.float32, device=device)

    model = build_model(cfg, args.variant, data.x.shape[1],
                        data.num_adj_channels, device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    metrics = evaluate(model, data)
    print(f"=== {args.variant} on {args.dataset} ===")
    print(f"checkpoint: {args.checkpoint}")
    for k, v in metrics.items():
        print(f"  {k:12s} {v}")

    with torch.no_grad():
        logits = torch.nan_to_num(model_forward(model, data), nan=0.0)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    os.makedirs(cfg.output_dir, exist_ok=True)
    out = os.path.join(cfg.output_dir,
                       f"predictions_{args.dataset}_{args.variant}_"
                       f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w") as f:
        json.dump({"metrics": metrics,
                   "prob_class1": probs[:, 1].round(6).tolist()}, f, indent=2)
    print(f"Saved predictions: {out}")


if __name__ == "__main__":
    main()
