import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from mcmf_gtn import __version__
from mcmf_gtn.config import IMPLEMENTATION_SETTINGS, VARIANTS, dataset_spec, load_config
from mcmf_gtn.data.protocol import prepare_data
from mcmf_gtn.data.readers import load_dataset
from mcmf_gtn.demo import create_demo_graph
from mcmf_gtn.evaluation import evaluate_imbalance, evaluate_robustness, read_predictions
from mcmf_gtn.metrics import summarize_metrics
from mcmf_gtn.training import checkpoint_graph, load_checkpoint, predict_with_checkpoint, run_experiment, write_json, write_predictions


def parser():
    root = argparse.ArgumentParser(prog="mcmf-gtn")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("config", "prepare", "train", "smoke"):
        command = commands.add_parser(name)
        command.add_argument("--config")
        command.add_argument("--dataset", choices=("elliptic", "etx", "rd", "all") if name == "smoke" else ("elliptic", "etx", "rd"), default="all" if name == "smoke" else None)
        command.add_argument("--device", choices=("auto", "cpu", "cuda"))
        command.add_argument("--threads", type=int, default=2)
        if name != "config":
            command.add_argument("--out", required=True)
        if name in ("prepare", "train"):
            command.add_argument("--data", required=True)
            command.add_argument("--trusted-pickle", action="store_true")
        if name in ("train", "smoke"):
            command.add_argument("--seeds", nargs="+", type=int)
            command.add_argument("--variant", choices=VARIANTS, default=None)
            command.add_argument("--all-variants", action="store_true")
            command.add_argument("--accept-implementation-defaults", action="store_true", default=None)
    for name in ("evaluate", "robustness"):
        command = commands.add_parser(name)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--data", required=True)
        command.add_argument("--out", required=True)
        command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
        command.add_argument("--threads", type=int, default=2)
        command.add_argument("--trusted-pickle", action="store_true")
        if name == "evaluate":
            command.add_argument("--all-test-targets", action="store_true")
        else:
            command.add_argument("--perturbation-seed", type=int)
    command = commands.add_parser("imbalance")
    command.add_argument("--predictions", required=True, nargs="+")
    command.add_argument("--ratios", type=int, nargs="+", default=(1, 5, 10))
    command.add_argument("--positive-count", type=int, default=300)
    command.add_argument("--sampling-seed", type=int, default=42)
    command.add_argument("--out", required=True)
    return root


def _configuration(args):
    overrides = {}
    if args.command == "smoke":
        overrides.update(epochs=2, batch_size=16, gcn_hidden=(16, 8, 4), embed_dim=8, top_k=4, walk_length=5, num_walks=2, window=2, negative_samples=2, skipgram_epochs=1, inference_epochs=2, inference_policy="frozen_context", profile="smoke", device="cpu")
    overrides.update({name: getattr(args, name, None) for name in ("device", "variant")})
    if args.dataset != "all":
        overrides["dataset"] = args.dataset
    if getattr(args, "accept_implementation_defaults", None):
        overrides["acknowledged_defaults"] = True
    return load_config(args.config, **overrides)


def _runs(args, cfg):
    datasets = ("elliptic", "etx", "rd") if args.command == "smoke" and args.dataset == "all" else (cfg.dataset,)
    variants = VARIANTS if args.all_variants else (cfg.variant,)
    seeds = tuple(args.seeds) if args.seeds else cfg.seeds
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be unique nonnegative integers")
    output = Path(args.out)
    if (output / "summary.json").exists():
        raise ValueError("output already contains a summary; choose a fresh directory")
    rows = []
    for dataset in datasets:
        raw = create_demo_graph(dataset) if args.command == "smoke" else load_dataset(dataset, args.data, args.trusted_pickle)
        for variant in variants:
            if args.all_variants and variant == "no_amount" and dataset != "etx":
                continue
            for seed in seeds:
                run_cfg = cfg.with_updates(dataset=dataset, variant=variant, seed=seed, seeds=seeds)
                directory = output / dataset / variant / f"seed_{seed}"
                outcome = run_experiment(raw, run_cfg, directory)
                if args.command == "smoke":
                    loaded = load_checkpoint(outcome["checkpoint"], run_cfg.device)
                    restored = predict_with_checkpoint(raw, loaded)
                    original_scores = [row["score"] for row in outcome["predictions"]]
                    restored_scores = [row["score"] for row in restored["predictions"]]
                    if not np.allclose(original_scores, restored_scores, atol=1e-6, rtol=0):
                        raise ValueError("checkpoint reload prediction mismatch")
                    write_json(directory / "reload_check.json", {"passed": True, "max_absolute_error": float(np.max(np.abs(np.array(original_scores) - restored_scores)))})
                rows.append({"dataset": dataset, "variant": variant, "seed": seed, "metrics": outcome["metrics"], "directory": str(directory)})
                print(f"completed {dataset}/{variant}/seed_{seed}", file=sys.stderr, flush=True)
    groups = {}
    for dataset in datasets:
        for variant in variants:
            selected = [row["metrics"] for row in rows if row["dataset"] == dataset and row["variant"] == variant]
            if selected:
                groups[f"{dataset}/{variant}"] = summarize_metrics(selected)
    report = {"status": "synthetic_smoke" if args.command == "smoke" else "reference_runs", "std_definition": "ddof_1_across_seeds", "runs": rows, "summary": groups}
    write_json(output / "summary.json", report)
    return {"status": report["status"], "runs": len(rows), "summary": str(output / "summary.json")}


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if hasattr(args, "threads"):
            if args.threads < 1:
                raise ValueError("threads must be positive")
            torch.set_num_threads(args.threads)
        if hasattr(args, "out"):
            output = Path(args.out)
            if output.exists() and (not output.is_dir() or any(output.iterdir())):
                raise ValueError("output directory is not empty; choose a fresh directory")
        if args.command in ("config", "prepare", "train", "smoke"):
            cfg = _configuration(args)
            if args.command == "config":
                spec = dataset_spec(cfg.dataset)
                result = {"config": cfg.to_dict(), "dataset": {"static_dim": spec.static_dim, "channels": spec.channels, "fused_dim": spec.static_dim + cfg.embed_dim}, "implementation_settings": IMPLEMENTATION_SETTINGS}
            elif args.command == "prepare":
                raw = load_dataset(cfg.dataset, args.data, args.trusted_pickle)
                result = prepare_data(raw, cfg).stats
                output = Path(args.out)
                output.mkdir(parents=True, exist_ok=True)
                write_json(output / "split_stats.json", result)
            else:
                result = _runs(args, cfg)
        elif args.command in ("evaluate", "robustness"):
            loaded = load_checkpoint(args.checkpoint, args.device)
            raw = load_dataset(loaded["config"].dataset, args.data, args.trusted_pickle)
            output = Path(args.out)
            output.mkdir(parents=True, exist_ok=True)
            if args.command == "evaluate":
                result = predict_with_checkpoint(raw, loaded, args.all_test_targets)
                write_predictions(output / "predictions.csv", result.pop("predictions"))
            else:
                result = evaluate_robustness(loaded["model"], checkpoint_graph(raw, loaded), loaded["config"], args.perturbation_seed)
            write_json(output / "metrics.json", result)
        else:
            result = evaluate_imbalance([read_predictions(path) for path in args.predictions], tuple(args.ratios), args.positive_count, args.sampling_seed)
            output = Path(args.out)
            output.mkdir(parents=True, exist_ok=True)
            write_json(output / "imbalance.json", result)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, FileNotFoundError, OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
