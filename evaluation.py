import csv
from pathlib import Path

import numpy as np
import torch

from mcmf_gtn.config import dataset_spec
from mcmf_gtn.metrics import compute_metrics, summarize_metrics
from mcmf_gtn.pipeline import TensorGraph
from mcmf_gtn.training import predict


def read_predictions(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    return [{"node_id": row["node_id"], "label": int(row["label"]), "score": float(row["score"]), "prediction": int(row["prediction"])} for row in records]


def _pool_arrays(records):
    ids = [str(row["node_id"]) for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("prediction pool contains duplicate node IDs")
    labels = np.array([row["label"] for row in records], dtype=int)
    scores = np.array([row["score"] for row in records], dtype=float)
    predictions = np.array([row["prediction"] for row in records], dtype=int)
    compute_metrics(labels, scores, predictions)
    return ids, labels, scores, predictions


def imbalance_masks(records, ratios=(1, 5, 10), positive_count=None, seed=42):
    _, labels, _, _ = _pool_arrays(records)
    ratios = sorted(set(ratios))
    if not ratios or any(isinstance(r, bool) or not isinstance(r, int) or r < 1 for r in ratios):
        raise ValueError("negative-to-positive ratios must be positive integers")
    positives, negatives = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    positive_count = len(positives) if positive_count is None else positive_count
    if not isinstance(positive_count, int) or isinstance(positive_count, bool) or not 0 < positive_count <= len(positives):
        raise ValueError(f"positive count exceeds the eligible pool: {len(positives)} available")
    if positive_count * max(ratios) > len(negatives):
        raise ValueError(f"insufficient negative pool: {positive_count * max(ratios)} required, {len(negatives)} available; no replacement is performed")
    generator = np.random.default_rng(seed)
    positives = generator.permutation(positives)[:positive_count]
    negatives = generator.permutation(negatives)
    return {ratio: np.concatenate([positives, negatives[:ratio * positive_count]]).tolist() for ratio in ratios}


def evaluate_imbalance(prediction_runs, ratios=(1, 5, 10), positive_count=None, seed=42):
    if not prediction_runs:
        raise ValueError("at least one prediction file is required")
    ids, labels, _, _ = _pool_arrays(prediction_runs[0])
    aligned = []
    for records in prediction_runs:
        run_ids, run_labels, scores, predictions = _pool_arrays(records)
        lookup = {node: i for i, node in enumerate(run_ids)}
        if set(run_ids) != set(ids):
            raise ValueError("each run must use the same eligible prediction pool")
        order = [lookup[node] for node in ids]
        if not np.array_equal(run_labels[order], labels):
            raise ValueError("prediction pool labels differ between runs")
        aligned.append((scores[order], predictions[order]))
    masks = imbalance_masks(prediction_runs[0], ratios, positive_count, seed)
    conditions = []
    for ratio, indices in masks.items():
        selected = labels[indices]
        runs = [compute_metrics(selected, scores[indices], predictions[indices]) for scores, predictions in aligned]
        conditions.append({"illicit_to_licit": f"1:{ratio}", "positive_count": int(selected.sum()), "negative_count": int(len(selected) - selected.sum()), "target_ids": [ids[i] for i in indices], "per_run": runs, "metrics": summarize_metrics(runs)})
    return {"protocol": "frozen_predictions_fixed_positives_nested_negative_subsets", "sampling_seed": seed, "pool_positive_count": int(labels.sum()), "pool_negative_count": int(len(labels) - labels.sum()), "std_definition": "sample_standard_deviation_ddof_1_across_model_runs", "conditions": conditions}


def perturb_inputs(graph, static_dim, edge_dropout=0., noise_sigma=0., seed=42):
    if not 0 <= edge_dropout <= 1 or not np.isfinite(noise_sigma) or noise_sigma < 0:
        raise ValueError("invalid perturbation strength")
    if not isinstance(static_dim, int) or not 0 <= static_dim <= graph.x.shape[1]:
        raise ValueError("invalid static feature dimension")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    keep = (torch.rand(graph.edge_index.shape[1], generator=generator) >= edge_dropout).to(graph.edge_index.device)
    x = graph.x.clone()
    if noise_sigma:
        noise = torch.randn((len(x), static_dim), generator=generator, dtype=x.dtype).to(x.device)
        x[:, :static_dim] += noise_sigma * noise
    return TensorGraph(graph.node_ids.copy(), x, graph.edge_index[:, keep].clone(), graph.edge_attr[keep].clone(), graph.y.clone(), graph.target_mask.clone())


def evaluate_robustness(model, graph, cfg, seed=None):
    seed = cfg.seed if seed is None else seed
    static_dim = 0 if cfg.variant == "embedding_only" else dataset_spec(cfg.dataset).static_dim
    if not static_dim:
        raise ValueError("feature-noise evaluation requires the static feature branch")
    settings = (("clean", 0., 0.), ("edge_dropout_0.1", .1, 0.), ("edge_dropout_0.3", .3, 0.), ("feature_noise_0.1", 0., .1), ("feature_noise_0.3", 0., .3))
    conditions, baseline = [], None
    for name, dropout, sigma in settings:
        perturbed = perturb_inputs(graph, static_dim, dropout, sigma, seed)
        metrics, _ = predict(model, perturbed, cfg.decision_threshold)
        baseline = metrics if baseline is None else baseline
        conditions.append({"condition": name, "metrics": metrics, "delta_f1_pp": 100 * (metrics["f1"] - baseline["f1"]), "delta_roc_auc_pp": 100 * (metrics["roc_auc"] - baseline["roc_auc"]) if metrics["roc_auc"] is not None and baseline["roc_auc"] is not None else None})
    return {"protocol": "test_only_frozen_checkpoint", "perturbation_seed": seed, "edge_channels_rebuilt": True, "embeddings": "held_fixed", "noise_space": "training_transformed_static_features", "retrained": False, "conditions": conditions}
