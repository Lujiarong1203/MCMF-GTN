import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mcmf_gtn.config import IMPLEMENTATION_SETTINGS, load_config, validate_config
from mcmf_gtn.data.protocol import prepare_data, prepare_evaluation_graph
from mcmf_gtn.features.skipgram import EmbeddingState
from mcmf_gtn.metrics import compute_metrics
from mcmf_gtn.models.network import build_model
from mcmf_gtn.pipeline import build_input, prepare_inputs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return torch.device("cuda" if name == "cuda" or name == "auto" and torch.cuda.is_available() else "cpu")


def graph_fingerprint(raw):
    digest = hashlib.sha256()
    digest.update("\n".join(map(str, raw.node_ids)).encode("utf-8"))
    for value in (raw.features, raw.labels, raw.node_times, raw.edge_index, raw.edge_times, raw.edge_amounts):
        if value is not None:
            array = np.ascontiguousarray(value)
            digest.update(str(array.shape).encode("ascii"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def predict(model, graph, threshold=0.5):
    model.eval()
    with torch.no_grad():
        logits = model(graph.x, graph.edge_index, graph.edge_attr)
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite classification logits")
        scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    mask = graph.target_mask.cpu().numpy()
    labels = graph.y.cpu().numpy()[mask]
    probabilities = scores[mask]
    predictions = (probabilities >= threshold).astype(int)
    metrics = compute_metrics(labels, probabilities, predictions)
    records = [{"node_id": str(node), "label": int(label), "score": float(score), "prediction": int(prediction)} for node, label, score, prediction in zip(graph.node_ids[mask], labels, probabilities, predictions)]
    return metrics, records


def train_network(model, train, validation, cfg):
    device = resolve_device(cfg.device)
    model.to(device)
    train, validation = train.to(device), validation.to(device)
    if not train.target_mask.any() or not validation.target_mask.any():
        raise ValueError("training and validation require labeled center nodes")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_step, gamma=cfg.lr_gamma)
    targets = torch.nonzero(train.target_mask, as_tuple=False).flatten()
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
    best_score, best_state, best_epoch, best_validation = -1., None, None, None
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        start, weighted_loss = time.perf_counter(), 0.0
        order = torch.randperm(len(targets), generator=generator).to(device)
        for batch in order.split(cfg.batch_size):
            indices = targets[batch]
            optimizer.zero_grad()
            logits = model(train.x, train.edge_index, train.edge_attr)
            loss = F.cross_entropy(logits[indices], train.y[indices])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite classification loss")
            loss.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                raise ValueError("nonfinite model gradient")
            optimizer.step()
            weighted_loss += float(loss.detach().cpu()) * len(indices)
        validation_metrics, _ = predict(model, validation, cfg.decision_threshold)
        row = {"epoch": epoch, "train_loss": weighted_loss / len(targets), "validation_f1": validation_metrics["f1"], "validation_roc_auc": validation_metrics["roc_auc"], "lr": optimizer.param_groups[0]["lr"], "seconds": time.perf_counter() - start}
        history.append(row)
        if validation_metrics["f1"] > best_score:
            best_score, best_epoch, best_validation = validation_metrics["f1"], epoch, validation_metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        scheduler.step()
    model.load_state_dict(best_state)
    return {"history": history, "best_epoch": best_epoch, "validation_metrics": best_validation}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def write_predictions(path, records):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("node_id", "label", "score", "prediction"))
        writer.writeheader()
        writer.writerows(records)


def run_experiment(raw, cfg, out_dir):
    validate_config(cfg, formal=True)
    pipeline_start = time.perf_counter()
    out_dir = Path(out_dir)
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise ValueError(f"run already exists: {out_dir}; use a new output directory")
    set_seed(cfg.seed)
    prepared = prepare_data(raw, cfg)
    bundle = prepare_inputs(prepared, cfg)
    set_seed(cfg.seed)
    model = build_model(bundle.train.x.shape[1], bundle.train.edge_attr.shape[1], cfg)
    start = time.perf_counter()
    outcome = train_network(model, bundle.train, bundle.validation, cfg)
    device = resolve_device(cfg.device)
    metrics, predictions = predict(model, bundle.test.to(device), cfg.decision_threshold)
    classifier_seconds = time.perf_counter() - start
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": 1, "config": cfg.to_dict(),
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "input_dim": bundle.train.x.shape[1], "num_channels": bundle.train.edge_attr.shape[1],
        "transforms": bundle.transforms,
        "embedding_state": bundle.embedding_state.to_payload() if bundle.embedding_state is not None else None,
        "split_stats": prepared.stats, "eligible_test_ids": list(map(str, prepared.eligible_test_ids)),
        "data_fingerprint": graph_fingerprint(raw), "best_epoch": outcome["best_epoch"],
        "validation_metrics": outcome["validation_metrics"],
    }
    checkpoint_path = out_dir / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    learned = cfg.variant not in ("mlp", "gat", "topology_gcn", "direct_ef", "fixed_mean", "fixed_product")
    weights = model.channel_weights().detach().cpu().tolist() if learned else None
    metadata = {"schema_version": 1, "config": cfg.to_dict(), "status": "synthetic_smoke" if cfg.profile == "smoke" else "reference_run", "split_stats": prepared.stats, "data_fingerprint": checkpoint["data_fingerprint"], "channel_names": list(bundle.channel_names), "learned_channel_weights": weights, "implementation_settings": {name: {"value": getattr(cfg, name), "provenance": origin} for name, origin in IMPLEMENTATION_SETTINGS.items()}, "embedding_fit": bundle.embedding_state.metadata if bundle.embedding_state is not None else None, "checkpoint_selection": "highest_validation_positive_class_f1_first_tie", "decision_threshold": cfg.decision_threshold, "classifier_train_test_seconds": classifier_seconds, "prepared_pipeline_seconds": time.perf_counter() - pipeline_start}
    write_json(out_dir / "run.json", metadata)
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "history.json", outcome["history"])
    write_predictions(out_dir / "predictions.csv", predictions)
    with (out_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["actual/predicted", "licit", "illicit"])
        writer.writerow(["licit", metrics["tn"], metrics["fp"]])
        writer.writerow(["illicit", metrics["fn"], metrics["tp"]])
    return {**outcome, "metrics": metrics, "predictions": predictions, "checkpoint": str(checkpoint_path), "metadata": metadata}


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != 1:
        raise ValueError("unsupported checkpoint schema")
    cfg = load_config(None, **{**checkpoint["config"], "device": device})
    model = build_model(checkpoint["input_dim"], checkpoint["num_channels"], cfg)
    model.load_state_dict(checkpoint["model_state"])
    model.to(resolve_device(device)).eval()
    state = EmbeddingState.from_payload(checkpoint["embedding_state"]) if checkpoint["embedding_state"] is not None else None
    return {"payload": checkpoint, "config": cfg, "model": model, "embedding_state": state}


def checkpoint_graph(raw, loaded, all_test_targets=False):
    payload, cfg = loaded["payload"], loaded["config"]
    if graph_fingerprint(raw) != payload["data_fingerprint"]:
        raise ValueError("dataset fingerprint does not match this checkpoint")
    prepared = prepare_data(raw, cfg)
    if prepared.stats["target_ids"] != payload["split_stats"]["target_ids"]:
        raise ValueError("target IDs do not match checkpoint splitting")
    split = prepare_evaluation_graph(raw, cfg, payload["eligible_test_ids"], payload["split_stats"]["bounds"]) if all_test_targets else prepared.test
    graph = build_input(split, cfg, payload["transforms"], loaded["embedding_state"])
    return graph.to(resolve_device(cfg.device))


def predict_with_checkpoint(raw, loaded, all_test_targets=False):
    graph = checkpoint_graph(raw, loaded, all_test_targets)
    metrics, predictions = predict(loaded["model"], graph, loaded["config"].decision_threshold)
    return {"metrics": metrics, "predictions": predictions, "scope": "eligible_test_pool" if all_test_targets else "paper_balanced_temporal_test"}
