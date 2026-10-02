"""Balanced centers, causal split-local context, and label-safe target masks."""

import numpy as np

from mcmf_gtn.config import dataset_spec
from mcmf_gtn.data.types import GraphData, PreparedData, SplitGraph


def _target_indices(raw, target_ids):
    ids = np.asarray(target_ids, dtype=str).reshape(-1)
    if not len(ids):
        raise ValueError("empty target split; supply at least one labeled center")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate target node IDs")
    lookup = {str(node_id): i for i, node_id in enumerate(raw.node_ids)}
    try:
        indices = np.array([lookup[node_id] for node_id in ids], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"unknown target node ID: {exc.args[0]}") from exc
    if (raw.labels[indices] < 0).any():
        raise ValueError("unknown labels cannot be supervised targets")
    return indices


def sample_targets(raw, seed):
    """Select every illicit center and an equal-size licit sample without replacement."""
    raw.validate()
    positives = np.flatnonzero(raw.labels == 1)
    negatives = np.flatnonzero(raw.labels == 0)
    if not len(positives):
        raise ValueError("no illicit prediction targets")
    if len(negatives) < len(positives):
        raise ValueError("insufficient licit nodes for sampling without replacement")
    chosen = np.random.default_rng(seed).choice(negatives, len(positives), replace=False)
    return raw.node_ids[np.sort(np.concatenate([positives, chosen]))].astype(str)


def split_targets(raw, target_ids, dataset, seed):
    """Return center indices, inclusive event bounds, and all-account memberships."""
    dataset_spec(dataset)
    indices = _target_indices(raw, target_ids)
    if dataset == "etx":
        order = np.lexsort((raw.node_ids.astype(str), raw.node_times))
        n_train, n_validation_end = int(0.7 * len(order)), int(0.8 * len(order))
        membership = {"train": order[:n_train], "validation": order[n_train:n_validation_end], "test": order[n_validation_end:]}
        if any(not len(part) for part in membership.values()):
            raise ValueError("ETX needs enough accounts for nonempty 70/10/20 chronological splits")
        bounds = {key: (float(raw.node_times[part].min()), float(raw.node_times[part].max())) for key, part in membership.items()}
        bounds["test"] = (bounds["test"][0], float(raw.edge_times.max()))
    else:
        membership = {
            "train": np.flatnonzero(raw.node_times <= 34),
            "validation": np.flatnonzero((raw.node_times >= 35) & (raw.node_times <= 44)),
            "test": np.flatnonzero((raw.node_times >= 45) & (raw.node_times <= 49)),
        }
        minimum = float(raw.node_times.min())
        bounds = {"train": (minimum, 34), "validation": (minimum, 44), "test": (45, 49)}
    selected = {key: part[np.isin(part, indices)] for key, part in membership.items()}
    if dataset != "etx":
        train = selected["train"]
        count = int(0.1 * len(train))
        heldout = np.random.default_rng(seed).permutation(train)[:count]
        selected["train"] = train[~np.isin(train, heldout)]
        selected["validation"] = np.sort(np.concatenate([selected["validation"], heldout]))
    for key, part in selected.items():
        if not len(part):
            raise ValueError(f"empty {key} target split after chronological assignment")
    return {**selected, "bounds": bounds, "chronological_membership": membership}


def _etx_features(edge_index, edge_times, amounts, n):
    if amounts is None:
        raise ValueError("ETX split-local features require transaction amounts")
    x = np.zeros((n, 14), dtype=np.float64)
    for side, offset in ((1, 0), (0, 1)):
        endpoint = edge_index[side]
        count = np.bincount(endpoint, minlength=n)
        total = np.bincount(endpoint, weights=amounts, minlength=n)
        maximum = np.zeros(n)
        minimum = np.full(n, np.inf)
        np.maximum.at(maximum, endpoint, amounts)
        np.minimum.at(minimum, endpoint, amounts)
        # Degree counts counterparties; counts retain repeated transaction events.
        pairs = np.unique(edge_index.T, axis=0)
        degree = np.bincount(pairs[:, side], minlength=n) if len(pairs) else np.zeros(n)
        x[:, offset] = degree
        x[:, 2 + offset] = total
        x[:, 4 + offset] = count
        x[:, 6 + offset] = maximum
        x[:, 8 + offset] = np.where(count, minimum, 0)
        x[:, 10 + offset] = np.divide(total, count, out=np.zeros(n), where=count > 0)
    first, last = np.full(n, np.inf), np.full(n, -np.inf)
    for endpoint in edge_index:
        np.minimum.at(first, endpoint, edge_times)
        np.maximum.at(last, endpoint, edge_times)
    active = np.isfinite(first)
    x[active, 12] = last[active]
    x[active, 13] = last[active] - first[active]
    return x


def extract_subgraph(raw, target_ids, lower, upper):
    """Filter events, collect bidirectional one-hop context, then induce local edges."""
    raw.validate()
    centers = _target_indices(raw, target_ids)
    if not np.isfinite([lower, upper]).all() or lower > upper:
        raise ValueError("invalid finite temporal window bounds")
    if (raw.node_times[centers] > upper).any():
        raise ValueError("target node is first observed after the temporal window upper bound")
    window = (raw.edge_times >= lower) & (raw.edge_times <= upper)
    if raw.features.shape[1] != 14:
        window &= (raw.node_times[raw.edge_index[0]] <= upper) & (raw.node_times[raw.edge_index[1]] <= upper)
    edge_index = raw.edge_index[:, window]
    touching = np.isin(edge_index[0], centers) | np.isin(edge_index[1], centers)
    nodes = np.unique(np.concatenate([centers, edge_index[:, touching].reshape(-1)]))
    induced = np.isin(edge_index[0], nodes) & np.isin(edge_index[1], nodes)
    selected_edges = edge_index[:, induced]
    lookup = np.full(len(raw.node_ids), -1, dtype=np.int64)
    lookup[nodes] = np.arange(len(nodes))
    local_edges = lookup[selected_edges]
    event_times = raw.edge_times[window][induced].copy()
    amounts = None if raw.edge_amounts is None else raw.edge_amounts[window][induced].copy()
    features = raw.features[nodes].copy()
    if features.shape[1] == 14:
        features = _etx_features(local_edges, event_times, amounts, len(nodes))
    result = SplitGraph(raw.node_ids[nodes].copy(), features, raw.labels[nodes].copy(), raw.node_times[nodes].copy(), local_edges, event_times, amounts, np.isin(nodes, centers))
    result.validate()
    return result


def _preprocess(raw, cfg):
    raw.validate()
    if raw.features.shape[1] != dataset_spec(cfg.dataset).static_dim:
        raise ValueError(f"{cfg.dataset} requires {dataset_spec(cfg.dataset).static_dim} static features")
    keep_edges = np.ones(raw.edge_index.shape[1], dtype=bool)
    if cfg.dataset == "etx":
        if raw.edge_amounts is None:
            raise ValueError("ETX requires edge transaction amounts")
        keep_edges &= raw.edge_amounts > 0
    keep_nodes = np.ones(len(raw.node_ids), dtype=bool)
    while True:
        edges = raw.edge_index[:, keep_edges]
        degree = np.bincount(edges.reshape(-1), minlength=len(raw.node_ids))
        next_nodes = keep_nodes & (degree >= cfg.min_degree)
        next_edges = keep_edges & next_nodes[raw.edge_index[0]] & next_nodes[raw.edge_index[1]]
        if np.array_equal(next_nodes, keep_nodes) and np.array_equal(next_edges, keep_edges):
            break
        keep_nodes, keep_edges = next_nodes, next_edges
    nodes = np.flatnonzero(keep_nodes)
    if not len(nodes):
        raise ValueError("preprocessing removed every node; check minimum degree and valid transactions")
    lookup = np.full(len(raw.node_ids), -1, dtype=np.int64)
    lookup[nodes] = np.arange(len(nodes))
    edge_index = lookup[raw.edge_index[:, keep_edges]]
    edge_times = raw.edge_times[keep_edges].copy()
    times = raw.node_times[nodes].copy()
    if cfg.dataset == "etx":
        times[:] = np.inf
        for endpoint in edge_index:
            np.minimum.at(times, endpoint, edge_times)
    result = GraphData(raw.node_ids[nodes].copy(), raw.features[nodes].copy(), raw.labels[nodes].copy(), times, edge_index, edge_times, None if raw.edge_amounts is None else raw.edge_amounts[keep_edges].copy())
    result.validate()
    return result


def prepare_data(raw, cfg):
    clean = _preprocess(raw, cfg)
    targets = sample_targets(clean, cfg.seed)
    splits = split_targets(clean, targets, cfg.dataset, cfg.seed)
    graphs = {key: extract_subgraph(clean, clean.node_ids[splits[key]], *splits["bounds"][key]) for key in ("train", "validation", "test")}
    excluded = np.concatenate([clean.node_ids[splits[key]] for key in ("train", "validation")])
    pool = splits["chronological_membership"]["test"]
    eligible = clean.node_ids[pool[(clean.labels[pool] >= 0) & ~np.isin(clean.node_ids[pool], excluded)]].copy()
    counts = {}
    for key, part in graphs.items():
        labels = part.labels[part.target_mask]
        counts[key] = {"nodes": len(part.node_ids), "edges": part.edge_index.shape[1], "targets": len(labels), "licit": int((labels == 0).sum()), "illicit": int((labels == 1).sum()), "unknown_context": int((part.labels == -1).sum())}
    stats = {
        "bounds": {key: [float(x) for x in value] for key, value in splits["bounds"].items()},
        "counts": counts,
        "target_ids": {key: clean.node_ids[splits[key]].tolist() for key in graphs},
        "raw_nodes": len(raw.node_ids), "preprocessed_nodes": len(clean.node_ids),
        "raw_edges": raw.edge_index.shape[1], "preprocessed_edges": clean.edge_index.shape[1],
        "eligible_test_count": len(eligible),
        "chronology_policy": "ETX account memberships precede balanced target selection; stable first-event time and ID ordering; test events through last observed event" if cfg.dataset == "etx" else "train <=34; validation 35..44 plus 10% heldout train centers with events through44; test45..49; nodes first seen after window upper bound excluded",
    }
    return PreparedData(graphs["train"], graphs["validation"], graphs["test"], stats, eligible)


def prepare_evaluation_graph(raw, cfg, eligible_test_ids, bounds):
    """Build a new frozen-model test graph from the eligible imbalance pool."""
    clean = _preprocess(raw, cfg)
    window = bounds["test"] if isinstance(bounds, dict) else bounds
    return extract_subgraph(clean, eligible_test_ids, *window)
