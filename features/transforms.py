"""Serializable imputation, scaling, and channel weights fitted to training only."""

import numpy as np

from mcmf_gtn.config import dataset_spec


def fit_transforms(train_split, cfg):
    train_split.validate()
    expected = dataset_spec(cfg.dataset).static_dim
    if train_split.features.shape[1] != expected:
        raise ValueError(f"{cfg.dataset} requires {expected} static features")
    x = np.asarray(train_split.features, dtype=np.float64)
    if (np.isnan(x).all(axis=0)).any():
        raise ValueError("training feature column is entirely missing; cannot fit imputation")
    medians = np.nanmedian(x, axis=0)
    filled = np.where(np.isnan(x), medians, x)
    mean, std = filled.mean(axis=0), filled.std(axis=0)
    std[std == 0] = 1
    if not len(train_split.edge_times):
        raise ValueError("training graph has no events to fit temporal channel statistics")
    minimum, maximum = float(train_split.edge_times.min()), float(train_split.edge_times.max())
    amount_max = None
    if cfg.dataset == "etx":
        if train_split.edge_amounts is None or not len(train_split.edge_amounts) or train_split.edge_amounts.max() <= 0:
            raise ValueError("training ETX graph needs positive transaction amounts")
        amount_max = float(train_split.edge_amounts.max())
    return {
        "dataset": cfg.dataset, "feature_scaling": cfg.feature_scaling,
        "timestamp_min": minimum, "timestamp_max": maximum, "time_scale": maximum - minimum if maximum > minimum else 1.0,
        "amount_max": amount_max,
        "feature_medians": medians.tolist(), "feature_mean": mean.tolist(), "feature_std": std.tolist(),
    }


def transform_split(split, state, dataset):
    split.validate()
    spec = dataset_spec(dataset)
    if state["dataset"] != dataset or split.features.shape[1] != spec.static_dim:
        raise ValueError("transform dataset or static feature dimensions do not match fitted training state")
    x = np.asarray(split.features, dtype=np.float64)
    medians = np.asarray(state["feature_medians"], dtype=np.float64)
    x = np.where(np.isnan(x), medians, x)
    if state["feature_scaling"] == "standard":
        x = (x - np.asarray(state["feature_mean"])) / np.asarray(state["feature_std"])
    time_weights = (split.edge_times - state["timestamp_min"]) / state["time_scale"]
    if (time_weights < 0).any():
        raise ValueError("edge timestamp earlier than training minimum produces a negative temporal channel; choose a compatible causal window")
    channels = [np.ones(len(split.edge_times)), time_weights]
    if dataset == "etx":
        if split.edge_amounts is None or state["amount_max"] is None or state["amount_max"] <= 0:
            raise ValueError("ETX amount channel requires fitted positive training amount statistics")
        channels.append(split.edge_amounts / state["amount_max"])
    attributes = np.column_stack(channels)
    x, attributes = x.astype(np.float32), attributes.astype(np.float32)
    if not np.isfinite(x).all() or not np.isfinite(attributes).all() or (attributes < 0).any():
        raise ValueError("transformed features and channel weights must be finite and channel weights nonnegative")
    return x, attributes
