import json
import math
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    static_dim: int
    channels: tuple
    split_type: str


SPECS = {
    "elliptic": DatasetSpec("elliptic", 166, ("topology", "timestamp"), "snapshot"),
    "etx": DatasetSpec("etx", 14, ("topology", "timestamp", "amount"), "chronological"),
    "rd": DatasetSpec("rd", 185, ("topology", "timestamp"), "snapshot"),
}

VARIANTS = (
    "full", "mlp", "gat", "topology_gcn", "fixed_mean", "direct_ef", "edge_weight",
    "topology_gtn", "no_amount", "no_timestamp", "no_embedding", "embedding_only",
    "uniform_trw", "fixed_product",
)

IMPLEMENTATION_SETTINGS = {
    "top_k": "legacy public implementation; numeric K is not reported in the manuscript",
    "walk_length": "legacy public implementation",
    "num_walks": "legacy public implementation",
    "window": "legacy public implementation",
    "negative_samples": "legacy public implementation",
    "skipgram_epochs": "legacy public implementation",
    "skipgram_lr": "reference implementation choice",
    "inference_epochs": "reference implementation choice",
    "inference_policy": "unknown-node handling is not specified in the manuscript",
    "weight_decay": "legacy public implementation; not specified in the manuscript",
    "feature_scaling": "reference implementation choice",
    "min_degree": "low-activity cutoff is not numerically specified in the manuscript",
    "decision_threshold": "reference implementation choice; fixed before test evaluation",
    "gat_hidden": "conventional baseline implementation choice",
    "gat_heads": "conventional baseline implementation choice",
    "gat_dropout": "conventional baseline implementation choice",
    "edge_fusion_hidden": "direct edge-feature baseline implementation choice",
}


@dataclass(frozen=True)
class RunConfig:
    dataset: str = "elliptic"
    seed: int = 42
    seeds: tuple = (42, 123, 456, 789, 1024)
    epochs: int = 100
    batch_size: int = 128
    lr: float = 0.0001
    lr_step: int = 10
    lr_gamma: float = 0.1
    gt_layers: int = 3
    gcn_hidden: tuple = (128, 64, 32)
    dropout: float = 0.5
    embed_dim: int = 128
    alpha: float = 0.6
    top_k: int = 10
    walk_length: int = 80
    num_walks: int = 10
    window: int = 10
    negative_samples: int = 5
    skipgram_epochs: int = 5
    skipgram_lr: float = 0.025
    inference_epochs: int = 5
    inference_policy: str = "strict"
    weight_decay: float = 1e-5
    feature_scaling: str = "standard"
    min_degree: int = 1
    decision_threshold: float = 0.5
    gat_hidden: tuple = (64, 32)
    gat_heads: int = 4
    gat_dropout: float = 0.3
    edge_fusion_hidden: int = 16
    device: str = "auto"
    variant: str = "full"
    profile: str = "paper"
    acknowledged_defaults: bool = False

    def to_dict(self):
        return asdict(self)

    def with_updates(self, **updates):
        result = replace(self, **updates)
        validate_config(result)
        return result


def dataset_spec(name):
    if name not in SPECS:
        raise ValueError(f"unknown dataset: {name}; choose elliptic, etx or rd")
    return SPECS[name]


def validate_config(cfg, formal=False):
    dataset_spec(cfg.dataset)
    if cfg.variant not in VARIANTS:
        raise ValueError(f"unknown variant: {cfg.variant}")
    integer_fields = ("epochs", "batch_size", "lr_step", "gt_layers", "embed_dim", "top_k", "walk_length", "num_walks", "window", "negative_samples", "skipgram_epochs", "inference_epochs", "gat_heads", "edge_fusion_hidden")
    for name in integer_fields:
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if cfg.walk_length < 2:
        raise ValueError("walk_length must be at least 2")
    for name in ("lr", "lr_gamma", "skipgram_lr"):
        value = getattr(cfg, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("alpha", "dropout", "gat_dropout", "decision_threshold"):
        value = getattr(cfg, name)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if cfg.dropout == 1 or cfg.gat_dropout == 1:
        raise ValueError("dropout must be below 1")
    if not math.isfinite(cfg.weight_decay) or cfg.weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if isinstance(cfg.min_degree, bool) or not isinstance(cfg.min_degree, int) or cfg.min_degree < 1:
        raise ValueError("min_degree must be a positive integer")
    for sizes in (cfg.gcn_hidden, cfg.gat_hidden):
        if not sizes or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in sizes):
            raise ValueError("hidden dimensions must be positive integers")
    if cfg.inference_policy not in ("strict", "frozen_context"):
        raise ValueError("inference_policy must be strict or frozen_context")
    if cfg.feature_scaling not in ("none", "standard"):
        raise ValueError("feature_scaling must be none or standard")
    if cfg.device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    if cfg.profile not in ("paper", "smoke", "custom"):
        raise ValueError("profile must be paper, smoke or custom")
    if not isinstance(cfg.seed, int) or isinstance(cfg.seed, bool) or cfg.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not cfg.seeds or len(set(cfg.seeds)) != len(cfg.seeds) or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in cfg.seeds):
        raise ValueError("seeds must be unique nonnegative integers")
    if formal and cfg.profile != "smoke" and not cfg.acknowledged_defaults:
        raise ValueError("confirm implementation settings with --accept-implementation-defaults or acknowledged_defaults=true in an author configuration")


def load_config(path=None, **overrides):
    values = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    if not isinstance(values, dict):
        raise ValueError("configuration must be a JSON object")
    values.update({key: value for key, value in overrides.items() if value is not None})
    allowed = {x.name for x in fields(RunConfig)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
    for name in ("seeds", "gcn_hidden", "gat_hidden"):
        if name in values:
            values[name] = tuple(values[name])
    result = RunConfig(**values)
    validate_config(result)
    return result
