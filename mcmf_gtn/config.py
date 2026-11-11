import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    data_dir: str = os.path.join(os.path.dirname(__file__), "..", "data")
    output_dir: str = "outputs"

    seed: int = 42
    lr: float = 1e-4
    batch_size: int = 128
    epochs: int = 100
    weight_decay: float = 1e-5

    gt_layers: int = 3
    gcn_hidden: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout: float = 0.5

    embed_dim: int = 128
    btrw_alpha: float = 0.6
    walk_length: int = 80
    num_walks: int = 10
    window_size: int = 10

    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])

    gt_top_k: int = 10

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
