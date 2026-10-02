import numpy as np


def _probabilities(weights):
    weights = np.asarray(weights, dtype=np.float64)
    total = weights.sum()
    return weights / total if total > 0 else np.full(len(weights), 1.0 / len(weights))


def transition_probabilities(times, amounts, previous_time, cfg, uniform=False):
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
        raise ValueError("candidate timestamps must be a finite nonempty vector")
    if uniform or previous_time is None:
        return np.full(len(times), 1.0 / len(times))
    time_weights = np.maximum(1.0 - (times - previous_time), 0.0)
    temporal = _probabilities(time_weights)
    if cfg.dataset != "etx":
        return temporal
    if amounts is None:
        raise ValueError("ETX BTRW requires genuine event amounts")
    amounts = np.asarray(amounts, dtype=np.float64)
    if amounts.shape != times.shape or not np.isfinite(amounts).all() or (amounts < 0).any():
        raise ValueError("candidate amounts must be finite and nonnegative")
    return cfg.alpha * temporal + (1.0 - cfg.alpha) * _probabilities(amounts)


def generate_walks(split, cfg, uniform=False, time_scale=None, all_nodes=False):
    split.validate()
    scale = time_scale
    if scale is None:
        scale = np.ptp(split.edge_times) if len(split.edge_times) else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    successors = [[] for _ in split.node_ids]
    for e, (source, target) in enumerate(split.edge_index.T):
        successors[source].append((int(target), float(split.edge_times[e]), e))
    rng = np.random.default_rng(cfg.seed)
    starts = np.arange(len(split.node_ids)) if all_nodes else np.flatnonzero(split.target_mask)
    result = []
    for _ in range(cfg.num_walks):
        for start in rng.permutation(starts):
            sequence = [str(split.node_ids[start])]
            current, previous_time = int(start), None
            for _ in range(cfg.walk_length - 1):
                candidates = [entry for entry in successors[current] if previous_time is None or entry[1] > previous_time]
                if not candidates:
                    break
                times = np.array([entry[1] for entry in candidates]) / scale
                amounts = None if split.edge_amounts is None else split.edge_amounts[[entry[2] for entry in candidates]]
                probabilities = transition_probabilities(times, amounts, None if previous_time is None else previous_time / scale, cfg, uniform)
                current, previous_time, _ = candidates[int(rng.choice(len(candidates), p=probabilities))]
                sequence.append(str(split.node_ids[current]))
            result.append(sequence)
    return result
