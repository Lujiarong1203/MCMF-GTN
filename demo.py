import numpy as np

from mcmf_gtn.config import dataset_spec
from mcmf_gtn.data.types import GraphData


def create_demo_graph(dataset="etx", seed=42):
    spec = dataset_spec(dataset)
    rng = np.random.default_rng(seed)
    ids, labels, times = ["anchor_a", "anchor_b", "anchor_c"], [-1] * 3, [1.] * 3
    sources, targets, event_times, amounts = [], [], [], []
    for time in range(1, 50):
        for category in range(5):
            center = len(ids)
            ids.append(f"node_{time}_{category}")
            labels.append(1 if category == 0 else 0)
            times.append(float(time))
            anchor = time % 3
            sources.append(center)
            targets.append(anchor)
            event_times.append(float(time))
            amounts.append(float(2 + category))
        for anchor in range(3):
            sources.append(anchor)
            targets.append((anchor + 1) % 3)
            event_times.append(float(time) + (0.01 if dataset == "etx" else 0))
            amounts.append(float(anchor + 1))
    features = rng.normal(size=(len(ids), spec.static_dim)).astype(np.float64)
    if dataset != "etx":
        features[:, 0] = times
    node_times = np.array(times)
    edge_index, edge_times = np.array([sources, targets], dtype=np.int64), np.array(event_times)
    if dataset == "etx":
        node_times[:] = np.inf
        for endpoint in edge_index:
            np.minimum.at(node_times, endpoint, edge_times)
    return GraphData(np.array(ids), features, np.array(labels, dtype=int), node_times, edge_index, edge_times, np.array(amounts) if dataset == "etx" else None)
