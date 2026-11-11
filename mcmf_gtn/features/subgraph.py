import numpy as np
import pandas as pd
import networkx as nx
from typing import Tuple, Set


def balanced_sample_target_nodes(labels: np.ndarray,
                                 rng: np.random.Generator) -> np.ndarray:
    """All anomalous nodes plus an equal number of sampled normal nodes."""
    anomalous_idx = np.where(labels == 1)[0]
    normal_idx = np.where(labels == 0)[0]

    if len(anomalous_idx) == 0:
        raise ValueError("No anomalous nodes found")

    sampled_normal = rng.choice(normal_idx, size=len(anomalous_idx), replace=False)

    target_mask = np.zeros(len(labels), dtype=bool)
    target_mask[anomalous_idx] = True
    target_mask[sampled_normal] = True
    return target_mask


def construct_subgraph_from_edgelist(
    all_node_ids: np.ndarray,
    labels: np.ndarray,
    edgelist: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Balanced target sampling + 1-hop neighbours over the complete graph.

    The subgraph spans the full temporal range; train/val/test masks are
    applied afterwards at the supervision stage.
    """
    target_mask = balanced_sample_target_nodes(labels, rng)
    target_ids = set(all_node_ids[target_mask])

    edge_df = pd.DataFrame(edgelist, columns=["src", "dst"])
    src_in = edge_df["src"].isin(target_ids)
    dst_in = edge_df["dst"].isin(target_ids)

    predecessors = edge_df.loc[dst_in, "src"].unique()
    successors = edge_df.loc[src_in, "dst"].unique()
    neighbour_ids = set(np.unique(np.concatenate([predecessors, successors])))

    v_sub_ids = target_ids | neighbour_ids
    node_mask = np.array([nid in v_sub_ids for nid in all_node_ids])

    src_in_vsub = edge_df["src"].isin(v_sub_ids)
    dst_in_vsub = edge_df["dst"].isin(v_sub_ids)
    edge_mask = src_in_vsub.values & dst_in_vsub.values

    subgraph_node_ids = all_node_ids[node_mask]
    return node_mask, edge_mask, subgraph_node_ids


def construct_subgraph_from_nx(
    G: nx.MultiDiGraph,
    labels: np.ndarray,
    all_node_ids: np.ndarray,
    edgelist: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_mask = balanced_sample_target_nodes(labels, rng)
    target_ids = set(all_node_ids[target_mask])

    neighbour_ids: Set = set()
    for node in target_ids:
        neighbour_ids.update(G.predecessors(node))
        neighbour_ids.update(G.successors(node))

    v_sub_ids = target_ids | neighbour_ids
    node_mask = np.array([nid in v_sub_ids for nid in all_node_ids])

    edge_df = pd.DataFrame(edgelist, columns=["src", "dst"])
    src_in_vsub = edge_df["src"].isin(v_sub_ids)
    dst_in_vsub = edge_df["dst"].isin(v_sub_ids)
    edge_mask = src_in_vsub.values & dst_in_vsub.values

    subgraph_node_ids = all_node_ids[node_mask]
    return node_mask, edge_mask, subgraph_node_ids
