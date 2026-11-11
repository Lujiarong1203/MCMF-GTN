import os
import numpy as np
import torch

from .btrw import (
    build_successor_table_from_edgelist,
    BiasedTemporalRandomWalk,
    train_skipgram,
)


def generate_btrw_embeddings(
    subgraph_node_ids,
    edgelist,
    timestamps,
    amounts,
    embed_dim=128,
    walk_length=80,
    num_walks=10,
    alpha=0.6,
    window=10,
    neg_count=5,
    epochs=5,
    seed=42,
    cache_dir="outputs",
    device="cpu",
    use_amount=True,
    verbose=True,
):
    """Biased temporal random walks over the whole subgraph + Skip-Gram.

    Walks start from every subgraph node, so the Skip-Gram corpus spans
    training, validation, and test nodes alike.
    """
    os.makedirs(cache_dir, exist_ok=True)

    cache_key = (f"btrw_emb_n{len(subgraph_node_ids)}_d{embed_dim}_"
                 f"wl{walk_length}_nw{num_walks}_a{alpha}_s{seed}.npy")
    cache_path = os.path.join(cache_dir, cache_key)
    if os.path.exists(cache_path):
        if verbose:
            print(f"  Loading cached BTRW embeddings from {cache_path}", flush=True)
        return np.load(cache_path)

    succ_table = build_successor_table_from_edgelist(
        subgraph_node_ids, edgelist, timestamps, amounts
    )

    walker = BiasedTemporalRandomWalk(
        succ_table,
        walk_length=walk_length,
        num_walks=num_walks,
        alpha=alpha,
        seed=seed,
        use_amount=use_amount,
    )
    walks = walker.generate_walks(list(subgraph_node_ids), verbose=verbose)

    if verbose:
        print(f"  Generated {len(walks)} walks, training Skip-Gram...", flush=True)

    embeddings_raw, vocab = train_skipgram(
        walks,
        embed_dim=embed_dim,
        window=window,
        neg_count=neg_count,
        epochs=epochs,
        device=device,
        verbose=verbose,
    )

    n = len(subgraph_node_ids)
    embeddings = np.zeros((n, embed_dim), dtype=np.float32)
    for i, nid in enumerate(subgraph_node_ids):
        node_str = str(nid)
        if node_str in vocab:
            embeddings[i] = embeddings_raw[vocab[node_str]]

    np.save(cache_path, embeddings)
    if verbose:
        print(f"  Saved BTRW embeddings to {cache_path}", flush=True)
    return embeddings


def add_btrw_to_features(raw_features, btrw_embeddings, mode="concat"):
    if mode == "replace":
        return btrw_embeddings.astype(np.float32)
    if mode == "concat":
        return np.concatenate([raw_features, btrw_embeddings], axis=1).astype(np.float32)
    if mode == "add":
        assert raw_features.shape[1] == btrw_embeddings.shape[1]
        return (raw_features + btrw_embeddings).astype(np.float32)
    raise ValueError(f"Unknown mode: {mode}")
