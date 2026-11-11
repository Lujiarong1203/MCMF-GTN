import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def build_successor_table_from_edgelist(
    subgraph_node_ids: np.ndarray,
    edgelist: np.ndarray,
    timestamps: np.ndarray,
    amounts: np.ndarray,
) -> Dict:
    node_set = set(subgraph_node_ids)
    succ = defaultdict(list)
    for i in range(len(edgelist)):
        src, dst = edgelist[i]
        if src in node_set and dst in node_set:
            succ[src].append((dst, timestamps[i], amounts[i]))
    return dict(succ)


def build_successor_table_from_nx(G: Any, subgraph_node_ids: np.ndarray) -> Dict:
    node_set = set(subgraph_node_ids)
    succ = defaultdict(list)
    for u, v, data in G.edges(data=True):
        if u in node_set and v in node_set:
            succ[u].append((v, data.get("timestamp", 0), data.get("amount", 0)))
    return dict(succ)


class BiasedTemporalRandomWalk:
    """Temporal walk with time-proximity and amount biases.

    Transition: p = alpha * p_t + (1 - alpha) * p_a, where p_t favors edges
    close to the previous timestamp and p_a favors larger amounts.
    With use_amount=False the walk reduces to the time-proximity bias only.
    """

    def __init__(self, successor_table, walk_length=80, num_walks=10,
                 alpha=0.6, t_sup=1.0, seed=42, use_amount=True):
        self.succ = successor_table
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.alpha = alpha
        self.t_sup = t_sup
        self.use_amount = use_amount
        self.rng = np.random.default_rng(seed)

        all_ts = [ts for neighs in successor_table.values() for _, ts, _ in neighs]
        self._ts_min = min(all_ts) if all_ts else 0
        self._ts_max = max(all_ts) if all_ts else 1

    def _normalise(self, ts):
        if self._ts_max == self._ts_min:
            return 0.0
        return (ts - self._ts_min) / (self._ts_max - self._ts_min)

    def _transition_probs(self, candidates, prev_ts_norm):
        n = len(candidates)
        if n == 0:
            return np.array([]), np.array([])

        ts_norm = np.array([self._normalise(c[1]) for c in candidates], dtype=float)
        delta_t = ts_norm - prev_ts_norm
        p_t_num = np.maximum(self.t_sup - delta_t, 1e-12)
        p_t = p_t_num / p_t_num.sum()

        amt_arr = np.array([c[2] for c in candidates], dtype=float)
        p_a = np.maximum(amt_arr, 1e-12)
        p_a = p_a / p_a.sum()
        return p_t, p_a

    def _walk(self, start_node):
        walk = [start_node]
        succ_start = self.succ.get(start_node, [])
        if not succ_start:
            return walk

        idx = self.rng.integers(len(succ_start))
        curr, ts, _ = succ_start[idx]
        walk.append(curr)
        prev_norm = self._normalise(ts)

        for _ in range(1, self.walk_length):
            candidates = self.succ.get(curr, [])
            if not candidates:
                break
            filtered = [c for c in candidates if self._normalise(c[1]) > prev_norm]
            if not filtered:
                break

            p_t, p_a = self._transition_probs(filtered, prev_norm)
            p = p_t if not self.use_amount else self.alpha * p_t + (1.0 - self.alpha) * p_a
            p = p / p.sum()

            chosen = self.rng.choice(len(filtered), p=p)
            nxt, nxt_ts, _ = filtered[chosen]
            walk.append(nxt)
            prev_norm = self._normalise(nxt_ts)
            curr = nxt

        return walk

    def generate_walks(self, nodes, verbose=True):
        walks = []
        it = tqdm(nodes, desc="BTRW") if verbose else nodes
        for node in it:
            for _ in range(self.num_walks):
                w = self._walk(node)
                if len(w) >= 2:
                    walks.append([str(n) for n in w])
        return walks


class _SkipGramDataset(Dataset):
    def __init__(self, walks, vocab, window):
        self.pairs = []
        for walk in walks:
            tokens = [vocab[w] for w in walk]
            for i, center in enumerate(tokens):
                start = max(0, i - window)
                end = min(len(tokens), i + window + 1)
                for j in range(start, end):
                    if j != i:
                        self.pairs.append((center, tokens[j]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        return torch.tensor(center), torch.tensor(context)


class _SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.center_emb = nn.Embedding(vocab_size, embed_dim)
        self.context_emb = nn.Embedding(vocab_size, embed_dim)
        nn.init.xavier_uniform_(self.center_emb.weight)
        nn.init.xavier_uniform_(self.context_emb.weight)

    def forward(self, center, context, neg_samples):
        v_c = self.center_emb(center)
        v_o = self.context_emb(context)
        v_n = self.context_emb(neg_samples)

        pos_score = torch.sum(v_c * v_o, dim=1)
        pos_loss = -F.logsigmoid(pos_score)

        neg_score = torch.bmm(v_n, v_c.unsqueeze(-1)).squeeze(-1)
        neg_loss = -F.logsigmoid(-neg_score).sum(dim=1)

        return (pos_loss + neg_loss).mean()


def train_skipgram(walks, embed_dim=128, window=10, neg_count=5,
                   batch_size=1024, epochs=5, lr=0.01, device="cpu",
                   verbose=True):
    word_counts = Counter()
    for walk in walks:
        word_counts.update(walk)
    vocab = {w: i for i, (w, _) in enumerate(word_counts.most_common())}
    vocab_size = len(vocab)

    ds = _SkipGramDataset(walks, vocab, window)
    if len(ds) == 0:
        return np.random.randn(vocab_size, embed_dim).astype(np.float32), vocab

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    word_freq = np.array([word_counts[w] for w, _ in vocab.items()], dtype=float)
    word_freq = word_freq ** 0.75
    neg_dist = word_freq / word_freq.sum()

    model = _SkipGramModel(vocab_size, embed_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    it = tqdm(range(epochs), desc="Skip-Gram") if verbose else range(epochs)
    for epoch in it:
        total_loss, n_batches = 0.0, 0
        for center, context in loader:
            B = center.size(0)
            center = center.to(device)
            context = context.to(device)

            neg_ids = np.random.choice(vocab_size, size=(B, neg_count), p=neg_dist)
            neg_t = torch.tensor(neg_ids, dtype=torch.long, device=device)

            loss = model(center, context, neg_t)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            n_batches += 1

        if verbose:
            tqdm.write(f"  Epoch {epoch + 1}/{epochs}  loss={total_loss / max(n_batches, 1):.4f}")

    embeddings = model.center_emb.weight.detach().cpu().numpy()
    return embeddings, vocab
