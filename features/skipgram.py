from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class EmbeddingState:
    vocabulary: dict
    input_vectors: np.ndarray
    context_vectors: np.ndarray
    frequencies: np.ndarray
    metadata: dict

    def to_payload(self):
        return {
            "vocabulary": self.vocabulary,
            "input_vectors": torch.from_numpy(self.input_vectors.copy()),
            "context_vectors": torch.from_numpy(self.context_vectors.copy()),
            "frequencies": torch.from_numpy(self.frequencies.copy()),
            "metadata": self.metadata,
        }

    @classmethod
    def from_payload(cls, payload):
        return cls(payload["vocabulary"], payload["input_vectors"].cpu().numpy(), payload["context_vectors"].cpu().numpy(), payload["frequencies"].cpu().numpy(), payload["metadata"])


def _device(cfg):
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return torch.device("cuda" if cfg.device == "cuda" or cfg.device == "auto" and torch.cuda.is_available() else "cpu")


def _pair_batches(walks, vocabulary, window, order, batch_size=1024):
    centers, contexts = [], []
    for position in order:
        tokens = [vocabulary[token] for token in walks[position]]
        for i, center in enumerate(tokens):
            for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
                if i != j:
                    centers.append(center)
                    contexts.append(tokens[j])
                    if len(centers) == batch_size:
                        yield centers, contexts
                        centers, contexts = [], []
    if centers:
        yield centers, contexts


def _loss(center, context, negatives):
    positive = F.logsigmoid((center * context).sum(-1))
    negative = F.logsigmoid(-torch.einsum("bd,bkd->bk", center, negatives)).sum(-1)
    return -(positive + negative).mean()


def fit_embeddings(walks, cfg):
    walks = [[str(token) for token in sequence] for sequence in walks if len(sequence) > 1]
    if not walks:
        raise ValueError("training walks contain no Skip-Gram context pairs")
    counts = Counter(token for sequence in walks for token in sequence)
    vocabulary = {token: i for i, token in enumerate(sorted(counts))}
    frequencies = np.array([counts[token] for token in vocabulary], dtype=np.float64)
    distribution = torch.tensor(frequencies ** 0.75 / (frequencies ** 0.75).sum(), dtype=torch.float32)
    device = _device(cfg)
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
    cpu_vectors = (torch.rand((len(vocabulary), cfg.embed_dim), generator=generator) - 0.5) / cfg.embed_dim
    input_layer = torch.nn.Embedding.from_pretrained(cpu_vectors.to(device), freeze=False)
    context_layer = torch.nn.Embedding.from_pretrained(torch.zeros_like(cpu_vectors).to(device), freeze=False)
    optimizer = torch.optim.Adam(list(input_layer.parameters()) + list(context_layer.parameters()), lr=cfg.skipgram_lr)
    rng, pair_count, last_loss = np.random.default_rng(cfg.seed), 0, None
    for epoch in range(cfg.skipgram_epochs):
        current_count = 0
        for centers, contexts in _pair_batches(walks, vocabulary, cfg.window, rng.permutation(len(walks))):
            center_ids = torch.tensor(centers, dtype=torch.long, device=device)
            context_ids = torch.tensor(contexts, dtype=torch.long, device=device)
            negative_ids = torch.multinomial(distribution, len(centers) * cfg.negative_samples, replacement=True, generator=generator).reshape(len(centers), cfg.negative_samples).to(device)
            optimizer.zero_grad()
            loss = _loss(input_layer(center_ids), context_layer(context_ids), context_layer(negative_ids))
            if not torch.isfinite(loss):
                raise ValueError("nonfinite Skip-Gram loss")
            loss.backward()
            optimizer.step()
            current_count += len(centers)
            last_loss = float(loss.detach().cpu())
        if epoch == 0:
            pair_count = current_count
    metadata = {"seed": cfg.seed, "walk_count": len(walks), "pair_count": pair_count, "vocabulary_size": len(vocabulary), "epochs": cfg.skipgram_epochs, "last_loss": last_loss, "fit_scope": "training_walk_corpus"}
    return EmbeddingState(vocabulary, input_layer.weight.detach().cpu().numpy(), context_layer.weight.detach().cpu().numpy(), frequencies, metadata)


def embed_nodes(split, state, cfg, walks=None):
    ids = list(map(str, split.node_ids))
    unknown = [token for token in ids if token not in state.vocabulary]
    if unknown and cfg.inference_policy == "strict":
        raise ValueError(f"unseen Skip-Gram node IDs: {len(unknown)}; select and disclose an inference policy before evaluating new nodes")
    result = np.empty((len(ids), state.input_vectors.shape[1]), dtype=np.float32)
    inferred = {}
    if unknown:
        if walks is None:
            raise ValueError("frozen-context inference requires evaluation walks")
        contexts = {token: [] for token in unknown}
        for sequence in walks:
            sequence = list(map(str, sequence))
            for i, token in enumerate(sequence):
                if token in contexts:
                    contexts[token].extend(state.vocabulary[sequence[j]] for j in range(max(0, i - cfg.window), min(len(sequence), i + cfg.window + 1)) if i != j and sequence[j] in state.vocabulary)
        missing = [token for token, entries in contexts.items() if not entries]
        if missing:
            raise ValueError(f"no training-context anchor for {len(missing)} unseen nodes; frozen Skip-Gram cannot identify their embedding space")
        device = _device(cfg)
        distribution = torch.tensor(state.frequencies ** 0.75 / (state.frequencies ** 0.75).sum(), dtype=torch.float32)
        fixed_context = torch.tensor(state.context_vectors, dtype=torch.float32, device=device)
        generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
        unknown_index = {token: index for index, token in enumerate(unknown)}
        initial = np.stack([state.input_vectors[entries].mean(0) for entries in contexts.values()])
        vectors = torch.nn.Parameter(torch.tensor(initial, dtype=torch.float32, device=device))
        optimizer = torch.optim.Adam([vectors], lr=cfg.skipgram_lr)
        pairs = [(unknown_index[token], context) for token, entries in contexts.items() for context in entries]
        rng = np.random.default_rng(cfg.seed)
        for _ in range(cfg.inference_epochs):
            for order in np.array_split(rng.permutation(len(pairs)), max(1, (len(pairs) + 1023) // 1024)):
                centers, positives = zip(*(pairs[int(i)] for i in order))
                center_ids = torch.tensor(centers, dtype=torch.long, device=device)
                positive_ids = torch.tensor(positives, dtype=torch.long, device=device)
                negative_ids = torch.multinomial(distribution, len(order) * cfg.negative_samples, replacement=True, generator=generator).reshape(len(order), cfg.negative_samples).to(device)
                optimizer.zero_grad()
                loss = _loss(vectors[center_ids], fixed_context[positive_ids], fixed_context[negative_ids])
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite frozen-context inference loss")
                loss.backward()
                optimizer.step()
        inferred = {token: vectors[i].detach().cpu().numpy() for token, i in unknown_index.items()}
    for i, token in enumerate(ids):
        result[i] = state.input_vectors[state.vocabulary[token]] if token in state.vocabulary else inferred[token]
    if not np.isfinite(result).all():
        raise ValueError("nonfinite node embeddings")
    return result
