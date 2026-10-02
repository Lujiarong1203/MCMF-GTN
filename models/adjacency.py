"""Differentiable sparse row-oriented adjacency operations."""

import torch


MAX_PRODUCT_ENTRIES = 20_000_000


def _coo(indices, values, num_nodes):
    if not torch.isfinite(values).all():
        raise ValueError("Adjacency values must be finite; reduce edge-weight magnitude.")
    matrix = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes), device=values.device).coalesce()
    if not torch.isfinite(matrix.values()).all():
        raise ValueError("Duplicate-edge summation overflowed; reduce edge-weight magnitude.")
    return matrix


def _checked(adj):
    if adj.layout != torch.sparse_coo or adj.ndim != 2 or adj.shape[0] != adj.shape[1] or adj.sparse_dim() != 2:
        raise ValueError("Adjacency must be a square sparse COO tensor.")
    indices = adj._indices()
    if indices.numel() and (torch.any(indices < 0) or torch.any(indices >= adj.shape[0])):
        raise ValueError("Adjacency contains an index outside its node range.")
    adj = adj.coalesce()
    values = adj.values()
    if not values.is_floating_point() or not torch.isfinite(values).all():
        raise ValueError("Adjacency values must be finite floating-point numbers.")
    if torch.any(values < 0):
        raise ValueError("Adjacency values must be nonnegative.")
    return _coo(adj.indices(), values, adj.shape[0])


def validate_edges(edge_index, edge_attr, num_nodes, num_channels=None, weighted=True):
    if not isinstance(num_nodes, int) or num_nodes < 0:
        raise ValueError("num_nodes must be a nonnegative integer.")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype != torch.long:
        raise ValueError("edge_index must have shape [2, E] and dtype torch.long.")
    if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[1]:
        raise ValueError("edge_attr must have shape [E, C].")
    if edge_attr.shape[1] == 0 or (num_channels is not None and edge_attr.shape[1] != num_channels):
        raise ValueError("edge_attr channel count does not match the model.")
    if edge_index.device != edge_attr.device:
        raise ValueError("edge_index and edge_attr must be on the same device.")
    if not edge_attr.is_floating_point():
        raise ValueError("edge_attr must contain floating-point numbers.")
    if edge_index.numel() and (torch.any(edge_index < 0) or torch.any(edge_index >= num_nodes)):
        raise ValueError("edge_index contains an endpoint outside [0, num_nodes).")
    if weighted and (not torch.isfinite(edge_attr).all() or torch.any(edge_attr < 0)):
        raise ValueError("Edge-channel values must be finite and nonnegative.")


def row_normalize(adj):
    """Normalize nonempty rows; empty rows stay zero."""
    adj = _checked(adj)
    rows = adj.indices()[0]
    values = adj.values()
    if adj.is_cuda and adj.dtype != torch.float64 and torch.any((values > 0) & (values < torch.finfo(adj.dtype).tiny)):
        values = values.to(torch.float64)
    totals = torch.zeros(adj.shape[0], dtype=values.dtype, device=adj.device).index_add(0, rows, values)
    if not torch.isfinite(totals).all():
        raise ValueError("Adjacency row sums overflowed; reduce edge-weight magnitude.")
    row_totals = totals[rows]
    denominators = torch.where(row_totals > 0, row_totals, torch.ones_like(row_totals))
    return _coo(adj.indices(), (values / denominators).to(adj.dtype), adj.shape[0])


def add_self_loops(adj):
    """Add one identity matrix, including to existing diagonal weights."""
    adj = _checked(adj)
    diagonal = torch.arange(adj.shape[0], device=adj.device)
    indices = torch.cat((adj.indices(), torch.stack((diagonal, diagonal))), dim=1)
    values = torch.cat((adj.values(), torch.ones(adj.shape[0], device=adj.device, dtype=adj.dtype)))
    return _coo(indices, values, adj.shape[0])


def prune_topk(adj, k):
    """Keep at most k largest stored entries in each row without rescaling."""
    if not isinstance(k, int) or k < 1:
        raise ValueError("top_k must be a positive integer.")
    adj = _checked(adj)
    if adj._nnz() == 0:
        return adj
    rows = adj.indices()[0]
    order = torch.argsort(adj.values(), descending=True, stable=True)
    order = order[torch.argsort(rows[order], stable=True)]
    counts = torch.bincount(rows, minlength=adj.shape[0])
    starts = counts.cumsum(0) - counts
    ranks = torch.arange(order.numel(), device=adj.device) - torch.repeat_interleave(starts, counts)
    selected = order[ranks < k]
    return _coo(adj.indices()[:, selected], adj.values()[selected], adj.shape[0])


def sparse_product(left, right):
    """Expand matching COO paths while retaining gradients of their products."""
    left, right = _checked(left), _checked(right)
    if left.shape != right.shape or left.device != right.device or left.dtype != right.dtype:
        raise ValueError("Sparse product operands must have matching shapes, devices and dtypes.")
    n = left.shape[0]
    right_counts = torch.bincount(right.indices()[0], minlength=n)
    repeats = right_counts[left.indices()[1]]
    candidates = int(repeats.sum().item())
    if candidates > MAX_PRODUCT_ENTRIES:
        raise RuntimeError(
            f"Sparse product candidate expansion has {candidates:,} entries, exceeding "
            f"the {MAX_PRODUCT_ENTRIES:,}-entry limit. Reduce graph size or top_k."
        )
    if candidates == 0:
        values = left.values()[:0] + right.values()[:0]
        return _coo(left.indices()[:, :0], values, n)
    left_positions = torch.repeat_interleave(torch.arange(left._nnz(), device=left.device), repeats)
    candidate_starts = repeats.cumsum(0) - repeats
    offsets = torch.arange(candidates, device=left.device) - torch.repeat_interleave(candidate_starts, repeats)
    right_starts = right_counts.cumsum(0) - right_counts
    right_positions = right_starts[left.indices()[1, left_positions]] + offsets
    indices = torch.stack((left.indices()[0, left_positions], right.indices()[1, right_positions]))
    values = left.values()[left_positions] * right.values()[right_positions]
    return _coo(indices, values, n)


def weighted_sum(adjacencies, coefficients):
    indices = torch.cat([adj.indices() for adj in adjacencies], dim=1)
    values = torch.cat([adj.values() * coefficient for adj, coefficient in zip(adjacencies, coefficients)])
    return _coo(indices, values, adjacencies[0].shape[0])


def compose_adjacency(edge_index, edge_attr, channel_logits, num_nodes, top_k, products=True):
    """Form normalized channel mixtures and optionally compose their L factors."""
    validate_edges(edge_index, edge_attr, num_nodes)
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    if channel_logits.ndim != 2 or channel_logits.shape[0] == 0 or channel_logits.shape[1] != edge_attr.shape[1]:
        raise ValueError("channel_logits must have shape [L, C] with L > 0.")
    if channel_logits.device != edge_attr.device or not torch.isfinite(channel_logits).all():
        raise ValueError("channel_logits must be finite and on the graph device.")
    weights = channel_logits.to(dtype=edge_attr.dtype).softmax(dim=-1)
    factors = [row_normalize(add_self_loops(_coo(edge_index, edge_attr @ weight, num_nodes))) for weight in weights]
    if not products:
        coefficients = torch.full((len(factors),), 1 / len(factors), dtype=edge_attr.dtype, device=edge_attr.device)
        return weighted_sum(factors, coefficients)
    result = factors[0]
    for factor in factors[1:]:
        result = prune_topk(sparse_product(result, factor), top_k)
    return result


def topology_adjacency(edge_index, edge_attr, num_nodes):
    validate_edges(edge_index, edge_attr, num_nodes, weighted=False)
    weights = torch.ones(edge_index.shape[1], device=edge_attr.device, dtype=edge_attr.dtype)
    topology = _coo(edge_index, weights, num_nodes)
    topology = _coo(topology.indices(), torch.ones_like(topology.values()), num_nodes)
    return row_normalize(add_self_loops(topology))
