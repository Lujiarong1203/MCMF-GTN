import torch
import torch.nn as nn
import torch.nn.functional as F


class GTNLayer(nn.Module):
    """Softmax-weighted channel mixing: Q = sum_c softmax(w)[c] * A^(c)."""

    def __init__(self, num_channels, layer_idx=0):
        super().__init__()
        base = torch.ones(num_channels)
        if num_channels > 1:
            base = base + 0.05 * (layer_idx - 1.0)
        self.channel_weight = nn.Parameter(base)

    def forward_edge(self, edge_attr):
        w = F.softmax(self.channel_weight, dim=0)
        return (edge_attr * w.unsqueeze(0)).sum(dim=1)

    def forward_matrix(self, adj_channels):
        w = F.softmax(self.channel_weight, dim=0)
        Q_vals = sum(w[c] * adj_channels[c].values() for c in range(len(adj_channels)))
        return torch.sparse_coo_tensor(
            adj_channels[0].indices(), Q_vals, adj_channels[0].size()
        ).coalesce()


class ChannelFusionGenerator(nn.Module):
    """L GTN layers; matrix mode multiplies the layer matrices into a
    meta-path adjacency A_P = Q_1 Q_2 ... Q_L with per-row top-K pruning."""

    def __init__(self, num_channels, num_layers=3, use_matrix_mult=False, top_k=10):
        super().__init__()
        self.num_layers = num_layers
        self.use_matrix_mult = use_matrix_mult
        self.top_k = top_k
        self.layers = nn.ModuleList(
            [GTNLayer(num_channels, layer_idx=i) for i in range(num_layers)]
        )
        self.layer_weights = nn.Parameter(torch.ones(num_layers))

    def forward_edge(self, edge_attr):
        Q_list = [layer.forward_edge(edge_attr) for layer in self.layers]
        w = F.softmax(self.layer_weights, dim=0)
        return sum(w[i] * Q_list[i] for i in range(self.num_layers))

    def _add_self_loops(self, Q):
        n = Q.size(0)
        eye_idx = torch.arange(n, device=Q.device).unsqueeze(0).repeat(2, 1)
        eye_val = torch.ones(n, device=Q.device, dtype=Q.dtype)
        all_idx = torch.cat([Q.indices(), eye_idx], dim=1)
        all_val = torch.cat([Q.values(), eye_val], dim=0)
        return torch.sparse_coo_tensor(all_idx, all_val, (n, n)).coalesce()

    def _row_normalize(self, Q):
        n = Q.size(0)
        row_sum = torch.sparse.sum(Q, dim=1).to_dense() + 1e-8
        src_nodes = Q.indices()[0]
        norm_vals = Q.values() / row_sum[src_nodes]
        return torch.sparse_coo_tensor(Q.indices(), norm_vals, (n, n)).coalesce()

    def _sparse_top_k_per_row(self, Q, k):
        Q = Q.coalesce()
        n = Q.size(0)

        Q_csr = Q.to_sparse_csr()
        crow = Q_csr.crow_indices()
        col = Q_csr.col_indices()
        val = Q_csr.values()

        # 逐行判断是否需要剪枝（不能用全局 nnz 早退：总非零数 <= n*k 时仍可能有单行超过 k 项）
        row_lens = crow[1:] - crow[:-1]
        if bool((row_lens <= k).all()):
            return Q

        new_rows, new_cols, new_vals = [], [], []
        for i in range(n):
            start, end = crow[i].item(), crow[i + 1].item()
            if start == end:
                continue
            row_vals = val[start:end]
            row_cols = col[start:end]
            if len(row_vals) <= k:
                top_vals = row_vals
                top_cols = row_cols
            else:
                topk = torch.topk(row_vals, k)
                top_vals = topk.values
                top_cols = row_cols[topk.indices]
            # 值张量保持 autograd 图（不用 tolist 重建），索引为结构性张量
            new_rows.append(torch.full((top_vals.numel(),), i, device=Q.device, dtype=torch.long))
            new_cols.append(top_cols.to(dtype=torch.long))
            new_vals.append(top_vals)

        idx = torch.stack([torch.cat(new_rows), torch.cat(new_cols)], dim=0)
        vals = torch.cat(new_vals)
        return torch.sparse_coo_tensor(idx, vals, (n, n)).coalesce()

    def forward_matrix(self, adj_channels):
        num_channels = len(adj_channels)
        if num_channels == 1:
            Q = self.layers[0].forward_matrix(adj_channels)
            Q = self._add_self_loops(Q)
            return self._row_normalize(Q)

        A_P = None
        for layer in self.layers:
            Q = layer.forward_matrix(adj_channels)
            Q = self._add_self_loops(Q)
            Q = self._row_normalize(Q)
            if A_P is None:
                A_P = Q
            else:
                A_P = torch.sparse.mm(A_P, Q).coalesce()
                if self.top_k is not None:
                    A_P = self._sparse_top_k_per_row(A_P, self.top_k)
        return A_P

    def forward(self, edge_attr_or_adj_channels):
        if isinstance(edge_attr_or_adj_channels, list):
            return self.forward_matrix(edge_attr_or_adj_channels)
        return self.forward_edge(edge_attr_or_adj_channels)
