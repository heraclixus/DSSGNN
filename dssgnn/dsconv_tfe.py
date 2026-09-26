"""
DS-Conv with TFE-style propagation.

Uses the same adjacency-based polynomial propagation as TFE-GNN:
  - Low-pass: A_lp = D^{-0.5} A D^{-0.5}, filter = sum_k coeffs_lp[k] * A_lp^k
  - High-pass: A_hp = I - D^{-eta} A D^{-eta}, filter = sum_k coeffs_hp[k] * A_hp^k

This aligns DS-GNN's graph spectral axis with TFE-GNN for better heterophily performance.
"""

import math
import torch
import torch.nn as nn


def _safe_sparse_dense_mm(adj, x):
    """Matrix multiply adj @ x. Handles both sparse and dense adj. Uses float32 for sparse (no fp16 support)."""
    if adj.is_sparse:
        orig_dtype = x.dtype
        if adj.is_cuda:
            with torch.cuda.amp.autocast(enabled=False):
                out = torch.sparse.mm(adj.float(), x.float())
        else:
            out = torch.sparse.mm(adj.float(), x.float())
        return out.to(orig_dtype)
    return torch.mm(adj, x)


def mix_prop(adj, x, coe):
    """Polynomial propagation: x + coe[1]*A*x + coe[2]*A^2*x + ... (TFE-style)."""
    x0 = x.clone()
    xx = x.clone()
    for i in range(1, len(coe)):
        x0 = _safe_sparse_dense_mm(adj, x0)
        xx = xx + coe[i] * x0
    return xx


class DSConvTFE(nn.Module):
    """
    DS-Conv with TFE-style adjacency polynomial propagation.

    Uses adj_lp and adj_hp (from propagate_adj) instead of Chebyshev on Laplacian.
    Aligns with TFE-GNN's mixed spectral approach for heterophilous graphs.
    """

    def __init__(self, in_dim, out_dim, K_lp, K_hp, P, activation=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.K_lp = K_lp
        self.K_hp = K_hp
        self.P = P
        self.activation = activation

        # Learnable polynomial coefficients (TFE-style: one per order 0..K)
        self.coeffs_lp = nn.Parameter(torch.ones(K_lp + 1) * 0.5)
        self.coeffs_hp = nn.Parameter(torch.ones(K_hp + 1) * 0.5)

        # Order-specific branch weights
        self.alpha = nn.Parameter(torch.ones(P + 1) * 0.7)
        self.beta = nn.Parameter(torch.ones(P + 1) * 0.3)

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.coeffs_lp.data.fill_(0.5)
            self.coeffs_hp.data.fill_(0.5)
        stdv = 1.0 / math.sqrt(self.W.weight.size(1))
        self.W.weight.data.normal_(-stdv, stdv)

    def forward(self, adj_lp, adj_hp, H_list):
        """
        Args:
            adj_lp: low-pass adjacency (N, N) from propagate_adj
            adj_hp: high-pass adjacency (N, N) from propagate_adj
            H_list: list of (P+1) tensors (N, in_dim)

        Returns:
            list of (P+1) tensors (N, out_dim)
        """
        out_list = []
        for n in range(self.P + 1):
            h_n = H_list[n]
            g_lp_h = mix_prop(adj_lp, h_n, self.coeffs_lp)
            g_hp_h = mix_prop(adj_hp, h_n, self.coeffs_hp)
            combined = self.alpha[n] * g_lp_h + self.beta[n] * g_hp_h
            out = self.W(combined)
            if self.activation:
                out = torch.relu(out)
            out_list.append(out)
        return out_list
