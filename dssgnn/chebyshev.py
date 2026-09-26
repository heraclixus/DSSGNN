"""
Chebyshev polynomial filters on the rescaled graph Laplacian.

Implements the scalable formulation from method.tex:
  L̃_G = (2/λ_max) L_G - I
  T_0(L̃)X = X, T_1(L̃)X = L̃X, T_k(L̃)X = 2*L̃*T_{k-1}(L̃)X - T_{k-2}(L̃)X
"""

import numpy as np
import scipy.sparse as sp
import torch


def _sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert scipy sparse to torch sparse (reused pattern from tfe_utils)."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)


def build_rescaled_laplacian(adj, lambda_max=2.0):
    """
    Build rescaled Laplacian L̃ = (2/λ_max) * L - I for Chebyshev filters.

    Uses normalized Laplacian L = I - D^{-1/2} A D^{-1/2} (A without self-loops).
    For normalized L, eigenvalues are in [0, 2], so λ_max=2 is standard.

    Args:
        adj: scipy sparse adjacency matrix (with self-loops)
        lambda_max: maximum eigenvalue (default 2 for normalized Laplacian)

    Returns:
        L_rescaled: torch sparse tensor of shape (N, N)
    """
    if sp.issparse(adj):
        adj = adj.tocsr()
    else:
        adj = sp.csr_matrix(adj)

    # Remove self-loops for Laplacian
    n = adj.shape[0]
    adj_no_self = adj - sp.diags(adj.diagonal(), format='csr')
    adj_no_self = adj_no_self + adj_no_self.T
    adj_no_self = adj_no_self / 2  # symmetrize
    adj_no_self.data = np.minimum(adj_no_self.data, 1.0)  # cap at 1

    # Degree matrix
    d = np.array(adj_no_self.sum(axis=1)).flatten()
    d[d == 0] = 1  # avoid division by zero for isolated nodes
    d_inv_sqrt = np.power(d, -0.5)
    D_inv_sqrt = sp.diags(d_inv_sqrt, format='csr')

    # Normalized adjacency A_norm = D^{-1/2} A D^{-1/2}
    A_norm = D_inv_sqrt.dot(adj_no_self).dot(D_inv_sqrt)

    # Normalized Laplacian L = I - A_norm
    L = sp.eye(n, format='csr') - A_norm

    # Rescaled: L̃ = (2/λ_max) * L - I  =>  eigenvalues in [-1, 1]
    L_rescaled = (2.0 / lambda_max) * L - sp.eye(n, format='csr')
    L_rescaled = L_rescaled.tocoo()

    return _sparse_mx_to_torch_sparse_tensor(L_rescaled.astype(np.float32))


def chebyshev_filter(L_rescaled, x, coeffs):
    """
    Evaluate Chebyshev polynomial filter: sum_k c_k * T_k(L̃) * x

    Uses recurrence: T_0(X)=X, T_1(X)=L̃X, T_k(X)=2*L̃*T_{k-1}(X) - T_{k-2}(X)

    Args:
        L_rescaled: torch sparse tensor (N, N)
        x: torch tensor (N, d)
        coeffs: list or tensor of K+1 coefficients [c_0, c_1, ..., c_K]

    Returns:
        out: torch tensor (N, d)
    """
    if isinstance(coeffs, (list, tuple)):
        coeffs = torch.tensor(coeffs, dtype=x.dtype, device=x.device)
    elif not isinstance(coeffs, torch.Tensor):
        coeffs = torch.tensor(coeffs, dtype=x.dtype, device=x.device)

    # Ensure L_rescaled is on same device as x
    if L_rescaled.device != x.device:
        L_rescaled = L_rescaled.to(x.device)

    # Sparse matmul: torch.sparse.mm does not support float16 on CUDA.
    # Disable autocast for this op to force float32.
    def matmul(L, X):
        if L.is_sparse:
            orig_dtype = X.dtype
            if L.is_cuda:
                with torch.cuda.amp.autocast(enabled=False):
                    out = torch.sparse.mm(L.float(), X.float())
            else:
                out = torch.sparse.mm(L.float(), X.float())
            return out.to(orig_dtype)
        return torch.mm(L, X)

    K = len(coeffs) - 1
    if K < 0:
        return torch.zeros_like(x)

    # T_0(X) = X
    T_prev = x
    out = coeffs[0] * T_prev

    if K == 0:
        return out

    # T_1(X) = L̃ X
    T_curr = matmul(L_rescaled, x)
    out = out + coeffs[1] * T_curr

    for k in range(2, K + 1):
        # T_k = 2 * L̃ * T_{k-1} - T_{k-2}
        T_next = 2 * matmul(L_rescaled, T_curr) - T_prev
        out = out + coeffs[k] * T_next
        T_prev, T_curr = T_curr, T_next

    return out
