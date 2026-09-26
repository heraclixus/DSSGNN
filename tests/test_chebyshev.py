"""Tests for Chebyshev filter module."""

import numpy as np
import scipy.sparse as sp
import torch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dssgnn.chebyshev import build_rescaled_laplacian, chebyshev_filter


def test_build_rescaled_laplacian_shape():
    """Rescaled Laplacian has correct shape."""
    n = 10
    adj = sp.eye(n) + sp.random(n, n, density=0.2, format='csr')
    adj = adj + adj.T
    adj.data = np.minimum(adj.data, 1.0)
    L = build_rescaled_laplacian(adj)
    assert L.shape == (n, n)
    assert L.is_sparse


def test_build_rescaled_laplacian_symmetric():
    """Rescaled Laplacian is symmetric."""
    n = 20
    adj = sp.eye(n) + sp.random(n, n, density=0.15, format='csr')
    adj = adj + adj.T
    adj.data = np.minimum(adj.data, 1.0)
    L = build_rescaled_laplacian(adj)
    L_dense = L.to_dense()
    np.testing.assert_allclose(L_dense.numpy(), L_dense.numpy().T, atol=1e-5)


def test_chebyshev_filter_identity():
    """T_0(L̃)X = X, so coeffs=[1,0,0,...] gives X."""
    n, d = 15, 4
    adj = sp.eye(n) + sp.random(n, n, density=0.2, format='csr')
    adj = adj + adj.T
    adj.data = np.minimum(adj.data, 1.0)
    L = build_rescaled_laplacian(adj)
    x = torch.randn(n, d)
    out = chebyshev_filter(L, x, [1.0, 0.0, 0.0])
    torch.testing.assert_close(out, x)


def test_chebyshev_filter_linear():
    """Filter with coeffs [0,1,0] gives L̃ @ x."""
    n, d = 12, 3
    adj = sp.eye(n) + sp.random(n, n, density=0.25, format='csr')
    adj = adj + adj.T
    adj.data = np.minimum(adj.data, 1.0)
    L = build_rescaled_laplacian(adj)
    x = torch.randn(n, d)
    out = chebyshev_filter(L, x, [0.0, 1.0, 0.0])
    expected = torch.sparse.mm(L, x)
    torch.testing.assert_close(out, expected)


def test_chebyshev_filter_output_shape():
    """Output shape matches input shape."""
    n, d = 8, 5
    adj = sp.eye(n) + sp.random(n, n, density=0.3, format='csr')
    adj = adj + adj.T
    adj.data = np.minimum(adj.data, 1.0)
    L = build_rescaled_laplacian(adj)
    x = torch.randn(n, d)
    coeffs = [0.5, 0.3, 0.2]
    out = chebyshev_filter(L, x, coeffs)
    assert out.shape == (n, d)


def test_chebyshev_filter_gradient():
    """Chebyshev filter supports gradients."""
    n, d = 6, 2
    adj = sp.eye(n) + sp.random(n, n, density=0.3, format='csr')
    adj = adj + adj.T
    adj.data = np.minimum(adj.data, 1.0)
    L = build_rescaled_laplacian(adj)
    x = torch.randn(n, d, requires_grad=True)
    coeffs = torch.tensor([0.5, 0.3], requires_grad=True)
    out = chebyshev_filter(L, x, coeffs)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert coeffs.grad is not None


if __name__ == "__main__":
    test_build_rescaled_laplacian_shape()
    test_build_rescaled_laplacian_symmetric()
    test_chebyshev_filter_identity()
    test_chebyshev_filter_linear()
    test_chebyshev_filter_output_shape()
    test_chebyshev_filter_gradient()
    print("All chebyshev tests passed.")
