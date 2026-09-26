"""Tests for DSS-Conv non-intrusive layer (method.tex Algorithm 1)."""

import numpy as np
import scipy.sparse as sp
import torch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dssgnn.chebyshev import build_rescaled_laplacian
from dssgnn.dsconv_nonintrusive import DSSConvNonIntrusive


def _make_adj(n, density=0.2):
    adj = sp.eye(n) + sp.random(n, n, density=density, format='csr')
    adj = adj + adj.T
    adj.data = np.minimum(adj.data, 1.0)
    return adj


def test_dssconv_nonintrusive_shape():
    """DSSConvNonIntrusive output has correct shape."""
    n, in_dim, out_dim = 10, 4, 8
    P, K, S = 1, 2, 4
    adj = _make_adj(n)
    L = build_rescaled_laplacian(adj)
    conv = DSSConvNonIntrusive(in_dim, out_dim, K_lp=K, K_hp=K, P=P, S=S)
    H_list = [torch.randn(n, in_dim) for _ in range(P + 1)]
    out_list = conv(L, H_list)
    assert len(out_list) == P + 1
    for out in out_list:
        assert out.shape == (n, out_dim)


def test_dssconv_nonintrusive_gradient():
    """Gradients flow through DSSConvNonIntrusive."""
    n, in_dim, out_dim = 6, 3, 4
    P, K, S = 1, 2, 4
    adj = _make_adj(n)
    L = build_rescaled_laplacian(adj)
    conv = DSSConvNonIntrusive(in_dim, out_dim, K_lp=K, K_hp=K, P=P, S=S)
    H_list = [torch.randn(n, in_dim, requires_grad=True) for _ in range(P + 1)]
    out_list = conv(L, H_list)
    loss = sum(o.sum() for o in out_list)
    loss.backward()
    for h in H_list:
        assert h.grad is not None


def test_dssconv_p0_single_channel():
    """When P=0, single channel works."""
    n, in_dim, out_dim = 8, 4, 6
    K, S = 2, 4
    adj = _make_adj(n)
    L = build_rescaled_laplacian(adj)
    conv = DSSConvNonIntrusive(in_dim, out_dim, K_lp=K, K_hp=K, P=0, S=S)
    H_list = [torch.randn(n, in_dim)]
    out_list = conv(L, H_list)
    assert len(out_list) == 1
    assert out_list[0].shape == (n, out_dim)


if __name__ == "__main__":
    test_dssconv_nonintrusive_shape()
    test_dssconv_nonintrusive_gradient()
    test_dssconv_p0_single_channel()
    print("All DSS-Conv tests passed.")
