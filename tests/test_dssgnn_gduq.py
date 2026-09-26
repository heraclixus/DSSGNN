"""Tests for DSS-GNN + GDUQ anchor model (DSSGNNWithGDUQAnchor)."""

import numpy as np
import scipy.sparse as sp
import torch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dssgnn.chebyshev import build_rescaled_laplacian
from dssgnn.model import DSSGNN, DSSGNNTFE
from dssgnn.dssgnn_gduq import DSSGNNWithGDUQAnchor, build_dssgnn_gduq


def _make_adj(n, density=0.2):
    adj = sp.eye(n) + sp.random(n, n, density=density, format='csr')
    adj = adj + adj.T
    adj.data = np.minimum(adj.data, 1.0)
    return adj


def test_dssgnn_gduq_forward_shape():
    """DSSGNNWithGDUQAnchor forward returns correct logits shape."""
    n, input_dim, hidden_dim, out_dim = 20, 10, 32, 5
    adj = _make_adj(n)
    L = build_rescaled_laplacian(adj)
    mean = torch.randn(input_dim)
    std = torch.ones(input_dim) * 0.5

    model = DSSGNNWithGDUQAnchor(
        base_model_class=DSSGNN,
        base_model_kwargs=dict(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=2,
            K_lp=2,
            K_hp=2,
            P=1,
            dropout=(0.0, 0.0),
        ),
        mean=mean,
        std=std,
    )
    x = torch.randn(n, input_dim)
    logits = model(L, x, n_anchors=3)
    assert logits.shape == (n, out_dim)


def test_dssgnn_gduq_forward_with_std():
    """DSSGNNWithGDUQAnchor returns (logits_mean, logits_std) when return_std=True."""
    n, input_dim, hidden_dim, out_dim = 15, 8, 16, 3
    adj = _make_adj(n)
    L = build_rescaled_laplacian(adj)
    mean = torch.randn(input_dim)
    std = torch.ones(input_dim) * 0.3

    model = DSSGNNWithGDUQAnchor(
        base_model_class=DSSGNN,
        base_model_kwargs=dict(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=2,
            K_lp=2,
            K_hp=2,
            P=1,
            dropout=(0.0, 0.0),
        ),
        mean=mean,
        std=std,
    )
    x = torch.randn(n, input_dim)
    model.eval()
    with torch.no_grad():
        mu, sigma = model(L, x, n_anchors=5, return_std=True)
    assert mu.shape == (n, out_dim)
    assert sigma.shape == (n, out_dim)
    assert (sigma >= 0).all()


def test_dssgnn_gduq_process_batch():
    """process_batch produces concat(x - anchor, anchor) with 2*input_dim."""
    n, input_dim = 10, 6
    mean = torch.randn(input_dim)
    std = torch.ones(input_dim) * 0.5

    model = DSSGNNWithGDUQAnchor(
        base_model_class=DSSGNN,
        base_model_kwargs=dict(
            input_dim=2 * input_dim,
            hidden_dim=8,
            out_dim=2,
            num_layers=1,
            K_lp=2,
            K_hp=2,
            P=0,
            dropout=(0.0, 0.0),
        ),
        mean=mean,
        std=std,
    )
    x = torch.randn(n, input_dim)
    out = model.process_batch(x, anchors=None)
    assert out.shape == (n, 2 * input_dim)

    # With explicit anchor
    anchor = torch.randn(1, input_dim)
    out = model.process_batch(x, anchors=anchor)
    assert out.shape == (n, 2 * input_dim)


def test_dssgnn_gduq_gradient_flow():
    """Gradients flow through DSSGNNWithGDUQAnchor."""
    n, input_dim, hidden_dim, out_dim = 8, 4, 8, 3
    adj = _make_adj(n)
    L = build_rescaled_laplacian(adj)
    mean = torch.randn(input_dim)
    std = torch.ones(input_dim) * 0.5

    model = DSSGNNWithGDUQAnchor(
        base_model_class=DSSGNN,
        base_model_kwargs=dict(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=2,
            K_lp=2,
            K_hp=2,
            P=1,
            dropout=(0.0, 0.0),
        ),
        mean=mean,
        std=std,
    )
    x = torch.randn(n, input_dim, requires_grad=True)
    logits = model(L, x, n_anchors=2)
    loss = logits.sum()
    loss.backward()
    assert x.grad is not None


def test_dssgnn_gduq_calibrate():
    """calibrate produces same shape as input."""
    n, input_dim, out_dim = 10, 4, 3
    adj = _make_adj(n)
    L = build_rescaled_laplacian(adj)
    mean = torch.randn(input_dim)
    std = torch.ones(input_dim) * 0.5
    model = DSSGNNWithGDUQAnchor(
        base_model_class=DSSGNN,
        base_model_kwargs=dict(
            input_dim=input_dim,
            hidden_dim=8,
            out_dim=out_dim,
            num_layers=1,
            K_lp=2,
            K_hp=2,
            P=0,
            dropout=(0.0, 0.0),
        ),
        mean=mean,
        std=std,
    )
    mu = torch.randn(n, out_dim)
    std_vals = torch.softmax(torch.randn(n, out_dim), dim=1)
    calibrated = model.calibrate(mu, std_vals)
    assert calibrated.shape == (n, out_dim)


def test_build_dssgnn_gduq():
    """build_dssgnn_gduq returns model with correct output shape."""
    n, input_dim, num_classes = 30, 8, 4
    adj = _make_adj(n)
    L = build_rescaled_laplacian(adj)
    features = torch.randn(n, input_dim)
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[:20] = True

    model = build_dssgnn_gduq(
        DSSGNN,
        features,
        train_mask,
        num_classes,
        hidden_dim=16,
        num_layers=2,
        K_lp=2,
        K_hp=2,
        P=1,
    )
    assert isinstance(model, DSSGNNWithGDUQAnchor)
    logits = model(L, features, n_anchors=2)
    assert logits.shape == (n, num_classes)


def test_dssgnn_gduq_with_tfe_base():
    """DSSGNNWithGDUQAnchor works with DSSGNNTFE base (tuple graph_input)."""
    n, input_dim, hidden_dim, out_dim = 12, 6, 16, 4
    adj = _make_adj(n)
    from tfe_utils import propagate_adj
    adj_lp = propagate_adj(adj, "low", -0.5, -0.5)
    adj_hp = propagate_adj(adj, "high", -0.3, -0.3)
    mean = torch.randn(input_dim)
    std = torch.ones(input_dim) * 0.4

    model = DSSGNNWithGDUQAnchor(
        base_model_class=DSSGNNTFE,
        base_model_kwargs=dict(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=2,
            K_lp=2,
            K_hp=2,
            P=1,
            dropout=(0.0, 0.0),
        ),
        mean=mean,
        std=std,
    )
    x = torch.randn(n, input_dim)
    graph_input = (adj_lp, adj_hp)
    logits = model(graph_input, x, n_anchors=2)
    assert logits.shape == (n, out_dim)


def test_dssgnn_gduq_training_step():
    """DSSGNNWithGDUQAnchor can run a training step (sanity check)."""
    n, input_dim, num_classes = 25, 6, 3
    adj = _make_adj(n)
    L = build_rescaled_laplacian(adj)
    features = torch.randn(n, input_dim)
    labels = torch.randint(0, num_classes, (n,))
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[:15] = True

    model = build_dssgnn_gduq(
        DSSGNN,
        features,
        train_mask,
        num_classes,
        hidden_dim=12,
        num_layers=2,
        K_lp=2,
        K_hp=2,
        P=1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    optimizer.zero_grad()
    logits = model(L, features, anchors=None, n_anchors=3)
    loss = torch.nn.functional.cross_entropy(logits[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()
    assert not torch.isnan(loss)


if __name__ == "__main__":
    test_dssgnn_gduq_forward_shape()
    test_dssgnn_gduq_forward_with_std()
    test_dssgnn_gduq_process_batch()
    test_dssgnn_gduq_gradient_flow()
    test_dssgnn_gduq_calibrate()
    test_build_dssgnn_gduq()
    test_dssgnn_gduq_with_tfe_base()
    test_dssgnn_gduq_training_step()
    print("All DSS-GNN+GDUQ tests passed.")
