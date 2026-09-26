"""Tests for DSS-GNN model (method.tex formulation)."""

import numpy as np
import scipy.sparse as sp
import torch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dssgnn.chebyshev import build_rescaled_laplacian
from dssgnn.model import DSSGNN, DSSGNNTFE
from dssgnn.dsconv_tfe_propfirst import DSSGNNTFEPropFirst
from gnnsafe_ood.backbone import GCNDSSResidualEncoder


def _make_adj(n, density=0.2):
    adj = sp.eye(n) + sp.random(n, n, density=density, format='csr')
    adj = adj + adj.T
    adj.data = np.minimum(adj.data, 1.0)
    return adj


def test_dssgnn_forward_shape():
    """DSS-GNN forward returns correct logits shape."""
    n, input_dim, hidden_dim, out_dim = 20, 10, 32, 5
    adj = _make_adj(n)
    L = build_rescaled_laplacian(adj)
    model = DSSGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_layers=2,
        K_lp=3,
        K_hp=3,
        P=2,
        dropout=(0.0, 0.0),
    )
    x = torch.randn(n, input_dim)
    logits = model(L, x)
    assert logits.shape == (n, out_dim)


def test_dssgnn_forward_with_uncertainty():
    """DSS-GNN returns uncertainty when requested."""
    n, input_dim, hidden_dim, out_dim = 15, 8, 16, 3
    adj = _make_adj(n)
    L = build_rescaled_laplacian(adj)
    model = DSSGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_layers=2,
        K_lp=2,
        K_hp=2,
        P=2,
        dropout=(0.0, 0.0),
    )
    x = torch.randn(n, input_dim)
    logits, uncertainty = model(L, x, return_uncertainty=True)
    assert logits.shape == (n, out_dim)
    assert uncertainty.shape == (n,)
    assert (uncertainty >= 0).all()


def test_dssgnn_p0_no_uncertainty():
    """When P=0, uncertainty is zero (single channel)."""
    n, input_dim, hidden_dim, out_dim = 10, 4, 8, 2
    adj = _make_adj(n)
    L = build_rescaled_laplacian(adj)
    model = DSSGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_layers=2,
        K_lp=2,
        K_hp=2,
        P=0,
        dropout=(0.0, 0.0),
    )
    x = torch.randn(n, input_dim)
    logits, uncertainty = model(L, x, return_uncertainty=True)
    torch.testing.assert_close(uncertainty, torch.zeros(n))


def test_dssgnn_gradient_flow():
    """Gradients flow through DSS-GNN."""
    n, input_dim, hidden_dim, out_dim = 8, 4, 8, 3
    adj = _make_adj(n)
    L = build_rescaled_laplacian(adj)
    model = DSSGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_layers=2,
        K_lp=2,
        K_hp=2,
        P=1,
        dropout=(0.0, 0.0),
    )
    x = torch.randn(n, input_dim, requires_grad=True)
    logits = model(L, x)
    loss = logits.sum()
    loss.backward()
    assert x.grad is not None


def test_dssgnn_deterministic():
    """Same input gives same output in eval mode."""
    n, input_dim, hidden_dim, out_dim = 12, 6, 16, 4
    adj = _make_adj(n)
    L = build_rescaled_laplacian(adj)
    model = DSSGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_layers=2,
        K_lp=2,
        K_hp=2,
        P=2,
        dropout=(0.5, 0.5),
    )
    model.eval()
    x = torch.randn(n, input_dim)
    with torch.no_grad():
        out1 = model(L, x)
        out2 = model(L, x)
    torch.testing.assert_close(out1, out2)


def test_dssgnn_tfe_forward():
    """DSSGNNTFE forward returns correct shape with adj_lp, adj_hp."""
    n, input_dim, hidden_dim, out_dim = 15, 8, 16, 3
    adj = _make_adj(n)
    from tfe_utils import propagate_adj
    adj_lp = propagate_adj(adj, "low", -0.5, -0.5)
    adj_hp = propagate_adj(adj, "high", -0.3, -0.3)
    model = DSSGNNTFE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_layers=2,
        K_lp=3,
        K_hp=3,
        P=2,
        dropout=(0.0, 0.0),
    )
    x = torch.randn(n, input_dim)
    logits, uncertainty = model(adj_lp, adj_hp, x, return_uncertainty=True)
    assert logits.shape == (n, out_dim)
    assert uncertainty.shape == (n,)


def test_dssgnn_nonintrusive_input_lift():
    """DSS-GNN (method.tex formulation) returns correct shape."""
    n, input_dim, hidden_dim, out_dim = 20, 10, 32, 5
    adj = _make_adj(n)
    L = build_rescaled_laplacian(adj)
    model = DSSGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_layers=2,
        K_lp=2,
        K_hp=2,
        P=1,
        dropout=(0.0, 0.0),
        S=4,
    )
    x = torch.randn(n, input_dim)
    logits, uncertainty = model(L, x, return_uncertainty=True)
    assert logits.shape == (n, out_dim)
    assert uncertainty.shape == (n,)
    assert (uncertainty >= 0).all()


def test_dssgnn_nonintrusive_random_gates():
    """DSS-GNN with random gates (S+G) returns correct shape."""
    n, input_dim, hidden_dim, out_dim = 15, 8, 16, 3
    adj = _make_adj(n)
    L = build_rescaled_laplacian(adj)
    model = DSSGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_layers=2,
        K_lp=2,
        K_hp=2,
        P=1,
        dropout=(0.0, 0.0),
        use_random_gates=True,
        P_gate=1,
        S=4,
    )
    x = torch.randn(n, input_dim)
    model.train()
    logits, uncertainty = model(L, x, return_uncertainty=True)
    assert logits.shape == (n, out_dim)
    assert uncertainty.shape == (n,)
    assert (uncertainty >= 0).all()


def test_dssgnn_shared_input_lift_forward():
    """Shared input lift keeps shapes intact and reduces input-lift parameters."""
    n, input_dim, hidden_dim, out_dim = 15, 8, 16, 3
    adj = _make_adj(n)
    L = build_rescaled_laplacian(adj)
    baseline = DSSGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_layers=2,
        K_lp=2,
        K_hp=2,
        P=2,
        dropout=(0.0, 0.0),
        shared_input_lift=False,
    )
    shared = DSSGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_layers=2,
        K_lp=2,
        K_hp=2,
        P=2,
        dropout=(0.0, 0.0),
        shared_input_lift=True,
    )
    x = torch.randn(n, input_dim)
    logits, uncertainty = shared(L, x, return_uncertainty=True)
    assert logits.shape == (n, out_dim)
    assert uncertainty.shape == (n,)
    baseline_params = sum(p.numel() for p in baseline.parameters())
    shared_params = sum(p.numel() for p in shared.parameters())
    assert shared_params < baseline_params


def test_dssgnn_forward_with_predictive_stats_shapes():
    """Quadrature predictive stats should be normalized and shape-consistent."""
    n, input_dim, hidden_dim, out_dim = 14, 6, 12, 3
    adj = _make_adj(n)
    L = build_rescaled_laplacian(adj)
    model = DSSGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_layers=2,
        K_lp=2,
        K_hp=2,
        P=2,
        dropout=(0.0, 0.0),
        S=4,
    )
    x = torch.randn(n, input_dim)
    stats = model.forward_with_predictive_stats(L, x)
    assert stats["logits"].shape == (n, out_dim)
    assert stats["sample_logits"].shape[1:] == (n, out_dim)
    assert stats["weights"].shape[0] == stats["sample_logits"].shape[0]
    assert stats["predictive_probs"].shape == (n, out_dim)
    assert stats["predictive_entropy"].shape == (n,)
    assert stats["expected_entropy"].shape == (n,)
    assert stats["mutual_info"].shape == (n,)
    torch.testing.assert_close(
        stats["predictive_probs"].sum(dim=-1),
        torch.ones(n, dtype=stats["predictive_probs"].dtype),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.isfinite(stats["mutual_info"]).all()


def test_dssgnn_forward_with_field_stats_shapes():
    """Continuous field statistics expose sample outputs and nonnegative variances."""
    n, input_dim, hidden_dim, out_dim = 9, 5, 12, 4
    adj = _make_adj(n)
    L = build_rescaled_laplacian(adj)
    model = DSSGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_layers=2,
        K_lp=2,
        K_hp=2,
        P=2,
        dropout=(0.0, 0.0),
        S=5,
    )
    x = torch.randn(n, input_dim)
    stats = model.forward_with_field_stats(L, x)
    assert stats["mean"].shape == (n, out_dim)
    assert stats["predictive_mean"].shape == (n, out_dim)
    assert stats["predictive_var"].shape == (n, out_dim)
    assert stats["predictive_std"].shape == (n, out_dim)
    assert stats["total_variance"].shape == (n,)
    assert stats["sample_outputs"].shape[1:] == (n, out_dim)
    assert torch.isfinite(stats["predictive_var"]).all()
    assert (stats["predictive_var"] >= 0).all()


def test_dssgnn_field_stats_p0_has_zero_variance():
    """Order-0 DSS-GNN is deterministic under field statistics."""
    n, input_dim, hidden_dim, out_dim = 8, 4, 10, 3
    adj = _make_adj(n)
    L = build_rescaled_laplacian(adj)
    model = DSSGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_layers=2,
        K_lp=2,
        K_hp=2,
        P=0,
        dropout=(0.0, 0.0),
        S=3,
    )
    x = torch.randn(n, input_dim)
    stats = model.forward_with_field_stats(L, x)
    torch.testing.assert_close(stats["predictive_var"], torch.zeros_like(stats["predictive_var"]), atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(stats["total_variance"], torch.zeros_like(stats["total_variance"]), atol=1e-7, rtol=1e-7)


def test_gcn_dss_residual_encoder_warmup_behavior():
    """Hybrid GCN+DSS encoder should act like plain GCN before residual warmup ends."""
    n, input_dim, hidden_dim, out_dim = 12, 6, 16, 4
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]],
        dtype=torch.long,
    )
    x = torch.randn(n, input_dim)
    encoder = GCNDSSResidualEncoder(
        in_channels=input_dim,
        hidden_channels=hidden_dim,
        out_channels=out_dim,
        num_layers=2,
        dropout=0.0,
        use_bn=False,
        K_lp=2,
        K_hp=2,
        P=1,
        warmup_base_epochs=2,
        residual_scale_init=0.1,
    )
    encoder.eval()
    encoder.set_train_epoch(0)
    base_logits = encoder.base_gcn(x, edge_index)
    warmup_logits = encoder(x, edge_index)
    torch.testing.assert_close(warmup_logits, base_logits)

    encoder.set_train_epoch(2)
    active_logits = encoder(x, edge_index)
    assert active_logits.shape == base_logits.shape
    assert not torch.allclose(active_logits, base_logits)
    _, uncertainty = encoder.forward_with_uncertainty(x, edge_index, uncertainty_type="chaos")
    assert uncertainty.shape == (n,)


def test_gcn_dss_residual_encoder_predictive_stats_warmup_is_deterministic():
    """During warmup, the hybrid predictive interface should collapse to the base GCN."""
    n, input_dim, hidden_dim, out_dim = 10, 5, 12, 3
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]],
        dtype=torch.long,
    )
    x = torch.randn(n, input_dim)
    encoder = GCNDSSResidualEncoder(
        in_channels=input_dim,
        hidden_channels=hidden_dim,
        out_channels=out_dim,
        num_layers=2,
        dropout=0.0,
        use_bn=False,
        K_lp=2,
        K_hp=2,
        P=1,
        warmup_base_epochs=5,
        residual_scale_init=0.1,
    )
    encoder.eval()
    encoder.set_train_epoch(0)
    stats = encoder.forward_with_predictive_stats(x, edge_index)
    base_logits = encoder.base_gcn(x, edge_index)
    torch.testing.assert_close(stats["logits"], base_logits)
    torch.testing.assert_close(stats["mutual_info"], torch.zeros(n, dtype=stats["mutual_info"].dtype), atol=1e-7, rtol=1e-7)


def test_dssgnn_reset_parameters_restores_spectral_and_gate_params():
    """Reset restores DSS-Conv spectral and gate defaults used by OOD multi-run eval."""
    model = DSSGNN(
        input_dim=8,
        hidden_dim=16,
        out_dim=3,
        num_layers=2,
        K_lp=2,
        K_hp=2,
        P=1,
        dropout=(0.0, 0.0),
        use_random_gates=True,
        P_gate=1,
        S=4,
    )
    conv = model.convs[0]
    with torch.no_grad():
        conv.coeffs_lp.fill_(42.0)
        conv.coeffs_hp.fill_(-42.0)
        conv.a_coeffs.fill_(5.0)
        conv.b_coeffs.fill_(6.0)

    model.reset_parameters()

    torch.testing.assert_close(conv.coeffs_lp, torch.tensor([1.0, 0.5, 0.25]))
    torch.testing.assert_close(conv.coeffs_hp, torch.tensor([0.0, -0.5, -0.25]))
    expected_a = torch.zeros_like(conv.a_coeffs)
    expected_b = torch.zeros_like(conv.b_coeffs)
    expected_a[:, 0] = 0.7
    expected_b[:, 0] = 0.3
    torch.testing.assert_close(conv.a_coeffs, expected_a)
    torch.testing.assert_close(conv.b_coeffs, expected_b)


def test_dssgnn_tfe_propfirst_forward():
    """DSSGNNTFEPropFirst forward returns correct shape with adj_lp, adj_hp."""
    n, input_dim, hidden_dim, out_dim = 15, 8, 16, 3
    adj = _make_adj(n)
    from tfe_utils import propagate_adj
    adj_lp = propagate_adj(adj, "low", -0.5, -0.5)
    adj_hp = propagate_adj(adj, "high", -0.3, -0.3)
    model = DSSGNNTFEPropFirst(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_layers=2,
        K_lp=3,
        K_hp=3,
        P=2,
        dropout=(0.0, 0.0),
    )
    x = torch.randn(n, input_dim)
    logits = model(adj_lp, adj_hp, x)
    assert logits.shape == (n, out_dim)


def test_dssgnn_tfe_propfirst_with_uncertainty():
    """DSSGNNTFEPropFirst returns uncertainty when requested."""
    n, input_dim, hidden_dim, out_dim = 12, 6, 12, 4
    adj = _make_adj(n)
    from tfe_utils import propagate_adj
    adj_lp = propagate_adj(adj, "low", -0.5, -0.5)
    adj_hp = propagate_adj(adj, "high", -0.3, -0.3)
    model = DSSGNNTFEPropFirst(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_layers=2,
        K_lp=2,
        K_hp=2,
        P=2,
        dropout=(0.0, 0.0),
    )
    x = torch.randn(n, input_dim)
    logits, uncertainty = model(adj_lp, adj_hp, x, return_uncertainty=True)
    assert logits.shape == (n, out_dim)
    assert uncertainty.shape == (n,)
    assert (uncertainty >= 0).all()


if __name__ == "__main__":
    test_dssgnn_forward_shape()
    test_dssgnn_forward_with_uncertainty()
    test_dssgnn_p0_no_uncertainty()
    test_dssgnn_gradient_flow()
    test_dssgnn_deterministic()
    test_dssgnn_tfe_forward()
    test_dssgnn_nonintrusive_input_lift()
    test_dssgnn_nonintrusive_random_gates()
    test_dssgnn_shared_input_lift_forward()
    test_gcn_dss_residual_encoder_warmup_behavior()
    test_dssgnn_reset_parameters_restores_spectral_and_gate_params()
    test_dssgnn_tfe_propfirst_forward()
    test_dssgnn_tfe_propfirst_with_uncertainty()
    print("All DSS-GNN tests passed.")
