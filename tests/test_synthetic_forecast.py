"""Tests for synthetic stochastic forecasting utilities."""

import numpy as np
import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dssgnn.synthetic_forecast import (
    NodeMLPForecaster,
    TemporalDSSForecaster,
    TemporalGraphDSSForecaster,
    TemporalMLPForecaster,
    TemporalCoupledSwingDSSForecaster,
    TemporalPhaseDSSForecaster,
    TemporalResidualDSSForecaster,
    TemporalSwingDSSForecaster,
    make_forecasting_windows,
    make_ieee118_style_grid,
    make_sensor_graph,
    simulate_stochastic_diffusion,
    simulate_stochastic_kuramoto,
    simulate_stochastic_swing,
    split_forecasting_windows,
)
from scripts.run_stochastic_field_forecast import apply_variance_calibrator, fit_variance_calibrator


def test_sensor_graph_and_diffusion_shapes():
    adj, coords = make_sensor_graph(num_nodes=12, k=3, seed=0)
    states = simulate_stochastic_diffusion(adj, coords, num_steps=40, seed=0)
    assert adj.shape == (12, 12)
    assert coords.shape == (12, 2)
    assert states.shape == (40, 12, 1)
    assert np.isfinite(states).all()


def test_kuramoto_windows_and_split_shapes():
    adj, _ = make_sensor_graph(num_nodes=10, k=3, seed=1)
    states = simulate_stochastic_kuramoto(adj, num_steps=50, seed=1)
    x, y = make_forecasting_windows(states, context=5, horizon=2)
    split = split_forecasting_windows(x, y, train_frac=0.5, val_frac=0.25)
    assert states.shape == (50, 10, 2)
    assert x.shape[1:] == (10, 10)
    assert y.shape[1:] == (10, 4)
    assert split.train_x.shape[0] > 0
    assert split.val_x.shape[0] > 0
    assert split.test_x.shape[0] > 0


def test_node_mlp_forecaster_shape():
    model = NodeMLPForecaster(input_dim=6, hidden_dim=8, out_dim=4)
    x = torch.randn(7, 6)
    y = model(x)
    assert y.shape == (7, 4)


def test_ieee118_style_grid_and_swing_shapes():
    adj, area_ids = make_ieee118_style_grid(seed=3)
    states = simulate_stochastic_swing(adj, area_ids, num_steps=32, seed=3)
    assert adj.shape == (118, 118)
    assert area_ids.shape == (118,)
    assert states.shape == (32, 118, 3)
    assert np.isfinite(states).all()


def test_temporal_mlp_forecaster_shape():
    model = TemporalMLPForecaster(
        input_state_dim=2,
        context=4,
        hidden_dim=10,
        out_dim=6,
        extra_static_dim=1,
    )
    x = torch.randn(5, 4 * 2 + 1)
    y = model(x)
    assert y.shape == (5, 6)


def test_temporal_dss_forecaster_field_stats_shape():
    from dssgnn.chebyshev import build_rescaled_laplacian

    adj, _ = make_sensor_graph(num_nodes=12, k=3, seed=0)
    L = build_rescaled_laplacian(adj)
    model = TemporalDSSForecaster(
        input_state_dim=2,
        context=3,
        temporal_hidden_dim=8,
        spatial_hidden_dim=10,
        out_dim=4,
        extra_static_dim=1,
        P=2,
        layers=2,
        K_lp=2,
        K_hp=2,
        S=4,
    )
    x = torch.randn(12, 3 * 2 + 1)
    stats = model.forward_with_field_stats(L, x)
    assert stats["mean"].shape == (12, 4)
    assert stats["predictive_var"].shape == (12, 4)


def test_temporal_graph_dss_forecaster_field_stats_shape():
    from dssgnn.chebyshev import build_rescaled_laplacian

    adj, _ = make_sensor_graph(num_nodes=10, k=3, seed=2)
    L = build_rescaled_laplacian(adj)
    model = TemporalGraphDSSForecaster(
        input_state_dim=3,
        context=2,
        temporal_hidden_dim=6,
        spatial_hidden_dim=8,
        out_dim=6,
        extra_static_dim=1,
        temporal_layers=2,
        recurrent_spatial_layers=1,
        P=2,
        layers=2,
        K_lp=2,
        K_hp=2,
        S=4,
    )
    x = torch.randn(10, 2 * 3 + 1)
    stats = model.forward_with_field_stats(L, x)
    assert stats["mean"].shape == (10, 6)
    assert stats["predictive_var"].shape == (10, 6)


def test_temporal_residual_dss_forecaster_field_stats_shape():
    from dssgnn.chebyshev import build_rescaled_laplacian

    adj, _ = make_sensor_graph(num_nodes=8, k=3, seed=4)
    L = build_rescaled_laplacian(adj)
    model = TemporalResidualDSSForecaster(
        input_state_dim=2,
        context=3,
        temporal_hidden_dim=7,
        spatial_hidden_dim=9,
        out_dim=4,
        extra_static_dim=1,
        P=2,
        layers=2,
        K_lp=2,
        K_hp=2,
        S=4,
    )
    x = torch.randn(8, 3 * 2 + 1)
    stats = model.forward_with_field_stats(L, x)
    assert stats["mean"].shape == (8, 4)
    assert stats["predictive_var"].shape == (8, 4)


def test_temporal_phase_dss_forecaster_field_stats_shape():
    from dssgnn.chebyshev import build_rescaled_laplacian

    adj, _ = make_sensor_graph(num_nodes=9, k=3, seed=5)
    L = build_rescaled_laplacian(adj)
    model = TemporalPhaseDSSForecaster(
        input_state_dim=2,
        context=4,
        temporal_hidden_dim=8,
        spatial_hidden_dim=10,
        horizon=3,
        extra_static_dim=1,
        P=2,
        layers=2,
        K_lp=2,
        K_hp=2,
        S=4,
    )
    x = torch.randn(9, 4 * 2 + 1)
    stats = model.forward_with_field_stats(L, x)
    assert stats["mean"].shape == (9, 6)
    assert stats["predictive_var"].shape == (9, 6)


def test_temporal_swing_dss_forecaster_field_stats_shape():
    from dssgnn.chebyshev import build_rescaled_laplacian

    adj, _ = make_sensor_graph(num_nodes=9, k=3, seed=6)
    L = build_rescaled_laplacian(adj)
    model = TemporalSwingDSSForecaster(
        input_state_dim=3,
        context=4,
        temporal_hidden_dim=8,
        spatial_hidden_dim=10,
        horizon=3,
        extra_static_dim=1,
        P=2,
        layers=2,
        K_lp=2,
        K_hp=2,
        S=4,
    )
    x = torch.randn(9, 4 * 3 + 1)
    stats = model.forward_with_field_stats(L, x)
    assert stats["mean"].shape == (9, 9)
    assert stats["predictive_var"].shape == (9, 9)


def test_temporal_coupled_swing_dss_forecaster_field_stats_shape():
    from dssgnn.chebyshev import build_rescaled_laplacian

    adj, _ = make_sensor_graph(num_nodes=9, k=3, seed=7)
    L = build_rescaled_laplacian(adj)
    model = TemporalCoupledSwingDSSForecaster(
        input_state_dim=3,
        context=4,
        temporal_hidden_dim=8,
        spatial_hidden_dim=10,
        horizon=3,
        extra_static_dim=1,
        P=2,
        layers=2,
        K_lp=2,
        K_hp=2,
        S=4,
    )
    x = torch.randn(9, 4 * 3 + 1)
    stats = model.forward_with_field_stats(L, x)
    assert stats["mean"].shape == (9, 9)
    assert stats["predictive_var"].shape == (9, 9)


def test_temporal_phase_dss_masked_nodes_use_graph_context():
    from dssgnn.chebyshev import build_rescaled_laplacian

    torch.manual_seed(0)
    adj = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    L = build_rescaled_laplacian(adj)
    model = TemporalPhaseDSSForecaster(
        input_state_dim=2,
        context=3,
        temporal_hidden_dim=8,
        spatial_hidden_dim=10,
        horizon=2,
        extra_static_dim=1,
        P=2,
        layers=2,
        K_lp=2,
        K_hp=2,
        S=4,
    )

    x_a = torch.zeros(3, 3 * 2 + 1)
    x_b = x_a.clone()

    # Node 1 is masked; only its neighbors change.
    x_a[1, -1] = 0.0
    x_b[1, -1] = 0.0
    x_a[0, :6] = torch.tensor([0.2, 0.9, 0.4, 0.8, 0.6, 0.7])
    x_a[2, :6] = torch.tensor([-0.3, 0.8, -0.1, 0.9, 0.1, 1.0])
    x_b[0, :6] = torch.tensor([0.9, 0.1, 0.8, -0.1, 0.7, -0.3])
    x_b[2, :6] = torch.tensor([-0.8, 0.1, -0.7, -0.2, -0.6, -0.4])
    x_a[[0, 2], -1] = 1.0
    x_b[[0, 2], -1] = 1.0

    out_a = model.forward_with_field_stats(L, x_a)["mean"][1]
    out_b = model.forward_with_field_stats(L, x_b)["mean"][1]
    assert not torch.allclose(out_a, out_b)


def test_grouped_scale_calibrator_inflates_masked_variance():
    mean = torch.zeros(2, 2, 1)
    target = torch.tensor([[[0.1], [1.0]], [[0.1], [1.2]]], dtype=torch.float32)
    variance = torch.full_like(target, 0.1)
    group_masks = {
        "observed": torch.tensor([[[True], [False]], [[True], [False]]]),
        "masked": torch.tensor([[[False], [True]], [[False], [True]]]),
    }

    calibrator = fit_variance_calibrator(mean, target, variance, var_floor=1e-4, group_masks=group_masks)
    calibrated = apply_variance_calibrator(mean.shape, variance, calibrator, var_floor=1e-4, group_masks=group_masks)

    assert calibrator["mode"] == "group_scale"
    assert calibrated[:, 1, :].mean() > calibrated[:, 0, :].mean()


def test_grouped_constant_calibrator_handles_missing_variance():
    mean = torch.zeros(1, 2, 1)
    target = torch.tensor([[[0.1], [0.8]]], dtype=torch.float32)
    group_masks = {
        "observed": torch.tensor([[[True], [False]]]),
        "masked": torch.tensor([[[False], [True]]]),
    }

    calibrator = fit_variance_calibrator(mean, target, None, var_floor=1e-4, group_masks=group_masks)
    calibrated = apply_variance_calibrator(mean.shape, None, calibrator, var_floor=1e-4, group_masks=group_masks)

    assert calibrator["mode"] == "group_constant"
    assert float(calibrated[0, 1, 0]) > float(calibrated[0, 0, 0])
