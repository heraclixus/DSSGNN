"""Run locally runnable synthetic stochastic graph-field forecasting experiments."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CACHE_ROOT = Path("results_local") / ".cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "mpl"))

from dssgnn.chebyshev import build_rescaled_laplacian
from dssgnn.model import DSSGNN
from dssgnn.synthetic_forecast import (
    ForecastSplit,
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

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - plotting is optional
    plt = None


SYSTEMS = ("diffusion", "kuramoto", "swing118")
MODELS = (
    "mlp",
    "graph_p0",
    "dss",
    "gru_mlp",
    "gru_graph_p0",
    "gru_dss",
    "graphgru_graph_p0",
    "graphgru_dss",
    "gru_dssres",
    "gru_phase_dss",
    "gru_swing_dss",
    "gru_swing_coupled_dss",
)
Z90 = 1.6448536269514722


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def simulate_system(system: str, args, seed: int):
    if system == "diffusion":
        adj, coords = make_sensor_graph(
            num_nodes=args.num_nodes,
            k=args.graph_k,
            length_scale=args.graph_length_scale,
            seed=seed,
        )
        states = simulate_stochastic_diffusion(
            adj,
            coords,
            num_steps=args.num_steps,
            dt=args.dt_diffusion,
            diffusion=args.diffusion_rate,
            damping=args.damping,
            forcing_scale=args.forcing_scale,
            noise_std=args.noise_std_diffusion,
            seed=seed,
        )
        plot_meta = {"coords": coords}
    elif system == "kuramoto":
        adj, coords = make_sensor_graph(
            num_nodes=args.num_nodes,
            k=args.graph_k,
            length_scale=args.graph_length_scale,
            seed=seed,
        )
        states = simulate_stochastic_kuramoto(
            adj,
            num_steps=args.num_steps,
            dt=args.dt_kuramoto,
            coupling=args.kuramoto_coupling,
            noise_std=args.noise_std_kuramoto,
            seed=seed,
        )
        plot_meta = {"coords": coords}
    elif system == "swing118":
        adj, area_ids = make_ieee118_style_grid(seed=seed)
        states = simulate_stochastic_swing(
            adj,
            area_ids,
            num_steps=args.num_steps_swing,
            dt=args.dt_swing,
            coupling=args.swing_coupling,
            damping=args.swing_damping,
            inertia=args.swing_inertia,
            forcing_scale=args.swing_forcing_scale,
            noise_std=args.noise_std_swing,
            seed=seed,
        )
        plot_meta = {"area_ids": area_ids}
    else:
        raise ValueError(f"Unknown system: {system}")
    return adj, states, plot_meta


def build_dataset(system: str, args, seed: int):
    if system == "swing118" and args.num_rollouts > 1:
        if args.num_rollouts < 3:
            raise ValueError("num_rollouts must be at least 3 for rollout-based train/val/test splits.")
        adj, area_ids = make_ieee118_style_grid(seed=args.topology_seed)
        plot_meta = {"area_ids": area_ids}
        rollout_x, rollout_y = [], []
        for rollout_idx in range(args.num_rollouts):
            states = simulate_stochastic_swing(
                adj,
                area_ids,
                num_steps=args.num_steps_swing,
                dt=args.dt_swing,
                coupling=args.swing_coupling,
                damping=args.swing_damping,
                inertia=args.swing_inertia,
                forcing_scale=args.swing_forcing_scale,
                noise_std=args.noise_std_swing,
                seed=seed + rollout_idx,
            )
            x_roll, y_roll = make_forecasting_windows(states, context=args.context, horizon=args.horizon)
            rollout_x.append(x_roll)
            rollout_y.append(y_roll)

        num_rollouts = len(rollout_x)
        train_end = max(1, int(num_rollouts * args.train_frac))
        val_end = max(train_end + 1, int(num_rollouts * (args.train_frac + args.val_frac)))
        val_end = min(val_end, num_rollouts - 1) if num_rollouts > 2 else val_end
        split = ForecastSplit(
            train_x=np.concatenate(rollout_x[:train_end], axis=0),
            train_y=np.concatenate(rollout_y[:train_end], axis=0),
            val_x=np.concatenate(rollout_x[train_end:val_end], axis=0),
            val_y=np.concatenate(rollout_y[train_end:val_end], axis=0),
            test_x=np.concatenate(rollout_x[val_end:], axis=0),
            test_y=np.concatenate(rollout_y[val_end:], axis=0),
        )
    else:
        adj, states, plot_meta = simulate_system(system, args, seed)
        x, y = make_forecasting_windows(states, context=args.context, horizon=args.horizon)
        split = split_forecasting_windows(x, y, train_frac=args.train_frac, val_frac=args.val_frac)
    masked_nodes = None
    if args.masked_sensor_frac > 0.0:
        rng = np.random.default_rng(seed + 137)
        num_nodes = split.train_x.shape[1]
        num_mask = max(1, int(round(num_nodes * args.masked_sensor_frac)))
        if system == "swing118":
            # Prefer masking part of the disturbed subsystem to mimic sparse instrumentation
            # during a localized stochastic event.
            disturbed_start = 24 + 22  # third area in make_ieee118_style_grid
            disturbed_end = disturbed_start + 25
            low = disturbed_start
            high = max(low + 1, disturbed_end - num_mask + 1)
            block_start = int(rng.integers(low, high))
            masked_nodes = np.arange(block_start, min(num_nodes, block_start + num_mask), dtype=np.int64)
        else:
            masked_nodes = np.sort(rng.choice(num_nodes, size=num_mask, replace=False)).astype(np.int64)

        def _apply_mask(arr):
            masked = arr.copy()
            masked[:, masked_nodes, :] = 0.0
            if args.append_obs_flag:
                flag = np.ones((masked.shape[0], masked.shape[1], 1), dtype=np.float32)
                flag[:, masked_nodes, 0] = 0.0
                masked = np.concatenate([masked, flag], axis=-1)
            return masked

        split = split.__class__(
            train_x=_apply_mask(split.train_x),
            train_y=split.train_y,
            val_x=_apply_mask(split.val_x),
            val_y=split.val_y,
            test_x=_apply_mask(split.test_x),
            test_y=split.test_y,
        )
    elif args.append_obs_flag:
        def _append_flag(arr):
            flag = np.ones((arr.shape[0], arr.shape[1], 1), dtype=np.float32)
            return np.concatenate([arr, flag], axis=-1)

        split = split.__class__(
            train_x=_append_flag(split.train_x),
            train_y=split.train_y,
            val_x=_append_flag(split.val_x),
            val_y=split.val_y,
            test_x=_append_flag(split.test_x),
            test_y=split.test_y,
        )
    return adj, split, masked_nodes, plot_meta


def maybe_limit_windows(split: ForecastSplit, max_train: int | None, max_val: int | None, max_test: int | None):
    def _slice(arr: np.ndarray, limit: int | None):
        if limit is None or limit <= 0 or arr.shape[0] <= limit:
            return arr
        return arr[:limit]

    return ForecastSplit(
        train_x=_slice(split.train_x, max_train),
        train_y=_slice(split.train_y, max_train),
        val_x=_slice(split.val_x, max_val),
        val_y=_slice(split.val_y, max_val),
        test_x=_slice(split.test_x, max_test),
        test_y=_slice(split.test_y, max_test),
    )


def target_weight_map(x: torch.Tensor, out_dim: int, masked_loss_weight: float, append_obs_flag: bool) -> torch.Tensor | None:
    if not append_obs_flag or masked_loss_weight <= 1.0:
        return None
    obs_flag = x[:, -1].clamp(0.0, 1.0)
    node_weights = 1.0 + (1.0 - obs_flag) * (masked_loss_weight - 1.0)
    return node_weights.unsqueeze(-1).expand(-1, out_dim)


def node_weight_vector(x: torch.Tensor, masked_loss_weight: float, append_obs_flag: bool) -> torch.Tensor | None:
    if not append_obs_flag or masked_loss_weight <= 1.0:
        return None
    obs_flag = x[:, -1].clamp(0.0, 1.0)
    return 1.0 + (1.0 - obs_flag) * (masked_loss_weight - 1.0)


def weighted_mean(values: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    if weights is None:
        return values.mean()
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def wrapped_angle_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))


def kuramoto_phase_loss_terms(
    mean: torch.Tensor,
    target: torch.Tensor,
    x: torch.Tensor,
    context: int,
    masked_loss_weight: float,
    append_obs_flag: bool,
):
    state_dim = 2
    horizon = mean.shape[-1] // state_dim
    pred_steps = mean.reshape(mean.shape[0], horizon, state_dim)
    target_steps = target.reshape(target.shape[0], horizon, state_dim)
    pred_phase = torch.atan2(pred_steps[..., 0], pred_steps[..., 1])
    target_phase = torch.atan2(target_steps[..., 0], target_steps[..., 1])
    node_weights = node_weight_vector(x, masked_loss_weight, append_obs_flag)
    phase_weights = node_weights.unsqueeze(-1).expand_as(pred_phase) if node_weights is not None else None

    phase_err = wrapped_angle_diff(pred_phase, target_phase)
    phase_loss = weighted_mean(phase_err.pow(2), phase_weights)

    if horizon > 1:
        pred_delta = wrapped_angle_diff(pred_phase[:, 1:], pred_phase[:, :-1])
        target_delta = wrapped_angle_diff(target_phase[:, 1:], target_phase[:, :-1])
        delta_weights = node_weights.unsqueeze(-1).expand_as(pred_delta) if node_weights is not None else None
        delta_loss = weighted_mean(wrapped_angle_diff(pred_delta, target_delta).pow(2), delta_weights)
    else:
        delta_loss = phase_loss.new_zeros(())
    return phase_loss, delta_loss


def kuramoto_selection_score(
    mean: torch.Tensor,
    target: torch.Tensor,
    xs: np.ndarray,
    args,
):
    rmse = torch.sqrt(F.mse_loss(mean, target))
    if mean.shape[-1] % 2 != 0:
        return float(rmse.item())

    x_tensor = torch.from_numpy(xs)
    phase_losses = []
    delta_losses = []
    for sample_idx in range(mean.shape[0]):
        phase_loss, delta_loss = kuramoto_phase_loss_terms(
            mean[sample_idx],
            target[sample_idx],
            x_tensor[sample_idx],
            args.context,
            args.masked_loss_weight,
            args.append_obs_flag,
        )
        phase_losses.append(phase_loss)
        delta_losses.append(delta_loss)
    phase_term = torch.stack(phase_losses).mean() if phase_losses else rmse.new_zeros(())
    delta_term = torch.stack(delta_losses).mean() if delta_losses else rmse.new_zeros(())
    score = rmse + args.val_phase_weight * phase_term.sqrt() + args.val_delta_weight * delta_term.sqrt()
    return float(score.item())


def swing_dynamics_loss_terms(
    mean: torch.Tensor,
    target: torch.Tensor,
    x: torch.Tensor,
    masked_loss_weight: float,
    append_obs_flag: bool,
):
    state_dim = 3
    horizon = mean.shape[-1] // state_dim
    pred_steps = mean.reshape(mean.shape[0], horizon, state_dim)
    target_steps = target.reshape(target.shape[0], horizon, state_dim)
    pred_phase = torch.atan2(pred_steps[..., 0], pred_steps[..., 1])
    target_phase = torch.atan2(target_steps[..., 0], target_steps[..., 1])
    pred_omega = pred_steps[..., 2]
    target_omega = target_steps[..., 2]
    node_weights = node_weight_vector(x, masked_loss_weight, append_obs_flag)
    phase_weights = node_weights.unsqueeze(-1).expand_as(pred_phase) if node_weights is not None else None
    omega_weights = node_weights.unsqueeze(-1).expand_as(pred_omega) if node_weights is not None else None

    phase_loss = weighted_mean(wrapped_angle_diff(pred_phase, target_phase).pow(2), phase_weights)
    omega_loss = weighted_mean((pred_omega - target_omega).pow(2), omega_weights)
    if horizon > 1:
        pred_domega = pred_omega[:, 1:] - pred_omega[:, :-1]
        target_domega = target_omega[:, 1:] - target_omega[:, :-1]
        domega_weights = node_weights.unsqueeze(-1).expand_as(pred_domega) if node_weights is not None else None
        domega_loss = weighted_mean((pred_domega - target_domega).pow(2), domega_weights)
    else:
        domega_loss = phase_loss.new_zeros(())
    return phase_loss, omega_loss, domega_loss


def swing_selection_score(
    mean: torch.Tensor,
    target: torch.Tensor,
    xs: np.ndarray,
    args,
):
    rmse = torch.sqrt(F.mse_loss(mean, target))
    x_tensor = torch.from_numpy(xs)
    phase_losses = []
    omega_losses = []
    domega_losses = []
    for sample_idx in range(mean.shape[0]):
        phase_loss, omega_loss, domega_loss = swing_dynamics_loss_terms(
            mean[sample_idx],
            target[sample_idx],
            x_tensor[sample_idx],
            args.masked_loss_weight,
            args.append_obs_flag,
        )
        phase_losses.append(phase_loss)
        omega_losses.append(omega_loss)
        domega_losses.append(domega_loss)
    phase_term = torch.stack(phase_losses).mean() if phase_losses else rmse.new_zeros(())
    omega_term = torch.stack(omega_losses).mean() if omega_losses else rmse.new_zeros(())
    domega_term = torch.stack(domega_losses).mean() if domega_losses else rmse.new_zeros(())
    score = (
        rmse
        + args.val_swing_phase_weight * phase_term.sqrt()
        + args.val_swing_omega_weight * omega_term.sqrt()
        + args.val_swing_domega_weight * domega_term.sqrt()
    )
    return float(score.item())


def observation_group_masks(xs: np.ndarray, out_dim: int, append_obs_flag: bool):
    if not append_obs_flag:
        return None
    obs_flag = torch.from_numpy(xs[..., -1]).float()
    observed = obs_flag > 0.5
    masked = ~observed
    group_masks = {}
    if bool(observed.any()):
        group_masks["observed"] = observed.unsqueeze(-1).expand(-1, -1, out_dim)
    if bool(masked.any()):
        group_masks["masked"] = masked.unsqueeze(-1).expand(-1, -1, out_dim)
    return group_masks or None


def build_model(model_name: str, input_dim: int, out_dim: int, args):
    state_dim = out_dim // args.horizon
    extra_static_dim = input_dim - args.context * state_dim
    if out_dim % args.horizon != 0 or extra_static_dim < 0:
        raise ValueError("Invalid forecasting dimensions for temporal encoder construction.")

    if model_name == "mlp":
        return NodeMLPForecaster(input_dim=input_dim, hidden_dim=args.hidden_dim, out_dim=out_dim)
    if model_name == "gru_mlp":
        return TemporalMLPForecaster(
            input_state_dim=state_dim,
            context=args.context,
            hidden_dim=args.temporal_hidden_dim,
            out_dim=out_dim,
            extra_static_dim=extra_static_dim,
            temporal_layers=args.temporal_layers,
            temporal_dropout=args.temporal_dropout,
        )
    if model_name == "graph_p0":
        return DSSGNN(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            out_dim=out_dim,
            num_layers=args.layers,
            K_lp=args.K_lp,
            K_hp=args.K_hp,
            P=0,
            dropout=(args.dropout, args.lin_dropout),
            activation=True,
            shared_input_lift=False,
        )
    if model_name == "gru_graph_p0":
        return TemporalDSSForecaster(
            input_state_dim=state_dim,
            context=args.context,
            temporal_hidden_dim=args.temporal_hidden_dim,
            spatial_hidden_dim=args.hidden_dim,
            out_dim=out_dim,
            extra_static_dim=extra_static_dim,
            temporal_layers=args.temporal_layers,
            temporal_dropout=args.temporal_dropout,
            K_lp=args.K_lp,
            K_hp=args.K_hp,
            P=0,
            layers=args.layers,
            dropout=(args.dropout, args.lin_dropout),
            shared_input_lift=False,
        )
    if model_name == "dss":
        return DSSGNN(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            out_dim=out_dim,
            num_layers=args.layers,
            K_lp=args.K_lp,
            K_hp=args.K_hp,
            P=args.P,
            dropout=(args.dropout, args.lin_dropout),
            activation=True,
            use_random_gates=args.use_random_gates,
            P_gate=args.P_gate,
            S=args.S,
            shared_input_lift=args.shared_input_lift,
        )
    if model_name == "gru_dss":
        return TemporalDSSForecaster(
            input_state_dim=state_dim,
            context=args.context,
            temporal_hidden_dim=args.temporal_hidden_dim,
            spatial_hidden_dim=args.hidden_dim,
            out_dim=out_dim,
            extra_static_dim=extra_static_dim,
            temporal_layers=args.temporal_layers,
            temporal_dropout=args.temporal_dropout,
            K_lp=args.K_lp,
            K_hp=args.K_hp,
            P=args.P,
            layers=args.layers,
            dropout=(args.dropout, args.lin_dropout),
            use_random_gates=args.use_random_gates,
            P_gate=args.P_gate,
            S=args.S,
            shared_input_lift=args.shared_input_lift,
        )
    if model_name == "graphgru_graph_p0":
        return TemporalGraphDSSForecaster(
            input_state_dim=state_dim,
            context=args.context,
            temporal_hidden_dim=args.temporal_hidden_dim,
            spatial_hidden_dim=args.hidden_dim,
            out_dim=out_dim,
            extra_static_dim=extra_static_dim,
            temporal_layers=args.temporal_layers,
            temporal_dropout=args.temporal_dropout,
            recurrent_spatial_layers=args.recurrent_spatial_layers,
            K_lp=args.K_lp,
            K_hp=args.K_hp,
            P=0,
            layers=args.layers,
            dropout=(args.dropout, args.lin_dropout),
            shared_input_lift=False,
        )
    if model_name == "graphgru_dss":
        return TemporalGraphDSSForecaster(
            input_state_dim=state_dim,
            context=args.context,
            temporal_hidden_dim=args.temporal_hidden_dim,
            spatial_hidden_dim=args.hidden_dim,
            out_dim=out_dim,
            extra_static_dim=extra_static_dim,
            temporal_layers=args.temporal_layers,
            temporal_dropout=args.temporal_dropout,
            recurrent_spatial_layers=args.recurrent_spatial_layers,
            K_lp=args.K_lp,
            K_hp=args.K_hp,
            P=args.P,
            layers=args.layers,
            dropout=(args.dropout, args.lin_dropout),
            use_random_gates=args.use_random_gates,
            P_gate=args.P_gate,
            S=args.S,
            shared_input_lift=args.shared_input_lift,
        )
    if model_name == "gru_dssres":
        return TemporalResidualDSSForecaster(
            input_state_dim=state_dim,
            context=args.context,
            temporal_hidden_dim=args.temporal_hidden_dim,
            spatial_hidden_dim=args.hidden_dim,
            out_dim=out_dim,
            extra_static_dim=extra_static_dim,
            temporal_layers=args.temporal_layers,
            temporal_dropout=args.temporal_dropout,
            K_lp=args.K_lp,
            K_hp=args.K_hp,
            P=args.P,
            layers=args.layers,
            dropout=(args.dropout, args.lin_dropout),
            use_random_gates=args.use_random_gates,
            P_gate=args.P_gate,
            S=args.S,
            shared_input_lift=args.shared_input_lift,
        )
    if model_name == "gru_phase_dss":
        if state_dim != 2:
            raise ValueError("gru_phase_dss is only supported for 2D Kuramoto-style sin/cos states.")
        return TemporalPhaseDSSForecaster(
            input_state_dim=state_dim,
            context=args.context,
            temporal_hidden_dim=args.temporal_hidden_dim,
            spatial_hidden_dim=args.hidden_dim,
            horizon=args.horizon,
            extra_static_dim=extra_static_dim,
            temporal_layers=args.temporal_layers,
            temporal_dropout=args.temporal_dropout,
            K_lp=args.K_lp,
            K_hp=args.K_hp,
            P=args.P,
            layers=args.layers,
            dropout=(args.dropout, args.lin_dropout),
            use_random_gates=args.use_random_gates,
            P_gate=args.P_gate,
            S=args.S,
            shared_input_lift=args.shared_input_lift,
        )
    if model_name == "gru_swing_dss":
        if state_dim != 3:
            raise ValueError("gru_swing_dss is only supported for 3D swing-style sin/cos/omega states.")
        return TemporalSwingDSSForecaster(
            input_state_dim=state_dim,
            context=args.context,
            temporal_hidden_dim=args.temporal_hidden_dim,
            spatial_hidden_dim=args.hidden_dim,
            horizon=args.horizon,
            extra_static_dim=extra_static_dim,
            temporal_layers=args.temporal_layers,
            temporal_dropout=args.temporal_dropout,
            K_lp=args.K_lp,
            K_hp=args.K_hp,
            P=args.P,
            layers=args.layers,
            dropout=(args.dropout, args.lin_dropout),
            use_random_gates=args.use_random_gates,
            P_gate=args.P_gate,
            S=args.S,
            shared_input_lift=args.shared_input_lift,
            time_step=args.dt_swing,
        )
    if model_name == "gru_swing_coupled_dss":
        if state_dim != 3:
            raise ValueError("gru_swing_coupled_dss is only supported for 3D swing-style sin/cos/omega states.")
        return TemporalCoupledSwingDSSForecaster(
            input_state_dim=state_dim,
            context=args.context,
            temporal_hidden_dim=args.temporal_hidden_dim,
            spatial_hidden_dim=args.hidden_dim,
            horizon=args.horizon,
            extra_static_dim=extra_static_dim,
            temporal_layers=args.temporal_layers,
            temporal_dropout=args.temporal_dropout,
            K_lp=args.K_lp,
            K_hp=args.K_hp,
            P=args.P,
            layers=args.layers,
            dropout=(args.dropout, args.lin_dropout),
            use_random_gates=args.use_random_gates,
            P_gate=args.P_gate,
            S=args.S,
            shared_input_lift=args.shared_input_lift,
            time_step=args.dt_swing,
        )
    raise ValueError(f"Unknown model: {model_name}")


def predict_once(model_name: str, model, graph_input, x: torch.Tensor, var_floor: float):
    if model_name in {"mlp", "gru_mlp"}:
        mean = model(x)
        return mean, None
    if model_name in {"graph_p0", "gru_graph_p0", "graphgru_graph_p0"}:
        mean = model(graph_input, x)
        return mean, None
    stats = model.forward_with_field_stats(graph_input, x, eps=var_floor)
    return stats["mean"], stats["predictive_var"].clamp_min(var_floor)


def collect_predictions(model_name: str, model, graph_input, xs: np.ndarray, device, var_floor: float):
    means, variances = [], []
    model.eval()
    with torch.no_grad():
        for sample in xs:
            x = torch.from_numpy(sample).to(device)
            mean, var = predict_once(model_name, model, graph_input, x, var_floor)
            means.append(mean.cpu())
            if var is not None:
                variances.append(var.cpu())
    mean_tensor = torch.stack(means, dim=0)
    if variances:
        return mean_tensor, torch.stack(variances, dim=0)
    return mean_tensor, None


def train_epoch(model_name: str, model, graph_input, train_x: np.ndarray, train_y: np.ndarray, optimizer, device, args):
    model.train()
    order = np.random.permutation(train_x.shape[0])
    total_loss = 0.0
    for idx in order:
        x = torch.from_numpy(train_x[idx]).to(device)
        y = torch.from_numpy(train_y[idx]).to(device)
        optimizer.zero_grad()
        mean, var = predict_once(model_name, model, graph_input, x, args.var_floor)
        weights = target_weight_map(x, y.shape[-1], args.masked_loss_weight, args.append_obs_flag)
        if var is None:
            loss = weighted_mean((mean - y).pow(2), weights)
        else:
            nll = 0.5 * (torch.log(var) + (y - mean).pow(2) / var)
            mse = weighted_mean((mean - y).pow(2), weights)
            loss = mse + args.nll_weight * weighted_mean(nll, weights) + args.var_reg * weighted_mean(var, weights)
        if model_name == "gru_phase_dss" and (args.phase_loss_weight > 0.0 or args.delta_loss_weight > 0.0):
            phase_loss, delta_loss = kuramoto_phase_loss_terms(
                mean,
                y,
                x,
                args.context,
                args.masked_loss_weight,
                args.append_obs_flag,
            )
            loss = loss + args.phase_loss_weight * phase_loss + args.delta_loss_weight * delta_loss
        if model_name in {"gru_swing_dss", "gru_swing_coupled_dss"} and (
            args.swing_phase_loss_weight > 0.0
            or args.swing_omega_loss_weight > 0.0
            or args.swing_domega_loss_weight > 0.0
        ):
            phase_loss, omega_loss, domega_loss = swing_dynamics_loss_terms(
                mean,
                y,
                x,
                args.masked_loss_weight,
                args.append_obs_flag,
            )
            loss = (
                loss
                + args.swing_phase_loss_weight * phase_loss
                + args.swing_omega_loss_weight * omega_loss
                + args.swing_domega_loss_weight * domega_loss
            )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(train_x.shape[0], 1)


def evaluate_rmse(model_name: str, model, graph_input, xs: np.ndarray, ys: np.ndarray, device, var_floor: float):
    mean, _ = collect_predictions(model_name, model, graph_input, xs, device, var_floor)
    target = torch.from_numpy(ys)
    return torch.sqrt(F.mse_loss(mean, target)).item()


def evaluate_selection_metric(model_name: str, model, graph_input, xs: np.ndarray, ys: np.ndarray, device, var_floor: float, args):
    mean, _ = collect_predictions(model_name, model, graph_input, xs, device, var_floor)
    target = torch.from_numpy(ys)
    if model_name == "gru_phase_dss":
        return kuramoto_selection_score(mean, target, xs, args)
    if model_name in {"gru_swing_dss", "gru_swing_coupled_dss"}:
        return swing_selection_score(mean, target, xs, args)
    return torch.sqrt(F.mse_loss(mean, target)).item()


def fit_variance_calibrator(
    mean: torch.Tensor,
    target: torch.Tensor,
    variance: torch.Tensor | None,
    var_floor: float,
    group_masks: dict[str, torch.Tensor] | None = None,
):
    sq_error = (target - mean).pow(2)
    if group_masks:
        fallback = fit_variance_calibrator(mean, target, variance, var_floor, group_masks=None)
        groups = {}
        raw = variance.clamp_min(var_floor) if variance is not None else None
        for name, mask in group_masks.items():
            if not bool(mask.any()):
                continue
            group_sq = sq_error[mask]
            if variance is None:
                groups[name] = float(group_sq.mean().clamp_min(var_floor).item())
            else:
                group_raw = raw[mask]
                scale = float((group_sq.mean() / group_raw.mean().clamp_min(var_floor)).item())
                groups[name] = max(scale, var_floor)
        return {
            "mode": "group_constant" if variance is None else "group_scale",
            "groups": groups,
            "fallback": fallback,
        }
    if variance is None:
        return {"mode": "constant", "value": float(sq_error.mean().clamp_min(var_floor).item())}
    raw = variance.clamp_min(var_floor)
    scale = float((sq_error.mean() / raw.mean().clamp_min(var_floor)).item())
    return {"mode": "scale", "value": max(scale, var_floor)}


def apply_variance_calibrator(
    shape,
    variance: torch.Tensor | None,
    calibrator,
    var_floor: float,
    group_masks: dict[str, torch.Tensor] | None = None,
):
    if calibrator["mode"] in {"group_constant", "group_scale"}:
        fallback = apply_variance_calibrator(shape, variance, calibrator["fallback"], var_floor, group_masks=None)
        if not group_masks:
            return fallback
        output = fallback.clone()
        if calibrator["mode"] == "group_constant":
            for name, value in calibrator["groups"].items():
                mask = group_masks.get(name)
                if mask is not None:
                    output[mask] = value
        else:
            if variance is None:
                raise ValueError("Group scale calibrator requires predictive variance.")
            raw = variance.clamp_min(var_floor)
            for name, value in calibrator["groups"].items():
                mask = group_masks.get(name)
                if mask is not None:
                    output[mask] = (raw[mask] * value).clamp_min(var_floor)
        return output
    if calibrator["mode"] == "constant":
        return torch.full(shape, calibrator["value"], dtype=torch.float32)
    if variance is None:
        raise ValueError("Scaled calibrator requires predictive variance.")
    return (variance * calibrator["value"]).clamp_min(var_floor)


def regression_metrics(mean: torch.Tensor, target: torch.Tensor, variance: torch.Tensor, node_idx=None):
    if node_idx is not None:
        mean = mean[:, node_idx, :]
        target = target[:, node_idx, :]
        variance = variance[:, node_idx, :]
    err = mean - target
    mse = err.pow(2).mean()
    rmse = torch.sqrt(mse)
    mae = err.abs().mean()
    variance = variance.clamp_min(1e-8)
    nll = 0.5 * (math.log(2.0 * math.pi) + torch.log(variance) + err.pow(2) / variance).mean()
    std = variance.sqrt()
    lower = mean - Z90 * std
    upper = mean + Z90 * std
    coverage90 = ((target >= lower) & (target <= upper)).float().mean()
    width90 = (upper - lower).mean()

    pred_std = std.flatten()
    abs_err = err.abs().flatten()
    if pred_std.numel() > 1 and pred_std.std() > 0 and abs_err.std() > 0:
        corr = torch.corrcoef(torch.stack([pred_std, abs_err]))[0, 1]
        corr_value = float(corr.item())
    else:
        corr_value = 0.0
    return {
        "rmse": float(rmse.item()),
        "mae": float(mae.item()),
        "nll": float(nll.item()),
        "coverage90": float(coverage90.item()),
        "width90": float(width90.item()),
        "std_error_corr": corr_value,
    }


def infer_state_dim(system: str) -> int:
    if system == "diffusion":
        return 1
    if system == "kuramoto":
        return 2
    if system == "swing118":
        return 3
    return 1


def swing_layout(area_ids: np.ndarray) -> np.ndarray:
    coords = np.zeros((len(area_ids), 2), dtype=np.float32)
    unique_areas = np.unique(area_ids)
    for row_idx, area in enumerate(unique_areas):
        nodes = np.where(area_ids == area)[0]
        x = np.linspace(0.0, 1.0, len(nodes), dtype=np.float32)
        y = np.full(len(nodes), -float(row_idx), dtype=np.float32)
        coords[nodes, 0] = x
        coords[nodes, 1] = y
    return coords


def choose_model_pair(predictions: dict[str, tuple[torch.Tensor, torch.Tensor]]):
    preferred_pairs = [
        ("gru_mlp", "gru_phase_dss"),
        ("gru_mlp", "gru_dssres"),
        ("gru_mlp", "gru_dss"),
        ("mlp", "dss"),
    ]
    for baseline, graph_model in preferred_pairs:
        if baseline in predictions and graph_model in predictions:
            return baseline, graph_model
    model_names = list(predictions.keys())
    if len(model_names) >= 2:
        return model_names[0], model_names[-1]
    if model_names:
        return model_names[0], model_names[0]
    raise ValueError("predictions must contain at least one model.")


def per_node_rmse(mean: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    err = mean - target
    return err.pow(2).mean(dim=(0, 2)).sqrt()


def choose_focus_node_and_channel(
    system: str,
    target: torch.Tensor,
    baseline_mean: torch.Tensor,
    graph_mean: torch.Tensor | None,
    masked_nodes: np.ndarray | None,
):
    cases = rank_focus_cases(system, target, baseline_mean, graph_mean, masked_nodes)
    if not cases:
        return {"kind": "raw", "node_idx": 0, "channel_idx": 0, "horizon_idx": 0, "feat_idx": 0}
    return cases[0]


def select_diverse_cases(
    cases: list[dict[str, Any]],
    max_cases: int,
    prefer_unique_nodes: bool = True,
) -> list[dict[str, Any]]:
    if max_cases <= 0 or not cases:
        return []
    if not prefer_unique_nodes:
        return cases[:max_cases]

    selected = []
    used_nodes = set()
    for case in cases:
        node_idx = case["node_idx"]
        if node_idx in used_nodes:
            continue
        selected.append(case)
        used_nodes.add(node_idx)
        if len(selected) >= max_cases:
            return selected

    for case in cases:
        if len(selected) >= max_cases:
            break
        if case not in selected:
            selected.append(case)
    return selected


def rank_focus_cases(
    system: str,
    target: torch.Tensor,
    baseline_mean: torch.Tensor,
    graph_mean: torch.Tensor | None,
    masked_nodes: np.ndarray | None,
):
    candidate_nodes = masked_nodes if masked_nodes is not None and len(masked_nodes) > 0 else np.arange(target.shape[1])
    state_dim = infer_state_dim(system)
    cases = []

    if system == "kuramoto":
        num_horizons = target.shape[-1] // state_dim
        for candidate_node in candidate_nodes:
            for horizon_candidate in range(num_horizons):
                target_series = extract_phase_series(target, int(candidate_node), horizon_candidate, state_dim)
                baseline_series = extract_phase_series(baseline_mean, int(candidate_node), horizon_candidate, state_dim)
                graph_series = extract_phase_series(graph_mean, int(candidate_node), horizon_candidate, state_dim) if graph_mean is not None else baseline_series
                target_std = float(np.std(target_series))
                graph_std = float(np.std(graph_series))
                baseline_mae = float(np.mean(np.abs(baseline_series - target_series)))
                graph_mae = float(np.mean(np.abs(graph_series - target_series)))
                improvement = baseline_mae - graph_mae
                if target_std > 1e-8 and graph_std > 1e-8:
                    corr = float(np.corrcoef(target_series, graph_series)[0, 1])
                else:
                    corr = -1.0
                if not np.isfinite(corr):
                    corr = -1.0
                score_tuple = (
                    1 if improvement > 0.0 else 0,
                    1 if target_std > 0.05 else 0,
                    corr,
                    improvement / max(baseline_mae, 1e-6),
                    graph_std,
                    -graph_mae,
                    -int(candidate_node),
                    -horizon_candidate,
                )
                cases.append(
                    {
                        "score": score_tuple,
                        "kind": "phase",
                        "node_idx": int(candidate_node),
                        "channel_idx": horizon_candidate * state_dim,
                        "horizon_idx": horizon_candidate,
                        "feat_idx": 0,
                    }
                )
    elif system == "swing118":
        num_horizons = target.shape[-1] // state_dim
        for candidate_node in candidate_nodes:
            for horizon_candidate in range(num_horizons):
                target_phase = extract_swing_phase_series(target, int(candidate_node), horizon_candidate)
                baseline_phase = extract_swing_phase_series(baseline_mean, int(candidate_node), horizon_candidate)
                graph_phase = extract_swing_phase_series(graph_mean, int(candidate_node), horizon_candidate) if graph_mean is not None else baseline_phase
                target_std = float(np.std(target_phase))
                graph_std = float(np.std(graph_phase))
                baseline_mae = float(np.mean(np.abs(baseline_phase - target_phase)))
                graph_mae = float(np.mean(np.abs(graph_phase - target_phase)))
                improvement = baseline_mae - graph_mae
                if target_std > 1e-8 and graph_std > 1e-8:
                    corr = float(np.corrcoef(target_phase, graph_phase)[0, 1])
                else:
                    corr = -1.0
                if not np.isfinite(corr):
                    corr = -1.0
                cases.append(
                    {
                        "score": (
                            1 if improvement > 0.0 else 0,
                            1 if target_std > 0.05 else 0,
                            corr,
                            improvement / max(baseline_mae, 1e-6),
                            graph_std,
                            -graph_mae,
                            -int(candidate_node),
                            -horizon_candidate,
                        ),
                        "kind": "phase",
                        "node_idx": int(candidate_node),
                        "channel_idx": horizon_candidate * state_dim,
                        "horizon_idx": horizon_candidate,
                        "feat_idx": 0,
                    }
                )

                omega_idx = horizon_candidate * state_dim + 2
                target_series = target[:, int(candidate_node), omega_idx].cpu().numpy()
                baseline_series = baseline_mean[:, int(candidate_node), omega_idx].cpu().numpy()
                graph_series = graph_mean[:, int(candidate_node), omega_idx].cpu().numpy() if graph_mean is not None else baseline_series
                target_std = float(np.std(target_series))
                graph_std = float(np.std(graph_series))
                baseline_mae = float(np.mean(np.abs(baseline_series - target_series)))
                graph_mae = float(np.mean(np.abs(graph_series - target_series)))
                improvement = baseline_mae - graph_mae
                if target_std > 1e-8 and graph_std > 1e-8:
                    corr = float(np.corrcoef(target_series, graph_series)[0, 1])
                else:
                    corr = -1.0
                if not np.isfinite(corr):
                    corr = -1.0
                cases.append(
                    {
                        "score": (
                            1 if improvement > 0.0 else 0,
                            1 if target_std > 0.02 else 0,
                            corr,
                            improvement / max(baseline_mae, 1e-6),
                            graph_std,
                            -graph_mae,
                            -int(candidate_node),
                            -omega_idx,
                        ),
                        "kind": "omega",
                        "node_idx": int(candidate_node),
                        "channel_idx": omega_idx,
                        "horizon_idx": horizon_candidate,
                        "feat_idx": 2,
                    }
                )
    else:
        num_channels = target.shape[-1]
        for candidate_node in candidate_nodes:
            for channel_idx in range(num_channels):
                feat_idx = channel_idx % state_dim
                horizon_idx = channel_idx // state_dim
                baseline_series = baseline_mean[:, int(candidate_node), channel_idx].cpu().numpy()
                target_series = target[:, int(candidate_node), channel_idx].cpu().numpy()
                graph_series = graph_mean[:, int(candidate_node), channel_idx].cpu().numpy() if graph_mean is not None else baseline_series
                target_std = float(np.std(target_series))
                graph_std = float(np.std(graph_series))
                baseline_mae = float(np.mean(np.abs(baseline_series - target_series)))
                graph_mae = float(np.mean(np.abs(graph_series - target_series)))
                improvement = baseline_mae - graph_mae
                if target_std > 1e-8 and graph_std > 1e-8:
                    corr = float(np.corrcoef(target_series, graph_series)[0, 1])
                else:
                    corr = -1.0
                if not np.isfinite(corr):
                    corr = -1.0
                score_tuple = (
                    1 if improvement > 0.0 else 0,
                    1 if target_std > 0.02 else 0,
                    corr,
                    improvement / max(baseline_mae, 1e-6),
                    graph_std,
                    -graph_mae,
                    -int(candidate_node),
                    -channel_idx,
                )
                cases.append(
                    {
                        "score": score_tuple,
                        "kind": "raw",
                        "node_idx": int(candidate_node),
                        "channel_idx": channel_idx,
                        "horizon_idx": horizon_idx,
                        "feat_idx": feat_idx,
                    }
                )

    cases.sort(key=lambda item: item["score"], reverse=True)
    return cases


def sample_uncertainty_error(
    mean: torch.Tensor,
    variance: torch.Tensor,
    target: torch.Tensor,
    node_subset: np.ndarray | None,
    max_points: int = 4000,
):
    if node_subset is not None and len(node_subset) > 0:
        mean = mean[:, node_subset, :]
        variance = variance[:, node_subset, :]
        target = target[:, node_subset, :]
    pred_std = variance.clamp_min(1e-8).sqrt().reshape(-1)
    abs_err = (mean - target).abs().reshape(-1)
    if pred_std.numel() > max_points:
        idx = torch.linspace(0, pred_std.numel() - 1, max_points).long()
        pred_std = pred_std[idx]
        abs_err = abs_err[idx]
    return pred_std.cpu().numpy(), abs_err.cpu().numpy()


def extract_phase_series(tensor: torch.Tensor, node_idx: int, horizon_idx: int, state_dim: int):
    sin_idx = horizon_idx * state_dim
    cos_idx = sin_idx + 1
    phase = torch.atan2(tensor[:, node_idx, sin_idx], tensor[:, node_idx, cos_idx])
    return np.unwrap(phase.cpu().numpy())


def extract_swing_phase_series(tensor: torch.Tensor, node_idx: int, horizon_idx: int):
    return extract_phase_series(tensor, node_idx, horizon_idx, state_dim=3)


def extract_case_series(system: str, tensor: torch.Tensor, case: dict[str, Any]):
    kind = case.get("kind", "raw")
    node_idx = case["node_idx"]
    if kind == "phase":
        state_dim = 2 if system == "kuramoto" else 3
        return extract_phase_series(tensor, node_idx, case["horizon_idx"], state_dim)
    return tensor[:, node_idx, case["channel_idx"]].cpu().numpy()


def plot_visual_summary(
    result_dir: Path,
    system: str,
    targets: torch.Tensor,
    predictions: dict[str, tuple[torch.Tensor, torch.Tensor]],
    plot_meta: dict[str, Any],
    masked_nodes: np.ndarray | None,
):
    if plt is None or len(predictions) < 2:  # pragma: no cover - plotting is optional
        return

    baseline_name, graph_name = choose_model_pair(predictions)
    baseline_mean, baseline_var = predictions[baseline_name]
    graph_mean, graph_var = predictions[graph_name]

    focus_case = choose_focus_node_and_channel(
        system,
        targets,
        baseline_mean,
        graph_mean,
        masked_nodes,
    )
    node_idx = focus_case["node_idx"]
    channel_idx = focus_case["channel_idx"]
    horizon_idx = focus_case["horizon_idx"]
    feat_idx = focus_case["feat_idx"]
    series_kind = focus_case.get("kind", "raw")
    x_axis = np.arange(targets.shape[0])
    phase_view = series_kind == "phase"
    if phase_view:
        target_series = extract_case_series(system, targets, focus_case)
        baseline_series = extract_case_series(system, baseline_mean, focus_case)
        graph_series = extract_case_series(system, graph_mean, focus_case)
        baseline_std = None
        graph_std = None
    else:
        target_series = targets[:, node_idx, channel_idx].cpu().numpy()
        baseline_series = baseline_mean[:, node_idx, channel_idx].cpu().numpy()
        graph_series = graph_mean[:, node_idx, channel_idx].cpu().numpy()
        baseline_std = baseline_var[:, node_idx, channel_idx].sqrt().cpu().numpy()
        graph_std = graph_var[:, node_idx, channel_idx].sqrt().cpu().numpy()

    coords = plot_meta.get("coords")
    if coords is None and "area_ids" in plot_meta:
        coords = swing_layout(plot_meta["area_ids"])
    baseline_rmse = per_node_rmse(baseline_mean, targets).cpu().numpy()
    graph_rmse = per_node_rmse(graph_mean, targets).cpu().numpy()
    gain = baseline_rmse - graph_rmse

    subset = masked_nodes if masked_nodes is not None and len(masked_nodes) > 0 else None
    base_std_pts, base_err_pts = sample_uncertainty_error(baseline_mean, baseline_var, targets, subset)
    graph_std_pts, graph_err_pts = sample_uncertainty_error(graph_mean, graph_var, targets, subset)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    ax.plot(x_axis, target_series, color="black", linewidth=2.0, label="truth")
    ax.plot(x_axis, baseline_series, linewidth=1.8, label=baseline_name)
    if baseline_std is not None:
        ax.fill_between(
            x_axis,
            baseline_series - Z90 * baseline_std,
            baseline_series + Z90 * baseline_std,
            alpha=0.16,
        )
    ax.plot(x_axis, graph_series, linewidth=1.8, label=graph_name)
    if graph_std is not None:
        ax.fill_between(
            x_axis,
            graph_series - Z90 * graph_std,
            graph_series + Z90 * graph_std,
            alpha=0.16,
        )
    masked_text = "masked" if masked_nodes is not None and node_idx in set(masked_nodes.tolist()) else "observed"
    if phase_view:
        label = "phase" if system in {"kuramoto", "swing118"} else "trajectory"
        ax.set_title(f"Best-recovered {label}: node {node_idx} ({masked_text}), h{horizon_idx + 1}")
        ax.set_ylabel("Unwrapped phase")
    elif system == "swing118" and series_kind == "omega":
        ax.set_title(f"Best-recovered frequency: node {node_idx} ({masked_text}), h{horizon_idx + 1}")
        ax.set_ylabel("Omega")
    else:
        ax.set_title(f"Representative trajectory: node {node_idx} ({masked_text}), h{horizon_idx + 1}/c{feat_idx + 1}")
        ax.set_ylabel("Forecast value")
    ax.set_xlabel("Test window index")
    ax.legend()

    ax = axes[0, 1]
    if coords is not None:
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=gain, cmap="coolwarm", s=60, edgecolor="black", linewidth=0.3)
        if masked_nodes is not None and len(masked_nodes) > 0:
            ax.scatter(
                coords[masked_nodes, 0],
                coords[masked_nodes, 1],
                facecolors="none",
                edgecolors="gold",
                s=120,
                linewidths=1.4,
                label="masked",
            )
            ax.legend(loc="best")
        fig.colorbar(scatter, ax=ax, shrink=0.85, label=f"RMSE gain: {baseline_name} - {graph_name}")
        ax.set_title("Node-wise RMSE gain map")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.plot(gain)
        ax.set_title("Node-wise RMSE gain profile")
        ax.set_xlabel("Node index")
        ax.set_ylabel(f"{baseline_name} - {graph_name}")

    ax = axes[1, 0]
    hb = ax.hexbin(base_std_pts, base_err_pts, gridsize=28, cmap="Blues", mincnt=1)
    fig.colorbar(hb, ax=ax, shrink=0.85, label="count")
    ax.set_title(f"{baseline_name}: std vs abs error")
    ax.set_xlabel("Predicted std")
    ax.set_ylabel("Absolute error")

    ax = axes[1, 1]
    hb = ax.hexbin(graph_std_pts, graph_err_pts, gridsize=28, cmap="Reds", mincnt=1)
    fig.colorbar(hb, ax=ax, shrink=0.85, label="count")
    ax.set_title(f"{graph_name}: std vs abs error")
    ax.set_xlabel("Predicted std")
    ax.set_ylabel("Absolute error")

    subset_label = "masked nodes" if subset is not None else "all nodes"
    fig.suptitle(f"{system}: visualization summary ({subset_label})", fontsize=14)
    fig.tight_layout()
    fig.savefig(result_dir / f"{system}_visual_summary.png", dpi=180)
    plt.close(fig)


def plot_trajectory_grid(
    result_dir: Path,
    system: str,
    targets: torch.Tensor,
    predictions: dict[str, tuple[torch.Tensor, torch.Tensor]],
    masked_nodes: np.ndarray | None = None,
    max_cases: int = 6,
):
    if plt is None or len(predictions) < 2:  # pragma: no cover - plotting is optional
        return

    baseline_name, graph_name = choose_model_pair(predictions)
    baseline_mean, baseline_var = predictions[baseline_name]
    graph_mean, graph_var = predictions[graph_name]
    ranked_cases = rank_focus_cases(system, targets, baseline_mean, graph_mean, masked_nodes)
    if not ranked_cases:
        return

    state_dim = infer_state_dim(system)
    phase_view = system == "kuramoto"
    x_axis = np.arange(targets.shape[0])
    selected_cases = select_diverse_cases(ranked_cases, max_cases=max_cases, prefer_unique_nodes=True)
    num_cases = min(max_cases, len(selected_cases))
    cols = 3
    rows = int(math.ceil(num_cases / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.3 * rows), squeeze=False)

    for ax in axes.flat:
        ax.axis("off")

    for ax, case in zip(axes.flat, selected_cases[:num_cases]):
        ax.axis("on")
        node_idx = case["node_idx"]
        channel_idx = case["channel_idx"]
        horizon_idx = case["horizon_idx"]
        feat_idx = case["feat_idx"]
        case_kind = case.get("kind", "raw")
        if case_kind == "phase":
            target_series = extract_case_series(system, targets, case)
            graph_series = extract_case_series(system, graph_mean, case)
            graph_std = None
        else:
            target_series = targets[:, node_idx, channel_idx].cpu().numpy()
            graph_series = graph_mean[:, node_idx, channel_idx].cpu().numpy()
            graph_std = graph_var[:, node_idx, channel_idx].sqrt().cpu().numpy()

        ax.plot(x_axis, target_series, color="black", linewidth=1.8, label="truth")
        ax.plot(x_axis, graph_series, color="#dd6b20", linewidth=1.8, label=graph_name)
        if graph_std is not None:
            ax.fill_between(
                x_axis,
                graph_series - Z90 * graph_std,
                graph_series + Z90 * graph_std,
                color="#dd6b20",
                alpha=0.14,
            )
        masked_text = "masked" if masked_nodes is not None and node_idx in set(masked_nodes.tolist()) else "observed"
        if case_kind == "phase":
            ax.set_title(f"node {node_idx} ({masked_text}), h{horizon_idx + 1}")
            ax.set_ylabel("phase")
        elif system == "swing118" and case_kind == "omega":
            ax.set_title(f"node {node_idx} ({masked_text}), h{horizon_idx + 1}")
            ax.set_ylabel("omega")
        else:
            ax.set_title(f"node {node_idx} ({masked_text}), h{horizon_idx + 1}/c{feat_idx + 1}")
            ax.set_ylabel("value")
        ax.set_xlabel("test window")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)), frameon=False)
    fig.suptitle(f"{system}: top recovered {graph_name} trajectories", fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(result_dir / f"{system}_trajectory_grid.png", dpi=180)
    plt.close(fig)


def plot_forecasts(
    result_dir: Path,
    system: str,
    targets: torch.Tensor,
    predictions: dict[str, tuple[torch.Tensor, torch.Tensor]],
    masked_nodes: np.ndarray | None = None,
):
    if plt is None:  # pragma: no cover - plotting is optional
        return

    baseline_name, _ = choose_model_pair(predictions)
    baseline_mean, _ = predictions[baseline_name]
    focus_case = choose_focus_node_and_channel(
        system,
        targets,
        baseline_mean,
        predictions[choose_model_pair(predictions)[1]][0] if len(predictions) > 1 else None,
        masked_nodes,
    )
    node_idx = focus_case["node_idx"]
    channel_idx = focus_case["channel_idx"]
    horizon_idx = focus_case["horizon_idx"]
    feat_idx = focus_case["feat_idx"]
    series_kind = focus_case.get("kind", "raw")
    x_axis = np.arange(targets.shape[0])
    phase_view = series_kind == "phase"
    if phase_view:
        target_series = extract_case_series(system, targets, focus_case)
    else:
        target_series = targets[:, node_idx, channel_idx].numpy()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x_axis, target_series, label="truth", color="black", linewidth=2.0)
    for model_name, (mean, variance) in predictions.items():
        if phase_view:
            mean_series = extract_case_series(system, mean, focus_case)
            std_series = None
        else:
            mean_series = mean[:, node_idx, channel_idx].numpy()
            std_series = variance[:, node_idx, channel_idx].sqrt().numpy()
        ax.plot(x_axis, mean_series, label=model_name)
        if std_series is not None:
            ax.fill_between(
                x_axis,
                mean_series - Z90 * std_series,
                mean_series + Z90 * std_series,
                alpha=0.15,
            )
    masked_text = "masked" if masked_nodes is not None and node_idx in set(masked_nodes.tolist()) else "observed"
    if phase_view:
        label = "phase" if system in {"kuramoto", "swing118"} else "trajectory"
        ax.set_title(f"{system}: best-recovered {label}, node {node_idx} ({masked_text}), h{horizon_idx + 1}")
        ax.set_ylabel("Unwrapped phase")
    elif system == "swing118" and series_kind == "omega":
        ax.set_title(f"{system}: best-recovered frequency, node {node_idx} ({masked_text}), h{horizon_idx + 1}")
        ax.set_ylabel("Omega")
    else:
        ax.set_title(f"{system}: node {node_idx} ({masked_text}), h{horizon_idx + 1}/c{feat_idx + 1}")
        ax.set_ylabel("Forecast value")
    ax.set_xlabel("Test window index")
    ax.legend()
    fig.tight_layout()
    fig.savefig(result_dir / f"{system}_forecast.png", dpi=160)
    plt.close(fig)


def run_single(system: str, args, train_seed: int, data_seed: int, result_dir: Path):
    set_seed(train_seed)
    adj, split, masked_nodes, plot_meta = build_dataset(system, args, data_seed)
    split = maybe_limit_windows(
        split,
        max_train=args.max_train_windows,
        max_val=args.max_val_windows,
        max_test=args.max_test_windows,
    )
    input_dim = split.train_x.shape[-1]
    out_dim = split.train_y.shape[-1]
    graph_input = build_rescaled_laplacian(adj, lambda_max=2.0).to(args.device)
    val_group_masks = observation_group_masks(split.val_x, out_dim, args.append_obs_flag)
    test_group_masks = observation_group_masks(split.test_x, out_dim, args.append_obs_flag)

    metrics_rows = []
    plot_data = {}

    for model_idx, model_name in enumerate(args.models):
        model = None
        best_state = None
        best_val_metric = float("inf")
        best_val_rmse = float("inf")
        best_restart_idx = 0

        for restart_idx in range(args.num_restarts):
            restart_seed = train_seed + 1000 * (model_idx + 1) + 37 * restart_idx
            set_seed(restart_seed)
            model = build_model(model_name, input_dim=input_dim, out_dim=out_dim, args=args).to(args.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            restart_best_state = copy.deepcopy(model.state_dict())
            restart_best_metric = float("inf")
            restart_best_rmse = float("inf")
            epochs_without_improve = 0

            for epoch in range(args.epochs):
                train_epoch(model_name, model, graph_input, split.train_x, split.train_y, optimizer, args.device, args)
                val_metric = evaluate_selection_metric(
                    model_name,
                    model,
                    graph_input,
                    split.val_x,
                    split.val_y,
                    args.device,
                    args.var_floor,
                    args,
                )
                val_rmse = evaluate_rmse(model_name, model, graph_input, split.val_x, split.val_y, args.device, args.var_floor)
                if val_metric + 1e-6 < restart_best_metric:
                    restart_best_metric = val_metric
                    restart_best_state = copy.deepcopy(model.state_dict())
                    epochs_without_improve = 0
                    restart_best_rmse = val_rmse
                else:
                    epochs_without_improve += 1
                    if epochs_without_improve >= args.patience:
                        break

            if (
                restart_best_metric + 1e-6 < best_val_metric
                or (
                    abs(restart_best_metric - best_val_metric) <= 1e-6
                    and restart_best_rmse < best_val_rmse
                )
            ):
                best_val_metric = restart_best_metric
                best_val_rmse = restart_best_rmse
                best_state = copy.deepcopy(restart_best_state)
                best_restart_idx = restart_idx

        if model is None or best_state is None:
            raise RuntimeError(f"Failed to train model {model_name}.")
        model.load_state_dict(best_state)

        val_mean, val_var = collect_predictions(model_name, model, graph_input, split.val_x, args.device, args.var_floor)
        val_target = torch.from_numpy(split.val_y)
        calibrator = fit_variance_calibrator(
            val_mean,
            val_target,
            val_var,
            args.var_floor,
            group_masks=val_group_masks,
        )

        test_mean, test_var = collect_predictions(model_name, model, graph_input, split.test_x, args.device, args.var_floor)
        test_target = torch.from_numpy(split.test_y)
        calibrated_var = apply_variance_calibrator(
            test_mean.shape,
            test_var,
            calibrator,
            args.var_floor,
            group_masks=test_group_masks,
        )
        metrics = regression_metrics(test_mean, test_target, calibrated_var)
        if masked_nodes is not None and len(masked_nodes) > 0:
            masked_metrics = regression_metrics(test_mean, test_target, calibrated_var, node_idx=masked_nodes)
            masked_record = {f"masked_{k}": v for k, v in masked_metrics.items()}
        else:
            masked_record = {
                f"masked_{key}": float("nan")
                for key in ("rmse", "mae", "nll", "coverage90", "width90", "std_error_corr")
            }
        metrics_rows.append(
            {
                "system": system,
                "model": model_name,
                "seed": train_seed,
                "data_seed": data_seed,
                "restart": best_restart_idx,
                "val_rmse": best_val_rmse,
                "val_metric": best_val_metric,
                "variance_mode": calibrator["mode"],
                "variance_value": calibrator.get("value", float("nan")),
                **metrics,
                **masked_record,
            }
        )
        plot_data[model_name] = (test_mean, calibrated_var)

    targets = torch.from_numpy(split.test_y)[:, :, :out_dim]
    plot_forecasts(result_dir, system, targets, plot_data, masked_nodes=masked_nodes)
    plot_visual_summary(result_dir, system, targets, plot_data, plot_meta, masked_nodes)
    plot_trajectory_grid(result_dir, system, targets, plot_data, masked_nodes=masked_nodes)
    return metrics_rows


def summarize_rows(rows):
    summary = {}
    for row in rows:
        key = (row["system"], row["model"])
        summary.setdefault(
            key,
            {
                k: []
                for k in (
                    "rmse",
                    "mae",
                    "nll",
                    "coverage90",
                    "width90",
                    "std_error_corr",
                    "masked_rmse",
                    "masked_mae",
                    "masked_nll",
                    "masked_coverage90",
                    "masked_width90",
                    "masked_std_error_corr",
                )
            },
        )
        for metric in summary[key]:
            value = row.get(metric, float("nan"))
            if not math.isnan(value):
                summary[key][metric].append(value)
    out = []
    for (system, model), metric_map in sorted(summary.items()):
        record = {"system": system, "model": model}
        for metric, values in metric_map.items():
            if values:
                record[f"{metric}_mean"] = float(np.mean(values))
                record[f"{metric}_std"] = float(np.std(values))
            else:
                record[f"{metric}_mean"] = float("nan")
                record[f"{metric}_std"] = float("nan")
        out.append(record)
    return out


def write_csv(path: Path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def preferred_visual_model(system: str, rows):
    model_names = {row["model"] for row in rows if row["system"] == system}
    preferred = [
        "gru_phase_dss",
        "gru_dssres",
        "gru_dss",
        "graphgru_dss",
        "dss",
        "gru_graph_p0",
        "graph_p0",
        "gru_mlp",
        "mlp",
    ]
    for model_name in preferred:
        if model_name in model_names:
            return model_name
    return None


def export_best_seed_visuals(result_dir: Path, rows):
    systems = sorted({row["system"] for row in rows})
    metadata = {}
    for system in systems:
        model_name = preferred_visual_model(system, rows)
        if model_name is None:
            continue
        candidates = [row for row in rows if row["system"] == system and row["model"] == model_name]
        if not candidates:
            continue
        best_row = min(candidates, key=lambda row: row.get("val_metric", row.get("val_rmse", float("inf"))))
        seed = best_row["seed"]
        seed_dir = result_dir / f"{system}_seed{seed}"
        copied = {}
        for stem in (f"{system}_visual_summary.png", f"{system}_forecast.png", f"{system}_trajectory_grid.png"):
            src = seed_dir / stem
            if src.exists():
                dst = result_dir / f"{system}_best_{stem.split('_', 1)[1]}"
                shutil.copy2(src, dst)
                copied[stem] = str(dst.relative_to(result_dir))
        metadata[system] = {
            "model": model_name,
            "seed": seed,
            "val_metric": best_row.get("val_metric", float("nan")),
            "val_rmse": best_row.get("val_rmse", float("nan")),
            "files": copied,
        }
    if metadata:
        with (result_dir / "best_seed_visuals.json").open("w") as f:
            json.dump(metadata, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic stochastic graph-field forecasting with DSS-GNN.")
    parser.add_argument("--systems", nargs="+", default=["diffusion", "swing118"], choices=list(SYSTEMS))
    parser.add_argument("--models", nargs="+", default=["mlp", "graph_p0", "dss", "gru_graph_p0", "gru_dss"], choices=list(MODELS))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--num_restarts", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=48)
    parser.add_argument("--temporal_hidden_dim", type=int, default=48)
    parser.add_argument("--temporal_layers", type=int, default=1)
    parser.add_argument("--temporal_dropout", type=float, default=0.0)
    parser.add_argument("--recurrent_spatial_layers", type=int, default=1)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--K_lp", type=int, default=3)
    parser.add_argument("--K_hp", type=int, default=3)
    parser.add_argument("--P", type=int, default=2)
    parser.add_argument("--S", type=int, default=5)
    parser.add_argument("--P_gate", type=int, default=1)
    parser.add_argument("--use_random_gates", action="store_true")
    parser.add_argument("--shared_input_lift", dest="shared_input_lift", action="store_true")
    parser.add_argument("--no_shared_input_lift", dest="shared_input_lift", action="store_false")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lin_dropout", type=float, default=0.0)
    parser.add_argument("--var_floor", type=float, default=1e-4)
    parser.add_argument("--var_reg", type=float, default=1e-4)
    parser.add_argument("--nll_weight", type=float, default=0.2)
    parser.add_argument("--masked_loss_weight", type=float, default=1.0)
    parser.add_argument("--phase_loss_weight", type=float, default=0.2)
    parser.add_argument("--delta_loss_weight", type=float, default=0.3)
    parser.add_argument("--val_phase_weight", type=float, default=0.25)
    parser.add_argument("--val_delta_weight", type=float, default=0.35)
    parser.add_argument("--swing_phase_loss_weight", type=float, default=0.15)
    parser.add_argument("--swing_omega_loss_weight", type=float, default=0.15)
    parser.add_argument("--swing_domega_loss_weight", type=float, default=0.25)
    parser.add_argument("--val_swing_phase_weight", type=float, default=0.2)
    parser.add_argument("--val_swing_omega_weight", type=float, default=0.15)
    parser.add_argument("--val_swing_domega_weight", type=float, default=0.3)
    parser.add_argument("--num_nodes", type=int, default=24)
    parser.add_argument("--num_steps", type=int, default=320)
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--train_frac", type=float, default=0.6)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_val_windows", type=int, default=None)
    parser.add_argument("--max_test_windows", type=int, default=None)
    parser.add_argument("--graph_k", type=int, default=6)
    parser.add_argument("--graph_length_scale", type=float, default=0.25)
    parser.add_argument("--dt_diffusion", type=float, default=0.08)
    parser.add_argument("--diffusion_rate", type=float, default=1.2)
    parser.add_argument("--damping", type=float, default=0.35)
    parser.add_argument("--forcing_scale", type=float, default=0.9)
    parser.add_argument("--noise_std_diffusion", type=float, default=0.18)
    parser.add_argument("--dt_kuramoto", type=float, default=0.05)
    parser.add_argument("--kuramoto_coupling", type=float, default=1.6)
    parser.add_argument("--noise_std_kuramoto", type=float, default=0.22)
    parser.add_argument("--num_steps_swing", type=int, default=520)
    parser.add_argument("--num_rollouts", type=int, default=1)
    parser.add_argument("--topology_seed", type=int, default=11)
    parser.add_argument("--dt_swing", type=float, default=0.03)
    parser.add_argument("--swing_coupling", type=float, default=4.5)
    parser.add_argument("--swing_damping", type=float, default=1.1)
    parser.add_argument("--swing_inertia", type=float, default=2.0)
    parser.add_argument("--swing_forcing_scale", type=float, default=0.45)
    parser.add_argument("--noise_std_swing", type=float, default=0.16)
    parser.add_argument("--masked_sensor_frac", type=float, default=0.0)
    parser.add_argument("--append_obs_flag", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data_seed", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--tag", type=str, default="local")
    parser.set_defaults(shared_input_lift=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    result_dir = Path("results_local") / "stochastic_field_forecast" / args.tag
    result_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for run_idx in range(args.runs):
        train_seed = args.seed + run_idx
        data_seed = args.data_seed if args.data_seed is not None else train_seed
        for system in args.systems:
            system_dir = result_dir / f"{system}_seed{train_seed}"
            system_dir.mkdir(parents=True, exist_ok=True)
            rows = run_single(system, args, train_seed, data_seed, system_dir)
            all_rows.extend(rows)

    summary_rows = summarize_rows(all_rows)
    write_csv(result_dir / "per_run_metrics.csv", all_rows)
    write_csv(result_dir / "summary.csv", summary_rows)
    export_best_seed_visuals(result_dir, all_rows)
    with (result_dir / "config.json").open("w") as f:
        json.dump(
            {
                "systems": args.systems,
                "models": args.models,
                "runs": args.runs,
                "num_restarts": args.num_restarts,
                "epochs": args.epochs,
                "seed": args.seed,
                "data_seed": args.data_seed,
                "device": str(args.device),
            },
            f,
            indent=2,
        )

    print(f"Saved results to {result_dir}")
    for row in summary_rows:
        print(
            f"{row['system']:>10s} | {row['model']:>8s} | "
            f"RMSE {row['rmse_mean']:.4f} +/- {row['rmse_std']:.4f} | "
            f"MaskedRMSE {row['masked_rmse_mean']:.4f} | "
            f"NLL {row['nll_mean']:.4f} +/- {row['nll_std']:.4f} | "
            f"Cov90 {row['coverage90_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
