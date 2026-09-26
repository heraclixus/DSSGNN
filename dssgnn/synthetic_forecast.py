"""Synthetic stochastic graph-field forecasting utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import DSSGNN, _field_stats_from_sample_outputs


@dataclass
class ForecastSplit:
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray


def make_ieee118_style_grid(seed: int = 0):
    """Create a sparse modular 118-node transmission-style graph."""
    rng = np.random.default_rng(seed)
    area_sizes = [24, 22, 25, 23, 24]
    num_nodes = sum(area_sizes)
    assert num_nodes == 118

    rows, cols, vals = [], [], []
    area_ids = np.zeros(num_nodes, dtype=np.int64)
    offset = 0
    boundaries = []

    for area_idx, size in enumerate(area_sizes):
        nodes = np.arange(offset, offset + size)
        area_ids[nodes] = area_idx
        boundaries.append((nodes[0], nodes[-1]))

        # Backbone chain within each area.
        for u, v in zip(nodes[:-1], nodes[1:]):
            weight = rng.uniform(0.9, 1.2)
            rows.extend([u, v])
            cols.extend([v, u])
            vals.extend([weight, weight])

        # Local chords to introduce loops and alternate paths.
        for idx in range(0, size - 3, 3):
            u = nodes[idx]
            v = nodes[min(idx + 3, size - 1)]
            weight = rng.uniform(0.7, 1.0)
            rows.extend([u, v])
            cols.extend([v, u])
            vals.extend([weight, weight])

        # A few random local tie-lines per area.
        for _ in range(max(2, size // 8)):
            u, v = rng.choice(nodes, size=2, replace=False)
            if abs(int(u) - int(v)) <= 1:
                continue
            weight = rng.uniform(0.6, 1.0)
            rows.extend([u, v])
            cols.extend([v, u])
            vals.extend([weight, weight])

        offset += size

    # Inter-area transmission lines.
    for (left_start, left_end), (right_start, right_end) in zip(boundaries[:-1], boundaries[1:]):
        candidates = [
            (left_end, right_start),
            (left_end - 2, right_start + 2),
            ((left_start + left_end) // 2, (right_start + right_end) // 2),
        ]
        for u, v in candidates:
            weight = rng.uniform(0.9, 1.3)
            rows.extend([u, v])
            cols.extend([v, u])
            vals.extend([weight, weight])

    # A few long-range transmission corridors.
    long_range_pairs = [(5, 70), (18, 92), (40, 108), (12, 55), (63, 101)]
    for u, v in long_range_pairs:
        weight = rng.uniform(0.7, 1.1)
        rows.extend([u, v])
        cols.extend([v, u])
        vals.extend([weight, weight])

    adj = sp.coo_matrix((vals, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float32)
    adj = adj.maximum(adj.T).tocsr()
    adj = adj + sp.eye(num_nodes, format="csr", dtype=np.float32)
    return adj, area_ids


def make_sensor_graph(num_nodes: int, k: int = 6, length_scale: float = 0.25, seed: int = 0):
    """Create a symmetric geometric sensor graph with RBF edge weights."""
    rng = np.random.default_rng(seed)
    coords = rng.random((num_nodes, 2))
    diffs = coords[:, None, :] - coords[None, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=-1))

    rows, cols, vals = [], [], []
    for i in range(num_nodes):
        nbrs = np.argsort(dists[i])[1 : k + 1]
        for j in nbrs:
            weight = math.exp(-(dists[i, j] ** 2) / (2.0 * length_scale ** 2))
            rows.append(i)
            cols.append(j)
            vals.append(weight)

    adj = sp.coo_matrix((vals, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float32)
    adj = adj.maximum(adj.T).tocsr()
    adj = adj + sp.eye(num_nodes, format="csr", dtype=np.float32)
    return adj, coords


def row_stochastic_weights(adj: sp.csr_matrix) -> np.ndarray:
    """Return a dense row-stochastic matrix without self-loop domination."""
    adj = adj.tocsr().astype(np.float64)
    off_diag = adj - sp.diags(adj.diagonal(), format="csr")
    deg = np.asarray(off_diag.sum(axis=1)).reshape(-1)
    deg[deg == 0.0] = 1.0
    inv_deg = sp.diags(1.0 / deg, format="csr")
    return inv_deg.dot(off_diag).toarray()


def dense_weight_matrix(adj: sp.csr_matrix) -> np.ndarray:
    """Return dense symmetric off-diagonal coupling weights."""
    adj = adj.tocsr().astype(np.float64)
    off_diag = adj - sp.diags(adj.diagonal(), format="csr")
    dense = off_diag.toarray()
    np.fill_diagonal(dense, 0.0)
    return dense


def simulate_stochastic_diffusion(
    adj: sp.csr_matrix,
    coords: np.ndarray,
    num_steps: int = 400,
    dt: float = 0.08,
    diffusion: float = 1.2,
    damping: float = 0.35,
    forcing_scale: float = 0.9,
    noise_std: float = 0.18,
    seed: int = 0,
) -> np.ndarray:
    """Simulate a noisy diffusion-reaction field on the graph."""
    rng = np.random.default_rng(seed)
    weights = row_stochastic_weights(adj)
    state = 0.2 * rng.normal(size=adj.shape[0])

    basis_1 = np.sin(2.0 * math.pi * coords[:, 0])
    basis_2 = np.cos(2.0 * math.pi * coords[:, 1])

    states = []
    for t in range(num_steps):
        force = forcing_scale * (
            math.sin(0.05 * t + 0.3) * basis_1 + math.cos(0.03 * t - 0.2) * basis_2
        )
        lap_term = weights @ state - state
        state = state + dt * (diffusion * lap_term - damping * state + force)
        state = state + noise_std * math.sqrt(dt) * rng.normal(size=state.shape[0])
        states.append(state[:, None].astype(np.float32))
    return np.stack(states, axis=0)


def simulate_stochastic_kuramoto(
    adj: sp.csr_matrix,
    num_steps: int = 400,
    dt: float = 0.05,
    coupling: float = 1.6,
    noise_std: float = 0.22,
    seed: int = 0,
) -> np.ndarray:
    """Simulate noisy coupled oscillators and return sine/cosine observations."""
    rng = np.random.default_rng(seed)
    weights = row_stochastic_weights(adj)
    num_nodes = adj.shape[0]
    theta = rng.uniform(-math.pi, math.pi, size=num_nodes)
    natural_freq = rng.normal(loc=0.0, scale=0.35, size=num_nodes)

    states = []
    for _ in range(num_steps):
        phase_diff = theta[None, :] - theta[:, None]
        coupling_term = (weights * np.sin(phase_diff)).sum(axis=1)
        theta = theta + dt * (natural_freq + coupling * coupling_term)
        theta = theta + noise_std * math.sqrt(dt) * rng.normal(size=num_nodes)
        theta = (theta + math.pi) % (2.0 * math.pi) - math.pi
        states.append(np.stack([np.sin(theta), np.cos(theta)], axis=-1).astype(np.float32))
    return np.stack(states, axis=0)


def simulate_stochastic_swing(
    adj: sp.csr_matrix,
    area_ids: np.ndarray,
    num_steps: int = 500,
    dt: float = 0.03,
    coupling: float = 4.5,
    damping: float = 1.1,
    inertia: float = 2.0,
    forcing_scale: float = 0.45,
    noise_std: float = 0.16,
    seed: int = 0,
) -> np.ndarray:
    """Simulate stochastic swing dynamics on a transmission-style graph."""
    rng = np.random.default_rng(seed)
    weights = dense_weight_matrix(adj)
    num_nodes = adj.shape[0]

    theta = rng.normal(loc=0.0, scale=0.1, size=num_nodes)
    omega = rng.normal(loc=0.0, scale=0.03, size=num_nodes)

    # Area-wise net injections, re-centered so total power is balanced.
    area_drive = np.array([0.22, 0.08, -0.12, -0.05, -0.13], dtype=np.float64)
    base_injection = area_drive[area_ids] + rng.normal(scale=0.04, size=num_nodes)
    base_injection = base_injection - base_injection.mean()

    disturbance_profile = rng.normal(size=num_nodes)
    disturbance_profile = disturbance_profile - disturbance_profile.mean()
    disturbance_profile = disturbance_profile / (np.linalg.norm(disturbance_profile) + 1e-8)

    states = []
    for t in range(num_steps):
        phase_diff = theta[None, :] - theta[:, None]
        electrical = coupling * (weights * np.sin(phase_diff)).sum(axis=1)
        forcing_t = forcing_scale * (
            math.sin(0.025 * t + 0.1) + 0.6 * math.cos(0.041 * t - 0.35)
        )
        disturbance_t = 0.35 * forcing_scale * math.sin(0.013 * t + 0.7)
        domega = (
            base_injection
            - damping * omega
            + electrical
            + forcing_t * disturbance_profile
            + disturbance_t * (area_ids == 2)
        ) / inertia
        omega = omega + dt * domega + noise_std * math.sqrt(dt) * rng.normal(size=num_nodes)
        theta = theta + dt * omega
        theta = (theta + math.pi) % (2.0 * math.pi) - math.pi
        states.append(np.stack([np.sin(theta), np.cos(theta), omega], axis=-1).astype(np.float32))
    return np.stack(states, axis=0)


def make_forecasting_windows(states: np.ndarray, context: int, horizon: int):
    """Build direct multi-step windows for per-node forecasting."""
    num_steps, num_nodes, state_dim = states.shape
    xs, ys = [], []
    for t in range(context, num_steps - horizon + 1):
        x = states[t - context : t].transpose(1, 0, 2).reshape(num_nodes, context * state_dim)
        y = states[t : t + horizon].transpose(1, 0, 2).reshape(num_nodes, horizon * state_dim)
        xs.append(x.astype(np.float32))
        ys.append(y.astype(np.float32))
    return np.stack(xs, axis=0), np.stack(ys, axis=0)


def split_forecasting_windows(
    x: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> ForecastSplit:
    """Temporal split of forecasting windows."""
    num_samples = x.shape[0]
    train_end = int(num_samples * train_frac)
    val_end = int(num_samples * (train_frac + val_frac))
    return ForecastSplit(
        train_x=x[:train_end],
        train_y=y[:train_end],
        val_x=x[train_end:val_end],
        val_y=y[train_end:val_end],
        test_x=x[val_end:],
        test_y=y[val_end:],
    )


class NodeMLPForecaster(nn.Module):
    """Node-wise MLP baseline that ignores graph structure."""

    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalNodeEncoder(nn.Module):
    """Per-node GRU encoder for context windows."""

    def __init__(
        self,
        input_state_dim: int,
        context: int,
        hidden_dim: int,
        extra_static_dim: int = 0,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_state_dim = input_state_dim
        self.context = context
        self.extra_static_dim = extra_static_dim
        self.sequence_dim = context * input_state_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_size=input_state_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    @property
    def output_dim(self) -> int:
        return self.hidden_dim + self.extra_static_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x[:, : self.sequence_dim].reshape(x.shape[0], self.context, self.input_state_dim)
        _, hidden = self.gru(seq)
        out = hidden[-1]
        if self.extra_static_dim > 0:
            extra = x[:, self.sequence_dim : self.sequence_dim + self.extra_static_dim]
            out = torch.cat([out, extra], dim=-1)
        return out


class TemporalMLPForecaster(nn.Module):
    """GRU encoder followed by a node-wise decoder."""

    def __init__(
        self,
        input_state_dim: int,
        context: int,
        hidden_dim: int,
        out_dim: int,
        extra_static_dim: int = 0,
        temporal_layers: int = 1,
        temporal_dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = TemporalNodeEncoder(
            input_state_dim=input_state_dim,
            context=context,
            hidden_dim=hidden_dim,
            extra_static_dim=extra_static_dim,
            num_layers=temporal_layers,
            dropout=temporal_dropout,
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class GraphGRUCell(nn.Module):
    """Graph-aware GRU cell built from deterministic DSS order-0 blocks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        K_lp: int = 3,
        K_hp: int = 3,
        spatial_layers: int = 1,
        dropout=(0.0, 0.0),
    ):
        super().__init__()
        gate_dim = input_dim + hidden_dim
        self.hidden_dim = hidden_dim
        self.gate_net = DSSGNN(
            input_dim=gate_dim,
            hidden_dim=hidden_dim,
            out_dim=2 * hidden_dim,
            num_layers=spatial_layers,
            K_lp=K_lp,
            K_hp=K_hp,
            P=0,
            dropout=dropout,
            activation=True,
            shared_input_lift=False,
        )
        self.candidate_net = DSSGNN(
            input_dim=gate_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_layers=spatial_layers,
            K_lp=K_lp,
            K_hp=K_hp,
            P=0,
            dropout=dropout,
            activation=True,
            shared_input_lift=False,
        )

    def forward(self, L_rescaled, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        gate_input = torch.cat([x_t, h_prev], dim=-1)
        gates = self.gate_net(L_rescaled, gate_input)
        update_gate, reset_gate = torch.sigmoid(gates).chunk(2, dim=-1)
        candidate_input = torch.cat([x_t, reset_gate * h_prev], dim=-1)
        candidate = torch.tanh(self.candidate_net(L_rescaled, candidate_input))
        return update_gate * h_prev + (1.0 - update_gate) * candidate


class TemporalGraphEncoder(nn.Module):
    """Graph-aware recurrent encoder for context windows."""

    def __init__(
        self,
        input_state_dim: int,
        context: int,
        hidden_dim: int,
        extra_static_dim: int = 0,
        num_layers: int = 1,
        dropout: float = 0.0,
        K_lp: int = 3,
        K_hp: int = 3,
        spatial_layers: int = 1,
        spatial_dropout=(0.0, 0.0),
    ):
        super().__init__()
        self.input_state_dim = input_state_dim
        self.context = context
        self.extra_static_dim = extra_static_dim
        self.sequence_dim = context * input_state_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.cells = nn.ModuleList(
            [
                GraphGRUCell(
                    input_dim=input_state_dim if layer_idx == 0 else hidden_dim,
                    hidden_dim=hidden_dim,
                    K_lp=K_lp,
                    K_hp=K_hp,
                    spatial_layers=spatial_layers,
                    dropout=spatial_dropout,
                )
                for layer_idx in range(num_layers)
            ]
        )

    @property
    def output_dim(self) -> int:
        return self.hidden_dim + self.extra_static_dim

    def forward(self, L_rescaled, x: torch.Tensor) -> torch.Tensor:
        seq = x[:, : self.sequence_dim].reshape(x.shape[0], self.context, self.input_state_dim)
        hidden_states = [
            x.new_zeros(x.shape[0], self.hidden_dim)
            for _ in range(self.num_layers)
        ]
        for step_idx in range(self.context):
            layer_input = seq[:, step_idx]
            for layer_idx, cell in enumerate(self.cells):
                h_next = cell(L_rescaled, layer_input, hidden_states[layer_idx])
                hidden_states[layer_idx] = h_next
                layer_input = h_next
                if self.dropout > 0.0 and layer_idx < self.num_layers - 1:
                    layer_input = F.dropout(layer_input, p=self.dropout, training=self.training)
        out = hidden_states[-1]
        if self.extra_static_dim > 0:
            extra = x[:, self.sequence_dim : self.sequence_dim + self.extra_static_dim]
            out = torch.cat([out, extra], dim=-1)
        return out


class TemporalDSSForecaster(nn.Module):
    """GRU encoder followed by a DSS spatial forecaster."""

    def __init__(
        self,
        input_state_dim: int,
        context: int,
        temporal_hidden_dim: int,
        spatial_hidden_dim: int,
        out_dim: int,
        extra_static_dim: int = 0,
        temporal_layers: int = 1,
        temporal_dropout: float = 0.0,
        K_lp: int = 3,
        K_hp: int = 3,
        P: int = 0,
        layers: int = 2,
        dropout=(0.0, 0.0),
        use_random_gates: bool = False,
        P_gate: int = 1,
        S: int = 4,
        shared_input_lift: bool = False,
    ):
        super().__init__()
        self.encoder = TemporalNodeEncoder(
            input_state_dim=input_state_dim,
            context=context,
            hidden_dim=temporal_hidden_dim,
            extra_static_dim=extra_static_dim,
            num_layers=temporal_layers,
            dropout=temporal_dropout,
        )
        self.backbone = DSSGNN(
            input_dim=self.encoder.output_dim,
            hidden_dim=spatial_hidden_dim,
            out_dim=out_dim,
            num_layers=layers,
            K_lp=K_lp,
            K_hp=K_hp,
            P=P,
            dropout=dropout,
            activation=True,
            use_random_gates=use_random_gates,
            P_gate=P_gate,
            S=S,
            shared_input_lift=shared_input_lift,
        )

    def forward(self, L_rescaled, x: torch.Tensor, return_uncertainty: bool = False, **kwargs):
        encoded = self.encoder(x)
        return self.backbone(L_rescaled, encoded, return_uncertainty=return_uncertainty, **kwargs)

    def forward_with_field_stats(self, L_rescaled, x: torch.Tensor, eps: float = 1e-12):
        encoded = self.encoder(x)
        return self.backbone.forward_with_field_stats(L_rescaled, encoded, eps=eps)


class TemporalGraphDSSForecaster(nn.Module):
    """Graph-aware recurrent encoder plus DSS spatial decoder."""

    def __init__(
        self,
        input_state_dim: int,
        context: int,
        temporal_hidden_dim: int,
        spatial_hidden_dim: int,
        out_dim: int,
        extra_static_dim: int = 0,
        temporal_layers: int = 1,
        temporal_dropout: float = 0.0,
        recurrent_spatial_layers: int = 1,
        K_lp: int = 3,
        K_hp: int = 3,
        P: int = 0,
        layers: int = 2,
        dropout=(0.0, 0.0),
        use_random_gates: bool = False,
        P_gate: int = 1,
        S: int = 4,
        shared_input_lift: bool = False,
    ):
        super().__init__()
        self.encoder = TemporalGraphEncoder(
            input_state_dim=input_state_dim,
            context=context,
            hidden_dim=temporal_hidden_dim,
            extra_static_dim=extra_static_dim,
            num_layers=temporal_layers,
            dropout=temporal_dropout,
            K_lp=K_lp,
            K_hp=K_hp,
            spatial_layers=recurrent_spatial_layers,
            spatial_dropout=dropout,
        )
        self.backbone = DSSGNN(
            input_dim=self.encoder.output_dim,
            hidden_dim=spatial_hidden_dim,
            out_dim=out_dim,
            num_layers=layers,
            K_lp=K_lp,
            K_hp=K_hp,
            P=P,
            dropout=dropout,
            activation=True,
            use_random_gates=use_random_gates,
            P_gate=P_gate,
            S=S,
            shared_input_lift=shared_input_lift,
        )

    def forward(self, L_rescaled, x: torch.Tensor, return_uncertainty: bool = False, **kwargs):
        encoded = self.encoder(L_rescaled, x)
        return self.backbone(L_rescaled, encoded, return_uncertainty=return_uncertainty, **kwargs)

    def forward_with_field_stats(self, L_rescaled, x: torch.Tensor, eps: float = 1e-12):
        encoded = self.encoder(L_rescaled, x)
        return self.backbone.forward_with_field_stats(L_rescaled, encoded, eps=eps)


class TemporalResidualDSSForecaster(nn.Module):
    """Strong local temporal base plus DSS graph residual correction."""

    def __init__(
        self,
        input_state_dim: int,
        context: int,
        temporal_hidden_dim: int,
        spatial_hidden_dim: int,
        out_dim: int,
        extra_static_dim: int = 0,
        temporal_layers: int = 1,
        temporal_dropout: float = 0.0,
        K_lp: int = 3,
        K_hp: int = 3,
        P: int = 0,
        layers: int = 2,
        dropout=(0.0, 0.0),
        use_random_gates: bool = False,
        P_gate: int = 1,
        S: int = 4,
        shared_input_lift: bool = False,
    ):
        super().__init__()
        self.encoder = TemporalNodeEncoder(
            input_state_dim=input_state_dim,
            context=context,
            hidden_dim=temporal_hidden_dim,
            extra_static_dim=extra_static_dim,
            num_layers=temporal_layers,
            dropout=temporal_dropout,
        )
        self.base_head = nn.Sequential(
            nn.Linear(self.encoder.output_dim, temporal_hidden_dim),
            nn.ReLU(),
            nn.Linear(temporal_hidden_dim, out_dim),
        )
        self.backbone = DSSGNN(
            input_dim=self.encoder.output_dim,
            hidden_dim=spatial_hidden_dim,
            out_dim=out_dim,
            num_layers=layers,
            K_lp=K_lp,
            K_hp=K_hp,
            P=P,
            dropout=dropout,
            activation=True,
            use_random_gates=use_random_gates,
            P_gate=P_gate,
            S=S,
            shared_input_lift=shared_input_lift,
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, L_rescaled, x: torch.Tensor, return_uncertainty: bool = False, **kwargs):
        encoded = self.encoder(x)
        base_mean = self.base_head(encoded)
        if return_uncertainty:
            residual_mean, residual_uncertainty = self.backbone(
                L_rescaled,
                encoded,
                return_uncertainty=True,
                **kwargs,
            )
            mean = base_mean + self.residual_scale * residual_mean
            uncertainty = self.residual_scale.abs() * residual_uncertainty
            return mean, uncertainty
        residual_mean = self.backbone(L_rescaled, encoded, return_uncertainty=False, **kwargs)
        return base_mean + self.residual_scale * residual_mean

    def forward_with_field_stats(self, L_rescaled, x: torch.Tensor, eps: float = 1e-12):
        encoded = self.encoder(x)
        base_mean = self.base_head(encoded)
        residual_stats = self.backbone.forward_with_field_stats(L_rescaled, encoded, eps=eps)
        sample_outputs = base_mean.unsqueeze(0) + self.residual_scale * residual_stats["sample_outputs"]
        stats = _field_stats_from_sample_outputs(sample_outputs, residual_stats["weights"], eps=eps)
        stats["mean"] = base_mean + self.residual_scale * residual_stats["mean"]
        return stats


class TemporalPhaseDSSForecaster(nn.Module):
    """Kuramoto-oriented residual forecaster that predicts phase increments on the unit circle."""

    def __init__(
        self,
        input_state_dim: int,
        context: int,
        temporal_hidden_dim: int,
        spatial_hidden_dim: int,
        horizon: int,
        extra_static_dim: int = 0,
        temporal_layers: int = 1,
        temporal_dropout: float = 0.0,
        K_lp: int = 3,
        K_hp: int = 3,
        P: int = 0,
        layers: int = 2,
        dropout=(0.0, 0.0),
        use_random_gates: bool = False,
        P_gate: int = 1,
        S: int = 4,
        shared_input_lift: bool = False,
    ):
        super().__init__()
        if input_state_dim != 2:
            raise ValueError("TemporalPhaseDSSForecaster expects Kuramoto-style sin/cos states (input_state_dim=2).")
        self.context = context
        self.input_state_dim = input_state_dim
        self.horizon = horizon
        self.extra_static_dim = extra_static_dim
        self.sequence_dim = context * input_state_dim
        self.local_encoder = TemporalNodeEncoder(
            input_state_dim=input_state_dim,
            context=context,
            hidden_dim=temporal_hidden_dim,
            extra_static_dim=extra_static_dim,
            num_layers=temporal_layers,
            dropout=temporal_dropout,
        )
        self.graph_encoder = TemporalGraphEncoder(
            input_state_dim=input_state_dim,
            context=context,
            hidden_dim=temporal_hidden_dim,
            extra_static_dim=extra_static_dim,
            num_layers=temporal_layers,
            dropout=temporal_dropout,
            K_lp=K_lp,
            K_hp=K_hp,
            spatial_layers=1,
            spatial_dropout=dropout,
        )
        fused_dim = self.local_encoder.output_dim + self.graph_encoder.output_dim
        self.base_head = nn.Sequential(
            nn.Linear(fused_dim, temporal_hidden_dim),
            nn.ReLU(),
            nn.Linear(temporal_hidden_dim, horizon),
        )
        self.anchor_state_head = nn.Sequential(
            nn.Linear(fused_dim, temporal_hidden_dim),
            nn.ReLU(),
            nn.Linear(temporal_hidden_dim, input_state_dim),
        )
        self.anchor_delta_head = nn.Sequential(
            nn.Linear(fused_dim, temporal_hidden_dim),
            nn.ReLU(),
            nn.Linear(temporal_hidden_dim, 1),
        )
        self.backbone = DSSGNN(
            input_dim=fused_dim,
            hidden_dim=spatial_hidden_dim,
            out_dim=horizon,
            num_layers=layers,
            K_lp=K_lp,
            K_hp=K_hp,
            P=P,
            dropout=dropout,
            activation=True,
            use_random_gates=use_random_gates,
            P_gate=P_gate,
            S=S,
            shared_input_lift=shared_input_lift,
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.phase_step_scale = 0.5

    def _context_states(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, : self.sequence_dim].reshape(x.shape[0], self.context, self.input_state_dim)

    def _observation_flag(self, x: torch.Tensor) -> torch.Tensor:
        if self.extra_static_dim <= 0:
            return x.new_ones(x.shape[0], 1)
        return x[:, self.sequence_dim : self.sequence_dim + 1].clamp(0.0, 1.0)

    def _encode(self, L_rescaled, x: torch.Tensor) -> torch.Tensor:
        local_encoded = self.local_encoder(x)
        graph_encoded = self.graph_encoder(L_rescaled, x)
        obs_flag = self._observation_flag(x)
        # Hidden nodes have zeroed local histories, so rely on graph context instead of
        # concatenating a nearly constant local-GRU state that encourages mean collapse.
        blended_encoded = obs_flag * local_encoded + (1.0 - obs_flag) * graph_encoded
        return torch.cat([blended_encoded, graph_encoded], dim=-1)

    def _last_theta(self, x: torch.Tensor) -> torch.Tensor:
        seq = self._context_states(x)
        last_state = seq[:, -1]
        return torch.atan2(last_state[:, 0], last_state[:, 1])

    def _last_delta(self, x: torch.Tensor) -> torch.Tensor:
        seq = self._context_states(x)
        if self.context < 2:
            return x.new_zeros(x.shape[0])
        prev_state = seq[:, -2]
        last_state = seq[:, -1]
        prev_theta = torch.atan2(prev_state[:, 0], prev_state[:, 1])
        last_theta = torch.atan2(last_state[:, 0], last_state[:, 1])
        raw_delta = last_theta - prev_theta
        return torch.atan2(torch.sin(raw_delta), torch.cos(raw_delta))

    def _inferred_anchor(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        anchor_state = self.anchor_state_head(encoded)
        anchor_state = anchor_state / anchor_state.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        anchor_delta = self.phase_step_scale * torch.tanh(self.anchor_delta_head(encoded)).squeeze(-1)
        return anchor_state, anchor_delta

    def _blended_last_state(self, x: torch.Tensor, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        obs_flag = self._observation_flag(x).squeeze(-1)
        raw_state = self._context_states(x)[:, -1]
        raw_delta = self._last_delta(x)
        anchor_state, anchor_delta = self._inferred_anchor(encoded)
        last_state = obs_flag.unsqueeze(-1) * raw_state + (1.0 - obs_flag).unsqueeze(-1) * anchor_state
        last_theta = torch.atan2(last_state[:, 0], last_state[:, 1])
        last_delta = obs_flag * raw_delta + (1.0 - obs_flag) * anchor_delta
        return last_theta, last_delta

    def _decode_phase_samples(self, last_theta: torch.Tensor, delta_samples: torch.Tensor) -> torch.Tensor:
        theta_steps = last_theta.unsqueeze(0).unsqueeze(-1) + torch.cumsum(delta_samples, dim=-1)
        sin_part = torch.sin(theta_steps)
        cos_part = torch.cos(theta_steps)
        outputs = torch.stack([sin_part, cos_part], dim=-1)
        return outputs.reshape(outputs.shape[0], outputs.shape[1], self.horizon * self.input_state_dim)

    def forward(self, L_rescaled, x: torch.Tensor, return_uncertainty: bool = False, **kwargs):
        stats = self.forward_with_field_stats(L_rescaled, x)
        mean = stats["mean"]
        if return_uncertainty:
            uncertainty = stats["predictive_var"].sum(dim=-1)
            return mean, uncertainty
        return mean

    def forward_with_field_stats(self, L_rescaled, x: torch.Tensor, eps: float = 1e-12):
        encoded = self._encode(L_rescaled, x)
        last_theta, last_delta = self._blended_last_state(x, encoded)
        last_delta = last_delta.unsqueeze(-1)
        base_delta = torch.tanh(self.base_head(encoded))
        residual_stats = self.backbone.forward_with_field_stats(L_rescaled, encoded, eps=eps)
        residual_delta = torch.tanh(residual_stats["sample_outputs"])
        delta_samples = last_delta.unsqueeze(0) + self.phase_step_scale * (
            base_delta.unsqueeze(0) + self.residual_scale * residual_delta
        )
        sample_outputs = self._decode_phase_samples(last_theta, delta_samples)
        stats = _field_stats_from_sample_outputs(sample_outputs, residual_stats["weights"], eps=eps)
        stats["mean"] = stats["predictive_mean"]
        return stats


class TemporalSwingDSSForecaster(nn.Module):
    """Swing-dynamics forecaster that rolls forward phase and frequency instead of regressing raw channels."""

    def __init__(
        self,
        input_state_dim: int,
        context: int,
        temporal_hidden_dim: int,
        spatial_hidden_dim: int,
        horizon: int,
        extra_static_dim: int = 0,
        temporal_layers: int = 1,
        temporal_dropout: float = 0.0,
        K_lp: int = 3,
        K_hp: int = 3,
        P: int = 0,
        layers: int = 2,
        dropout=(0.0, 0.0),
        use_random_gates: bool = False,
        P_gate: int = 1,
        S: int = 4,
        shared_input_lift: bool = False,
        time_step: float = 0.03,
    ):
        super().__init__()
        if input_state_dim != 3:
            raise ValueError("TemporalSwingDSSForecaster expects swing-style sin/cos/omega states (input_state_dim=3).")
        self.context = context
        self.input_state_dim = input_state_dim
        self.horizon = horizon
        self.extra_static_dim = extra_static_dim
        self.sequence_dim = context * input_state_dim
        self.time_step = time_step
        self.local_encoder = TemporalNodeEncoder(
            input_state_dim=input_state_dim,
            context=context,
            hidden_dim=temporal_hidden_dim,
            extra_static_dim=extra_static_dim,
            num_layers=temporal_layers,
            dropout=temporal_dropout,
        )
        self.graph_encoder = TemporalGraphEncoder(
            input_state_dim=input_state_dim,
            context=context,
            hidden_dim=temporal_hidden_dim,
            extra_static_dim=extra_static_dim,
            num_layers=temporal_layers,
            dropout=temporal_dropout,
            K_lp=K_lp,
            K_hp=K_hp,
            spatial_layers=1,
            spatial_dropout=dropout,
        )
        fused_dim = self.local_encoder.output_dim + self.graph_encoder.output_dim
        self.base_head = nn.Sequential(
            nn.Linear(fused_dim, temporal_hidden_dim),
            nn.ReLU(),
            nn.Linear(temporal_hidden_dim, 2 * horizon),
        )
        self.anchor_state_head = nn.Sequential(
            nn.Linear(fused_dim, temporal_hidden_dim),
            nn.ReLU(),
            nn.Linear(temporal_hidden_dim, input_state_dim),
        )
        self.anchor_domega_head = nn.Sequential(
            nn.Linear(fused_dim, temporal_hidden_dim),
            nn.ReLU(),
            nn.Linear(temporal_hidden_dim, 1),
        )
        self.backbone = DSSGNN(
            input_dim=fused_dim,
            hidden_dim=spatial_hidden_dim,
            out_dim=2 * horizon,
            num_layers=layers,
            K_lp=K_lp,
            K_hp=K_hp,
            P=P,
            dropout=dropout,
            activation=True,
            use_random_gates=use_random_gates,
            P_gate=P_gate,
            S=S,
            shared_input_lift=shared_input_lift,
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.omega_step_scale = 0.25
        self.phase_correction_scale = 0.12
        self.anchor_omega_scale = 0.8

    def _context_states(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, : self.sequence_dim].reshape(x.shape[0], self.context, self.input_state_dim)

    def _observation_flag(self, x: torch.Tensor) -> torch.Tensor:
        if self.extra_static_dim <= 0:
            return x.new_ones(x.shape[0], 1)
        return x[:, self.sequence_dim : self.sequence_dim + 1].clamp(0.0, 1.0)

    def _encode(self, L_rescaled, x: torch.Tensor) -> torch.Tensor:
        local_encoded = self.local_encoder(x)
        graph_encoded = self.graph_encoder(L_rescaled, x)
        obs_flag = self._observation_flag(x)
        # For masked buses, graph features are the only informative temporal signal.
        blended_encoded = obs_flag * local_encoded + (1.0 - obs_flag) * graph_encoded
        return torch.cat([blended_encoded, graph_encoded], dim=-1)

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        sincos = state[:, :2]
        sincos = sincos / sincos.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        omega = self.anchor_omega_scale * torch.tanh(state[:, 2:3])
        return torch.cat([sincos, omega], dim=-1)

    def _last_theta(self, x: torch.Tensor) -> torch.Tensor:
        seq = self._context_states(x)
        last_state = seq[:, -1]
        return torch.atan2(last_state[:, 0], last_state[:, 1])

    def _last_omega(self, x: torch.Tensor) -> torch.Tensor:
        return self._context_states(x)[:, -1, 2]

    def _last_domega(self, x: torch.Tensor) -> torch.Tensor:
        seq = self._context_states(x)
        if self.context < 2:
            return x.new_zeros(x.shape[0])
        return seq[:, -1, 2] - seq[:, -2, 2]

    def _inferred_anchor(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        anchor_state = self._normalize_state(self.anchor_state_head(encoded))
        anchor_domega = self.omega_step_scale * torch.tanh(self.anchor_domega_head(encoded)).squeeze(-1)
        return anchor_state, anchor_domega

    def _blended_last_state(self, x: torch.Tensor, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_flag = self._observation_flag(x).squeeze(-1)
        raw_state = self._context_states(x)[:, -1]
        raw_domega = self._last_domega(x)
        anchor_state, anchor_domega = self._inferred_anchor(encoded)
        last_state = obs_flag.unsqueeze(-1) * raw_state + (1.0 - obs_flag).unsqueeze(-1) * anchor_state
        last_theta = torch.atan2(last_state[:, 0], last_state[:, 1])
        last_omega = last_state[:, 2]
        last_domega = obs_flag * raw_domega + (1.0 - obs_flag) * anchor_domega
        return last_theta, last_omega, last_domega

    def _decode_swing_samples(
        self,
        last_theta: torch.Tensor,
        last_omega: torch.Tensor,
        last_domega: torch.Tensor,
        control_samples: torch.Tensor,
    ) -> torch.Tensor:
        control_samples = control_samples.reshape(control_samples.shape[0], control_samples.shape[1], self.horizon, 2)
        domega_updates = last_domega.unsqueeze(0).unsqueeze(-1) + self.omega_step_scale * torch.tanh(control_samples[..., 0])
        omega_steps = last_omega.unsqueeze(0).unsqueeze(-1) + torch.cumsum(domega_updates, dim=-1)
        phase_corrections = self.phase_correction_scale * torch.tanh(control_samples[..., 1])
        theta_steps = last_theta.unsqueeze(0).unsqueeze(-1) + torch.cumsum(self.time_step * omega_steps + phase_corrections, dim=-1)
        sin_part = torch.sin(theta_steps)
        cos_part = torch.cos(theta_steps)
        outputs = torch.stack([sin_part, cos_part, omega_steps], dim=-1)
        return outputs.reshape(outputs.shape[0], outputs.shape[1], self.horizon * self.input_state_dim)

    def forward(self, L_rescaled, x: torch.Tensor, return_uncertainty: bool = False, **kwargs):
        stats = self.forward_with_field_stats(L_rescaled, x)
        mean = stats["mean"]
        if return_uncertainty:
            uncertainty = stats["predictive_var"].sum(dim=-1)
            return mean, uncertainty
        return mean

    def forward_with_field_stats(self, L_rescaled, x: torch.Tensor, eps: float = 1e-12):
        encoded = self._encode(L_rescaled, x)
        last_theta, last_omega, last_domega = self._blended_last_state(x, encoded)
        base_controls = self.base_head(encoded).reshape(encoded.shape[0], self.horizon, 2)
        residual_stats = self.backbone.forward_with_field_stats(L_rescaled, encoded, eps=eps)
        residual_controls = residual_stats["sample_outputs"].reshape(residual_stats["sample_outputs"].shape[0], encoded.shape[0], self.horizon, 2)
        control_samples = base_controls.unsqueeze(0) + self.residual_scale * residual_controls
        sample_outputs = self._decode_swing_samples(last_theta, last_omega, last_domega, control_samples)
        stats = _field_stats_from_sample_outputs(sample_outputs, residual_stats["weights"], eps=eps)
        stats["mean"] = stats["predictive_mean"]
        return stats


class TemporalCoupledSwingDSSForecaster(TemporalSwingDSSForecaster):
    """Swing forecaster with graph-coupled rollout in the decoder."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coupling_gain = nn.Parameter(torch.tensor(1.4, dtype=torch.float32))
        self.damping_gain = nn.Parameter(torch.tensor(0.7, dtype=torch.float32))
        self.control_gain = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.residual_domega_scale = 0.2

    def _normalized_adjacency_apply(self, L_rescaled, x: torch.Tensor) -> torch.Tensor:
        A_norm = (-L_rescaled).coalesce()
        if A_norm.device != x.device:
            A_norm = A_norm.to(x.device)
        if x.dim() == 2:
            return torch.sparse.mm(A_norm.float(), x.float()).to(x.dtype)
        if x.dim() != 3:
            raise ValueError("Expected tensor with shape (S, N, C) or (N, C).")
        outputs = []
        for sample in x:
            outputs.append(torch.sparse.mm(A_norm.float(), sample.float()).to(sample.dtype))
        return torch.stack(outputs, dim=0)

    def _decode_coupled_swing_samples(
        self,
        L_rescaled,
        last_theta: torch.Tensor,
        last_omega: torch.Tensor,
        last_domega: torch.Tensor,
        control_samples: torch.Tensor,
    ) -> torch.Tensor:
        control_samples = control_samples.reshape(control_samples.shape[0], control_samples.shape[1], self.horizon, 2)
        theta = last_theta.unsqueeze(0)
        omega = last_omega.unsqueeze(0)
        domega_bias = last_domega.unsqueeze(0)
        coupling_gain = F.softplus(self.coupling_gain)
        damping_gain = F.softplus(self.damping_gain)
        control_gain = F.softplus(self.control_gain)

        outputs = []
        for step_idx in range(self.horizon):
            sin_theta = torch.sin(theta)
            cos_theta = torch.cos(theta)
            nbr_sin = self._normalized_adjacency_apply(L_rescaled, sin_theta.unsqueeze(-1)).squeeze(-1)
            nbr_cos = self._normalized_adjacency_apply(L_rescaled, cos_theta.unsqueeze(-1)).squeeze(-1)
            electrical = coupling_gain * (cos_theta * nbr_sin - sin_theta * nbr_cos)

            step_controls = control_samples[..., step_idx, :]
            control_drive = control_gain * torch.tanh(step_controls[..., 0])
            residual_drive = self.residual_domega_scale * torch.tanh(step_controls[..., 1])
            if step_idx == 0:
                residual_drive = residual_drive + 0.25 * domega_bias

            domega = control_drive - damping_gain * omega + electrical + residual_drive
            omega = omega + self.time_step * domega
            theta = theta + self.time_step * omega
            outputs.append(torch.stack([torch.sin(theta), torch.cos(theta), omega], dim=-1))

        stacked = torch.stack(outputs, dim=2)
        return stacked.reshape(stacked.shape[0], stacked.shape[1], self.horizon * self.input_state_dim)

    def forward_with_field_stats(self, L_rescaled, x: torch.Tensor, eps: float = 1e-12):
        encoded = self._encode(L_rescaled, x)
        last_theta, last_omega, last_domega = self._blended_last_state(x, encoded)
        base_controls = self.base_head(encoded).reshape(encoded.shape[0], self.horizon, 2)
        residual_stats = self.backbone.forward_with_field_stats(L_rescaled, encoded, eps=eps)
        residual_controls = residual_stats["sample_outputs"].reshape(
            residual_stats["sample_outputs"].shape[0],
            encoded.shape[0],
            self.horizon,
            2,
        )
        control_samples = base_controls.unsqueeze(0) + self.residual_scale * residual_controls
        sample_outputs = self._decode_coupled_swing_samples(
            L_rescaled,
            last_theta,
            last_omega,
            last_domega,
            control_samples,
        )
        stats = _field_stats_from_sample_outputs(sample_outputs, residual_stats["weights"], eps=eps)
        stats["mean"] = stats["predictive_mean"]
        return stats
