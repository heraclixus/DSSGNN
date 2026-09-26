"""Lightweight Graph-EBM post-hoc scorer for the GNNSafe pipeline.

This module implements the core post-hoc path from the official Graph-EBM
wrapper in a dependency-light form:

- fit a Gaussian per class on propagated train embeddings
- rescale the Mahalanobis correction into logit space
- evaluate energy on unpropagated embeddings/logits
- optionally diffuse both class evidence and scalar energy over the graph

Reference:
https://github.com/dfuchsgruber/graph-ebm
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _quantile_interval(x: torch.Tensor, quantile: float = 0.95) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-column robust interval used by the official wrapper for rescaling."""
    alpha = (1.0 - quantile) / 2.0
    lo = torch.quantile(x, alpha, dim=0)
    hi = torch.quantile(x, 1.0 - alpha, dim=0)
    return lo, hi


def _row_normalized_self_loop_weights(edge_index: torch.Tensor, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-normalized adjacency with self-loops for label-propagation diffusion."""
    device = edge_index.device
    self_loops = torch.arange(num_nodes, device=device, dtype=edge_index.dtype)
    loop_edges = torch.stack([self_loops, self_loops], dim=0)
    edge_index = torch.cat([edge_index, loop_edges], dim=1)
    row = edge_index[0]
    deg = torch.bincount(row, minlength=num_nodes).to(torch.float32)
    weight = torch.reciprocal(deg.clamp_min(1.0))[row]
    return edge_index, weight


def _label_propagation_diffusion(
    signal: torch.Tensor,
    edge_index: torch.Tensor,
    k: int,
    alpha: float,
) -> torch.Tensor:
    if k <= 0:
        return signal
    num_nodes = signal.size(0)
    edge_index, weight = _row_normalized_self_loop_weights(edge_index, num_nodes)
    src, dst = edge_index
    out = signal
    for _ in range(k):
        messages = out[dst] * weight.unsqueeze(-1)
        aggregated = torch.zeros_like(out)
        aggregated.index_add_(0, src, messages)
        out = alpha * out + (1.0 - alpha) * aggregated
    return out


def _energy(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    return -temperature * torch.logsumexp(logits / temperature, dim=-1)


@dataclass
class _GaussianStats:
    mean: torch.Tensor
    covariance: torch.Tensor


class GraphEBMPosthocScorer:
    """Dependency-light Graph-EBM scorer aligned with the official wrapper defaults."""

    def __init__(
        self,
        covariance_type: str = "diagonal",
        tied_covariance: bool = False,
        gamma_correction: float = 1.0,
        lambda_independent_energy: float = 1.0,
        lambda_local_energy: float = 1.0,
        lambda_group_energy: float = 1.0,
        alpha: float = 0.5,
        num_diffusion_steps: int = 10,
        aggregation: str = "sum",
        temperature: float = 1.0,
        eps: float = 1e-5,
    ) -> None:
        self.covariance_type = covariance_type
        self.tied_covariance = bool(tied_covariance)
        self.gamma_correction = float(gamma_correction)
        self.lambda_independent_energy = float(lambda_independent_energy)
        self.lambda_local_energy = float(lambda_local_energy)
        self.lambda_group_energy = float(lambda_group_energy)
        self.alpha = float(alpha)
        self.num_diffusion_steps = int(num_diffusion_steps)
        self.aggregation = aggregation
        self.temperature = float(temperature)
        self.eps = float(eps)

        self._stats: list[_GaussianStats] = []
        self._class_prior: torch.Tensor | None = None
        self._mahalanobis_scale: tuple[torch.Tensor, torch.Tensor] | None = None
        self._logit_scale: tuple[torch.Tensor, torch.Tensor] | None = None
        self._fitted = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    def _fit_covariance(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - x.mean(dim=0, keepdim=True)
        denom = max(x.size(0) - 1, 1)
        cov = (x_centered.T @ x_centered) / float(denom)
        cov = cov + self.eps * torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
        if self.covariance_type == "full":
            return cov
        if self.covariance_type == "diagonal":
            return torch.diag(torch.diag(cov))
        if self.covariance_type == "identity":
            return torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
        if self.covariance_type == "isotropic":
            scale = torch.diag(cov).mean()
            return scale * torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
        raise ValueError(f"Unsupported covariance_type: {self.covariance_type}")

    def _mahalanobis_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        assert self._stats, "Graph-EBM must be fitted before scoring."
        distances = []
        for stats in self._stats:
            delta = embeddings - stats.mean
            if self.covariance_type == "diagonal":
                inv_diag = torch.reciprocal(torch.diag(stats.covariance).clamp_min(self.eps))
                dist = (delta.square() * inv_diag.unsqueeze(0)).sum(dim=-1)
            else:
                precision = torch.linalg.pinv(stats.covariance)
                dist = torch.sum((delta @ precision) * delta, dim=-1)
            distances.append(dist)
        return torch.stack(distances, dim=1)

    def fit(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        del edge_index  # fitting uses propagated features only
        if y.dim() > 1:
            y = y.squeeze(-1)
        mask = mask.to(torch.bool)
        y = y.to(torch.long)
        train_embeddings = embeddings[mask]
        train_logits = logits[mask]
        train_y = y[mask]
        num_classes = logits.size(1)

        class_counts = torch.bincount(train_y, minlength=num_classes).to(train_logits.dtype)
        self._class_prior = class_counts.clamp_min(1.0)
        self._class_prior = self._class_prior / self._class_prior.sum()

        if self.tied_covariance:
            tied_cov = self._fit_covariance(train_embeddings)
        else:
            tied_cov = None

        self._stats = []
        global_mean = train_embeddings.mean(dim=0)
        for cls in range(num_classes):
            cls_embeddings = train_embeddings[train_y == cls]
            if cls_embeddings.numel() == 0:
                mean = global_mean
                cov = tied_cov if tied_cov is not None else self._fit_covariance(train_embeddings)
            else:
                mean = cls_embeddings.mean(dim=0)
                cov = tied_cov if tied_cov is not None else self._fit_covariance(cls_embeddings)
            self._stats.append(_GaussianStats(mean=mean, covariance=cov))

        mahalanobis = self._mahalanobis_distances(train_embeddings)
        self._mahalanobis_scale = _quantile_interval(mahalanobis, quantile=0.95)
        self._logit_scale = _quantile_interval(train_logits, quantile=0.95)
        self._fitted = True

    def _conditional_evidence(self, logits_unpropagated: torch.Tensor, embeddings_unpropagated: torch.Tensor) -> torch.Tensor:
        mahalanobis = self._mahalanobis_distances(embeddings_unpropagated)
        assert self._mahalanobis_scale is not None and self._logit_scale is not None
        mahal_lo, mahal_hi = self._mahalanobis_scale
        logit_lo, logit_hi = self._logit_scale
        denom = (mahal_hi - mahal_lo).clamp_min(self.eps)
        scaled = (mahalanobis - mahal_lo) / denom
        scaled = scaled * (logit_hi - logit_lo) + logit_lo
        correction = -scaled
        return logits_unpropagated + self.gamma_correction * correction

    def get_uncertainty(
        self,
        logits_unpropagated: torch.Tensor,
        embeddings_unpropagated: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        assert self._fitted, "Graph-EBM must be fitted before scoring."
        conditional_evidence = self._conditional_evidence(logits_unpropagated, embeddings_unpropagated)

        terms = []
        if self.lambda_local_energy > 0:
            diffused_evidence = _label_propagation_diffusion(
                conditional_evidence,
                edge_index,
                k=self.num_diffusion_steps,
                alpha=self.alpha,
            )
            terms.append(self.lambda_local_energy * _energy(diffused_evidence, self.temperature))

        if self.lambda_group_energy > 0:
            raw_energy = _energy(conditional_evidence, self.temperature).unsqueeze(-1)
            diffused_energy = _label_propagation_diffusion(
                raw_energy,
                edge_index,
                k=self.num_diffusion_steps,
                alpha=self.alpha,
            ).squeeze(-1)
            terms.append(self.lambda_group_energy * diffused_energy)

        if self.lambda_independent_energy > 0:
            terms.append(self.lambda_independent_energy * _energy(conditional_evidence, self.temperature))

        if not terms:
            return _energy(conditional_evidence, self.temperature)
        if self.aggregation == "sum":
            return sum(terms)
        if self.aggregation == "logsumexp":
            return torch.logsumexp(torch.stack(terms, dim=0), dim=0)
        raise ValueError(f"Unsupported aggregation: {self.aggregation}")
