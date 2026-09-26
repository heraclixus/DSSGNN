"""
DSS-GNN with GDUQ-style anchor mechanism for calibration experiments.

Strategy A (Direct Port): Input anchoring with Monte Carlo at inference.
See docs/GDUQ_DSGNN_INTEGRATION.md for design and principled integration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import DSSGNN, DSSGNNTFE, DSSGNNTFEPropFirst


class DSSGNNWithGDUQAnchor(nn.Module):
    """
    DSS-GNN wrapped with GDUQ-style input anchoring.

    - Anchor distribution: N(mean, std) from train features
    - Input: concat(x - ξ, ξ) when anchor ξ is used
    - Inner model's first layer accepts 2*input_dim
    - At inference: n_anchors forward passes → mean logits (and std if return_std)
    """

    def __init__(
        self,
        base_model_class,
        base_model_kwargs,
        mean,
        std,
        anchor_type="node",
    ):
        super().__init__()
        self.anchor_type = anchor_type
        self.mean = mean
        self.std = std
        std_safe = std.clone()
        std_safe[std_safe == 0] = 1e-3
        self.anchor_dist = torch.distributions.Normal(loc=mean, scale=std_safe)

        # Inner model expects 2x input (concat of x-ξ and ξ)
        kwargs = dict(base_model_kwargs)
        orig_input_dim = kwargs["input_dim"]
        kwargs["input_dim"] = 2 * orig_input_dim
        self.base_model = base_model_class(**kwargs)
        self.orig_input_dim = orig_input_dim

    def process_batch(self, x, anchors=None):
        """
        Form GDUQ-style input: concat(x - ξ, ξ).
        x: (N, d), anchors: (N, d) or (1, d) or None
        """
        N = x.shape[0]
        if anchors is None:
            anchors = self.anchor_dist.sample([N]).to(x.device)
        else:
            # anchors: (n_anchors, N, d) or (N, d) — need to broadcast
            if anchors.dim() == 3:
                raise ValueError("process_batch expects per-call anchors, not batched")
            if anchors.shape[0] == 1:
                anchors = anchors.expand(N, -1)
            elif anchors.shape[0] != N:
                anchors = anchors.repeat_interleave(N // anchors.shape[0], dim=0)
        return torch.cat([x - anchors, anchors], dim=1)

    def _forward_base(self, graph_input, new_x, return_uncertainty=False, uncertainty_type="chaos"):
        """Single forward through base model."""
        if isinstance(graph_input, tuple):
            return self.base_model(
                *graph_input,
                new_x,
                return_uncertainty=return_uncertainty,
                uncertainty_type=uncertainty_type,
            )
        return self.base_model(
            graph_input,
            new_x,
            return_uncertainty=return_uncertainty,
            uncertainty_type=uncertainty_type,
        )

    def forward(
        self,
        graph_input,
        x,
        anchors=None,
        n_anchors=1,
        return_std=False,
        return_uncertainty=False,
        uncertainty_type="chaos",
    ):
        """
        Args:
            graph_input: L_rescaled (DSSGNN) or (adj_lp, adj_hp) (TFE variants)
            x: node features (N, input_dim)
            anchors: optional pre-sampled anchors for inference
            n_anchors: number of anchor samples (MC)
            return_std: if True, return (logits_mean, logits_std)
            return_uncertainty: if True, also return chaos uncertainty from base model

        Returns:
            logits: (N, out_dim) — mean over anchors
            optional: std, uncertainty
        """
        if n_anchors == 1 and return_std:
            n_anchors = 5  # need >1 for std

        preds = []
        for _ in range(n_anchors):
            new_x = self.process_batch(x, anchors)
            out = self._forward_base(
                graph_input,
                new_x,
                return_uncertainty=False,
                uncertainty_type=uncertainty_type,
            )
            preds.append(out)

        preds = torch.stack(preds, dim=0)
        mu = preds.mean(dim=0)

        result = [mu]
        if return_std:
            result.append(preds.std(dim=0))
        if return_uncertainty:
            new_x = self.process_batch(x, anchors)
            _, unc = self._forward_base(
                graph_input,
                new_x,
                return_uncertainty=True,
                uncertainty_type=uncertainty_type,
            )
            result.append(unc)

        if len(result) == 1:
            return result[0]
        return tuple(result)

    def calibrate(self, mu, std):
        """GDUQ-style calibration: mu_hat = mu / (1 + exp(c)), c = mean(std)."""
        c = std.mean(dim=1, keepdim=True).expand_as(mu)
        return mu / (1 + torch.exp(c))


def build_dssgnn_gduq(
    base_model_class,
    features,
    train_mask,
    num_classes,
    **model_kwargs,
):
    """
    Build DSS-GNN + GDUQ anchor model with anchor dist from train features.

    Returns model, graph_input_factory (callable that takes adj and returns graph_input).
    """
    train_x = features[train_mask]
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0)
    std[std == 0] = 1e-3

    model = DSSGNNWithGDUQAnchor(
        base_model_class=base_model_class,
        base_model_kwargs=dict(
            input_dim=features.shape[1],
            out_dim=num_classes,
            **model_kwargs,
        ),
        mean=mean,
        std=std,
        anchor_type="node",
    )
    return model
