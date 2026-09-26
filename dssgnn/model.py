"""
DSS-GNN: Doubly-spectral stochastic graph neural network model.

Implements method.tex (Algorithm 1, Eq. 3.35, 3.105-3.120, 3.147-3.154):
- Input lift: H_n^(0) = X W_in^(n) (Eq. 3.35)
- DSS layer: filter, sample at quadrature, sigma, project (Eq. 3.105-3.120)
- Readout: Z_n = H_n^(L) W_out, mean=Z_0, V_chaos = sum_{n>=1} ||Z_n||^2 (Eq. 3.147-3.154)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .chaos import quadrature_sample_logits
from .dsconv_nonintrusive import DSSConvNonIntrusive
from .dsconv_tfe import DSConvTFE
from .dsconv_tfe_random_gates import DSConvTFERandomGates
from .dsconv_tfe_random_filter import DSConvTFERandomFilter
from .dsconv_tfe_propfirst import DSSGNNTFEPropFirst


def _nodewise_chaos_energy(H_list):
    chaos = torch.zeros(H_list[0].shape[0], device=H_list[0].device, dtype=H_list[0].dtype)
    for h in H_list[1:]:
        chaos = chaos + (h ** 2).sum(dim=1)
    return chaos


def _nodewise_mean_energy(H_list):
    return (H_list[0] ** 2).sum(dim=1)


def _predictive_stats_from_sample_logits(sample_logits, weights, eps=1e-12):
    """Compute predictive probabilities, entropy, and mutual information from weighted sample logits."""
    sample_probs = torch.softmax(sample_logits, dim=-1)
    weight_probs = weights.view(-1, 1, 1)
    predictive_probs = (weight_probs * sample_probs).sum(dim=0)
    predictive_probs = predictive_probs.clamp_min(eps)
    predictive_probs = predictive_probs / predictive_probs.sum(dim=-1, keepdim=True)

    predictive_entropy = -(predictive_probs * predictive_probs.log()).sum(dim=-1)
    sample_entropy = -(sample_probs.clamp_min(eps) * sample_probs.clamp_min(eps).log()).sum(dim=-1)
    expected_entropy = (weights.view(-1, 1) * sample_entropy).sum(dim=0)
    mutual_info = (predictive_entropy - expected_entropy).clamp_min(0.0)

    return {
        "sample_logits": sample_logits,
        "weights": weights,
        "predictive_probs": predictive_probs,
        "predictive_entropy": predictive_entropy,
        "expected_entropy": expected_entropy,
        "mutual_info": mutual_info,
    }


def _field_stats_from_sample_outputs(sample_outputs, weights, eps=1e-12):
    """Compute mean/variance statistics for continuous-valued stochastic outputs."""
    weight_view = weights.view(-1, 1, 1)
    predictive_mean = (weight_view * sample_outputs).sum(dim=0)
    centered = sample_outputs - predictive_mean.unsqueeze(0)
    predictive_var = (weight_view * centered.pow(2)).sum(dim=0).clamp_min(0.0)
    predictive_std = predictive_var.add(eps).sqrt()
    total_variance = predictive_var.sum(dim=-1)
    return {
        "sample_outputs": sample_outputs,
        "weights": weights,
        "predictive_mean": predictive_mean,
        "predictive_var": predictive_var,
        "predictive_std": predictive_std,
        "total_variance": total_variance,
    }


def _summarize_uncertainty(layer_states, uncertainty_type="chaos", eps=1e-8):
    if not layer_states:
        raise ValueError("layer_states must contain at least one coefficient stack.")

    final_state = layer_states[-1]
    final_chaos = _nodewise_chaos_energy(final_state)

    if uncertainty_type == "chaos":
        return final_chaos
    if uncertainty_type == "chaos_norm":
        return final_chaos / (_nodewise_mean_energy(final_state) + eps)
    if uncertainty_type == "chaos_ratio":
        mean_energy = _nodewise_mean_energy(final_state)
        return final_chaos / (final_chaos + mean_energy + eps)
    if uncertainty_type == "chaos_layernorm":
        total = torch.zeros_like(final_chaos)
        for state in layer_states:
            total = total + _nodewise_chaos_energy(state) / (_nodewise_mean_energy(state) + eps)
        return total / float(len(layer_states))
    if uncertainty_type == "chaos_layerratio":
        total = torch.zeros_like(final_chaos)
        for state in layer_states:
            chaos = _nodewise_chaos_energy(state)
            mean_energy = _nodewise_mean_energy(state)
            total = total + chaos / (chaos + mean_energy + eps)
        return total / float(len(layer_states))
    raise ValueError(f"Unknown uncertainty_type: {uncertainty_type}")


class DSSGNN(nn.Module):
    """
    Doubly-spectral stochastic GNN for node classification.

    Forward returns (logits, uncertainty) where:
    - logits: predictive mean from order-0 channel
    - uncertainty: variance proxy from higher-order channels (optional)
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        out_dim,
        num_layers,
        K_lp,
        K_hp,
        P,
        dropout=(0.5, 0.0),
        activation=True,
        use_random_gates=False,
        use_random_filter=False,
        P_gate=1,
        P_filter=1,
        S=4,
        shared_input_lift=False,
        act_fn="relu",
        use_bn=False,
    ):
        """
        Args:
            input_dim: node feature dimension
            hidden_dim: hidden dimension
            out_dim: number of classes
            num_layers: number of DSS layers
            K_lp, K_hp: Chebyshev degrees (Eq. 3.71-3.74)
            P: chaos truncation order (P+1 channels)
            dropout: (propagation_dropout, linear_dropout)
            activation: apply pointwise activation after each layer
            use_random_gates: S+G variant, stochastic alpha_n(omega), beta_n(omega)
            P_gate: chaos order for gate expansions
            use_random_filter: deprecated, not supported (use TFE variant for S+F)
            P_filter: deprecated
            S: quadrature nodes
            act_fn: activation function ("relu", "gelu", or "tanh")
            use_bn: per-chaos-channel BatchNorm1d between hidden DSS layers
                (opt-in; default False preserves the published architecture)
        """
        if use_random_filter:
            raise NotImplementedError(
                "S+F (use_random_filter) is not supported for DSSGNN (Laplacian backbone). "
                "Use TFE variant (--propagate_first) for S+F."
            )
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.K_lp = K_lp
        self.K_hp = K_hp
        self.P = P
        self.dropout = dropout
        self.activation = activation
        self.use_random_gates = use_random_gates
        self.shared_input_lift = shared_input_lift

        # Input lift: H_n^(0) = X W_in^(n) (Eq. 3.35)
        if shared_input_lift:
            self.W_in_base = nn.Linear(input_dim, hidden_dim, bias=False)
            self.input_gains = nn.Parameter(torch.zeros(P + 1))
            self.W_in = None
        else:
            self.W_in = nn.ModuleList([
                nn.Linear(input_dim, hidden_dim, bias=False)
                for _ in range(P + 1)
            ])
            self.W_in_base = None
            self.input_gains = None

        # Algorithm 1: non-intrusive projection (method.tex)
        ConvClass = DSSConvNonIntrusive
        conv_kwargs = {"P_gate": P_gate, "S": S, "use_random_gates": use_random_gates, "act_fn": act_fn}

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_d = hidden_dim if i == 0 else hidden_dim
            out_d = hidden_dim if i < num_layers - 1 else out_dim
            act = activation and (i < num_layers - 1)
            self.convs.append(
                ConvClass(in_d, out_d, K_lp, K_hp, P, activation=act, **conv_kwargs)
            )

        self.use_bn = use_bn
        if use_bn:
            self.bns = nn.ModuleList([
                nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(P + 1)])
                for _ in range(num_layers - 1)
            ])
        else:
            self.bns = None

        self.reset_parameters()

    def reset_parameters(self):
        if self.shared_input_lift:
            stdv = 1.0 / math.sqrt(self.W_in_base.weight.size(1))
            self.W_in_base.weight.data.normal_(-stdv, stdv)
            with torch.no_grad():
                self.input_gains.zero_()
                if self.input_gains.numel() > 0:
                    self.input_gains[0] = 1.0
                if self.input_gains.numel() > 1:
                    self.input_gains[1:] = 0.1
        else:
            for linear in self.W_in:
                stdv = 1.0 / math.sqrt(linear.weight.size(1))
                linear.weight.data.normal_(-stdv, stdv)
        for conv in self.convs:
            if hasattr(conv, "reset_parameters"):
                conv.reset_parameters()
        if getattr(self, "bns", None) is not None:
            for layer_bns in self.bns:
                for bn in layer_bns:
                    bn.reset_parameters()

    def _init_parameters(self):
        self.reset_parameters()

    def _forward_coefficients(self, L_rescaled, x):
        """
        Run the DSS stack and return the final coefficient stack plus intermediate layer states.
        """
        if self.shared_input_lift:
            base = self.W_in_base(x)
            H_list = [self.input_gains[n] * base for n in range(self.P + 1)]
        elif self.W_in is not None:
            H_list = [self.W_in[n](x) for n in range(self.P + 1)]
        else:
            H_list = [x for _ in range(self.P + 1)]

        if self.dropout[0] > 0 and self.training:
            H_list = [F.dropout(h, self.dropout[0], training=True) for h in H_list]

        layer_states = []
        for i, conv in enumerate(self.convs):
            H_list = conv(L_rescaled, H_list)
            if self.use_bn and i < self.num_layers - 1:
                H_list = [self.bns[i][n](h) for n, h in enumerate(H_list)]
            layer_states.append(list(H_list))
            if i < self.num_layers - 1 and self.dropout[1] > 0 and self.training:
                H_list = [F.dropout(h, self.dropout[1], training=True) for h in H_list]
        return H_list, layer_states

    def _sample_logits_from_coefficients(self, final_state):
        """Reconstruct quadrature-node logits from the final chaos coefficient stack."""
        ref_conv = self.convs[-1]
        device = final_state[0].device
        dtype = final_state[0].dtype
        psi = ref_conv._get_psi(device).to(dtype)[:, : self.P + 1]
        coeffs = torch.stack(final_state, dim=0)  # (P+1, N, C)
        sample_logits = torch.einsum("sp,pnc->snc", psi, coeffs)
        weights = ref_conv._w.to(device=device, dtype=dtype)
        return sample_logits, weights

    def forward(self, L_rescaled, x, return_uncertainty=False, uncertainty_type="chaos", uncertainty_eps=1e-8):
        """
        Args:
            L_rescaled: rescaled Laplacian (N, N) sparse
            x: node features (N, input_dim)
            return_uncertainty: if True, also return variance proxy (Eq. 3.150)

        Returns:
            logits: (N, out_dim) = Z_0 = E[Y]
            uncertainty: (N,) optional, V_chaos = sum_{n>=1} ||Z_n||^2
        """
        final_state, layer_states = self._forward_coefficients(L_rescaled, x)
        Z_0 = final_state[0]
        logits = Z_0

        if return_uncertainty:
            uncertainty = _summarize_uncertainty(
                layer_states if layer_states else [final_state],
                uncertainty_type=uncertainty_type,
                eps=uncertainty_eps,
            )
            return logits, uncertainty

        return logits

    def forward_with_sample_logits(self, L_rescaled, x):
        """Return mean logits, sample logits, quadrature weights, and chaos energy.

        Used for the quadrature-averaged training loss.
        Returns:
            logits: (N, C) mean logits Z_0
            sample_logits: (S, N, C) quadrature-node logits
            weights: (S,) quadrature weights
            chaos_energy: (N,) per-node chaos energy
        """
        final_state, _ = self._forward_coefficients(L_rescaled, x)
        logits = final_state[0]
        sample_logits, weights = self._sample_logits_from_coefficients(final_state)
        if len(final_state) > 1:
            chaos_energy = sum(h.pow(2).sum(dim=-1) for h in final_state[1:])
        else:
            chaos_energy = logits.new_zeros(logits.shape[0])
        return logits, sample_logits, weights, chaos_energy

    def forward_with_predictive_stats(self, L_rescaled, x, eps=1e-12):
        """Return predictive uncertainty statistics from quadrature-reconstructed sample logits."""
        final_state, _ = self._forward_coefficients(L_rescaled, x)
        logits = final_state[0]
        sample_logits, weights = self._sample_logits_from_coefficients(final_state)
        stats = _predictive_stats_from_sample_logits(sample_logits, weights, eps=eps)
        stats["logits"] = logits
        return stats

    def forward_with_field_stats(self, L_rescaled, x, eps=1e-12):
        """Return mean/variance statistics for continuous stochastic outputs."""
        final_state, _ = self._forward_coefficients(L_rescaled, x)
        mean = final_state[0]
        sample_outputs, weights = self._sample_logits_from_coefficients(final_state)
        stats = _field_stats_from_sample_outputs(sample_outputs, weights, eps=eps)
        stats["mean"] = mean
        return stats


class DSSGNNTFE(nn.Module):
    """
    DSS-GNN with TFE-style adjacency propagation (aligns with TFE-GNN on heterophilous graphs).

    Uses adj_lp and adj_hp from propagate_adj instead of Chebyshev on Laplacian.
    Same chaos-channel structure, different graph spectral basis.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        out_dim,
        num_layers,
        K_lp,
        K_hp,
        P,
        dropout=(0.5, 0.0),
        activation=True,
        use_random_gates=False,
        use_random_filter=False,
        P_gate=1,
        P_filter=1,
        shared_input_lift=False,
        use_bn=False,
    ):
        super().__init__()
        self.P = P
        self.dropout = dropout
        self.use_random_gates = use_random_gates
        self.use_random_filter = use_random_filter
        self.shared_input_lift = shared_input_lift

        if shared_input_lift:
            self.W_in_base = nn.Linear(input_dim, hidden_dim, bias=False)
            self.input_gains = nn.Parameter(torch.zeros(P + 1))
            first_in_dim = hidden_dim
        else:
            self.W_in_base = None
            self.input_gains = None
            first_in_dim = input_dim

        if use_random_gates:
            ConvClass = DSConvTFERandomGates
            conv_kwargs = {"P_gate": P_gate}
        elif use_random_filter:
            ConvClass = DSConvTFERandomFilter
            conv_kwargs = {"P_filter": P_filter}
        else:
            ConvClass = DSConvTFE
            conv_kwargs = {}

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_d = first_in_dim if i == 0 else hidden_dim
            out_d = hidden_dim if i < num_layers - 1 else out_dim
            act = activation and (i < num_layers - 1)
            self.convs.append(
                ConvClass(in_d, out_d, K_lp, K_hp, P, activation=act, **conv_kwargs)
            )

        # Per-chaos-channel BatchNorm1d between hidden layers (opt-in; default
        # False preserves the published architecture). Mirrors DSSGNN.use_bn.
        self.use_bn = use_bn
        if use_bn:
            self.bns = nn.ModuleList([
                nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(P + 1)])
                for _ in range(num_layers - 1)
            ])
        else:
            self.bns = None

        self._init_parameters()

    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                stdv = 1.0 / math.sqrt(m.weight.size(1))
                m.weight.data.normal_(-stdv, stdv)
        if self.shared_input_lift:
            with torch.no_grad():
                self.input_gains.zero_()
                if self.input_gains.numel() > 0:
                    self.input_gains[0] = 1.0
                if self.input_gains.numel() > 1:
                    self.input_gains[1:] = 0.1

    def _forward_coefficients(self, adj_lp, adj_hp, x):
        if self.shared_input_lift:
            base = self.W_in_base(x)
            H_list = [self.input_gains[n] * base for n in range(self.P + 1)]
        else:
            H_list = [x for _ in range(self.P + 1)]

        if self.dropout[0] > 0 and self.training:
            H_list = [F.dropout(h, self.dropout[0], training=True) for h in H_list]

        layer_states = []
        for i, conv in enumerate(self.convs):
            H_list = conv(adj_lp, adj_hp, H_list)
            if self.use_bn and i < len(self.convs) - 1:
                H_list = [self.bns[i][n](h) for n, h in enumerate(H_list)]
            layer_states.append(list(H_list))
            if i < len(self.convs) - 1 and self.dropout[1] > 0 and self.training:
                H_list = [F.dropout(h, self.dropout[1], training=True) for h in H_list]

        return H_list, layer_states

    def forward(self, adj_lp, adj_hp, x, return_uncertainty=False, uncertainty_type="chaos", uncertainty_eps=1e-8):
        """
        Args:
            adj_lp: low-pass adjacency from propagate_adj
            adj_hp: high-pass adjacency from propagate_adj
            x: node features (N, input_dim)
        """
        H_list, layer_states = self._forward_coefficients(adj_lp, adj_hp, x)

        Z_0 = H_list[0]
        logits = Z_0

        if return_uncertainty:
            uncertainty = _summarize_uncertainty(
                layer_states if layer_states else [H_list],
                uncertainty_type=uncertainty_type,
                eps=uncertainty_eps,
            )
            return logits, uncertainty

        return logits

    def forward_with_sample_logits(self, adj_lp, adj_hp, x, S=4):
        """Return mean logits, quadrature-node logits, weights, and chaos energy.

        The TFE convs carry no quadrature buffers, so nodes/weights are built on
        the fly from the final chaos coefficient stack (Eq. 8 reconstruction).
        Returns:
            logits: (N, C) mean logits Z_0
            sample_logits: (S, N, C) quadrature-node logits
            weights: (S,) quadrature weights
            chaos_energy: (N,) per-node chaos energy
        """
        final_state, _ = self._forward_coefficients(adj_lp, adj_hp, x)
        logits = final_state[0]
        sample_logits, weights = quadrature_sample_logits(final_state, S)
        if len(final_state) > 1:
            chaos_energy = sum(h.pow(2).sum(dim=-1) for h in final_state[1:])
        else:
            chaos_energy = logits.new_zeros(logits.shape[0])
        return logits, sample_logits, weights, chaos_energy
