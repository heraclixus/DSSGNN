"""
DSS-GNN with TFE's propagate-first structure.

Key TFE insight: propagate ONCE at the input, then stack MLP layers.
This matches TFE-GNN's architecture and improves heterophilous performance.

Flow: x -> [mix_prop(lp), mix_prop(hp)] -> combine per chaos order -> MLP layers.

S+G variant: use_random_gates=True makes alpha_n(omega), beta_n(omega) stochastic.
S+F variant: use_random_filter=True makes c_k^(lp)(omega), c_k^(hp)(omega) stochastic.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dsconv_tfe import mix_prop
from .chaos import eval_chaos_basis, sample_omega, quadrature_sample_logits


class DSSGNNTFEPropFirst(nn.Module):
    """
    DSS-GNN with TFE's propagate-first structure.

    - Propagate ONCE: h_lp = mix_prop(adj_lp, x), h_hp = mix_prop(adj_hp, x)
    - Chaos channels: H[n] = alpha[n]*h_lp + beta[n]*h_hp (order-specific combine)
    - MLP layers: no more propagation, just H -> W(H) -> ReLU per layer
    - Optional: combine='con' doubles input dim (concat lp and hp before per-order weights)
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
        dropout=(0.6, 0.0),
        activation=True,
        combine="sum",
        use_ense_coe=False,
        use_random_gates=False,
        use_random_filter=False,
        P_gate=1,
        P_filter=1,
        use_bn=False,
    ):
        super().__init__()
        self.P = P
        self.dropout = dropout
        self.combine = combine
        self.activation = activation
        self.K_lp = K_lp
        self.K_hp = K_hp
        self.use_ense_coe = use_ense_coe
        self.use_random_gates = use_random_gates
        self.use_random_filter = use_random_filter
        self.P_gate = P_gate
        self.P_filter = P_filter

        # TFE-style: learnable polynomial coefficients (one set for lp, one for hp)
        if use_random_filter:
            # S+F: c_k(omega) = sum_r gamma_{k,r} Psi_r(omega)
            self.gamma_lp = nn.Parameter(torch.ones(K_lp + 1, P_filter + 1) * 0.5)
            self.gamma_hp = nn.Parameter(torch.ones(K_hp + 1, P_filter + 1) * 0.5)
            self.coeffs_lp = self.coeffs_hp = None
        else:
            self.coeffs_lp = nn.Parameter(torch.ones(K_lp + 1) * 0.5)
            self.coeffs_hp = nn.Parameter(torch.ones(K_hp + 1) * 0.5)
            self.gamma_lp = self.gamma_hp = None

        if use_ense_coe:
            # TFE exact: 2 params for lp/hp combine (ense_coe)
            self.ense_coe = nn.Parameter(torch.ones(2))
            self.alpha = self.beta = None
            self.a_coeffs = self.b_coeffs = None
        elif use_random_gates:
            # S+G: alpha_n(omega), beta_n(omega) as chaos expansions
            self.a_coeffs = nn.Parameter(torch.zeros(P + 1, P_gate + 1))
            self.b_coeffs = nn.Parameter(torch.zeros(P + 1, P_gate + 1))
            with torch.no_grad():
                self.a_coeffs[:, 0] = 0.7
                self.b_coeffs[:, 0] = 0.3
            self.alpha = self.beta = None
            self.ense_coe = None
        else:
            # Order-specific branch weights (per chaos order)
            self.alpha = nn.Parameter(torch.ones(P + 1) * 0.7)
            self.beta = nn.Parameter(torch.ones(P + 1) * 0.3)
            self.a_coeffs = self.b_coeffs = None
            self.ense_coe = None

        # MLP layers (no propagation after first combine)
        # con: each channel gets [alpha[n]*h_lp; beta[n]*h_hp] -> 2*input_dim (TFE-style)
        combine_dim = 2 * input_dim if combine == "con" else input_dim

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_d = combine_dim if i == 0 else hidden_dim
            out_d = hidden_dim if i < num_layers - 1 else out_dim
            self.layers.append(nn.Linear(in_d, out_d, bias=False))

        # Per-chaos-channel BatchNorm1d between hidden MLP layers (opt-in;
        # default False preserves the published architecture). Mirrors
        # DSSGNN.use_bn.
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
        with torch.no_grad():
            if self.coeffs_lp is not None:
                self.coeffs_lp.data.fill_(0.5)
                self.coeffs_hp.data.fill_(0.5)
            if self.ense_coe is not None:
                self.ense_coe.data.fill_(1.0)
        for m in self.layers:
            if isinstance(m, nn.Linear):
                stdv = 1.0 / math.sqrt(m.weight.size(1))
                m.weight.data.normal_(-stdv, stdv)

    def _get_gates(self, device, dtype, omega=None):
        """Compute alpha, beta from chaos expansion when use_random_gates."""
        if omega is None:
            if self.training:
                omega = sample_omega(1, device, dtype)
            else:
                omega = torch.zeros(1, device=device, dtype=dtype)
        Psi = eval_chaos_basis(omega, self.P_gate, device)
        alpha = (self.a_coeffs * Psi).sum(dim=1)
        beta = (self.b_coeffs * Psi).sum(dim=1)
        return alpha, beta

    def _get_filter_coeffs(self, device, dtype, omega=None):
        """Compute coeffs_lp, coeffs_hp from chaos expansion when use_random_filter."""
        if omega is None:
            if self.training:
                omega = sample_omega(1, device, dtype)
            else:
                omega = torch.zeros(1, device=device, dtype=dtype)
        Psi = eval_chaos_basis(omega, self.P_filter, device)
        coeffs_lp = (self.gamma_lp * Psi).sum(dim=1)
        coeffs_hp = (self.gamma_hp * Psi).sum(dim=1)
        return coeffs_lp, coeffs_hp

    def _forward_coefficients(self, adj_lp, adj_hp, x, omega=None):
        if self.dropout[0] > 0 and self.training:
            x = F.dropout(x, self.dropout[0], training=True)

        device, dtype = x.device, x.dtype
        if self.use_random_filter:
            coeffs_lp, coeffs_hp = self._get_filter_coeffs(device, dtype, omega)
        else:
            coeffs_lp, coeffs_hp = self.coeffs_lp, self.coeffs_hp

        # Single propagation (TFE key insight)
        h_lp = mix_prop(adj_lp, x, coeffs_lp)
        h_hp = mix_prop(adj_hp, x, coeffs_hp)

        # Chaos channels: different lp/hp balance per order (TFE-style)
        if self.use_random_gates:
            device, dtype = x.device, x.dtype
            alpha, beta = self._get_gates(device, dtype, omega)
            if self.combine == "con":
                H_list = [
                    torch.cat([alpha[n] * h_lp, beta[n] * h_hp], dim=1)
                    for n in range(self.P + 1)
                ]
            else:
                H_list = [alpha[n] * h_lp + beta[n] * h_hp for n in range(self.P + 1)]
        elif self.use_ense_coe:
            # TFE exact: single combine, all channels same
            if self.combine == "con":
                h = torch.cat([self.ense_coe[0] * h_lp, self.ense_coe[1] * h_hp], dim=1)
            else:
                h = self.ense_coe[0] * h_lp + self.ense_coe[1] * h_hp
            H_list = [h for _ in range(self.P + 1)]
        elif self.combine == "con":
            H_list = [
                torch.cat([self.alpha[n] * h_lp, self.beta[n] * h_hp], dim=1)
                for n in range(self.P + 1)
            ]
        else:
            H_list = [
                self.alpha[n] * h_lp + self.beta[n] * h_hp
                for n in range(self.P + 1)
            ]

        # MLP layers (no more propagation)
        for i, layer in enumerate(self.layers):
            H_list = [layer(h) for h in H_list]
            if i < len(self.layers) - 1:
                if self.activation:
                    H_list = [F.relu(h) for h in H_list]
                if self.use_bn:
                    H_list = [self.bns[i][n](h) for n, h in enumerate(H_list)]
                if self.dropout[1] > 0 and self.training:
                    H_list = [F.dropout(h, self.dropout[1], training=True) for h in H_list]

        return H_list

    def forward(self, adj_lp, adj_hp, x, return_uncertainty=False, omega=None):
        """
        TFE-style: propagate once, then MLP.
        omega: optional for S+G / S+F (used when use_random_gates or use_random_filter)
        """
        H_list = self._forward_coefficients(adj_lp, adj_hp, x, omega=omega)

        Z_0 = H_list[0]
        logits = Z_0

        if return_uncertainty:
            uncertainty = torch.zeros(Z_0.shape[0], device=Z_0.device, dtype=Z_0.dtype)
            for n in range(1, self.P + 1):
                uncertainty = uncertainty + (H_list[n] ** 2).sum(dim=1)
            return logits, uncertainty

        return logits

    def forward_with_sample_logits(self, adj_lp, adj_hp, x, S=4, omega=None):
        """Return mean logits, quadrature-node logits, weights, and chaos energy.

        Nodes/weights are built on the fly from the final chaos coefficient
        stack (Eq. 8 reconstruction). For S+G / S+F variants the gate/filter
        coefficients follow the same omega convention as forward (deterministic
        omega=0 at eval time), so the stack matches what forward returns.
        Returns:
            logits: (N, C) mean logits Z_0
            sample_logits: (S, N, C) quadrature-node logits
            weights: (S,) quadrature weights
            chaos_energy: (N,) per-node chaos energy
        """
        H_list = self._forward_coefficients(adj_lp, adj_hp, x, omega=omega)
        logits = H_list[0]
        sample_logits, weights = quadrature_sample_logits(H_list, S)
        if len(H_list) > 1:
            chaos_energy = sum(h.pow(2).sum(dim=-1) for h in H_list[1:])
        else:
            chaos_energy = logits.new_zeros(logits.shape[0])
        return logits, sample_logits, weights, chaos_energy
