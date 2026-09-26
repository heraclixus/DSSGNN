"""
DS-Conv TFE with random branch gates (S+G variant).

Same as DSConvTFE but alpha_n(omega), beta_n(omega) are chaos expansions.
"""

import math
import torch
import torch.nn as nn

from .dsconv_tfe import mix_prop
from .chaos import eval_chaos_basis, sample_omega


class DSConvTFERandomGates(nn.Module):
    """
    DS-Conv TFE with stochastic branch gates (S+G).
    """

    def __init__(self, in_dim, out_dim, K_lp, K_hp, P, P_gate=1, activation=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.K_lp = K_lp
        self.K_hp = K_hp
        self.P = P
        self.P_gate = P_gate
        self.activation = activation

        self.coeffs_lp = nn.Parameter(torch.ones(K_lp + 1) * 0.5)
        self.coeffs_hp = nn.Parameter(torch.ones(K_hp + 1) * 0.5)

        self.a_coeffs = nn.Parameter(torch.zeros(P + 1, P_gate + 1))
        self.b_coeffs = nn.Parameter(torch.zeros(P + 1, P_gate + 1))
        with torch.no_grad():
            self.a_coeffs[:, 0] = 0.7
            self.b_coeffs[:, 0] = 0.3

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.coeffs_lp.data.fill_(0.5)
            self.coeffs_hp.data.fill_(0.5)
        stdv = 1.0 / math.sqrt(self.W.weight.size(1))
        self.W.weight.data.normal_(-stdv, stdv)

    def _get_gates(self, device, dtype, omega=None):
        if omega is None:
            if self.training:
                omega = sample_omega(1, device, dtype)
            else:
                omega = torch.zeros(1, device=device, dtype=dtype)
        Psi = eval_chaos_basis(omega, self.P_gate, device)
        alpha = (self.a_coeffs * Psi).sum(dim=1)
        beta = (self.b_coeffs * Psi).sum(dim=1)
        return alpha, beta

    def forward(self, adj_lp, adj_hp, H_list, omega=None):
        device = H_list[0].device
        dtype = H_list[0].dtype
        alpha, beta = self._get_gates(device, dtype, omega)

        out_list = []
        for n in range(self.P + 1):
            h_n = H_list[n]
            g_lp_h = mix_prop(adj_lp, h_n, self.coeffs_lp)
            g_hp_h = mix_prop(adj_hp, h_n, self.coeffs_hp)
            combined = alpha[n] * g_lp_h + beta[n] * g_hp_h
            out = self.W(combined)
            if self.activation:
                out = torch.relu(out)
            out_list.append(out)
        return out_list
