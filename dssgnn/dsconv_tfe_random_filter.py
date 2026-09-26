"""
DS-Conv TFE with random filter coefficients (S+F variant).

Same as DSConvTFE but c_k^(lp)(omega), c_k^(hp)(omega) are chaos expansions.
"""

import math
import torch
import torch.nn as nn

from .dsconv_tfe import mix_prop
from .chaos import eval_chaos_basis, sample_omega


class DSConvTFERandomFilter(nn.Module):
    """
    DS-Conv TFE with stochastic polynomial filter coefficients (S+F).
    """

    def __init__(self, in_dim, out_dim, K_lp, K_hp, P, P_filter=1, activation=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.K_lp = K_lp
        self.K_hp = K_hp
        self.P = P
        self.P_filter = P_filter
        self.activation = activation

        self.gamma_lp = nn.Parameter(torch.zeros(K_lp + 1, P_filter + 1))
        self.gamma_hp = nn.Parameter(torch.zeros(K_hp + 1, P_filter + 1))
        with torch.no_grad():
            self.gamma_lp.data.fill_(0.5)
            self.gamma_hp.data.fill_(0.5)

        self.alpha = nn.Parameter(torch.ones(P + 1) * 0.7)
        self.beta = nn.Parameter(torch.ones(P + 1) * 0.3)
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self._init_parameters()

    def _init_parameters(self):
        stdv = 1.0 / math.sqrt(self.W.weight.size(1))
        self.W.weight.data.normal_(-stdv, stdv)

    def _get_filter_coeffs(self, device, dtype, omega=None):
        if omega is None:
            if self.training:
                omega = sample_omega(1, device, dtype)
            else:
                omega = torch.zeros(1, device=device, dtype=dtype)
        Psi = eval_chaos_basis(omega, self.P_filter, device)
        coeffs_lp = (self.gamma_lp * Psi).sum(dim=1)
        coeffs_hp = (self.gamma_hp * Psi).sum(dim=1)
        return coeffs_lp, coeffs_hp

    def forward(self, adj_lp, adj_hp, H_list, omega=None):
        device = H_list[0].device
        dtype = H_list[0].dtype
        coeffs_lp, coeffs_hp = self._get_filter_coeffs(device, dtype, omega)

        out_list = []
        for n in range(self.P + 1):
            h_n = H_list[n]
            g_lp_h = mix_prop(adj_lp, h_n, coeffs_lp)
            g_hp_h = mix_prop(adj_hp, h_n, coeffs_hp)
            combined = self.alpha[n] * g_lp_h + self.beta[n] * g_hp_h
            out = self.W(combined)
            if self.activation:
                out = torch.relu(out)
            out_list.append(out)
        return out_list
