"""
DSS-Conv: Non-intrusive layer per Algorithm 1 (method.tex).

Implements Eq. (3.105)-(3.120) and Algorithm 1:
- Filter per order: U_n^lp = G_lp(L) bar_H_n, U_n^hp = G_hp(L) bar_H_n (line 4)
- For each quadrature node omega_s: evaluate gates, sample pre-activation (lines 5-9)
- Pre-activation: sum_n psi_{s,n}(alpha_n U_n^lp + beta_n U_n^hp) then W_ell (Eq. 3.108)
- Activation: sigma (line 9)
- Project back: H_m = sum_s w_s H(omega_s) psi_{s,m} (Eq. 3.92, line 11)
"""

import math
import torch
import torch.nn as nn

from .chebyshev import chebyshev_filter
from .chaos import eval_chaos_basis, gauss_hermite_quadrature


class DSSConvNonIntrusive(nn.Module):
    """
    DSS layer with non-intrusive projection (Algorithm 1).

    For each quadrature node omega_s: evaluate pre-activation, apply sigma,
    then project back to chaos coefficients.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        K_lp,
        K_hp,
        P,
        P_gate=1,
        S=4,
        activation=True,
        use_random_gates=False,
        act_fn="relu",
    ):
        """
        Args:
            in_dim, out_dim, K_lp, K_hp, P: standard
            P_gate: chaos order for gates when use_random_gates (ignored otherwise)
            S: number of Gauss-Hermite quadrature nodes
            activation: apply pointwise activation after projection
            use_random_gates: if True, alpha_n(omega), beta_n(omega) are chaos expansions
            act_fn: activation function name ("relu", "gelu", or "tanh")
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.K_lp = K_lp
        self.K_hp = K_hp
        self.P = P
        self.P_gate = max(P_gate, 1) if use_random_gates else 0
        self.S = S
        self.activation = activation
        self.use_random_gates = use_random_gates
        _act_fns = {"relu": torch.relu, "gelu": torch.nn.functional.gelu, "tanh": torch.tanh}
        self.act_fn = _act_fns.get(act_fn, torch.relu)

        self.coeffs_lp = nn.Parameter(torch.ones(K_lp + 1) * 0.1)
        self.coeffs_hp = nn.Parameter(torch.ones(K_hp + 1) * 0.1)

        if use_random_gates:
            self.a_coeffs = nn.Parameter(torch.zeros(P + 1, self.P_gate + 1))
            self.b_coeffs = nn.Parameter(torch.zeros(P + 1, self.P_gate + 1))
            with torch.no_grad():
                self.a_coeffs[:, 0] = 0.7
                self.b_coeffs[:, 0] = 0.3
            self.alpha = self.beta = None
        else:
            self.alpha = nn.Parameter(torch.ones(P + 1) * 0.7)
            self.beta = nn.Parameter(torch.ones(P + 1) * 0.3)
            self.a_coeffs = self.b_coeffs = None

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.reset_parameters()

        # Precompute quadrature (registered as buffer so it moves with model)
        omega, w = gauss_hermite_quadrature(S, device=None, dtype=torch.float32)
        self.register_buffer("_omega", omega)
        self.register_buffer("_w", w)
        # psi[s,n] = Psi_n(omega_s), max_n = max(P, P_gate)
        self._psi_max_n = max(P, self.P_gate) if use_random_gates else P

    def reset_parameters(self):
        with torch.no_grad():
            for k in range(self.K_lp + 1):
                self.coeffs_lp[k] = 0.5 ** k
            self.coeffs_hp.zero_()
            self.coeffs_hp[0] = -self.coeffs_lp[0]
            for k in range(1, min(self.K_hp + 1, len(self.coeffs_lp))):
                self.coeffs_hp[k] = -self.coeffs_lp[k]
            if self.K_hp >= 0:
                self.coeffs_hp[0] = self.coeffs_hp[0] + 1.0
            if self.use_random_gates:
                self.a_coeffs.zero_()
                self.b_coeffs.zero_()
                self.a_coeffs[:, 0] = 0.7
                self.b_coeffs[:, 0] = 0.3
            else:
                self.alpha.fill_(0.7)
                self.beta.fill_(0.3)
        stdv = 1.0 / math.sqrt(self.W.weight.size(1))
        self.W.weight.data.normal_(-stdv, stdv)

    def _init_parameters(self):
        self.reset_parameters()

    def _get_psi(self, device):
        """psi[s,n] = Psi_n(omega_s) for s=0..S-1, n=0..P."""
        omega = self._omega.to(device)
        psi = eval_chaos_basis(omega, self._psi_max_n, device)  # (S, P+1) or (S, P_gate+1)
        return psi

    def _get_gates(self, psi_gate, device):
        """Alpha_n, beta_n at each quadrature node. Returns (S, P+1) each."""
        if self.use_random_gates:
            # alpha[s,n] = sum_r a_{n,r} psi[s,r]
            psi_g = psi_gate[:, : self.P_gate + 1]  # (S, P_gate+1)
            alpha = torch.einsum("sr,nr->sn", psi_g, self.a_coeffs)  # (S, P+1)
            beta = torch.einsum("sr,nr->sn", psi_g, self.b_coeffs)
        else:
            alpha = self.alpha.unsqueeze(0).expand(self.S, -1)  # (S, P+1)
            beta = self.beta.unsqueeze(0).expand(self.S, -1)
        return alpha, beta

    def forward(self, L_rescaled, H_list):
        """
        Args:
            L_rescaled: (N, N) sparse
            H_list: list of (P+1) tensors (N, in_dim)

        Returns:
            list of (P+1) tensors (N, out_dim)
        """
        device = H_list[0].device
        dtype = H_list[0].dtype
        w = self._w.to(device=device, dtype=dtype)
        psi = self._get_psi(device).to(dtype)  # (S, max_n+1)
        psi_state = psi[:, : self.P + 1]  # (S, P+1)
        psi_gate = psi if self.use_random_gates else None

        # Filter per order: U_n^lp, U_n^hp
        U_lp = []
        U_hp = []
        for n in range(self.P + 1):
            h_n = H_list[n]
            U_lp.append(chebyshev_filter(L_rescaled, h_n, self.coeffs_lp))
            U_hp.append(chebyshev_filter(L_rescaled, h_n, self.coeffs_hp))
        U_lp = torch.stack(U_lp, dim=0)  # (P+1, N, in_dim)
        U_hp = torch.stack(U_hp, dim=0)

        # Gates at each quadrature node: alpha[s,n], beta[s,n]
        if self.use_random_gates:
            alpha, beta = self._get_gates(psi_gate, device)  # (S, P+1) each
        else:
            alpha = self.alpha.to(dtype).unsqueeze(0).expand(self.S, -1)  # (S, P+1)
            beta = self.beta.to(dtype).unsqueeze(0).expand(self.S, -1)

        # For each s: pre_act = sum_n psi[s,n] * (alpha[s,n] U_n^lp + beta[s,n] U_n^hp)
        # alpha, beta: (S, P+1), U_lp, U_hp: (P+1, N, in_dim)
        alpha = alpha.unsqueeze(-1).unsqueeze(-1)  # (S, P+1, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        psi_n = psi_state.unsqueeze(-1).unsqueeze(-1)  # (S, P+1, 1, 1)
        combined = alpha * U_lp.unsqueeze(0) + beta * U_hp.unsqueeze(0)  # (S, P+1, N, in_dim)
        pre_act = (psi_n * combined).sum(dim=1)  # (S, N, in_dim)
        pre_act = self.W(pre_act)  # (S, N, out_dim)
        H_sample = self.act_fn(pre_act) if self.activation else pre_act  # (S, N, out_dim)

        # Project back: H_m = sum_s w_s * H_sample[s] * psi[s,m]
        # H_sample: (S, N, out_dim), psi: (S, P+1), w: (S,)
        out_list = []
        for m in range(self.P + 1):
            # H_m = sum_s w_s * H_sample[s] * psi[s,m]
            coeff = w.unsqueeze(-1).unsqueeze(-1) * psi_state[:, m].unsqueeze(-1).unsqueeze(-1)
            H_m = (coeff * H_sample).sum(dim=0)  # (N, out_dim)
            out_list.append(H_m)
        return out_list
