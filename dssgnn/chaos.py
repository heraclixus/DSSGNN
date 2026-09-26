"""
Chaos basis utilities for DSS-GNN (method.tex).

Orthonormal Hermite polynomials Psi_r(omega) for omega ~ N(0,1):
  Psi_0(omega) = 1
  Psi_1(omega) = omega
  Psi_2(omega) = (omega^2 - 1) / sqrt(2)
  Psi_r(omega) = He_r(omega) / sqrt(r!)

where He_r are probabilist's Hermite polynomials.
"""

import math
import torch
import torch.nn as nn

try:
    from numpy.polynomial.hermite import hermgauss
except ImportError:
    hermgauss = None


def gauss_hermite_quadrature(S: int, device=None, dtype=torch.float32):
    """
    Gauss-Hermite quadrature nodes and weights for omega ~ N(0,1).

    For integral E[f(omega)] = int f(omega) * (1/sqrt(2pi)) exp(-omega^2/2) d omega.
    Uses transformation: omega = sqrt(2)*u, then int w.r.t. exp(-u^2).
    Returns (omega_s, w_s) with sum_s w_s = 1 (normalized for N(0,1) measure).

    Eq. (3.92-3.96) in method.tex: nodes/weights {omega_s, w_s}_{s=1}^S.
    """
    if hermgauss is None:
        raise ImportError("numpy.polynomial.hermite required for Gauss-Hermite quadrature")
    u, w_raw = hermgauss(S)
    # omega_s = sqrt(2) * u_s for N(0,1)
    omega = torch.tensor(u * (2 ** 0.5), dtype=dtype)
    # w_s = w_raw / sqrt(pi) for correct N(0,1) measure
    w = torch.tensor(w_raw / (math.pi ** 0.5), dtype=dtype)
    if device is not None:
        omega = omega.to(device)
        w = w.to(device)
    return omega, w


def eval_chaos_basis(omega: torch.Tensor, P: int, device=None) -> torch.Tensor:
    """
    Evaluate orthonormal Hermite chaos basis Psi_0,...,Psi_P at omega.

    Args:
        omega: (...,) tensor, typically scalar or (batch,). Standard normal samples.
        P: max chaos order (returns P+1 values)
        device: optional device for output

    Returns:
        Psi: (..., P+1) tensor with Psi[r] = Psi_r(omega)
    """
    if device is None:
        device = omega.device
    dtype = omega.dtype
    shape = omega.shape

    # Reshape for broadcasting: omega (..., 1) for Psi output (..., P+1)
    o = omega.unsqueeze(-1)  # (..., 1)

    Psi_list = []
    # Psi_0 = 1
    Psi_list.append(torch.ones(*shape, 1, device=device, dtype=dtype))
    if P == 0:
        return torch.cat(Psi_list, dim=-1)

    # Psi_1 = omega
    Psi_list.append(o)
    if P == 1:
        return torch.cat(Psi_list, dim=-1)

    # Recurrence: He_n(x) = x * He_{n-1}(x) - (n-1) * He_{n-2}(x)
    # Psi_n = He_n / sqrt(n!)
    He_prev = o.clone()  # He_1 = omega
    He_prev2 = torch.ones_like(o)  # He_0 = 1
    for r in range(2, P + 1):
        He_r = o * He_prev - (r - 1) * He_prev2
        norm = math.sqrt(math.factorial(r))
        Psi_r = He_r / norm
        Psi_list.append(Psi_r)
        He_prev2 = He_prev
        He_prev = He_r

    return torch.cat(Psi_list, dim=-1)


def sample_omega(batch_size: int, device, dtype=torch.float32) -> torch.Tensor:
    """Sample omega ~ N(0,1) for operator stochasticity."""
    return torch.randn(batch_size, device=device, dtype=dtype)


def quadrature_sample_logits(coefficient_stack, S: int):
    """
    Reconstruct quadrature-node logits from a final chaos coefficient stack.

    Args:
        coefficient_stack: list of P+1 tensors (N, C) — the final H_list / Z_n stack
        S: number of Gauss-Hermite quadrature nodes

    Returns:
        sample_logits: (S, N, C) with z^(s) = sum_n Psi_n(omega_s) Z_n
        weights: (S,) quadrature weights summing to 1
    """
    P = len(coefficient_stack) - 1
    device = coefficient_stack[0].device
    dtype = coefficient_stack[0].dtype
    omega, weights = gauss_hermite_quadrature(S, device=device, dtype=dtype)
    psi = eval_chaos_basis(omega, P)  # (S, P+1)
    coeffs = torch.stack(coefficient_stack, dim=0)  # (P+1, N, C)
    sample_logits = torch.einsum("sp,pnc->snc", psi, coeffs)
    return sample_logits, weights
