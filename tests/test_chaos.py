"""Tests for chaos basis and Gauss-Hermite quadrature (method.tex)."""

import math
import torch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dssgnn.chaos import eval_chaos_basis, gauss_hermite_quadrature, sample_omega


def test_eval_chaos_basis_psi0():
    """Psi_0(omega) = 1 for all omega."""
    omega = torch.tensor([0.0, 1.0, -2.5])
    psi = eval_chaos_basis(omega, P=0)
    assert psi.shape == (3, 1)
    torch.testing.assert_close(psi[:, 0], torch.ones(3))


def test_eval_chaos_basis_psi1():
    """Psi_1(omega) = omega."""
    omega = torch.tensor([0.0, 1.0, -2.5])
    psi = eval_chaos_basis(omega, P=1)
    assert psi.shape == (3, 2)
    torch.testing.assert_close(psi[:, 0], torch.ones(3))
    torch.testing.assert_close(psi[:, 1], omega)


def test_eval_chaos_basis_orthonormal():
    """E[Psi_n Psi_m] = delta_nm for standard normal omega (via quadrature)."""
    omega, w = gauss_hermite_quadrature(10)
    psi = eval_chaos_basis(omega, P=3)  # (10, 4)
    # E[Psi_i Psi_j] approx sum_s w_s * psi[s,i] * psi[s,j]
    gram = torch.einsum("s,si,sj->ij", w, psi, psi)
    torch.testing.assert_close(gram, torch.eye(4), atol=1e-5, rtol=1e-5)


def test_gauss_hermite_quadrature_normalized():
    """Quadrature weights sum to 1 for N(0,1) measure."""
    omega, w = gauss_hermite_quadrature(5)
    assert omega.shape == (5,)
    assert w.shape == (5,)
    torch.testing.assert_close(w.sum(), torch.tensor(1.0), atol=1e-5, rtol=1e-5)


def test_gauss_hermite_quadrature_expectation():
    """E[omega^2] = 1 for omega ~ N(0,1)."""
    omega, w = gauss_hermite_quadrature(10)
    e_x2 = (w * omega ** 2).sum()
    torch.testing.assert_close(e_x2, torch.tensor(1.0), atol=1e-5, rtol=1e-5)


def test_sample_omega_shape():
    """sample_omega returns correct shape."""
    omega = sample_omega(100, device=torch.device("cpu"))
    assert omega.shape == (100,)


if __name__ == "__main__":
    test_eval_chaos_basis_psi0()
    test_eval_chaos_basis_psi1()
    test_eval_chaos_basis_orthonormal()
    test_gauss_hermite_quadrature_normalized()
    test_gauss_hermite_quadrature_expectation()
    test_sample_omega_shape()
    print("All chaos tests passed.")
