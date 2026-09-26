#!/usr/bin/env python3
"""Compute second-order loss correction analysis for DSS-GNN (Proposition 2).

For each node, computes:
  - Base loss: ell_i = CE(z_{0,i}, y_i)
  - Quadrature-averaged loss: bar_ell_i = sum_s mu_s CE(z_i^{(s)}, y_i)
  - Second-order approximation: ell_i + 0.5 * tr(H_i @ Sigma_i)
  - Correction term: c_i = 0.5 * tr(H_i @ Sigma_i)
  - Chaos energy: E_i = tr(Sigma_i)
  - Prediction entropy: H(p_i)
  - Hessian trace: tr(diag(p) - pp^T)

Saves results as .npz for downstream plotting.
"""

import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tfe_utils import (set_seed, load_data, propagate_adj, random_walk_adj,
                       compute_brier)
from dssgnn.chebyshev import build_rescaled_laplacian
from dssgnn.model import DSSGNN
from dssgnn.dsconv_tfe_propfirst import DSSGNNTFEPropFirst


def compute_correction_quantities(model, graph_input, x, labels, use_tfe=False):
    """Compute all second-order correction quantities on ALL nodes.

    Returns dict of numpy arrays, each of shape (N,) or (N, C).
    """
    model.eval()
    device = x.device

    with torch.no_grad():
        if use_tfe:
            # PropFirst models don't have forward_with_sample_logits directly;
            # fall back to coefficient-based computation
            raise NotImplementedError("Use non-TFE DSSGNN for correction analysis")
        logits, sample_logits, weights, chaos_energy = \
            model.forward_with_sample_logits(graph_input, x)

    N, C = logits.shape
    S = sample_logits.shape[0]
    w = weights.to(device=device, dtype=logits.dtype)  # (S,)

    # --- Per-node base loss: CE(z_{0,i}, y_i) ---
    base_loss = F.cross_entropy(logits, labels, reduction='none')  # (N,)

    # --- Quadrature-averaged loss: sum_s mu_s CE(z_i^{(s)}, y_i) ---
    per_sample_loss = torch.stack([
        F.cross_entropy(sample_logits[s], labels, reduction='none')
        for s in range(S)
    ], dim=0)  # (S, N)
    quad_loss = (w.unsqueeze(1) * per_sample_loss).sum(dim=0)  # (N,)

    # --- Logit covariance Sigma_{z_i} from sample logits ---
    centered = sample_logits - logits.unsqueeze(0)  # (S, N, C)
    # Sigma_{z_i} = sum_s mu_s (z_i^{(s)} - z_{0,i})(z_i^{(s)} - z_{0,i})^T
    logit_cov = torch.zeros(N, C, C, device=device, dtype=logits.dtype)
    for s in range(S):
        outer = centered[s].unsqueeze(2) * centered[s].unsqueeze(1)  # (N, C, C)
        logit_cov += w[s] * outer

    # --- Softmax probabilities ---
    p = F.softmax(logits, dim=1)  # (N, C)

    # --- Hessian of CE: H_i = diag(p_i) - p_i p_i^T ---
    hessian = torch.diag_embed(p) - p.unsqueeze(2) * p.unsqueeze(1)  # (N, C, C)

    # --- Correction: c_i = 0.5 * tr(H_i @ Sigma_i) ---
    H_Sigma = torch.bmm(hessian, logit_cov)  # (N, C, C)
    correction = 0.5 * torch.diagonal(H_Sigma, dim1=1, dim2=2).sum(dim=1)  # (N,)

    # --- Second-order approximation ---
    second_order_approx = base_loss + correction  # (N,)

    # --- Prediction entropy: H(p_i) ---
    pred_entropy = -(p * (p + 1e-12).log()).sum(dim=1)  # (N,)

    # --- Hessian trace: tr(H_i) = 1 - sum_c p_c^2 ---
    hessian_trace = 1.0 - (p * p).sum(dim=1)  # (N,)

    # --- Logit covariance trace = chaos energy (sanity check) ---
    cov_trace = torch.diagonal(logit_cov, dim1=1, dim2=2).sum(dim=1)  # (N,)

    # --- Quadrature-averaged predictive ---
    sample_probs = F.softmax(sample_logits, dim=-1)  # (S, N, C)
    pred_probs = (w.view(-1, 1, 1) * sample_probs).sum(dim=0)  # (N, C)

    # --- Brier scores: point vs integrated ---
    # Brier = mean_c (p_c - 1[y=c])^2
    one_hot = F.one_hot(labels, C).float()
    brier_point = ((p - one_hot) ** 2).sum(dim=1)  # (N,)
    brier_integrated = ((pred_probs - one_hot) ** 2).sum(dim=1)  # (N,)

    # --- Correctness ---
    pred_class = logits.argmax(dim=1)
    correct = (pred_class == labels).float()

    return {
        'base_loss': base_loss.cpu().numpy(),
        'quad_loss': quad_loss.cpu().numpy(),
        'second_order_approx': second_order_approx.cpu().numpy(),
        'correction': correction.cpu().numpy(),
        'chaos_energy': chaos_energy.cpu().numpy(),
        'cov_trace': cov_trace.cpu().numpy(),
        'pred_entropy': pred_entropy.cpu().numpy(),
        'hessian_trace': hessian_trace.cpu().numpy(),
        'brier_point': brier_point.cpu().numpy(),
        'brier_integrated': brier_integrated.cpu().numpy(),
        'correct': correct.cpu().numpy(),
        'labels': labels.cpu().numpy(),
        'pred_probs': pred_probs.cpu().numpy(),
        'softmax_probs': p.cpu().numpy(),
    }


def train_epoch(model, optimizer, graph_input, x, labels, train_mask,
                lambda_reg=0.01):
    """One training epoch with mean-logit loss."""
    model.train()
    optimizer.zero_grad()
    logits, uncertainty = model(graph_input, x, return_uncertainty=True)
    out = F.log_softmax(logits, dim=1)
    loss = F.cross_entropy(out[train_mask], labels[train_mask])
    loss = loss + lambda_reg * uncertainty.mean()
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, graph_input, x, labels, mask):
    """Evaluate accuracy and loss."""
    model.eval()
    with torch.no_grad():
        logits = model(graph_input, x)
    logits = F.log_softmax(logits, dim=1)
    loss = F.cross_entropy(logits[mask], labels[mask]).item()
    pred = logits[mask].argmax(dim=1)
    acc = (pred == labels[mask]).float().mean().item()
    return acc, loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--P', type=int, default=2)
    parser.add_argument('--S', type=int, default=4)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--K_lp', type=int, default=10)
    parser.add_argument('--K_hp', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--lambda_reg', type=float, default=0.01)
    parser.add_argument('--lambda_max', type=float, default=2.0)
    parser.add_argument('--pro_dropout', type=float, default=0.5)
    parser.add_argument('--lin_dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--outdir', type=str, default='results/correction_analysis')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available()
                          else 'cpu')

    # Accumulate results across runs
    all_results = []

    for run_idx in range(args.runs):
        set_seed(run_idx)
        print(f"\n=== {args.dataset} | P={args.P} | run {run_idx} ===")

        adj, features, labels, train_mask, val_mask, test_mask = load_data(
            args.dataset, full=True, random_split=False,
            train_rate=0.6, val_rate=0.2, i=run_idx
        )
        num_classes = int(labels.max()) + 1
        N = features.shape[0]

        model = DSSGNN(
            input_dim=features.shape[1],
            hidden_dim=args.hidden,
            out_dim=num_classes,
            num_layers=args.layers,
            K_lp=args.K_lp, K_hp=args.K_hp,
            P=args.P,
            dropout=(args.pro_dropout, args.lin_dropout),
            activation=True,
            S=args.S,
        ).to(device)

        graph_input = build_rescaled_laplacian(
            adj, lambda_max=args.lambda_max).to(device)
        features_d = features.clone().detach().to(device)
        labels_d = labels.clone().detach().to(device)
        train_mask_d = train_mask.clone().detach().to(device)
        val_mask_d = val_mask.clone().detach().to(device)
        test_mask_d = test_mask.clone().detach().to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.wd)

        # --- Train ---
        best_val_loss = float('inf')
        best_state = None
        bad_epoch = 0
        for epoch in range(args.epochs):
            train_epoch(model, optimizer, graph_input, features_d, labels_d,
                        train_mask_d, lambda_reg=args.lambda_reg)
            val_acc, val_loss = evaluate(model, graph_input, features_d,
                                        labels_d, val_mask_d)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
                bad_epoch = 0
            else:
                bad_epoch += 1
            if bad_epoch >= args.patience:
                break

        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        test_acc, test_loss = evaluate(model, graph_input, features_d,
                                       labels_d, test_mask_d)
        print(f"  Test acc: {test_acc:.4f}")

        # --- Compute correction on ALL nodes ---
        result = compute_correction_quantities(
            model, graph_input, features_d, labels_d)

        # Add split indicators
        split = np.full(N, -1, dtype=np.int32)
        tm = train_mask.numpy()
        vm = val_mask.numpy()
        tsm = test_mask.numpy()
        # Handle both boolean masks and index arrays
        if tm.dtype == bool:
            split[tm] = 0
            split[vm] = 1
            split[tsm] = 2
        else:
            split[tm] = 0
            split[vm] = 1
            split[tsm] = 2
        result['split'] = split

        # Compute node degrees from adjacency
        if sp.issparse(adj):
            degrees = np.array(adj.sum(axis=1)).flatten()
        else:
            degrees = adj.sum(axis=1).numpy() if torch.is_tensor(adj) \
                else adj.sum(axis=1)
        result['degree'] = np.array(degrees, dtype=np.float32)

        all_results.append(result)

    # Average across runs (save all runs for flexibility)
    outpath = os.path.join(args.outdir,
                           f'{args.dataset}_P{args.P}_correction.npz')
    # Stack arrays across runs: shape (runs, N) for each quantity
    stacked = {}
    for key in all_results[0]:
        stacked[key] = np.stack([r[key] for r in all_results], axis=0)
    stacked['dataset'] = np.array(args.dataset)
    stacked['P'] = np.array(args.P)
    np.savez_compressed(outpath, **stacked)
    print(f"\nSaved to {outpath}")


if __name__ == '__main__':
    main()
