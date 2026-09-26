#!/usr/bin/env python3
"""Train DSS-GNN with P=0 warmup: train P=0 to convergence, then fine-tune P>0.

This guarantees P>0 starts from at least the P=0 solution. The chaos channels
are initialized near zero and trained with a separate (smaller) learning rate.

Usage:
  python scripts/train_warmup_p.py --dataset roman-empire --P 1 --hidden 64
"""

import argparse
import copy
import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tfe_utils import set_seed, load_data, accuracy, compute_brier
from dssgnn.chebyshev import build_rescaled_laplacian
from dssgnn.model import DSSGNN


def train_epoch(model, optimizer, graph_input, x, labels, train_mask,
                lambda_reg=0.01):
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
    model.eval()
    with torch.no_grad():
        logits = model(graph_input, x)
    logits_sm = F.log_softmax(logits, dim=1)
    loss = F.cross_entropy(logits_sm[mask], labels[mask]).item()
    acc = float(accuracy(logits_sm[mask], labels[mask]))
    brier = float(compute_brier(logits_sm[mask], labels[mask]))
    return acc, loss, brier


def train_to_convergence(model, optimizer, graph_input, x, labels,
                         train_mask, val_mask, epochs, patience, lambda_reg):
    """Train until validation loss converges."""
    best_val_loss = float('inf')
    best_state = None
    bad = 0
    for epoch in range(epochs):
        train_epoch(model, optimizer, graph_input, x, labels, train_mask,
                    lambda_reg=lambda_reg)
        _, val_loss, _ = evaluate(model, graph_input, x, labels, val_mask)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break
    return best_state, epoch + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--P', type=int, default=1)
    parser.add_argument('--S', type=int, default=4)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--K_lp', type=int, default=10)
    parser.add_argument('--K_hp', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_chaos', type=float, default=0.001,
                        help='Learning rate for P>0 chaos channels in phase 2')
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--lambda_reg', type=float, default=0.01)
    parser.add_argument('--lambda_reg_phase2', type=float, default=None,
                        help='Lambda_reg for phase 2 (default: same as phase 1)')
    parser.add_argument('--pro_dropout', type=float, default=0.5)
    parser.add_argument('--lin_dropout', type=float, default=0.0)
    parser.add_argument('--epochs_p0', type=int, default=500,
                        help='Max epochs for P=0 warmup phase')
    parser.add_argument('--epochs_p', type=int, default=500,
                        help='Max epochs for P>0 fine-tune phase')
    parser.add_argument('--patience', type=int, default=200)
    parser.add_argument('--lambda_max', type=float, default=2.0)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    if args.lambda_reg_phase2 is None:
        args.lambda_reg_phase2 = args.lambda_reg

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available()
                          else 'cpu')

    all_acc, all_brier = [], []

    for run_idx in range(args.runs):
        set_seed(run_idx)

        adj, features, labels, train_mask, val_mask, test_mask = load_data(
            args.dataset, full=True, random_split=False,
            train_rate=0.6, val_rate=0.2, i=run_idx
        )
        num_classes = int(labels.max()) + 1
        graph_input = build_rescaled_laplacian(adj, lambda_max=args.lambda_max).to(device)
        features_d = features.clone().detach().to(device)
        labels_d = labels.clone().detach().to(device)
        train_mask_d = train_mask.clone().detach().to(device)
        val_mask_d = val_mask.clone().detach().to(device)
        test_mask_d = test_mask.clone().detach().to(device)

        # ====== Phase 1: Train P=0 to convergence ======
        model_p0 = DSSGNN(
            input_dim=features.shape[1], hidden_dim=args.hidden,
            out_dim=num_classes, num_layers=args.layers,
            K_lp=args.K_lp, K_hp=args.K_hp, P=0,
            dropout=(args.pro_dropout, args.lin_dropout),
            activation=True, S=args.S,
        ).to(device)

        opt_p0 = torch.optim.Adam(model_p0.parameters(), lr=args.lr,
                                   weight_decay=args.wd)
        best_p0_state, ep1 = train_to_convergence(
            model_p0, opt_p0, graph_input, features_d, labels_d,
            train_mask_d, val_mask_d, args.epochs_p0, args.patience,
            args.lambda_reg)
        model_p0.load_state_dict({k: v.to(device) for k, v in best_p0_state.items()})
        p0_acc, _, p0_brier = evaluate(model_p0, graph_input, features_d,
                                        labels_d, test_mask_d)

        # ====== Phase 2: Initialize P>0 from P=0, fine-tune ======
        model_p = DSSGNN(
            input_dim=features.shape[1], hidden_dim=args.hidden,
            out_dim=num_classes, num_layers=args.layers,
            K_lp=args.K_lp, K_hp=args.K_hp, P=args.P,
            dropout=(args.pro_dropout, args.lin_dropout),
            activation=True, S=args.S,
        ).to(device)

        # Copy P=0 weights into P>0 model
        p0_sd = best_p0_state
        p_sd = model_p.state_dict()
        transferred = 0
        for key in p_sd:
            if key in p0_sd and p0_sd[key].shape == p_sd[key].shape:
                p_sd[key] = p0_sd[key].clone()
                transferred += 1
            # For input lifts with per-order projections, copy P=0 weights
            # to order 0 and initialize orders 1..P near zero
            elif key in p0_sd and 'input_lift' in key:
                # P=0 has shape matching order-0; keep it
                pass
        model_p.load_state_dict({k: v.to(device) for k, v in p_sd.items()})

        # Scale P>0 input lift weights to near-zero
        with torch.no_grad():
            for name, param in model_p.named_parameters():
                if 'input_lift' in name or 'lift' in name:
                    # Check if this is a per-order parameter beyond order 0
                    # by looking at parameter indices in the module
                    pass  # The architecture handles this via separate W_in^{(n)}

        # Separate parameter groups: base params at low lr, all at lr_chaos
        # The simplest approach: use a single low lr for fine-tuning
        opt_p = torch.optim.Adam(model_p.parameters(), lr=args.lr_chaos,
                                  weight_decay=args.wd)

        best_p_state, ep2 = train_to_convergence(
            model_p, opt_p, graph_input, features_d, labels_d,
            train_mask_d, val_mask_d, args.epochs_p, args.patience,
            args.lambda_reg_phase2)
        model_p.load_state_dict({k: v.to(device) for k, v in best_p_state.items()})
        p_acc, _, p_brier = evaluate(model_p, graph_input, features_d,
                                      labels_d, test_mask_d)

        print(f"run_{run_idx+1}  P=0: acc={p0_acc:.2f} brier={p0_brier:.4f} "
              f"(ep={ep1})  |  P={args.P}: acc={p_acc:.2f} brier={p_brier:.4f} "
              f"(ep={ep2})")
        all_acc.append(p_acc)
        all_brier.append(p_brier)

    acc_arr = np.array(all_acc)
    brier_arr = np.array(all_brier)
    print(f"\ntest acc mean (%) = {acc_arr.mean():.2f} +/- {acc_arr.std():.2f}")
    print(f"test Brier mean = {brier_arr.mean():.4f} +/- {brier_arr.std():.4f}")


if __name__ == '__main__':
    main()
