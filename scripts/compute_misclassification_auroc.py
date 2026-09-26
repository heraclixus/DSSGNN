#!/usr/bin/env python3
"""
Compute misclassification AUROC: how well does chaos energy predict which
nodes the model gets wrong? Higher AUROC = better uncertainty.

This is the definitive metric for "the model knows what it doesn't know."
P>0 should give higher AUROC than P=0 (deterministic has no chaos energy).

Usage:
  python scripts/compute_misclassification_auroc.py --dataset cora --P 2
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

from sklearn.metrics import roc_auc_score
from tfe_utils import set_seed, load_data, accuracy
from dssgnn.chebyshev import build_rescaled_laplacian
from dssgnn.model import DSSGNN


def run(args):
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    adj, features, labels, train_mask, val_mask, test_mask = load_data(
        args.dataset, True, True, 0.6, 0.2, args.seed
    )
    num_classes = int(max(labels)) + 1
    graph_input = build_rescaled_laplacian(adj, lambda_max=2.0).to(device)

    model = DSSGNN(
        input_dim=features.shape[1], hidden_dim=args.hidden, out_dim=num_classes,
        num_layers=2, K_lp=4, K_hp=4, P=args.P, dropout=(args.dropout, 0.0),
        activation=True, S=args.S,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    # Train
    best_val_loss = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits, unc = model(graph_input, features, return_uncertainty=True)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss = loss + args.lambda_reg * unc.mean()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits, _ = model(graph_input, features, return_uncertainty=True)
            val_loss = F.cross_entropy(logits[val_mask], labels[val_mask]).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model = model.to(device)

    # Evaluate
    model.eval()
    with torch.no_grad():
        logits, uncertainty = model(graph_input, features, return_uncertainty=True)
        probs = F.softmax(logits, dim=1)

    test_idx = test_mask.nonzero(as_tuple=True)[0].cpu()
    test_logits = logits[test_idx].cpu()
    test_probs = probs[test_idx].cpu().numpy()
    test_labels = labels[test_idx].cpu().numpy()
    test_uncertainty = uncertainty[test_idx].cpu().numpy()

    preds = test_probs.argmax(axis=1)
    is_wrong = (preds != test_labels).astype(int)

    acc = (1 - is_wrong.mean()) * 100
    brier = ((test_probs - np.eye(num_classes)[test_labels]) ** 2).sum(axis=1).mean()

    # Misclassification AUROC using different scores
    results = {"acc": acc, "brier": brier, "n_test": len(test_idx), "n_wrong": int(is_wrong.sum())}

    # 1. Chaos energy (only meaningful for P>0)
    if test_uncertainty.max() > 1e-10:
        try:
            results["auroc_chaos"] = roc_auc_score(is_wrong, test_uncertainty)
        except ValueError:
            results["auroc_chaos"] = 0.5
    else:
        results["auroc_chaos"] = 0.5  # no signal for P=0

    # 2. Max probability (1 - max_prob as uncertainty, works for any P)
    max_prob_unc = 1.0 - test_probs.max(axis=1)
    try:
        results["auroc_maxprob"] = roc_auc_score(is_wrong, max_prob_unc)
    except ValueError:
        results["auroc_maxprob"] = 0.5

    # 3. Entropy of prediction
    entropy = -np.sum(test_probs * np.log(test_probs + 1e-10), axis=1)
    try:
        results["auroc_entropy"] = roc_auc_score(is_wrong, entropy)
    except ValueError:
        results["auroc_entropy"] = 0.5

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--P", type=int, default=2)
    parser.add_argument("--S", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lambda_reg", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    all_results = []
    for seed in range(args.runs):
        args.seed = seed
        r = run(args)
        all_results.append(r)
        print(f"  seed={seed} acc={r['acc']:.2f} brier={r['brier']:.4f} "
              f"auroc_chaos={r['auroc_chaos']:.4f} auroc_maxprob={r['auroc_maxprob']:.4f} "
              f"auroc_entropy={r['auroc_entropy']:.4f}")

    # Aggregate
    for key in ["acc", "brier", "auroc_chaos", "auroc_maxprob", "auroc_entropy"]:
        vals = [r[key] for r in all_results]
        print(f"METRIC dataset={args.dataset} P={args.P} {key}_mean={np.mean(vals):.4f} {key}_std={np.std(vals):.4f}")


if __name__ == "__main__":
    main()
