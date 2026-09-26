#!/usr/bin/env python3
"""
Minimal synthetic graph test for DSS-GNN + GDUQ.
Runs in ~5 seconds. Use for quick sanity check.

Run from project root: python scripts/test_dssgnn_gduq_synthetic.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from scipy import sparse

from tfe_utils import set_seed, accuracy, compute_ece, compute_mce, compute_brier
from dssgnn.chebyshev import build_rescaled_laplacian
from dssgnn.model import DSSGNN
from dssgnn.dssgnn_gduq import DSSGNNWithGDUQAnchor


def make_synthetic_graph(N=50, d=8, C=3, p_edge=0.1):
    """Small random graph with random features and labels."""
    set_seed(42)
    # Sparse random graph
    i = np.random.randint(0, N, size=int(N * N * p_edge))
    j = np.random.randint(0, N, size=int(N * N * p_edge))
    v = np.ones(len(i))
    adj = sparse.coo_matrix((v, (i, j)), shape=(N, N))
    adj = adj + adj.T
    adj = adj + sparse.eye(N)
    adj = sparse.csr_matrix(adj)

    features = torch.randn(N, d).float()
    labels = torch.randint(0, C, (N,))

    n_train, n_val = int(0.6 * N), int(0.2 * N)
    perm = torch.randperm(N)
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train : n_train + n_val]] = True
    test_mask[perm[n_train + n_val :]] = True

    return adj, features, labels, train_mask, val_mask, test_mask, C


def main():
    print("=== Synthetic graph (N=50, d=8, C=3) ===")
    adj, features, labels, train_mask, val_mask, test_mask, num_classes = make_synthetic_graph()

    device = torch.device("cpu")
    mean = features[train_mask].mean(0)
    std = features[train_mask].std(0) + 1e-3

    model = DSSGNNWithGDUQAnchor(
        base_model_class=DSSGNN,
        base_model_kwargs=dict(
            input_dim=features.shape[1],
            hidden_dim=16,
            out_dim=num_classes,
            num_layers=2,
            K_lp=2,
            K_hp=2,
            P=1,
        ),
        mean=mean,
        std=std,
    )
    model = model.to(device)
    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    graph_input = build_rescaled_laplacian(adj, lambda_max=2.0)
    graph_input = graph_input.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(20):
        model.train()
        optimizer.zero_grad()
        out = model(graph_input, features, anchors=None, n_anchors=3)
        loss = torch.nn.functional.cross_entropy(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(graph_input, features, anchors=None, n_anchors=5)
        logits = torch.nn.functional.log_softmax(out, dim=1)
        acc = accuracy(logits[test_mask], labels[test_mask])
        ece = compute_ece(logits[test_mask], labels[test_mask]).item()
        mce = compute_mce(logits[test_mask], labels[test_mask]).item()
        brier = compute_brier(logits[test_mask], labels[test_mask]).item()

    print(f"  acc: {acc:.4f}  ECE: {ece:.4f}  MCE: {mce:.4f}  Brier: {brier:.4f}")
    print("OK: DS-GNN + GDUQ runs on synthetic graph")


if __name__ == "__main__":
    main()
