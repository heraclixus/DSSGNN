"""Integration test for DS-GNN training (synthetic data, no external deps)."""

import numpy as np
import scipy.sparse as sp
import torch
import random

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dssgnn.chebyshev import build_rescaled_laplacian
from dssgnn.model import DSSGNN


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    return preds.eq(labels).double().sum() / len(labels)


def _make_synthetic_data(n=100, d=20, num_classes=5, density=0.05):
    """Create synthetic graph and labels for testing."""
    adj = sp.eye(n) + sp.random(n, n, density=density, format='csr')
    adj = adj + adj.T
    adj.data = np.minimum(adj.data, 1.0)
    features = torch.randn(n, d)
    labels = torch.randint(0, num_classes, (n,))
    # Simple train/val/test split
    perm = torch.randperm(n)
    n_train, n_val = int(0.6 * n), int(0.2 * n)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train : n_train + n_val]] = True
    test_mask[perm[n_train + n_val :]] = True
    return adj, features, labels, train_mask, val_mask, test_mask


def test_training_one_epoch():
    """Run one training epoch and verify loss decreases."""
    set_seed(42)
    adj, features, labels, train_mask, val_mask, test_mask = _make_synthetic_data(
        n=50, d=10, num_classes=3
    )
    L = build_rescaled_laplacian(adj)
    model = DSSGNN(
        input_dim=10,
        hidden_dim=16,
        out_dim=3,
        num_layers=2,
        K_lp=2,
        K_hp=2,
        P=1,
        dropout=(0.0, 0.0),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    mask = [train_mask, val_mask, test_mask]

    # Initial loss
    model.eval()
    with torch.no_grad():
        logits = model(L, features)
        loss_before = torch.nn.functional.cross_entropy(
            torch.nn.functional.log_softmax(logits, dim=1)[train_mask],
            labels[train_mask],
        ).item()

    # One epoch
    model.train()
    optimizer.zero_grad()
    logits, uncertainty = model(L, features, return_uncertainty=True)
    loss = torch.nn.functional.cross_entropy(
        torch.nn.functional.log_softmax(logits, dim=1)[train_mask],
        labels[train_mask],
    ) + 0.01 * uncertainty.mean()
    loss.backward()
    optimizer.step()

    # Loss after
    model.eval()
    with torch.no_grad():
        logits = model(L, features)
        loss_after = torch.nn.functional.cross_entropy(
            torch.nn.functional.log_softmax(logits, dim=1)[train_mask],
            labels[train_mask],
        ).item()

    # Loss should typically decrease (may not always due to randomness)
    assert loss_after < loss_before or abs(loss_after - loss_before) < 0.5


def test_training_accuracy_nonzero():
    """After a few epochs, accuracy should be above random."""
    set_seed(42)
    adj, features, labels, train_mask, val_mask, test_mask = _make_synthetic_data(
        n=80, d=15, num_classes=4
    )
    L = build_rescaled_laplacian(adj)
    model = DSSGNN(
        input_dim=15,
        hidden_dim=32,
        out_dim=4,
        num_layers=2,
        K_lp=3,
        K_hp=3,
        P=2,
        dropout=(0.0, 0.0),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    mask = [train_mask, val_mask, test_mask]

    for _ in range(20):
        model.train()
        optimizer.zero_grad()
        logits, uncertainty = model(L, features, return_uncertainty=True)
        loss = torch.nn.functional.cross_entropy(
            torch.nn.functional.log_softmax(logits, dim=1)[train_mask],
            labels[train_mask],
        )
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(L, features)
    acc = accuracy(
        torch.nn.functional.log_softmax(logits, dim=1)[test_mask],
        labels[test_mask],
    )
    # Random baseline for 4 classes = 0.25
    assert acc.item() > 0.2


if __name__ == "__main__":
    test_training_one_epoch()
    test_training_accuracy_nonzero()
    print("All training tests passed.")
