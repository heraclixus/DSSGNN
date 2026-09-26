"""
Standalone GDUQ (G-Δ-UQ) runner using tfe_utils load_data.
Runs GraphANTNode on node classification without GOOD dependency.
"""

import argparse
import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm

from tfe_utils import set_seed, load_data, accuracy, compute_ece, compute_mce, compute_brier

# GDUQ imports (requires torch_geometric)
from torch_geometric.data import Data

# Add gduq to path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gduq", "src"))


def adj_to_edge_index(adj):
    """Convert scipy sparse adj to PyG edge_index."""
    if sp.issparse(adj):
        adj = adj.tocoo()
    else:
        adj = sp.coo_matrix(adj)
    edge_index = torch.tensor(np.stack([adj.row, adj.col]), dtype=torch.long)
    return edge_index


def run_gduq(args, dataset, i):
    set_seed(args.seed if args.random_split else i)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    adj, features, labels, train_mask, val_mask, test_mask = load_data(
        dataset, args.full, args.random_split, args.train_rate, args.val_rate, i
    )

    # Convert to PyG Data
    if sp.issparse(adj):
        adj = adj.tocsr()
    edge_index = adj_to_edge_index(adj)
    x = features.float() if not isinstance(features, torch.Tensor) else features.float()
    num_classes = int(max(labels)) + 1

    y = torch.tensor(labels, dtype=torch.long) if not isinstance(labels, torch.Tensor) else labels.long()
    data = Data(x=x, edge_index=edge_index, y=y.unsqueeze(1))
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.num_nodes = x.shape[0]

    # Build config for GDUQ (encoder expects config.model.*, config.dataset.*)
    from types import SimpleNamespace
    config = SimpleNamespace()
    config.model = SimpleNamespace(
        model_layer=args.layers,
        dim_hidden=args.hidden,
        dropout_rate=args.dropout,
        model_level="node",
    )
    config.dataset = SimpleNamespace(
        dim_node=x.shape[1] * 2,  # GDUQ uses 2x for anchor
        num_classes=num_classes,
    )
    config.device = device

    # Anchor distribution from train features
    train_x = x[train_mask]
    mu = train_x.mean(dim=0)
    std = train_x.std(dim=0)
    std[std == 0] = 1e-3

    # Build model
    from models.encoders import GCNEncoder
    from models.gduq_models import baseModelNode, GraphANTNode

    gcn_enc = GCNEncoder(config)
    base_net = baseModelNode(encoder=gcn_enc, num_classes=num_classes)
    model = GraphANTNode(
        base_network=base_net,
        mean=mu,
        std=std,
        anchor_type="node",
        num_classes=num_classes,
    )
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # Use anchors=None so model samples anchors on the fly (matches GDUQ training behavior)

    best_val_loss = float("inf")
    test_acc = 0
    test_ece = test_mce = test_brier = 0.0
    bad_epoch = 0

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data, anchors=None, n_anchors=args.num_anchors)
        loss = torch.nn.functional.cross_entropy(out[data.train_mask], data.y[data.train_mask].squeeze())
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(data, anchors=None, n_anchors=args.num_anchors)
            logits = torch.nn.functional.log_softmax(out, dim=1)
            val_loss = torch.nn.functional.cross_entropy(
                logits[data.val_mask], data.y[data.val_mask].squeeze()
            ).item()
            tmp_test_acc = accuracy(logits[data.test_mask], data.y[data.test_mask].squeeze())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            test_logits = logits[data.test_mask]
            test_labels = data.y[data.test_mask].squeeze()
            test_ece = compute_ece(test_logits, test_labels).item()
            test_mce = compute_mce(test_logits, test_labels).item()
            test_brier = compute_brier(test_logits, test_labels).item()
            bad_epoch = 0
        else:
            bad_epoch += 1
        if bad_epoch == args.patience:
            break

    return test_acc, test_ece, test_mce, test_brier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=300)  # GDUQ baseModelNode uses 300
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--num_anchors", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full", type=bool, default=True)
    parser.add_argument("--random_split", type=bool, default=True)
    args = parser.parse_args()
    args.train_rate = 0.6 if args.full else 0.025
    args.val_rate = 0.2 if args.full else 0.025

    print(args)
    all_accs = []
    all_eces = []
    all_mces = []
    all_briers = []
    for i in tqdm(range(args.runs)):
        acc, ece, mce, brier = run_gduq(args, args.dataset, i)
        all_accs.append(acc.item())
        all_eces.append(ece)
        all_mces.append(mce)
        all_briers.append(brier)
        print(f"run_{i+1}  acc: {acc:.4f}  ECE: {ece:.4f}  MCE: {mce:.4f}  Brier: {brier:.4f}")
    print(f"test acc mean (%) = {np.mean(all_accs)*100:.2f} ± {np.std(all_accs)*100:.2f}")
    print(f"test ECE mean = {np.mean(all_eces):.4f} ± {np.std(all_eces):.4f}")
    print(f"test MCE mean = {np.mean(all_mces):.4f} ± {np.std(all_mces):.4f}")
    print(f"test Brier mean = {np.mean(all_briers):.4f} ± {np.std(all_briers):.4f}")


if __name__ == "__main__":
    main()
