#!/usr/bin/env python3
"""
Generate data for visualization figures (runs on GPU).
Outputs numpy/csv files that can be plotted locally.

Usage:
  python scripts/generate_visualization_data.py --task chaos_energy --dataset cora
  python scripts/generate_visualization_data.py --task reliability --dataset cora
  python scripts/generate_visualization_data.py --task ood_scores --dataset cora --ood_type feature
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

from tfe_utils import set_seed, load_data, accuracy, propagate_adj, random_walk_adj
from dssgnn.chebyshev import build_rescaled_laplacian
from dssgnn.model import DSSGNN


def train_dssgnn(dataset, device, P=2, hidden=256, epochs=500, lr=0.01, lambda_reg=0.01):
    """Train a DSS-GNN and return model + data."""
    set_seed(42)
    adj, features, labels, train_mask, val_mask, test_mask = load_data(
        dataset, True, True, 0.6, 0.2, 0
    )
    num_classes = int(max(labels)) + 1
    graph_input = build_rescaled_laplacian(adj, lambda_max=2.0).to(device)
    model = DSSGNN(
        input_dim=features.shape[1], hidden_dim=hidden, out_dim=num_classes,
        num_layers=2, K_lp=4, K_hp=4, P=P, dropout=(0.5, 0.0), activation=True,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    test_mask = test_mask.to(device)

    best_val_loss = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits, unc = model(graph_input, features, return_uncertainty=True)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss = loss + lambda_reg * unc.mean()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits, _ = model(graph_input, features, return_uncertainty=True)
            val_loss = F.cross_entropy(logits[val_mask.to(device)], labels[val_mask.to(device)]).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model = model.to(device)
    return model, graph_input, features, labels, train_mask, test_mask, adj


def generate_chaos_energy(args):
    """Generate chaos energy per node for heatmap visualization."""
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, graph_input, features, labels, train_mask, test_mask, adj = train_dssgnn(
        args.dataset, device
    )
    model.eval()
    with torch.no_grad():
        logits, uncertainty = model(graph_input, features, return_uncertainty=True)
        # uncertainty is per-node chaos energy (sum of higher-order coefficient norms)
        chaos_energy = uncertainty

    preds = logits.argmax(dim=1).cpu().numpy()
    correct = (preds == labels.cpu().numpy()).astype(float)

    output = {
        "chaos_energy": chaos_energy.cpu().numpy().tolist(),
        "predictions": preds.tolist(),
        "labels": labels.cpu().numpy().tolist(),
        "correct": correct.tolist(),
        "train_mask": train_mask.cpu().numpy().tolist(),
        "test_mask": test_mask.cpu().numpy().tolist(),
        "dataset": args.dataset,
    }

    # Save edge list for graph layout
    if hasattr(adj, 'tocoo'):
        coo = adj.tocoo()
        output["edge_src"] = coo.row.tolist()
        output["edge_dst"] = coo.col.tolist()

    out_path = os.path.join(args.output_dir, f"chaos_energy_{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump(output, f)
    print(f"Saved chaos energy data to {out_path}")
    print(f"  Nodes: {len(output['chaos_energy'])}, Energy range: [{min(output['chaos_energy']):.4f}, {max(output['chaos_energy']):.4f}]")


def generate_reliability(args):
    """Generate reliability diagram data (binned accuracy vs confidence)."""
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, graph_input, features, labels, train_mask, test_mask, adj = train_dssgnn(
        args.dataset, device
    )
    model.eval()
    with torch.no_grad():
        logits, _ = model(graph_input, features, return_uncertainty=True)
        probs = F.softmax(logits, dim=1)
        # Also get quadrature-averaged predictive
        stats = model.forward_with_predictive_stats(graph_input, features)
        sample_logits = stats.get("sample_logits", None)
        weights = stats.get("weights", None)

    test_idx = test_mask.nonzero(as_tuple=True)[0]
    test_probs = probs[test_idx].cpu().numpy()
    test_labels = labels[test_idx].cpu().numpy()

    # Point-logit confidence
    confidences = test_probs.max(axis=1)
    predictions = test_probs.argmax(axis=1)
    correct = (predictions == test_labels).astype(float)

    # Quadrature-averaged confidence
    if sample_logits is not None and weights is not None:
        sample_probs = F.softmax(sample_logits[:, test_idx, :], dim=2)
        w = weights.to(sample_probs.device).unsqueeze(1).unsqueeze(2)
        avg_probs = (w * sample_probs).sum(dim=0).cpu().numpy()
        avg_confidences = avg_probs.max(axis=1)
        avg_predictions = avg_probs.argmax(axis=1)
        avg_correct = (avg_predictions == test_labels).astype(float)
    else:
        avg_confidences = confidences
        avg_correct = correct

    # Bin into 10 bins
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)

    def compute_bins(conf, corr):
        bin_acc = []
        bin_conf = []
        bin_count = []
        for i in range(n_bins):
            mask = (conf > bins[i]) & (conf <= bins[i + 1])
            if mask.sum() > 0:
                bin_acc.append(corr[mask].mean())
                bin_conf.append(conf[mask].mean())
                bin_count.append(int(mask.sum()))
            else:
                bin_acc.append(0.0)
                bin_conf.append((bins[i] + bins[i + 1]) / 2)
                bin_count.append(0)
        return bin_acc, bin_conf, bin_count

    point_acc, point_conf, point_count = compute_bins(confidences, correct)
    avg_acc, avg_conf, avg_count = compute_bins(avg_confidences, avg_correct)

    def to_py(lst):
        return [float(x) if isinstance(x, (np.floating, float)) else int(x) for x in lst]

    output = {
        "dataset": args.dataset,
        "bins": bins.tolist(),
        "point_logit": {"accuracy": to_py(point_acc), "confidence": to_py(point_conf), "count": to_py(point_count)},
        "quadrature_avg": {"accuracy": to_py(avg_acc), "confidence": to_py(avg_conf), "count": to_py(avg_count)},
    }
    out_path = os.path.join(args.output_dir, f"reliability_{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump(output, f)
    print(f"Saved reliability data to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True,
                        choices=["chaos_energy", "reliability"])
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="results_local/viz_data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.task == "chaos_energy":
        generate_chaos_energy(args)
    elif args.task == "reliability":
        generate_reliability(args)


if __name__ == "__main__":
    main()
