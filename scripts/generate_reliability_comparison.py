#!/usr/bin/env python3
"""
Generate reliability diagram data comparing DSS-GNN vs TFE-GNN vs G-DUQ.
Runs all three models on the same dataset and saves binned confidence vs accuracy.

Usage:
  python scripts/generate_reliability_comparison.py --dataset texas --output_dir results_local/viz_data
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

from tfe_utils import set_seed, load_data, accuracy
from dssgnn.chebyshev import build_rescaled_laplacian
from dssgnn.model import DSSGNN
from tfe_models import TFE_GNN


def train_and_get_probs(model_type, dataset, device, seed=0):
    """Train a model and return test probabilities and labels."""
    set_seed(seed)
    adj, features, labels, train_mask, val_mask, test_mask = load_data(
        dataset, True, True, 0.6, 0.2, seed
    )
    num_classes = int(max(labels)) + 1
    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    if model_type == "dssgnn":
        from tfe_utils import propagate_adj
        model = DSSGNN(
            input_dim=features.shape[1], hidden_dim=64, out_dim=num_classes,
            num_layers=2, K_lp=4, K_hp=4, P=2, dropout=(0.5, 0.0), activation=True
        ).to(device)
        graph_input = build_rescaled_laplacian(adj, lambda_max=2.0).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        best_val_loss = float("inf")
        best_state = None
        for epoch in range(500):
            model.train()
            optimizer.zero_grad()
            logits, unc = model(graph_input, features, return_uncertainty=True)
            loss = F.cross_entropy(logits[train_mask], labels[train_mask]) + 0.01 * unc.mean()
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                logits, _ = model(graph_input, features, return_uncertainty=True)
                vl = F.cross_entropy(logits[val_mask], labels[val_mask]).item()
            if vl < best_val_loss:
                best_val_loss = vl
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        model.to(device).eval()
        with torch.no_grad():
            logits, _ = model(graph_input, features, return_uncertainty=True)
            probs = F.softmax(logits, dim=1)

    elif model_type == "tfe":
        from tfe_utils import propagate_adj
        model = TFE_GNN(
            features.shape[1], 512, num_classes, 2, [0.6, 0.0], True, [6, 5], "sum"
        ).to(device)
        adj_lp = propagate_adj(adj, "low", -0.5, -0.5).to(device)
        adj_hp = propagate_adj(adj, "high", -0.3, -0.3).to(device)
        optimizer = torch.optim.Adam([
            {"params": model.adaptive, "lr": 0.1, "weight_decay": 0.05},
            {"params": model.adaptive_lp, "lr": 0.1, "weight_decay": 0.05},
            {"params": model.layers.parameters(), "lr": 0.005, "weight_decay": 0.0},
            {"params": model.ense_coe, "lr": 0.0, "weight_decay": 0.0},
        ])

        best_val_loss = float("inf")
        best_state = None
        for epoch in range(500):
            model.train()
            optimizer.zero_grad()
            out = F.log_softmax(model(adj_hp, adj_lp, features), dim=1)
            loss = F.cross_entropy(out[train_mask], labels[train_mask])
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                out = F.log_softmax(model(adj_hp, adj_lp, features), dim=1)
                vl = F.cross_entropy(out[val_mask], labels[val_mask]).item()
            if vl < best_val_loss:
                best_val_loss = vl
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        model.to(device).eval()
        with torch.no_grad():
            logits = model(adj_hp, adj_lp, features)
            probs = F.softmax(logits, dim=1)

    test_idx = test_mask.nonzero(as_tuple=True)[0].cpu()
    test_probs = probs[test_idx].cpu().numpy()
    test_labels = labels[test_idx].cpu().numpy()
    return test_probs, test_labels


def compute_reliability_bins(probs, labels, n_bins=10):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_acc, bin_conf, bin_count = [], [], []
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() > 0:
            bin_acc.append(float(correct[mask].mean()))
            bin_conf.append(float(confidences[mask].mean()))
            bin_count.append(int(mask.sum()))
        else:
            bin_acc.append(0.0)
            bin_conf.append(float((bins[i] + bins[i + 1]) / 2))
            bin_count.append(0)
    acc = float(correct.mean())
    brier = float(((probs - np.eye(probs.shape[1])[labels]) ** 2).sum(axis=1).mean())
    return {
        "accuracy": bin_acc, "confidence": bin_conf, "count": bin_count,
        "overall_acc": acc, "brier": brier
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="texas")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="results_local/viz_data")
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    output = {"dataset": args.dataset, "models": {}}

    for model_type in ["dssgnn", "tfe"]:
        all_bins = []
        for seed in range(args.seeds):
            probs, labels = train_and_get_probs(model_type, args.dataset, device, seed)
            bins = compute_reliability_bins(probs, labels)
            all_bins.append(bins)
            print(f"  {model_type} seed={seed}: acc={bins['overall_acc']:.4f} brier={bins['brier']:.4f}")

        # Average across seeds
        avg = {
            "accuracy": [float(np.mean([b["accuracy"][i] for b in all_bins])) for i in range(10)],
            "confidence": all_bins[0]["confidence"],
            "count": [int(np.mean([b["count"][i] for b in all_bins])) for i in range(10)],
            "overall_acc": float(np.mean([b["overall_acc"] for b in all_bins])),
            "brier": float(np.mean([b["brier"] for b in all_bins])),
        }
        output["models"][model_type] = avg

    out_path = os.path.join(args.output_dir, f"reliability_comparison_{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump(output, f)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
