#!/usr/bin/env python3
"""
Plot chaos energy heatmap and reliability diagrams from generated data.

Usage:
  python scripts/plot_viz_figures.py --output_dir tex/figures
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def plot_chaos_energy_heatmap(data_path, output_dir):
    """Plot chaos energy as a scatter on a 2D graph layout."""
    with open(data_path) as f:
        data = json.load(f)

    dataset = data["dataset"]
    chaos_energy = np.array(data["chaos_energy"])
    labels = np.array(data["labels"])
    correct = np.array(data["correct"])
    test_mask = np.array(data["test_mask"], dtype=bool)

    if chaos_energy.max() < 1e-8:
        print(f"  Skipping {dataset}: chaos energy is all zeros")
        return

    # Use a simple spectral layout from the adjacency
    edge_src = np.array(data.get("edge_src", []))
    edge_dst = np.array(data.get("edge_dst", []))
    N = len(chaos_energy)

    # Compute a quick 2D layout using eigenvectors of the Laplacian
    import scipy.sparse as sp
    if len(edge_src) > 0:
        adj = sp.csr_matrix((np.ones(len(edge_src)), (edge_src, edge_dst)), shape=(N, N))
        adj = adj + adj.T
        adj.data = np.minimum(adj.data, 1)
        degree = np.array(adj.sum(axis=1)).flatten()
        degree[degree == 0] = 1
        D_inv_sqrt = sp.diags(1.0 / np.sqrt(degree))
        L_norm = sp.eye(N) - D_inv_sqrt @ adj @ D_inv_sqrt

        from scipy.sparse.linalg import eigsh
        try:
            _, vecs = eigsh(L_norm.astype(np.float64), k=3, which="SM", maxiter=5000)
            pos = vecs[:, 1:3]  # Fiedler vector + next
        except Exception:
            np.random.seed(42)
            pos = np.random.randn(N, 2)
    else:
        np.random.seed(42)
        pos = np.random.randn(N, 2)

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.0))

    # Panel 1: Chaos energy heatmap
    ax = axes[0]
    ce_clipped = np.clip(chaos_energy, 0, np.percentile(chaos_energy, 98))
    scatter = ax.scatter(pos[:, 0], pos[:, 1], c=ce_clipped, cmap="YlOrRd",
                         s=3, alpha=0.7, edgecolors="none", rasterized=True)
    plt.colorbar(scatter, ax=ax, shrink=0.8, label="Chaos energy $\\mathcal{E}_i$")
    ax.set_title("Chaos Energy", fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    # Panel 2: Correct vs incorrect (test nodes only)
    ax = axes[1]
    # Draw all nodes faintly
    ax.scatter(pos[:, 0], pos[:, 1], c="#eeeeee", s=2, alpha=0.3, edgecolors="none", rasterized=True)
    # Test correct in blue, incorrect in red
    test_idx = np.where(test_mask)[0]
    correct_idx = test_idx[correct[test_idx] == 1]
    wrong_idx = test_idx[correct[test_idx] == 0]
    ax.scatter(pos[correct_idx, 0], pos[correct_idx, 1], c="#2166ac", s=3, alpha=0.5,
               label=f"Correct ({len(correct_idx)})", edgecolors="none", rasterized=True)
    ax.scatter(pos[wrong_idx, 0], pos[wrong_idx, 1], c="#b2182b", s=8, alpha=0.8,
               label=f"Wrong ({len(wrong_idx)})", edgecolors="none", marker="x", rasterized=True)
    ax.legend(loc="upper right", markerscale=2, framealpha=0.9)
    ax.set_title("Test Predictions", fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    # Panel 3: Energy vs correctness (scatter)
    ax = axes[2]
    test_energy = chaos_energy[test_mask]
    test_correct = correct[test_mask]
    bins = np.linspace(0, np.percentile(test_energy, 99), 20)
    correct_counts = np.histogram(test_energy[test_correct == 1], bins=bins)[0]
    wrong_counts = np.histogram(test_energy[test_correct == 0], bins=bins)[0]
    bin_centers = (bins[:-1] + bins[1:]) / 2
    total = correct_counts + wrong_counts
    total[total == 0] = 1
    error_rate = wrong_counts / total
    ax.bar(bin_centers, error_rate, width=bins[1]-bins[0], color="#d95f02", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Chaos energy $\\mathcal{E}_i$")
    ax.set_ylabel("Error rate")
    ax.set_title("Energy vs Error Rate", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(f"DSS-GNN Uncertainty Structure on {dataset.capitalize()}", y=1.02, fontsize=11)
    plt.tight_layout()
    path = os.path.join(output_dir, f"chaos_energy_{dataset}.pdf")
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)


def plot_reliability(data_path, output_dir):
    """Plot reliability diagram comparing point logit vs quadrature average."""
    with open(data_path) as f:
        data = json.load(f)

    dataset = data["dataset"]
    bins = np.array(data["bins"])
    bin_centers = (bins[:-1] + bins[1:]) / 2

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0))

    # Diagonal (perfect calibration)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Perfect calibration")

    # Point logit
    point = data["point_logit"]
    point_acc = np.array(point["accuracy"][:len(bin_centers)])
    point_count = np.array(point["count"][:len(bin_centers)])
    mask_p = point_count > 0
    ax.bar(bin_centers[mask_p] - 0.02, point_acc[mask_p],
           width=0.04, alpha=0.6, color="#2b8cbe", label="Point logit")

    # Quadrature average
    quad = data["quadrature_avg"]
    quad_acc = np.array(quad["accuracy"][:len(bin_centers)])
    quad_count = np.array(quad["count"][:len(bin_centers)])
    mask_q = quad_count > 0
    ax.bar(bin_centers[mask_q] + 0.02, quad_acc[mask_q],
           width=0.04, alpha=0.6, color="#d95f02", label="Quadrature avg")

    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Actual accuracy")
    ax.set_title(f"Reliability Diagram — {dataset.capitalize()}", fontweight="bold")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, f"reliability_{dataset}.pdf")
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="results_local/viz_data")
    parser.add_argument("--output_dir", type=str, default="tex/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for ds in ["cora", "texas"]:
        ce_path = os.path.join(args.data_dir, f"chaos_energy_{ds}.json")
        if os.path.exists(ce_path):
            plot_chaos_energy_heatmap(ce_path, args.output_dir)

        rel_path = os.path.join(args.data_dir, f"reliability_{ds}.json")
        if os.path.exists(rel_path):
            plot_reliability(rel_path, args.output_dir)


if __name__ == "__main__":
    main()
