#!/usr/bin/env python3
"""
Plot OOD score density plots from saved score data.

Usage:
  python scripts/plot_ood_density.py --data_dir results_local/ood_scores --output_dir tex/figures
"""

import argparse
import json
import os
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

METHOD_LABELS = {
    "gnnsafe_gcn": "GNNSafe",
    "gnnsafe++_gcn": "GNNSafe++",
    "graph_ebm_gcn": "Graph-EBM",
    "gnnsafe++_gcn_dssres": "DSS-Hybrid",
}

METHOD_COLORS = {
    "GNNSafe": "#999999",
    "GNNSafe++": "#d95f02",
    "Graph-EBM": "#7570b3",
    "DSS-Hybrid": "#2b8cbe",
}


def plot_density_for_shift(data_dir, ood_type, output_dir):
    """Plot ID vs OOD score densities for all methods on one shift type."""
    files = glob.glob(os.path.join(data_dir, f"scores_cora_{ood_type}_*.json"))
    if not files:
        print(f"  No data for cora/{ood_type}")
        return

    methods = {}
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        method_key = d["method"]
        label = METHOD_LABELS.get(method_key, method_key)
        methods[label] = d

    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(2.5 * n_methods, 2.2),
                              sharey=True)
    if n_methods == 1:
        axes = [axes]

    # Order: GNNSafe, GNNSafe++, Graph-EBM, DSS-Hybrid
    order = ["GNNSafe", "GNNSafe++", "Graph-EBM", "DSS-Hybrid"]
    ordered = [(name, methods[name]) for name in order if name in methods]

    for idx, (name, data) in enumerate(ordered):
        ax = axes[idx]
        id_scores = np.array(data["id_scores"])
        ood_scores = np.array(data["ood_scores"])

        # Remove outliers for KDE
        all_scores = np.concatenate([id_scores, ood_scores])
        lo, hi = np.percentile(all_scores, 1), np.percentile(all_scores, 99)
        x_range = np.linspace(lo, hi, 200)

        try:
            kde_id = gaussian_kde(id_scores, bw_method=0.3)
            kde_ood = gaussian_kde(ood_scores, bw_method=0.3)
            ax.fill_between(x_range, kde_id(x_range), alpha=0.4, color="#2166ac",
                            label="ID")
            ax.fill_between(x_range, kde_ood(x_range), alpha=0.4, color="#b2182b",
                            label="OOD")
            ax.plot(x_range, kde_id(x_range), color="#2166ac", linewidth=1)
            ax.plot(x_range, kde_ood(x_range), color="#b2182b", linewidth=1)
        except Exception:
            ax.hist(id_scores, bins=30, alpha=0.4, color="#2166ac",
                    label="ID", density=True)
            ax.hist(ood_scores, bins=30, alpha=0.4, color="#b2182b",
                    label="OOD", density=True)

        ax.set_title(name, fontweight="bold", fontsize=9)
        ax.set_xlabel("Energy score")
        if idx == 0:
            ax.set_ylabel("Density")
            ax.legend(fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_yticks([])

    fig.suptitle(f"Cora / {ood_type} shift", y=1.03, fontsize=11)
    plt.tight_layout()
    path = os.path.join(output_dir, f"ood_density_cora_{ood_type}.pdf")
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="results_local/ood_scores")
    parser.add_argument("--output_dir", type=str, default="tex/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for ood_type in ["structure", "feature", "label"]:
        plot_density_for_shift(args.data_dir, ood_type, args.output_dir)


if __name__ == "__main__":
    main()
