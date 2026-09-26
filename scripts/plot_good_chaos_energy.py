#!/usr/bin/env python3
"""Plot chaos energy vs node degree on GOOD-Arxiv/degree.

Shows that chaos energy is higher on nodes whose degree deviates from the
training distribution, demonstrating topology-aware uncertainty.
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 10, "font.family": "serif",
    "axes.labelsize": 11, "axes.titlesize": 11,
    "legend.fontsize": 9, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        default='results/good_chaos_energy/good_arxiv_degree_P2.npz')
    parser.add_argument('--outdir', type=str, default='tex/figures')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    data = np.load(args.input)

    chaos = data['chaos_energy']
    degrees = data['degrees']
    env = data['env']  # 0=train, 1=val, 2=test
    correct = data['correct']

    # Log-transform degree for better visualization
    log_deg = np.log10(degrees + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # --- Panel (a): Chaos energy vs degree, colored by split ---
    ax = axes[0]
    colors = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}
    labels = {0: 'Train', 1: 'Val', 2: 'Test (shifted)'}
    # Use log scale for chaos energy (clip outliers)
    chaos_plot = np.log10(chaos + 1e-6)
    for split_id in [0, 2]:  # Show train and test only for clarity
        mask_s = env == split_id
        ax.scatter(log_deg[mask_s], chaos_plot[mask_s], s=2, alpha=0.15,
                   c=colors[split_id], label=labels[split_id], rasterized=True)

    # Binned means
    for split_id, ls in [(0, '-'), (2, '--')]:
        mask_s = env == split_id
        if mask_s.sum() < 10:
            continue
        bins = np.percentile(log_deg[mask_s], np.linspace(0, 100, 20))
        bin_idx = np.digitize(log_deg[mask_s], bins) - 1
        bin_means, bin_centers = [], []
        for b in range(len(bins) - 1):
            in_bin = bin_idx == b
            if in_bin.sum() > 5:
                bin_means.append(chaos_plot[mask_s][in_bin].mean())
                bin_centers.append(log_deg[mask_s][in_bin].mean())
        ax.plot(bin_centers, bin_means, ls, color=colors[split_id],
                linewidth=2.5, zorder=5)

    ax.set_xlabel(r"$\log_{10}$(degree + 1)", fontsize=11)
    ax.set_ylabel(r"$\log_{10}$ Chaos energy $\mathcal{E}_i$", fontsize=11)
    ax.set_title("(a) Chaos energy vs node degree")
    ax.legend(markerscale=5, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Panel (b): Chaos energy distribution by split (log scale) ---
    ax = axes[1]
    for split_id in [0, 2]:
        mask_s = env == split_id
        vals = chaos_plot[mask_s]
        lo_h, hi_h = np.percentile(vals, [1, 99])
        bins = np.linspace(lo_h, hi_h, 50)
        ax.hist(vals, bins=bins, density=True, alpha=0.5,
                color=colors[split_id], label=labels[split_id])
        ax.axvline(vals.mean(), color=colors[split_id], ls='--', lw=1.5)

    ax.set_xlabel(r"$\log_{10}$ Chaos energy $\mathcal{E}_i$", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("(b) Distribution by environment")
    ax.legend(framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(args.outdir, 'good_chaos_energy_degree.pdf')
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)


if __name__ == '__main__':
    main()
