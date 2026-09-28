#!/usr/bin/env python3
"""Generate Proposition 2 correction decomposition figure.

Shows c_i = (1/2) tr(H_i Sigma_{z_i}) across multiple datasets.
The correction requires both factors: the Hessian H_i (which scales with
prediction uncertainty) and the chaos covariance Sigma_{z_i} (which captures
stochastic structure). Nodes with low chaos energy get near-zero correction
regardless of prediction uncertainty.

The x-axis shows tr(H_i) = 1 - ||p_i||^2 (the Hessian trace, i.e., local
cross-entropy loss curvature).
The y-axis shows the chaos energy E_i = tr(Sigma_{z_i}).
Color shows the correction c_i.

This makes the interaction transparent: c_i ~ tr(H_i) * E_i, so large
correction only appears in the upper-right corner (high Hessian trace AND
high chaos energy).
"""

import argparse
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 9, "font.family": "serif",
    "axes.labelsize": 10, "axes.titlesize": 11,
    "legend.fontsize": 8, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default='results/correction_analysis')
    parser.add_argument('--outdir', type=str, default='tex/figures')
    parser.add_argument('--datasets', nargs='*',
                        default=['minesweeper'])
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load datasets
    npz_files = sorted(glob.glob(os.path.join(args.indir, '*_correction.npz')))
    all_data = {}
    for f in npz_files:
        d = dict(np.load(f, allow_pickle=True))
        name = str(d['dataset'])
        all_data[name] = d

    datasets = [d for d in args.datasets if d in all_data]
    n = len(datasets)

    pretty_names = {
        'cora': 'Cora', 'citeseer': 'Citeseer',
        'amazon-ratings': 'Amazon-Ratings', 'texas': 'Texas',
        'roman-empire': 'Roman-Empire', 'minesweeper': 'Minesweeper',
    }

    # Use the first dataset for both panels
    ds = datasets[0]
    data = all_data[ds]
    mask = data['split'][0] == 2

    hessian_trace = data['hessian_trace'][0][mask]
    chaos = data['chaos_energy'][0][mask]
    corr = data['correction'][0][mask]
    correct = data['correct'][0][mask]
    chaos_log = np.log10(chaos + 1e-10)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8),
                              gridspec_kw={"width_ratios": [1.2, 0.8]})

    # --- (a) Correction decomposition scatter ---
    ax = axes[0]
    vmax = max(np.percentile(corr, 97), 1e-6)
    sc = ax.scatter(hessian_trace, chaos_log, s=8, alpha=0.6,
                    c=corr, cmap='YlOrRd', edgecolors='none',
                    norm=Normalize(vmin=0, vmax=vmax),
                    rasterized=True)
    ax.set_xlabel(
        r'Loss Curvature (Hessian Trace)', fontsize=11, fontweight='bold')
    ax.set_ylabel(
        r'$\log_{10}$ Chaos Energy', fontsize=11, fontweight='bold')
    ax.set_title(f'(a) Correction decomposition ({pretty_names.get(ds, ds)})',
                 fontsize=11, fontweight='bold')
    ax.tick_params(axis='both', labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Second-order correction',
                   fontsize=10, fontweight='bold')

    # --- (b) Correction by prediction quality ---
    ax = axes[1]

    # Quartile breakdown
    q25 = np.percentile(corr, 25)
    q75 = np.percentile(corr, 75)
    low_mask = corr <= q25
    mid_mask = (corr > q25) & (corr <= q75)
    high_mask = corr > q75

    categories = ['Low $c_i$\n(Q1)', 'Mid $c_i$\n(Q2\u2013Q3)',
                  'High $c_i$\n(Q4)']
    accs = [correct[low_mask].mean(), correct[mid_mask].mean(),
            correct[high_mask].mean()]
    briers = [data['brier_point'][0][mask][low_mask].mean(),
              data['brier_point'][0][mask][mid_mask].mean(),
              data['brier_point'][0][mask][high_mask].mean()]

    x = np.arange(3)
    width = 0.35
    bars1 = ax.bar(x - width/2, [a * 100 for a in accs], width,
                   color='#6baed6', alpha=0.9, label='Accuracy (%)')
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, briers, width,
                    color='#fb6a4a', alpha=0.9, label='Brier score')

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel('Accuracy (%)', color='#2171b5', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Brier Score', color='#cb181d', fontsize=11, fontweight='bold')
    ax.set_title('(b) Correction is largest on hard nodes',
                 fontsize=11, fontweight='bold', pad=18)
    ax.tick_params(axis='both', labelsize=10)
    ax2.tick_params(axis='y', labelsize=10)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    # Legend inside panel at bottom-center (bars are short there)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9,
              loc='center', bbox_to_anchor=(0.5, 0.5),
              ncol=1, framealpha=0.95)

    plt.tight_layout()
    path = os.path.join(args.outdir, 'proposition2_figure.pdf')
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)


if __name__ == '__main__':
    main()
