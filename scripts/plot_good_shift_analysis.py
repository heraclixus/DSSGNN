#!/usr/bin/env python3
"""GOOD-Arxiv/degree: show chaos energy tracks the distribution shift.

Panel (a): Test accuracy by degree bin -- shows where shift hurts
Panel (b): Chaos energy by degree bin for correct vs incorrect test nodes
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
    env = data['env']
    correct = data['correct']

    # Define degree bins (log-spaced)
    log_deg = np.log10(degrees + 1)
    bin_edges = np.percentile(log_deg[env >= 0], np.linspace(0, 100, 9))
    bin_edges = np.unique(np.round(bin_edges, 2))
    nbins = len(bin_edges) - 1

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # === Panel (a): Accuracy by degree bin, train vs test ===
    ax = axes[0]
    train_accs, test_accs, bin_labels = [], [], []
    for i in range(nbins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask_bin = (log_deg >= lo) & (log_deg < hi)
        train_in = mask_bin & (env == 0)
        test_in = mask_bin & (env == 2)
        if train_in.sum() > 50 and test_in.sum() > 50:
            train_accs.append(correct[train_in].mean() * 100)
            test_accs.append(correct[test_in].mean() * 100)
            deg_lo = int(10 ** lo - 1)
            deg_hi = int(10 ** hi - 1)
            bin_labels.append(f'{deg_lo}-{deg_hi}')

    x = np.arange(len(train_accs))
    width = 0.35
    ax.bar(x - width / 2, train_accs, width, color='#6baed6', alpha=0.9,
           label='Train (ID)')
    ax.bar(x + width / 2, test_accs, width, color='#fb6a4a', alpha=0.9,
           label='Test (shifted)')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=8, rotation=20, ha='right')
    ax.set_xlabel('Degree range', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('(a) Shift impact by degree', fontweight='bold', fontsize=11)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # === Panel (b): Chaos energy distribution for hard vs easy shifted nodes ===
    ax = axes[1]
    from scipy.stats import gaussian_kde

    test_mask = env == 2
    test_chaos = chaos[test_mask]
    test_logdeg = log_deg[test_mask]
    test_correct = correct[test_mask]

    # Hard-shift: low degrees (0-6) where accuracy drops 15-20pp
    hard = test_logdeg < np.log10(6 + 1)
    # Easy-shift: mid degrees (6-24) where gap is near zero
    easy = (test_logdeg >= np.log10(6 + 1)) & (test_logdeg < np.log10(24 + 1))

    log_ce = np.log10(test_chaos + 1e-8)
    lo_h, hi_h = np.percentile(log_ce, [1, 99])
    xs = np.linspace(lo_h, hi_h, 200)

    kde_hard = gaussian_kde(log_ce[hard], bw_method='silverman')
    kde_easy = gaussian_kde(log_ce[easy], bw_method='silverman')

    ax.fill_between(xs, kde_hard(xs), alpha=0.4, color='#fb6a4a',
                     label=f'Hard shift (deg < 6, acc={test_correct[hard].mean()*100:.0f}%)')
    ax.plot(xs, kde_hard(xs), color='#fb6a4a', lw=1.5)
    ax.fill_between(xs, kde_easy(xs), alpha=0.4, color='#6baed6',
                     label=f'Easy shift (deg 6-24, acc={test_correct[easy].mean()*100:.0f}%)')
    ax.plot(xs, kde_easy(xs), color='#6baed6', lw=1.5)

    # Mean lines
    ax.axvline(log_ce[hard].mean(), color='#fb6a4a', ls='--', lw=1.5)
    ax.axvline(log_ce[easy].mean(), color='#6baed6', ls='--', lw=1.5)

    ax.set_xlabel(r'$\log_{10}$ Chaos energy $\mathcal{E}_i$', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(b) Chaos energy on shifted test nodes',
                 fontweight='bold', fontsize=11)
    ax.legend(fontsize=8, loc='upper right')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(args.outdir, 'good_shift_analysis.pdf')
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)


if __name__ == '__main__':
    main()
