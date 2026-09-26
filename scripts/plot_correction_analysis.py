#!/usr/bin/env python3
"""Plot second-order loss correction analysis figures.

Reads .npz files produced by compute_correction_analysis.py and generates:
  1. Scatter: exact quadrature loss vs second-order approximation (validates Prop 2)
  2. 2D scatter: prediction entropy × chaos energy, colored by correction magnitude
  3. Correction decomposition: Hessian trace × covariance trace vs correction
  4. Correction on correct vs incorrect predictions
  5. Brier improvement (point - integrated) vs correction
"""

import argparse
import glob
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm


def load_results(pattern):
    """Load all .npz files matching pattern, return list of dicts."""
    files = sorted(glob.glob(pattern))
    results = []
    for f in files:
        data = dict(np.load(f, allow_pickle=True))
        results.append(data)
    return results


def get_test_mask(split):
    """Return boolean mask for test nodes. split shape: (runs, N)."""
    return split == 2


def fig_quad_vs_approx(result, dataset, P, outdir):
    """Fig 1: Scatter of exact quadrature loss vs second-order approximation."""
    # Average across runs
    mask = get_test_mask(result['split'][0])
    quad = result['quad_loss'].mean(axis=0)[mask]
    approx = result['second_order_approx'].mean(axis=0)[mask]
    base = result['base_loss'].mean(axis=0)[mask]
    correction = result['correction'].mean(axis=0)[mask]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: quad vs second-order approx
    ax = axes[0]
    vmin = min(quad.min(), approx.min())
    vmax = np.percentile(np.concatenate([quad, approx]), 98)
    ax.scatter(quad, approx, s=6, alpha=0.4, c='steelblue', edgecolors='none')
    ax.plot([vmin, vmax], [vmin, vmax], 'k--', lw=1, alpha=0.7, label='$y=x$')
    ax.set_xlabel(r'Exact quadrature loss $\bar{\ell}_i$', fontsize=11)
    ax.set_ylabel(r'Second-order approx $\ell_i + c_i$', fontsize=11)
    ax.set_title('Proposition 2 validation', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect('equal')

    # Inset: residual histogram
    residual = approx - quad
    inset = ax.inset_axes([0.55, 0.08, 0.42, 0.35])
    inset.hist(residual, bins=50, color='steelblue', alpha=0.7, density=True)
    inset.axvline(0, color='k', ls='--', lw=0.8)
    inset.set_xlabel('Residual', fontsize=7)
    inset.set_ylabel('Density', fontsize=7)
    inset.tick_params(labelsize=6)

    # Right: correction magnitude vs base loss
    ax = axes[1]
    sc = ax.scatter(base, correction, s=6, alpha=0.4,
                    c=result['chaos_energy'].mean(axis=0)[mask],
                    cmap='viridis', edgecolors='none')
    ax.set_xlabel(r'Base loss $\ell(z_{0,i}, y_i)$', fontsize=11)
    ax.set_ylabel(r'Correction $c_i$', fontsize=11)
    ax.set_title('Correction vs base loss', fontsize=12)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(r'Chaos energy $\mathcal{E}_i$', fontsize=9)

    fig.suptitle(f'{dataset} (P={P})', fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(outdir, f'correction_validation_{dataset}_P{P}.pdf')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


def fig_entropy_chaos_correction(result, dataset, P, outdir):
    """Fig 2: 2D scatter of prediction entropy × chaos energy, colored by correction."""
    mask = get_test_mask(result['split'][0])
    entropy = result['pred_entropy'].mean(axis=0)[mask]
    chaos = result['chaos_energy'].mean(axis=0)[mask]
    corr = result['correction'].mean(axis=0)[mask]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    # Use log scale for chaos energy if range is large
    chaos_plot = np.log10(chaos + 1e-10)

    sc = ax.scatter(entropy, chaos_plot, s=8, alpha=0.5,
                    c=corr, cmap='magma', edgecolors='none',
                    norm=Normalize(vmin=0, vmax=np.percentile(corr, 95)))
    ax.set_xlabel(r'Prediction entropy $H(p_i)$', fontsize=11)
    ax.set_ylabel(r'$\log_{10}$ Chaos energy $\mathcal{E}_i$', fontsize=11)
    ax.set_title(f'Correction decomposition ({dataset}, P={P})', fontsize=12)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(r'Correction $c_i = \frac{1}{2}\mathrm{tr}(H_i \Sigma_i)$',
                   fontsize=9)

    plt.tight_layout()
    path = os.path.join(outdir, f'correction_decomposition_{dataset}_P{P}.pdf')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


def fig_correct_vs_incorrect(result, dataset, P, outdir):
    """Fig 3: Correction, chaos energy, entropy distributions for correct vs incorrect."""
    mask = get_test_mask(result['split'][0])
    correct = result['correct'].mean(axis=0)[mask] > 0.5  # majority vote across runs
    corr = result['correction'].mean(axis=0)[mask]
    chaos = result['chaos_energy'].mean(axis=0)[mask]
    entropy = result['pred_entropy'].mean(axis=0)[mask]

    quantities = [
        (corr, r'Correction $c_i$', 'correction'),
        (chaos, r'Chaos energy $\mathcal{E}_i$', 'chaos_energy'),
        (entropy, r'Prediction entropy $H(p_i)$', 'entropy'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, (vals, xlabel, name) in zip(axes, quantities):
        vmax = np.percentile(vals, 98)
        bins = np.linspace(0, vmax, 40)
        ax.hist(vals[correct], bins=bins, alpha=0.6, density=True,
                label='Correct', color='#2ecc71')
        ax.hist(vals[~correct], bins=bins, alpha=0.6, density=True,
                label='Incorrect', color='#e74c3c')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=8)
        # Add median lines
        ax.axvline(np.median(vals[correct]), color='#2ecc71', ls='--', lw=1.2)
        ax.axvline(np.median(vals[~correct]), color='#e74c3c', ls='--', lw=1.2)

    fig.suptitle(f'Correct vs Incorrect predictions ({dataset}, P={P})',
                 fontsize=12)
    plt.tight_layout()
    path = os.path.join(outdir, f'correction_correct_incorrect_{dataset}_P{P}.pdf')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


def fig_brier_improvement(result, dataset, P, outdir):
    """Fig 4: Per-node Brier improvement (point - integrated) vs correction."""
    mask = get_test_mask(result['split'][0])
    brier_pt = result['brier_point'].mean(axis=0)[mask]
    brier_int = result['brier_integrated'].mean(axis=0)[mask]
    corr = result['correction'].mean(axis=0)[mask]
    improvement = brier_pt - brier_int  # positive = integrated is better

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: scatter
    ax = axes[0]
    sc = ax.scatter(corr, improvement, s=6, alpha=0.4, c='steelblue',
                    edgecolors='none')
    ax.axhline(0, color='k', ls='--', lw=0.8, alpha=0.5)
    ax.set_xlabel(r'Correction $c_i$', fontsize=11)
    ax.set_ylabel(r'Brier improvement (point $-$ integrated)', fontsize=11)
    ax.set_title('Calibration gain vs correction', fontsize=12)

    # Bin and show trend
    nbins = 20
    valid = np.isfinite(corr) & np.isfinite(improvement)
    if valid.sum() > nbins * 5:
        bins = np.percentile(corr[valid], np.linspace(0, 100, nbins + 1))
        bin_idx = np.digitize(corr, bins) - 1
        bin_means = []
        bin_centers = []
        for b in range(nbins):
            in_bin = (bin_idx == b) & valid
            if in_bin.sum() > 3:
                bin_means.append(improvement[in_bin].mean())
                bin_centers.append(corr[in_bin].mean())
        ax.plot(bin_centers, bin_means, 'o-', color='orangered', lw=2,
                markersize=4, label='Binned mean')
        ax.legend(fontsize=9)

    # Right: histogram of improvement
    ax = axes[1]
    ax.hist(improvement, bins=50, color='steelblue', alpha=0.7, density=True)
    ax.axvline(0, color='k', ls='--', lw=0.8)
    ax.axvline(improvement.mean(), color='orangered', ls='-', lw=1.5,
               label=f'Mean = {improvement.mean():.4f}')
    ax.set_xlabel(r'Brier improvement (point $-$ integrated)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Distribution of calibration gain', fontsize=12)
    ax.legend(fontsize=9)

    fig.suptitle(f'{dataset} (P={P})', fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(outdir, f'correction_brier_{dataset}_P{P}.pdf')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


def fig_degree_dependence(result, dataset, P, outdir):
    """Fig 5: Correction vs node degree (topology-awareness)."""
    mask = get_test_mask(result['split'][0])
    corr = result['correction'].mean(axis=0)[mask]
    chaos = result['chaos_energy'].mean(axis=0)[mask]
    degree = result['degree'].mean(axis=0)[mask]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, (vals, ylabel) in zip(axes, [
        (corr, r'Correction $c_i$'),
        (chaos, r'Chaos energy $\mathcal{E}_i$'),
    ]):
        ax.scatter(degree, vals, s=6, alpha=0.3, c='steelblue',
                   edgecolors='none')
        ax.set_xlabel('Node degree', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)

        # Binned trend
        unique_deg = np.unique(degree.astype(int))
        if len(unique_deg) > 5:
            nbins = min(20, len(unique_deg))
            bins = np.percentile(degree, np.linspace(0, 100, nbins + 1))
            bin_idx = np.digitize(degree, bins) - 1
            bin_means, bin_centers = [], []
            for b in range(nbins):
                in_bin = bin_idx == b
                if in_bin.sum() > 3:
                    bin_means.append(vals[in_bin].mean())
                    bin_centers.append(degree[in_bin].mean())
            ax.plot(bin_centers, bin_means, 'o-', color='orangered', lw=2,
                    markersize=5, label='Binned mean')
            ax.legend(fontsize=9)

    fig.suptitle(f'Topology dependence ({dataset}, P={P})', fontsize=12)
    plt.tight_layout()
    path = os.path.join(outdir, f'correction_degree_{dataset}_P{P}.pdf')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


def fig_multi_dataset_summary(results_list, outdir):
    """Summary figure across multiple datasets."""
    if len(results_list) < 2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    colors = plt.cm.Set2(np.linspace(0, 1, len(results_list)))
    for i, result in enumerate(results_list):
        dataset = str(result['dataset'])
        P = int(result['P'])
        mask = get_test_mask(result['split'][0])
        quad = result['quad_loss'].mean(axis=0)[mask]
        approx = result['second_order_approx'].mean(axis=0)[mask]
        corr = result['correction'].mean(axis=0)[mask]
        brier_pt = result['brier_point'].mean(axis=0)[mask]
        brier_int = result['brier_integrated'].mean(axis=0)[mask]

        label = f'{dataset} (P={P})'

        # Panel 1: R^2 of second-order approx
        ss_res = ((approx - quad) ** 2).sum()
        ss_tot = ((quad - quad.mean()) ** 2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-12)
        axes[0].bar(i, r2, color=colors[i], label=label, alpha=0.8)

        # Panel 2: Mean correction
        axes[1].bar(i, corr.mean(), color=colors[i], alpha=0.8,
                    yerr=corr.std() / np.sqrt(len(corr)))

        # Panel 3: Mean Brier improvement
        improvement = (brier_pt - brier_int).mean()
        axes[2].bar(i, improvement, color=colors[i], alpha=0.8)

    axes[0].set_ylabel(r'$R^2$ (approx vs exact)', fontsize=10)
    axes[0].set_title('Proposition 2 accuracy', fontsize=11)
    axes[0].set_ylim(0, 1.05)

    axes[1].set_ylabel(r'Mean correction $\bar{c}$', fontsize=10)
    axes[1].set_title('Correction magnitude', fontsize=11)

    axes[2].set_ylabel('Mean Brier improvement', fontsize=10)
    axes[2].set_title('Calibration gain', fontsize=11)
    axes[2].axhline(0, color='k', ls='--', lw=0.8)

    for ax in axes:
        ax.set_xticks(range(len(results_list)))
        ax.set_xticklabels(
            [f"{str(r['dataset'])}\nP={int(r['P'])}" for r in results_list],
            fontsize=8, rotation=0)

    plt.tight_layout()
    path = os.path.join(outdir, 'correction_summary.pdf')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default='results/correction_analysis')
    parser.add_argument('--outdir', type=str, default='tex/figures')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    npz_files = sorted(glob.glob(os.path.join(args.indir, '*_correction.npz')))
    if not npz_files:
        print(f'No .npz files found in {args.indir}')
        return

    all_results = []
    for f in npz_files:
        print(f'\nProcessing {os.path.basename(f)}')
        result = dict(np.load(f, allow_pickle=True))
        dataset = str(result['dataset'])
        P = int(result['P'])
        all_results.append(result)

        fig_quad_vs_approx(result, dataset, P, args.outdir)
        fig_entropy_chaos_correction(result, dataset, P, args.outdir)
        fig_correct_vs_incorrect(result, dataset, P, args.outdir)
        fig_brier_improvement(result, dataset, P, args.outdir)
        fig_degree_dependence(result, dataset, P, args.outdir)

    fig_multi_dataset_summary(all_results, args.outdir)
    print('\nDone.')


if __name__ == '__main__':
    main()
