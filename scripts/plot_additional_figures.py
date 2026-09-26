#!/usr/bin/env python3
"""
Generate additional visualization figures from grid results.

Usage:
  python scripts/plot_additional_figures.py --output_dir tex/figures
"""

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def load_grid_results(path):
    results = []
    with open(path) as f:
        for line in f:
            d = {}
            for m in re.finditer(r'(\w+)=([\w./\-]+)', line):
                k, v = m.group(1), m.group(2)
                try:
                    v = float(v)
                except ValueError:
                    pass
                d[k] = v
            results.append(d)
    return results


def plot_p_sweep_lines(results, output_dir):
    """P sweep line plot: Brier vs P, one line per dataset."""
    cal = [r for r in results if r.get('type') == 'calibration']
    by_ds = defaultdict(dict)
    for r in cal:
        cfg = r.get('config', '')
        if '_lreg0.01_lossmean' not in cfg:
            continue
        P_match = re.search(r'_P(\d)_', cfg)
        if not P_match:
            continue
        P = int(P_match.group(1))
        ds = r['dataset']
        brier = r.get('brier_mean', None)
        if brier is not None:
            if P not in by_ds[ds] or brier < by_ds[ds][P]:
                by_ds[ds][P] = brier

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5))

    homo = ['cora', 'citeseer', 'pubmed', 'cs']
    hetero = ['texas', 'cornell', 'wisconsin', 'chameleon', 'squirrel',
              'roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions']

    # Filter out datasets with collapsed runs (any P with Brier > 0.75)
    skip = set()
    for ds in by_ds:
        for P in [0, 1, 2, 3]:
            if by_ds[ds].get(P, 0) > 0.75:
                skip.add(ds)

    P_values = [0, 1, 2, 3]
    for ds in sorted(by_ds):
        if ds in skip:
            continue
        vals = [by_ds[ds].get(P, None) for P in P_values]
        if any(v is None for v in vals):
            continue
        is_homo = ds in homo
        color = '#2b8cbe' if is_homo else '#d95f02'
        style = '-' if is_homo else '--'
        marker = 'o' if is_homo else 's'
        # Label key datasets
        labeled = ['cora', 'texas', 'cornell', 'pubmed', 'squirrel', 'citeseer']
        label = ds if ds in labeled else None
        ax.plot(P_values, vals, marker=marker, color=color, linestyle=style,
                markersize=4, linewidth=1.2, alpha=0.7, label=label)

    # Add legend entries for categories
    ax.plot([], [], 'o-', color='#2b8cbe', label='Homophilous', linewidth=1.2)
    ax.plot([], [], 's--', color='#d95f02', label='Heterophilous', linewidth=1.2)
    ax.set_xlabel('Chaos order $P$')
    ax.set_ylabel('Brier Score ($\\downarrow$)')
    ax.set_xticks(P_values)
    ax.legend(loc='upper right', ncol=2, framealpha=0.9, fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if skip:
        print(f'  Filtered out collapsed datasets: {skip}')

    path = os.path.join(output_dir, 'p_sweep_lines.pdf')
    fig.savefig(path)
    print(f'Saved {path}')
    plt.close(fig)

    # Box plot of Brier change (P>0 minus P=0) across datasets
    fig, ax = plt.subplots(1, 1, figsize=(4.0, 2.8))
    box_data = {1: [], 2: [], 3: []}
    for ds in sorted(by_ds):
        if ds in skip:
            continue
        vals = [by_ds[ds].get(P, None) for P in P_values]
        if any(v is None for v in vals) or vals[0] == 0:
            continue
        p0 = vals[0]
        for P in [1, 2, 3]:
            delta = (p0 - vals[P]) / p0 * 100  # % improvement
            if abs(delta) < 200:  # filter extreme collapses
                box_data[P].append(delta)

    positions = [1, 2, 3]
    bp = ax.boxplot([box_data[P] for P in positions], positions=positions,
                     widths=0.5, patch_artist=True,
                     boxprops=dict(facecolor='#74a9cf', alpha=0.7),
                     medianprops=dict(color='black', linewidth=1.5),
                     whiskerprops=dict(color='#333333'),
                     flierprops=dict(marker='o', markersize=3, alpha=0.5))
    ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
    ax.set_xlabel('Chaos order $P$')
    ax.set_ylabel('Brier improvement\\nover $P{=}0$ (\\%)')
    ax.set_xticks(positions)
    ax.set_xticklabels(['$P$=1', '$P$=2', '$P$=3'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotate median
    for P in positions:
        med = np.median(box_data[P])
        n_pos = sum(1 for v in box_data[P] if v > 0)
        n_tot = len(box_data[P])
        ax.text(P, ax.get_ylim()[1] * 0.9, f'{n_pos}/{n_tot}',
                ha='center', fontsize=8, color='#045a8d')

    path = os.path.join(output_dir, 'p_sweep_boxplot.pdf')
    fig.savefig(path)
    print(f'Saved {path}')
    plt.close(fig)


def plot_p_lreg_heatmap(results, output_dir):
    """P x lambda_reg heatmap for representative datasets."""
    cal = [r for r in results if r.get('type') == 'calibration']
    datasets = ['cora', 'texas', 'chameleon', 'amazon-ratings']
    P_values = [0, 1, 2, 3]
    lreg_values = [0.0, 0.01]

    fig, axes = plt.subplots(1, len(datasets), figsize=(8, 2.0))

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        grid = np.full((len(lreg_values), len(P_values)), np.nan)
        for r in cal:
            if r.get('dataset') != ds:
                continue
            cfg = r.get('config', '')
            if '_lossmean' not in cfg:
                continue
            P_match = re.search(r'_P(\d)_', cfg)
            lreg_match = re.search(r'_lreg([\d.]+)_', cfg)
            if not P_match or not lreg_match:
                continue
            P = int(P_match.group(1))
            lreg = float(lreg_match.group(1))
            brier = r.get('brier_mean', None)
            if brier is None or brier > 1.0:
                continue
            pi = P_values.index(P)
            li = lreg_values.index(lreg) if lreg in lreg_values else -1
            if li >= 0:
                grid[li, pi] = brier

        im = ax.imshow(grid, cmap='YlOrRd_r', aspect='auto',
                       vmin=np.nanmin(grid) * 0.95 if not np.all(np.isnan(grid)) else 0,
                       vmax=np.nanmax(grid) * 1.05 if not np.all(np.isnan(grid)) else 1)
        # Annotate cells
        for li in range(len(lreg_values)):
            for pi in range(len(P_values)):
                if not np.isnan(grid[li, pi]):
                    ax.text(pi, li, f'{grid[li, pi]:.3f}', ha='center', va='center',
                            fontsize=7, color='black')
        ax.set_xticks(range(len(P_values)))
        ax.set_xticklabels([f'$P$={p}' for p in P_values])
        ax.set_yticks(range(len(lreg_values)))
        ax.set_yticklabels([f'$\\lambda$={v}' for v in lreg_values])
        ax.set_title(ds.capitalize(), fontweight='bold', fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, 'p_lreg_heatmap.pdf')
    fig.savefig(path)
    print(f'Saved {path}')
    plt.close(fig)


def plot_good_p_bars(results, output_dir):
    """P effect on GOOD: grouped bars per setting."""
    good = [r for r in results if r.get('type') == 'good']
    by_setting = defaultdict(dict)
    for r in good:
        cfg = r.get('config', '')
        if '_erm' not in cfg:
            continue
        P_match = re.search(r'_P(\d)_', cfg)
        if not P_match:
            continue
        P = int(P_match.group(1))
        ds = r['dataset']
        acc = r.get('acc', 0)
        by_setting[ds][P] = acc * 100

    settings = sorted(by_setting.keys())
    P_values = [0, 1, 2]
    colors = ['#bdbdbd', '#74a9cf', '#2b8cbe']

    fig, ax = plt.subplots(1, 1, figsize=(7, 2.5))
    x = np.arange(len(settings))
    width = 0.25

    for i, P in enumerate(P_values):
        vals = [by_setting[s].get(P, 0) for s in settings]
        bars = ax.bar(x + (i - 1) * width, vals, width, color=colors[i],
                      label=f'$P$={P}', edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(settings, rotation=25, ha='right', fontsize=7)
    ax.set_ylabel('Shifted Test Acc (%)')
    ax.legend(fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add TAR reference lines
    tar_values = {
        'Arxiv-degree': 59.26, 'Arxiv-time': 66.08, 'CBAS-color': 87.29,
        'Cora-degree': 61.73, 'Cora-word': 64.73, 'Twitch-lang': 57.20,
        'WebKB-univ': 30.83,
    }
    for i, s in enumerate(settings):
        if s in tar_values:
            ax.plot([i - 0.35, i + 0.35], [tar_values[s], tar_values[s]],
                    color='#e41a1c', linewidth=1.5, linestyle='--', zorder=5)
    ax.plot([], [], '--', color='#e41a1c', linewidth=1.5, label='TAR')
    ax.legend(fontsize=7, ncol=2)

    path = os.path.join(output_dir, 'good_p_bars.pdf')
    fig.savefig(path)
    print(f'Saved {path}')
    plt.close(fig)


def plot_brier_vs_auroc(output_dir):
    """ID Brier vs OOD AUROC scatter for Cora shift settings."""
    # Data from OOD Brier experiment (collected earlier)
    methods = {
        'GNNSafe':    {'brier': [0.394, 0.394, 0.198], 'auroc': [67.65, 50.92, 39.51]},
        'GNNSafe++':  {'brier': [0.365, 0.364, 0.183], 'auroc': [78.24, 50.67, 60.75]},
        'Graph-EBM':  {'brier': [0.394, 0.394, 0.198], 'auroc': [50.49, 42.14, 19.69]},
        'DSS-Hybrid': {'brier': [0.361, 0.360, 0.184], 'auroc': [92.96, 96.65, 84.80]},
    }
    shifts = ['structure', 'feature', 'label']
    colors = {'GNNSafe': '#999999', 'GNNSafe++': '#d95f02',
              'Graph-EBM': '#7570b3', 'DSS-Hybrid': '#2b8cbe'}
    markers = {'GNNSafe': 'o', 'GNNSafe++': 's', 'Graph-EBM': '^', 'DSS-Hybrid': 'D'}

    fig, ax = plt.subplots(1, 1, figsize=(4.0, 3.0))

    for method, data in methods.items():
        for i, shift in enumerate(shifts):
            label = method if i == 0 else None
            ax.scatter(data['brier'][i], data['auroc'][i],
                       c=colors[method], marker=markers[method], s=50,
                       label=label, zorder=3, edgecolors='white', linewidth=0.5)

    ax.set_xlabel('ID Brier ($\\downarrow$)')
    ax.set_ylabel('OOD AUROC (\\%)')
    ax.legend(fontsize=7, loc='lower left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # DSS-Hybrid should be top-left (low Brier, high AUROC)
    ax.annotate('better', xy=(0.18, 95), fontsize=7, color='#666666',
                arrowprops=dict(arrowstyle='->', color='#666666'),
                xytext=(0.30, 75))

    path = os.path.join(output_dir, 'brier_vs_auroc.pdf')
    fig.savefig(path)
    print(f'Saved {path}')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='tex/figures')
    parser.add_argument('--grid_v2', type=str, default='results_local/grid_v2_results.txt')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.exists(args.grid_v2):
        results = load_grid_results(args.grid_v2)
        print(f'Loaded {len(results)} grid v2 results')
        plot_p_sweep_lines(results, args.output_dir)
        plot_p_lreg_heatmap(results, args.output_dir)
        plot_good_p_bars(results, args.output_dir)
    else:
        print(f'Grid v2 results not found at {args.grid_v2}, skipping grid plots')

    plot_brier_vs_auroc(args.output_dir)
    print('Done.')


if __name__ == '__main__':
    main()
