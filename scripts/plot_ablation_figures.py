#!/usr/bin/env python3
"""
Generate ablation and visualization figures for the paper.
Run locally after collecting experiment results.

Usage:
  python scripts/plot_ablation_figures.py --output_dir tex/figures
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Use a clean style
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


# ============================================================
# Data from ablation experiments (from SLURM results)
# ============================================================

# Full ablation data from E-appendix table (12 datasets, P=0,1,2,3)
P_ABLATION = {
    "Cora":        {"acc": [85.50, 85.44, 85.60, 86.49], "brier": [0.220, 0.228, 0.223, 0.212]},
    "Citeseer":    {"acc": [65.39, 78.59, 79.26, 78.43], "brier": [0.427, 0.351, 0.318, 0.326]},
    "PubMed":      {"acc": [89.02, 89.29, 89.23, 89.23], "brier": [0.164, 0.160, 0.161, 0.161]},
    "Texas":       {"acc": [74.75, 88.52, 90.98, 89.02], "brier": [0.362, 0.190, 0.214, 0.207]},
    "Cornell":     {"acc": [82.46, 83.44, 81.48, 83.11], "brier": [0.293, 0.262, 0.287, 0.262]},
    "Wisconsin":   {"acc": [95.38, 93.75, 93.00, 92.38], "brier": [0.084, 0.089, 0.096, 0.121]},
    "Chameleon":   {"acc": [74.25, 75.12, 74.81, 74.90], "brier": [0.363, 0.338, 0.346, 0.348]},
    "Squirrel":    {"acc": [67.20, 67.61, 68.55, 67.56], "brier": [0.444, 0.393, 0.399, 0.441]},
    "CS":          {"acc": [95.94, 96.17, 96.00, 96.01], "brier": [0.065, 0.062, 0.063, 0.064]},
    "Amazon-Rat.": {"acc": [45.31, 48.24, 49.18, 49.24], "brier": [0.666, 0.641, 0.635, 0.636]},
    "Roman-Emp.":  {"acc": [77.11, 58.32, 49.91, None],  "brier": [0.329, 0.340, 0.340, None]},
    "Minesweeper": {"acc": [83.80, 83.88, 83.68, 83.72], "brier": [0.223, 0.179, 0.169, 0.225]},
    "Tolokers":    {"acc": [78.62, 78.18, None, None],    "brier": [0.293, 0.268, 0.273, 0.308]},
    "Questions":   {"acc": [97.15, 97.05, None, None],    "brier": [0.055, 0.054, None, None]},
}

REG_ABLATION = {
    "Cora": {
        "lreg":  [0, 0.001, 0.01, 0.1],
        "acc":   [84.50, 85.29, 85.57, 85.76],
        "brier": [0.232, 0.227, 0.224, 0.222],
    },
    "PubMed": {
        "lreg":  [0, 0.001, 0.01, 0.1],
        "acc":   [89.23, 89.34, 89.22, 89.29],
        "brier": [0.161, 0.160, 0.161, 0.162],
    },
    "Texas": {
        "lreg":  [0, 0.001, 0.01, 0.1],
        "acc":   [74.92, 87.87, 91.31, 88.20],
        "brier": [0.369, 0.241, 0.212, 0.253],
    },
    "Chameleon": {
        "lreg":  [0, 0.001, 0.01, 0.1],
        "acc":   [75.08, 74.75, 74.84, 74.53],
        "brier": [0.357, 0.359, 0.345, 0.353],
    },
}

GOOD_ADAPTIVE = {
    "GOOD-Twitch/language": {
        "lambda": [0.0, 0.2, 0.5, 0.8],
        "acc":    [63.24, 63.22, 63.99, 63.99],
        "tar":    57.20,
        "erm":    47.10,
    },
    "GOOD-Arxiv/time": {
        "lambda": [0.0, 0.2, 0.5, 0.8],
        "acc":    [64.12, 64.17, 64.19, 64.68],
        "tar":    66.08,
        "erm":    66.76,  # ERM is best here
    },
    "GOOD-CBAS/color": {
        "lambda": [0.0, 0.2, 0.5, 0.8],
        "acc":    [71.43, 72.14, 71.43, 70.71],
        "tar":    87.29,
        "erm":    70.71,
    },
}


def plot_p_ablation(output_dir):
    """Appendix figure: P ablation bars for representative datasets (those with full P=0..3 data)."""
    # Only include datasets with complete P=0..3 data
    datasets = [ds for ds in P_ABLATION
                if all(v is not None for v in P_ABLATION[ds]["brier"])]
    P_values = [0, 1, 2, 3]
    colors = ["#bdbdbd", "#74a9cf", "#2b8cbe", "#045a8d"]
    x = np.arange(len(P_values))
    width = 0.6
    n_ds = len(datasets)

    ncols = min(n_ds, 5)
    nrows = 2 * ((n_ds + ncols - 1) // ncols)  # 2 rows per batch (acc + brier)
    fig, axes = plt.subplots(2, ncols, figsize=(2.2 * ncols, 4.0),
                              gridspec_kw={"hspace": 0.35, "wspace": 0.35})
    if ncols == 1:
        axes = axes.reshape(2, 1)

    for col_idx in range(ncols):
        if col_idx >= n_ds:
            axes[0, col_idx].set_visible(False)
            axes[1, col_idx].set_visible(False)
            continue
        ds = datasets[col_idx]
        data = P_ABLATION[ds]

        # --- Top row: Accuracy ---
        ax = axes[0, col_idx]
        ax.bar(x, data["acc"], width, color=colors, edgecolor="white", linewidth=0.5)
        best_acc_idx = np.argmax(data["acc"])
        ax.bar(x[best_acc_idx], data["acc"][best_acc_idx], width,
               color=colors[best_acc_idx], edgecolor="black", linewidth=1.2)
        ax.axhline(data["acc"][0], color="#999999", linestyle="--", linewidth=0.8, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.set_title(ds, fontweight="bold", fontsize=9)
        if col_idx == 0:
            ax.set_ylabel("Accuracy (%)")
        ymin = min(data["acc"]) - 3
        ymax = max(data["acc"]) + 3
        ax.set_ylim(ymin, ymax)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # --- Bottom row: Brier ---
        ax = axes[1, col_idx]
        ax.bar(x, data["brier"], width, color=colors, edgecolor="white", linewidth=0.5)
        best_brier_idx = np.argmin(data["brier"])
        ax.bar(x[best_brier_idx], data["brier"][best_brier_idx], width,
               color=colors[best_brier_idx], edgecolor="black", linewidth=1.2)
        ax.axhline(data["brier"][0], color="#999999", linestyle="--", linewidth=0.8, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels([f"$P$={p}" for p in P_values])
        if col_idx == 0:
            ax.set_ylabel("Brier ($\\downarrow$)")
        ymin_b = min(data["brier"]) * 0.9
        ymax_b = max(data["brier"]) * 1.1
        ax.set_ylim(ymin_b, ymax_b)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "p_ablation.pdf")
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)


# Large-P sweep: best Brier across ALL configs (lambda_reg={0.0, 0.01}) at each P
# Uses Table 2 configs: Cora --hidden 256, Citeseer --hidden 64,
# Texas --propagate_first --optimizer_prop RMSprop,
# Amazon-Rat. --propagate_first --hidden 128, Chameleon --propagate_first --combine sum
LARGE_P_SWEEP = {
    "Citeseer": {
        "P": [0, 1, 2, 3, 4, 5, 6, 8],
        "brier": [0.4273, 0.3505, 0.3178, 0.3198, 0.3158, 0.3826, 0.4000, 0.3941],
    },
    "Texas": {
        "P": [0, 1, 2, 3, 4, 5, 6, 8],
        "brier": [0.2433, 0.1867, 0.2140, 0.2070, 0.2274, 0.2274, 0.2227, 0.2158],
    },
    "Cora": {
        "P": [0, 1, 2, 3, 4, 5, 6, 8],
        "brier": [0.2277, 0.2100, 0.2146, 0.2120, 0.2106, 0.2178, 0.2141, 0.2143],
    },
    "PubMed": {
        "P": [0, 1, 2, 3, 4, 5, 6, 8],
        "brier": [0.1623, 0.1580, 0.1601, 0.1588, 0.1608, 0.1556, 0.1587, 0.1594],
    },
    "Amazon-Rat.": {
        "P": [0, 1, 2, 3, 4, 5, 6, 8],
        "brier": [0.6655, 0.6417, 0.6353, 0.6328, 0.6362, 0.6367, 0.6358, 0.6344],
    },
    "Chameleon": {
        "P": [0, 1, 2, 3, 4, 5, 6, 8],
        "brier": [0.3621, 0.3385, 0.3466, 0.3504, 0.3515, 0.3508, 0.3545, 0.3599],
    },
    "Squirrel": {
        "P": [0, 1, 2, 3, 4, 5, 6, 8],
        "brier": [0.4457, 0.4235, 0.4353, 0.4412, 0.4275, 0.4304, 0.4364, 0.4228],
    },
}


def plot_p_ablation_summary(output_dir):
    """Two-panel figure: (a) % Brier reduction across datasets,
    (b) Brier vs P curves showing plateau at P=1-2.
    """
    # --- Compute percentage improvements ---
    datasets = []
    for ds in P_ABLATION:
        brier_vals = [b for b in P_ABLATION[ds]["brier"][1:] if b is not None]
        if brier_vals:
            datasets.append(ds)

    n = len(datasets)
    p0_brier = [P_ABLATION[ds]["brier"][0] for ds in datasets]
    best_p_brier = [min(b for b in P_ABLATION[ds]["brier"][1:] if b is not None)
                    for ds in datasets]
    best_p_idx = [1 + np.argmin([b if b is not None else 1e9
                                  for b in P_ABLATION[ds]["brier"][1:]])
                  for ds in datasets]
    pct = [100.0 * (p0 - bp) / p0 for p0, bp in zip(p0_brier, best_p_brier)]

    order = np.argsort(pct)[::-1]
    datasets_sorted = [datasets[i] for i in order]
    pct_sorted = [pct[i] for i in order]
    best_p_sorted = [best_p_idx[i] for i in order]

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0),
                              gridspec_kw={"width_ratios": [1.3, 1]})

    # (a) Percentage Brier reduction
    ax = axes[0]
    y = np.arange(n)
    colors = ["#2b8cbe" if p > 0 else "#e74c3c" for p in pct_sorted]
    # Clip display values for visual balance (cap at ±35%)
    pct_display = [max(min(p, 55), -35) for p in pct_sorted]
    ax.barh(y, pct_display, color=colors, edgecolor="white",
            height=0.6, zorder=3)
    ax.axvline(0, color="black", linewidth=0.8, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(datasets_sorted, fontsize=9, fontweight="bold")
    ax.set_xlabel(r"Brier score reduction (\%)", fontsize=11)
    ax.tick_params(axis='x', labelsize=10)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, (p, pd, pi) in enumerate(zip(pct_sorted, pct_display, best_p_sorted)):
        if p > 0:
            ax.text(pd + 0.8, i, f"$P$={pi}", va="center", fontsize=8,
                    color="#045a8d")
        else:
            # Red bar: label to the right of the dashed line (consistent)
            ax.text(1.0, i, f"$P$={pi}", va="center",
                    fontsize=8, color="#c0392b")
    n_improved = sum(1 for p in pct_sorted if p > 0)
    ax.set_title(f"(a) $P{{>}}0$ improves Brier on {n_improved}/{n} datasets",
                 fontsize=11, fontweight="bold")

    # (b) Brier vs P curves (large-P sweep)
    ax = axes[1]
    markers = ["o", "s", "^", "D", "v", "p", "h"]
    colors_line = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#a6761d", "#e6ab02"]
    for i, (ds, data) in enumerate(LARGE_P_SWEEP.items()):
        P_vals = data["P"]
        brier_vals = data["brier"]
        # Normalize to P=0 value for relative comparison
        b0 = brier_vals[0]
        relative = [b / b0 for b in brier_vals]
        ax.plot(P_vals, relative, marker=markers[i], color=colors_line[i],
                label=ds, markersize=5, linewidth=1.5, zorder=3)
    ax.axhline(1.0, color="#999999", linestyle="--", linewidth=0.8, zorder=1,
               label="$P{=}0$ baseline")
    ax.set_xlabel("Chaos order $P$", fontsize=10)
    ax.set_ylabel("Brier / Brier($P{=}0$)", fontsize=10)
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 8])
    ax.set_ylim(0.65, 1.08)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    ax.tick_params(axis='both', labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Shade the P=1-2 sweet spot
    ax.axvspan(0.5, 2.5, alpha=0.08, color="#2b8cbe", zorder=0)
    ax.text(1.5, 0.67, "$P{=}1$--$2$\nsweet spot", ha="center", fontsize=9,
            color="#045a8d", style="italic")
    ax.set_title("(b) Performance plateaus at low $P$", fontsize=11,
                 fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "p_ablation_summary.pdf")
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)


def plot_reg_ablation(output_dir):
    """Figure: λ_reg ablation — line plots."""
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.5))
    datasets = list(REG_ABLATION.keys())
    markers = ["o", "s", "^", "D"]
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]

    # Left: Accuracy
    ax = axes[0]
    for i, ds in enumerate(datasets):
        data = REG_ABLATION[ds]
        x = [1e-4 if v == 0 else v for v in data["lreg"]]  # shift 0 for log scale
        ax.plot(x, data["acc"], marker=markers[i], color=colors[i], label=ds,
                markersize=5, linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("$\\lambda_{\\mathrm{reg}}$")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1])
    ax.set_xticklabels(["0", "0.001", "0.01", "0.1"])
    ax.legend(loc="lower right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: Brier
    ax = axes[1]
    for i, ds in enumerate(datasets):
        data = REG_ABLATION[ds]
        x = [1e-4 if v == 0 else v for v in data["lreg"]]
        ax.plot(x, data["brier"], marker=markers[i], color=colors[i], label=ds,
                markersize=5, linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("$\\lambda_{\\mathrm{reg}}$")
    ax.set_ylabel("Brier Score")
    ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1])
    ax.set_xticklabels(["0", "0.001", "0.01", "0.1"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "reg_ablation.pdf")
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)


def plot_good_adaptive(output_dir):
    """Figure 5: Adaptive λ sweep for GOOD settings."""
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    settings = list(GOOD_ADAPTIVE.keys())
    color_dss = "#2b8cbe"
    color_tar = "#d95f02"
    color_erm = "#999999"

    for idx, setting in enumerate(settings):
        ax = axes[idx]
        data = GOOD_ADAPTIVE[setting]
        ax.plot(data["lambda"], data["acc"], "o-", color=color_dss, linewidth=1.8,
                markersize=5, label="DSS-Hybrid", zorder=3)
        ax.axhline(data["tar"], color=color_tar, linestyle="--", linewidth=1.2,
                   label="TAR", zorder=2)
        ax.axhline(data["erm"], color=color_erm, linestyle=":", linewidth=1.0,
                   label="DSS ERM", zorder=1)
        ax.set_xlabel("$\\lambda$ (spectral coupling)")
        if idx == 0:
            ax.set_ylabel("Shifted Test Acc (%)")
        ax.set_title(setting, fontsize=9, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == 0:
            ax.legend(fontsize=7, loc="lower right")

    plt.tight_layout()
    path = os.path.join(output_dir, "good_adaptive_lambda.pdf")
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="tex/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    plot_p_ablation(args.output_dir)
    plot_p_ablation_summary(args.output_dir)
    plot_reg_ablation(args.output_dir)
    plot_good_adaptive(args.output_dir)
    print("Done. All figures saved.")


if __name__ == "__main__":
    main()
