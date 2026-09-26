#!/usr/bin/env python3
"""Generate OOD density plots combining baseline methods with best DSS-Hybrid.

Reads baseline scores from fair_local runs and best DSS scores from
density_best runs, producing a 1x4 (or 2x4) panel figure.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 9, "font.family": "serif",
    "axes.labelsize": 10, "axes.titlesize": 10,
    "legend.fontsize": 8, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

BASEDIR = "results_local/ood_density"


def load_and_combine(dataset):
    """Load scores, return combined DataFrame with standardized labels."""
    keep = {
        "gcn_gnnsafe": "GNNSafe",
        "gcn_gnnsafepp": "GNNSafe++",
        "gcn_graph_ebm": "Graph-EBM",
    }
    dss_configs = {
        "gcn_dssres_gnnsafepp": "DSS-Hybrid",
        "gcn_dssres_gnnsafe": "DSS-Hybrid",
        "gcn_dssres_fusion_probunc": "DSS-Hybrid",
    }

    if dataset == "cora":
        # All methods in one file (from combined run)
        df = pd.read_csv(os.path.join(BASEDIR, "cora_final_scores.csv"))
        parts = []
        for cfg, lbl in {**keep, **dss_configs}.items():
            sub = df[df["config"] == cfg].copy()
            if not sub.empty:
                sub["label"] = lbl
                parts.append(sub)
        return pd.concat(parts, ignore_index=True)
    elif dataset == "twitch":
        # Baselines from BN run, DSS from no-BN run
        baselines_file = os.path.join(BASEDIR, "twitch_baselines_final.csv")
        dss_file = os.path.join(BASEDIR, "twitch_dss_final.csv")
        parts = []
        if os.path.exists(baselines_file):
            bl = pd.read_csv(baselines_file)
            for cfg, lbl in keep.items():
                sub = bl[bl["config"] == cfg].copy()
                if not sub.empty:
                    sub["label"] = lbl
                    parts.append(sub)
        if os.path.exists(dss_file):
            dss = pd.read_csv(dss_file)
            dss["label"] = "DSS-Hybrid"
            parts.append(dss)
        elif os.path.exists(os.path.join(BASEDIR, "twitch_best_dss_scores.csv")):
            dss = pd.read_csv(os.path.join(BASEDIR, "twitch_best_dss_scores.csv"))
            dss["label"] = "DSS-Hybrid"
            parts.append(dss)
        return pd.concat(parts, ignore_index=True)
    else:
        # Fallback: fair + best DSS
        fair = pd.read_csv(os.path.join(BASEDIR, f"{dataset}_fair_scores.csv"))
        best_dss = pd.read_csv(os.path.join(BASEDIR, f"{dataset}_best_dss_scores.csv"))
        parts = []
        for cfg, lbl in keep.items():
            sub = fair[fair["config"] == cfg].copy()
            if not sub.empty:
                sub["label"] = lbl
                parts.append(sub)
        best_dss["label"] = "DSS-Hybrid"
        parts.append(best_dss)
        return pd.concat(parts, ignore_index=True)


def plot_density_row(df, dataset_label, axes):
    """Plot 4-panel density for one dataset using smooth KDE curves."""
    from scipy.stats import gaussian_kde
    methods = ["GNNSafe", "GNNSafe++", "Graph-EBM", "DSS-Hybrid"]
    colors_id = "#6baed6"
    colors_ood = "#fb6a4a"

    for idx, method in enumerate(methods):
        ax = axes[idx]
        sub = df[df["label"] == method]
        if sub.empty:
            ax.set_visible(False)
            continue

        # Negate scores: our code stores -E(x) (higher=ID),
        # but GNNSafe paper convention is E(x) (lower=ID, higher=OOD)
        ind = -sub[sub["split"] == "IND"]["score"].values
        ood = -sub[sub["split"] == "OOD"]["score"].values

        all_scores = np.concatenate([ind, ood])
        lo, hi = np.percentile(all_scores, [1, 99])
        margin = (hi - lo) * 0.05
        lo, hi = lo - margin, hi + margin
        xs = np.linspace(lo, hi, 300)

        # KDE
        bw = 0.15 * (hi - lo) / max(len(ind), 1) ** 0.2  # adaptive bandwidth
        try:
            kde_ind = gaussian_kde(ind, bw_method="silverman")
            kde_ood = gaussian_kde(ood, bw_method="silverman")
            y_ind = kde_ind(xs)
            y_ood = kde_ood(xs)
        except Exception:
            # Fallback to histogram
            y_ind, _ = np.histogram(ind, bins=50, range=(lo, hi), density=True)
            y_ood, _ = np.histogram(ood, bins=50, range=(lo, hi), density=True)
            xs = np.linspace(lo, hi, 50)

        ax.fill_between(xs, y_ind, alpha=0.4, color=colors_id, label="ID")
        ax.plot(xs, y_ind, color=colors_id, linewidth=1.2)
        ax.fill_between(xs, y_ood, alpha=0.4, color=colors_ood, label="OOD")
        ax.plot(xs, y_ood, color=colors_ood, linewidth=1.2)

        ax.set_title(method, fontweight="bold", fontsize=11)
        ax.set_xlabel("Energy $E(x)$", fontsize=10)
        ax.tick_params(axis='both', labelsize=9)
        if idx == 0:
            ax.set_ylabel("Density", fontsize=10)
            ax.legend(fontsize=9, framealpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(lo, hi)
        ax.set_ylim(bottom=0)


def main():
    outdir = "tex/figures"
    os.makedirs(outdir, exist_ok=True)

    # --- Cora / structure ---
    if os.path.exists(os.path.join(BASEDIR, "cora_final_scores.csv")):
        df_cora = load_and_combine("cora")
    elif os.path.exists(os.path.join(BASEDIR, "cora_fair_scores.csv")):
        df_cora = load_and_combine("cora")
    else:
        df_cora = None

    if df_cora is not None:
        fig, axes = plt.subplots(1, 4, figsize=(12, 2.5))
        plot_density_row(df_cora, "Cora / structure shift", axes)
        fig.suptitle("Cora / structure shift", fontsize=11, y=1.02)
        plt.tight_layout()
        path = os.path.join(outdir, "ood_density_cora_structure.pdf")
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved {path}")
        plt.close(fig)

    # --- Twitch ---
    if os.path.exists(os.path.join(BASEDIR, "twitch_fair_scores.csv")):
        df_twitch = load_and_combine("twitch")
        fig, axes = plt.subplots(1, 4, figsize=(12, 2.5), sharey=True)
        plot_density_row(df_twitch, "Twitch (cross-graph)", axes)
        fig.suptitle("Twitch (cross-graph OOD)", fontsize=11, y=1.02)
        plt.tight_layout()
        path = os.path.join(outdir, "ood_density_twitch.pdf")
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved {path}")
        plt.close(fig)

    # --- Combined 2-row figure ---
    if (os.path.exists(os.path.join(BASEDIR, "cora_fair_scores.csv")) and
        os.path.exists(os.path.join(BASEDIR, "twitch_fair_scores.csv"))):
        fig, axes = plt.subplots(2, 4, figsize=(12, 4.5))
        df_cora = load_and_combine("cora")
        df_twitch = load_and_combine("twitch")
        plot_density_row(df_cora, "Cora", axes[0])
        plot_density_row(df_twitch, "Twitch", axes[1])
        axes[0][0].set_ylabel("Cora\nDensity", fontsize=12, fontweight="bold")
        axes[1][0].set_ylabel("Twitch\nDensity", fontsize=12, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(outdir, "ood_density_combined.pdf")
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved {path}")
        plt.close(fig)

    print("Done.")


if __name__ == "__main__":
    main()
