#!/usr/bin/env python3
"""
Local GOOD comparison with score-density visualization.

This mirrors the GNNSafe local analysis workflow, but for GOOD node-level
domain-generalization splits. The primary classifier metrics are reported on
GOOD's `id_test` and `test` splits, and we additionally visualize how scalar
confidence / uncertainty scores separate `id_test` from `test`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

MPL_CACHE_DIR = Path("results_local") / ".matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR.resolve()))
CACHE_ROOT = Path("results_local") / ".cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT.resolve()))

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from matplotlib import use as matplotlib_use
from scipy.stats import gaussian_kde

matplotlib_use("Agg")
import matplotlib.pyplot as plt

from gnnsafe_ood.data_utils import get_measures
from gnnsafe_ood.graph_ebm import GraphEBMPosthocScorer
from gnnsafe_ood.scores import propagate_scalar_score
from run_dssgnn_good import (
    build_model,
    ensure_good_dataset_registered,
    forward_model,
    forward_predictive_stats,
    load_good_config,
    restore_best_state,
    train_epoch,
    _good_data_to_dssgnn_format,
)
from dssgnn.chebyshev import build_rescaled_laplacian, chebyshev_filter
from tfe_utils import compute_brier, compute_ece, compute_mce, set_seed


PRESET_DATASETS = {
    "goodcbas": "GOOD_clean/configs/GOOD_configs/GOODCBAS/color/concept/ERM.yaml",
    "goodcbas_color_concept": "GOOD_clean/configs/GOOD_configs/GOODCBAS/color/concept/ERM.yaml",
    "goodcbas_color_covariate": "GOOD_clean/configs/GOOD_configs/GOODCBAS/color/covariate/ERM.yaml",
    "goodwebkb": "GOOD_clean/configs/GOOD_configs/GOODWebKB/university/concept/ERM.yaml",
    "goodwebkb_university_concept": "GOOD_clean/configs/GOOD_configs/GOODWebKB/university/concept/ERM.yaml",
    "goodwebkb_university_covariate": "GOOD_clean/configs/GOOD_configs/GOODWebKB/university/covariate/ERM.yaml",
    "goodcora": "GOOD_clean/configs/GOOD_configs/GOODCora/degree/concept/ERM.yaml",
    "goodcora_degree_concept": "GOOD_clean/configs/GOOD_configs/GOODCora/degree/concept/ERM.yaml",
    "goodcora_degree_covariate": "GOOD_clean/configs/GOOD_configs/GOODCora/degree/covariate/ERM.yaml",
    "goodcora_word_concept": "GOOD_clean/configs/GOOD_configs/GOODCora/word/concept/ERM.yaml",
    "goodcora_word_covariate": "GOOD_clean/configs/GOOD_configs/GOODCora/word/covariate/ERM.yaml",
    "goodarxiv_degree_concept": "GOOD_clean/configs/GOOD_configs/GOODArxiv/degree/concept/ERM.yaml",
    "goodarxiv_degree_covariate": "GOOD_clean/configs/GOOD_configs/GOODArxiv/degree/covariate/ERM.yaml",
    "goodarxiv_time_concept": "GOOD_clean/configs/GOOD_configs/GOODArxiv/time/concept/ERM.yaml",
    "goodarxiv_time_covariate": "GOOD_clean/configs/GOOD_configs/GOODArxiv/time/covariate/ERM.yaml",
    "goodtwitch_language_concept": "GOOD_clean/configs/GOOD_configs/GOODTwitch/language/concept/ERM.yaml",
    "goodtwitch_language_covariate": "GOOD_clean/configs/GOOD_configs/GOODTwitch/language/covariate/ERM.yaml",
}

MODEL_SPECS = {
    "gcn": {
        "label": "GCN ERM",
        "backbone": "gcn",
        "train_mode": "erm",
        "score_types": ("msp", "energy", "entropy"),
    },
    "sage": {
        "label": "GraphSAGE ERM",
        "backbone": "sage",
        "train_mode": "erm",
        "score_types": ("msp", "energy", "entropy"),
    },
    "appnp": {
        "label": "APPNP ERM",
        "backbone": "appnp",
        "train_mode": "erm",
        "score_types": ("msp", "energy", "entropy"),
    },
    "gpr": {
        "label": "GPR-GNN ERM",
        "backbone": "gpr",
        "train_mode": "erm",
        "score_types": ("msp", "energy", "entropy"),
    },
    "gcn_gnnsafe": {
        "label": "GCN GNNSafe",
        "backbone": "gcn",
        "train_mode": "erm",
        "score_types": ("energy_prop",),
    },
    "gcn_gnnsafepp": {
        "label": "GCN GNNSafe++ (GOOD-adapted)",
        "backbone": "gcn",
        "train_mode": "gnnsafepp",
        "score_types": ("energy_prop",),
    },
    "gcn_graph_ebm": {
        "label": "GCN Graph-EBM",
        "backbone": "gcn",
        "train_mode": "erm",
        "score_types": ("graph_ebm",),
    },
    "dssgnn": {
        "label": "DSS-GNN",
        "backbone": "dssgnn",
        "train_mode": "erm",
        "score_types": ("msp", "energy", "entropy", "chaos", "pred_entropy", "mutual_info"),
    },
    "gcn_dssres": {
        "label": "GCN+DSS Residual",
        "backbone": "gcn_dssres",
        "train_mode": "erm",
        "score_types": ("msp", "energy", "entropy", "chaos", "pred_entropy", "mutual_info"),
    },
    "sage_dssres": {
        "label": "GraphSAGE+DSS Residual",
        "backbone": "sage_dssres",
        "train_mode": "erm",
        "score_types": ("msp", "energy", "entropy", "chaos", "pred_entropy", "mutual_info"),
    },
    "appnp_dssres": {
        "label": "APPNP+DSS Residual",
        "backbone": "appnp_dssres",
        "train_mode": "erm",
        "score_types": ("msp", "energy", "entropy", "chaos", "pred_entropy", "mutual_info"),
    },
    "gpr_dssres": {
        "label": "GPR-GNN+DSS Residual",
        "backbone": "gpr_dssres",
        "train_mode": "erm",
        "score_types": ("msp", "energy", "entropy", "chaos", "pred_entropy", "mutual_info"),
    },
    "gcn_dssres_tar": {
        "label": "GCN+DSS Residual TAR",
        "backbone": "gcn_dssres",
        "train_mode": "tar_reweight",
        "score_types": ("msp", "energy", "entropy", "chaos", "pred_entropy", "mutual_info"),
    },
    "sage_dssres_tar": {
        "label": "GraphSAGE+DSS Residual TAR",
        "backbone": "sage_dssres",
        "train_mode": "tar_reweight",
        "score_types": ("msp", "energy", "entropy", "chaos", "pred_entropy", "mutual_info"),
    },
    "gpr_dssres_tar": {
        "label": "GPR-GNN+DSS Residual TAR",
        "backbone": "gpr_dssres",
        "train_mode": "tar_reweight",
        "score_types": ("msp", "energy", "entropy", "chaos", "pred_entropy", "mutual_info"),
    },
    "gcn_dssres_tarspec": {
        "label": "GCN+DSS Residual Spectral TAR",
        "backbone": "gcn_dssres",
        "train_mode": "tar_spectral_reweight",
        "score_types": ("msp", "energy", "entropy", "chaos", "pred_entropy", "mutual_info"),
    },
    "sage_dssres_tarspec": {
        "label": "GraphSAGE+DSS Residual Spectral TAR",
        "backbone": "sage_dssres",
        "train_mode": "tar_spectral_reweight",
        "score_types": ("msp", "energy", "entropy", "chaos", "pred_entropy", "mutual_info"),
    },
    "gpr_dssres_tarspec": {
        "label": "GPR-GNN+DSS Residual Spectral TAR",
        "backbone": "gpr_dssres",
        "train_mode": "tar_spectral_reweight",
        "score_types": ("msp", "energy", "entropy", "chaos", "pred_entropy", "mutual_info"),
    },
    "gcn_dssres_taredge": {
        "label": "GCN+DSS Residual Edge-Spectral TAR",
        "backbone": "gcn_dssres",
        "train_mode": "tar_spectral_reweight",
        "score_types": ("msp", "energy", "entropy", "chaos", "pred_entropy", "mutual_info"),
    },
    "sage_dssres_taredge": {
        "label": "GraphSAGE+DSS Residual Edge-Spectral TAR",
        "backbone": "sage_dssres",
        "train_mode": "tar_spectral_reweight",
        "score_types": ("msp", "energy", "entropy", "chaos", "pred_entropy", "mutual_info"),
    },
    "gpr_dssres_taredge": {
        "label": "GPR-GNN+DSS Residual Edge-Spectral TAR",
        "backbone": "gpr_dssres",
        "train_mode": "tar_spectral_reweight",
        "score_types": ("msp", "energy", "entropy", "chaos", "pred_entropy", "mutual_info"),
    },
    "gpr_dssres_tarmix": {
        "label": "GPR-GNN+DSS Residual Mixed Edge-Spectral TAR",
        "backbone": "gpr_dssres",
        "train_mode": "tar_spectral_reweight",
        "score_types": ("msp", "energy", "entropy", "chaos", "pred_entropy", "mutual_info"),
    },
    "gcn_dssres_targate": {
        "label": "GCN+DSS Residual Local-Gated Edge TAR",
        "backbone": "gcn_dssres",
        "train_mode": "tar_spectral_reweight",
        "score_types": ("msp", "energy", "entropy", "chaos", "pred_entropy", "mutual_info"),
    },
    "sage_dssres_targate": {
        "label": "GraphSAGE+DSS Residual Local-Gated Edge TAR",
        "backbone": "sage_dssres",
        "train_mode": "tar_spectral_reweight",
        "score_types": ("msp", "energy", "entropy", "chaos", "pred_entropy", "mutual_info"),
    },
    "gpr_dssres_targate": {
        "label": "GPR-GNN+DSS Residual Local-Gated Edge TAR",
        "backbone": "gpr_dssres",
        "train_mode": "tar_spectral_reweight",
        "score_types": ("msp", "energy", "entropy", "chaos", "pred_entropy", "mutual_info"),
    },
    "gcn_dssres_tarloss": {
        "label": "GCN+DSS Residual Loss-Field TAR",
        "backbone": "gcn_dssres",
        "train_mode": "tar_lossfield_reweight",
        "score_types": ("msp", "energy", "entropy", "chaos", "pred_entropy", "mutual_info"),
    },
    "sage_dssres_tarloss": {
        "label": "GraphSAGE+DSS Residual Loss-Field TAR",
        "backbone": "sage_dssres",
        "train_mode": "tar_lossfield_reweight",
        "score_types": ("msp", "energy", "entropy", "chaos", "pred_entropy", "mutual_info"),
    },
    "gpr_dssres_tarloss": {
        "label": "GPR-GNN+DSS Residual Loss-Field TAR",
        "backbone": "gpr_dssres",
        "train_mode": "tar_lossfield_reweight",
        "score_types": ("msp", "energy", "entropy", "chaos", "pred_entropy", "mutual_info"),
    },
    "gcn_dssres_taradapt": {
        "label": "GCN+DSS Residual Adaptive TAR",
        "backbone": "gcn_dssres",
        "train_mode": "tar_adaptive_reweight",
        "score_types": ("msp", "energy", "entropy", "chaos", "pred_entropy", "mutual_info"),
    },
    "sage_dssres_taradapt": {
        "label": "GraphSAGE+DSS Residual Adaptive TAR",
        "backbone": "sage_dssres",
        "train_mode": "tar_adaptive_reweight",
        "score_types": ("msp", "energy", "entropy", "chaos", "pred_entropy", "mutual_info"),
    },
    "gpr_dssres_taradapt": {
        "label": "GPR-GNN+DSS Residual Adaptive TAR",
        "backbone": "gpr_dssres",
        "train_mode": "tar_adaptive_reweight",
        "score_types": ("msp", "energy", "entropy", "chaos", "pred_entropy", "mutual_info"),
    },
    "gcn_dssres_gnnsafepp": {
        "label": "GCN+DSS Residual GNNSafe++ (GOOD-adapted)",
        "backbone": "gcn_dssres",
        "train_mode": "gnnsafepp",
        "score_types": ("energy_prop", "pred_entropy_prop", "mutual_info_prop"),
    },
    "sage_dssres_gnnsafepp": {
        "label": "GraphSAGE+DSS Residual GNNSafe++ (GOOD-adapted)",
        "backbone": "sage_dssres",
        "train_mode": "gnnsafepp",
        "score_types": ("energy_prop", "pred_entropy_prop", "mutual_info_prop"),
    },
    "sage_dssres_gnnsafepp_support": {
        "label": "GraphSAGE+DSS Residual GNNSafe++ + Support Prior",
        "backbone": "sage_dssres",
        "train_mode": "gnnsafepp",
        "posthoc_mode": "support_add",
        "score_types": ("energy_prop", "pred_entropy_prop", "mutual_info_prop"),
    },
    "sage_dssres_gnnsafepp_supporttrain": {
        "label": "GraphSAGE+DSS Residual GNNSafe++ + Support-Weighted Train Loss",
        "backbone": "sage_dssres",
        "train_mode": "gnnsafepp",
        "aux_mode": "support_weighted_train",
        "score_types": ("energy_prop", "pred_entropy_prop", "mutual_info_prop"),
    },
    "sage_dssres_gnnsafepp_protolp": {
        "label": "GraphSAGE+DSS Residual GNNSafe++ + Prototype/LP Aux",
        "backbone": "sage_dssres",
        "train_mode": "gnnsafepp",
        "aux_mode": "prototype_lp_aux",
        "score_types": ("energy_prop", "pred_entropy_prop", "mutual_info_prop"),
    },
    "sage_dssres_gnnsafepp_retrievalaux": {
        "label": "GraphSAGE+DSS Residual GNNSafe++ + Retrieval Teacher Aux",
        "backbone": "sage_dssres",
        "train_mode": "gnnsafepp",
        "aux_mode": "retrieval_teacher_aux",
        "score_types": ("energy_prop", "pred_entropy_prop", "mutual_info_prop"),
    },
    "sage_dssres_gnnsafepp_retrievalmix": {
        "label": "GraphSAGE+DSS Residual GNNSafe++ + Retrieval Mix",
        "backbone": "sage_dssres",
        "train_mode": "gnnsafepp",
        "posthoc_mode": "retrieval_support_add",
        "score_types": ("energy_prop", "pred_entropy_prop", "mutual_info_prop"),
    },
    "sage_dssres_gnnsafepp_retrievalswitch": {
        "label": "GraphSAGE+DSS Residual GNNSafe++ + Retrieval Switch",
        "backbone": "sage_dssres",
        "train_mode": "gnnsafepp",
        "posthoc_mode": "retrieval_switch",
        "score_types": ("energy_prop", "pred_entropy_prop", "mutual_info_prop"),
    },
    "appnp_dssres_gnnsafepp": {
        "label": "APPNP+DSS Residual GNNSafe++ (GOOD-adapted)",
        "backbone": "appnp_dssres",
        "train_mode": "gnnsafepp",
        "score_types": ("energy_prop", "pred_entropy_prop", "mutual_info_prop"),
    },
    "gcn_dssres_tarspec_uncmix": {
        "label": "GCN+DSS Residual Spectral TAR + Uncertainty Mix",
        "backbone": "gcn_dssres",
        "train_mode": "tar_spectral_reweight",
        "posthoc_mode": "uncertainty_mix",
        "score_types": ("msp", "energy", "entropy", "chaos", "pred_entropy", "mutual_info"),
    },
    "gcn_dssres_tarspec_uncmargin": {
        "label": "GCN+DSS Residual Spectral TAR + Uncertainty Margin",
        "backbone": "gcn_dssres",
        "train_mode": "tar_spectral_reweight",
        "aux_mode": "uncertainty_margin",
        "score_types": ("msp", "energy", "entropy", "chaos", "pred_entropy", "mutual_info"),
    },
    "gcn_dssres_tarspec_pairmargin": {
        "label": "GCN+DSS Residual Spectral TAR + Pair Margin",
        "backbone": "gcn_dssres",
        "train_mode": "tar_spectral_reweight",
        "aux_mode": "pair_confusion_margin",
        "score_types": ("msp", "energy", "entropy", "chaos", "pred_entropy", "mutual_info"),
    },
    "gcn_dssres_tarspec_pairproto": {
        "label": "GCN+DSS Residual Spectral TAR + Pair Prototype Separation",
        "backbone": "gcn_dssres",
        "train_mode": "tar_spectral_reweight",
        "aux_mode": "pair_prototype_separation",
        "score_types": ("msp", "energy", "entropy", "chaos", "pred_entropy", "mutual_info"),
    },
    "gcn_dssres_tarspec_pairspecialist": {
        "label": "GCN+DSS Residual Spectral TAR + Pair Specialist",
        "backbone": "gcn_dssres",
        "train_mode": "tar_spectral_reweight",
        "posthoc_mode": "pair_proto_specialist",
        "score_types": ("msp", "energy", "entropy", "chaos", "pred_entropy", "mutual_info"),
    },
    "gcn_dssres_tarspec_pairswitch": {
        "label": "GCN+DSS Residual Spectral TAR + Pair Switch",
        "backbone": "gcn_dssres",
        "train_mode": "tar_spectral_reweight",
        "posthoc_mode": "pair_proto_switch",
        "score_types": ("msp", "energy", "entropy", "chaos", "pred_entropy", "mutual_info"),
    },
}

NODE_PREDICTION_SCORE_FIELDS = tuple(
    sorted({score_type for spec in MODEL_SPECS.values() for score_type in spec["score_types"]})
)


def build_base_args():
    return SimpleNamespace(
        layers=3,
        K_lp=3,
        K_hp=2,
        P=1,
        P_gate=1,
        quadrature_nodes=4,
        pro_dropout=0.5,
        lin_dropout=0.0,
        lambda_max=2.0,
        lambda_reg=0.01,
        appnp_K=10,
        appnp_alpha=0.1,
        gpr_K=10,
        gpr_alpha=0.1,
        tar_inner_steps=3,
        tar_beta=0.1,
        tar_eta=0.0,
        tar_step_size=0.5,
        tar_eta0=0.0,
        tar_eta_lp=0.0,
        tar_eta_hp=0.0,
        tar_alpha_lp=0.0,
        tar_alpha_partial=0.0,
        tar_lambda_edge=1.0,
        tar_edge_onset_epoch=0,
        tar_gate_bias=0.0,
        tar_gate_lp=0.0,
        tar_gate_boundary=0.0,
        support_graph_gamma=0.0,
        support_proto_gamma=0.0,
        support_twohop_weight=0.5,
        support_smoothing=1.0,
        support_gain=1.0,
        support_proto_temp=0.5,
        uncertainty_mix_gamma=0.0,
        uncertainty_mix_cap=1.0,
        support_train_gamma=0.0,
        unc_margin_gamma=0.0,
        unc_margin_target=0.0,
        proto_aux_gamma=0.0,
        proto_lp_gamma=0.0,
        proto_aux_temp=0.5,
        retrieval_aux_gamma=0.0,
        retrieval_topk=8,
        retrieval_temp=0.2,
        retrieval_gate_power=1.0,
        retrieval_posthoc_gamma=0.0,
        retrieval_posthoc_topk=8,
        retrieval_posthoc_temp=0.2,
        retrieval_posthoc_power=1.0,
        retrieval_switch_gamma=1.0,
        retrieval_switch_support_thresh=0.6,
        retrieval_switch_conf_thresh=0.6,
        pair_margin_gamma=0.0,
        pair_margin_target=0.0,
        pair_proto_gamma=0.0,
        pair_proto_target=0.0,
        pair_specialist_gamma=0.0,
        pair_specialist_temp=0.2,
        pair_specialist_unc_scale=0.0,
        pair_switch_gamma=1.0,
        pair_switch_pairmass_thresh=0.6,
        pair_switch_margin_thresh=0.25,
        pair_switch_unc_thresh=0.0,
        use_random_gates=False,
        shared_input_lift=False,
    )


def summarize(values):
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std())


def plot_density(ax, ind_scores, ood_scores, title, shared_x_limits=None):
    ind_scores = np.asarray(ind_scores, dtype=float)
    ood_scores = np.asarray(ood_scores, dtype=float)
    if ind_scores.size == 0 or ood_scores.size == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No scores", ha="center", va="center")
        ax.axis("off")
        return

    combined = np.concatenate([ind_scores, ood_scores])
    x_min = combined.min()
    x_max = combined.max()
    if shared_x_limits is not None:
        x_min, x_max = shared_x_limits
    elif math.isclose(x_min, x_max):
        x_min -= 0.5
        x_max += 0.5

    xs = np.linspace(x_min, x_max, 256)
    for values, color, label in (
        (ind_scores, "#0f766e", "ID test"),
        (ood_scores, "#b91c1c", "OOD test"),
    ):
        std = float(np.std(values))
        if values.size >= 3 and std > 1e-12:
            kde = gaussian_kde(values)
            ax.plot(xs, kde(xs), color=color, lw=2, label=label)
        else:
            ax.hist(values, bins=20, density=True, alpha=0.35, color=color, label=label)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.2)
    ax.set_xlim(x_min, x_max)


def save_density_grid(path: Path, config_names, summary_rows, all_scores, fig_title, shared_x_limits=None):
    n_configs = len(config_names)
    n_cols = 3
    n_rows = math.ceil(n_configs / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.8 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, config_name in zip(axes_flat, config_names):
        summary = next(row for row in summary_rows if row["config"] == config_name)
        ind_scores = np.concatenate(all_scores[config_name]["ind"], axis=0)
        ood_scores = np.concatenate(all_scores[config_name]["ood"], axis=0)
        title = (
            f"{summary['label']}\n"
            f"Score AUROC {100 * summary['score_auroc_mean']:.1f} +/- {100 * summary['score_auroc_std']:.1f} | "
            f"ID Acc {100 * summary['id_test_acc_mean']:.1f} | OOD Acc {100 * summary['test_acc_mean']:.1f}"
        )
        plot_density(ax, ind_scores, ood_scores, title, shared_x_limits=shared_x_limits)

    for ax in axes_flat[n_configs:]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False)
    fig.suptitle(fig_title, fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_summary_csv(path: Path, rows):
    fieldnames = [
        "config",
        "label",
        "model",
        "score_type",
        "runs",
        "score_auroc_mean",
        "score_auroc_std",
        "score_aupr_mean",
        "score_aupr_std",
        "score_fpr95_mean",
        "score_fpr95_std",
        "id_test_acc_mean",
        "id_test_acc_std",
        "test_acc_mean",
        "test_acc_std",
        "id_test_ece_mean",
        "id_test_ece_std",
        "test_ece_mean",
        "test_ece_std",
        "val_loss_mean",
        "val_loss_std",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_split_metrics_csv(path: Path, rows):
    fieldnames = [
        "model",
        "label",
        "split",
        "acc_mean",
        "acc_std",
        "ece_mean",
        "ece_std",
        "mce_mean",
        "mce_std",
        "brier_mean",
        "brier_std",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_scores_csv(path: Path, rows):
    fieldnames = ["config", "label", "model", "score_type", "split", "score"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_node_predictions_csv(path: Path, rows):
    fieldnames = [
        "config",
        "label",
        "model",
        "run_idx",
        "seed",
        "split",
        "node_idx",
        "true_label",
        "pred_label",
        "correct",
        "confidence",
        "true_prob",
        "margin",
        "env_id",
        "domain_id",
    ] + list(NODE_PREDICTION_SCORE_FIELDS)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _maybe_dataset_tensor_numpy(dataset, attr_name):
    data = getattr(dataset, "data", None)
    if data is None or not hasattr(data, attr_name):
        return None
    value = getattr(data, attr_name)
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return None


def accuracy_from_logits(logits, labels):
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item())


def energy_from_logits(logits):
    return torch.logsumexp(logits, dim=1)


def evaluate_split(logits, labels):
    log_probs = F.log_softmax(logits, dim=1)
    return {
        "acc": accuracy_from_logits(log_probs, labels),
        "ece": float(compute_ece(log_probs, labels).item()),
        "mce": float(compute_mce(log_probs, labels).item()),
        "brier": float(compute_brier(log_probs, labels).item()),
    }


def compute_score_map(logits, uncertainty, predictive_stats=None, extra_scores=None):
    probs = F.softmax(logits, dim=1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
    scores = {
        "msp": probs.max(dim=1).values.detach().cpu().numpy(),
        "energy": torch.logsumexp(logits, dim=1).detach().cpu().numpy(),
        "entropy": (-entropy).detach().cpu().numpy(),
    }
    if uncertainty is not None:
        scores["chaos"] = (-uncertainty).detach().cpu().numpy()
    if predictive_stats is not None:
        scores["pred_entropy"] = (-predictive_stats["predictive_entropy"]).detach().cpu().numpy()
        scores["mutual_info"] = (-predictive_stats["mutual_info"]).detach().cpu().numpy()
    if extra_scores:
        for name, value in extra_scores.items():
            scores[name] = value.detach().cpu().numpy()
    return scores


def build_graph_ebm_scorer(args):
    return GraphEBMPosthocScorer(
        covariance_type=args.graph_ebm_covariance_type,
        tied_covariance=args.graph_ebm_tied_covariance,
        gamma_correction=args.graph_ebm_gamma_correction,
        lambda_independent_energy=args.graph_ebm_lambda_independent_energy,
        lambda_local_energy=args.graph_ebm_lambda_local_energy,
        lambda_group_energy=args.graph_ebm_lambda_group_energy,
        alpha=args.graph_ebm_alpha,
        num_diffusion_steps=args.graph_ebm_num_diffusion_steps,
        aggregation=args.graph_ebm_aggregation,
    )


def _gnnsafe_margin_loss(signal_in, signal_out, args):
    min_n = min(signal_in.shape[0], signal_out.shape[0])
    if min_n <= 0:
        return signal_in.new_tensor(0.0)
    signal_in = signal_in[:min_n]
    signal_out = signal_out[:min_n]
    return torch.mean(F.relu(signal_in - args.m_in) ** 2 + F.relu(args.m_out - signal_out) ** 2)


def build_transport_graph(edge_index, train_mask):
    train_nodes = train_mask.nonzero(as_tuple=False).view(-1)
    num_train = int(train_nodes.numel())
    if num_train <= 1:
        return {
            "num_train": num_train,
            "pairs": edge_index.new_empty((0, 2)),
            "weights": edge_index.new_empty((0,), dtype=torch.float),
        }

    local_index = edge_index.new_full((train_mask.numel(),), -1)
    local_index[train_nodes] = torch.arange(num_train, device=edge_index.device, dtype=edge_index.dtype)

    src, dst = edge_index
    keep = train_mask[src] & train_mask[dst] & (src != dst)
    if not bool(keep.any()):
        return {
            "num_train": num_train,
            "pairs": edge_index.new_empty((0, 2)),
            "weights": edge_index.new_empty((0,), dtype=torch.float),
        }

    src_local = local_index[src[keep]]
    dst_local = local_index[dst[keep]]
    pair_lo = torch.minimum(src_local, dst_local)
    pair_hi = torch.maximum(src_local, dst_local)
    pairs = torch.stack([pair_lo, pair_hi], dim=1)
    pairs = torch.unique(pairs, dim=0)

    deg = torch.bincount(pairs.reshape(-1), minlength=num_train).float()
    pair_deg = torch.maximum(deg[pairs[:, 0]], deg[pairs[:, 1]]).clamp_min(1.0)
    weights = pair_deg.reciprocal()
    return {"num_train": num_train, "pairs": pairs, "weights": weights}


def compute_tar_reweight(node_risk, transport_graph, args, edge_weights=None):
    num_train = transport_graph["num_train"]
    if num_train <= 1:
        return node_risk.new_ones((node_risk.shape[0],))

    pairs = transport_graph["pairs"]
    if edge_weights is None:
        weights = transport_graph["weights"].to(node_risk.device, dtype=node_risk.dtype)
    else:
        weights = edge_weights.to(node_risk.device, dtype=node_risk.dtype)
    if pairs.numel() == 0:
        return node_risk.new_ones((node_risk.shape[0],))
    pairs = pairs.to(node_risk.device)

    q = node_risk.new_full((num_train,), 1.0 / num_train)
    eps = 1e-8

    for _ in range(max(int(args.tar_inner_steps), 0)):
        qi = q[pairs[:, 0]].clamp_min(eps)
        qj = q[pairs[:, 1]].clamp_min(eps)
        ri = node_risk[pairs[:, 0]]
        rj = node_risk[pairs[:, 1]]
        velocity = ri - rj + args.tar_beta * (torch.log(qj) - torch.log(qi))
        xi = torch.where(velocity > 0, qj, qi)
        raw_flux = args.tar_step_size * weights * velocity * xi
        capacity = 0.5 * torch.minimum(qi, qj)
        flux = torch.clamp(raw_flux, min=-capacity, max=capacity)

        delta = q.new_zeros(q.shape)
        delta.index_add_(0, pairs[:, 0], flux)
        delta.index_add_(0, pairs[:, 1], -flux)
        q = (q + delta).clamp_min(eps)
        q = q / q.sum()

    return (num_train * q).detach()


def _default_tar_filter_coeffs(order, mode, device, dtype):
    coeffs = torch.zeros(order + 1, device=device, dtype=dtype)
    if mode == "lp":
        for k in range(order + 1):
            coeffs[k] = 0.5 ** k
        return coeffs
    if mode == "hp":
        lp_coeffs = _default_tar_filter_coeffs(order, "lp", device, dtype)
        coeffs[: lp_coeffs.numel()] = -lp_coeffs
        if coeffs.numel() > 0:
            coeffs[0] = coeffs[0] + 1.0
        return coeffs
    raise ValueError(f"Unknown TAR filter mode: {mode}")


def _extract_tar_filter_coeffs(model, args, device, dtype):
    dss_module = None
    if hasattr(model, "residual_dss") and hasattr(model.residual_dss, "dssgnn"):
        dss_module = model.residual_dss.dssgnn
    elif hasattr(model, "dssgnn"):
        dss_module = model.dssgnn

    if dss_module is not None and hasattr(dss_module, "convs") and len(dss_module.convs) > 0:
        conv = dss_module.convs[-1]
        if hasattr(conv, "coeffs_lp") and hasattr(conv, "coeffs_hp"):
            coeffs_lp = conv.coeffs_lp.detach().to(device=device, dtype=dtype)
            coeffs_hp = conv.coeffs_hp.detach().to(device=device, dtype=dtype)
            return coeffs_lp, coeffs_hp

    coeffs_lp = _default_tar_filter_coeffs(int(args.K_lp), "lp", device=device, dtype=dtype)
    coeffs_hp = _default_tar_filter_coeffs(int(args.K_hp), "hp", device=device, dtype=dtype)
    return coeffs_lp, coeffs_hp


def _normalize_positive_signal(values, eps=1e-8):
    return values / values.mean().clamp_min(eps)


def _normalize_signed_signal(values, eps=1e-8):
    return values / values.abs().mean().clamp_min(eps)


def compute_tar_edge_weights(transport_graph, args, smooth_signal=None, boundary_signal=None, current_epoch=None):
    has_anisotropy = args.tar_alpha_lp > 0 or args.tar_alpha_partial > 0
    gate_bias = float(getattr(args, "tar_gate_bias", 0.0))
    gate_lp = float(getattr(args, "tar_gate_lp", 0.0))
    gate_boundary = float(getattr(args, "tar_gate_boundary", 0.0))
    has_local_gate = abs(gate_bias) > 0 or abs(gate_lp) > 0 or abs(gate_boundary) > 0
    if not has_anisotropy:
        return None
    if float(getattr(args, "tar_lambda_edge", 1.0)) <= 0 and not has_local_gate:
        return None
    if current_epoch is not None and int(current_epoch) < int(getattr(args, "tar_edge_onset_epoch", 0)):
        return None
    if transport_graph["num_train"] <= 1:
        return None
    pairs = transport_graph["pairs"]
    if pairs.numel() == 0:
        return None

    ref = smooth_signal if smooth_signal is not None else boundary_signal
    if ref is None:
        return None

    device = ref.device
    dtype = ref.dtype
    pairs = pairs.to(device)
    base_weights = transport_graph["weights"].to(device=device, dtype=dtype)

    smooth_term = base_weights.new_zeros(base_weights.shape)
    boundary_term = base_weights.new_zeros(base_weights.shape)
    if smooth_signal is not None:
        smooth_term = 0.5 * (smooth_signal[pairs[:, 0]] + smooth_signal[pairs[:, 1]])
    if boundary_signal is not None:
        boundary_term = (boundary_signal[pairs[:, 0]] - boundary_signal[pairs[:, 1]]).abs()

    log_scale = args.tar_alpha_lp * smooth_term - args.tar_alpha_partial * boundary_term
    anisotropic_weights = base_weights * torch.exp(log_scale.clamp(min=-8.0, max=8.0))
    lambda_edge = float(getattr(args, "tar_lambda_edge", 1.0))
    lambda_edge = max(0.0, min(1.0, lambda_edge))
    if has_local_gate:
        local_lambda = lambda_edge + gate_bias + gate_lp * smooth_term - gate_boundary * boundary_term
        local_lambda = local_lambda.clamp(min=0.0, max=1.0)
        return (1.0 - local_lambda) * base_weights + local_lambda * anisotropic_weights
    if lambda_edge >= 1.0:
        return anisotropic_weights
    return (1.0 - lambda_edge) * base_weights + lambda_edge * anisotropic_weights


def compute_tar_spectral_components(per_node_loss, full_uncertainty, graph_input, train_mask, model, args):
    node_risk = per_node_loss.detach()
    if full_uncertainty is None:
        return node_risk, None, None, None

    full_uncertainty = full_uncertainty.detach().to(dtype=per_node_loss.dtype)
    coeffs_lp, coeffs_hp = _extract_tar_filter_coeffs(model, args, device=full_uncertainty.device, dtype=full_uncertainty.dtype)

    uncertainty_column = full_uncertainty.unsqueeze(-1)
    spectral_lp = chebyshev_filter(graph_input, uncertainty_column, coeffs_lp).squeeze(-1)
    spectral_hp = chebyshev_filter(graph_input, uncertainty_column, coeffs_hp).squeeze(-1).abs()

    base_signal = _normalize_positive_signal(full_uncertainty[train_mask])
    lp_signal = _normalize_signed_signal(spectral_lp[train_mask])
    hp_signal = _normalize_positive_signal(spectral_hp[train_mask])

    if args.tar_eta0 > 0:
        node_risk = node_risk + args.tar_eta0 * base_signal
    if args.tar_eta_lp > 0:
        node_risk = node_risk + args.tar_eta_lp * lp_signal
    if args.tar_eta_hp > 0:
        node_risk = node_risk + args.tar_eta_hp * hp_signal

    return node_risk, base_signal, lp_signal, hp_signal


def compute_tar_spectral_node_risk(per_node_loss, full_uncertainty, graph_input, train_mask, model, args):
    node_risk, _base_signal, _lp_signal, _hp_signal = compute_tar_spectral_components(
        per_node_loss=per_node_loss,
        full_uncertainty=full_uncertainty,
        graph_input=graph_input,
        train_mask=train_mask,
        model=model,
        args=args,
    )
    return node_risk


def compute_stochastic_train_loss_field(sample_logits, sample_weights, labels, train_index):
    train_logits = sample_logits[:, train_index, :]
    num_samples, num_train, num_classes = train_logits.shape
    repeated_labels = labels[train_index].unsqueeze(0).expand(num_samples, -1).reshape(-1)
    sample_losses = F.cross_entropy(
        train_logits.reshape(num_samples * num_train, num_classes),
        repeated_labels,
        reduction="none",
    ).view(num_samples, num_train)
    sample_weights = sample_weights.to(device=sample_losses.device, dtype=sample_losses.dtype)
    mean_loss = (sample_weights.view(-1, 1) * sample_losses).sum(dim=0)
    centered = sample_losses - mean_loss.unsqueeze(0)
    loss_var = (sample_weights.view(-1, 1) * centered.pow(2)).sum(dim=0)
    return mean_loss, loss_var


def compute_tar_lossfield_node_risk(mean_loss, loss_var, train_graph_input, model, args):
    node_risk = mean_loss.detach()
    loss_std = loss_var.detach().clamp_min(0.0).add(1e-8).sqrt()
    coeffs_lp, coeffs_hp = _extract_tar_filter_coeffs(
        model,
        args,
        device=loss_std.device,
        dtype=loss_std.dtype,
    )

    uncertainty_column = loss_std.unsqueeze(-1)
    spectral_lp = chebyshev_filter(train_graph_input, uncertainty_column, coeffs_lp).squeeze(-1)
    spectral_hp = chebyshev_filter(train_graph_input, uncertainty_column, coeffs_hp).squeeze(-1).abs()

    if args.tar_eta0 > 0:
        node_risk = node_risk + args.tar_eta0 * _normalize_positive_signal(loss_std)
    if args.tar_eta_lp > 0:
        node_risk = node_risk + args.tar_eta_lp * _normalize_signed_signal(spectral_lp)
    if args.tar_eta_hp > 0:
        node_risk = node_risk + args.tar_eta_hp * _normalize_positive_signal(spectral_hp)

    return node_risk


def build_support_prior_tensors(adj, features, labels, train_mask, num_classes, args, device):
    train_mask_cpu = train_mask.detach().cpu().numpy().astype(bool)
    labels_cpu = labels.detach().cpu().numpy()
    num_nodes = int(labels.shape[0])

    train_onehot = np.zeros((num_nodes, num_classes), dtype=np.float32)
    train_onehot[train_mask_cpu, labels_cpu[train_mask_cpu]] = 1.0

    one_hop = adj.dot(train_onehot).astype(np.float32)
    if float(getattr(args, "support_twohop_weight", 0.0)) != 0.0:
        two_hop = adj.dot(one_hop).astype(np.float32)
    else:
        two_hop = np.zeros_like(one_hop)

    support_counts = one_hop + float(args.support_twohop_weight) * two_hop
    train_class_counts = train_onehot[train_mask_cpu].sum(axis=0)
    if float(train_class_counts.sum()) <= 0:
        train_class_prior = np.full((num_classes,), 1.0 / max(num_classes, 1), dtype=np.float32)
    else:
        train_class_prior = train_class_counts / float(train_class_counts.sum())

    support_probs = support_counts + float(args.support_smoothing) * train_class_prior[None, :]
    support_probs = support_probs / np.clip(support_probs.sum(axis=1, keepdims=True), 1e-8, None)
    graph_logits = np.log(np.clip(support_probs, 1e-8, None))
    graph_logits = graph_logits - graph_logits.mean(axis=1, keepdims=True)

    support_mass = support_counts.sum(axis=1)
    nonzero_mass = support_mass[support_mass > 0]
    ref_mass = float(nonzero_mass.mean()) if nonzero_mass.size > 0 else 1.0
    support_gate = 1.0 / (1.0 + support_mass / max(ref_mass, 1e-8))

    feature_tensor = features.detach()
    feat_dtype = feature_tensor.dtype
    feature_norm = F.normalize(feature_tensor, dim=1)
    prototype_rows = []
    for class_idx in range(num_classes):
        class_mask = (labels == class_idx) & train_mask
        if int(class_mask.sum().item()) == 0:
            prototype_rows.append(torch.zeros((feature_tensor.shape[1],), device=feature_tensor.device, dtype=feat_dtype))
            continue
        prototype_rows.append(feature_tensor[class_mask].mean(dim=0))
    prototype_matrix = torch.stack(prototype_rows, dim=0)
    prototype_matrix = F.normalize(prototype_matrix, dim=1)
    proto_scores = (feature_norm @ prototype_matrix.t()) / max(float(args.support_proto_temp), 1e-6)
    proto_scores = proto_scores - proto_scores.mean(dim=1, keepdim=True)

    return {
        "graph_logits": torch.from_numpy(graph_logits).to(device=device, dtype=feat_dtype),
        "proto_scores": proto_scores.to(device=device, dtype=feat_dtype),
        "support_gate": torch.from_numpy(support_gate).to(device=device, dtype=feat_dtype),
    }


def build_row_normalized_adj_tensor(adj, device, dtype):
    row_sums = np.asarray(adj.sum(axis=1)).reshape(-1)
    inv_row = np.zeros_like(row_sums, dtype=np.float32)
    nonzero = row_sums > 0
    inv_row[nonzero] = 1.0 / row_sums[nonzero]
    adj_norm = sp.diags(inv_row).dot(adj).tocoo()
    indices = torch.from_numpy(np.vstack([adj_norm.row, adj_norm.col]).astype(np.int64)).to(device=device)
    values = torch.from_numpy(adj_norm.data.astype(np.float32)).to(device=device, dtype=dtype)
    return torch.sparse_coo_tensor(indices, values, size=adj_norm.shape, device=device, dtype=dtype).coalesce()


def extract_model_embeddings(model, backbone_name, x, edge_index):
    base_module = None
    if hasattr(model, "base_sage"):
        base_module = model.base_sage
    elif hasattr(model, "base_gcn"):
        base_module = model.base_gcn
    elif hasattr(model, "base_appnp"):
        base_module = model.base_appnp
    elif hasattr(model, "base_gpr"):
        base_module = model.base_gpr
    elif hasattr(model, "forward_with_graph_ebm_views"):
        base_module = model

    if base_module is not None and hasattr(base_module, "forward_with_graph_ebm_views"):
        views = base_module.forward_with_graph_ebm_views(x, edge_index)
        return views["embeddings"]
    return None


def compute_train_class_prototypes(embeddings, labels, train_mask, num_classes):
    train_embeddings = embeddings[train_mask]
    train_labels = labels[train_mask]
    prototype_rows = []
    for class_idx in range(num_classes):
        class_mask = train_labels == class_idx
        if int(class_mask.sum().item()) == 0:
            prototype_rows.append(
                torch.zeros((embeddings.shape[1],), device=embeddings.device, dtype=embeddings.dtype)
            )
        else:
            prototype_rows.append(train_embeddings[class_mask].mean(dim=0))
    return torch.stack(prototype_rows, dim=0), train_embeddings, train_labels


def build_embedding_retrieval_logits(embeddings, labels, train_mask, num_classes, topk, temp):
    emb_norm = F.normalize(embeddings, dim=1)
    train_emb = emb_norm[train_mask]
    train_labels = labels[train_mask]
    if train_emb.shape[0] == 0:
        return emb_norm.new_zeros((emb_norm.shape[0], num_classes))
    sim = emb_norm @ train_emb.t()
    topk = max(1, min(int(topk), train_emb.shape[0]))
    topk_vals, topk_idx = torch.topk(sim, k=topk, dim=1)
    topk_labels = train_labels[topk_idx]
    weights = F.softmax(topk_vals / max(float(temp), 1e-6), dim=1)
    target_probs = emb_norm.new_zeros((emb_norm.shape[0], num_classes))
    target_probs.scatter_add_(1, topk_labels, weights)
    target_probs = target_probs / target_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
    retrieval_logits = torch.log(target_probs.clamp_min(1e-8))
    retrieval_logits = retrieval_logits - retrieval_logits.mean(dim=1, keepdim=True)
    return retrieval_logits


def build_pair_specialist_logits(embeddings, labels, train_mask, num_classes, pair_classes, temp):
    prototypes, _train_embeddings, _train_labels = compute_train_class_prototypes(
        embeddings, labels, train_mask, num_classes
    )
    emb_norm = F.normalize(embeddings, dim=1)
    proto_norm = F.normalize(prototypes, dim=1)
    specialist = emb_norm.new_zeros((embeddings.shape[0], num_classes))
    if not pair_classes:
        return specialist
    pair_idx = torch.tensor(pair_classes, device=embeddings.device, dtype=torch.long)
    pair_logits = (emb_norm @ proto_norm[pair_idx].t()) / max(float(temp), 1e-6)
    pair_logits = pair_logits - pair_logits.mean(dim=1, keepdim=True)
    specialist[:, pair_idx] = pair_logits
    return specialist


def update_dynamic_support_prior(support_prior, model, backbone_name, features, edge_index, labels, train_mask, num_classes, args):
    embeddings = extract_model_embeddings(model, backbone_name, features, edge_index)
    support_prior["model_embeddings"] = embeddings
    if embeddings is None:
        support_prior["retrieval_logits"] = None
        support_prior["pair_specialist_logits"] = None
        return
    support_prior["retrieval_logits"] = build_embedding_retrieval_logits(
        embeddings,
        labels,
        train_mask,
        num_classes=num_classes,
        topk=int(getattr(args, "retrieval_posthoc_topk", 8)),
        temp=float(getattr(args, "retrieval_posthoc_temp", 0.2)),
    )
    support_prior["pair_specialist_logits"] = build_pair_specialist_logits(
        embeddings,
        labels,
        train_mask,
        num_classes=num_classes,
        pair_classes=(1, 2, 3),
        temp=float(getattr(args, "pair_specialist_temp", 0.2)),
    )


def apply_posthoc_decision_correction(logits, uncertainty, support_prior, spec, args, train_mask):
    posthoc_mode = spec.get("posthoc_mode")
    if posthoc_mode is None:
        return logits

    support_logits = (
        float(getattr(args, "support_graph_gamma", 0.0)) * support_prior["graph_logits"]
        + float(getattr(args, "support_proto_gamma", 0.0)) * support_prior["proto_scores"]
    )
    if torch.allclose(support_logits.abs().max(), support_logits.new_tensor(0.0)):
        return logits

    if posthoc_mode == "support_add":
        alpha = float(getattr(args, "support_gain", 1.0)) * support_prior["support_gate"]
        return logits + alpha.unsqueeze(-1) * support_logits

    if posthoc_mode == "uncertainty_mix":
        if uncertainty is None:
            alpha = logits.new_full((logits.shape[0],), float(getattr(args, "uncertainty_mix_gamma", 0.0)))
        else:
            train_uncertainty = uncertainty[train_mask]
            unc_scale = train_uncertainty.mean().clamp_min(1e-8)
            alpha = (float(getattr(args, "uncertainty_mix_gamma", 0.0)) * (uncertainty / unc_scale)).clamp(
                min=0.0,
                max=float(getattr(args, "uncertainty_mix_cap", 1.0)),
            )
        return logits + alpha.unsqueeze(-1) * support_logits

    if posthoc_mode == "retrieval_support_add":
        retrieval_logits = support_prior.get("retrieval_logits")
        if retrieval_logits is None:
            return logits
        alpha = float(getattr(args, "retrieval_posthoc_gamma", 0.0)) * support_prior["support_gate"].clamp_min(0.0).pow(
            float(getattr(args, "retrieval_posthoc_power", 1.0))
        )
        return logits + alpha.unsqueeze(-1) * retrieval_logits

    if posthoc_mode == "pair_proto_specialist":
        specialist_logits = support_prior.get("pair_specialist_logits")
        if specialist_logits is None:
            return logits
        base_probs = torch.softmax(logits, dim=1)
        alpha = float(getattr(args, "pair_specialist_gamma", 0.0)) * base_probs[:, [1, 2, 3]].sum(dim=1)
        if uncertainty is not None and float(getattr(args, "pair_specialist_unc_scale", 0.0)) > 0:
            unc = uncertainty / uncertainty.mean().clamp_min(1e-8)
            alpha = alpha * (1.0 + float(getattr(args, "pair_specialist_unc_scale", 0.0)) * unc)
        return logits + alpha.unsqueeze(-1) * specialist_logits

    if posthoc_mode == "retrieval_switch":
        retrieval_logits = support_prior.get("retrieval_logits")
        if retrieval_logits is None:
            return logits
        base_probs = torch.softmax(logits, dim=1)
        support_gate = support_prior["support_gate"].clamp_min(0.0)
        low_support = support_gate >= float(getattr(args, "retrieval_switch_support_thresh", 0.6))
        low_conf = base_probs.max(dim=1).values <= float(getattr(args, "retrieval_switch_conf_thresh", 0.6))
        switch_mask = low_support & low_conf
        if not torch.any(switch_mask):
            return logits
        gamma = float(getattr(args, "retrieval_switch_gamma", 1.0))
        corrected = logits.clone()
        corrected[switch_mask] = (1.0 - gamma) * logits[switch_mask] + gamma * retrieval_logits[switch_mask]
        return corrected

    if posthoc_mode == "pair_proto_switch":
        specialist_logits = support_prior.get("pair_specialist_logits")
        if specialist_logits is None:
            return logits
        base_probs = torch.softmax(logits, dim=1)
        pair_mass = base_probs[:, [1, 2, 3]].sum(dim=1)
        top2 = torch.topk(base_probs, k=2, dim=1).indices
        pair_set = torch.tensor([1, 2, 3], device=logits.device)
        top2_in_pair = torch.isin(top2, pair_set).all(dim=1)
        top2_vals = torch.topk(base_probs, k=2, dim=1).values
        low_margin = (top2_vals[:, 0] - top2_vals[:, 1]) <= float(getattr(args, "pair_switch_margin_thresh", 0.25))
        switch_mask = (pair_mass >= float(getattr(args, "pair_switch_pairmass_thresh", 0.6))) & top2_in_pair & low_margin
        if uncertainty is not None and float(getattr(args, "pair_switch_unc_thresh", 0.0)) > 0:
            switch_mask = switch_mask & (uncertainty >= float(getattr(args, "pair_switch_unc_thresh", 0.0)))
        if not torch.any(switch_mask):
            return logits
        gamma = float(getattr(args, "pair_switch_gamma", 1.0))
        corrected = logits.clone()
        corrected[switch_mask, 1:4] = (
            (1.0 - gamma) * logits[switch_mask, 1:4] + gamma * specialist_logits[switch_mask, 1:4]
        )
        return corrected

    raise ValueError(f"Unknown posthoc_mode: {posthoc_mode}")


def apply_auxiliary_training_loss(loss, logits, uncertainty, per_node_loss, labels, train_mask, support_prior, spec, args):
    aux_mode = spec.get("aux_mode")
    if aux_mode is None:
        return loss

    if aux_mode == "support_weighted_train":
        gamma = float(getattr(args, "support_train_gamma", 0.0))
        if gamma <= 0:
            return loss
        weights = 1.0 + gamma * support_prior["support_gate"][train_mask]
        weights = weights / weights.mean().clamp_min(1e-8)
        return torch.mean(weights * per_node_loss)

    if aux_mode == "uncertainty_margin":
        gamma = float(getattr(args, "unc_margin_gamma", 0.0))
        target = float(getattr(args, "unc_margin_target", 0.0))
        if gamma <= 0 or target <= 0:
            return loss
        train_logits = logits[train_mask]
        train_labels = labels[train_mask]
        true_logits = train_logits.gather(1, train_labels.view(-1, 1)).squeeze(1)
        masked_logits = train_logits.clone()
        masked_logits[torch.arange(train_logits.shape[0], device=train_logits.device), train_labels] = float("-inf")
        max_wrong = masked_logits.max(dim=1).values
        margin_violation = (target - (true_logits - max_wrong)).clamp_min(0.0)
        if uncertainty is not None:
            unc = uncertainty[train_mask].detach()
            unc = unc / unc.mean().clamp_min(1e-8)
        else:
            unc = margin_violation.new_ones(margin_violation.shape)
        return loss + gamma * torch.mean(unc * margin_violation)

    if aux_mode == "prototype_lp_aux":
        proto_gamma = float(getattr(args, "proto_aux_gamma", 0.0))
        lp_gamma = float(getattr(args, "proto_lp_gamma", 0.0))
        if proto_gamma <= 0 and lp_gamma <= 0:
            return loss
        embeddings = support_prior.get("model_embeddings")
        adj_norm = support_prior.get("adj_row_norm")
        if embeddings is None or adj_norm is None:
            return loss
        num_classes = logits.shape[1]
        prototypes, _train_embeddings, train_labels = compute_train_class_prototypes(
            embeddings, labels, train_mask, num_classes
        )
        emb_norm = F.normalize(embeddings, dim=1)
        proto_norm = F.normalize(prototypes, dim=1)
        proto_logits = (emb_norm @ proto_norm.t()) / max(float(getattr(args, "proto_aux_temp", 0.5)), 1e-6)
        aux_loss = proto_logits.new_tensor(0.0)
        if proto_gamma > 0:
            aux_loss = aux_loss + proto_gamma * F.cross_entropy(proto_logits[train_mask], train_labels)
        if lp_gamma > 0:
            lp_proto_logits = torch.sparse.mm(adj_norm, proto_logits)
            aux_loss = aux_loss + lp_gamma * F.cross_entropy(lp_proto_logits[train_mask], train_labels)
        return loss + aux_loss

    if aux_mode == "retrieval_teacher_aux":
        gamma = float(getattr(args, "retrieval_aux_gamma", 0.0))
        if gamma <= 0:
            return loss
        embeddings = support_prior.get("model_embeddings")
        if embeddings is None:
            return loss
        train_logits = logits[train_mask]
        train_labels = labels[train_mask]
        train_embeddings = F.normalize(embeddings[train_mask], dim=1)
        if train_embeddings.shape[0] <= 1:
            return loss
        sim = train_embeddings @ train_embeddings.t()
        sim.fill_diagonal_(float("-inf"))
        topk = max(1, min(int(getattr(args, "retrieval_topk", 8)), train_embeddings.shape[0] - 1))
        topk_vals, topk_idx = torch.topk(sim, k=topk, dim=1)
        topk_labels = train_labels[topk_idx]
        weights = F.softmax(topk_vals / max(float(getattr(args, "retrieval_temp", 0.2)), 1e-6), dim=1)
        target_probs = train_logits.new_zeros(train_logits.shape)
        target_probs.scatter_add_(1, topk_labels, weights)
        target_probs = target_probs / target_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
        gate = support_prior["support_gate"][train_mask].detach().clamp_min(0.0)
        gate = gate.pow(float(getattr(args, "retrieval_gate_power", 1.0)))
        gate = gate / gate.mean().clamp_min(1e-8)
        teacher_loss = F.kl_div(
            F.log_softmax(train_logits, dim=1),
            target_probs.detach(),
            reduction="none",
        ).sum(dim=1)
        return loss + gamma * torch.mean(gate * teacher_loss)

    if aux_mode == "pair_confusion_margin":
        gamma = float(getattr(args, "pair_margin_gamma", 0.0))
        target = float(getattr(args, "pair_margin_target", 0.0))
        if gamma <= 0 or target <= 0:
            return loss
        pair_map = {1: (2,), 2: (1, 3), 3: (2,)}
        train_logits = logits[train_mask]
        train_labels = labels[train_mask]
        penalties = []
        for class_idx, bad_classes in pair_map.items():
            class_mask = train_labels == class_idx
            if int(class_mask.sum().item()) == 0:
                continue
            true_logits = train_logits[class_mask, class_idx]
            class_penalty = true_logits.new_zeros(true_logits.shape)
            for bad_class in bad_classes:
                class_penalty = class_penalty + (target - (true_logits - train_logits[class_mask, bad_class])).clamp_min(0.0)
            penalties.append(class_penalty)
        if not penalties:
            return loss
        pair_penalty = torch.cat(penalties, dim=0)
        return loss + gamma * pair_penalty.mean()

    if aux_mode == "pair_prototype_separation":
        gamma = float(getattr(args, "pair_proto_gamma", 0.0))
        target = float(getattr(args, "pair_proto_target", 0.0))
        if gamma <= 0 or target <= 0:
            return loss
        embeddings = support_prior.get("model_embeddings")
        if embeddings is None:
            return loss
        pair_map = {1: (2, 3), 2: (1, 3), 3: (1, 2)}
        prototypes, _train_embeddings, train_labels = compute_train_class_prototypes(
            embeddings, labels, train_mask, logits.shape[1]
        )
        emb_norm = F.normalize(embeddings[train_mask], dim=1)
        proto_norm = F.normalize(prototypes, dim=1)
        sim = emb_norm @ proto_norm.t()
        penalties = []
        for class_idx, bad_classes in pair_map.items():
            class_mask = train_labels == class_idx
            if int(class_mask.sum().item()) == 0:
                continue
            true_sim = sim[class_mask, class_idx]
            class_penalty = true_sim.new_zeros(true_sim.shape)
            for bad_class in bad_classes:
                class_penalty = class_penalty + (target - (true_sim - sim[class_mask, bad_class])).clamp_min(0.0)
            penalties.append(class_penalty)
        if not penalties:
            return loss
        return loss + gamma * torch.cat(penalties, dim=0).mean()

    raise ValueError(f"Unknown aux_mode: {aux_mode}")


def train_and_collect(model_key, dataset, config, args, seed):
    set_seed(seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.device}")
    spec = MODEL_SPECS[model_key]
    backbone_name = spec["backbone"]
    train_mode = spec["train_mode"]

    adj, edge_index, features, labels, masks = _good_data_to_dssgnn_format(dataset, config)
    train_mask_cpu = masks["train_mask"]
    train_nodes_cpu = train_mask_cpu.nonzero(as_tuple=False).view(-1).cpu().numpy()
    train_mask = masks["train_mask"].to(device)
    val_mask = masks["val_mask"].to(device)
    test_mask = masks["test_mask"].to(device)
    id_val_mask = masks.get("id_val_mask", masks["val_mask"]).to(device)
    id_test_mask = masks.get("id_test_mask", masks["test_mask"]).to(device)

    graph_input = build_rescaled_laplacian(adj, lambda_max=args.lambda_max).to(device)
    train_graph_input = build_rescaled_laplacian(adj[train_nodes_cpu][:, train_nodes_cpu], lambda_max=args.lambda_max).to(device)
    edge_index = edge_index.to(device)
    features = features.to(device)
    labels = labels.to(device)
    transport_graph = build_transport_graph(edge_index, train_mask)
    train_index = train_mask.nonzero(as_tuple=False).view(-1)

    num_classes = int(labels.max().item()) + 1
    input_dim = features.shape[1]
    model = build_model(backbone_name, input_dim, args.hidden, num_classes, args).to(device)
    support_prior = build_support_prior_tensors(adj, features, labels, train_mask, num_classes, args, device)
    support_prior["adj_row_norm"] = build_row_normalized_adj_tensor(adj, device=device, dtype=features.dtype)
    supports_uncertainty = backbone_name == "dssgnn" or hasattr(model, "forward_with_uncertainty")
    supports_predictive_stats = backbone_name == "dssgnn" or hasattr(model, "forward_with_predictive_stats")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0
    selection_mask = id_val_mask if train_mode == "gnnsafepp" else val_mask

    for _epoch in range(args.epochs):
        if hasattr(model, "set_train_epoch"):
            model.set_train_epoch(_epoch)
        model.train()
        optimizer.zero_grad()
        logits, uncertainty = forward_model(model, backbone_name, graph_input, edge_index, features)
        per_node_loss = F.cross_entropy(logits[train_mask], labels[train_mask], reduction="none")
        loss = per_node_loss.mean()
        if supports_uncertainty:
            loss = loss + args.lambda_reg * uncertainty.mean()

        if train_mode == "gnnsafepp":
            signal = energy_from_logits(logits)
            signal = propagate_scalar_score(signal, edge_index, args.gnnsafe_K, args.gnnsafe_alpha)
            reg_loss = _gnnsafe_margin_loss(signal[train_mask], signal[val_mask], args)
            loss = loss + args.lamda * reg_loss
        elif train_mode == "tar_reweight":
            node_risk = per_node_loss.detach()
            if supports_uncertainty and args.tar_eta > 0:
                train_uncertainty = uncertainty[train_mask].detach()
                train_uncertainty = train_uncertainty / train_uncertainty.mean().clamp_min(1e-8)
                node_risk = node_risk + args.tar_eta * train_uncertainty
            rho = compute_tar_reweight(node_risk, transport_graph, args)
            loss = torch.mean(rho * per_node_loss)
            if supports_uncertainty:
                loss = loss + args.lambda_reg * uncertainty.mean()
        elif train_mode == "tar_spectral_reweight":
            node_risk, raw_signal, lp_signal, _hp_signal = compute_tar_spectral_components(
                per_node_loss=per_node_loss,
                full_uncertainty=uncertainty if supports_uncertainty else None,
                graph_input=graph_input,
                train_mask=train_mask,
                model=model,
                args=args,
            )
            edge_weights = compute_tar_edge_weights(
                transport_graph,
                args,
                smooth_signal=lp_signal,
                boundary_signal=raw_signal,
                current_epoch=_epoch,
            )
            rho = compute_tar_reweight(node_risk, transport_graph, args, edge_weights=edge_weights)
            loss = torch.mean(rho * per_node_loss)
            if supports_uncertainty:
                loss = loss + args.lambda_reg * uncertainty.mean()
        elif train_mode == "tar_lossfield_reweight":
            if not supports_predictive_stats:
                raise ValueError(f"{backbone_name} does not expose predictive stats required for tar_lossfield_reweight.")
            predictive_stats = forward_predictive_stats(model, backbone_name, graph_input, edge_index, features)
            logits = predictive_stats["logits"]
            mean_loss, loss_var = compute_stochastic_train_loss_field(
                predictive_stats["sample_logits"],
                predictive_stats["weights"],
                labels,
                train_index,
            )
            node_risk = compute_tar_lossfield_node_risk(
                mean_loss=mean_loss,
                loss_var=loss_var,
                train_graph_input=train_graph_input,
                model=model,
                args=args,
            )
            rho = compute_tar_reweight(node_risk, transport_graph, args)
            loss = torch.mean(rho * mean_loss)
            if supports_uncertainty and args.lambda_reg > 0:
                _, uncertainty = forward_model(model, backbone_name, graph_input, edge_index, features)
                loss = loss + args.lambda_reg * uncertainty.mean()
        elif train_mode == "tar_adaptive_reweight":
            # Hybrid reweighting: TAR loss-based base + adaptive spectral correction.
            # Pure-loss transport (TAR baseline) always on:
            node_risk_base = per_node_loss.detach()
            rho_base = compute_tar_reweight(node_risk_base, transport_graph, args)
            # Spectral correction from DSS uncertainty (may or may not help):
            if supports_uncertainty:
                node_risk_spectral, raw_signal, lp_signal, _hp_signal = compute_tar_spectral_components(
                    per_node_loss=per_node_loss,
                    full_uncertainty=uncertainty,
                    graph_input=graph_input,
                    train_mask=train_mask,
                    model=model,
                    args=args,
                )
                edge_weights = compute_tar_edge_weights(
                    transport_graph,
                    args,
                    smooth_signal=lp_signal,
                    boundary_signal=raw_signal,
                    current_epoch=_epoch,
                )
                rho_spectral = compute_tar_reweight(node_risk_spectral, transport_graph, args, edge_weights=edge_weights)
                # Adaptive gate: interpolate between loss-only and spectral reweighting
                lam = args.tar_adaptive_lambda
                rho = (1.0 - lam) * rho_base + lam * rho_spectral
            else:
                rho = rho_base
            loss = torch.mean(rho * per_node_loss)
            if supports_uncertainty:
                loss = loss + args.lambda_reg * uncertainty.mean()

        update_dynamic_support_prior(
            support_prior,
            model,
            backbone_name,
            features,
            edge_index,
            labels,
            train_mask,
            num_classes,
            args,
        )
        loss = apply_auxiliary_training_loss(
            loss,
            logits=logits,
            uncertainty=uncertainty if supports_uncertainty else None,
            per_node_loss=per_node_loss,
            labels=labels,
            train_mask=train_mask,
            support_prior=support_prior,
            spec=spec,
            args=args,
        )

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            logits, eval_uncertainty = forward_model(model, backbone_name, graph_input, edge_index, features)
            logits = apply_posthoc_decision_correction(logits, eval_uncertainty, support_prior, spec, args, train_mask)
        val_loss = F.cross_entropy(logits[selection_mask], labels[selection_mask]).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = _epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    restore_best_state(model, best_state, best_epoch)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        update_dynamic_support_prior(
            support_prior,
            model,
            backbone_name,
            features,
            edge_index,
            labels,
            train_mask,
            num_classes,
            args,
        )
        logits, uncertainty = forward_model(model, backbone_name, graph_input, edge_index, features)
        logits = apply_posthoc_decision_correction(logits, uncertainty, support_prior, spec, args, train_mask)
        predictive_stats = None
        if supports_predictive_stats:
            predictive_stats = forward_predictive_stats(model, backbone_name, graph_input, edge_index, features)
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        topk = probs.topk(k=min(2, probs.shape[1]), dim=1).values
        if topk.shape[1] == 1:
            margin = topk[:, 0]
        else:
            margin = topk[:, 0] - topk[:, 1]
        true_prob = probs.gather(1, labels.view(-1, 1)).squeeze(1)

    split_metrics = {}
    split_masks = {
        "train": train_mask,
        "id_val": id_val_mask,
        "id_test": id_test_mask,
        "val": val_mask,
        "test": test_mask,
    }
    for split_name, mask in split_masks.items():
        split_metrics[split_name] = evaluate_split(logits[mask], labels[mask])

    extra_scores = {}
    energy = energy_from_logits(logits)
    if any(score.endswith("_prop") for score in spec["score_types"]):
        extra_scores["energy_prop"] = propagate_scalar_score(energy, edge_index, args.gnnsafe_K, args.gnnsafe_alpha)
    if predictive_stats is not None:
        pred_entropy_score = -predictive_stats["predictive_entropy"]
        mutual_info_score = -predictive_stats["mutual_info"]
        if "pred_entropy_prop" in spec["score_types"]:
            extra_scores["pred_entropy_prop"] = propagate_scalar_score(
                pred_entropy_score, edge_index, args.gnnsafe_K, args.gnnsafe_alpha
            )
        if "mutual_info_prop" in spec["score_types"]:
            extra_scores["mutual_info_prop"] = propagate_scalar_score(
                mutual_info_score, edge_index, args.gnnsafe_K, args.gnnsafe_alpha
            )
    if "graph_ebm" in spec["score_types"]:
        scorer = build_graph_ebm_scorer(args)
        views = model.forward_with_graph_ebm_views(features, edge_index)
        scorer.fit(
            logits=views["logits"],
            embeddings=views["embeddings"],
            edge_index=edge_index,
            y=labels,
            mask=train_mask,
        )
        extra_scores["graph_ebm"] = -scorer.get_uncertainty(
            logits_unpropagated=views["logits_unpropagated"],
            embeddings_unpropagated=views["embeddings_unpropagated"],
            edge_index=edge_index,
        )

    score_map = compute_score_map(
        logits,
        uncertainty if supports_uncertainty else None,
        predictive_stats=predictive_stats,
        extra_scores=extra_scores,
    )

    return {
        "split_metrics": split_metrics,
        "score_map": score_map,
        "best_val_loss": best_val_loss,
        "id_test_mask": id_test_mask.detach().cpu().numpy().astype(bool),
        "test_mask": test_mask.detach().cpu().numpy().astype(bool),
        "labels": labels.detach().cpu().numpy(),
        "preds": preds.detach().cpu().numpy(),
        "confidence": probs.max(dim=1).values.detach().cpu().numpy(),
        "true_prob": true_prob.detach().cpu().numpy(),
        "margin": margin.detach().cpu().numpy(),
        "env_id": _maybe_dataset_tensor_numpy(dataset, "env_id"),
        "domain_id": _maybe_dataset_tensor_numpy(dataset, "domain_id"),
    }


def dataset_slug_from_config(config):
    return f"{config.dataset.dataset_name}-{config.dataset.domain}-{config.dataset.shift_type}".lower()


def main():
    parser = argparse.ArgumentParser(description="Local GOOD comparison with density plots")
    parser.add_argument("--dataset", type=str, default="goodcbas", choices=sorted(PRESET_DATASETS.keys()))
    parser.add_argument("--config_path", type=str, default=None, help="Optional explicit GOOD config path")
    parser.add_argument(
        "--models",
        nargs="*",
        default=["gcn", "dssgnn", "gcn_dssres"],
        help=f"Model keys to evaluate. Choices: {', '.join(sorted(MODEL_SPECS.keys()))}",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--generate", action="store_true", help="Regenerate GOOD datasets when supported.")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--pro_dropout", type=float, default=0.5)
    parser.add_argument("--lin_dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", "--wd", dest="weight_decay", type=float, default=5e-4)
    parser.add_argument("--lambda_reg", type=float, default=0.01)
    parser.add_argument("--lambda_max", type=float, default=2.0)
    parser.add_argument("--use_random_gates", action="store_true")
    parser.add_argument("--P_gate", type=int, default=1)
    parser.add_argument("--quadrature_nodes", type=int, default=4)
    parser.add_argument("--shared_input_lift", action="store_true")
    parser.add_argument("--appnp_K", type=int, default=10)
    parser.add_argument("--appnp_alpha", type=float, default=0.1)
    parser.add_argument("--gpr_K", type=int, default=10)
    parser.add_argument("--gpr_alpha", type=float, default=0.1)
    parser.add_argument("--tar_inner_steps", type=int, default=3, help="Number of topology-aware transport steps for TAR-style reweighting.")
    parser.add_argument("--tar_beta", type=float, default=0.1, help="Entropy smoothing coefficient for TAR-style reweighting.")
    parser.add_argument("--tar_eta", type=float, default=0.0, help="Weight on normalized DSS uncertainty in the TAR-style node risk.")
    parser.add_argument("--tar_step_size", type=float, default=0.5, help="Explicit transport step size for TAR-style reweighting.")
    parser.add_argument("--tar_eta0", type=float, default=0.0, help="Weight on the raw normalized uncertainty component in the spectral TAR transport potential.")
    parser.add_argument("--tar_eta_lp", type=float, default=0.0, help="Weight on the low-pass filtered uncertainty component in the spectral TAR transport potential.")
    parser.add_argument("--tar_eta_hp", type=float, default=0.0, help="Weight on the high-pass filtered uncertainty component in the spectral TAR transport potential.")
    parser.add_argument("--tar_alpha_lp", type=float, default=0.0, help="Edge-conductance gain on the low-pass DSS uncertainty field for anisotropic TAR transport.")
    parser.add_argument("--tar_alpha_partial", type=float, default=0.0, help="Edge-conductance penalty on DSS uncertainty jumps for anisotropic TAR transport.")
    parser.add_argument("--tar_lambda_edge", type=float, default=1.0, help="Interpolation weight between isotropic and anisotropic TAR edge transport.")
    parser.add_argument("--tar_adaptive_lambda", type=float, default=0.5, help="Interpolation between loss-only TAR (0) and spectrally coupled TAR (1) in tar_adaptive_reweight mode.")
    parser.add_argument("--tar_edge_onset_epoch", type=int, default=0, help="Epoch at which anisotropic TAR edge transport becomes active.")
    parser.add_argument("--tar_gate_bias", type=float, default=0.0, help="Constant offset for local edge-wise anisotropy gating in TAR transport.")
    parser.add_argument("--tar_gate_lp", type=float, default=0.0, help="Gain on smooth uncertainty when computing local edge-wise anisotropy gates.")
    parser.add_argument("--tar_gate_boundary", type=float, default=0.0, help="Penalty on uncertainty jumps when computing local edge-wise anisotropy gates.")
    parser.add_argument("--support_graph_gamma", type=float, default=0.0, help="Weight on the train-neighbor class-support prior in the decision correction.")
    parser.add_argument("--support_proto_gamma", type=float, default=0.0, help="Weight on the train feature-prototype prior in the decision correction.")
    parser.add_argument("--support_twohop_weight", type=float, default=0.5, help="Relative weight of 2-hop train-label support in the graph prior.")
    parser.add_argument("--support_smoothing", type=float, default=1.0, help="Train-class prior smoothing added to graph support counts.")
    parser.add_argument("--support_gain", type=float, default=1.0, help="Global gain on the low-support support-prior correction.")
    parser.add_argument("--support_proto_temp", type=float, default=0.5, help="Temperature for the train feature-prototype similarity prior.")
    parser.add_argument("--uncertainty_mix_gamma", type=float, default=0.0, help="Gain on DSS uncertainty when turning on uncertainty-aware support mixing.")
    parser.add_argument("--uncertainty_mix_cap", type=float, default=1.0, help="Maximum per-node uncertainty mix coefficient.")
    parser.add_argument("--support_train_gamma", type=float, default=0.0, help="Extra weight placed on low-support train nodes in the supervised loss.")
    parser.add_argument("--unc_margin_gamma", type=float, default=0.0, help="Weight on the uncertainty-aware margin penalty over train nodes.")
    parser.add_argument("--unc_margin_target", type=float, default=0.0, help="Target true-vs-top-wrong logit margin for the uncertainty-aware penalty.")
    parser.add_argument("--proto_aux_gamma", type=float, default=0.0, help="Weight on the prototype classification auxiliary over model embeddings.")
    parser.add_argument("--proto_lp_gamma", type=float, default=0.0, help="Weight on the propagated prototype auxiliary over model embeddings.")
    parser.add_argument("--proto_aux_temp", type=float, default=0.5, help="Temperature for the embedding-prototype auxiliary logits.")
    parser.add_argument("--retrieval_aux_gamma", type=float, default=0.0, help="Weight on the embedding-space retrieval teacher auxiliary for low-support train nodes.")
    parser.add_argument("--retrieval_topk", type=int, default=8, help="Number of train neighbors used by the retrieval teacher auxiliary.")
    parser.add_argument("--retrieval_temp", type=float, default=0.2, help="Softmax temperature for retrieval-neighbor weights.")
    parser.add_argument("--retrieval_gate_power", type=float, default=1.0, help="Exponent applied to the low-support gate in the retrieval teacher auxiliary.")
    parser.add_argument("--retrieval_posthoc_gamma", type=float, default=0.0, help="Gain on the low-support retrieval post-hoc logit correction.")
    parser.add_argument("--retrieval_posthoc_topk", type=int, default=8, help="Number of train neighbors used by the retrieval post-hoc correction.")
    parser.add_argument("--retrieval_posthoc_temp", type=float, default=0.2, help="Softmax temperature for the retrieval post-hoc correction.")
    parser.add_argument("--retrieval_posthoc_power", type=float, default=1.0, help="Exponent on the low-support gate in the retrieval post-hoc correction.")
    parser.add_argument("--retrieval_switch_gamma", type=float, default=1.0, help="Blend weight for the retrieval fallback switch.")
    parser.add_argument("--retrieval_switch_support_thresh", type=float, default=0.6, help="Minimum low-support gate to activate retrieval switching.")
    parser.add_argument("--retrieval_switch_conf_thresh", type=float, default=0.6, help="Maximum confidence to activate retrieval switching.")
    parser.add_argument("--pair_margin_gamma", type=float, default=0.0, help="Weight on the class-pair-aware confusion margin penalty.")
    parser.add_argument("--pair_margin_target", type=float, default=0.0, help="Target margin against the configured confusion classes.")
    parser.add_argument("--pair_proto_gamma", type=float, default=0.0, help="Weight on the pair-specific prototype-separation auxiliary.")
    parser.add_argument("--pair_proto_target", type=float, default=0.0, help="Target embedding-similarity gap for the pair-specific prototype-separation auxiliary.")
    parser.add_argument("--pair_specialist_gamma", type=float, default=0.0, help="Gain on the pair-specialist post-hoc logits for classes 1/2/3.")
    parser.add_argument("--pair_specialist_temp", type=float, default=0.2, help="Temperature for the pair-specialist embedding prototypes.")
    parser.add_argument("--pair_specialist_unc_scale", type=float, default=0.0, help="Optional uncertainty scaling applied to the pair-specialist gate.")
    parser.add_argument("--pair_switch_gamma", type=float, default=1.0, help="Blend weight for the pair-specialist switch on classes 1/2/3.")
    parser.add_argument("--pair_switch_pairmass_thresh", type=float, default=0.6, help="Minimum probability mass on classes 1/2/3 to activate the pair switch.")
    parser.add_argument("--pair_switch_margin_thresh", type=float, default=0.25, help="Maximum top-1 vs top-2 probability margin to activate the pair switch.")
    parser.add_argument("--pair_switch_unc_thresh", type=float, default=0.0, help="Optional minimum uncertainty to activate the pair switch.")
    parser.add_argument("--warmup_base_epochs", type=int, default=25)
    parser.add_argument("--residual_scale_init", type=float, default=0.1)
    parser.add_argument("--K_lp", type=int, default=3)
    parser.add_argument("--K_hp", type=int, default=2)
    parser.add_argument("--P", type=int, default=1)
    parser.add_argument("--gnnsafe_K", type=int, default=2, help="Propagation depth for GOOD-adapted GNNSafe scores.")
    parser.add_argument("--gnnsafe_alpha", type=float, default=0.5, help="Propagation alpha for GOOD-adapted GNNSafe scores.")
    parser.add_argument("--lamda", type=float, default=1.0, help="OOD regularization weight for GOOD-adapted GNNSafe++.")
    parser.add_argument("--m_in", type=float, default=-5.0, help="IND energy margin for GOOD-adapted GNNSafe++.")
    parser.add_argument("--m_out", type=float, default=-1.0, help="OOD energy margin for GOOD-adapted GNNSafe++.")
    parser.add_argument("--graph_ebm_covariance_type", type=str, default="diagonal", choices=["diagonal", "full", "identity", "isotropic"])
    parser.add_argument("--graph_ebm_tied_covariance", action="store_true")
    parser.add_argument("--graph_ebm_gamma_correction", type=float, default=1.0)
    parser.add_argument("--graph_ebm_lambda_independent_energy", type=float, default=1.0)
    parser.add_argument("--graph_ebm_lambda_local_energy", type=float, default=1.0)
    parser.add_argument("--graph_ebm_lambda_group_energy", type=float, default=1.0)
    parser.add_argument("--graph_ebm_alpha", type=float, default=0.5)
    parser.add_argument("--graph_ebm_num_diffusion_steps", type=int, default=10)
    parser.add_argument("--graph_ebm_aggregation", type=str, default="sum", choices=["sum", "logsumexp"])
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument(
        "--save_artifacts_if_test_acc_above",
        type=float,
        default=None,
        help=(
            "Only write scores_long.csv and density grids if the best shifted-test "
            "accuracy strictly exceeds this threshold."
        ),
    )
    args = parser.parse_args()

    unknown_models = [model for model in args.models if model not in MODEL_SPECS]
    if unknown_models:
        raise ValueError(f"Unknown GOOD model keys: {unknown_models}. Expected subset of {sorted(MODEL_SPECS)}")
    selected_models = args.models

    config_path = args.config_path or PRESET_DATASETS[args.dataset]
    base_config = load_good_config(
        config_path=config_path,
        hidden=args.hidden,
        seed=args.seed,
        generate=args.generate,
    )

    from GOOD.data import load_dataset

    ensure_good_dataset_registered(base_config.dataset.dataset_name)
    dataset = load_dataset(base_config.dataset.dataset_name, base_config)
    dataset_slug = dataset_slug_from_config(base_config)
    tag = args.tag or f"r{args.runs}_e{args.epochs}"
    out_dir = REPO_ROOT / "results_local" / "good_density" / f"{dataset_slug}-{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_metrics = {name: [] for name in selected_models}
    score_metrics = {}
    all_scores = {}
    run_artifacts = []
    for model_name in selected_models:
        spec = MODEL_SPECS[model_name]
        for score_type in spec["score_types"]:
            key = f"{model_name}_{score_type}"
            score_metrics[key] = []
            all_scores[key] = {"ind": [], "ood": []}

    for run_idx in range(args.runs):
        seed = args.seed + run_idx
        for model_name in selected_models:
            run_result = train_and_collect(model_name, dataset, deepcopy(base_config), args, seed)
            run_artifacts.append(
                {
                    "run_idx": run_idx,
                    "seed": seed,
                    "model_name": model_name,
                    "run_result": run_result,
                }
            )
            model_metrics[model_name].append(
                {
                    "seed": seed,
                    "val_loss": run_result["best_val_loss"],
                    **run_result["split_metrics"],
                }
            )
            id_mask = run_result["id_test_mask"]
            test_mask = run_result["test_mask"]
            for score_type in MODEL_SPECS[model_name]["score_types"]:
                key = f"{model_name}_{score_type}"
                scores = run_result["score_map"][score_type]
                ind_scores = scores[id_mask]
                ood_scores = scores[test_mask]
                auroc, aupr, fpr95, _ = get_measures(ind_scores, ood_scores)
                score_metrics[key].append(
                    {
                        "auroc": auroc,
                        "aupr": aupr,
                        "fpr95": fpr95,
                        "id_test_acc": run_result["split_metrics"]["id_test"]["acc"],
                        "test_acc": run_result["split_metrics"]["test"]["acc"],
                        "id_test_ece": run_result["split_metrics"]["id_test"]["ece"],
                        "test_ece": run_result["split_metrics"]["test"]["ece"],
                        "val_loss": run_result["best_val_loss"],
                    }
                )
                all_scores[key]["ind"].append(ind_scores)
                all_scores[key]["ood"].append(ood_scores)

    split_rows = []
    node_prediction_rows = []
    for model_name, runs in model_metrics.items():
        label = MODEL_SPECS[model_name]["label"]
        for split in ("train", "id_val", "id_test", "val", "test"):
            split_rows.append(
                {
                    "model": model_name,
                    "label": label,
                    "split": split,
                    "acc_mean": summarize([m[split]["acc"] for m in runs])[0],
                    "acc_std": summarize([m[split]["acc"] for m in runs])[1],
                    "ece_mean": summarize([m[split]["ece"] for m in runs])[0],
                    "ece_std": summarize([m[split]["ece"] for m in runs])[1],
                    "mce_mean": summarize([m[split]["mce"] for m in runs])[0],
                    "mce_std": summarize([m[split]["mce"] for m in runs])[1],
                    "brier_mean": summarize([m[split]["brier"] for m in runs])[0],
                    "brier_std": summarize([m[split]["brier"] for m in runs])[1],
                }
            )

    for artifact in run_artifacts:
        run_idx = artifact["run_idx"]
        seed = artifact["seed"]
        model_name = artifact["model_name"]
        run_result = artifact["run_result"]
        spec = MODEL_SPECS[model_name]
        score_map = run_result["score_map"]
        split_masks = {
            "id_test": run_result["id_test_mask"],
            "test": run_result["test_mask"],
        }
        for split_name, split_mask in split_masks.items():
            for node_idx in np.flatnonzero(split_mask):
                row = {
                    "config": model_name,
                    "label": spec["label"],
                    "model": model_name,
                    "run_idx": run_idx,
                    "seed": seed,
                    "split": split_name,
                    "node_idx": int(node_idx),
                    "true_label": int(run_result["labels"][node_idx]),
                    "pred_label": int(run_result["preds"][node_idx]),
                    "correct": int(run_result["labels"][node_idx] == run_result["preds"][node_idx]),
                    "confidence": float(run_result["confidence"][node_idx]),
                    "true_prob": float(run_result["true_prob"][node_idx]),
                    "margin": float(run_result["margin"][node_idx]),
                    "env_id": "" if run_result["env_id"] is None else int(run_result["env_id"][node_idx]),
                    "domain_id": "" if run_result["domain_id"] is None else int(run_result["domain_id"][node_idx]),
                }
                for score_name in NODE_PREDICTION_SCORE_FIELDS:
                    row[score_name] = (
                        float(score_map[score_name][node_idx])
                        if score_name in score_map
                        else ""
                    )
                node_prediction_rows.append(row)

    summary_rows = []
    config_names = []
    for model_name in selected_models:
        spec = MODEL_SPECS[model_name]
        for score_type in spec["score_types"]:
            key = f"{model_name}_{score_type}"
            config_names.append(key)
            label = f"{spec['label']} {score_type.upper()}"
            auroc_mean, auroc_std = summarize([m["auroc"] for m in score_metrics[key]])
            aupr_mean, aupr_std = summarize([m["aupr"] for m in score_metrics[key]])
            fpr_mean, fpr_std = summarize([m["fpr95"] for m in score_metrics[key]])
            id_acc_mean, id_acc_std = summarize([m["id_test_acc"] for m in score_metrics[key]])
            test_acc_mean, test_acc_std = summarize([m["test_acc"] for m in score_metrics[key]])
            id_ece_mean, id_ece_std = summarize([m["id_test_ece"] for m in score_metrics[key]])
            test_ece_mean, test_ece_std = summarize([m["test_ece"] for m in score_metrics[key]])
            val_loss_mean, val_loss_std = summarize([m["val_loss"] for m in score_metrics[key]])
            summary_rows.append(
                {
                    "config": key,
                    "label": label,
                    "model": model_name,
                    "score_type": score_type,
                    "runs": args.runs,
                    "score_auroc_mean": auroc_mean,
                    "score_auroc_std": auroc_std,
                    "score_aupr_mean": aupr_mean,
                    "score_aupr_std": aupr_std,
                    "score_fpr95_mean": fpr_mean,
                    "score_fpr95_std": fpr_std,
                    "id_test_acc_mean": id_acc_mean,
                    "id_test_acc_std": id_acc_std,
                    "test_acc_mean": test_acc_mean,
                    "test_acc_std": test_acc_std,
                    "id_test_ece_mean": id_ece_mean,
                    "id_test_ece_std": id_ece_std,
                    "test_ece_mean": test_ece_mean,
                    "test_ece_std": test_ece_std,
                    "val_loss_mean": val_loss_mean,
                    "val_loss_std": val_loss_std,
                }
            )

    write_summary_csv(out_dir / "summary.csv", summary_rows)
    write_split_metrics_csv(out_dir / "split_metrics.csv", split_rows)

    best_test_acc = max(row["test_acc_mean"] for row in summary_rows)
    save_artifacts = True
    if args.save_artifacts_if_test_acc_above is not None:
        save_artifacts = best_test_acc > args.save_artifacts_if_test_acc_above

    if save_artifacts:
        long_score_rows = []
        global_scores = []
        for model_name in selected_models:
            spec = MODEL_SPECS[model_name]
            for score_type in spec["score_types"]:
                key = f"{model_name}_{score_type}"
                label = f"{spec['label']} {score_type.upper()}"
                ind_scores = np.concatenate(all_scores[key]["ind"], axis=0)
                ood_scores = np.concatenate(all_scores[key]["ood"], axis=0)
                global_scores.extend([ind_scores, ood_scores])
                for value in ind_scores:
                    long_score_rows.append(
                        {
                            "config": key,
                            "label": label,
                            "model": model_name,
                            "score_type": score_type,
                            "split": "ID_test",
                            "score": float(value),
                        }
                    )
                for value in ood_scores:
                    long_score_rows.append(
                        {
                            "config": key,
                            "label": label,
                            "model": model_name,
                            "score_type": score_type,
                            "split": "OOD_test",
                            "score": float(value),
                        }
                    )

        write_scores_csv(out_dir / "scores_long.csv", long_score_rows)
        write_node_predictions_csv(out_dir / "node_predictions_long.csv", node_prediction_rows)

        score_min = min(float(arr.min()) for arr in global_scores)
        score_max = max(float(arr.max()) for arr in global_scores)
        padding = 0.05 * max(score_max - score_min, 1e-6)
        shared_x_limits = (score_min - padding, score_max + padding)
        fig_title = (
            f"GOOD Local Score Density Comparison: "
            f"{base_config.dataset.dataset_name} / {base_config.dataset.domain} / {base_config.dataset.shift_type}"
        )
        save_density_grid(out_dir / "density_grid.png", config_names, summary_rows, all_scores, fig_title)
        save_density_grid(
            out_dir / "density_grid_shared_x.png",
            config_names,
            summary_rows,
            all_scores,
            f"{fig_title} (shared x-scale)",
            shared_x_limits,
        )
    else:
        print(
            "Skipping scores_long.csv and density grids: "
            f"best shifted-test accuracy {100 * best_test_acc:.2f} did not exceed "
            f"threshold {100 * args.save_artifacts_if_test_acc_above:.2f}."
        )

    metadata = {
        "dataset": base_config.dataset.dataset_name,
        "domain": base_config.dataset.domain,
        "shift_type": base_config.dataset.shift_type,
        "config_path": config_path,
        "runs": args.runs,
        "epochs": args.epochs,
        "patience": args.patience,
        "seed": args.seed,
        "models": selected_models,
        "shared_input_lift": args.shared_input_lift,
        "use_random_gates": args.use_random_gates,
        "P_gate": args.P_gate,
        "quadrature_nodes": args.quadrature_nodes,
        "gnnsafe_K": args.gnnsafe_K,
        "gnnsafe_alpha": args.gnnsafe_alpha,
        "support_graph_gamma": args.support_graph_gamma,
        "support_proto_gamma": args.support_proto_gamma,
        "support_twohop_weight": args.support_twohop_weight,
        "support_smoothing": args.support_smoothing,
        "support_gain": args.support_gain,
        "support_proto_temp": args.support_proto_temp,
        "uncertainty_mix_gamma": args.uncertainty_mix_gamma,
        "uncertainty_mix_cap": args.uncertainty_mix_cap,
        "support_train_gamma": args.support_train_gamma,
        "unc_margin_gamma": args.unc_margin_gamma,
        "unc_margin_target": args.unc_margin_target,
        "proto_aux_gamma": args.proto_aux_gamma,
        "proto_lp_gamma": args.proto_lp_gamma,
        "proto_aux_temp": args.proto_aux_temp,
        "retrieval_aux_gamma": args.retrieval_aux_gamma,
        "retrieval_topk": args.retrieval_topk,
        "retrieval_temp": args.retrieval_temp,
        "retrieval_gate_power": args.retrieval_gate_power,
        "retrieval_posthoc_gamma": args.retrieval_posthoc_gamma,
        "retrieval_posthoc_topk": args.retrieval_posthoc_topk,
        "retrieval_posthoc_temp": args.retrieval_posthoc_temp,
        "retrieval_posthoc_power": args.retrieval_posthoc_power,
        "retrieval_switch_gamma": args.retrieval_switch_gamma,
        "retrieval_switch_support_thresh": args.retrieval_switch_support_thresh,
        "retrieval_switch_conf_thresh": args.retrieval_switch_conf_thresh,
        "pair_margin_gamma": args.pair_margin_gamma,
        "pair_margin_target": args.pair_margin_target,
        "pair_proto_gamma": args.pair_proto_gamma,
        "pair_proto_target": args.pair_proto_target,
        "pair_specialist_gamma": args.pair_specialist_gamma,
        "pair_specialist_temp": args.pair_specialist_temp,
        "pair_specialist_unc_scale": args.pair_specialist_unc_scale,
        "pair_switch_gamma": args.pair_switch_gamma,
        "pair_switch_pairmass_thresh": args.pair_switch_pairmass_thresh,
        "pair_switch_margin_thresh": args.pair_switch_margin_thresh,
        "pair_switch_unc_thresh": args.pair_switch_unc_thresh,
        "best_test_acc_mean": best_test_acc,
        "save_artifacts_if_test_acc_above": args.save_artifacts_if_test_acc_above,
        "artifacts_saved": save_artifacts,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"\nSaved results to {out_dir}")
    print(f"- {out_dir / 'summary.csv'}")
    print(f"- {out_dir / 'split_metrics.csv'}")
    if save_artifacts:
        print(f"- {out_dir / 'scores_long.csv'}")
        print(f"- {out_dir / 'node_predictions_long.csv'}")
        print(f"- {out_dir / 'density_grid.png'}")
        print(f"- {out_dir / 'density_grid_shared_x.png'}")


if __name__ == "__main__":
    main()
