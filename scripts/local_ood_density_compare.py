#!/usr/bin/env python3
"""
Local small-scale OOD comparison with score-density visualization.

Focus:
- GNNSafe baselines on GCN
- DSS-GNN score variants (MSP / energy / chaos, with optional propagation)
- learned DSS-GNN score fusion on IND-valid vs OOD-train

Outputs:
- summary.csv
- scores_long.csv
- density_grid.png
- density_grid_shared_x.png
- metadata.json

Example:
  python scripts/local_ood_density_compare.py \
    --dataset cora --ood_type structure --epochs 30 --runs 1 --cpu
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
import torch
import torch.nn as nn
from matplotlib import use as matplotlib_use
from scipy.stats import gaussian_kde

from gnnsafe_ood.baselines import GDUQ, GraphEBM, MSP
from gnnsafe_ood.data_utils import (
    evaluate_detect,
    eval_acc,
    eval_rocauc,
    get_measures,
    rand_splits,
)
from gnnsafe_ood.dataset import load_dataset
from gnnsafe_ood.fusion import fit_logistic_score_fusion, score_with_fusion
from gnnsafe_ood.gnnsafe import GNNSafe

matplotlib_use("Agg")

import matplotlib.pyplot as plt


GNNSAFE_HYPER = {
    ("cora", "structure"): (-5, -1, 0.01),
    ("cora", "feature"): (-5, -1, 0.01),
    ("cora", "label"): (-5, -4, 1.0),
    ("amazon-photo", "structure"): (-9, -1, 0.1),
    ("amazon-photo", "feature"): (-9, -1, 1.0),
    ("amazon-photo", "label"): (-9, -4, 1.0),
    ("coauthor-cs", "structure"): (-5, -1, 0.1),
    ("coauthor-cs", "feature"): (-7, -1, 0.1),
    ("coauthor-cs", "label"): (-9, -2, 0.01),
    ("twitch", None): (-5, -1, 0.1),
    ("arxiv", None): (-9, -2, 0.01),
}


TRAIN_SPECS = {
    "gcn_sup": {
        "label": "GCN Sup",
        "method": "msp",
        "backbone": "gcn",
        "use_reg": False,
        "use_prop": False,
    },
    "gcn_graph_ebm_sup": {
        "label": "GCN + Graph-EBM",
        "method": "graph_ebm",
        "backbone": "gcn",
        "use_reg": False,
        "use_prop": False,
    },
    "gcn_reg": {
        "label": "GCN Energy FT",
        "method": "gnnsafe",
        "backbone": "gcn",
        "use_reg": True,
        "use_prop": False,
    },
    "gcn_reg_prop": {
        "label": "GCN GNNSafe++",
        "method": "gnnsafe",
        "backbone": "gcn",
        "use_reg": True,
        "use_prop": True,
    },
    "gcn_dssres_sup": {
        "label": "GCN+DSS Residual Sup",
        "method": "msp",
        "backbone": "gcn_dssres",
        "use_reg": False,
        "use_prop": False,
        "arg_overrides": {
            "K_lp": 4,
            "K_hp": 4,
            "P": 2,
            "dropout": 0.5,
            "weight_decay": 5e-4,
            "dss_lambda_reg": 0.01,
            "dss_reg_score_type": "chaos",
            "warmup_base_epochs": 50,
            "residual_scale_init": 0.1,
        },
    },
    "gcn_dssres_reg": {
        "label": "GCN+DSS Residual Energy FT",
        "method": "gnnsafe",
        "backbone": "gcn_dssres",
        "use_reg": True,
        "use_prop": False,
        "reg_score_type": "energy",
        "arg_overrides": {
            "K_lp": 4,
            "K_hp": 4,
            "P": 2,
            "dropout": 0.5,
            "weight_decay": 5e-4,
            "dss_lambda_reg": 0.01,
            "dss_reg_score_type": "chaos",
            "warmup_base_epochs": 50,
            "residual_scale_init": 0.1,
        },
    },
    "gcn_dssres_reg_prop": {
        "label": "GCN+DSS Residual GNNSafe++",
        "method": "gnnsafe",
        "backbone": "gcn_dssres",
        "use_reg": True,
        "use_prop": True,
        "reg_score_type": "energy",
        "arg_overrides": {
            "K_lp": 4,
            "K_hp": 4,
            "P": 2,
            "dropout": 0.5,
            "weight_decay": 5e-4,
            "dss_lambda_reg": 0.01,
            "dss_reg_score_type": "chaos",
            "warmup_base_epochs": 50,
            "residual_scale_init": 0.1,
        },
    },
    "gcn_dssres_predent_reg": {
        "label": "GCN+DSS Residual PredEnt FT",
        "method": "gnnsafe",
        "backbone": "gcn_dssres",
        "use_reg": True,
        "use_prop": False,
        "reg_score_type": "pred_entropy",
        "arg_overrides": {
            "K_lp": 4,
            "K_hp": 4,
            "P": 2,
            "dropout": 0.5,
            "weight_decay": 5e-4,
            "dss_lambda_reg": 0.01,
            "dss_reg_score_type": "chaos",
            "warmup_base_epochs": 50,
            "residual_scale_init": 0.1,
        },
    },
    "gcn_dssres_predent_reg_prop": {
        "label": "GCN+DSS Residual PredEnt Safe",
        "method": "gnnsafe",
        "backbone": "gcn_dssres",
        "use_reg": True,
        "use_prop": True,
        "reg_score_type": "pred_entropy",
        "arg_overrides": {
            "K_lp": 4,
            "K_hp": 4,
            "P": 2,
            "dropout": 0.5,
            "weight_decay": 5e-4,
            "dss_lambda_reg": 0.01,
            "dss_reg_score_type": "chaos",
            "warmup_base_epochs": 50,
            "residual_scale_init": 0.1,
        },
    },
    "dssgnn_sup": {
        "label": "DSS-GNN Sup",
        "method": "msp",
        "backbone": "dssgnn",
        "use_reg": False,
        "use_prop": False,
    },
    "dssgnn_sup_tuned": {
        "label": "DSS-GNN Sup Tuned",
        "method": "msp",
        "backbone": "dssgnn",
        "use_reg": False,
        "use_prop": False,
        "arg_overrides": {
            "K_lp": 4,
            "K_hp": 4,
            "P": 2,
            "dropout": 0.5,
            "weight_decay": 5e-4,
            "dss_lambda_reg": 0.01,
            "dss_reg_score_type": "chaos",
        },
    },
    "dssgnn_sup_lowlabel": {
        "label": "DSS-GNN Sup LowLabel",
        "method": "msp",
        "backbone": "dssgnn",
        "use_reg": False,
        "use_prop": False,
        "arg_overrides": {
            "K_lp": 4,
            "K_hp": 4,
            "P": 2,
            "dropout": 0.5,
            "weight_decay": 5e-4,
            "dss_lambda_reg": 0.01,
            "dss_reg_score_type": "chaos",
            "shared_input_lift": True,
        },
    },
    "dssgnn_reg": {
        "label": "DSS-GNN Energy FT",
        "method": "gnnsafe",
        "backbone": "dssgnn",
        "use_reg": True,
        "use_prop": False,
    },
    "dssgnn_reg_tuned": {
        "label": "DSS-GNN Energy FT Tuned",
        "method": "gnnsafe",
        "backbone": "dssgnn",
        "use_reg": True,
        "use_prop": False,
        "arg_overrides": {
            "K_lp": 4,
            "K_hp": 4,
            "P": 2,
            "dropout": 0.5,
            "weight_decay": 5e-4,
            "dss_lambda_reg": 0.01,
            "dss_reg_score_type": "chaos",
        },
    },
    "dssgnn_reg_prop": {
        "label": "DSS-GNN GNNSafe++",
        "method": "gnnsafe",
        "backbone": "dssgnn",
        "use_reg": True,
        "use_prop": True,
        "reg_score_type": "energy",
    },
    "dssgnn_reg_prop_tuned": {
        "label": "DSS-GNN GNNSafe++ Tuned",
        "method": "gnnsafe",
        "backbone": "dssgnn",
        "use_reg": True,
        "use_prop": True,
        "reg_score_type": "energy",
        "arg_overrides": {
            "K_lp": 4,
            "K_hp": 4,
            "P": 2,
            "dropout": 0.5,
            "weight_decay": 5e-4,
            "dss_lambda_reg": 0.01,
            "dss_reg_score_type": "chaos",
        },
    },
    "dssgnn_chaos_reg": {
        "label": "DSS-GNN Chaos FT",
        "method": "gnnsafe",
        "backbone": "dssgnn",
        "use_reg": True,
        "use_prop": False,
        "reg_score_type": "chaos",
    },
    "dssgnn_chaos_reg_prop": {
        "label": "DSS-GNN Chaos Safe",
        "method": "gnnsafe",
        "backbone": "dssgnn",
        "use_reg": True,
        "use_prop": True,
        "reg_score_type": "chaos",
    },
    "dssgnn_chaos_norm_reg": {
        "label": "DSS-GNN ChaosNorm FT",
        "method": "gnnsafe",
        "backbone": "dssgnn",
        "use_reg": True,
        "use_prop": False,
        "reg_score_type": "chaos_norm",
    },
    "dssgnn_chaos_layernorm_reg": {
        "label": "DSS-GNN ChaosLayer FT",
        "method": "gnnsafe",
        "backbone": "dssgnn",
        "use_reg": True,
        "use_prop": False,
        "reg_score_type": "chaos_layernorm",
    },
    "dssgnn_chaos_ratio_reg": {
        "label": "DSS-GNN ChaosRatio FT",
        "method": "gnnsafe",
        "backbone": "dssgnn",
        "use_reg": True,
        "use_prop": False,
        "reg_score_type": "chaos_ratio",
    },
    "dssgnn_chaos_ratio_reg_tuned": {
        "label": "DSS-GNN ChaosRatio FT Tuned",
        "method": "gnnsafe",
        "backbone": "dssgnn",
        "use_reg": True,
        "use_prop": False,
        "reg_score_type": "chaos_ratio",
        "arg_overrides": {
            "K_lp": 4,
            "K_hp": 4,
            "P": 2,
            "dropout": 0.5,
            "weight_decay": 5e-4,
            "dss_lambda_reg": 0.01,
            "dss_reg_score_type": "chaos",
        },
    },
    "dssgnn_chaos_ratio_reg_lowlabel": {
        "label": "DSS-GNN ChaosRatio FT LowLabel",
        "method": "gnnsafe",
        "backbone": "dssgnn",
        "use_reg": True,
        "use_prop": False,
        "reg_score_type": "chaos_ratio",
        "arg_overrides": {
            "K_lp": 4,
            "K_hp": 4,
            "P": 2,
            "dropout": 0.5,
            "weight_decay": 5e-4,
            "dss_lambda_reg": 0.01,
            "dss_reg_score_type": "chaos",
            "shared_input_lift": True,
        },
    },
    "dssgnn_chaos_layerratio_reg": {
        "label": "DSS-GNN ChaosLayerRatio FT",
        "method": "gnnsafe",
        "backbone": "dssgnn",
        "use_reg": True,
        "use_prop": False,
        "reg_score_type": "chaos_layerratio",
    },
}


CONFIG_SPECS = {
    "gcn_energy": {
        "label": "GCN Energy",
        "train_spec": "gcn_sup",
        "score_type": "energy",
        "use_prop": False,
    },
    "gcn_graph_ebm": {
        "label": "GCN Graph-EBM",
        "train_spec": "gcn_graph_ebm_sup",
        "score_type": "auto",
        "use_prop": False,
    },
    "gcn_gnnsafe": {
        "label": "GCN GNNSafe",
        "train_spec": "gcn_sup",
        "score_type": "energy",
        "use_prop": True,
    },
    "gcn_energy_ft": {
        "label": "GCN Energy FT",
        "train_spec": "gcn_reg",
        "score_type": "energy",
        "use_prop": False,
    },
    "gcn_gnnsafepp": {
        "label": "GCN GNNSafe++",
        "train_spec": "gcn_reg_prop",
        "score_type": "energy",
        "use_prop": True,
    },
    "gcn_dssres_energy": {
        "label": "GCN+DSS Residual Energy",
        "train_spec": "gcn_dssres_sup",
        "score_type": "energy",
        "use_prop": False,
    },
    "gcn_dssres_gnnsafe": {
        "label": "GCN+DSS Residual Energy+Prop",
        "train_spec": "gcn_dssres_sup",
        "score_type": "energy",
        "use_prop": True,
    },
    "gcn_dssres_energy_ft": {
        "label": "GCN+DSS Residual Energy FT",
        "train_spec": "gcn_dssres_reg",
        "score_type": "energy",
        "use_prop": False,
    },
    "gcn_dssres_gnnsafepp": {
        "label": "GCN+DSS Residual GNNSafe++",
        "train_spec": "gcn_dssres_reg_prop",
        "score_type": "energy",
        "use_prop": True,
    },
    "gcn_dssres_gnnsafepp_pred_entropy": {
        "label": "GCN+DSS Residual GNNSafe++ -> PredEnt",
        "train_spec": "gcn_dssres_reg_prop",
        "score_type": "pred_entropy",
        "use_prop": False,
    },
    "gcn_dssres_gnnsafepp_pred_entropy_prop": {
        "label": "GCN+DSS Residual GNNSafe++ -> PredEnt+Prop",
        "train_spec": "gcn_dssres_reg_prop",
        "score_type": "pred_entropy",
        "use_prop": True,
    },
    "gcn_dssres_gnnsafepp_energy_mi_prop": {
        "label": "GCN+DSS Residual GNNSafe++ -> Energy-MI+Prop",
        "train_spec": "gcn_dssres_reg_prop",
        "score_type": "energy_mutual_info",
        "use_prop": True,
    },
    "gcn_dssres_pred_entropy": {
        "label": "GCN+DSS Residual PredEnt",
        "train_spec": "gcn_dssres_sup",
        "score_type": "pred_entropy",
        "use_prop": False,
    },
    "gcn_dssres_pred_entropy_prop": {
        "label": "GCN+DSS Residual PredEnt+Prop",
        "train_spec": "gcn_dssres_sup",
        "score_type": "pred_entropy",
        "use_prop": True,
    },
    "gcn_dssres_mutual_info": {
        "label": "GCN+DSS Residual MI",
        "train_spec": "gcn_dssres_sup",
        "score_type": "mutual_info",
        "use_prop": False,
    },
    "gcn_dssres_mutual_info_prop": {
        "label": "GCN+DSS Residual MI+Prop",
        "train_spec": "gcn_dssres_sup",
        "score_type": "mutual_info",
        "use_prop": True,
    },
    "gcn_dssres_energy_mi_prop": {
        "label": "GCN+DSS Residual Energy-MI+Prop",
        "train_spec": "gcn_dssres_sup",
        "score_type": "energy_mutual_info",
        "use_prop": True,
    },
    "gcn_dssres_fusion_prop": {
        "label": "GCN+DSS Residual Fusion+Prop",
        "train_spec": "gcn_dssres_sup",
        "score_type": "energy",
        "use_prop": False,
        "fusion_features": [
            "energy",
            "energy_prop",
            "chaos_ratio",
            "chaos_ratio_prop",
            "chaos_layerratio",
            "chaos_layerratio_prop",
        ],
        "fusion_ind_split": "valid",
    },
    "gcn_dssres_fusion_probunc": {
        "label": "GCN+DSS Residual Fusion+ProbUnc",
        "train_spec": "gcn_dssres_sup",
        "score_type": "energy",
        "use_prop": False,
        "fusion_features": [
            "energy_prop",
            "pred_entropy_prop",
            "mutual_info_prop",
        ],
        "fusion_ind_split": "valid",
    },
    "gcn_dssres_pred_entropy_ft": {
        "label": "GCN+DSS Residual PredEnt FT",
        "train_spec": "gcn_dssres_predent_reg",
        "score_type": "pred_entropy",
        "use_prop": False,
    },
    "gcn_dssres_pred_entropy_safe": {
        "label": "GCN+DSS Residual PredEnt Safe",
        "train_spec": "gcn_dssres_predent_reg_prop",
        "score_type": "pred_entropy",
        "use_prop": True,
    },
    "gcn_dssres_predsafe_energy_mi_prop": {
        "label": "GCN+DSS Residual PredSafe -> Energy-MI+Prop",
        "train_spec": "gcn_dssres_predent_reg_prop",
        "score_type": "energy_mutual_info",
        "use_prop": True,
    },
    "dssgnn_msp": {
        "label": "DSS-GNN MSP",
        "train_spec": "dssgnn_sup",
        "score_type": "msp",
        "use_prop": False,
    },
    "dssgnn_energy": {
        "label": "DSS-GNN Energy",
        "train_spec": "dssgnn_sup",
        "score_type": "energy",
        "use_prop": False,
    },
    "dssgnn_energy_chaos_ratio": {
        "label": "DSS-GNN Energy-ChaosRatio",
        "train_spec": "dssgnn_sup",
        "score_type": "energy_chaos_ratio",
        "use_prop": False,
    },
    "dssgnn_energy_chaos_layerratio": {
        "label": "DSS-GNN Energy-ChaosLayerRatio",
        "train_spec": "dssgnn_sup",
        "score_type": "energy_chaos_layerratio",
        "use_prop": False,
    },
    "dssgnn_energy_chaos_ratio_prop": {
        "label": "DSS-GNN Energy-ChaosRatio+Prop",
        "train_spec": "dssgnn_sup",
        "score_type": "energy_chaos_ratio",
        "use_prop": True,
    },
    "dssgnn_energy_chaos_layerratio_prop": {
        "label": "DSS-GNN Energy-ChaosLayerRatio+Prop",
        "train_spec": "dssgnn_sup",
        "score_type": "energy_chaos_layerratio",
        "use_prop": True,
    },
    "dssgnn_fusion": {
        "label": "DSS-GNN Fusion",
        "train_spec": "dssgnn_sup",
        "score_type": "energy",
        "use_prop": False,
        "fusion_features": ["energy", "chaos_ratio", "chaos_layerratio"],
        "fusion_ind_split": "valid",
    },
    "dssgnn_fusion_prop": {
        "label": "DSS-GNN Fusion+PropFeat",
        "train_spec": "dssgnn_sup",
        "score_type": "energy",
        "use_prop": False,
        "fusion_features": [
            "energy",
            "energy_prop",
            "chaos_ratio",
            "chaos_ratio_prop",
            "chaos_layerratio",
            "chaos_layerratio_prop",
        ],
        "fusion_ind_split": "valid",
    },
    "dssgnn_fusion_propcompact": {
        "label": "DSS-GNN Fusion+Prop3",
        "train_spec": "dssgnn_sup",
        "score_type": "energy",
        "use_prop": False,
        "fusion_features": [
            "energy_prop",
            "chaos_ratio_prop",
            "chaos_layerratio_prop",
        ],
        "fusion_ind_split": "train+valid",
    },
    "dssgnn_fusion_prop_tuned": {
        "label": "DSS-GNN Fusion+Prop Tuned",
        "train_spec": "dssgnn_sup_tuned",
        "score_type": "energy",
        "use_prop": False,
        "fusion_features": [
            "energy",
            "energy_prop",
            "chaos_ratio",
            "chaos_ratio_prop",
            "chaos_layerratio",
            "chaos_layerratio_prop",
        ],
        "fusion_ind_split": "valid",
    },
    "dssgnn_fusion_prop_lowlabel": {
        "label": "DSS-GNN Fusion+Prop LowLabel",
        "train_spec": "dssgnn_sup_lowlabel",
        "score_type": "energy",
        "use_prop": False,
        "fusion_features": [
            "energy",
            "energy_prop",
            "chaos_ratio",
            "chaos_ratio_prop",
            "chaos_layerratio",
            "chaos_layerratio_prop",
        ],
        "fusion_ind_split": "valid",
    },
    "dssgnn_fusion_chaosft": {
        "label": "DSS-GNN Fusion+ChaosFT",
        "train_spec": "dssgnn_chaos_ratio_reg",
        "score_type": "energy",
        "use_prop": False,
        "fusion_features": [
            "energy",
            "energy_prop",
            "chaos_ratio",
            "chaos_ratio_prop",
            "chaos_layerratio",
            "chaos_layerratio_prop",
        ],
        "fusion_ind_split": "valid",
    },
    "dssgnn_fusion_chaosft_tuned": {
        "label": "DSS-GNN Fusion+ChaosFT Tuned",
        "train_spec": "dssgnn_chaos_ratio_reg_tuned",
        "score_type": "energy",
        "use_prop": False,
        "fusion_features": [
            "energy",
            "energy_prop",
            "chaos_ratio",
            "chaos_ratio_prop",
            "chaos_layerratio",
            "chaos_layerratio_prop",
        ],
        "fusion_ind_split": "valid",
    },
    "dssgnn_fusion_chaosft_lowlabel": {
        "label": "DSS-GNN Fusion+ChaosFT LowLabel",
        "train_spec": "dssgnn_chaos_ratio_reg_lowlabel",
        "score_type": "energy",
        "use_prop": False,
        "fusion_features": [
            "energy",
            "energy_prop",
            "chaos_ratio",
            "chaos_ratio_prop",
            "chaos_layerratio",
            "chaos_layerratio_prop",
        ],
        "fusion_ind_split": "valid",
    },
    "dssgnn_fusion_chaosft_compact": {
        "label": "DSS-GNN Fusion+ChaosFT3",
        "train_spec": "dssgnn_chaos_ratio_reg",
        "score_type": "energy",
        "use_prop": False,
        "fusion_features": [
            "energy_prop",
            "chaos_ratio_prop",
            "chaos_layerratio_prop",
        ],
        "fusion_ind_split": "train+valid",
    },
    "dssgnn_gnnsafe": {
        "label": "DSS-GNN Energy+Prop",
        "train_spec": "dssgnn_sup",
        "score_type": "energy",
        "use_prop": True,
    },
    "dssgnn_gnnsafe_tuned": {
        "label": "DSS-GNN Energy+Prop Tuned",
        "train_spec": "dssgnn_sup_tuned",
        "score_type": "energy",
        "use_prop": True,
    },
    "dssgnn_gnnsafe_lowlabel": {
        "label": "DSS-GNN Energy+Prop LowLabel",
        "train_spec": "dssgnn_sup_lowlabel",
        "score_type": "energy",
        "use_prop": True,
    },
    "dssgnn_energy_ft": {
        "label": "DSS-GNN Energy FT",
        "train_spec": "dssgnn_reg",
        "score_type": "energy",
        "use_prop": False,
    },
    "dssgnn_energy_ft_tuned": {
        "label": "DSS-GNN Energy FT Tuned",
        "train_spec": "dssgnn_reg_tuned",
        "score_type": "energy",
        "use_prop": False,
    },
    "dssgnn_gnnsafepp": {
        "label": "DSS-GNN GNNSafe++",
        "train_spec": "dssgnn_reg_prop",
        "score_type": "energy",
        "use_prop": True,
    },
    "dssgnn_gnnsafepp_tuned": {
        "label": "DSS-GNN GNNSafe++ Tuned",
        "train_spec": "dssgnn_reg_prop_tuned",
        "score_type": "energy",
        "use_prop": True,
    },
    "dssgnn_chaos": {
        "label": "DSS-GNN Chaos",
        "train_spec": "dssgnn_sup",
        "score_type": "chaos",
        "use_prop": False,
    },
    "dssgnn_chaos_norm": {
        "label": "DSS-GNN ChaosNorm",
        "train_spec": "dssgnn_sup",
        "score_type": "chaos_norm",
        "use_prop": False,
    },
    "dssgnn_chaos_layernorm": {
        "label": "DSS-GNN ChaosLayer",
        "train_spec": "dssgnn_sup",
        "score_type": "chaos_layernorm",
        "use_prop": False,
    },
    "dssgnn_chaos_ratio": {
        "label": "DSS-GNN ChaosRatio",
        "train_spec": "dssgnn_sup",
        "score_type": "chaos_ratio",
        "use_prop": False,
    },
    "dssgnn_chaos_layerratio": {
        "label": "DSS-GNN ChaosLayerRatio",
        "train_spec": "dssgnn_sup",
        "score_type": "chaos_layerratio",
        "use_prop": False,
    },
    "dssgnn_chaos_prop": {
        "label": "DSS-GNN Chaos+Prop",
        "train_spec": "dssgnn_sup",
        "score_type": "chaos",
        "use_prop": True,
    },
    "dssgnn_chaos_ft": {
        "label": "DSS-GNN Chaos FT",
        "train_spec": "dssgnn_chaos_reg",
        "score_type": "chaos",
        "use_prop": False,
    },
    "dssgnn_chaos_safe": {
        "label": "DSS-GNN Chaos Safe",
        "train_spec": "dssgnn_chaos_reg_prop",
        "score_type": "chaos",
        "use_prop": True,
    },
    "dssgnn_chaos_norm_ft": {
        "label": "DSS-GNN ChaosNorm FT",
        "train_spec": "dssgnn_chaos_norm_reg",
        "score_type": "chaos_norm",
        "use_prop": False,
    },
    "dssgnn_chaos_layernorm_ft": {
        "label": "DSS-GNN ChaosLayer FT",
        "train_spec": "dssgnn_chaos_layernorm_reg",
        "score_type": "chaos_layernorm",
        "use_prop": False,
    },
    "dssgnn_chaos_ratio_ft": {
        "label": "DSS-GNN ChaosRatio FT",
        "train_spec": "dssgnn_chaos_ratio_reg",
        "score_type": "chaos_ratio",
        "use_prop": False,
    },
    "dssgnn_chaos_layerratio_ft": {
        "label": "DSS-GNN ChaosLayerRatio FT",
        "train_spec": "dssgnn_chaos_layerratio_reg",
        "score_type": "chaos_layerratio",
        "use_prop": False,
    },
}


PRESETS = {
    "gnnsafe_matrix": [
        "gcn_energy",
        "gcn_gnnsafe",
        "gcn_energy_ft",
        "gcn_gnnsafepp",
        "gcn_graph_ebm",
        "gcn_dssres_energy",
        "gcn_dssres_gnnsafe",
        "gcn_dssres_energy_ft",
        "gcn_dssres_gnnsafepp",
    ],
    "quick": [
        "gcn_gnnsafe",
        "gcn_gnnsafepp",
        "gcn_dssres_energy",
        "gcn_dssres_gnnsafe",
        "gcn_dssres_gnnsafepp",
        "gcn_dssres_pred_entropy",
        "gcn_dssres_pred_entropy_prop",
        "gcn_dssres_mutual_info",
        "gcn_dssres_mutual_info_prop",
        "gcn_dssres_energy_mi_prop",
        "gcn_dssres_fusion_prop",
        "gcn_dssres_fusion_probunc",
        "dssgnn_energy",
        "dssgnn_energy_chaos_ratio",
        "dssgnn_energy_chaos_layerratio",
        "dssgnn_energy_chaos_ratio_prop",
        "dssgnn_energy_chaos_layerratio_prop",
        "dssgnn_fusion",
        "dssgnn_fusion_prop",
        "dssgnn_fusion_propcompact",
        "dssgnn_gnnsafe",
        "dssgnn_chaos",
        "dssgnn_chaos_norm",
        "dssgnn_chaos_layernorm",
        "dssgnn_chaos_ratio",
        "dssgnn_chaos_layerratio",
        "dssgnn_chaos_prop",
        "dssgnn_chaos_ft",
        "dssgnn_chaos_safe",
        "dssgnn_chaos_norm_ft",
        "dssgnn_chaos_layernorm_ft",
        "dssgnn_chaos_ratio_ft",
        "dssgnn_chaos_layerratio_ft",
    ],
    "full": list(CONFIG_SPECS.keys()),
}


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def build_base_args() -> SimpleNamespace:
    return SimpleNamespace(
        dataset="cora",
        ood_type="structure",
        data_dir="data/",
        device=0,
        cpu=False,
        seed=123,
        train_prop=0.1,
        valid_prop=0.1,
        runs=1,
        epochs=50,
        method="msp",
        backbone="gcn",
        score_type="auto",
        hidden_channels=64,
        num_layers=2,
        dropout=0.0,
        use_bn=False,
        dss_lambda_reg=0.0,
        dss_reg_score_type="chaos",
        T=1.0,
        use_reg=False,
        lamda=1.0,
        m_in=-5,
        m_out=-1,
        reg_score_type="energy",
        chaos_margin=0.01,
        chaos_weight=1.0,
        mi_weight=1.0,
        use_prop=False,
        K=2,
        alpha=0.5,
        weight_decay=1e-2,
        lr=0.01,
        display_step=50,
        mode="detect",
        K_lp=3,
        K_hp=2,
        P=1,
        use_random_gates=False,
        P_gate=1,
        S=4,
        lambda_max=2.0,
        shared_input_lift=False,
        warmup_base_epochs=50,
        residual_scale_init=0.1,
        hop_lp=2,
        hop_hp=2,
        tfe_combine="sum",
        eta=0.5,
        gduq_n_anchors=5,
        graph_ebm_covariance_type="diagonal",
        graph_ebm_tied_covariance=False,
        graph_ebm_gamma_correction=1.0,
        graph_ebm_lambda_independent_energy=1.0,
        graph_ebm_lambda_local_energy=1.0,
        graph_ebm_lambda_group_energy=1.0,
        graph_ebm_alpha=0.5,
        graph_ebm_num_diffusion_steps=10,
        graph_ebm_aggregation="sum",
    )


def make_args(base_args: SimpleNamespace, **overrides) -> SimpleNamespace:
    args = deepcopy(base_args)
    for key, value in overrides.items():
        setattr(args, key, value)
    hp = GNNSAFE_HYPER.get((args.dataset, args.ood_type), GNNSAFE_HYPER.get((args.dataset, None), None))
    if hp is not None:
        args.m_in, args.m_out, args.lamda = hp
    return args


def instantiate_model(args: SimpleNamespace, in_dim: int, out_dim: int, device: torch.device):
    if args.method == "msp":
        model = MSP(in_dim, out_dim, args)
    elif args.method == "gnnsafe":
        model = GNNSafe(in_dim, out_dim, args)
    elif args.method == "gduq":
        model = GDUQ(in_dim, out_dim, args)
    elif args.method == "graph_ebm":
        model = GraphEBM(in_dim, out_dim, args)
    else:
        raise ValueError(f"Unknown method: {args.method}")
    return model.to(device)


def maybe_set_anchor_dist(model, args: SimpleNamespace, dataset_ind, device: torch.device) -> None:
    if args.backbone != "gduq":
        return
    train_idx = dataset_ind.splits["train"]
    train_x = dataset_ind.x[train_idx]
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0)
    std[std == 0] = 1e-3
    model.set_anchor_dist(mean.to(device), std.to(device))


def prepare_datasets(args: SimpleNamespace):
    dataset_ind, dataset_ood_tr, dataset_ood_te = load_dataset(args)

    if len(dataset_ind.y.shape) == 1:
        dataset_ind.y = dataset_ind.y.unsqueeze(1)
    if len(dataset_ood_tr.y.shape) == 1:
        dataset_ood_tr.y = dataset_ood_tr.y.unsqueeze(1)
    if isinstance(dataset_ood_te, list):
        for dataset in dataset_ood_te:
            if len(dataset.y.shape) == 1:
                dataset.y = dataset.y.unsqueeze(1)
    elif len(dataset_ood_te.y.shape) == 1:
        dataset_ood_te.y = dataset_ood_te.y.unsqueeze(1)

    if args.dataset not in ("cora", "citeseer", "pubmed"):
        dataset_ind.splits = rand_splits(
            dataset_ind.node_idx, train_prop=args.train_prop, valid_prop=args.valid_prop
        )

    return dataset_ind, dataset_ood_tr, dataset_ood_te


def select_single_ood_dataset(dataset_ood_te, ood_test_index: int):
    if isinstance(dataset_ood_te, list):
        if not dataset_ood_te:
            raise ValueError("OOD test dataset list is empty.")
        if ood_test_index < 0 or ood_test_index >= len(dataset_ood_te):
            raise ValueError(f"ood_test_index={ood_test_index} out of range for {len(dataset_ood_te)} OOD sets.")
        return dataset_ood_te[ood_test_index], f"ood{ood_test_index + 1}"
    return dataset_ood_te, "ood1"


def metric_setup(dataset_name: str):
    criterion = nn.BCEWithLogitsLoss() if dataset_name in ("proteins", "ppi") else nn.NLLLoss()
    eval_func = eval_rocauc if dataset_name in ("proteins", "ppi", "twitch") else eval_acc
    return criterion, eval_func


def clone_state_dict(model) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_best_state(model, dataset_ind, dataset_ood_tr, dataset_ood_eval, train_args, device, verbose=False):
    criterion, eval_func = metric_setup(train_args.dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_args.lr, weight_decay=train_args.weight_decay)
    best_state = clone_state_dict(model)
    best_valid_loss = float("inf")
    best_epoch = 0

    for epoch in range(train_args.epochs):
        if hasattr(model, "set_train_epoch"):
            model.set_train_epoch(epoch)
        model.train()
        optimizer.zero_grad()
        loss = model.loss_compute(dataset_ind, dataset_ood_tr, criterion, device, train_args)
        loss.backward()
        optimizer.step()

        result = evaluate_detect(model, dataset_ind, dataset_ood_eval, criterion, eval_func, train_args, device)
        valid_loss = float(result[-1].detach() if hasattr(result[-1], 'detach') else result[-1])
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = clone_state_dict(model)
            best_epoch = epoch + 1

        if verbose and (epoch == 0 or (epoch + 1) % train_args.display_step == 0 or epoch + 1 == train_args.epochs):
            auroc, aupr, fpr95, ind_acc = result[0], result[1], result[2], result[-2]
            print(
                f"[{train_args.method}/{train_args.backbone}] epoch={epoch + 1} "
                f"loss={float(loss.detach()):.4f} val={valid_loss:.4f} "
                f"auroc={100*auroc:.2f} aupr={100*aupr:.2f} fpr95={100*fpr95:.2f} ind={100*ind_acc:.2f}"
            )

    return best_state, best_valid_loss, best_epoch


def evaluate_config(model, dataset_ind, dataset_ood_eval, eval_args, device):
    criterion, eval_func = metric_setup(eval_args.dataset)
    result, ind_scores, ood_scores = evaluate_detect(
        model, dataset_ind, dataset_ood_eval, criterion, eval_func, eval_args, device, return_score=True
    )
    return {
        "auroc": float(result[0]),
        "aupr": float(result[1]),
        "fpr95": float(result[2]),
        "ind_acc": float(result[-2]),
        "valid_loss": float(result[-1].detach() if hasattr(result[-1], 'detach') else result[-1]),
        "ind_scores": ind_scores.detach().cpu().numpy() if hasattr(ind_scores, 'detach') else ind_scores.cpu().numpy(),
        "ood_scores": ood_scores.detach().cpu().numpy() if hasattr(ood_scores, 'detach') else ood_scores.cpu().numpy(),
    }


def evaluate_fusion_config(model, dataset_ind, dataset_ood_tr, dataset_ood_eval, eval_args, device, config):
    criterion, eval_func = metric_setup(eval_args.dataset)
    fusion = fit_logistic_score_fusion(
        model.encoder,
        dataset_ind,
        dataset_ood_tr,
        device,
        eval_args,
        feature_names=config["fusion_features"],
        ind_split=config.get("fusion_ind_split", "valid"),
        c=config.get("fusion_c", 1.0),
    )

    model.eval()
    with torch.no_grad():
        ind_score_all = score_with_fusion(model.encoder, dataset_ind, device, eval_args, fusion).cpu()
        ood_score_all = score_with_fusion(model.encoder, dataset_ood_eval, device, eval_args, fusion).cpu()
        test_ind_score = ind_score_all[dataset_ind.splits["test"]]
        test_ood_score = ood_score_all[dataset_ood_eval.node_idx]
        auroc, aupr, fpr95, _ = get_measures(test_ind_score, test_ood_score)

        out = model(dataset_ind, device).cpu()
        test_idx = dataset_ind.splits["test"]
        valid_idx = dataset_ind.splits["valid"]
        ind_acc = eval_func(dataset_ind.y[test_idx], out[test_idx])
        if eval_args.dataset in ("proteins", "ppi"):
            valid_loss = criterion(out[valid_idx], dataset_ind.y[valid_idx].to(torch.float))
        else:
            valid_out = torch.nn.functional.log_softmax(out[valid_idx], dim=1)
            valid_loss = criterion(valid_out, dataset_ind.y[valid_idx].squeeze(1))

    return {
        "auroc": float(auroc),
        "aupr": float(aupr),
        "fpr95": float(fpr95),
        "ind_acc": float(ind_acc),
        "valid_loss": float(valid_loss.detach()),
        "ind_scores": test_ind_score.detach().cpu().numpy(),
        "ood_scores": test_ood_score.detach().cpu().numpy(),
        "fusion": fusion.to_dict(),
    }


def restore_model_phase(model, best_epoch):
    if hasattr(model, "set_train_epoch"):
        model.set_train_epoch(max(int(best_epoch) - 1, 0))


def score_limits(ind_scores, ood_scores):
    all_scores = np.concatenate([np.asarray(ind_scores, dtype=float), np.asarray(ood_scores, dtype=float)])
    score_min = float(all_scores.min())
    score_max = float(all_scores.max())
    padding = 0.05 * max(score_max - score_min, 1e-6)
    return score_min - padding, score_max + padding


def plot_density(ax, ind_scores, ood_scores, title, x_limits=None):
    ind_scores = np.asarray(ind_scores, dtype=float)
    ood_scores = np.asarray(ood_scores, dtype=float)
    x_min, x_max = x_limits if x_limits is not None else score_limits(ind_scores, ood_scores)
    xs = np.linspace(x_min, x_max, 256)

    def _draw(values, color, label):
        unique = np.unique(values)
        if values.size >= 3 and unique.size >= 2:
            try:
                kde = gaussian_kde(values)
                ys = kde(xs)
                ax.plot(xs, ys, color=color, linewidth=2, label=label)
                ax.fill_between(xs, ys, color=color, alpha=0.20)
                return
            except np.linalg.LinAlgError:
                pass
        ax.hist(values, bins=25, density=True, histtype="step", linewidth=2, color=color, label=label)

    _draw(ind_scores, "#1f77b4", "IND")
    _draw(ood_scores, "#d62728", "OOD")
    ax.axvline(ind_scores.mean(), color="#1f77b4", linestyle="--", linewidth=1)
    ax.axvline(ood_scores.mean(), color="#d62728", linestyle="--", linewidth=1)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(x_min, x_max)
    ax.grid(alpha=0.2)


def save_density_grid(path: Path, config_names, summary_rows, all_scores, fig_title, shared_x_limits=None):
    n_configs = len(config_names)
    n_cols = 3
    n_rows = math.ceil(n_configs / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.8 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, config_name in zip(axes_flat, config_names):
        config = CONFIG_SPECS[config_name]
        summary = next(row for row in summary_rows if row["config"] == config_name)
        ind_scores = np.concatenate(all_scores[config_name]["ind"], axis=0)
        ood_scores = np.concatenate(all_scores[config_name]["ood"], axis=0)
        title = (
            f"{config['label']}\n"
            f"AUROC {100*summary['auroc_mean']:.1f} +/- {100*summary['auroc_std']:.1f} | "
            f"IND Acc {100*summary['ind_acc_mean']:.1f}"
        )
        plot_density(ax, ind_scores, ood_scores, title, shared_x_limits)

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
        "train_spec",
        "runs",
        "auroc_mean",
        "auroc_std",
        "aupr_mean",
        "aupr_std",
        "fpr95_mean",
        "fpr95_std",
        "ind_acc_mean",
        "ind_acc_std",
        "valid_loss_mean",
        "valid_loss_std",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_scores_csv(path: Path, long_rows):
    fieldnames = ["config", "label", "split", "score"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in long_rows:
            writer.writerow(row)


def summarize(values):
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std())


def main():
    parser = argparse.ArgumentParser(description="Local OOD comparison with density plots")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--ood_type", type=str, default="structure", choices=["structure", "feature", "label"])
    parser.add_argument("--ood_test_index", type=int, default=0, help="For multi-OOD datasets (e.g. arxiv/twitch), choose which OOD split to visualize.")
    parser.add_argument("--preset", type=str, default="full", choices=sorted(PRESETS.keys()))
    parser.add_argument("--configs", nargs="*", help="Optional explicit config list; overrides --preset.")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--dss_lambda_reg", type=float, default=0.0)
    parser.add_argument(
        "--dss_reg_score_type",
        type=str,
        default="chaos",
        choices=["chaos", "chaos_norm", "chaos_layernorm", "chaos_ratio", "chaos_layerratio"],
    )
    parser.add_argument("--K_lp", type=int, default=3)
    parser.add_argument("--K_hp", type=int, default=2)
    parser.add_argument("--P", type=int, default=1)
    parser.add_argument("--use_random_gates", action="store_true")
    parser.add_argument("--P_gate", type=int, default=1)
    parser.add_argument("--S", type=int, default=4)
    parser.add_argument("--lambda_max", type=float, default=2.0)
    parser.add_argument("--shared_input_lift", action="store_true")
    parser.add_argument("--warmup_base_epochs", type=int, default=50)
    parser.add_argument("--residual_scale_init", type=float, default=0.1)
    parser.add_argument("--K", type=int, default=2, help="Propagation steps for score smoothing.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Residual weight for score propagation.")
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--lamda", type=float, default=1.0, help="GNNSafe OOD regularization weight.")
    parser.add_argument("--m_in", type=float, default=-5.0, help="GNNSafe in-distribution energy margin.")
    parser.add_argument("--m_out", type=float, default=-1.0, help="GNNSafe OOD energy margin.")
    parser.add_argument("--chaos_margin", type=float, default=0.01, help="Margin used by chaos-based OOD regularization.")
    parser.add_argument("--chaos_weight", type=float, default=1.0, help="Weight for hybrid energy-minus-chaos scores.")
    parser.add_argument("--mi_weight", type=float, default=1.0, help="Weight for hybrid energy-minus-mutual-information scores.")
    parser.add_argument("--display_step", type=int, default=25)
    parser.add_argument("--tag", type=str, default=None, help="Optional output tag.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--use_bn", action="store_true", default=True,
                        help="Use batch norm (default: True, matching GNNSafe paper)")
    parser.add_argument("--no_bn", action="store_true", help="Disable batch norm")
    args = parser.parse_args()
    if args.no_bn:
        args.use_bn = False

    config_names = args.configs if args.configs else PRESETS[args.preset]
    unknown = [name for name in config_names if name not in CONFIG_SPECS]
    if unknown:
        raise ValueError(f"Unknown configs: {unknown}")

    base_args = build_base_args()
    for key in (
        "dataset",
        "ood_type",
        "data_dir",
        "device",
        "cpu",
        "seed",
        "runs",
        "epochs",
        "hidden_channels",
        "num_layers",
        "dropout",
        "lr",
        "weight_decay",
        "dss_lambda_reg",
        "dss_reg_score_type",
        "K_lp",
        "K_hp",
        "P",
        "use_random_gates",
        "P_gate",
        "S",
        "lambda_max",
        "shared_input_lift",
        "warmup_base_epochs",
        "residual_scale_init",
        "K",
        "alpha",
        "T",
        "lamda",
        "m_in",
        "m_out",
        "chaos_margin",
        "chaos_weight",
        "mi_weight",
        "display_step",
    ):
        setattr(base_args, key, getattr(args, key))
    base_args.use_bn = args.use_bn

    device = torch.device("cpu" if args.cpu else f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    default_tag = args.preset if not args.configs else "custom"
    tag = args.tag or f"{default_tag}_r{args.runs}_e{args.epochs}"
    out_dir = Path("results_local") / "ood_density" / f"{args.dataset}-{args.ood_type}-{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {name: [] for name in config_names}
    all_scores = {name: {"ind": [], "ood": []} for name in config_names}

    for run_idx in range(args.runs):
        run_seed = args.seed + run_idx
        fix_seed(run_seed)
        run_args = make_args(base_args, seed=run_seed)
        dataset_ind, dataset_ood_tr, dataset_ood_te = prepare_datasets(run_args)
        dataset_ood_eval, ood_label = select_single_ood_dataset(dataset_ood_te, args.ood_test_index)

        num_classes = max(dataset_ind.y.max().item() + 1, dataset_ind.y.shape[1])
        in_dim = dataset_ind.x.shape[1]

        trained_cache = {}
        for config_name in config_names:
            config = CONFIG_SPECS[config_name]
            train_key = config["train_spec"]
            if train_key in trained_cache:
                continue

            train_spec = TRAIN_SPECS[train_key]
            train_overrides = dict(train_spec.get("arg_overrides", {}))
            train_args = make_args(
                run_args,
                method=train_spec["method"],
                backbone=train_spec["backbone"],
                use_reg=train_spec["use_reg"],
                use_prop=train_spec["use_prop"],
                reg_score_type=train_spec.get("reg_score_type", "energy"),
                score_type="auto",
                **train_overrides,
            )
            model = instantiate_model(train_args, in_dim, num_classes, device)
            maybe_set_anchor_dist(model, train_args, dataset_ind, device)
            best_state, best_valid_loss, best_epoch = train_best_state(
                model,
                dataset_ind,
                dataset_ood_tr,
                dataset_ood_eval,
                train_args,
                device,
                verbose=args.verbose,
            )
            trained_cache[train_key] = {
                "state": best_state,
                "best_valid_loss": best_valid_loss,
                "best_epoch": best_epoch,
            }

        for config_name in config_names:
            config = CONFIG_SPECS[config_name]
            train_key = config["train_spec"]
            train_spec = TRAIN_SPECS[train_key]
            train_overrides = dict(train_spec.get("arg_overrides", {}))
            eval_args = make_args(
                run_args,
                method=train_spec["method"],
                backbone=train_spec["backbone"],
                use_reg=train_spec["use_reg"],
                use_prop=config["use_prop"],
                reg_score_type=train_spec.get("reg_score_type", "energy"),
                score_type=config["score_type"],
                **train_overrides,
            )
            model = instantiate_model(eval_args, in_dim, num_classes, device)
            maybe_set_anchor_dist(model, eval_args, dataset_ind, device)
            model.load_state_dict(trained_cache[train_key]["state"])
            restore_model_phase(model, trained_cache[train_key]["best_epoch"])
            model.to(device)
            model.eval()

            if "fusion_features" in config:
                metrics = evaluate_fusion_config(model, dataset_ind, dataset_ood_tr, dataset_ood_eval, eval_args, device, config)
            else:
                metrics = evaluate_config(model, dataset_ind, dataset_ood_eval, eval_args, device)
            metrics["best_epoch"] = trained_cache[train_key]["best_epoch"]
            metrics["train_spec"] = train_key
            all_metrics[config_name].append(metrics)
            all_scores[config_name]["ind"].append(metrics["ind_scores"])
            all_scores[config_name]["ood"].append(metrics["ood_scores"])

            print(
                f"[run {run_idx + 1}/{args.runs}] {config_name:<18} "
                f"epoch={metrics['best_epoch']:<3d} "
                f"AUROC={100*metrics['auroc']:.2f} "
                f"AUPR={100*metrics['aupr']:.2f} "
                f"FPR95={100*metrics['fpr95']:.2f} "
                f"IND={100*metrics['ind_acc']:.2f}"
            )

    summary_rows = []
    long_score_rows = []
    global_scores = []
    for config_name in config_names:
        config = CONFIG_SPECS[config_name]
        label = config["label"]
        metrics = all_metrics[config_name]
        auroc_mean, auroc_std = summarize([m["auroc"] for m in metrics])
        aupr_mean, aupr_std = summarize([m["aupr"] for m in metrics])
        fpr_mean, fpr_std = summarize([m["fpr95"] for m in metrics])
        ind_mean, ind_std = summarize([m["ind_acc"] for m in metrics])
        valid_mean, valid_std = summarize([m["valid_loss"] for m in metrics])
        summary_rows.append(
            {
                "config": config_name,
                "label": label,
                "train_spec": config["train_spec"],
                "runs": args.runs,
                "auroc_mean": auroc_mean,
                "auroc_std": auroc_std,
                "aupr_mean": aupr_mean,
                "aupr_std": aupr_std,
                "fpr95_mean": fpr_mean,
                "fpr95_std": fpr_std,
                "ind_acc_mean": ind_mean,
                "ind_acc_std": ind_std,
                "valid_loss_mean": valid_mean,
                "valid_loss_std": valid_std,
            }
        )

        ind_scores = np.concatenate(all_scores[config_name]["ind"], axis=0)
        ood_scores = np.concatenate(all_scores[config_name]["ood"], axis=0)
        global_scores.extend([ind_scores, ood_scores])
        for value in ind_scores:
            long_score_rows.append({"config": config_name, "label": label, "split": "IND", "score": float(value)})
        for value in ood_scores:
            long_score_rows.append({"config": config_name, "label": label, "split": "OOD", "score": float(value)})

    write_summary_csv(out_dir / "summary.csv", summary_rows)
    write_scores_csv(out_dir / "scores_long.csv", long_score_rows)

    score_min = min(float(arr.min()) for arr in global_scores)
    score_max = max(float(arr.max()) for arr in global_scores)
    padding = 0.05 * max(score_max - score_min, 1e-6)
    shared_x_limits = (score_min - padding, score_max + padding)
    fig_title = f"Local OOD Score Density Comparison: {args.dataset} / {args.ood_type} / {ood_label}"
    save_density_grid(out_dir / "density_grid.png", config_names, summary_rows, all_scores, fig_title)
    save_density_grid(
        out_dir / "density_grid_shared_x.png",
        config_names,
        summary_rows,
        all_scores,
        f"{fig_title} (shared x-scale)",
        shared_x_limits,
    )

    metadata = {
        "dataset": args.dataset,
        "ood_type": args.ood_type,
        "ood_test_index": args.ood_test_index,
        "selected_ood_label": ood_label,
        "preset": args.preset,
        "configs": config_names,
        "runs": args.runs,
        "epochs": args.epochs,
        "device": str(device),
        "seed": args.seed,
        "fusion_configs": {
            name: [m["fusion"] for m in all_metrics[name] if "fusion" in m]
            for name in config_names
            if any("fusion" in m for m in all_metrics[name])
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"\nSaved results to {out_dir}")
    print(f"- {out_dir / 'summary.csv'}")
    print(f"- {out_dir / 'scores_long.csv'}")
    print(f"- {out_dir / 'density_grid.png'}")
    print(f"- {out_dir / 'density_grid_shared_x.png'}")


if __name__ == "__main__":
    main()
