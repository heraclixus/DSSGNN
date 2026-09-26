#!/usr/bin/env python3
"""
Server-side DSS hybrid OOD sweep launcher for GNNSafe datasets.

This script defines dataset-aware search grids around the best-performing DSS
hybrid family found locally, and can either:

1. list/count the sweep jobs, or
2. execute a single job by index (for Slurm array use).

The underlying experiment runner is `scripts/local_ood_density_compare.py`,
invoked with a single config at a time so each Slurm array element is
independent and easy to resubmit.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE = "search"
DEFAULT_TAG_PREFIX = {
    "search": "serversearch",
    "rescue": "serverrescue",
    "phase2": "serverphase2",
    "phase3": "serverphase3",
    "phase3b": "serverphase3b",
}


GNNSAFEPP_FILTER_BUNDLES = [
    {"K_lp": 4, "K_hp": 4},
    {"K_lp": 3, "K_hp": 2},
]

GNNSAFEPP_REG_BUNDLES = [
    {
        "dropout": 0.5,
        "weight_decay": 5e-4,
        "warmup_base_epochs": 50,
        "residual_scale_init": 0.1,
        "dss_lambda_reg": 0.01,
    },
    {
        "dropout": 0.3,
        "weight_decay": 1e-3,
        "warmup_base_epochs": 25,
        "residual_scale_init": 0.05,
        "dss_lambda_reg": 0.005,
    },
]

CORA_LOWLABEL_BUNDLES = [
    {
        "dropout": 0.5,
        "weight_decay": 5e-4,
        "warmup_base_epochs": 50,
        "residual_scale_init": 0.1,
        "dss_lambda_reg": 0.01,
        "shared_input_lift": False,
    },
    {
        "dropout": 0.5,
        "weight_decay": 5e-4,
        "warmup_base_epochs": 25,
        "residual_scale_init": 0.05,
        "dss_lambda_reg": 0.01,
        "shared_input_lift": True,
    },
]

FUSION_BACKBONE_BUNDLES = [
    {
        "K_lp": 4,
        "K_hp": 4,
        "dropout": 0.5,
        "weight_decay": 5e-4,
        "warmup_base_epochs": 50,
        "residual_scale_init": 0.1,
        "dss_lambda_reg": 0.01,
    },
    {
        "K_lp": 3,
        "K_hp": 3,
        "dropout": 0.3,
        "weight_decay": 1e-3,
        "warmup_base_epochs": 25,
        "residual_scale_init": 0.05,
        "dss_lambda_reg": 0.0,
    },
]

FUSION_PROP_BUNDLES = [
    {"K": 2, "alpha": 0.5},
    {"K": 4, "alpha": 0.7},
]


TARGET_SPECS = {
    "cora_structure": {
        "dataset": "cora",
        "ood_type": "structure",
        "config": "gcn_dssres_gnnsafepp",
        "family": "gnnsafepp",
        "note": "Local winner on cora structure was the energy-trained hybrid.",
        "axes": [
            ("lr", [0.01, 0.005]),
            ("P", [1, 2]),
            ("filters", GNNSAFEPP_FILTER_BUNDLES),
            ("lowlabel", CORA_LOWLABEL_BUNDLES),
        ],
    },
    "cora_feature": {
        "dataset": "cora",
        "ood_type": "feature",
        "config": "gcn_dssres_gnnsafepp",
        "family": "gnnsafepp",
        "note": "Local cora feature results still favored the energy-trained hybrid backbone.",
        "axes": [
            ("lr", [0.01, 0.005]),
            ("P", [1, 2]),
            ("filters", GNNSAFEPP_FILTER_BUNDLES),
            ("lowlabel", CORA_LOWLABEL_BUNDLES),
        ],
    },
    "cora_label": {
        "dataset": "cora",
        "ood_type": "label",
        "config": "gcn_dssres_gnnsafepp",
        "family": "gnnsafepp",
        "note": "Local cora label OOD also favored the energy-trained hybrid backbone.",
        "axes": [
            ("lr", [0.01, 0.005]),
            ("P", [1, 2]),
            ("filters", GNNSAFEPP_FILTER_BUNDLES),
            ("lowlabel", CORA_LOWLABEL_BUNDLES),
        ],
    },
    "amazon_photo_structure": {
        "dataset": "amazon-photo",
        "ood_type": "structure",
        "config": "gcn_dssres_fusion_probunc",
        "family": "fusion_probunc",
        "note": "Local amazon-photo structure was saturated by the probabilistic fusion hybrid.",
        "axes": [
            ("lr", [0.01, 0.005]),
            ("P", [1, 2]),
            ("backbone", FUSION_BACKBONE_BUNDLES),
            ("prop", FUSION_PROP_BUNDLES),
        ],
    },
    "amazon_photo_feature": {
        "dataset": "amazon-photo",
        "ood_type": "feature",
        "config": "gcn_dssres_fusion_probunc",
        "family": "fusion_probunc",
        "note": "Local amazon-photo feature favored fusion strongly over raw GNNSafe++.",
        "axes": [
            ("lr", [0.01, 0.005]),
            ("P", [1, 2]),
            ("backbone", FUSION_BACKBONE_BUNDLES),
            ("prop", FUSION_PROP_BUNDLES),
        ],
    },
    "amazon_photo_label": {
        "dataset": "amazon-photo",
        "ood_type": "label",
        "config": "gcn_dssres_fusion_probunc",
        "family": "fusion_probunc",
        "note": "Local amazon-photo label shift also favored the fusion detector.",
        "axes": [
            ("lr", [0.01, 0.005]),
            ("P", [1, 2]),
            ("backbone", FUSION_BACKBONE_BUNDLES),
            ("prop", FUSION_PROP_BUNDLES),
        ],
    },
    "coauthor_cs_structure": {
        "dataset": "coauthor-cs",
        "ood_type": "structure",
        "config": "gcn_dssres_gnnsafepp",
        "family": "gnnsafepp",
        "note": "Local coauthor-cs structure showed a very strong GNNSafe++ hybrid win.",
        "axes": [
            ("lr", [0.01, 0.005]),
            ("P", [1, 2]),
            ("filters", GNNSAFEPP_FILTER_BUNDLES),
            ("regularization", GNNSAFEPP_REG_BUNDLES),
        ],
    },
    "coauthor_cs_feature": {
        "dataset": "coauthor-cs",
        "ood_type": "feature",
        "config": "gcn_dssres_gnnsafepp",
        "family": "gnnsafepp",
        "note": "Local coauthor-cs feature OOD was already nearly saturated by GNNSafe++ hybrid training.",
        "axes": [
            ("lr", [0.01, 0.005]),
            ("P", [1, 2]),
            ("filters", GNNSAFEPP_FILTER_BUNDLES),
            ("regularization", GNNSAFEPP_REG_BUNDLES),
        ],
    },
    "coauthor_cs_label": {
        "dataset": "coauthor-cs",
        "ood_type": "label",
        "config": "gcn_dssres_gnnsafepp",
        "family": "gnnsafepp",
        "note": "Local coauthor-cs label shift also favored the energy-trained hybrid.",
        "axes": [
            ("lr", [0.01, 0.005]),
            ("P", [1, 2]),
            ("filters", GNNSAFEPP_FILTER_BUNDLES),
            ("regularization", GNNSAFEPP_REG_BUNDLES),
        ],
    },
    "twitch": {
        "dataset": "twitch",
        "ood_type": "structure",
        "config": "gcn_dssres_fusion_probunc",
        "family": "fusion_probunc",
        "note": "Local twitch runs strongly favored the probabilistic fusion detector.",
        "axes": [
            ("lr", [0.01, 0.005]),
            ("P", [1, 2]),
            ("backbone", FUSION_BACKBONE_BUNDLES),
            ("prop", FUSION_PROP_BUNDLES),
        ],
    },
    "arxiv": {
        "dataset": "arxiv",
        "ood_type": "structure",
        "config": "gcn_dssres_gnnsafepp",
        "family": "gnnsafepp",
        "note": "Local arxiv results modestly favored the energy-trained hybrid backbone.",
        "axes": [
            ("lr", [0.01, 0.005]),
            ("P", [1, 2]),
            ("filters", GNNSAFEPP_FILTER_BUNDLES),
            ("regularization", GNNSAFEPP_REG_BUNDLES),
        ],
    },
}


CORA_LABEL_RESCUE_VARIANTS = [
    {
        "config_name": "gcn_dssres_gnnsafepp",
        "family": "label_gnnsafepp_k0",
        "note": "Cora label OOD with propagation disabled to avoid oversmoothing on the same graph.",
        "K": 0,
        "alpha": 0.5,
    },
    {
        "config_name": "gcn_dssres_gnnsafepp",
        "family": "label_gnnsafepp_k1",
        "note": "Cora label OOD with one propagation step as a lighter GNNSafe++ variant.",
        "K": 1,
        "alpha": 0.7,
    },
    {
        "config_name": "gcn_dssres_gnnsafepp_pred_entropy",
        "family": "label_predent",
        "note": "Energy-trained hybrid scored by predictive entropy without extra propagation.",
    },
    {
        "config_name": "gcn_dssres_fusion_probunc",
        "family": "label_fusion_k0",
        "note": "Fusion detector on unpropagated predictive features for same-graph label shift.",
        "K": 0,
        "alpha": 0.5,
    },
    {
        "config_name": "gcn_dssres_fusion_probunc",
        "family": "label_fusion_k1",
        "note": "Fusion detector with one propagation step for a milder smoothing baseline.",
        "K": 1,
        "alpha": 0.7,
    },
]

CORA_LABEL_RESCUE_BUNDLES = [
    {
        "lr": 0.01,
        "dropout": 0.5,
        "weight_decay": 5e-4,
        "warmup_base_epochs": 50,
        "residual_scale_init": 0.1,
        "dss_lambda_reg": 0.01,
        "shared_input_lift": False,
        "lamda": 1.0,
        "m_in": -5,
        "m_out": -1,
        "epochs": 400,
    },
    {
        "lr": 0.005,
        "dropout": 0.5,
        "weight_decay": 5e-4,
        "warmup_base_epochs": 100,
        "residual_scale_init": 0.05,
        "dss_lambda_reg": 0.005,
        "shared_input_lift": False,
        "lamda": 0.5,
        "m_in": -3,
        "m_out": 0,
        "epochs": 400,
    },
]

TWITCH_RESCUE_VARIANTS = [
    {
        "config_name": "gcn_dssres_fusion_probunc",
        "family": "fusion_probunc",
        "note": "Primary twitch candidate: fusion over propagated predictive uncertainty features.",
    },
    {
        "config_name": "gcn_dssres_gnnsafepp",
        "family": "gnnsafepp",
        "note": "Energy-trained hybrid baseline on twitch.",
    },
    {
        "config_name": "gcn_dssres_gnnsafepp_pred_entropy_prop",
        "family": "predent_prop",
        "note": "Energy-trained hybrid scored by propagated predictive entropy.",
    },
    {
        "config_name": "gcn_dssres_gnnsafepp_energy_mi_prop",
        "family": "energy_mi_prop",
        "note": "Energy-mutual-information hybrid score on top of the GNNSafe++-trained backbone.",
    },
]

TWITCH_RESCUE_BACKBONE_BUNDLES = [
    {
        "K_lp": 4,
        "K_hp": 4,
        "dropout": 0.5,
        "weight_decay": 5e-4,
        "warmup_base_epochs": 50,
        "residual_scale_init": 0.1,
        "dss_lambda_reg": 0.01,
        "lr": 0.01,
        "lamda": 1.0,
        "m_in": -5,
        "m_out": -1,
        "epochs": 300,
    },
    {
        "K_lp": 5,
        "K_hp": 5,
        "dropout": 0.3,
        "weight_decay": 1e-3,
        "warmup_base_epochs": 75,
        "residual_scale_init": 0.05,
        "dss_lambda_reg": 0.005,
        "lr": 0.005,
        "lamda": 0.5,
        "m_in": -3,
        "m_out": 0,
        "epochs": 300,
    },
]

TWITCH_RESCUE_STOCHASTIC_BUNDLES = [
    {
        "P": 2,
        "use_random_gates": False,
        "S": 4,
    },
    {
        "P": 3,
        "use_random_gates": True,
        "P_gate": 1,
        "S": 8,
    },
]

TWITCH_RESCUE_PROP_BUNDLES = [
    {"K": 4, "alpha": 0.7},
    {"K": 6, "alpha": 0.8},
]

ARXIV_RESCUE_VARIANTS = [
    {
        "config_name": "gcn_dssres_gnnsafepp",
        "family": "gnnsafepp",
        "note": "Energy-trained hybrid baseline on arxiv.",
    },
    {
        "config_name": "gcn_dssres_gnnsafepp_pred_entropy_prop",
        "family": "predent_prop",
        "note": "Predictive-entropy scoring on the GNNSafe++-trained hybrid backbone.",
    },
    {
        "config_name": "gcn_dssres_gnnsafepp_energy_mi_prop",
        "family": "energy_mi_prop",
        "note": "Energy-mutual-information scoring on the GNNSafe++-trained hybrid backbone.",
    },
    {
        "config_name": "gcn_dssres_fusion_probunc",
        "family": "fusion_probunc",
        "note": "Probabilistic fusion detector on top of the supervised hybrid backbone.",
    },
]

ARXIV_RESCUE_CAPACITY_BUNDLES = [
    {
        "hidden_channels": 64,
        "num_layers": 2,
        "P": 1,
        "K_lp": 4,
        "K_hp": 4,
        "use_random_gates": False,
        "S": 4,
    },
    {
        "hidden_channels": 128,
        "num_layers": 2,
        "P": 2,
        "K_lp": 4,
        "K_hp": 4,
        "use_random_gates": False,
        "S": 4,
    },
    {
        "hidden_channels": 128,
        "num_layers": 3,
        "P": 3,
        "K_lp": 5,
        "K_hp": 4,
        "use_random_gates": True,
        "P_gate": 1,
        "S": 8,
    },
]

ARXIV_RESCUE_TRAIN_BUNDLES = [
    {
        "lr": 0.01,
        "dropout": 0.5,
        "weight_decay": 5e-4,
        "warmup_base_epochs": 50,
        "residual_scale_init": 0.1,
        "dss_lambda_reg": 0.01,
        "lamda": 1.0,
        "m_in": -5,
        "m_out": -1,
        "epochs": 350,
    },
    {
        "lr": 0.005,
        "dropout": 0.3,
        "weight_decay": 1e-3,
        "warmup_base_epochs": 100,
        "residual_scale_init": 0.05,
        "dss_lambda_reg": 0.005,
        "lamda": 0.5,
        "m_in": -3,
        "m_out": 0,
        "epochs": 350,
    },
]

ARXIV_RESCUE_PROP_BUNDLES = [
    {"K": 2, "alpha": 0.5},
    {"K": 4, "alpha": 0.7},
]

RESCUE_TARGET_SPECS = {
    "cora_label": {
        "dataset": "cora",
        "ood_type": "label",
        "note": "Focused rescue sweep for same-graph label OOD on cora.",
        "axes": [
            ("variant", CORA_LABEL_RESCUE_VARIANTS),
            ("P", [1, 2]),
            ("training", CORA_LABEL_RESCUE_BUNDLES),
        ],
    },
    "twitch": {
        "dataset": "twitch",
        "ood_type": "structure",
        "note": "Focused rescue sweep for twitch with richer stochasticity and stronger propagation.",
        "axes": [
            ("variant", TWITCH_RESCUE_VARIANTS),
            ("backbone", TWITCH_RESCUE_BACKBONE_BUNDLES),
            ("stochasticity", TWITCH_RESCUE_STOCHASTIC_BUNDLES),
            ("prop", TWITCH_RESCUE_PROP_BUNDLES),
        ],
    },
    "arxiv": {
        "dataset": "arxiv",
        "ood_type": "structure",
        "note": "Focused rescue sweep for arxiv with longer training and higher-capacity DSS hybrids.",
        "axes": [
            ("variant", ARXIV_RESCUE_VARIANTS),
            ("capacity", ARXIV_RESCUE_CAPACITY_BUNDLES),
            ("training", ARXIV_RESCUE_TRAIN_BUNDLES),
            ("prop", ARXIV_RESCUE_PROP_BUNDLES),
        ],
    },
}


CORA_LABEL_PHASE2_VARIANTS = [
    {
        "config_name": "gcn_dssres_fusion_probunc",
        "family": "fusion_probunc",
        "note": "Phase 2 cora-label anchor: fusion over low-propagation predictive uncertainty features.",
    },
    {
        "config_name": "gcn_dssres_gnnsafepp_pred_entropy_prop",
        "family": "predent_prop",
        "note": "Phase 2 cora-label hedge: predictive-entropy scoring on the GNNSafe++ backbone.",
    },
]

CORA_LABEL_PHASE2_TRAIN_BUNDLES = [
    {
        "lr": 0.005,
        "dropout": 0.5,
        "weight_decay": 5e-4,
        "warmup_base_epochs": 100,
        "residual_scale_init": 0.05,
        "dss_lambda_reg": 0.005,
        "shared_input_lift": False,
        "lamda": 0.5,
        "m_in": -3,
        "m_out": 0,
        "epochs": 500,
    },
    {
        "lr": 0.003,
        "dropout": 0.5,
        "weight_decay": 5e-4,
        "warmup_base_epochs": 150,
        "residual_scale_init": 0.03,
        "dss_lambda_reg": 0.01,
        "shared_input_lift": True,
        "lamda": 0.75,
        "m_in": -4,
        "m_out": -0.5,
        "epochs": 600,
    },
]

CORA_LABEL_PHASE2_STOCHASTIC_BUNDLES = [
    {
        "P": 2,
        "use_random_gates": False,
        "S": 4,
    },
    {
        "P": 3,
        "use_random_gates": False,
        "S": 6,
    },
]

CORA_LABEL_PHASE2_PROP_BUNDLES = [
    {"K": 0, "alpha": 0.5},
    {"K": 1, "alpha": 0.7},
    {"K": 2, "alpha": 0.5},
]

TWITCH_PHASE2_VARIANTS = [
    {
        "config_name": "gcn_dssres_fusion_probunc",
        "family": "fusion_probunc",
        "note": "Phase 2 twitch anchor: the local winner family based on probabilistic fusion.",
    },
    {
        "config_name": "gcn_dssres_gnnsafepp_energy_mi_prop",
        "family": "energy_mi_prop",
        "note": "Phase 2 twitch hedge: AUROC-oriented energy-plus-MI scoring on the GNNSafe++ backbone.",
    },
]

TWITCH_PHASE2_BACKBONE_BUNDLES = [
    {
        "K_lp": 4,
        "K_hp": 4,
        "dropout": 0.5,
        "weight_decay": 5e-4,
        "warmup_base_epochs": 50,
        "residual_scale_init": 0.1,
        "dss_lambda_reg": 0.01,
        "lr": 0.01,
        "lamda": 1.0,
        "m_in": -5,
        "m_out": -1,
        "epochs": 300,
    },
    {
        "K_lp": 5,
        "K_hp": 4,
        "dropout": 0.4,
        "weight_decay": 5e-4,
        "warmup_base_epochs": 75,
        "residual_scale_init": 0.08,
        "dss_lambda_reg": 0.01,
        "lr": 0.005,
        "lamda": 0.75,
        "m_in": -4,
        "m_out": -0.5,
        "epochs": 400,
    },
]

TWITCH_PHASE2_STOCHASTIC_BUNDLES = [
    {
        "P": 1,
        "use_random_gates": False,
        "S": 4,
    },
    {
        "P": 2,
        "use_random_gates": False,
        "S": 6,
    },
    {
        "P": 3,
        "use_random_gates": True,
        "P_gate": 1,
        "S": 8,
    },
]

TWITCH_PHASE2_PROP_BUNDLES = [
    {"K": 4, "alpha": 0.7},
    {"K": 6, "alpha": 0.8},
]

ARXIV_PHASE2_VARIANTS = [
    {
        "config_name": "gcn_dssres_gnnsafepp_energy_mi_prop",
        "family": "energy_mi_prop",
        "note": "Phase 2 arxiv anchor: the strongest current AUROC family from the server sweep.",
    },
    {
        "config_name": "gcn_dssres_gnnsafepp_pred_entropy_prop",
        "family": "predent_prop",
        "note": "Phase 2 arxiv hedge: predictive-entropy scoring on the GNNSafe++ hybrid backbone.",
    },
]

ARXIV_PHASE2_CAPACITY_BUNDLES = [
    {
        "hidden_channels": 128,
        "num_layers": 3,
        "P": 3,
        "K_lp": 5,
        "K_hp": 4,
        "use_random_gates": True,
        "P_gate": 1,
        "S": 8,
    },
    {
        "hidden_channels": 128,
        "num_layers": 2,
        "P": 3,
        "K_lp": 5,
        "K_hp": 4,
        "use_random_gates": True,
        "P_gate": 1,
        "S": 8,
    },
    {
        "hidden_channels": 96,
        "num_layers": 3,
        "P": 2,
        "K_lp": 4,
        "K_hp": 4,
        "use_random_gates": True,
        "P_gate": 1,
        "S": 6,
    },
]

ARXIV_PHASE2_TRAIN_BUNDLES = [
    {
        "lr": 0.01,
        "dropout": 0.5,
        "weight_decay": 5e-4,
        "warmup_base_epochs": 50,
        "residual_scale_init": 0.1,
        "dss_lambda_reg": 0.01,
        "lamda": 1.0,
        "m_in": -5,
        "m_out": -1,
        "epochs": 450,
    },
    {
        "lr": 0.005,
        "dropout": 0.4,
        "weight_decay": 5e-4,
        "warmup_base_epochs": 100,
        "residual_scale_init": 0.05,
        "dss_lambda_reg": 0.01,
        "lamda": 0.75,
        "m_in": -4,
        "m_out": -0.5,
        "epochs": 500,
    },
]

ARXIV_PHASE2_PROP_BUNDLES = [
    {"K": 2, "alpha": 0.5},
    {"K": 4, "alpha": 0.7},
]

PHASE2_TARGET_SPECS = {
    "cora_label": {
        "dataset": "cora",
        "ood_type": "label",
        "note": "Phase 2 AUROC-focused sweep for cora label OOD around the current fusion winner.",
        "axes": [
            ("variant", CORA_LABEL_PHASE2_VARIANTS),
            ("training", CORA_LABEL_PHASE2_TRAIN_BUNDLES),
            ("stochasticity", CORA_LABEL_PHASE2_STOCHASTIC_BUNDLES),
            ("prop", CORA_LABEL_PHASE2_PROP_BUNDLES),
        ],
    },
    "twitch": {
        "dataset": "twitch",
        "ood_type": "structure",
        "note": "Phase 2 AUROC-focused sweep for twitch around the current fusion winner plus one AUROC hedge.",
        "axes": [
            ("variant", TWITCH_PHASE2_VARIANTS),
            ("backbone", TWITCH_PHASE2_BACKBONE_BUNDLES),
            ("stochasticity", TWITCH_PHASE2_STOCHASTIC_BUNDLES),
            ("prop", TWITCH_PHASE2_PROP_BUNDLES),
        ],
    },
    "arxiv": {
        "dataset": "arxiv",
        "ood_type": "structure",
        "note": "Phase 2 AUROC-focused sweep for arxiv around the current energy+MI winner with H100-sized capacity.",
        "axes": [
            ("variant", ARXIV_PHASE2_VARIANTS),
            ("capacity", ARXIV_PHASE2_CAPACITY_BUNDLES),
            ("training", ARXIV_PHASE2_TRAIN_BUNDLES),
            ("prop", ARXIV_PHASE2_PROP_BUNDLES),
        ],
    },
}


TWITCH_PHASE3_TRAIN_BUNDLES = [
    {
        "hidden_channels": 64,
        "num_layers": 2,
        "lr": 0.01,
        "dropout": 0.5,
        "weight_decay": 5e-4,
        "warmup_base_epochs": 50,
        "residual_scale_init": 0.1,
        "dss_lambda_reg": 0.01,
        "epochs": 350,
    },
    {
        "hidden_channels": 128,
        "num_layers": 3,
        "lr": 0.0075,
        "dropout": 0.4,
        "weight_decay": 3e-4,
        "warmup_base_epochs": 75,
        "residual_scale_init": 0.08,
        "dss_lambda_reg": 0.01,
        "epochs": 500,
    },
]

TWITCH_PHASE3_FILTER_BUNDLES = [
    {
        "K_lp": 4,
        "K_hp": 4,
    },
    {
        "K_lp": 5,
        "K_hp": 4,
    },
]

TWITCH_PHASE3_STOCHASTIC_BUNDLES = [
    {
        "P": 1,
        "S": 4,
        "use_random_gates": False,
    },
    {
        "P": 2,
        "S": 6,
        "use_random_gates": False,
    },
    {
        "P": 3,
        "P_gate": 1,
        "S": 8,
        "use_random_gates": True,
    },
]

TWITCH_PHASE3_PROP_BUNDLES = [
    {"K": 3, "alpha": 0.6},
    {"K": 4, "alpha": 0.7},
    {"K": 5, "alpha": 0.75},
    {"K": 6, "alpha": 0.8},
]

PHASE3_TARGET_SPECS = {
    "twitch": {
        "dataset": "twitch",
        "ood_type": "structure",
        "config": "gcn_dssres_fusion_probunc",
        "family": "fusion_probunc_phase3",
        "note": "Phase 3 twitch-only AUROC sweep centered on the strongest fusion-based DSS hybrid family.",
        "axes": [
            ("training", TWITCH_PHASE3_TRAIN_BUNDLES),
            ("filters", TWITCH_PHASE3_FILTER_BUNDLES),
            ("stochasticity", TWITCH_PHASE3_STOCHASTIC_BUNDLES),
            ("prop", TWITCH_PHASE3_PROP_BUNDLES),
        ],
    },
}


TWITCH_PHASE3B_TRAIN_BUNDLES = [
    {
        "hidden_channels": 128,
        "num_layers": 3,
        "lr": 0.0075,
        "dropout": 0.4,
        "weight_decay": 3e-4,
        "warmup_base_epochs": 75,
        "residual_scale_init": 0.08,
        "dss_lambda_reg": 0.01,
        "epochs": 200,
    },
]

TWITCH_PHASE3B_FILTER_BUNDLES = [
    {
        "K_lp": 4,
        "K_hp": 4,
    },
    {
        "K_lp": 5,
        "K_hp": 4,
    },
]

TWITCH_PHASE3B_STOCHASTIC_BUNDLES = [
    {
        "P": 1,
        "S": 4,
        "use_random_gates": False,
    },
    {
        "P": 2,
        "S": 6,
        "use_random_gates": False,
    },
    {
        "P": 3,
        "P_gate": 1,
        "S": 8,
        "use_random_gates": True,
    },
]

TWITCH_PHASE3B_PROP_BUNDLES = [
    {"K": 4, "alpha": 0.70},
    {"K": 5, "alpha": 0.72},
    {"K": 5, "alpha": 0.75},
    {"K": 5, "alpha": 0.78},
]

PHASE3B_TARGET_SPECS = {
    "twitch": {
        "dataset": "twitch",
        "ood_type": "structure",
        "config": "gcn_dssres_fusion_probunc",
        "family": "fusion_probunc_phase3b",
        "note": "Phase 3b twitch-only 200-epoch sweep centered on the best phase-3 region, intended as a fair tuned comparison against 200-epoch baselines.",
        "axes": [
            ("training", TWITCH_PHASE3B_TRAIN_BUNDLES),
            ("filters", TWITCH_PHASE3B_FILTER_BUNDLES),
            ("stochasticity", TWITCH_PHASE3B_STOCHASTIC_BUNDLES),
            ("prop", TWITCH_PHASE3B_PROP_BUNDLES),
        ],
    },
}


TARGET_ALIASES = {
    "table1": [
        "cora_structure",
        "cora_feature",
        "cora_label",
        "amazon_photo_structure",
        "amazon_photo_feature",
        "amazon_photo_label",
        "coauthor_cs_structure",
        "coauthor_cs_feature",
        "coauthor_cs_label",
    ],
    "table2": ["twitch", "arxiv"],
    "cora": ["cora_structure", "cora_feature", "cora_label"],
    "amazon-photo": ["amazon_photo_structure", "amazon_photo_feature", "amazon_photo_label"],
    "coauthor-cs": ["coauthor_cs_structure", "coauthor_cs_feature", "coauthor_cs_label"],
    "twitch": ["twitch"],
    "arxiv": ["arxiv"],
    "underperformers": ["cora_label", "twitch", "arxiv"],
    "phase2": ["cora_label", "twitch", "arxiv"],
    "phase3": ["twitch"],
    "phase3b": ["twitch"],
}


def _normalize_targets(requested: list[str] | None, target_specs: dict[str, dict]) -> list[str]:
    if not requested:
        return list(target_specs.keys())
    resolved: list[str] = []
    for item in requested:
        if item == "all":
            resolved.extend(target_specs.keys())
        elif item in TARGET_ALIASES:
            resolved.extend(target for target in TARGET_ALIASES[item] if target in target_specs)
        elif item in target_specs:
            resolved.append(item)
        else:
            raise ValueError(f"Unknown target selector: {item}")
    seen = set()
    ordered = []
    for item in resolved:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _flatten_axes(target_id: str, spec: dict, *, runs: int, epochs: int, tag_prefix: str) -> list[dict]:
    axis_names = [name for name, _ in spec["axes"]]
    axis_values = [values for _, values in spec["axes"]]
    jobs = []
    for combo_idx, chosen in enumerate(itertools.product(*axis_values)):
        params = {}
        for axis_name, value in zip(axis_names, chosen):
            if isinstance(value, dict):
                params.update(value)
            else:
                params[axis_name] = value
        config_name = params.pop("config_name", spec.get("config"))
        if config_name is None:
            raise ValueError(f"Target '{target_id}' did not resolve to a config_name.")
        family = params.pop("family", spec["family"] if "family" in spec else config_name)
        note = params.pop("note", spec["note"])
        job_runs = params.pop("runs", runs)
        job_epochs = params.pop("epochs", epochs)
        tag = f"{tag_prefix}_{target_id}_g{combo_idx:02d}"
        jobs.append(
            {
                "target_id": target_id,
                "dataset": spec["dataset"],
                "ood_type": spec["ood_type"],
                "config_name": config_name,
                "family": family,
                "note": note,
                "runs": job_runs,
                "epochs": job_epochs,
                "tag": tag,
                "params": params,
                "combo_index": combo_idx,
            }
        )
    return jobs


def build_jobs(*, target_specs: dict[str, dict], targets: list[str], runs: int, epochs: int, tag_prefix: str) -> list[dict]:
    jobs = []
    for target_id in targets:
        spec = target_specs[target_id]
        jobs.extend(_flatten_axes(target_id, spec, runs=runs, epochs=epochs, tag_prefix=tag_prefix))
    for job_index, job in enumerate(jobs):
        job["job_index"] = job_index
    return jobs


def build_command(job: dict, *, device: int, data_dir: str, cpu: bool) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/local_ood_density_compare.py",
        "--dataset",
        job["dataset"],
        "--ood_type",
        job["ood_type"],
        "--configs",
        job["config_name"],
        "--runs",
        str(job["runs"]),
        "--epochs",
        str(job["epochs"]),
        "--device",
        str(device),
        "--data_dir",
        data_dir,
        "--display_step",
        "50",
        "--tag",
        job["tag"],
    ]
    if cpu:
        cmd.append("--cpu")

    for key, value in job["params"].items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
            continue
        cmd.extend([f"--{key}", str(value)])
    return cmd


def write_job_record(job: dict, out_dir: Path, command: list[str]) -> None:
    record = {
        "target_id": job["target_id"],
        "dataset": job["dataset"],
        "ood_type": job["ood_type"],
        "config_name": job["config_name"],
        "family": job["family"],
        "note": job["note"],
        "runs": job["runs"],
        "epochs": job["epochs"],
        "tag": job["tag"],
        "combo_index": job["combo_index"],
        "job_index": job["job_index"],
        "params": job["params"],
        "command": command,
    }
    (out_dir / "sweep_job.json").write_text(json.dumps(record, indent=2))


def print_target_table(target_specs: dict[str, dict], target_ids: list[str]) -> None:
    print("Target sweep plan:")
    for target_id in target_ids:
        spec = target_specs[target_id]
        grid_size = 1
        for _, values in spec["axes"]:
            grid_size *= len(values)
        print(
            f"- {target_id}: {spec.get('config', '<multi-config>')} on {spec['dataset']} / {spec['ood_type']} "
            f"({grid_size} jobs) - {spec['note']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset-aware server sweep for DSS OOD models")
    parser.add_argument("--stage", type=str, default=DEFAULT_STAGE, choices=["search", "rescue", "phase2", "phase3", "phase3b"])
    parser.add_argument("--targets", nargs="*", help="Subset of targets, dataset aliases, or table1/table2.")
    parser.add_argument("--runs", type=int, default=3, help="Runs per sweep job.")
    parser.add_argument("--epochs", type=int, default=200, help="Epochs per sweep job.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--job-id", type=int, default=None, help="Execute one job by flattened index.")
    parser.add_argument("--count", action="store_true", help="Print the number of selected jobs.")
    parser.add_argument("--list", action="store_true", help="Print selected targets and their grid sizes.")
    parser.add_argument("--show-jobs", type=int, default=0, help="Print the first N job specs.")
    parser.add_argument("--tag-prefix", type=str, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.stage == "search":
        target_specs = TARGET_SPECS
    elif args.stage == "rescue":
        target_specs = RESCUE_TARGET_SPECS
    elif args.stage == "phase2":
        target_specs = PHASE2_TARGET_SPECS
    elif args.stage == "phase3":
        target_specs = PHASE3_TARGET_SPECS
    else:
        target_specs = PHASE3B_TARGET_SPECS
    tag_prefix = args.tag_prefix or DEFAULT_TAG_PREFIX[args.stage]

    selected_targets = _normalize_targets(args.targets, target_specs)
    jobs = build_jobs(
        target_specs=target_specs,
        targets=selected_targets,
        runs=args.runs,
        epochs=args.epochs,
        tag_prefix=tag_prefix,
    )

    if args.count:
        print(len(jobs))
        return

    if args.list:
        print_target_table(target_specs, selected_targets)
        print(f"Total jobs: {len(jobs)}")
        return

    if args.show_jobs:
        for job in jobs[: args.show_jobs]:
            print(json.dumps(job, indent=2))
        print(f"Total jobs: {len(jobs)}")
        return

    if args.job_id is None:
        raise ValueError("--job-id is required unless using --count, --list, or --show-jobs.")
    if args.job_id < 0 or args.job_id >= len(jobs):
        raise ValueError(f"--job-id {args.job_id} out of range for {len(jobs)} selected jobs.")

    job = jobs[args.job_id]
    cmd = build_command(job, device=args.device, data_dir=args.data_dir, cpu=args.cpu)
    out_dir = REPO_ROOT / "results_local" / "ood_density" / f"{job['dataset']}-{job['ood_type']}-{job['tag']}"

    print(json.dumps(job, indent=2))
    print("Command:")
    print(" ".join(cmd))

    if args.skip_existing and (out_dir / "summary.csv").exists():
        print(f"Skipping existing result bundle: {out_dir}")
        return

    if args.dry_run:
        return

    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    if out_dir.exists():
        write_job_record(job, out_dir, cmd)
        print(f"Wrote sweep metadata to {out_dir / 'sweep_job.json'}")


if __name__ == "__main__":
    main()
