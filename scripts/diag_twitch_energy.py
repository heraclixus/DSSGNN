#!/usr/bin/env python3
"""
Diagnostic: why does the standalone DSS-GNN energy score fail on cross-graph
twitch OOD transfer while the hybrid's works?

Trains (1) standalone dssgnn (lr 0.001, hidden 128, GNNSafe, no reg) and
(2) the published hybrid gcn_dssres GNNSafe++ config, then reports per-graph
score distributions (mean/std) and AUROC-vs-ID for several detection scores,
with and without propagation. One seed; distributions are the point here.

Usage (from repo root, needs the twitch dataset):
  .venv/bin/python scripts/diag_twitch_energy.py
"""

import argparse
import copy
import os
import random
import sys

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gnnsafe_ood.data_utils import rand_splits, get_measures, eval_rocauc
from gnnsafe_ood.dataset import load_dataset
from gnnsafe_ood.parse import parser_add_main_args
from gnnsafe_ood.gnnsafe import GNNSafe


def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def build_args(overrides):
    parser = argparse.ArgumentParser()
    parser_add_main_args(parser)
    args = parser.parse_args([])
    args.dataset = "twitch"
    args.method = "gnnsafe"
    args.mode = "detect"
    args.epochs = 200
    args.runs = 1
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def train_model(args, dataset_ind, dataset_ood_tr, device):
    c = max(dataset_ind.y.max().item() + 1, dataset_ind.y.shape[1])
    d = dataset_ind.x.shape[1]
    model = GNNSafe(d, c, args).to(device)
    model.reset_parameters()
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        if hasattr(model, "set_train_epoch"):
            model.set_train_epoch(epoch)
        model.train()
        optimizer.zero_grad()
        loss = model.loss_compute(dataset_ind, dataset_ood_tr, criterion, device, args)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def score_stats(model, args, dataset_ind, ood_list, device, score_type, use_prop):
    a = copy.copy(args)
    a.score_type = score_type
    a.use_prop = use_prop
    with torch.no_grad():
        ind_score = model.detect(dataset_ind, dataset_ind.splits["test"], device, a).cpu()
        train_score = model.detect(dataset_ind, dataset_ind.splits["train"], device, a).cpu()
    row = [
        f"score={score_type:<11s} prop={'on ' if use_prop else 'off'}",
        f"ID-train {train_score.mean():+8.3f} ± {train_score.std():6.3f}",
        f"ID-test {ind_score.mean():+8.3f} ± {ind_score.std():6.3f}",
    ]
    for i, d_ood in enumerate(ood_list):
        with torch.no_grad():
            ood_score = model.detect(d_ood, d_ood.node_idx, device, a).cpu()
        auroc, _, _, _ = get_measures(ind_score, ood_score)
        row.append(
            f"OOD{i + 1} {ood_score.mean():+8.3f} ± {ood_score.std():6.3f} AUROC {100 * auroc:5.1f}"
        )
    print(" | ".join(row))


def main():
    fix_seed()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    base = build_args({})
    dataset_ind, dataset_ood_tr, dataset_ood_te = load_dataset(base)
    if len(dataset_ind.y.shape) == 1:
        dataset_ind.y = dataset_ind.y.unsqueeze(1)
    if len(dataset_ood_tr.y.shape) == 1:
        dataset_ood_tr.y = dataset_ood_tr.y.unsqueeze(1)
    ood_list = dataset_ood_te if isinstance(dataset_ood_te, list) else [dataset_ood_te]
    for d_ood in ood_list:
        if len(d_ood.y.shape) == 1:
            d_ood.y = d_ood.y.unsqueeze(1)
    dataset_ind.splits = rand_splits(
        dataset_ind.node_idx, train_prop=base.train_prop, valid_prop=base.valid_prop
    )
    print(f"IND nodes {dataset_ind.num_nodes}; OOD test graphs: {len(ood_list)} "
          f"({[d.num_nodes for d in ood_list]})")

    configs = {
        "standalone dssgnn (lr1e-3 h128, no reg)": build_args(
            {"backbone": "dssgnn", "lr": 0.001, "hidden_channels": 128,
             "m_in": -5.0, "m_out": -1.0, "lamda": 0.1, "use_prop": True}
        ),
        "hybrid gcn_dssres GNNSafe++ (published cfg)": build_args(
            {"backbone": "gcn_dssres", "use_bn": True, "use_reg": True, "use_prop": True,
             "m_in": -5.0, "m_out": -1.0, "lamda": 0.1}
        ),
    }

    for name, args in configs.items():
        print(f"\n=== {name} ===")
        fix_seed()
        model = train_model(args, dataset_ind, dataset_ood_tr, device)
        with torch.no_grad():
            out = model(dataset_ind, device).cpu()
        test_idx = dataset_ind.splits["test"]
        id_metric = eval_rocauc(dataset_ind.y[test_idx], out[test_idx])
        print(f"ID test ROC-AUC: {100 * id_metric:.2f}")
        score_types = ["energy", "msp", "chaos", "chaos_norm", "chaos_ratio"]
        for st in score_types:
            for prop in (False, True):
                try:
                    score_stats(model, args, dataset_ind, ood_list, device, st, prop)
                except Exception as exc:  # keep the diagnostic going per score type
                    print(f"score={st} prop={prop}: FAILED ({exc})")


if __name__ == "__main__":
    main()
