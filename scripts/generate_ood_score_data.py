#!/usr/bin/env python3
"""
Generate per-node OOD scores for density plot visualization.
Saves ID and OOD score arrays for each method.

Usage:
  python scripts/generate_ood_score_data.py --dataset cora --ood_type feature --method gnnsafe --backbone gcn
"""

import argparse
import json
import os
import sys
import numpy as np
import torch

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

from gnnsafe_ood.dataset import load_dataset
from gnnsafe_ood.parse import parser_add_main_args
from gnnsafe_ood.data_utils import evaluate_detect, rand_splits
from gnnsafe_ood.baselines import MSP, GraphEBM
from gnnsafe_ood.gnnsafe import GNNSafe


def main():
    parser = argparse.ArgumentParser()
    parser_add_main_args(parser)
    parser.add_argument("--output_dir", type=str, default="results_local/ood_scores")
    args = parser.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu" if args.cpu else f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    dataset_ind, dataset_ood_tr, dataset_ood_te = load_dataset(args)

    if len(dataset_ind.y.shape) == 1:
        dataset_ind.y = dataset_ind.y.unsqueeze(1)
    if isinstance(dataset_ood_te, list):
        for d in dataset_ood_te:
            if len(d.y.shape) == 1:
                d.y = d.y.unsqueeze(1)
    elif len(dataset_ood_te.y.shape) == 1:
        dataset_ood_te.y = dataset_ood_te.y.unsqueeze(1)

    if args.dataset not in ("cora", "citeseer", "pubmed"):
        dataset_ind.splits = rand_splits(dataset_ind.node_idx, train_prop=args.train_prop, valid_prop=args.valid_prop)

    c = max(dataset_ind.y.max().item() + 1, dataset_ind.y.shape[1])
    d = dataset_ind.x.shape[1]

    if args.method == "gnnsafe":
        model = GNNSafe(d, c, args).to(device)
    elif args.method == "graph_ebm":
        model = GraphEBM(d, c, args).to(device)
    else:
        model = MSP(d, c, args).to(device)

    from torch import nn
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Train
    for epoch in range(args.epochs):
        if hasattr(model, "set_train_epoch"):
            model.set_train_epoch(epoch)
        model.train()
        optimizer.zero_grad()
        loss = model.loss_compute(dataset_ind, dataset_ood_tr, criterion, device, args)
        loss.backward()
        optimizer.step()

    # Get scores
    model.eval()
    with torch.no_grad():
        test_ind_score = model.detect(dataset_ind, dataset_ind.splits["test"], device, args).cpu().numpy()
        if isinstance(dataset_ood_te, list):
            test_ood_score = model.detect(dataset_ood_te[0], dataset_ood_te[0].node_idx, device, args).cpu().numpy()
        else:
            test_ood_score = model.detect(dataset_ood_te, dataset_ood_te.node_idx, device, args).cpu().numpy()

    # Build method name
    method_name = args.method
    if args.method == "gnnsafe":
        if args.use_prop and args.use_reg:
            method_name = "gnnsafe++"
        elif args.use_prop:
            method_name = "gnnsafe"

    backbone_name = args.backbone
    name = f"{method_name}_{backbone_name}"

    os.makedirs(args.output_dir, exist_ok=True)
    output = {
        "dataset": args.dataset,
        "ood_type": args.ood_type,
        "method": name,
        "id_scores": test_ind_score.tolist(),
        "ood_scores": test_ood_score.tolist(),
    }
    out_path = os.path.join(args.output_dir,
                             f"scores_{args.dataset}_{args.ood_type}_{name}.json")
    with open(out_path, "w") as f:
        json.dump(output, f)
    print(f"Saved {out_path}: {len(test_ind_score)} ID, {len(test_ood_score)} OOD scores")


if __name__ == "__main__":
    main()
