#!/usr/bin/env python3
"""
Run OOD detection using GNNSafe pipeline (ported from GraphOOD-GNNSafe).

Supports methods including MSP, GNNSafe, GDUQ, Graph-EBM, MC-Dropout, and Deep Ensemble.
Supports backbones: gcn, dssgnn (DSS-GNN), gcn_dssres, tfe, gduq.
OOD types: structure, feature, label (for cora/amazon/coauthor).
Datasets: cora, citeseer, pubmed, amazon-photo, amazon-computer, coauthor-cs, coauthor-physics, twitch, arxiv, proteins.

Example:
  python run_ood_gnnsafe.py --method gnnsafe --backbone dssgnn --dataset cora --ood_type structure --mode detect --use_bn
  python run_ood_gnnsafe.py --method msp --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn
  python run_ood_gnnsafe.py --method graph_ebm --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn
  python run_ood_gnnsafe.py --method mc_dropout --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn --dropout 0.5
  python run_ood_gnnsafe.py --method ensemble --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn
"""

import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn

from gnnsafe_ood.logger import Logger_classify, Logger_detect
from gnnsafe_ood.data_utils import (
    rand_splits,
    evaluate_classify,
    evaluate_detect,
    eval_acc,
    eval_rocauc,
    save_result,
)
from gnnsafe_ood.dataset import load_dataset
from gnnsafe_ood.parse import parser_add_main_args
from gnnsafe_ood.baselines import MSP, GDUQ, GraphEBM, MCDropout, Ensemble
from gnnsafe_ood.gnnsafe import GNNSafe


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description="GNNSafe OOD Pipeline")
    parser_add_main_args(parser)
    args = parser.parse_args()
    print(args)

    fix_seed(args.seed)
    device = torch.device("cpu" if args.cpu else f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    dataset_ind, dataset_ood_tr, dataset_ood_te = load_dataset(args)

    if len(dataset_ind.y.shape) == 1:
        dataset_ind.y = dataset_ind.y.unsqueeze(1)
    if len(dataset_ood_tr.y.shape) == 1:
        dataset_ood_tr.y = dataset_ood_tr.y.unsqueeze(1)
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
    print(f"IND: {dataset_ind.num_nodes} nodes, {c} classes, {d} feats")
    print(f"OOD tr: {dataset_ood_tr.num_nodes} nodes")

    if args.method == "msp":
        model = MSP(d, c, args).to(device)
    elif args.method == "gnnsafe":
        model = GNNSafe(d, c, args).to(device)
    elif args.method == "gduq":
        model = GDUQ(d, c, args).to(device)
    elif args.method == "graph_ebm":
        model = GraphEBM(d, c, args).to(device)
    elif args.method == "mc_dropout":
        model = MCDropout(d, c, args).to(device)
    elif args.method == "ensemble":
        model = Ensemble(d, c, args).to(device)
    else:
        raise ValueError(f"method {args.method}")

    if args.backbone == "gduq":
        train_idx = dataset_ind.splits["train"]
        train_x = dataset_ind.x[train_idx]
        mean = train_x.mean(dim=0)
        std = train_x.std(dim=0)
        std[std == 0] = 1e-3
        model.set_anchor_dist(mean.to(device), std.to(device))

    criterion = nn.BCEWithLogitsLoss() if args.dataset in ("proteins", "ppi") else nn.NLLLoss()
    eval_func = eval_rocauc if args.dataset in ("proteins", "ppi", "twitch") else eval_acc
    logger = Logger_classify(args.runs, args) if args.mode == "classify" else Logger_detect(args.runs, args)

    model.train()
    print("MODEL:", model)

    if not args.cpu and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    for run in range(args.runs):
        run_t0 = time.time()
        model.reset_parameters()
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        for epoch in range(args.epochs):
            if hasattr(model, "set_train_epoch"):
                model.set_train_epoch(epoch)
            model.train()
            optimizer.zero_grad()
            loss = model.loss_compute(dataset_ind, dataset_ood_tr, criterion, device, args)
            loss.backward()
            optimizer.step()

            if args.mode == "classify":
                result = evaluate_classify(model, dataset_ind, eval_func, criterion, args, device)
                logger.add_result(run, result)
            else:
                result = evaluate_detect(model, dataset_ind, dataset_ood_te, criterion, eval_func, args, device)
                logger.add_result(run, result)

            if epoch % args.display_step == 0:
                if args.mode == "classify":
                    print(f"Epoch {epoch:02d} Loss {loss:.4f} Train {100*result[0]:.2f}% Valid {100*result[1]:.2f}% Test {100*result[2]:.2f}%")
                else:
                    print(f"Epoch {epoch:02d} Loss {loss:.4f} AUROC {100*result[0]:.2f}% AUPR {100*result[1]:.2f}% FPR95 {100*result[2]:.2f}% IND {100*result[-2]:.2f}%")

        logger.print_statistics(run)
        print(f"Run {run + 1} wall time: {time.time() - run_t0:.1f}s "
              f"({(time.time() - run_t0) * 1000 / max(args.epochs, 1):.1f}ms/epoch)")

    if not args.cpu and torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"Peak GPU memory allocated: {peak_mb:.1f} MiB")

    results = logger.print_statistics()

    if args.mode == "detect":
        save_result(results, args)

    # Release GPU memory before exit (helps when run as subprocess in a loop)
    if not args.cpu and torch.cuda.is_available():
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    print("Done.")


if __name__ == "__main__":
    main()
