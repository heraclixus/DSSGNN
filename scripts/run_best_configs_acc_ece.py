#!/usr/bin/env python3
"""
Run best DS-GNN configs, TFE-GNN, and G-ΔUQ on node classification.
Reports Acc, ECE, MCE, Brier for Table 2 and calibration tables.

Usage:
  python scripts/run_best_configs_acc_ece.py --dataset cora --model dssgnn
  python scripts/run_best_configs_acc_ece.py --dataset cora --model tfe
  python scripts/run_best_configs_acc_ece.py --dataset cora --model gduq

Output (parseable):
  RESULT dataset=<name> model=<name> config=<config> acc_mean=<v> acc_std=<v> ece_mean=<v> ece_std=<v> mce_mean=<v> mce_std=<v> brier_mean=<v> brier_std=<v>
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Best DS-GNN config per dataset (from parse_sweep_results.py)
DSSGNN_BEST_CONFIG = {
    "cora": "prop_first_gfrw_dropout08",
    "citeseer": "P1",
    "pubmed": "hidden128",
    "texas": "prop_first_rmsprop",
    "cornell": "heterophily_tfe_layers3",
    "wisconsin": "dropout06",
    "chameleon": "prop_first_sum",
    "squirrel": "prop_first_gfrw_rmsprop",
    "cora-full": "dropout06",
    "cs": "heterophily_tfe",
    "physics": "dropout06",
    "roman-empire": "heterophily_tfe_layers3",
    "amazon-ratings": "heterophily_tfe_layers3",
    "minesweeper": "prop_first_gfrw_rmsprop",
    "tolokers": "prop_first_gfrw_rmsprop",
    "questions": "prop_first_rmsprop",
}

# Config name -> extra args for dssgnn_training.py
DSSGNN_CONFIG_ARGS = {
    "default": "",
    "hidden128": "--hidden 128",
    "deep_filter": "--hidden 128 --K_lp 6 --K_hp 5",
    "P1": "--P 1",
    "dropout06": "--pro_dropout 0.6",
    "heterophily_tfe": "--heterophily_tfe",
    "prop_first": "--propagate_first",
    "prop_first_gfrw": "--propagate_first --gf rw",
    "prop_first_sum": "--propagate_first --combine sum",
    "prop_first_rmsprop": "--propagate_first --optimizer_prop RMSprop",
    "prop_first_ense": "--propagate_first --use_ense_coe",
    "prop_first_gfrw_rmsprop": "--propagate_first --gf rw --optimizer_prop RMSprop",
    "prop_first_gfrw_dropout08": "--propagate_first --gf rw --pro_dropout 0.8",
    "prop_first_gfrw_layers3": "--propagate_first --gf rw --layers_heterophilous 3",
    "heterophily_tfe_dropout06": "--heterophily_tfe --pro_dropout 0.6",
    "heterophily_tfe_layers3": "--heterophily_tfe --layers_heterophilous 3",
}

DATASETS = list(DSSGNN_BEST_CONFIG.keys())


def run_dssgnn(dataset: str, runs: int = 10, device: int = 0) -> dict:
    config = DSSGNN_BEST_CONFIG[dataset]
    extra = DSSGNN_CONFIG_ARGS.get(config, "")
    cmd = [
        sys.executable, "dssgnn_training.py",
        "--dataset", dataset,
        "--runs", str(runs),
        "--device", str(device),
        "--amp", "--reduced_memory",
    ] + (extra.split() if extra else [])
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
    return parse_output(result.stdout + result.stderr, dataset, "dssgnn", config, result.returncode)


def run_tfe(dataset: str, runs: int = 10, device: int = 0) -> dict:
    cmd = [
        sys.executable, "tfe_training.py",
        "--dataset", dataset,
        "--runs", str(runs),
        "--device", str(device),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
    return parse_output(result.stdout + result.stderr, dataset, "tfe", "default", result.returncode)


def run_gduq(dataset: str, runs: int = 10, device: int = 0) -> dict:
    cmd = [
        sys.executable, "run_gduq_standalone.py",
        "--dataset", dataset,
        "--runs", str(runs),
        "--device", str(device),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
    return parse_output(result.stdout + result.stderr, dataset, "gduq", "default", result.returncode)


def parse_output(text: str, dataset: str, model: str, config: str, returncode: int) -> dict:
    acc_mean = acc_std = ece_mean = ece_std = mce_mean = mce_std = brier_mean = brier_std = None
    # Support both "83.09 ± 0.00" and "83.09 0.0" (tfe_training uses comma-separated print)
    _num = r"([\d.]+)\s+(?:[±]\s+)?([\d.]+)"
    m = re.search(r"test acc mean \(%\) = " + _num, text)
    if m:
        acc_mean, acc_std = float(m.group(1)), float(m.group(2))
    m = re.search(r"test ECE mean = " + _num, text)
    if m:
        ece_mean, ece_std = float(m.group(1)), float(m.group(2))
    m = re.search(r"test MCE mean = " + _num, text)
    if m:
        mce_mean, mce_std = float(m.group(1)), float(m.group(2))
    m = re.search(r"test Brier mean = " + _num, text)
    if m:
        brier_mean, brier_std = float(m.group(1)), float(m.group(2))
    return {
        "dataset": dataset,
        "model": model,
        "config": config,
        "acc_mean": acc_mean,
        "acc_std": acc_std,
        "ece_mean": ece_mean,
        "ece_std": ece_std,
        "mce_mean": mce_mean,
        "mce_std": mce_std,
        "brier_mean": brier_mean,
        "brier_std": brier_std,
        "ok": returncode == 0 and acc_mean is not None and ece_mean is not None,
    }


def main():
    parser = argparse.ArgumentParser(description="Run best configs and report Acc + ECE")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS)
    parser.add_argument("--model", type=str, required=True, choices=["dssgnn", "tfe", "gduq"])
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    if args.model == "dssgnn":
        out = run_dssgnn(args.dataset, args.runs, args.device)
    elif args.model == "tfe":
        out = run_tfe(args.dataset, args.runs, args.device)
    else:
        out = run_gduq(args.dataset, args.runs, args.device)

    # Parseable output for log aggregation
    if out["ok"]:
        parts = [f"dataset={out['dataset']}", f"model={out['model']}", f"config={out['config']}",
                 f"acc_mean={out['acc_mean']:.2f}", f"acc_std={out['acc_std']:.2f}",
                 f"ece_mean={out['ece_mean']:.4f}", f"ece_std={out['ece_std']:.4f}"]
        if out.get("mce_mean") is not None:
            parts.extend([f"mce_mean={out['mce_mean']:.4f}", f"mce_std={out['mce_std']:.4f}"])
        if out.get("brier_mean") is not None:
            parts.extend([f"brier_mean={out['brier_mean']:.4f}", f"brier_std={out['brier_std']:.4f}"])
        print(f"RESULT " + " ".join(parts))
    else:
        print(f"RESULT dataset={out['dataset']} model={out['model']} config={out['config']} FAILED", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
