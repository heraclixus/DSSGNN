#!/usr/bin/env python3
"""
Run DSS-GNN variants sweep on Table 1 datasets.

Variants (from method.tex):
  - dssgnn:      S (base, best config per dataset)
  - dssgnn_sg:   S+G (random branch gates)
  - dssgnn_sf:   S+F (random filter coefficients)
  - dssgnn_gduq: S+A (GDUQ anchoring)
  - dssgnn_gduq_sg: S+G+A (anchoring + random gates)
  - dssgnn_gduq_sf: S+F+A (anchoring + random filter)

Output (parseable): RESULT dataset=... variant=... config=... acc_mean=... ece_mean=... mce_mean=... brier_mean=...

Usage:
  python scripts/run_dssgnn_variants_sweep.py --dataset cora --variant dssgnn
  python scripts/run_dssgnn_variants_sweep.py --dataset cora --variant dssgnn_sg
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Best DSS-GNN config per dataset (from parse_sweep_results.py)
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

# Datasets where best config uses prop_first -> need --propagate_first for run_dssgnn_gduq
GDUQ_PROP_FIRST_DATASETS = {
    "cora", "texas", "cornell", "wisconsin", "chameleon", "squirrel",
    "roman-empire", "amazon-ratings", "minesweeper", "tolokers", "questions",
}

DATASETS = list(DSSGNN_BEST_CONFIG.keys())
VARIANTS = ["dssgnn", "dssgnn_sg", "dssgnn_sf", "dssgnn_gduq", "dssgnn_gduq_sg", "dssgnn_gduq_sf"]


def parse_output(text: str, dataset: str, variant: str, config: str, returncode: int) -> dict:
    acc_mean = acc_std = ece_mean = ece_std = mce_mean = mce_std = brier_mean = brier_std = None
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
        "variant": variant,
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


def run_dssgnn_variant(dataset: str, variant: str, runs: int = 10, device: int = 0) -> dict:
    config = DSSGNN_BEST_CONFIG[dataset]
    extra = DSSGNN_CONFIG_ARGS.get(config, "")
    base_args = [
        sys.executable, "dssgnn_training.py",
        "--dataset", dataset,
        "--runs", str(runs),
        "--device", str(device),
        "--amp", "--reduced_memory",
    ] + (extra.split() if extra else [])

    if variant == "dssgnn":
        cmd = base_args
    elif variant == "dssgnn_sg":
        cmd = base_args + ["--use_random_gates", "--P_gate", "1", "--lambda_op", "0.01"]
    elif variant == "dssgnn_sf":
        cmd = base_args + ["--use_random_filter", "--P_filter", "1", "--lambda_filter", "0.01"]
    else:
        raise ValueError(f"Unknown variant: {variant}")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
    return parse_output(result.stdout + result.stderr, dataset, variant, config, result.returncode)


def run_dssgnn_gduq_variant(dataset: str, variant: str, runs: int = 10, device: int = 0) -> dict:
    config = DSSGNN_BEST_CONFIG[dataset]
    extra = ["--propagate_first"] if dataset in GDUQ_PROP_FIRST_DATASETS else []
    extra += ["--num_anchors", "5"]

    if variant == "dssgnn_gduq":
        cmd = [
            sys.executable, "run_dssgnn_gduq.py",
            "--dataset", dataset,
            "--runs", str(runs),
            "--device", str(device),
        ] + extra
    elif variant == "dssgnn_gduq_sg":
        cmd = [
            sys.executable, "run_dssgnn_gduq.py",
            "--dataset", dataset,
            "--runs", str(runs),
            "--device", str(device),
            "--use_random_gates", "--P_gate", "1", "--lambda_op", "0.01",
        ] + extra
    elif variant == "dssgnn_gduq_sf":
        cmd = [
            sys.executable, "run_dssgnn_gduq.py",
            "--dataset", dataset,
            "--runs", str(runs),
            "--device", str(device),
            "--use_random_filter", "--P_filter", "1", "--lambda_filter", "0.01",
        ] + extra
    else:
        raise ValueError(f"Unknown variant: {variant}")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
    return parse_output(result.stdout + result.stderr, dataset, variant, config, result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run DS-GNN variants sweep")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS)
    parser.add_argument("--variant", type=str, required=True, choices=VARIANTS)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    if args.variant in ("dssgnn", "dssgnn_sg", "dssgnn_sf"):
        out = run_dssgnn_variant(args.dataset, args.variant, args.runs, args.device)
    else:
        out = run_dssgnn_gduq_variant(args.dataset, args.variant, args.runs, args.device)

    if out["ok"]:
        parts = [
            f"dataset={out['dataset']}", f"variant={out['variant']}", f"config={out['config']}",
            f"acc_mean={out['acc_mean']:.2f}", f"acc_std={out['acc_std']:.2f}",
            f"ece_mean={out['ece_mean']:.4f}", f"ece_std={out['ece_std']:.4f}",
        ]
        if out.get("mce_mean") is not None:
            parts.extend([f"mce_mean={out['mce_mean']:.4f}", f"mce_std={out['mce_std']:.4f}"])
        if out.get("brier_mean") is not None:
            parts.extend([f"brier_mean={out['brier_mean']:.4f}", f"brier_std={out['brier_std']:.4f}"])
        print(f"RESULT " + " ".join(parts))
    else:
        print(f"RESULT dataset={out['dataset']} variant={out['variant']} config={out['config']} FAILED", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
