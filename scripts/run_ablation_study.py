#!/usr/bin/env python3
"""
Ablation studies for DSS-GNN.
Three ablations: chaos order P, quadrature nodes S, regularization λ_reg.

Usage:
  python scripts/run_ablation_study.py --task_idx 0
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Representative datasets (homophilous, heterophilous, large)
ABLATION_DATASETS = ["cora", "pubmed", "texas", "chameleon", "physics", "cora-full"]

# Base config per dataset (best non-P, non-S, non-lambda_reg config)
# These hold everything fixed except the ablation variable.
BASE_CONFIGS = {
    "cora":      "--hidden 256 --propagate_first --gf rw --pro_dropout 0.8",
    "pubmed":    "--hidden 128",
    "texas":     "--propagate_first --optimizer_prop RMSprop",
    "chameleon": "--propagate_first --combine sum",
    "physics":   "--hidden 64",
    "cora-full": "--hidden 512 --lr 0.001 --pro_dropout 0.7",
}

TASKS = []

# ============================================================
# Ablation 1: Chaos order P (0, 1, 2, 3)
# ============================================================
for ds in ABLATION_DATASETS:
    for P in [0, 1, 2, 3]:
        name = f"P_ablation_{ds}_P{P}"
        extra = f"{BASE_CONFIGS[ds]} --P {P} --lambda_reg 0.01"
        # For cora-full with P>0, use lambda_reg=0 (P>0 is unstable with reg)
        if ds == "cora-full" and P > 0:
            extra = f"{BASE_CONFIGS[ds]} --P {P} --lambda_reg 0.0"
        TASKS.append((ds, name, extra, "P_ablation"))

# ============================================================
# Ablation 2: Quadrature nodes S (2, 4, 6, 8), fix P=2
# Only on datasets using base DSSGNN (Chebyshev), not prop_first
# (prop_first doesn't use quadrature)
# ============================================================
S_DATASETS = ["pubmed", "physics"]  # These use Chebyshev mode
S_BASE_CONFIGS = {
    "pubmed": "--hidden 128 --P 2",
    "physics": "--hidden 64 --P 2",
}
for ds in S_DATASETS:
    for S in [2, 4, 6, 8]:
        name = f"S_ablation_{ds}_S{S}"
        extra = f"{S_BASE_CONFIGS[ds]} --S {S}"
        TASKS.append((ds, name, extra, "S_ablation"))

# ============================================================
# Ablation 3: Regularization λ_reg (0, 0.001, 0.01, 0.1), fix P=2
# ============================================================
REG_DATASETS = ["cora", "pubmed", "texas", "chameleon"]
for ds in REG_DATASETS:
    for lreg in [0.0, 0.001, 0.01, 0.1]:
        name = f"reg_ablation_{ds}_lreg{lreg}"
        extra = f"{BASE_CONFIGS[ds]} --P 2 --lambda_reg {lreg}"
        TASKS.append((ds, name, extra, "reg_ablation"))

NUM_TASKS = len(TASKS)


def parse_output(text):
    vals = {}
    _num = r"([\d.]+)\s+(?:[±]\s+)?([\d.]+)"
    for key, pattern in [("acc", r"test acc mean \(%\)"), ("ece", "test ECE mean"),
                          ("mce", "test MCE mean"), ("brier", "test Brier mean")]:
        m = re.search(pattern + r" = " + _num, text)
        if m:
            vals[f"{key}_mean"] = float(m.group(1))
            vals[f"{key}_std"] = float(m.group(2))
    return vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_idx", type=int, required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    if args.task_idx >= NUM_TASKS:
        print(f"task_idx {args.task_idx} >= {NUM_TASKS}, skipping")
        return

    dataset, config_name, extra_args, ablation_type = TASKS[args.task_idx]

    cmd = [
        sys.executable, "dssgnn_training.py",
        "--dataset", dataset,
        "--runs", str(args.runs),
        "--device", str(args.device),
        "--amp",
    ] + extra_args.split()

    print(f"Ablation: type={ablation_type} dataset={dataset} config={config_name} (task {args.task_idx}/{NUM_TASKS})")
    print(f"  cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)

    combined = result.stdout + result.stderr
    for line in combined.strip().split('\n')[-6:]:
        print(line)

    vals = parse_output(combined)
    if "acc_mean" in vals:
        parts = [f"dataset={dataset}", f"ablation={ablation_type}", f"config={config_name}"]
        for k in ["acc", "ece", "mce", "brier"]:
            if f"{k}_mean" in vals:
                parts.append(f"{k}_mean={vals[k+'_mean']:.4f}")
                parts.append(f"{k}_std={vals[k+'_std']:.4f}")
        print(f"RESULT " + " ".join(parts))
    else:
        print(f"RESULT dataset={dataset} ablation={ablation_type} config={config_name} FAILED", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    print(f"Total ablation tasks: {NUM_TASKS}")
    main()
