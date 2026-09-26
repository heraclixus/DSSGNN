#!/usr/bin/env python3
"""
Comprehensive hyperparameter grid search for DSS-GNN.
Covers calibration (node classification) and GOOD (distribution shift).

Grid dimensions:
  Calibration: P × λ_reg × loss_type (per dataset, fixed architecture/hidden)
  GOOD: P × loss_type × train_mode (per setting, fixed architecture)

Usage:
  python scripts/run_comprehensive_grid.py --task_idx 0
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

# ============================================================
# CALIBRATION GRID: 8 datasets × P={0,1,2,3} × λ_reg={0,0.01} × loss={mean,quad}
# ============================================================

# (dataset, base_args) — architecture fixed per dataset from best prior config
CAL_DATASETS = [
    ("cora",           "--hidden 256"),
    ("pubmed",         "--hidden 128"),
    ("texas",          "--hidden 64"),
    ("chameleon",      "--hidden 64"),
    ("cornell",        "--hidden 128"),
    ("wisconsin",      "--hidden 64"),
    ("physics",        "--hidden 128"),
    ("amazon-ratings", "--propagate_first --hidden 128"),
]

P_VALUES = [0, 1, 2, 3]
LREG_VALUES = [0.0, 0.01]
LOSS_VALUES = ["mean", "quad"]

TASKS = []

for ds, base_args in CAL_DATASETS:
    for P in P_VALUES:
        for lreg in LREG_VALUES:
            for loss in LOSS_VALUES:
                name = f"cal_{ds}_P{P}_lreg{lreg}_loss{loss}"
                extra = f"{base_args} --P {P} --lambda_reg {lreg} --loss_type {loss}"
                TASKS.append(("calibration", ds, name, extra))

# ============================================================
# GOOD GRID: 3 settings × P={0,1,2} × loss={mean,quad} × mode={erm,tar_adaptive}
# ============================================================

GOOD_SETTINGS = [
    ("GOODArxiv/degree",  "GOOD_clean/configs/GOOD_configs/GOODArxiv/degree/concept/ERM.yaml",  "gpr_dssres"),
    ("GOODCora/degree",   "GOOD_clean/configs/GOOD_configs/GOODCora/degree/concept/ERM.yaml",   "gcn_dssres"),
    ("GOODTwitch/lang",   "GOOD_clean/configs/GOOD_configs/GOODTwitch/language/concept/ERM.yaml","gcn_dssres"),
]

GOOD_P_VALUES = [0, 1, 2]
GOOD_MODES = ["erm", "tar_adaptive"]

for setting_name, config_path, model in GOOD_SETTINGS:
    for P in GOOD_P_VALUES:
        for loss in LOSS_VALUES:
            for mode in GOOD_MODES:
                name = f"good_{setting_name}_P{P}_loss{loss}_{mode}"
                extra = f"--config_path {config_path} --model {model} --P {P} --loss_type {loss} --train_mode {mode}"
                if mode == "tar_adaptive":
                    extra += " --tar_adaptive_lambda 0.5 --tar_eta0 0.2 --tar_eta_lp 0.2 --tar_eta_hp 0.1"
                TASKS.append(("good", setting_name, name, extra))

NUM_TASKS = len(TASKS)


def parse_cal_output(text):
    vals = {}
    _num = r"([\d.]+)\s+(?:[±]\s+)?([\d.]+)"
    for key, pattern in [("acc", r"test acc mean \(%\)"), ("ece", "test ECE mean"),
                          ("mce", "test MCE mean"), ("brier", "test Brier mean")]:
        m = re.search(pattern + r" = " + _num, text)
        if m:
            vals[f"{key}_mean"] = float(m.group(1))
            vals[f"{key}_std"] = float(m.group(2))
    return vals


def parse_good_output(text):
    vals = {}
    for line in text.strip().split('\n'):
        if '  test:' in line:
            for key in ['Acc', 'ECE', 'Brier']:
                m = re.search(f'{key}=([\\d.]+)', line)
                if m:
                    vals[key.lower()] = float(m.group(1))
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

    task_type, dataset, config_name, extra_args = TASKS[args.task_idx]

    print(f"GRID: type={task_type} dataset={dataset} config={config_name} (task {args.task_idx}/{NUM_TASKS})")

    if task_type == "calibration":
        cmd = [
            sys.executable, "dssgnn_training.py",
            "--dataset", dataset,
            "--runs", str(args.runs),
            "--device", str(args.device),
            "--amp",
        ] + extra_args.split()

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        combined = result.stdout + result.stderr
        for line in combined.strip().split('\n')[-6:]:
            print(line)
        vals = parse_cal_output(combined)
        if "acc_mean" in vals:
            parts = [f"type={task_type}", f"dataset={dataset}", f"config={config_name}"]
            for k in ["acc", "ece", "mce", "brier"]:
                if f"{k}_mean" in vals:
                    parts.append(f"{k}_mean={vals[k+'_mean']:.4f}")
                    parts.append(f"{k}_std={vals[k+'_std']:.4f}")
            print(f"GRID_RESULT " + " ".join(parts))
        else:
            print(f"GRID_RESULT type={task_type} dataset={dataset} config={config_name} FAILED", file=sys.stderr)
            sys.exit(1)

    elif task_type == "good":
        cmd = [
            sys.executable, "run_dssgnn_good.py",
            "--hidden", "300",
            "--layers", "3",
            "--pro_dropout", "0.5",
            "--lr", "3e-3",
            "--epochs", "500",
            "--patience", "200",
        ] + extra_args.split()

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        combined = result.stdout + result.stderr
        for line in combined.strip().split('\n')[-8:]:
            print(line)
        vals = parse_good_output(combined)
        if "acc" in vals:
            parts = [f"type={task_type}", f"dataset={dataset}", f"config={config_name}",
                     f"acc={vals['acc']:.4f}"]
            if "brier" in vals:
                parts.append(f"brier={vals['brier']:.4f}")
            print(f"GRID_RESULT " + " ".join(parts))
        else:
            print(f"GRID_RESULT type={task_type} dataset={dataset} config={config_name} FAILED", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    print(f"Total grid tasks: {NUM_TASKS}")
    cal_count = sum(1 for t in TASKS if t[0] == "calibration")
    good_count = sum(1 for t in TASKS if t[0] == "good")
    print(f"  Calibration: {cal_count} tasks ({len(CAL_DATASETS)} datasets × {len(P_VALUES)} P × {len(LREG_VALUES)} λ_reg × {len(LOSS_VALUES)} loss)")
    print(f"  GOOD: {good_count} tasks ({len(GOOD_SETTINGS)} settings × {len(GOOD_P_VALUES)} P × {len(LOSS_VALUES)} loss × {len(GOOD_MODES)} mode)")
    main()
