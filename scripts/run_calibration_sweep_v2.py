#!/usr/bin/env python3
"""
Aggressive DS-GNN sweep v2 for cora-full, physics, cora.
Targets training instability (cora-full) and closing TFE gap (physics).
NO --reduced_memory. Varies lr, lambda_reg, P, hidden aggressively.

Usage:
  python scripts/run_calibration_sweep_v2.py --task_idx 0
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Each task: (dataset, config_name, extra_args_string)
TASKS = []

# ============================================================
# CORA-FULL (70 classes): training instability, need stability
# ============================================================
for h in [128, 256, 512]:
    for lr in ["0.01", "0.005", "0.001"]:
        for lreg in ["0.0", "0.001", "0.01"]:
            # P=0: deterministic spectral filter (no chaos overhead, baseline)
            TASKS.append(("cora-full", f"cf_P0_h{h}_lr{lr}_lreg{lreg}",
                          f"--hidden {h} --P 0 --lr {lr} --lambda_reg {lreg}"))
            # P=1: minimal chaos
            TASKS.append(("cora-full", f"cf_P1_h{h}_lr{lr}_lreg{lreg}",
                          f"--hidden {h} --P 1 --lr {lr} --lambda_reg {lreg}"))

# cora-full with prop_first (what works on heterophilous datasets)
for h in [128, 256]:
    for opt in ["Adam", "RMSprop"]:
        TASKS.append(("cora-full", f"cf_pf_h{h}_{opt}",
                      f"--propagate_first --hidden {h} --optimizer_prop {opt}"))
        TASKS.append(("cora-full", f"cf_pf_rw_h{h}_{opt}",
                      f"--propagate_first --gf rw --hidden {h} --optimizer_prop {opt}"))

# ============================================================
# PHYSICS: close the gap to TFE (97.22) and original (97.53)
# ============================================================
for h in [64, 128, 256]:
    for P in [1, 2, 3]:
        TASKS.append(("physics", f"ph_h{h}_P{P}",
                      f"--hidden {h} --P {P}"))
        TASKS.append(("physics", f"ph_h{h}_P{P}_d06",
                      f"--hidden {h} --P {P} --pro_dropout 0.6"))

# physics with prop_first
for h in [64, 128, 256, 512]:
    TASKS.append(("physics", f"ph_pf_h{h}",
                  f"--propagate_first --hidden {h}"))
    TASKS.append(("physics", f"ph_pf_rms_h{h}",
                  f"--propagate_first --optimizer_prop RMSprop --hidden {h}"))
    TASKS.append(("physics", f"ph_pf_rw_h{h}",
                  f"--propagate_first --gf rw --hidden {h}"))

# physics with lower lr
for lr in ["0.005", "0.001"]:
    for h in [64, 128]:
        TASKS.append(("physics", f"ph_h{h}_lr{lr}",
                      f"--hidden {h} --lr {lr}"))

# ============================================================
# CORA: try to close gap to original 87.54
# ============================================================
for h in [128, 256, 512]:
    TASKS.append(("cora", f"co_h{h}_P2",
                  f"--hidden {h} --P 2"))
    TASKS.append(("cora", f"co_h{h}_P2_d06",
                  f"--hidden {h} --P 2 --pro_dropout 0.6"))
    TASKS.append(("cora", f"co_pf_rw_d08_h{h}",
                  f"--propagate_first --gf rw --pro_dropout 0.8 --hidden {h}"))

# cora with lower lr
for lr in ["0.005", "0.001"]:
    TASKS.append(("cora", f"co_h256_P1_lr{lr}",
                  f"--hidden 256 --P 1 --lr {lr}"))


NUM_TASKS = len(TASKS)


def parse_output(text):
    vals = {}
    _num = r"([\d.]+)\s+(?:[±]\s+)?([\d.]+)"
    for key in ["acc", "ECE", "MCE", "Brier"]:
        pattern = f"test {key} mean" if key != "acc" else r"test acc mean \(%\)"
        m = re.search(pattern + r" = " + _num, text)
        if m:
            vals[key.lower() + "_mean"] = float(m.group(1))
            vals[key.lower() + "_std"] = float(m.group(2))
    return vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_idx", type=int, required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    if args.task_idx >= NUM_TASKS:
        print(f"task_idx {args.task_idx} >= NUM_TASKS {NUM_TASKS}, skipping")
        return

    dataset, config_name, extra_args = TASKS[args.task_idx]

    cmd = [
        sys.executable, "dssgnn_training.py",
        "--dataset", dataset,
        "--runs", str(args.runs),
        "--device", str(args.device),
        "--amp",
    ] + extra_args.split()

    print(f"Sweep v2: dataset={dataset} config={config_name} (task {args.task_idx}/{NUM_TASKS})")
    print(f"  cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)

    combined = result.stdout + result.stderr
    # Print last lines for debugging
    lines = combined.strip().split('\n')
    for line in lines[-10:]:
        print(line)

    vals = parse_output(combined)
    if "acc_mean" in vals:
        parts = [f"dataset={dataset}", f"model=dssgnn", f"config={config_name}"]
        for k in ["acc", "ece", "mce", "brier"]:
            if f"{k}_mean" in vals:
                parts.append(f"{k}_mean={vals[k+'_mean']:.4f}")
                parts.append(f"{k}_std={vals[k+'_std']:.4f}")
        print(f"RESULT " + " ".join(parts))
    else:
        print(f"RESULT dataset={dataset} model=dssgnn config={config_name} FAILED", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    print(f"Total tasks defined: {NUM_TASKS}")
    main()
