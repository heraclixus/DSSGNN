#!/usr/bin/env python3
"""
Targeted DS-GNN sweep for datasets that regressed under --reduced_memory.
Runs WITHOUT --reduced_memory, sweeping hidden/P/dropout/propagation.

Usage:
  python scripts/run_calibration_sweep.py --dataset cora-full --config_idx 0
  python scripts/run_calibration_sweep.py --dataset physics --config_idx 3

Output (parseable):
  RESULT dataset=<name> model=dssgnn config=<name> acc_mean=... acc_std=... ece_mean=... ece_std=... mce_mean=... mce_std=... brier_mean=... brier_std=...
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Datasets that need re-sweeping (regressed under --reduced_memory)
SWEEP_DATASETS = ["cora-full", "physics", "amazon-ratings", "cora"]

# Configs to sweep — NO --reduced_memory, but keep --amp for memory
# Each config: (name, extra_args)
SWEEP_CONFIGS = [
    # --- Chebyshev propagation (default) ---
    ("cheb_h64_P2",           "--hidden 64 --P 2"),
    ("cheb_h128_P2",          "--hidden 128 --P 2"),
    ("cheb_h128_P1",          "--hidden 128 --P 1"),
    ("cheb_h256_P1",          "--hidden 256 --P 1"),
    ("cheb_h64_P2_d06",       "--hidden 64 --P 2 --pro_dropout 0.6"),
    ("cheb_h128_P2_d06",      "--hidden 128 --P 2 --pro_dropout 0.6"),
    ("cheb_h128_P1_d06",      "--hidden 128 --P 1 --pro_dropout 0.6"),
    ("cheb_h256_P1_d06",      "--hidden 256 --P 1 --pro_dropout 0.6"),
    # --- TFE-style propagation (heterophily_tfe) ---
    ("htfe_h64",              "--heterophily_tfe"),
    ("htfe_h128",             "--heterophily_tfe --hidden 128"),
    ("htfe_h64_L3",           "--heterophily_tfe --layers_heterophilous 3"),
    ("htfe_h128_L3",          "--heterophily_tfe --layers_heterophilous 3 --hidden 128"),
    ("htfe_h64_d06",          "--heterophily_tfe --pro_dropout 0.6"),
    ("htfe_h128_d06",         "--heterophily_tfe --pro_dropout 0.6 --hidden 128"),
    # --- Propagate-first (prop_first) ---
    ("pf_h64",                "--propagate_first"),
    ("pf_h128",               "--propagate_first --hidden 128"),
    ("pf_rw_h64",             "--propagate_first --gf rw"),
    ("pf_rw_h128",            "--propagate_first --gf rw --hidden 128"),
    ("pf_rms_h64",            "--propagate_first --optimizer_prop RMSprop"),
    ("pf_rms_h128",           "--propagate_first --optimizer_prop RMSprop --hidden 128"),
    ("pf_rw_rms_h64",         "--propagate_first --gf rw --optimizer_prop RMSprop"),
    ("pf_rw_rms_h128",        "--propagate_first --gf rw --optimizer_prop RMSprop --hidden 128"),
    ("pf_rw_d08_h64",         "--propagate_first --gf rw --pro_dropout 0.8"),
    ("pf_rw_d08_h128",        "--propagate_first --gf rw --pro_dropout 0.8 --hidden 128"),
]

NUM_CONFIGS = len(SWEEP_CONFIGS)


def parse_output(text: str, dataset: str, config: str, returncode: int) -> dict:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=SWEEP_DATASETS)
    parser.add_argument("--config_idx", type=int, required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    config_name, extra_args = SWEEP_CONFIGS[args.config_idx]

    cmd = [
        sys.executable, "dssgnn_training.py",
        "--dataset", args.dataset,
        "--runs", str(args.runs),
        "--device", str(args.device),
        "--amp",  # keep AMP for memory, but NO --reduced_memory
    ] + extra_args.split()

    print(f"Calibration sweep: dataset={args.dataset} config={config_name} ({args.config_idx})")
    print(f"  cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)

    # Print full output for debugging
    if result.stdout:
        print(result.stdout[-2000:])  # last 2000 chars
    if result.stderr:
        print(result.stderr[-1000:], file=sys.stderr)

    out = parse_output(result.stdout + result.stderr, args.dataset, config_name, result.returncode)

    if out["ok"]:
        parts = [f"dataset={out['dataset']}", f"model=dssgnn", f"config={out['config']}",
                 f"acc_mean={out['acc_mean']:.2f}", f"acc_std={out['acc_std']:.2f}",
                 f"ece_mean={out['ece_mean']:.4f}", f"ece_std={out['ece_std']:.4f}",
                 f"mce_mean={out['mce_mean']:.4f}", f"mce_std={out['mce_std']:.4f}",
                 f"brier_mean={out['brier_mean']:.4f}", f"brier_std={out['brier_std']:.4f}"]
        print(f"RESULT " + " ".join(parts))
    else:
        print(f"RESULT dataset={out['dataset']} model=dssgnn config={out['config']} FAILED", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
