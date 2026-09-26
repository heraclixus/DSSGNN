#!/usr/bin/env python3
"""
Run OOD detection experiments matching GNNSafe Tables 1 & 2 (arxiv 2302.02914).

Table 1: Cora, Amazon-Photo, Coauthor-CS × structure/feature/label OOD
Table 2: Twitch, Arxiv (multi-graph / time-based)

Methods:
- MSP (baseline)
- Energy (gnnsafe w/o prop, w/o reg)
- Energy FT (gnnsafe w/o prop, w/ reg)
- GNNSafe (gnnsafe w/ prop, w/o reg)
- GNNSafe++ (gnnsafe w/ prop, w/ reg)
- DSS-GNN (MSP + dssgnn backbone)
- TFEGNN (MSP + tfe backbone)
- GDUQ (gduq method)

Usage:
  python scripts/run_ood_table1_table2.py --table 1 --quick   # Table 1, 1 run
  python scripts/run_ood_table1_table2.py --table 2          # Table 2
  python scripts/run_ood_table1_table2.py --all               # Full sweep
  python scripts/run_ood_table1_table2.py --config_set dssgnn_gnnsafe --all   # DSS-GNN + GNNSafe sweep
  python scripts/run_ood_table1_table2.py --config_set dssgnn_chaos_reg --all  # DSS-GNN chaos-regularized sweep
  python scripts/run_ood_table1_table2.py --config_set baselines --all      # MSP, GDUQ, TFEGNN baselines
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Hyperparams from GNNSafe run.sh (dataset, ood_type) -> (m_in, m_out, lamda)
GNNSAFE_HYPER = {
    ("cora", "structure"): (-5, -1, 0.01),
    ("cora", "feature"): (-5, -1, 0.01),
    ("cora", "label"): (-5, -4, 1.0),
    ("amazon-photo", "structure"): (-9, -1, 0.1),  # GNNSafe++
    ("amazon-photo", "feature"): (-9, -1, 1.0),
    ("amazon-photo", "label"): (-9, -4, 1.0),
    ("coauthor-cs", "structure"): (-5, -1, 0.1),
    ("coauthor-cs", "feature"): (-7, -1, 0.1),
    ("coauthor-cs", "label"): (-9, -2, 0.01),
    ("twitch", None): (-5, -1, 0.1),  # GNNSafe++
    ("arxiv", None): (-9, -2, 0.01),   # GNNSafe++
}

# Config sets for SLURM split
CONFIG_SETS = {
    "dssgnn_gnnsafe": [
        ("gnnsafe", "dssgnn", False, False),   # Energy
        ("gnnsafe", "dssgnn", False, True),   # Energy FT
        ("gnnsafe", "dssgnn", True, False),   # GNNSafe
        ("gnnsafe", "dssgnn", True, True),    # GNNSafe++
    ],
    "baselines": [
        ("msp", "gcn", None, None),
        ("msp", "tfe", None, None),
        ("msp", "gduq", None, None),
        ("gduq", "gduq", None, None),
    ],
    "dssgnn_scores": [
        ("msp", "dssgnn", False, False, ["--score_type", "msp"]),      # MSP
        ("msp", "dssgnn", False, False, ["--score_type", "energy"]),   # Raw energy
        ("msp", "dssgnn", True, False, ["--score_type", "energy"]),    # Propagated energy
        ("msp", "dssgnn", False, False, ["--score_type", "chaos"]),    # Raw chaos score
        ("msp", "dssgnn", True, False, ["--score_type", "chaos"]),     # Propagated chaos score
    ],
    "dssgnn_chaos_reg": [
        ("gnnsafe", "dssgnn", False, True, ["--reg_score_type", "chaos"]),  # Chaos FT
        ("gnnsafe", "dssgnn", True, True, ["--reg_score_type", "chaos"]),   # Chaos Safe
    ],
}

# Method configs: (method, backbone, use_prop, use_reg) for gnnsafe; None for others
METHOD_CONFIGS = [
    ("msp", "gcn", None, None),
    ("msp", "dssgnn", None, None),
    ("msp", "tfe", None, None),
    ("msp", "gduq", None, None),
    ("gduq", "gduq", None, None),
    ("gnnsafe", "gcn", False, False),   # Energy
    ("gnnsafe", "gcn", False, True),   # Energy FT
    ("gnnsafe", "gcn", True, False),   # GNNSafe
    ("gnnsafe", "gcn", True, True),    # GNNSafe++
    ("gnnsafe", "dssgnn", False, False),
    ("gnnsafe", "dssgnn", False, True),
    ("gnnsafe", "dssgnn", True, False),
    ("gnnsafe", "dssgnn", True, True),
    ("gnnsafe", "tfe", False, False),
    ("gnnsafe", "tfe", True, False),
]


def build_cmd(method, backbone, dataset, ood_type, runs, epochs, device, data_dir, extra_args):
    cmd = [
        sys.executable, "run_ood_gnnsafe.py",
        "--method", method,
        "--backbone", backbone,
        "--dataset", dataset,
        "--mode", "detect",
        "--runs", str(runs),
        "--epochs", str(epochs),
        "--device", str(device),
        "--data_dir", str(data_dir),
    ]
    if dataset in ("cora", "amazon-photo", "coauthor-cs") and ood_type:
        cmd += ["--ood_type", ood_type]
    if dataset in ("amazon-photo", "amazon-computer", "coauthor-cs", "coauthor-physics"):
        cmd += ["--use_bn"]
    if extra_args.get("use_prop"):
        cmd += ["--use_prop"]
    if method == "gnnsafe" and extra_args.get("use_reg"):
        hp = GNNSAFE_HYPER.get((dataset, ood_type if ood_type else None), (-5, -1, 0.01))
        m_in, m_out, lamda = hp
        cmd += ["--use_reg", "--m_in", str(m_in), "--m_out", str(m_out), "--lamda", str(lamda)]
    cmd += extra_args.get("args", [])
    return cmd


def unpack_method_config(config):
    if len(config) == 4:
        method, backbone, use_prop, use_reg = config
        return method, backbone, use_prop, use_reg, []
    method, backbone, use_prop, use_reg, extra_cli = config
    return method, backbone, use_prop, use_reg, extra_cli


def main():
    parser = argparse.ArgumentParser(description="Run OOD Table 1 & 2 experiments")
    parser.add_argument("--table", type=int, choices=[1, 2], help="Table 1 or 2")
    parser.add_argument("--all", action="store_true", help="Run both tables")
    parser.add_argument("--quick", action="store_true", help="1 run, 50 epochs")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--methods", type=str, nargs="*", help="Filter by method")
    parser.add_argument("--datasets", type=str, nargs="*", help="Filter by dataset")
    parser.add_argument("--job_id", type=int, default=None, help="SLURM array: run only this job index")
    parser.add_argument("--total_jobs", type=int, default=None, help="SLURM array: total jobs")
    parser.add_argument(
        "--config_set",
        type=str,
        choices=["dssgnn_gnnsafe", "baselines", "dssgnn_scores", "dssgnn_chaos_reg"],
        help="Run only dssgnn+GNNSafe, baseline models, or DSS-GNN score variants",
    )
    args = parser.parse_args()

    runs = 1 if args.quick else args.runs
    epochs = 50 if args.quick else args.epochs

    if args.table == 1 or args.all:
        table1_configs = [
            ("cora", "structure"), ("cora", "feature"), ("cora", "label"),
            ("amazon-photo", "structure"), ("amazon-photo", "feature"), ("amazon-photo", "label"),
            ("coauthor-cs", "structure"), ("coauthor-cs", "feature"), ("coauthor-cs", "label"),
        ]
    else:
        table1_configs = []

    if args.table == 2 or args.all:
        table2_configs = [("twitch", None), ("arxiv", None)]
    else:
        table2_configs = []

    configs = table1_configs + table2_configs

    root = Path(__file__).resolve().parent.parent
    os.chdir(root)

    # Resolve method configs
    method_configs = CONFIG_SETS[args.config_set] if args.config_set else METHOD_CONFIGS

    # Build flat list of (dataset, ood_type, method, backbone, use_prop, use_reg)
    jobs = []
    for dataset, ood_type in configs:
        for config in method_configs:
            method, backbone, use_prop, use_reg, extra_cli = unpack_method_config(config)
            if args.methods and method not in args.methods:
                continue
            if args.datasets and dataset not in args.datasets:
                continue
            jobs.append((dataset, ood_type, method, backbone, use_prop, use_reg, extra_cli))

    if args.job_id is not None and args.total_jobs is not None:
        jobs = [jobs[args.job_id]] if args.job_id < len(jobs) else []
        if not jobs:
            print(f"Job {args.job_id} out of range (total {len(jobs)})")
            return

    total = 0
    for dataset, ood_type, method, backbone, use_prop, use_reg, extra_cli in jobs:
        extra = {
            "use_prop": bool(use_prop),
            "use_reg": bool(use_reg),
            "args": list(extra_cli),
        }
        cmd = build_cmd(method, backbone, dataset, ood_type, runs, epochs,
                        args.device, args.data_dir, extra)
        print(" ".join(cmd))
        ret = subprocess.run(cmd, cwd=root)
        if ret.returncode != 0:
            print(f"FAILED: {' '.join(cmd)}", file=sys.stderr)
        total += 1

    print(f"Total jobs run: {total}")


if __name__ == "__main__":
    main()
