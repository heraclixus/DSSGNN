"""
Run baseline models (TFE-GNN or GDUQ) on node classification datasets.

Usage:
  python run_baseline.py --baseline tfe --dataset cora --runs 10
  python run_baseline.py --baseline gduq --dataset cora --runs 10
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Run TFE or GDUQ baseline")
    parser.add_argument("--baseline", type=str, required=True, choices=["tfe", "gduq"])
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    args, extra = parser.parse_known_args()

    # Use native defaults from tfe_training.py and run_gduq_standalone.py for epochs/patience
    if args.baseline == "tfe":
        cmd = [
            sys.executable, "tfe_training.py",
            "--dataset", args.dataset,
            "--runs", str(args.runs),
            "--device", str(args.device),
        ] + extra
        subprocess.run(cmd, check=True)
    elif args.baseline == "gduq":
        cmd = [
            sys.executable, "run_gduq_standalone.py",
            "--dataset", args.dataset,
            "--runs", str(args.runs),
            "--device", str(args.device),
        ] + extra
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
