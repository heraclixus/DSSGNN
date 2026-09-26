#!/usr/bin/env python3
"""Parse DSS-GNN sweep logs and report best config per dataset."""

import re
from pathlib import Path
from collections import defaultdict

DATASETS = ["cora", "citeseer", "pubmed", "texas", "cornell", "wisconsin",
            "chameleon", "squirrel", "cora-full", "cs", "physics",
            "roman-empire", "amazon-ratings", "minesweeper", "tolokers", "questions"]
CONFIG_NAMES = ["default", "hidden128", "deep_filter", "P1", "dropout06",
                "heterophily_tfe", "prop_first", "prop_first_gfrw", "prop_first_sum",
                "prop_first_rmsprop", "prop_first_ense", "prop_first_gfrw_rmsprop",
                "prop_first_gfrw_dropout08", "prop_first_gfrw_layers3",
                "heterophily_tfe_dropout06", "heterophily_tfe_layers3",
                "gfrw", "gfrw_rmsprop", "gfrw_dropout08", "gfrw_layers3",
                "gfrw_dropout08_layers3", "gfrw_patience400"]
NUM_CONFIGS = 7

def parse_log(path):
    """Extract (dataset, config, acc_mean, acc_std) from a sweep log."""
    with open(path) as f:
        lines = f.readlines()
    dataset, config = None, None
    acc_mean, acc_std = None, None
    for line in lines:
        m = re.match(r"(?:DS|DSS)-GNN (?:sweep|cornell/squirrel|heterophilous PyG): dataset=(\S+) config=(\S+)", line)
        if m:
            dataset, config = m.group(1), m.group(2)
        m = re.search(r"test acc mean \(%\) = ([\d.]+) ± ([\d.]+)", line)
        if m:
            acc_mean, acc_std = float(m.group(1)), float(m.group(2))
        elif re.search(r"test acc mean", line):
            # Format: "87.18 0.82" (no ±)
            parts = line.split("=")[-1].strip().split()
            if len(parts) >= 2:
                acc_mean, acc_std = float(parts[0]), float(parts[1])
    return dataset, config, acc_mean, acc_std

def main():
    log_dir = Path(__file__).parent.parent / "slurm" / "logs"
    files = (list(log_dir.glob("dssgnn_sweep_*.out")) +
             list(log_dir.glob("dssgnn_cornell_squirrel_*.out")) +
             list(log_dir.glob("dssgnn_heterophilous_pyg_*.out")) +
             list(log_dir.glob("dsgnn_sweep_*.out")) +
             list(log_dir.glob("dsgnn_cornell_squirrel_*.out")) +
             list(log_dir.glob("dsgnn_heterophilous_pyg_*.out")))
    print(f"Found {len(files)} sweep log files")

    # Collect all results: (dataset, config) -> list of (acc_mean, acc_std)
    results = defaultdict(list)
    for path in files:
        dataset, config, acc_mean, acc_std = parse_log(path)
        if dataset and config and acc_mean is not None:
            results[(dataset, config)].append((acc_mean, acc_std))

    # Best per (dataset, config): use max acc_mean; if tie, prefer lower std
    best_per_config = {}
    for (dataset, config), vals in results.items():
        best = max(vals, key=lambda x: (x[0], -x[1]))
        best_per_config[(dataset, config)] = best

    # Best config per dataset (consider all configs that have results for this dataset)
    print("\n" + "=" * 60)
    print("Best DSS-GNN config per dataset (from sweep)")
    print("=" * 60)
    table = []
    for dataset in DATASETS:
        best_acc, best_config, best_std = 0, "", 0
        for (ds, config), (acc, std) in best_per_config.items():
            if ds == dataset and acc > best_acc:
                best_acc, best_config, best_std = acc, config, std
        table.append((dataset, best_acc, best_std, best_config))
        if best_config:
            print(f"  {dataset:12} {best_acc:.2f} ± {best_std:.2f}  ({best_config})")

    # prop_first variants on heterophilous
    print("\n" + "=" * 60)
    print("prop_first variants on heterophilous datasets")
    print("=" * 60)
    pf_configs = ["prop_first", "prop_first_gfrw", "prop_first_sum", "prop_first_rmsprop", "prop_first_ense",
                  "prop_first_gfrw_rmsprop", "prop_first_gfrw_dropout08", "prop_first_gfrw_layers3",
                  "prop_first_sum_rmsprop", "prop_first_layers3", "prop_first_gfrw_layers3",
                  "prop_first_ense_rmsprop", "prop_first_sum_rw", "prop_first_dropout07",
                  "prop_first_patience300", "heterophily_tfe_patience300", "prop_first_sum_rw_rmsprop",
                  "prop_first_gfrw_dropout07",
                  "gfrw", "gfrw_rmsprop", "gfrw_dropout08", "gfrw_layers3", "gfrw_dropout08_layers3",
                  "gfrw_patience400", "heterophily_tfe", "heterophily_tfe_dropout06", "heterophily_tfe_layers3"]
    for dataset in ["chameleon", "squirrel", "texas", "cornell", "wisconsin",
                    "roman-empire", "amazon-ratings", "minesweeper", "tolokers", "questions"]:
        best_pf = [(c, best_per_config[(dataset, c)]) for c in pf_configs if (dataset, c) in best_per_config]
        if best_pf:
            best_pf.sort(key=lambda x: -x[1][0])
            print(f"  {dataset}: " + ", ".join(f"{c}={acc:.1f}" for c, (acc, _) in best_pf[:5]))
        else:
            print(f"  {dataset}: (no data)")

    return table

if __name__ == "__main__":
    main()
