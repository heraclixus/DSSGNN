#!/usr/bin/env python3
"""
Parse OOD experiment results from results/*.csv into a LaTeX table or summary CSV.

Results are written by gnnsafe_ood/data_utils.save_result() in format:
  method_backbone
  OOD Test 1 AUROC: X.XX AUPR: X.XX FPR: X.XX
  IND Test Score: X.XX

Usage:
  python scripts/parse_ood_results.py                    # print summary
  python scripts/parse_ood_results.py --latex            # LaTeX table
  python scripts/parse_ood_results.py --csv out.csv      # CSV output
"""

import argparse
import re
from pathlib import Path


def parse_results_dir(results_dir="results"):
    """Parse all CSV files in results dir. Returns dict: (dataset, ood_type) -> {method: (auroc, aupr, fpr, ind)}"""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return {}

    data = {}
    for f in sorted(results_dir.glob("*.csv")):
        stem = f.stem
        if "-" in stem:
            dataset, ood_type = stem.rsplit("-", 1)
        else:
            dataset, ood_type = stem, None

        with open(f) as fp:
            content = fp.read()

        key = (dataset, ood_type)
        if key not in data:
            data[key] = {}

        lines = content.strip().split("\n") if content.strip() else []
        i = 0
        while i < len(lines):
            line = lines[i]
            if not line.strip() or "AUROC" in line or "IND" in line or "Score" in line:
                i += 1
                continue
            method = line.strip()
            auroc = aupr = fpr = ind = None
            for j in range(i + 1, min(i + 4, len(lines))):
                ln = lines[j]
                if "AUROC" in ln:
                    m = re.search(r"AUROC:\s*([\d.]+)", ln)
                    if m:
                        auroc = float(m.group(1))
                if "AUPR" in ln:
                    m = re.search(r"AUPR:\s*([\d.]+)", ln)
                    if m:
                        aupr = float(m.group(1))
                if "FPR" in ln:
                    m = re.search(r"FPR:\s*([\d.]+)", ln)
                    if m:
                        fpr = float(m.group(1))
                if "IND" in ln or "Score" in ln:
                    m = re.search(r"Score:\s*([\d.]+)", ln)
                    if m:
                        ind = float(m.group(1))
            if method and (auroc is not None or ind is not None):
                data[key][method] = (auroc, aupr, fpr, ind)
            i += 1

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--latex", action="store_true")
    parser.add_argument("--csv", type=str, help="Output CSV path")
    args = parser.parse_args()

    data = parse_results_dir(args.results_dir)
    if not data:
        print("No results found.")
        return

    for (dataset, ood_type), methods in sorted(data.items()):
        label = f"{dataset}-{ood_type}" if ood_type else dataset
        print(f"\n=== {label} ===")
        for method, (auroc, aupr, fpr, ind) in sorted(methods.items()):
            print(f"  {method}: AUROC={auroc:.2f} AUPR={aupr:.2f} FPR95={fpr:.2f} IND={ind:.2f}")

    if args.csv:
        rows = []
        for (dataset, ood_type), methods in sorted(data.items()):
            for method, (auroc, aupr, fpr, ind) in sorted(methods.items()):
                rows.append({
                    "dataset": dataset,
                    "ood_type": ood_type or "",
                    "method": method,
                    "auroc": auroc,
                    "aupr": aupr,
                    "fpr95": fpr,
                    "ind": ind,
                })
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["dataset", "ood_type", "method", "auroc", "aupr", "fpr95", "ind"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
