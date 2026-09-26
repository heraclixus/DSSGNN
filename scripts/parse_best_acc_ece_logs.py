#!/usr/bin/env python3
"""
Parse best_acc_ece SLURM logs and generate Table 2 (Acc + ECE).

Usage:
  python scripts/parse_best_acc_ece_logs.py                    # print summary + LaTeX
  python scripts/parse_best_acc_ece_logs.py --update-table     # write tex/table2.tex
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict

LOG_DIR = Path(__file__).parent.parent / "slurm" / "logs"
DATASETS_ORDER = [
    "cora", "citeseer", "pubmed", "texas", "cornell", "wisconsin",
    "chameleon", "squirrel", "cora-full", "cs", "physics",
    "roman-empire", "amazon-ratings", "minesweeper", "tolokers", "questions",
]
MODELS = ["dssgnn", "tfe", "gduq"]


def parse_log(path: Path) -> list[dict]:
    """Extract RESULT lines from a log file. Supports both old (acc,ece) and new (acc,ece,mce,brier) format."""
    results = []
    with open(path) as f:
        for line in f:
            # New format with mce and brier
            m = re.match(
                r"RESULT dataset=(\S+) model=(\S+) config=\S+ "
                r"acc_mean=([\d.]+) acc_std=([\d.]+) "
                r"ece_mean=([\d.]+) ece_std=([\d.]+) "
                r"mce_mean=([\d.]+) mce_std=([\d.]+) "
                r"brier_mean=([\d.]+) brier_std=([\d.]+)",
                line.strip(),
            )
            if m:
                results.append({
                    "dataset": m.group(1),
                    "model": m.group(2),
                    "acc_mean": float(m.group(3)),
                    "acc_std": float(m.group(4)),
                    "ece_mean": float(m.group(5)),
                    "ece_std": float(m.group(6)),
                    "mce_mean": float(m.group(7)),
                    "mce_std": float(m.group(8)),
                    "brier_mean": float(m.group(9)),
                    "brier_std": float(m.group(10)),
                })
                continue
            # Old format (acc, ece only)
            m = re.match(
                r"RESULT dataset=(\S+) model=(\S+) config=\S+ "
                r"acc_mean=([\d.]+) acc_std=([\d.]+) "
                r"ece_mean=([\d.]+) ece_std=([\d.]+)",
                line.strip(),
            )
            if m:
                results.append({
                    "dataset": m.group(1),
                    "model": m.group(2),
                    "acc_mean": float(m.group(3)),
                    "acc_std": float(m.group(4)),
                    "ece_mean": float(m.group(5)),
                    "ece_std": float(m.group(6)),
                    "mce_mean": None,
                    "mce_std": None,
                    "brier_mean": None,
                    "brier_std": None,
                })
    return results


def collect_results() -> dict:
    """Collect all results from best_acc_ece logs."""
    data = defaultdict(dict)  # (dataset, model) -> {acc_mean, acc_std, ece_mean, ece_std}
    for path in sorted(LOG_DIR.glob("best_acc_ece_*.out")):
        for r in parse_log(path):
            key = (r["dataset"], r["model"])
            if key not in data or r["acc_mean"] > data[key]["acc_mean"]:
                data[key] = r
    return data


def format_cell(acc_mean, acc_std, ece_mean, ece_std, bold=False, mce_mean=None, mce_std=None, brier_mean=None, brier_std=None) -> str:
    """Format as 'acc ± std (ECE ± std)' for table2."""
    acc = f"{acc_mean:.2f} $\\pm$ {acc_std:.2f}"
    ece = f"{ece_mean:.3f} $\\pm$ {ece_std:.3f}"
    s = f"{acc} ({ece})"
    return f"\\textbf{{{s}}}" if bold else s


def generate_latex(data: dict, full_calibration: bool = False) -> str:
    """Generate LaTeX for Table 2."""
    lines = [
        "% Table 2: Node classification test accuracy (%) and ECE. Mean ± std over 10 runs.",
        "% Best accuracy per dataset in bold.",
        "%",
        "% Usage: \\input{tex/table2} or \\include{tex/table2}",
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Node classification: test accuracy (\\%) and ECE. Mean $\\pm$ standard deviation over 10 runs. Best accuracy per dataset in \\textbf{bold}.}",
        "\\label{tab:acc_ece}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Dataset    & DS-GNN        & TFE-GNN       & G-$\\Delta$UQ \\\\",
        "\\midrule",
    ]
    for ds in DATASETS_ORDER:
        name = ds.replace("-", " ").title().replace(" ", "-")  # roman-empire -> Roman-Empire
        if ds == "cora-full":
            name = "Cora-Full"
        row = [name]
        best_acc = 0
        for model in MODELS:
            key = (ds, model)
            if key in data:
                r = data[key]
                best_acc = max(best_acc, r["acc_mean"])
        for model in MODELS:
            key = (ds, model)
            if key in data:
                r = data[key]
                bold = r["acc_mean"] >= best_acc - 0.01
                row.append(format_cell(
                    r["acc_mean"], r["acc_std"], r["ece_mean"], r["ece_std"], bold,
                    r.get("mce_mean"), r.get("mce_std"), r.get("brier_mean"), r.get("brier_std")
                ))
            else:
                row.append("--")
        lines.append(" & ".join(row) + " \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def generate_calibration_latex(data: dict) -> str:
    """Generate LaTeX table with Acc, ECE, MCE, Brier."""
    lines = [
        "% Table: Node classification - Acc, ECE, MCE, Brier. Mean ± std over 10 runs.",
        "% Usage: \\input{tex/table2_calibration}",
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Node classification: Acc (\\%), ECE, MCE, Brier. Mean $\\pm$ std.}",
        "\\label{tab:calibration}",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Dataset & Model & Acc (\\%) & ECE & MCE & Brier \\\\",
        "\\midrule",
    ]
    for ds in DATASETS_ORDER:
        name = ds.replace("-", " ").title().replace(" ", "-")
        if ds == "cora-full":
            name = "Cora-Full"
        for model in MODELS:
            key = (ds, model)
            if key in data:
                r = data[key]
                acc = f"{r['acc_mean']:.2f} $\\pm$ {r['acc_std']:.2f}"
                ece = f"{r['ece_mean']:.3f} $\\pm$ {r['ece_std']:.3f}"
                mce = f"{r.get('mce_mean', 0):.3f} $\\pm$ {r.get('mce_std', 0):.3f}" if r.get("mce_mean") is not None else "--"
                brier = f"{r.get('brier_mean', 0):.4f} $\\pm$ {r.get('brier_std', 0):.4f}" if r.get("brier_mean") is not None else "--"
                lines.append(f"{name} & {model} & {acc} & {ece} & {mce} & {brier} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-table", action="store_true", help="Write tex/table2.tex")
    parser.add_argument("--update-calibration", action="store_true", help="Write tex/table2_calibration.tex")
    args = parser.parse_args()

    data = collect_results()
    if not data:
        print("No RESULT lines found in best_acc_ece_*.out logs.")
        print("Run: scripts/run_best_configs_acc_ece_local.sh")
        return

    print(f"Parsed {len(data)} (dataset, model) results")
    for ds in DATASETS_ORDER:
        for m in MODELS:
            if (ds, m) in data:
                r = data[(ds, m)]
                s = f"  {ds:15} {m:6}  acc={r['acc_mean']:.2f}±{r['acc_std']:.2f}  ECE={r['ece_mean']:.3f}±{r['ece_std']:.3f}"
                if r.get("mce_mean") is not None:
                    s += f"  MCE={r['mce_mean']:.3f}±{r['mce_std']:.3f}  Brier={r['brier_mean']:.4f}±{r['brier_std']:.4f}"
                print(s)

    latex = generate_latex(data)
    if args.update_table:
        out = Path(__file__).parent.parent / "tex" / "table2.tex"
        out.write_text(latex)
        print(f"\nWrote {out}")
    if args.update_calibration:
        cal_latex = generate_calibration_latex(data)
        out = Path(__file__).parent.parent / "tex" / "table2_calibration.tex"
        out.write_text(cal_latex)
        print(f"Wrote {out}")
    if not args.update_table and not args.update_calibration:
        print("\n" + "=" * 60 + "\nLaTeX:\n" + latex)


if __name__ == "__main__":
    main()
