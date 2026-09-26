#!/usr/bin/env python3
"""
Parse DSS-GNN variants sweep logs and generate summary tables.

Usage:
  python scripts/parse_dssgnn_variants_logs.py                    # print summary
  python scripts/parse_dssgnn_variants_logs.py --update-table      # write tex/table_variants.tex
  python scripts/parse_dssgnn_variants_logs.py --csv               # write csv summary
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
VARIANTS_ORDER = ["dssgnn", "dssgnn_sg", "dssgnn_sf", "dssgnn_gduq", "dssgnn_gduq_sg", "dssgnn_gduq_sf"]
def _get_result(data: dict, ds: str, v: str):
    """Look up result; support old logs with dsgnn_* variant names."""
    key = (ds, v)
    if key in data:
        return data[key]
    key_legacy = (ds, v.replace("dssgnn", "dsgnn"))
    return data.get(key_legacy)


VARIANT_LABELS = {
    "dssgnn": "S",
    "dssgnn_sg": "S+G",
    "dssgnn_sf": "S+F",
    "dssgnn_gduq": "S+A",
    "dssgnn_gduq_sg": "S+G+A",
    "dssgnn_gduq_sf": "S+F+A",
    # Backward compat with old logs
    "dsgnn": "S",
    "dsgnn_sg": "S+G",
    "dsgnn_sf": "S+F",
    "dsgnn_gduq": "S+A",
    "dsgnn_gduq_sg": "S+G+A",
    "dsgnn_gduq_sf": "S+F+A",
}


def parse_log(path: Path) -> list[dict]:
    """Extract RESULT lines from dssgnn_variants logs."""
    results = []
    with open(path) as f:
        for line in f:
            # Full format with mce, brier
            m = re.match(
                r"RESULT dataset=(\S+) variant=(\S+) config=\S+ "
                r"acc_mean=([\d.]+) acc_std=([\d.]+) "
                r"ece_mean=([\d.]+) ece_std=([\d.]+) "
                r"mce_mean=([\d.]+) mce_std=([\d.]+) "
                r"brier_mean=([\d.]+) brier_std=([\d.]+)",
                line.strip(),
            )
            if m:
                results.append({
                    "dataset": m.group(1),
                    "variant": m.group(2),
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
            # Acc, ECE only
            m = re.match(
                r"RESULT dataset=(\S+) variant=(\S+) config=\S+ "
                r"acc_mean=([\d.]+) acc_std=([\d.]+) "
                r"ece_mean=([\d.]+) ece_std=([\d.]+)",
                line.strip(),
            )
            if m:
                results.append({
                    "dataset": m.group(1),
                    "variant": m.group(2),
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
    """Collect all results from dssgnn_variants logs."""
    data = defaultdict(dict)
    for path in sorted(LOG_DIR.glob("dssgnn_variants_*.out")) + sorted(LOG_DIR.glob("dsgnn_variants_*.out")):
        for r in parse_log(path):
            key = (r["dataset"], r["variant"])
            if key not in data or r["acc_mean"] > data[key]["acc_mean"]:
                data[key] = r
    return data


def dataset_display_name(ds: str) -> str:
    if ds == "cora-full":
        return "Cora-Full"
    return ds.replace("-", " ").title().replace(" ", "-")


def generate_latex(data: dict) -> str:
    """Generate LaTeX table: datasets x variants (Acc, ECE)."""
    lines = [
        "% DSS-GNN variants sweep: Acc (%), ECE. Mean ± std over 10 runs.",
        "% Variants: S (base), S+G (random gates), S+F (random filter), S+A (anchor), S+G+A, S+F+A.",
        "% Requires: \\usepackage{booktabs,graphicx}",
        "%",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{DSS-GNN variants: test accuracy (\\%) and ECE. Mean $\\pm$ std. Best acc per dataset in \\textbf{bold}.}",
        "\\label{tab:dssgnn_variants}",
        "\\scriptsize",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{l" + "cc" * len(VARIANTS_ORDER) + "}",
        "\\toprule",
        "Dataset & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{VARIANT_LABELS.get(v, v)}}}" for v in VARIANTS_ORDER) + " \\\\",
        " & " + " & ".join(["Acc & ECE"] * len(VARIANTS_ORDER)) + " \\\\",
        "\\midrule",
    ]
    for ds in DATASETS_ORDER:
        name = dataset_display_name(ds)
        best_acc = 0
        for v in VARIANTS_ORDER:
            r = _get_result(data, ds, v)
            if r:
                best_acc = max(best_acc, r["acc_mean"])
        row = [name]
        for v in VARIANTS_ORDER:
            r = _get_result(data, ds, v)
            if r:
                bold = r["acc_mean"] >= best_acc - 0.01
                acc = f"{r['acc_mean']:.2f} $\\pm$ {r['acc_std']:.2f}"
                ece = f"{r['ece_mean']:.3f} $\\pm$ {r['ece_std']:.3f}"
                cell = f"\\textbf{{{acc}}} ({ece})" if bold else f"{acc} ({ece})"
                row.append(cell)
            else:
                row.append("--")
        lines.append(" & ".join(row) + " \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-table", action="store_true", help="Write tex/table_variants.tex")
    parser.add_argument("--csv", action="store_true", help="Write csv summary to stdout")
    args = parser.parse_args()

    data = collect_results()
    if not data:
        print("No RESULT lines found in dssgnn_variants_*.out logs.")
        print("Run: scripts/run_dssgnn_variants_local.sh")
        return

    print(f"Parsed {len(data)} (dataset, variant) results")
    for ds in DATASETS_ORDER:
        for v in VARIANTS_ORDER:
            r = _get_result(data, ds, v)
            if r:
                s = f"  {ds:15} {v:12}  acc={r['acc_mean']:.2f}±{r['acc_std']:.2f}  ECE={r['ece_mean']:.3f}±{r['ece_std']:.3f}"
                if r.get("mce_mean") is not None:
                    s += f"  MCE={r['mce_mean']:.3f}  Brier={r['brier_mean']:.4f}"
                print(s)

    if args.update_table:
        latex = generate_latex(data)
        out = Path(__file__).parent.parent / "tex" / "table_variants.tex"
        out.write_text(latex)
        print(f"\nWrote {out}")

    if args.csv:
        print("\n" + "dataset,variant,acc_mean,acc_std,ece_mean,ece_std,mce_mean,brier_mean")
        for ds in DATASETS_ORDER:
            for v in VARIANTS_ORDER:
                r = _get_result(data, ds, v)
                if r:
                    mce = r.get("mce_mean") or ""
                    brier = r.get("brier_mean") or ""
                    print(f"{ds},{v},{r['acc_mean']:.2f},{r['acc_std']:.2f},{r['ece_mean']:.4f},{r['ece_std']:.4f},{mce},{brier}")

    if not args.update_table and not args.csv:
        print("\n" + "=" * 60 + "\nLaTeX preview:\n" + generate_latex(data))


if __name__ == "__main__":
    main()
