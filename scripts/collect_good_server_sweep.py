#!/usr/bin/env python3
"""
Collect GOOD sweep bundles into a sortable CSV.

The current collector is classification-first: each bundle is reduced to the row
with the highest shifted-test accuracy (`test_acc_mean`), with `id_test_acc_mean`
and `score_auroc_mean` used as tie-breaker context.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_summary_rows(path: Path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def best_row(rows: list[dict]) -> dict:
    def key(row):
        return (
            float(row["test_acc_mean"]),
            float(row["id_test_acc_mean"]),
            float(row["score_auroc_mean"]),
        )

    return max(rows, key=key)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect GOOD server sweep results")
    parser.add_argument("--tag-prefix", default="goodphase1", help="Bundle tag prefix to include.")
    parser.add_argument(
        "--results-root",
        default=str(REPO_ROOT / "results_local" / "good_density"),
        help="Root directory containing GOOD result bundles.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path. Defaults to results_local/good_density/<tag-prefix>_summary.csv",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    out_path = Path(args.out) if args.out else results_root / f"{args.tag_prefix}_summary.csv"

    rows_out = []
    for bundle in sorted(results_root.iterdir()):
        if not bundle.is_dir() or args.tag_prefix not in bundle.name:
            continue
        summary_path = bundle / "summary.csv"
        if not summary_path.exists():
            continue
        summary_rows = load_summary_rows(summary_path)
        if not summary_rows:
            continue
        best = best_row(summary_rows)
        job_meta = {}
        job_path = bundle / "sweep_job.json"
        if job_path.exists():
            job_meta = json.loads(job_path.read_text())

        record = {
            "bundle": bundle.name,
            "target": job_meta.get("target", ""),
            "dataset": job_meta.get("dataset", ""),
            "model": job_meta.get("model", best.get("model", "")),
            "tag": job_meta.get("tag", ""),
            "score_type": best["score_type"],
            "test_acc_mean": best["test_acc_mean"],
            "id_test_acc_mean": best["id_test_acc_mean"],
            "score_auroc_mean": best["score_auroc_mean"],
            "score_aupr_mean": best["score_aupr_mean"],
            "score_fpr95_mean": best["score_fpr95_mean"],
            "val_loss_mean": best["val_loss_mean"],
            "runs": best["runs"],
            "epochs": job_meta.get("epochs", ""),
            "patience": job_meta.get("patience", ""),
            "hidden": job_meta.get("hidden", ""),
            "layers": job_meta.get("layers", ""),
            "K_lp": job_meta.get("K_lp", ""),
            "K_hp": job_meta.get("K_hp", ""),
            "P": job_meta.get("P", ""),
            "quadrature_nodes": job_meta.get("quadrature_nodes", ""),
            "lr": job_meta.get("lr", ""),
            "weight_decay": job_meta.get("weight_decay", ""),
            "lambda_reg": job_meta.get("lambda_reg", ""),
            "gnnsafe_K": job_meta.get("gnnsafe_K", ""),
            "gnnsafe_alpha": job_meta.get("gnnsafe_alpha", ""),
            "lamda": job_meta.get("lamda", ""),
            "m_in": job_meta.get("m_in", ""),
            "m_out": job_meta.get("m_out", ""),
            "warmup_base_epochs": job_meta.get("warmup_base_epochs", ""),
            "residual_scale_init": job_meta.get("residual_scale_init", ""),
            "tar_inner_steps": job_meta.get("tar_inner_steps", ""),
            "tar_beta": job_meta.get("tar_beta", ""),
            "tar_eta": job_meta.get("tar_eta", ""),
            "tar_eta0": job_meta.get("tar_eta0", ""),
            "tar_eta_lp": job_meta.get("tar_eta_lp", ""),
            "tar_eta_hp": job_meta.get("tar_eta_hp", ""),
            "tar_alpha_lp": job_meta.get("tar_alpha_lp", ""),
            "tar_alpha_partial": job_meta.get("tar_alpha_partial", ""),
            "tar_lambda_edge": job_meta.get("tar_lambda_edge", ""),
            "tar_edge_onset_epoch": job_meta.get("tar_edge_onset_epoch", ""),
            "tar_gate_bias": job_meta.get("tar_gate_bias", ""),
            "tar_gate_lp": job_meta.get("tar_gate_lp", ""),
            "tar_gate_boundary": job_meta.get("tar_gate_boundary", ""),
            "tar_step_size": job_meta.get("tar_step_size", ""),
            "use_random_gates": job_meta.get("use_random_gates", ""),
            "shared_input_lift": job_meta.get("shared_input_lift", ""),
            "family": job_meta.get("family", ""),
            "note": job_meta.get("note", ""),
        }
        rows_out.append(record)

    fieldnames = sorted({key for row in rows_out for key in row.keys()})
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_out:
            writer.writerow(row)

    print(f"Wrote {len(rows_out)} GOOD sweep rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
