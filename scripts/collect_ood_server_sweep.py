#!/usr/bin/env python3
"""
Collect DSS OOD server-sweep results into a single sortable CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import chain
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def flatten_params(prefix: str, value, out: dict) -> None:
    if isinstance(value, dict):
        for key, sub_value in value.items():
            flatten_params(f"{prefix}{key}_", sub_value, out)
    else:
        out[prefix[:-1]] = value


def get_first_present(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        if key in row:
            return row[key]
    available = ", ".join(sorted(row.keys()))
    requested = ", ".join(keys)
    raise KeyError(f"None of the requested columns were found: {requested}. Available columns: {available}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect server OOD sweep results")
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT / "results_local" / "ood_density",
        help="Sweep result root directory.",
    )
    parser.add_argument("--tag-prefix", type=str, default="serversearch")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results_local" / "ood_density" / "serversearch_summary.csv",
    )
    args = parser.parse_args()

    rows = []
    pattern = f"*{args.tag_prefix}*"
    for bundle in sorted(args.root.glob(pattern)):
        job_path = bundle / "sweep_job.json"
        summary_path = bundle / "summary.csv"
        if not job_path.exists() or not summary_path.exists():
            continue

        job = json.loads(job_path.read_text())
        with summary_path.open() as handle:
            summary_rows = list(csv.DictReader(handle))
        if not summary_rows:
            continue

        if len(summary_rows) != 1:
            raise ValueError(f"Expected exactly one summary row in {summary_path}, found {len(summary_rows)}")
        summary = summary_rows[0]

        row = {
            "bundle": bundle.name,
            "target_id": job["target_id"],
            "dataset": job["dataset"],
            "ood_type": job["ood_type"],
            "config_name": job["config_name"],
            "family": job["family"],
            "runs": job["runs"],
            "epochs": job["epochs"],
            "job_index": job["job_index"],
            "combo_index": job["combo_index"],
            "score_auroc_mean": get_first_present(summary, "score_auroc_mean", "auroc_mean"),
            "score_auroc_std": get_first_present(summary, "score_auroc_std", "auroc_std"),
            "score_aupr_mean": get_first_present(summary, "score_aupr_mean", "aupr_mean"),
            "score_fpr95_mean": get_first_present(summary, "score_fpr95_mean", "fpr95_mean"),
            "ind_acc_mean": get_first_present(summary, "ind_acc_mean"),
            "valid_loss_mean": get_first_present(summary, "valid_loss_mean"),
            "note": job["note"],
        }
        flatten_params("param_", job["params"], row)
        rows.append(row)

    rows.sort(
        key=lambda row: (
            row["target_id"],
            -float(row["score_auroc_mean"]),
            float(row["score_fpr95_mean"]),
            -float(row["ind_acc_mean"]),
        )
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        args.out.write_text("")
        print(f"No matching bundles found under {args.root} for tag prefix '{args.tag_prefix}'.")
        return

    fieldnames = []
    seen = set()
    for key in chain.from_iterable(row.keys() for row in rows):
        if key not in seen:
            seen.add(key)
            fieldnames.append(key)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
