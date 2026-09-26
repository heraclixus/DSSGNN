#!/usr/bin/env python3
"""
Compare persistent GOOD failure cases across multiple bundles/models.

Example:
python scripts/compare_good_failure_overlap.py \
  --case webkb_gcn=results_local/good_density/goodwebkb-...-gcn_nodepred:gcn \
  --case webkb_dss=results_local/good_density/goodwebkb-...-nodepred:gcn_dssres_gnnsafepp
"""

from __future__ import annotations

import argparse
import csv
import sys
from itertools import combinations
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_good_failure_cases import (
    aggregate_predictions,
    choose_model,
    compute_node_diagnostics,
    load_graph_context,
    load_node_predictions,
)


def parse_case_spec(spec: str):
    if "=" not in spec:
        raise ValueError(f"Invalid --case {spec!r}. Expected name=bundle[:model].")
    name, rest = spec.split("=", 1)
    if ":" in rest:
        bundle, model = rest.rsplit(":", 1)
    else:
        bundle, model = rest, None
    return {"name": name, "bundle": Path(bundle).resolve(), "model": model}


def write_rows(path: Path, fieldnames, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Compare persistent GOOD failure cases across models/bundles.")
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        help="Case spec: name=bundle[:model]. Repeat for multiple models.",
    )
    parser.add_argument("--split", default="test", choices=["id_test", "test"])
    parser.add_argument("--error-threshold", type=float, default=2 / 3)
    parser.add_argument("--output-dir", default=None, help="Optional output directory. Defaults to the first bundle.")
    args = parser.parse_args()

    cases = [parse_case_spec(spec) for spec in args.case]
    if len(cases) < 2:
        raise ValueError("Pass at least two --case entries.")

    first_metadata = None
    graph_ctx = None
    persistent_sets = {}
    chosen_models = {}

    for case in cases:
        metadata, rows = load_node_predictions(case["bundle"])
        model = choose_model(rows, case["model"])
        aggregates = aggregate_predictions(rows, model=model, split=args.split)
        if first_metadata is None:
            first_metadata = metadata
            graph_ctx = load_graph_context(metadata)
        else:
            if metadata["config_path"] != first_metadata["config_path"]:
                raise ValueError(
                    "All compared bundles must come from the same GOOD dataset/config. "
                    f"Found {first_metadata['config_path']} and {metadata['config_path']}."
                )
        chosen_models[case["name"]] = model
        persistent_sets[case["name"]] = {
            node_idx: row for node_idx, row in aggregates.items() if row["error_rate"] >= args.error_threshold
        }

    output_dir = Path(args.output_dir).resolve() if args.output_dir else cases[0]["bundle"]
    output_dir.mkdir(parents=True, exist_ok=True)

    pairwise_rows = []
    for case_a, case_b in combinations(cases, 2):
        name_a = case_a["name"]
        name_b = case_b["name"]
        set_a = set(persistent_sets[name_a])
        set_b = set(persistent_sets[name_b])
        intersection = set_a & set_b
        union = set_a | set_b
        pairwise_rows.append(
            {
                "case_a": name_a,
                "model_a": chosen_models[name_a],
                "case_b": name_b,
                "model_b": chosen_models[name_b],
                "size_a": len(set_a),
                "size_b": len(set_b),
                "intersection": len(intersection),
                "union": len(union),
                "jaccard": len(intersection) / max(len(union), 1),
                "overlap_frac_a": len(intersection) / max(len(set_a), 1),
                "overlap_frac_b": len(intersection) / max(len(set_b), 1),
            }
        )
    pairwise_rows.sort(key=lambda row: row["jaccard"], reverse=True)
    pairwise_path = output_dir / f"failure_overlap_pairwise_{args.split}.csv"
    write_rows(
        pairwise_path,
        [
            "case_a",
            "model_a",
            "case_b",
            "model_b",
            "size_a",
            "size_b",
            "intersection",
            "union",
            "jaccard",
            "overlap_frac_a",
            "overlap_frac_b",
        ],
        pairwise_rows,
    )

    union_nodes = sorted({node_idx for nodes in persistent_sets.values() for node_idx in nodes})
    diagnostics = compute_node_diagnostics(graph_ctx, union_nodes)
    membership_rows = []
    for node_idx in union_nodes:
        row = {
            "node_idx": node_idx,
            "true_label": int(graph_ctx["labels"][node_idx]),
            "domain_id": int(graph_ctx["domain_id"][node_idx]) if graph_ctx["domain_id"] is not None else None,
            "degree": diagnostics[node_idx]["degree"],
            "same_label_neighbor_frac": diagnostics[node_idx]["same_label_neighbor_frac"],
            "train_same_label_neighbor_frac": diagnostics[node_idx]["train_same_label_neighbor_frac"],
            "num_cases": 0,
            "cases": "",
        }
        present_cases = []
        for case in cases:
            name = case["name"]
            case_row = persistent_sets[name].get(node_idx)
            row[f"{name}_member"] = int(case_row is not None)
            row[f"{name}_pred"] = case_row["majority_pred_label"] if case_row is not None else None
            row[f"{name}_error_rate"] = case_row["error_rate"] if case_row is not None else None
            row[f"{name}_margin"] = case_row["mean_margin"] if case_row is not None else None
            row[f"{name}_chaos"] = case_row.get("mean_chaos") if case_row is not None else None
            if case_row is not None:
                present_cases.append(name)
        row["num_cases"] = len(present_cases)
        row["cases"] = ",".join(present_cases)
        membership_rows.append(row)

    membership_rows.sort(key=lambda row: (row["num_cases"], row["degree"]), reverse=True)
    membership_fieldnames = [
        "node_idx",
        "true_label",
        "domain_id",
        "degree",
        "same_label_neighbor_frac",
        "train_same_label_neighbor_frac",
        "num_cases",
        "cases",
    ]
    for case in cases:
        name = case["name"]
        membership_fieldnames.extend(
            [
                f"{name}_member",
                f"{name}_pred",
                f"{name}_error_rate",
                f"{name}_margin",
                f"{name}_chaos",
            ]
        )
    membership_path = output_dir / f"failure_overlap_nodes_{args.split}.csv"
    write_rows(membership_path, membership_fieldnames, membership_rows)

    print(f"Wrote: {pairwise_path}")
    print(f"Wrote: {membership_path}")
    print("\nPersistent failure sizes:")
    for case in cases:
        name = case["name"]
        print(f"- {name} ({chosen_models[name]}): {len(persistent_sets[name])}")
    print("\nTop pairwise overlaps:")
    for row in pairwise_rows[: min(10, len(pairwise_rows))]:
        print(
            f"- {row['case_a']} vs {row['case_b']}: "
            f"intersection={row['intersection']} jaccard={row['jaccard']:.3f} "
            f"overlap=({row['overlap_frac_a']:.3f}, {row['overlap_frac_b']:.3f})"
        )
    shared_all = [row for row in membership_rows if row["num_cases"] == len(cases)]
    print(f"\nNodes shared by all cases: {len(shared_all)}")
    for row in shared_all[: min(15, len(shared_all))]:
        print(
            f"node={row['node_idx']} true={row['true_label']} degree={row['degree']} "
            f"cases={row['cases']}"
        )


if __name__ == "__main__":
    main()
