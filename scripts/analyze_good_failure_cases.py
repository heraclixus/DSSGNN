#!/usr/bin/env python3
"""
Analyze persistent node-level failure cases from GOOD result bundles.

This script expects a bundle directory containing:

- metadata.json
- node_predictions_long.csv

It aggregates repeated predictions across seeds, computes simple graph-side
diagnostics for the chosen split, and writes a node-level summary CSV that can
be inspected directly or joined with visualization code later.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_dssgnn_good import ensure_good_dataset_registered, load_good_config


NODE_SCORE_FIELDS = (
    "msp",
    "energy",
    "entropy",
    "chaos",
    "pred_entropy",
    "mutual_info",
    "energy_prop",
    "pred_entropy_prop",
    "mutual_info_prop",
    "graph_ebm",
)


def _optional_tensor_numpy(data, attr_name):
    if not hasattr(data, attr_name):
        return None
    value = getattr(data, attr_name)
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return None


def _safe_float(value):
    if value in ("", None):
        return None
    return float(value)


def _safe_int(value):
    if value in ("", None):
        return None
    return int(value)


def load_node_predictions(bundle_dir: Path):
    metadata = json.loads((bundle_dir / "metadata.json").read_text())
    node_predictions_path = bundle_dir / "node_predictions_long.csv"
    if not node_predictions_path.exists():
        raise FileNotFoundError(
            f"Bundle {bundle_dir} does not contain node_predictions_long.csv. "
            "Rerun the bundle with the updated local_good_density_compare.py export."
        )
    with node_predictions_path.open() as handle:
        rows = list(csv.DictReader(handle))
    return metadata, rows


def choose_model(rows, requested_model: str | None):
    models = sorted({row["model"] for row in rows})
    if requested_model is not None:
        if requested_model not in models:
            raise ValueError(f"Requested model {requested_model!r} not present. Available: {models}")
        return requested_model
    if len(models) != 1:
        raise ValueError(f"Bundle contains multiple models {models}; pass --model.")
    return models[0]


def aggregate_predictions(rows, *, model: str, split: str):
    grouped = defaultdict(list)
    for row in rows:
        if row["model"] != model or row["split"] != split:
            continue
        grouped[int(row["node_idx"])].append(row)

    aggregates = {}
    for node_idx, node_rows in grouped.items():
        pred_counter = Counter(int(row["pred_label"]) for row in node_rows)
        error_count = sum(1 - int(row["correct"]) for row in node_rows)
        agg = {
            "node_idx": node_idx,
            "obs_count": len(node_rows),
            "error_count": error_count,
            "error_rate": error_count / max(len(node_rows), 1),
            "true_label": int(node_rows[0]["true_label"]),
            "majority_pred_label": pred_counter.most_common(1)[0][0],
            "majority_pred_count": pred_counter.most_common(1)[0][1],
            "mean_confidence": float(np.mean([float(row["confidence"]) for row in node_rows])),
            "mean_true_prob": float(np.mean([float(row["true_prob"]) for row in node_rows])),
            "mean_margin": float(np.mean([float(row["margin"]) for row in node_rows])),
            "env_id": _safe_int(node_rows[0]["env_id"]),
            "domain_id": _safe_int(node_rows[0]["domain_id"]),
        }
        for score_name in NODE_SCORE_FIELDS:
            values = [_safe_float(row.get(score_name)) for row in node_rows]
            values = [value for value in values if value is not None]
            agg[f"mean_{score_name}"] = float(np.mean(values)) if values else None
        aggregates[node_idx] = agg
    return aggregates


def load_graph_context(metadata):
    from GOOD.data import load_dataset

    config = load_good_config(
        config_path=metadata["config_path"],
        hidden=64,
        seed=metadata.get("seed", 123),
        generate=False,
    )
    ensure_good_dataset_registered(config.dataset.dataset_name)
    dataset = load_dataset(config.dataset.dataset_name, config)
    data = dataset.data

    labels = data.y.detach().cpu().numpy()
    if labels.ndim > 1:
        labels = labels.argmax(axis=1)
    edge_index = data.edge_index.detach().cpu().numpy()
    train_mask = data.train_mask.detach().cpu().numpy().astype(bool)
    env_id = _optional_tensor_numpy(data, "env_id")
    domain_id = _optional_tensor_numpy(data, "domain_id")

    num_nodes = int(data.x.shape[0])
    adjacency = sp.csr_matrix(
        (np.ones(edge_index.shape[1], dtype=np.float32), (edge_index[0], edge_index[1])),
        shape=(num_nodes, num_nodes),
    )
    return {
        "labels": labels,
        "train_mask": train_mask,
        "env_id": env_id,
        "domain_id": domain_id,
        "adjacency": adjacency,
        "num_nodes": num_nodes,
    }


def compute_node_diagnostics(graph_ctx, node_indices):
    adjacency = graph_ctx["adjacency"]
    labels = graph_ctx["labels"]
    train_mask = graph_ctx["train_mask"]
    domain_id = graph_ctx["domain_id"]

    diagnostics = {}
    for node_idx in node_indices:
        start = adjacency.indptr[node_idx]
        end = adjacency.indptr[node_idx + 1]
        neighbors = adjacency.indices[start:end]
        degree = int(neighbors.shape[0])
        if degree == 0:
            diagnostics[node_idx] = {
                "degree": 0,
                "same_label_neighbor_frac": None,
                "train_neighbor_frac": None,
                "train_same_label_neighbor_frac": None,
                "same_domain_neighbor_frac": None,
                "has_train_same_label_neighbor": 0,
            }
            continue

        neighbor_labels = labels[neighbors]
        same_label = neighbor_labels == labels[node_idx]
        train_neighbors = train_mask[neighbors]
        train_same_label = train_neighbors & same_label

        same_domain_frac = None
        if domain_id is not None:
            same_domain_frac = float(np.mean(domain_id[neighbors] == domain_id[node_idx]))

        diagnostics[node_idx] = {
            "degree": degree,
            "same_label_neighbor_frac": float(np.mean(same_label)),
            "train_neighbor_frac": float(np.mean(train_neighbors)),
            "train_same_label_neighbor_frac": float(np.mean(train_same_label)),
            "same_domain_neighbor_frac": same_domain_frac,
            "has_train_same_label_neighbor": int(train_same_label.any()),
        }
    return diagnostics


def write_failure_summary(path: Path, rows):
    fieldnames = [
        "node_idx",
        "obs_count",
        "error_count",
        "error_rate",
        "true_label",
        "majority_pred_label",
        "majority_pred_count",
        "env_id",
        "domain_id",
        "degree",
        "same_label_neighbor_frac",
        "train_neighbor_frac",
        "train_same_label_neighbor_frac",
        "same_domain_neighbor_frac",
        "has_train_same_label_neighbor",
        "mean_confidence",
        "mean_true_prob",
        "mean_margin",
    ] + [f"mean_{score_name}" for score_name in NODE_SCORE_FIELDS]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_rows(path: Path, fieldnames, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_confusion_summary(rows, *, error_threshold: float):
    persistent_rows = [row for row in rows if row["error_rate"] >= error_threshold]
    grouped = defaultdict(list)
    for row in persistent_rows:
        key = (row["true_label"], row["majority_pred_label"])
        grouped[key].append(row)

    summary = []
    for (true_label, pred_label), group_rows in grouped.items():
        summary.append(
            {
                "true_label": true_label,
                "pred_label": pred_label,
                "count": len(group_rows),
                "mean_error_rate": float(np.mean([row["error_rate"] for row in group_rows])),
                "mean_degree": float(np.mean([row["degree"] for row in group_rows])),
                "mean_margin": float(np.mean([row["mean_margin"] for row in group_rows])),
                "mean_confidence": float(np.mean([row["mean_confidence"] for row in group_rows])),
                "mean_chaos": float(
                    np.mean(
                        [
                            row["mean_chaos"]
                            for row in group_rows
                            if row.get("mean_chaos") is not None
                        ]
                    )
                )
                if any(row.get("mean_chaos") is not None for row in group_rows)
                else None,
            }
        )

    summary.sort(key=lambda row: (row["count"], row["mean_error_rate"]), reverse=True)
    return summary


def build_domain_summary(rows, *, error_threshold: float):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["domain_id"]].append(row)

    summary = []
    for domain_id, group_rows in grouped.items():
        persistent_rows = [row for row in group_rows if row["error_rate"] >= error_threshold]
        summary.append(
            {
                "domain_id": domain_id,
                "node_count": len(group_rows),
                "persistent_error_count": len(persistent_rows),
                "persistent_error_frac": len(persistent_rows) / max(len(group_rows), 1),
                "mean_error_rate": float(np.mean([row["error_rate"] for row in group_rows])),
                "mean_degree": float(np.mean([row["degree"] for row in group_rows])),
                "mean_same_label_neighbor_frac": float(
                    np.mean(
                        [
                            row["same_label_neighbor_frac"]
                            for row in group_rows
                            if row["same_label_neighbor_frac"] is not None
                        ]
                    )
                )
                if any(row["same_label_neighbor_frac"] is not None for row in group_rows)
                else None,
                "mean_train_same_label_neighbor_frac": float(
                    np.mean(
                        [
                            row["train_same_label_neighbor_frac"]
                            for row in group_rows
                            if row["train_same_label_neighbor_frac"] is not None
                        ]
                    )
                )
                if any(row["train_same_label_neighbor_frac"] is not None for row in group_rows)
                else None,
                "mean_margin": float(np.mean([row["mean_margin"] for row in group_rows])),
                "mean_confidence": float(np.mean([row["mean_confidence"] for row in group_rows])),
                "mean_chaos": float(
                    np.mean(
                        [
                            row["mean_chaos"]
                            for row in group_rows
                            if row.get("mean_chaos") is not None
                        ]
                    )
                )
                if any(row.get("mean_chaos") is not None for row in group_rows)
                else None,
            }
        )

    summary.sort(key=lambda row: (row["persistent_error_frac"], row["node_count"]), reverse=True)
    return summary


def build_domain_confusion_summary(rows, *, error_threshold: float):
    persistent_rows = [row for row in rows if row["error_rate"] >= error_threshold]
    grouped = defaultdict(list)
    for row in persistent_rows:
        key = (row["domain_id"], row["true_label"], row["majority_pred_label"])
        grouped[key].append(row)

    summary = []
    for (domain_id, true_label, pred_label), group_rows in grouped.items():
        summary.append(
            {
                "domain_id": domain_id,
                "true_label": true_label,
                "pred_label": pred_label,
                "count": len(group_rows),
                "mean_error_rate": float(np.mean([row["error_rate"] for row in group_rows])),
                "mean_margin": float(np.mean([row["mean_margin"] for row in group_rows])),
                "mean_confidence": float(np.mean([row["mean_confidence"] for row in group_rows])),
            }
        )

    summary.sort(key=lambda row: (row["count"], row["mean_error_rate"]), reverse=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze persistent GOOD node-level failure cases.")
    parser.add_argument("--bundle", required=True, help="Path to a GOOD result bundle directory.")
    parser.add_argument("--model", default=None, help="Model key within node_predictions_long.csv.")
    parser.add_argument("--split", default="test", choices=["id_test", "test"], help="Prediction split to analyze.")
    parser.add_argument("--error-threshold", type=float, default=2 / 3, help="Threshold for persistent-error reporting.")
    parser.add_argument("--top-k", type=int, default=20, help="How many highest-error nodes to print.")
    parser.add_argument("--output", default=None, help="Optional output CSV path.")
    args = parser.parse_args()

    bundle_dir = Path(args.bundle).resolve()
    metadata, rows = load_node_predictions(bundle_dir)
    model = choose_model(rows, args.model)

    aggregates = aggregate_predictions(rows, model=model, split=args.split)
    graph_ctx = load_graph_context(metadata)
    diagnostics = compute_node_diagnostics(graph_ctx, sorted(aggregates.keys()))

    merged_rows = []
    for node_idx, agg in aggregates.items():
        row = dict(agg)
        row.update(diagnostics[node_idx])
        merged_rows.append(row)

    merged_rows.sort(
        key=lambda row: (
            row["error_rate"],
            row.get("mean_chaos") if row.get("mean_chaos") is not None else float("-inf"),
            -row["mean_margin"],
        ),
        reverse=True,
    )

    output_path = Path(args.output) if args.output else bundle_dir / f"failure_case_summary_{model}_{args.split}.csv"
    write_failure_summary(output_path, merged_rows)

    confusion_rows = build_confusion_summary(merged_rows, error_threshold=args.error_threshold)
    confusion_path = bundle_dir / f"failure_case_confusion_{model}_{args.split}.csv"
    write_rows(
        confusion_path,
        ["true_label", "pred_label", "count", "mean_error_rate", "mean_degree", "mean_margin", "mean_confidence", "mean_chaos"],
        confusion_rows,
    )

    domain_rows = build_domain_summary(merged_rows, error_threshold=args.error_threshold)
    domain_path = bundle_dir / f"failure_case_domain_summary_{model}_{args.split}.csv"
    write_rows(
        domain_path,
        [
            "domain_id",
            "node_count",
            "persistent_error_count",
            "persistent_error_frac",
            "mean_error_rate",
            "mean_degree",
            "mean_same_label_neighbor_frac",
            "mean_train_same_label_neighbor_frac",
            "mean_margin",
            "mean_confidence",
            "mean_chaos",
        ],
        domain_rows,
    )

    domain_confusion_rows = build_domain_confusion_summary(merged_rows, error_threshold=args.error_threshold)
    domain_confusion_path = bundle_dir / f"failure_case_domain_confusion_{model}_{args.split}.csv"
    write_rows(
        domain_confusion_path,
        ["domain_id", "true_label", "pred_label", "count", "mean_error_rate", "mean_margin", "mean_confidence"],
        domain_confusion_rows,
    )

    persistent = [row for row in merged_rows if row["error_rate"] >= args.error_threshold]
    print(f"Bundle: {bundle_dir}")
    print(f"Model: {model}")
    print(f"Split: {args.split}")
    print(f"Nodes analyzed: {len(merged_rows)}")
    print(f"Persistent errors (threshold={args.error_threshold:.2f}): {len(persistent)}")
    print(f"Wrote: {output_path}")
    print(f"Wrote: {confusion_path}")
    print(f"Wrote: {domain_path}")
    print(f"Wrote: {domain_confusion_path}")
    print("\nTop persistent confusions:")
    for row in confusion_rows[: min(10, len(confusion_rows))]:
        print(
            f"true={row['true_label']} pred={row['pred_label']} count={row['count']} "
            f"deg={row['mean_degree']:.2f} margin={row['mean_margin']:.4f} chaos={row['mean_chaos']}"
        )
    print("\nTop persistent-error domains:")
    for row in domain_rows[: min(10, len(domain_rows))]:
        print(
            f"domain={row['domain_id']} persistent={row['persistent_error_count']}/{row['node_count']} "
            f"({row['persistent_error_frac']:.2f}) trainSameLabelNbr={row['mean_train_same_label_neighbor_frac']} "
            f"margin={row['mean_margin']:.4f}"
        )
    print("\nTop failure nodes:")
    for row in merged_rows[: args.top_k]:
        print(
            f"node={row['node_idx']} err={row['error_rate']:.2f} "
            f"deg={row['degree']} true={row['true_label']} pred={row['majority_pred_label']} "
            f"sameLabelNbr={row['same_label_neighbor_frac']} trainSameLabelNbr={row['train_same_label_neighbor_frac']} "
            f"chaos={row.get('mean_chaos')} margin={row['mean_margin']:.4f}"
        )


if __name__ == "__main__":
    main()
