#!/usr/bin/env python3
"""
Prepare the RDKit-backed GOOD molecule datasets by downloading GOOD's processed
archives through the official GOOD dataset loaders.

This targets the representative chemistry configs used by the dependency
checker:
  - GOODHIV / scaffold / concept
  - GOODPCBA / scaffold / concept
  - GOODZINC / scaffold / concept

Examples:
  python scripts/download_good_rdkit_datasets.py
  python scripts/download_good_rdkit_datasets.py --datasets goodhiv_scaffold_concept goodzinc_scaffold_concept
  python scripts/download_good_rdkit_datasets.py --root GOOD_clean/storage/datasets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

for good_root in (
    REPO_ROOT / "GOOD_clean",
    REPO_ROOT / "GOOD",
):
    if good_root.is_dir() and str(good_root) not in sys.path:
        sys.path.insert(0, str(good_root))

from run_dssgnn_good import ensure_good_dataset_registered, load_good_config

DATASET_SPECS = {
    "goodhiv_scaffold_concept": {
        "config_rel": "configs/GOOD_configs/GOODHIV/scaffold/concept/ERM.yaml",
        "dataset_name": "GOODHIV",
        "domain": "scaffold",
    },
    "goodpcba_scaffold_concept": {
        "config_rel": "configs/GOOD_configs/GOODPCBA/scaffold/concept/ERM.yaml",
        "dataset_name": "GOODPCBA",
        "domain": "scaffold",
    },
    "goodzinc_scaffold_concept": {
        "config_rel": "configs/GOOD_configs/GOODZINC/scaffold/concept/ERM.yaml",
        "dataset_name": "GOODZINC",
        "domain": "scaffold",
    },
}


def default_root() -> Path:
    if (REPO_ROOT / "GOOD_clean" / "storage" / "datasets").is_dir():
        return REPO_ROOT / "GOOD_clean" / "storage" / "datasets"
    return REPO_ROOT / "GOOD" / "storage" / "datasets"


def default_good_home() -> Path:
    if (REPO_ROOT / "GOOD_clean").is_dir():
        return REPO_ROOT / "GOOD_clean"
    return REPO_ROOT / "GOOD"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GOOD processed chemistry datasets.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DATASET_SPECS.keys()),
        choices=sorted(DATASET_SPECS.keys()),
        help="Which GOOD chemistry presets to prepare.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(default_root()),
        help="GOOD dataset root.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.root).resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)
    good_home = default_good_home()

    from GOOD.data import load_dataset

    print(f"Preparing GOOD chemistry datasets under: {dataset_root}")
    for dataset_key in args.datasets:
        spec = DATASET_SPECS[dataset_key]
        config_path = good_home / spec["config_rel"]
        processed_dir = dataset_root / spec["dataset_name"] / spec["domain"] / "processed"
        print("============================================================")
        print(f"Dataset: {dataset_key}")
        print(f"Config:   {config_path}")
        print(f"Target:   {processed_dir}")
        config = load_good_config(
            config_path=str(config_path),
            dataset_root=str(dataset_root),
            generate=False,
        )
        ensure_good_dataset_registered(config.dataset.dataset_name)
        dataset = load_dataset(config.dataset.dataset_name, config)
        train_dataset = dataset["train"]
        print(
            f"Loaded: train_graphs={len(train_dataset)} | "
            f"val_graphs={len(dataset['val'])} | test_graphs={len(dataset['test'])}"
        )
        print(
            "Processed files: "
            + ", ".join(sorted(p.name for p in processed_dir.glob("*.pt")))
        )

    print("Done.")


if __name__ == "__main__":
    main()
