#!/usr/bin/env python3
"""
Download the processed GOODArxiv archive via the official GOOD loader.

This prepares the dataset cache expected by GOOD's `GOODArxiv` dataset class:

  GOOD_clean/storage/datasets/GOODArxiv/degree/processed
  GOOD_clean/storage/datasets/GOODArxiv/time/processed

Unlike the base `ogbn_arxiv` cache under `data/ogb/ogbn_arxiv`, the GOOD
loader expects the separate processed GOOD archive. If that folder is missing,
GOOD will try to download it from Google Drive.

Examples:
  python scripts/download_good_arxiv_dataset.py
  python scripts/download_good_arxiv_dataset.py --domains degree
  python scripts/download_good_arxiv_dataset.py --root GOOD_clean/storage/datasets
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


CONFIG_SPECS = {
    "degree": "configs/GOOD_configs/GOODArxiv/degree/concept/ERM.yaml",
    "time": "configs/GOOD_configs/GOODArxiv/time/concept/ERM.yaml",
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
    parser = argparse.ArgumentParser(description="Download GOODArxiv processed dataset cache.")
    parser.add_argument(
        "--domains",
        nargs="*",
        default=["degree", "time"],
        choices=["degree", "time"],
        help="Which GOODArxiv domains to prepare.",
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

    ensure_good_dataset_registered("GOODArxiv")
    print(f"Preparing GOODArxiv under: {dataset_root}")

    for domain in args.domains:
        config_path = good_home / CONFIG_SPECS[domain]
        processed_dir = dataset_root / "GOODArxiv" / domain / "processed"
        print("============================================================")
        print(f"Domain:   {domain}")
        print(f"Config:   {config_path}")
        print(f"Target:   {processed_dir}")

        config = load_good_config(
            config_path=str(config_path),
            dataset_root=str(dataset_root),
            generate=False,
        )
        dataset = load_dataset(config.dataset.dataset_name, config)
        train_dataset = dataset["train"]
        print(
            f"Loaded: train_graphs={len(train_dataset)} | "
            f"val_graphs={len(dataset['val'])} | test_graphs={len(dataset['test'])}"
        )
        files = sorted(p.name for p in processed_dir.glob("*.pt"))
        print("Processed files: " + ", ".join(files))

    print("Done.")


if __name__ == "__main__":
    main()
