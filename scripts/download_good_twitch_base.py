#!/usr/bin/env python3
"""
Download the Twitch assets needed by GOODTwitch.

Two modes are supported:

1. processed-only (recommended)
   Download GOOD's processed GOODTwitch archive via the GOOD dataset loader.
   This is the path needed for our current GOOD experiments.

2. raw-base
   Download the six PyG Twitch language graphs:
     DE, EN, ES, FR, PT, RU
   and optionally rebuild GOODTwitch from raw.

The raw PyG source currently points to an upstream mirror that may return 404.
So for actual GOOD experiments, processed-only is the safer default.

Examples:
  python scripts/download_good_twitch_base.py
  python scripts/download_good_twitch_base.py --processed-only
  python scripts/download_good_twitch_base.py --raw-base --prepare-good
  python scripts/download_good_twitch_base.py --root GOOD_clean/storage/datasets
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.error import HTTPError

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

for good_root in (
    REPO_ROOT / "GOOD_clean",
    REPO_ROOT / "GOOD",
):
    if good_root.is_dir() and str(good_root) not in sys.path:
        sys.path.insert(0, str(good_root))

LANGUAGES = ("DE", "EN", "ES", "FR", "PT", "RU")


def default_root() -> Path:
    if (REPO_ROOT / "GOOD_clean" / "storage" / "datasets").is_dir():
        return REPO_ROOT / "GOOD_clean" / "storage" / "datasets"
    return REPO_ROOT / "GOOD" / "storage" / "datasets"


def main():
    parser = argparse.ArgumentParser(description="Download Twitch data for GOODTwitch.")
    parser.add_argument(
        "--root",
        type=str,
        default=str(default_root()),
        help="GOOD dataset root (the directory passed to GOODTwitch/Twitch).",
    )
    parser.add_argument(
        "--prepare-good",
        action="store_true",
        help="After downloading the six raw language graphs, build GOODTwitch processed splits locally.",
    )
    parser.add_argument(
        "--processed-only",
        action="store_true",
        help="Download GOOD's processed GOODTwitch archive only. Recommended for experiments.",
    )
    parser.add_argument(
        "--raw-base",
        action="store_true",
        help="Download the six raw PyG Twitch language graphs. This may fail if the old PyG mirror is unavailable.",
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Force PyG to reprocess the raw Twitch graphs.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    if not args.raw_base and not args.processed_only and not args.prepare_good:
        args.processed_only = True

    if args.processed_only:
        print("Downloading GOODTwitch processed archive via GOOD loader ...")
        from GOOD.data.good_datasets.good_twitch import GOODTwitch

        dataset = GOODTwitch(root=str(root), domain="language", shift="concept", generate=False)
        processed_dir = root / "GOODTwitch" / "language" / "processed"
        print(f"  GOODTwitch processed dir: {processed_dir}")
        print(f"  present: {processed_dir.is_dir()} | files: {sorted(p.name for p in processed_dir.glob('*.pt'))}")
        print(f"  loaded graph nodes: {dataset[0].num_nodes}")

    if args.raw_base:
        from torch_geometric.datasets import Twitch

        print(f"Downloading Twitch raw base graphs into: {root}")
        for language in LANGUAGES:
            print(f"[raw] Ensuring Twitch {language} is present...")
            try:
                dataset = Twitch(root=str(root), name=language, force_reload=args.force_reload)
            except HTTPError as exc:
                if exc.code == 404:
                    print(
                        f"  {language}: upstream PyG mirror returned 404.\n"
                        "  The old graphmining.ai Twitch mirror appears unavailable.\n"
                        "  Use --processed-only for GOOD experiments, or provide the raw .npz files manually."
                    )
                raise
            raw_path = root / language / "raw" / f"{language}.npz"
            processed_path = root / language / "processed" / "data.pt"
            print(
                f"  {language}: nodes={dataset[0].num_nodes}, "
                f"raw={'yes' if raw_path.is_file() else 'no'}, "
                f"processed={'yes' if processed_path.is_file() else 'no'}"
            )

    if args.prepare_good:
        print("Building GOODTwitch processed splits with generate=True ...")
        from GOOD.data.good_datasets.good_twitch import GOODTwitch

        dataset = GOODTwitch(root=str(root), domain="language", shift="concept", generate=True)
        processed_dir = root / "GOODTwitch" / "language" / "processed"
        print(f"  GOODTwitch processed dir: {processed_dir}")
        print(f"  present: {processed_dir.is_dir()} | files: {sorted(p.name for p in processed_dir.glob('*.pt'))}")
        print(f"  loaded graph nodes: {dataset[0].num_nodes}")

    print("Done.")


if __name__ == "__main__":
    main()
