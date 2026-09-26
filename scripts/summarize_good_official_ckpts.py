#!/usr/bin/env python3
"""Summarize native GOOD checkpoint metrics across experiment rounds."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def load_metric(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "epoch": int(ckpt["epoch"]),
        "val_score": float(ckpt["val_score"]),
        "test_score": float(ckpt["test_score"]),
        "id_val_score": float(ckpt["id_val_score"]),
        "id_test_score": float(ckpt["id_test_score"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize native GOOD checkpoint results")
    parser.add_argument("--ckpt-root", type=Path, required=True)
    parser.add_argument("--rounds", type=int, nargs="+", required=True)
    parser.add_argument("--subpath", type=Path, default=Path("."))
    parser.add_argument("--filename", type=str, default="best.ckpt")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for rnd in args.rounds:
        path = args.ckpt_root / f"round{rnd}" / args.subpath / args.filename
        if not path.exists():
            raise FileNotFoundError(path)
        row = {"round": rnd, "path": str(path)}
        row.update(load_metric(path))
        rows.append(row)

    summary = {
        "filename": args.filename,
        "rounds": rows,
        "test_score_mean": mean([r["test_score"] for r in rows]),
        "test_score_std": std([r["test_score"] for r in rows]),
        "val_score_mean": mean([r["val_score"] for r in rows]),
        "val_score_std": std([r["val_score"] for r in rows]),
        "id_test_score_mean": mean([r["id_test_score"] for r in rows]),
        "id_test_score_std": std([r["id_test_score"] for r in rows]),
    }

    text = json.dumps(summary, indent=2)
    print(text)
    if args.output is not None:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
