#!/usr/bin/env python3
"""Job list for the Graph-EBM re-run under the current OOD pipeline (BatchNorm on),
the same protocol as the MC-dropout and Deep Ensemble rows: 3 runs, 200 epochs.
"""
import os, sys
PY = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DSS_PY", "python")
BASE = f"{PY} run_ood_gnnsafe.py --method graph_ebm --backbone gcn --mode detect --use_bn --runs 3 --epochs 200 --device 0"
lines = []
for ds in ["cora", "amazon-photo", "coauthor-cs"]:
    for t in ["structure", "feature", "label"]:
        lines.append(f"gebm_{ds}_{t}\t{BASE} --dataset {ds} --ood_type {t}")
for ds in ["twitch", "arxiv"]:
    lines.append(f"gebm_{ds}\t{BASE} --dataset {ds} --ood_type structure")
print("\n".join(lines))
