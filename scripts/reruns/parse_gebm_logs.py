#!/usr/bin/env python3
"""Parse Graph-EBM re-run logs (OOD subset 1, the published Table convention):
mean +- SD over runs of AUROC / AUPR / FPR95 and ID test accuracy.

Usage: python scripts/reruns/parse_gebm_logs.py results_local/reruns/gebm
"""
import re, statistics as st, sys
from pathlib import Path

logdir = Path(sys.argv[1])
print("setting | AUROC | AUPR | FPR95 | ID acc | runs")
for log in sorted(logdir.glob("gebm_*.log")):
    txt = log.read_text()
    if "JOB_DONE_OK" not in txt:
        print(f"# incomplete: {log.name}", file=sys.stderr); continue
    ood1 = re.findall(r"OOD 1 AUROC: ([\d.]+) AUPR: ([\d.]+) FPR95: ([\d.]+)", txt)
    ind = re.findall(r"IND Test: ([\d.]+)\s+Brier", txt)
    if not ood1:
        print(f"# no OOD-1 block: {log.name}", file=sys.stderr); continue
    cols = list(zip(*[(float(a), float(p), float(f)) for a, p, f in ood1]))
    def ms(v): return f"{st.mean(v):.2f} ± {st.pstdev(v):.2f}"
    idacc = ms([float(x) for x in ind]) if ind else "n/a"
    print(f"{log.stem[5:]} | {ms(cols[0])} | {ms(cols[1])} | {ms(cols[2])} | {idacc} | {len(ood1)}")
