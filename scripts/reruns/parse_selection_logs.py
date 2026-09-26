#!/usr/bin/env python3
"""Parse selection-table logs: pick, per dataset, the lambda_reg with the lowest
mean validation Brier and print the table rows (P, lambda, val Brier, test acc, test Brier).

Usage: python scripts/reruns/parse_selection_logs.py results_local/reruns/selection
"""
import re, sys
from collections import defaultdict
from pathlib import Path

logdir = Path(sys.argv[1])
rows = defaultdict(dict)  # dataset-tag -> lambda -> metrics
pat = {
    "acc": re.compile(r"test acc mean \(%\) = ([\d.]+) ± ([\d.]+)"),
    "brier": re.compile(r"test Brier mean = ([\d.]+) ± ([\d.]+)"),
    "val_brier": re.compile(r"val Brier mean = ([\d.]+) ± ([\d.]+)"),
    "val_acc": re.compile(r"val acc mean \(%\) = ([\d.]+) ± ([\d.]+)"),
    "quad_brier": re.compile(r"test quad Brier mean = ([\d.]+) ± ([\d.]+)"),
    "disagree": re.compile(r"disagreement \(%\) = ([\d.]+) ± ([\d.]+)"),
}
for log in sorted(logdir.glob("*.log")):
    m = re.match(r"(.+)_P(\d+)_lreg([\d.]+)\.log", log.name)
    if not m:
        continue
    tag, P, lam = m.group(1), int(m.group(2)), m.group(3)
    txt = log.read_text()
    if "JOB_DONE_OK" not in txt:
        print(f"# incomplete: {log.name}", file=sys.stderr); continue
    met = {"P": P}
    ok = True
    for k, r in pat.items():
        mm = r.search(txt)
        if mm:
            met[k] = (float(mm.group(1)), float(mm.group(2)))
        elif k in ("acc", "brier", "val_brier"):
            ok = False
    if ok:
        rows[tag][lam] = met

print("tag | P | lambda_reg (val-selected) | val Brier | test acc | test Brier | quad Brier | disagree%")
for tag in sorted(rows):
    best = min(rows[tag].items(), key=lambda kv: kv[1]["val_brier"][0])
    lam, m = best
    print(f"{tag} | {m['P']} | {lam} | {m['val_brier'][0]:.3f} ± {m['val_brier'][1]:.3f} | "
          f"{m['acc'][0]:.2f} ± {m['acc'][1]:.2f} | {m['brier'][0]:.3f} ± {m['brier'][1]:.3f} | "
          f"{m.get('quad_brier',(float('nan'),0))[0]:.4f} | {m.get('disagree',(float('nan'),0))[0]:.2f}")
    others = ", ".join(f"lam={l}: val {v['val_brier'][0]:.3f} / test {v['brier'][0]:.3f} / acc {v['acc'][0]:.2f}" for l, v in sorted(rows[tag].items()))
    print(f"    all arms: {others}")
