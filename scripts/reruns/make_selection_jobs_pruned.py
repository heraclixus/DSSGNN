#!/usr/bin/env python3
"""Economical job list for the selection table: the lambda_reg grid only on the
small graphs; the large graphs at the lambda identified for the published row
(tolokers at both candidates). Cornell is excluded here because its four grid
jobs were already launched from the full list.
"""
import os, sys
PY = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DSS_PY", "python")
BASE = f"{PY} dssgnn_training.py --runs 10 --device 0 --amp"
GRID = ["0", "0.001", "0.01", "0.1"]
SMALL = {  # dataset -> (P, arch args): full lambda grid
    "cora":      (1, "--propagate_first --gf rw --pro_dropout 0.8"),
    "citeseer":  (2, ""),
    "texas":     (2, "--propagate_first --optimizer_prop RMSprop"),
    "wisconsin": (2, "--pro_dropout 0.6"),
    "chameleon": (1, "--propagate_first --combine sum"),
    "squirrel":  (1, "--propagate_first --gf rw --optimizer_prop RMSprop"),
}
LARGE = {  # tag -> (dataset, P, lambdas, arch args): identified lambda only
    "pubmed":                 ("pubmed", 5, ["0.01"], "--hidden 128"),
    "cs":                     ("cs", 1, ["0.01"], "--heterophily_tfe"),
    "roman-empire":           ("roman-empire", 6, ["0"], "--heterophily_tfe --layers_heterophilous 3"),
    "amazon-ratings":         ("amazon-ratings", 8, ["0.01"], "--propagate_first --hidden 128"),
    "amazon-ratings-mapcfg":  ("amazon-ratings", 8, ["0.01"], "--heterophily_tfe --layers_heterophilous 3"),
    "minesweeper":            ("minesweeper", 6, ["0.001"], "--lr 0.005"),
    "tolokers":               ("tolokers", 2, ["0", "0.01"], "--propagate_first --gf rw --optimizer_prop RMSprop"),
    "questions":              ("questions", 4, ["0.001"], "--propagate_first --optimizer_prop RMSprop"),
}
lines = []
for ds, (P, arch) in SMALL.items():
    for lam in GRID:
        lines.append(f"{ds}_P{P}_lreg{lam}\t{BASE} --dataset {ds} --P {P} --lambda_reg {lam} {arch}".rstrip())
for tag, (ds, P, lams, arch) in LARGE.items():
    for lam in lams:
        lines.append(f"{tag}_P{P}_lreg{lam}\t{BASE} --dataset {ds} --P {P} --lambda_reg {lam} {arch}".rstrip())
print("\n".join(lines))
