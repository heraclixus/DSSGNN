#!/usr/bin/env python3
"""Job list for the per-dataset selection table (Appendix D, tab:selection).

For each dataset, runs the selected chaos order P at each lambda_reg in the
paper's grid {0, 1e-3, 1e-2, 1e-1} under the full 10-split Table 1 protocol;
the validation Brier printed by dssgnn_training.py is the selection criterion.
Mixed precision is used but NOT --reduced_memory, which caps P to 1 and the
hidden width to 32 on the large datasets and would override the selected order.
Architecture arguments follow scripts/run_best_configs_acc_ece.py, with two
documented corrections: minesweeper uses the published Chebyshev/Adam arm
(lr 0.005), and amazon-ratings runs both the sweep arm (propagate-first,
hidden 128) and the best-config-map arm (TFE-adjacency, 3 layers).
"""
import os, sys
PY = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DSS_PY", "python")
BASE = f"{PY} dssgnn_training.py --runs 10 --device 0 --amp"
LAMBDAS = ["0", "0.001", "0.01", "0.1"]

# dataset -> (selected P, architecture args)
CONFIGS = {
    "cora":           (1, "--propagate_first --gf rw --pro_dropout 0.8"),
    "citeseer":       (2, ""),
    "pubmed":         (5, "--hidden 128"),
    "texas":          (2, "--propagate_first --optimizer_prop RMSprop"),
    "cornell":        (1, "--hidden 128 --pro_dropout 0.8"),
    "wisconsin":      (2, "--pro_dropout 0.6"),
    "chameleon":      (1, "--propagate_first --combine sum"),
    "squirrel":       (1, "--propagate_first --gf rw --optimizer_prop RMSprop"),
    "cs":             (1, "--heterophily_tfe"),
    "roman-empire":   (6, "--heterophily_tfe --layers_heterophilous 3"),
    "amazon-ratings": (8, "--propagate_first --hidden 128"),
    "minesweeper":    (6, "--lr 0.005"),
    "tolokers":       (2, "--propagate_first --gf rw --optimizer_prop RMSprop"),
    "questions":      (4, "--propagate_first --optimizer_prop RMSprop"),
}
EXTRA_ARMS = {  # secondary architecture arms, run as a check
    "amazon-ratings-mapcfg": ("amazon-ratings", 8, "--heterophily_tfe --layers_heterophilous 3"),
}

lines = []
for ds, (P, arch) in CONFIGS.items():
    for lam in LAMBDAS:
        name = f"{ds}_P{P}_lreg{lam}"
        lines.append(f"{name}\t{BASE} --dataset {ds} --P {P} --lambda_reg {lam} {arch}".rstrip())
for tag, (ds, P, arch) in EXTRA_ARMS.items():
    for lam in LAMBDAS:
        name = f"{tag}_P{P}_lreg{lam}"
        lines.append(f"{name}\t{BASE} --dataset {ds} --P {P} --lambda_reg {lam} {arch}".rstrip())
print("\n".join(lines))
