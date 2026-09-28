#!/usr/bin/env python3
"""Job lists for the Cornell row (corrected geom-gcn Cornell files).

Stage "arch": the architecture grid of Appendix D at the grid defaults
(P=2, lambda_reg=0.01, S=4); the architecture is selected on validation Brier.
For Cornell the TFE-adjacency and propagate-first variants fix the hidden
width (128 and 512) and raise the dropout to at least 0.6, so the grid has 32
distinct configurations.
Stage "sweep": at the selected architecture, the chaos order P and lambda_reg
grid (selected on validation Brier), the quadrature sweep at P=2, and the
baselines of the Cornell row.

Usage: python make_cornell_jobs.py arch|sweep [python] > jobs.tsv
Run the list with launch_jobs.py; on a CPU-only machine pass e.g. --gpus 0,1,2.
"""
import os, sys
stage = sys.argv[1] if len(sys.argv) > 1 else "arch"
PY = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("DSS_PY", "python")
DSS = f"{PY} dssgnn_training.py --runs 10 --device 0 --dataset cornell"
ARCH = "--hidden 128 --layers 2 --pro_dropout 0.8"   # selected on validation Brier
jobs = []
if stage == "arch":
    base = f"{DSS} --P 2 --lambda_reg 0.01"
    for h in (64, 128):
        for L in (2, 3):
            for d in (0.5, 0.6, 0.8):
                jobs.append((f"cheb_h{h}_L{L}_d{d}", f"{base} --hidden {h} --layers {L} --pro_dropout {d}"))
    for L in (2, 3):
        for d in (0.6, 0.8):
            jobs.append((f"tfeadj_L{L}_d{d}", f"{base} --heterophily_tfe --layers_heterophilous {L} --pro_dropout {d}"))
    for g in ("sym", "rw"):
        for o in ("Adam", "RMSprop"):
            for L in (2, 3):
                for d in (0.6, 0.8):
                    jobs.append((f"propfirst_{g}_{o}_L{L}_d{d}",
                                 f"{base} --propagate_first --gf {g} --optimizer_prop {o} --layers_heterophilous {L} --pro_dropout {d}"))
else:
    jobs.append(("P0_lreg0.01", f"{DSS} {ARCH} --P 0 --lambda_reg 0.01"))
    for P in (1, 2, 3, 4, 5, 6, 8):
        for lam in ("0", "0.001", "0.01", "0.1"):
            jobs.append((f"P{P}_lreg{lam}", f"{DSS} {ARCH} --P {P} --lambda_reg {lam}"))
    for S in (2, 3, 6, 8):
        jobs.append((f"S{S}_P2_lreg0.01", f"{DSS} {ARCH} --P 2 --lambda_reg 0.01 --S {S}"))
    sb = f"{PY} run_simple_baselines.py --dataset cornell --runs 10 --device 0"
    jobs += [
        ("gcn", f"{sb} --model gcn"),
        ("gat", f"{sb} --model gat"),
        ("mcdropout", f"{sb} --model gcn --mc_samples 20"),
        ("ensemble", f"{sb} --model gcn --ensemble_size 5"),
        ("hybrid", f"{PY} run_hybrid_calibration.py --dataset cornell --runs 10 --device 0"),
        ("gcnbase", f"{PY} run_hybrid_calibration.py --dataset cornell --runs 10 --device 0 --model gcn"),
        ("tfe", f"{PY} scripts/run_best_configs_acc_ece.py --dataset cornell --model tfe --runs 10"),
        ("gduq", f"{PY} scripts/run_best_configs_acc_ece.py --dataset cornell --model gduq --runs 10"),
        ("chebnet_single", f"{DSS} --P 0 --branch lp --K_lp 4 --hidden 64 --layers 2"),
    ]
print("\n".join(f"{n}\t{c}" for n, c in jobs))
