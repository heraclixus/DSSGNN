#!/usr/bin/env python3
"""
Generate per-node confidence data for GOOD shifted test nodes.
Compares P=0 vs P=2 on shifted test set.

Usage:
  python scripts/generate_good_confidence_data.py --setting arxiv_degree
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)
for good_root in (os.path.join(PROJ_ROOT, "GOOD_clean"), os.path.join(PROJ_ROOT, "GOOD")):
    if os.path.isdir(good_root):
        sys.path.insert(0, good_root)

from tfe_utils import set_seed
from dssgnn.chebyshev import build_rescaled_laplacian
from run_dssgnn_good import (
    load_good_config, build_model, forward_model, ensure_good_dataset_registered,
    _good_data_to_dssgnn_format,
)

SETTINGS = {
    "arxiv_degree": ("GOOD_clean/configs/GOOD_configs/GOODArxiv/degree/concept/ERM.yaml", "gpr_dssres"),
    "cora_degree": ("GOOD_clean/configs/GOOD_configs/GOODCora/degree/concept/ERM.yaml", "gcn_dssres"),
}


def run_and_collect(config_path, model_name, P, device, seed=42):
    set_seed(seed)
    config = load_good_config(config_path=config_path, hidden=300, seed=seed, generate=False)
    from GOOD.data import load_dataset
    ensure_good_dataset_registered(config.dataset.dataset_name)
    dataset = load_dataset(config.dataset.dataset_name, config)

    adj, edge_index, features, labels, masks = _good_data_to_dssgnn_format(dataset, config)
    train_mask = masks["train_mask"].to(device)
    test_mask = masks["test_mask"].to(device)
    num_classes = int(labels.max()) + 1

    graph_input = build_rescaled_laplacian(adj, lambda_max=2.0).to(device)
    edge_index = edge_index.to(device)
    features = features.to(device)
    labels = labels.to(device)

    args_obj = argparse.Namespace(
        model=model_name, hidden=300, layers=3, K_lp=3, K_hp=2, P=P,
        pro_dropout=0.5, lin_dropout=0.0, lambda_max=2.0,
        use_random_gates=False, P_gate=1, quadrature_nodes=4,
        shared_input_lift=False, appnp_K=10, appnp_alpha=0.1,
        gpr_K=10, gpr_alpha=0.1, warmup_base_epochs=25,
        residual_scale_init=0.1, S=4, loss_type="mean",
    )

    model = build_model(model_name, features.shape[1], 300, num_classes, args_obj).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=5e-4)

    best_val_loss = float("inf")
    best_state = None
    val_mask = masks["val_mask"].to(device)
    for epoch in range(500):
        if hasattr(model, "set_train_epoch"):
            model.set_train_epoch(epoch)
        model.train()
        optimizer.zero_grad()
        logits, unc = forward_model(model, model_name, graph_input, edge_index, features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask]) + 0.01 * unc.mean()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            logits, _ = forward_model(model, model_name, graph_input, edge_index, features)
            vl = F.cross_entropy(logits[val_mask], labels[val_mask]).item()
        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.to(device).eval()
    with torch.no_grad():
        logits, _ = forward_model(model, model_name, graph_input, edge_index, features)
        probs = F.softmax(logits, dim=1)

    test_idx = test_mask.nonzero(as_tuple=True)[0].cpu()
    test_probs = probs[test_idx].cpu().numpy()
    test_labels = labels[test_idx].cpu().numpy()
    preds = test_probs.argmax(axis=1)
    correct = (preds == test_labels)
    confidence = test_probs.max(axis=1)

    return {
        "confidence_correct": confidence[correct].tolist(),
        "confidence_wrong": confidence[~correct].tolist(),
        "acc": float(correct.mean()),
        "n_test": len(test_idx),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", type=str, required=True, choices=list(SETTINGS.keys()))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="results_local/viz_data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    config_path, model_name = SETTINGS[args.setting]

    output = {"setting": args.setting}
    for P in [0, 2]:
        print(f"Running {args.setting} P={P}...")
        data = run_and_collect(config_path, model_name, P, device)
        output[f"P{P}"] = data
        print(f"  P={P}: acc={data['acc']:.4f}, n_correct={len(data['confidence_correct'])}, n_wrong={len(data['confidence_wrong'])}")

    out_path = os.path.join(args.output_dir, f"good_confidence_{args.setting}.json")
    with open(out_path, "w") as f:
        json.dump(output, f)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
