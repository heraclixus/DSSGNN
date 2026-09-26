#!/usr/bin/env python3
"""Compute per-node chaos energy on GOOD-Arxiv/degree for visualization.

Uses the actual GOOD pipeline from run_dssgnn_good.py.
Saves per-node chaos energy, degree, environment, and correctness.
"""

import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

from run_dssgnn_good import (
    load_good_config,
    ensure_good_dataset_registered,
    _good_data_to_dssgnn_format,
    build_model,
    forward_model,
)
from dssgnn.chebyshev import build_rescaled_laplacian
from tfe_utils import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        default='GOOD_clean/configs/GOOD_configs/GOODArxiv/degree/concept/ERM.yaml')
    parser.add_argument('--model', type=str, default='gpr_dssres')
    parser.add_argument('--P', type=int, default=2)
    parser.add_argument('--hidden', type=int, default=300)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--K_lp', type=int, default=10)
    parser.add_argument('--K_hp', type=int, default=10)
    parser.add_argument('--pro_dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--lambda_reg', type=float, default=0.01)
    parser.add_argument('--lambda_max', type=float, default=2.0)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=200)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--outdir', type=str, default='results/good_chaos_energy')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available()
                          else 'cpu')
    set_seed(42)

    # Load GOOD dataset
    config = load_good_config(config_path=args.config_path)
    dataset_name = config.dataset.dataset_name
    ensure_good_dataset_registered(dataset_name)

    from GOOD.data.good_datasets.good_arxiv import GOODArxiv
    dataset, meta = GOODArxiv.load(
        config.dataset.dataset_root,
        domain=config.dataset.domain,
        shift=config.dataset.shift_type,
        generate=False,
    )

    adj, edge_index, features, labels, masks = _good_data_to_dssgnn_format(
        dataset, config
    )

    N = features.shape[0]
    num_classes = int(labels.max()) + 1
    print(f"Dataset: {dataset_name}, N={N}, classes={num_classes}, "
          f"features={features.shape[1]}")

    train_mask = masks["train_mask"]
    val_mask = masks["val_mask"]
    test_mask = masks["test_mask"]

    # Build model
    from types import SimpleNamespace
    model_args = SimpleNamespace(
        layers=args.layers, K_lp=args.K_lp, K_hp=args.K_hp,
        P=args.P, pro_dropout=args.pro_dropout, lin_dropout=0.0,
        use_random_gates=False, P_gate=1, quadrature_nodes=4,
        shared_input_lift=False, lambda_max=args.lambda_max,
        warmup_base_epochs=50, residual_scale_init=0.1,
        gpr_K=10, gpr_alpha=0.1,
        appnp_K=10, appnp_alpha=0.1,
    )
    model = build_model(
        args.model, features.shape[1], args.hidden, num_classes, model_args
    ).to(device)

    graph_input = build_rescaled_laplacian(adj, lambda_max=args.lambda_max).to(device)
    features_d = features.to(device)
    labels_d = labels.to(device)
    edge_index_d = edge_index.to(device)
    train_mask_d = train_mask.to(device)
    val_mask_d = val_mask.to(device)
    test_mask_d = test_mask.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    best_val_loss = float('inf')
    best_state = None
    bad = 0
    model_name = args.model

    for epoch in range(args.epochs):
        model.train()
        if hasattr(model, 'set_train_epoch'):
            model.set_train_epoch(epoch)
        optimizer.zero_grad()
        logits, uncertainty = forward_model(
            model, model_name, graph_input, edge_index_d, features_d)
        out = F.log_softmax(logits, dim=1)
        loss = F.cross_entropy(out[train_mask_d], labels_d[train_mask_d])
        loss = loss + args.lambda_reg * uncertainty.mean()
        loss.backward()
        optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_logits, _ = forward_model(
                model, model_name, graph_input, edge_index_d, features_d)
            val_out = F.log_softmax(val_logits, dim=1)
            val_loss = F.cross_entropy(
                val_out[val_mask_d], labels_d[val_mask_d]).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= args.patience:
            print(f"Early stop at epoch {epoch+1}")
            break
        if (epoch + 1) % 50 == 0:
            pred = val_logits[test_mask_d].argmax(dim=1)
            test_acc = (pred == labels_d[test_mask_d]).float().mean().item()
            print(f"Epoch {epoch+1}: val_loss={val_loss:.4f} test_acc={test_acc:.4f}")

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Extract chaos energy
    model.eval()
    with torch.no_grad():
        logits, chaos_energy = forward_model(
            model, model_name, graph_input, edge_index_d, features_d)

    # Compute degrees from original edge_index (without self-loops)
    graph = dataset[0]
    orig_ei = graph.edge_index
    degrees = torch.zeros(N, dtype=torch.float)
    degrees.scatter_add_(0, orig_ei[1], torch.ones(orig_ei.shape[1]))

    # Environment info
    env = np.full(N, -1, dtype=np.int32)
    env[train_mask.numpy()] = 0
    env[val_mask.numpy()] = 1
    env[test_mask.numpy()] = 2

    # Predictions
    pred = logits.argmax(dim=1).cpu()
    correct = (pred == labels).float().numpy()
    test_acc = correct[test_mask.numpy()].mean()
    print(f"\nFinal test acc: {test_acc:.4f}")

    # Save
    result = {
        'chaos_energy': chaos_energy.cpu().numpy(),
        'degrees': degrees.numpy(),
        'env': env,
        'correct': correct,
        'labels': labels.numpy(),
        'predictions': pred.numpy(),
    }

    tag = os.path.basename(args.config_path).replace('.yaml', '')
    outpath = os.path.join(args.outdir, f'good_arxiv_degree_P{args.P}.npz')
    np.savez_compressed(outpath, **result)
    print(f"Saved {outpath}")
    print(f"Chaos energy range: [{chaos_energy.min():.4f}, {chaos_energy.max():.4f}]")


if __name__ == '__main__':
    main()
