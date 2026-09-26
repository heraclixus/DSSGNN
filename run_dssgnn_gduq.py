"""
Run DSS-GNN with GDUQ-style anchor mechanism for calibration experiments.

Usage:
  python run_dssgnn_gduq.py --dataset cora --runs 2 --epochs 50 --patience 20  # quick local test
  python run_dssgnn_gduq.py --dataset cora --runs 10  # full run
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from tfe_utils import set_seed, load_data, accuracy, compute_ece, compute_mce, compute_brier, propagate_adj
from dssgnn.chebyshev import build_rescaled_laplacian
from dssgnn.model import DSSGNN, DSSGNNTFE
from dssgnn.dsconv_tfe_propfirst import DSSGNNTFEPropFirst
from dssgnn.dssgnn_gduq import DSSGNNWithGDUQAnchor


def _gate_reg_loss(model):
    """Penalize higher-order gate coefficients (r>=1) for S+G."""
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    if getattr(model, "use_random_gates", False) and hasattr(model, "a_coeffs") and model.a_coeffs is not None:
        loss = loss + (model.a_coeffs[:, 1:] ** 2).sum() + (model.b_coeffs[:, 1:] ** 2).sum()
    for m in model.modules():
        if hasattr(m, "a_coeffs") and m.a_coeffs is not None and m is not model:
            loss = loss + (m.a_coeffs[:, 1:] ** 2).sum() + (m.b_coeffs[:, 1:] ** 2).sum()
    return loss


def _filter_reg_loss(model):
    """Penalize higher-order filter coefficients (r>=1) for S+F (TFE variants only)."""
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for m in model.modules():
        if hasattr(m, "gamma_lp") and m.gamma_lp is not None and m is not model:
            loss = loss + (m.gamma_lp[:, 1:] ** 2).sum() + (m.gamma_hp[:, 1:] ** 2).sum()
    return loss


def run_dssgnn_gduq(args, dataset, i):
    set_seed(args.seed if args.random_split else i)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    adj, features, labels, train_mask, val_mask, test_mask = load_data(
        dataset, args.full, args.random_split, args.train_rate, args.val_rate, i
    )

    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features, dtype=torch.float32)
    else:
        features = features.float()
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.long)
    else:
        labels = labels.long()
    num_classes = int(max(labels)) + 1

    # Anchor distribution from train features (GDUQ-style)
    train_x = features[train_mask]
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0)
    std[std == 0] = 1e-3

    # Build graph input and base model
    use_tfe = args.propagation == "tfe" or (dataset in {"chameleon", "squirrel", "texas", "cornell", "wisconsin",
        "roman-empire", "amazon-ratings", "minesweeper", "tolokers", "questions"})
    use_prop_first = args.propagate_first or use_tfe

    if use_prop_first:
        base_class = DSSGNNTFEPropFirst
        adj_lp = propagate_adj(adj, "low", -0.5, -0.5)
        adj_hp = propagate_adj(adj, "high", args.eta, args.eta)
        graph_input = (adj_lp.to(device), adj_hp.to(device))
        base_kwargs = dict(
            input_dim=features.shape[1],
            hidden_dim=args.hidden,
            out_dim=num_classes,
            num_layers=args.layers,
            K_lp=6,
            K_hp=5,
            P=args.P,
            dropout=(args.pro_dropout, args.lin_dropout),
            activation=True,
            combine="sum" if use_tfe else "con",
            use_random_gates=args.use_random_gates,
            use_random_filter=args.use_random_filter,
            P_gate=args.P_gate,
            P_filter=args.P_filter,
        )
    elif use_tfe:
        base_class = DSSGNNTFE
        adj_lp = propagate_adj(adj, "low", -0.5, -0.5)
        adj_hp = propagate_adj(adj, "high", args.eta, args.eta)
        graph_input = (adj_lp.to(device), adj_hp.to(device))
        base_kwargs = dict(
            input_dim=features.shape[1],
            hidden_dim=args.hidden,
            out_dim=num_classes,
            num_layers=args.layers,
            K_lp=6,
            K_hp=5,
            P=args.P,
            dropout=(args.pro_dropout, args.lin_dropout),
            activation=True,
            use_random_gates=args.use_random_gates,
            use_random_filter=args.use_random_filter,
            P_gate=args.P_gate,
            P_filter=args.P_filter,
        )
    else:
        base_class = DSSGNN
        graph_input = build_rescaled_laplacian(adj, lambda_max=args.lambda_max).to(device)
        base_kwargs = dict(
            input_dim=features.shape[1],
            hidden_dim=args.hidden,
            out_dim=num_classes,
            num_layers=args.layers,
            K_lp=args.K_lp,
            K_hp=args.K_hp,
            P=args.P,
            dropout=(args.pro_dropout, args.lin_dropout),
            activation=True,
            use_random_gates=args.use_random_gates,
            P_gate=args.P_gate,
        )

    model = DSSGNNWithGDUQAnchor(
        base_model_class=base_class,
        base_model_kwargs=base_kwargs,
        mean=mean,
        std=std,
        anchor_type="node",
    )
    model = model.to(device)
    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val_loss = float("inf")
    test_acc = 0
    test_ece = test_mce = test_brier = 0.0
    bad_epoch = 0

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(graph_input, features, anchors=None, n_anchors=args.num_anchors)
        loss = F.cross_entropy(out[train_mask], labels[train_mask])
        if args.use_random_gates and args.lambda_op > 0:
            loss = loss + args.lambda_op * _gate_reg_loss(model.base_model)
        if args.use_random_filter and args.lambda_filter > 0:
            loss = loss + args.lambda_filter * _filter_reg_loss(model.base_model)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(graph_input, features, anchors=None, n_anchors=args.num_anchors)
            logits = F.log_softmax(out, dim=1)
            val_loss = F.cross_entropy(logits[val_mask], labels[val_mask]).item()
            tmp_test_acc = accuracy(logits[test_mask], labels[test_mask])
            tmp_test_ece = compute_ece(logits[test_mask], labels[test_mask]).item()
            tmp_test_mce = compute_mce(logits[test_mask], labels[test_mask]).item()
            tmp_test_brier = compute_brier(logits[test_mask], labels[test_mask]).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            test_ece = tmp_test_ece
            test_mce = tmp_test_mce
            test_brier = tmp_test_brier
            bad_epoch = 0
        else:
            bad_epoch += 1
        if bad_epoch == args.patience:
            break

    return test_acc, test_ece, test_mce, test_brier


def main():
    parser = argparse.ArgumentParser(description="DS-GNN + GDUQ anchor for calibration")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--K_lp", type=int, default=4)
    parser.add_argument("--K_hp", type=int, default=4)
    parser.add_argument("--P", type=int, default=2)
    parser.add_argument("--pro_dropout", type=float, default=0.5)
    parser.add_argument("--lin_dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--num_anchors", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full", type=bool, default=True)
    parser.add_argument("--random_split", type=bool, default=True)
    parser.add_argument("--propagation", type=str, default="chebyshev", choices=["chebyshev", "tfe"])
    parser.add_argument("--propagate_first", action="store_true")
    parser.add_argument("--eta", type=float, default=-0.3)
    parser.add_argument("--lambda_max", type=float, default=2.0)
    parser.add_argument("--use_random_gates", action="store_true",
                        help="S+G+A: combine anchoring with random branch gates")
    parser.add_argument("--use_random_filter", action="store_true",
                        help="S+F+A: combine anchoring with random filter coefficients")
    parser.add_argument("--P_gate", type=int, default=1)
    parser.add_argument("--P_filter", type=int, default=1)
    parser.add_argument("--lambda_op", type=float, default=0.01, help="Gate reg for S+G")
    parser.add_argument("--lambda_filter", type=float, default=0.01, help="Filter reg for S+F")
    args = parser.parse_args()
    args.train_rate = 0.6 if args.full else 0.025
    args.val_rate = 0.2 if args.full else 0.025

    print(args)
    all_accs = []
    all_eces = []
    all_mces = []
    all_briers = []
    for i in tqdm(range(args.runs)):
        acc, ece, mce, brier = run_dssgnn_gduq(args, args.dataset, i)
        all_accs.append(acc.item())
        all_eces.append(ece)
        all_mces.append(mce)
        all_briers.append(brier)
        print(f"run_{i+1}  acc: {acc:.4f}  ECE: {ece:.4f}  MCE: {mce:.4f}  Brier: {brier:.4f}")
    print(f"test acc mean (%) = {np.mean(all_accs)*100:.2f} ± {np.std(all_accs)*100:.2f}")
    print(f"test ECE mean = {np.mean(all_eces):.4f} ± {np.std(all_eces):.4f}")
    print(f"test MCE mean = {np.mean(all_mces):.4f} ± {np.std(all_mces):.4f}")
    print(f"test Brier mean = {np.mean(all_briers):.4f} ± {np.std(all_briers):.4f}")


if __name__ == "__main__":
    main()
