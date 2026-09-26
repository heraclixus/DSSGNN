"""
DSS-Hybrid (GCN base + DSS residual branch) calibration script for node classification.

Evaluates GCNDSSResidualEncoder under the same calibration protocol as
dssgnn_training.py: 10 random 60/20/20 splits (same SEEDS via tfe_utils.load_data),
early stopping on validation loss, and test acc/ECE/MCE/Brier recorded at the
best-val-loss epoch. Reports two predictive heads:
  mean_head: log_softmax of the hybrid mean logits (base + scale * Z_0)
  quad_head: quadrature predictive p_bar = sum_s w_s * softmax(z^(s))

--model gcn trains the plain GCN base alone for a controlled comparison.

Reuses data loading and metric utilities from tfe_utils.
"""

import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import time
import argparse
from tqdm import tqdm

from tfe_utils import set_seed, accuracy, load_data, compute_ece, compute_mce, compute_brier
from gnnsafe_ood.backbone import GCN, GCNDSSResidualEncoder


def adj_to_edge_index(adj):
    """
    Convert load_data's scipy adjacency to an edge_index (2, E) LongTensor.

    load_data returns adj with explicit self-loops (adj + sp.eye). GCNConv and the
    DSS residual branch (_edge_index_to_adj) both add their own self-loops, so the
    diagonal is stripped here to avoid double self-loops.
    """
    adj = sp.csr_matrix(adj)
    adj = adj - sp.diags(adj.diagonal(), format="csr")
    adj.eliminate_zeros()
    coo = adj.tocoo()
    edge_index = torch.stack(
        [torch.from_numpy(coo.row).long(), torch.from_numpy(coo.col).long()], dim=0
    )
    return edge_index


def cache_residual_laplacian(model):
    """
    Memoize the DSS branch's rescaled Laplacian.

    GCNDSSResidualEncoder rebuilds the Laplacian from edge_index on every forward;
    the graph is static across epochs, so cache it per device. No behavior change,
    and gnnsafe_ood stays unmodified (instance-level override only).
    """
    residual = model.residual_dss
    original = residual._build_graph_input
    cache = {}

    def cached_build(x, edge_index):
        if x.device not in cache:
            cache[x.device] = original(x, edge_index)
        return cache[x.device]

    residual._build_graph_input = cached_build


def train(model, optimizer, edge_index, x, y, mask, epoch, lambda_reg=0.01, is_hybrid=True):
    """
    Train one epoch with the hybrid mean-logit loss:
      L_sup(logits, Y) + lambda_reg * chaos_uncertainty.mean()
    set_train_epoch(epoch) drives the warmup schedule (DSS residual activates
    automatically after warmup_base_epochs; uncertainty is zero before that).
    """
    if is_hybrid:
        model.set_train_epoch(epoch)
    model.train()
    optimizer.zero_grad()
    if is_hybrid:
        logits, uncertainty = model.forward_with_uncertainty(x, edge_index)
    else:
        logits, uncertainty = model(x, edge_index), None
    out = F.log_softmax(logits, dim=1)
    loss = F.cross_entropy(out[mask[0]], y[mask[0]])
    if uncertainty is not None:
        loss = loss + lambda_reg * uncertainty.mean()
    loss.backward()
    optimizer.step()


def test(model, edge_index, x, y, mask):
    """Evaluate the mean-logit head on train/val/test splits (mirrors dssgnn_training.test)."""
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
    logits = F.log_softmax(logits, dim=1)
    accs, losses = [], []
    for i in range(3):
        acc = accuracy(logits[mask[i]], y[mask[i]])
        loss = F.cross_entropy(logits[mask[i]], y[mask[i]])
        accs.append(acc)
        losses.append(loss)
    return accs, losses, logits


def quad_test(model, edge_index, x, y, test_mask):
    """
    Evaluate the quadrature-predictive head on the test split.

    p_bar = sum_s w_s * softmax(z^(s)) from forward_with_predictive_stats. Also
    reports the fraction of test nodes where argmax p_bar != argmax mean-logits.
    """
    model.eval()
    with torch.no_grad():
        stats = model.forward_with_predictive_stats(x, edge_index)
    sample_probs = torch.softmax(stats["sample_logits"], dim=-1)  # (S, N, C)
    weights = stats["weights"].view(-1, 1, 1).to(sample_probs.dtype)
    p_bar = (weights * sample_probs).sum(dim=0)  # (N, C)
    quad_acc = accuracy(p_bar[test_mask], y[test_mask])
    quad_ece = compute_ece(p_bar[test_mask], y[test_mask]).item()
    quad_mce = compute_mce(p_bar[test_mask], y[test_mask]).item()
    quad_brier = compute_brier(p_bar[test_mask], y[test_mask]).item()
    mean_pred = stats["logits"][test_mask].argmax(dim=1)
    quad_pred = p_bar[test_mask].argmax(dim=1)
    disagree = (quad_pred != mean_pred).float().mean().item()
    return quad_acc, quad_ece, quad_mce, quad_brier, disagree


def run(args, dataset, full, random_split, i):
    if args.random_split:
        set_seed(args.seed)
    else:
        set_seed(i)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    # Reuse load_data from tfe_utils (same splits/SEEDS as dssgnn_training.py)
    adj, features, labels, train_mask, val_mask, test_mask = load_data(
        dataset, full, random_split, args.train_rate, args.val_rate, i
    )

    num_classes = int(max(labels)) + 1
    edge_index = adj_to_edge_index(adj)

    is_hybrid = args.model == "hybrid"
    if is_hybrid:
        model = GCNDSSResidualEncoder(
            in_channels=features.shape[1],
            hidden_channels=args.hidden,
            out_channels=num_classes,
            num_layers=args.layers,
            dropout=args.dropout,
            use_bn=not args.no_bn,
            K_lp=args.K_lp,
            K_hp=args.K_hp,
            P=args.P,
            lambda_max=args.lambda_max,
            use_random_gates=args.use_random_gates,
            P_gate=args.P_gate,
            quadrature_nodes=args.S,
            shared_input_lift=args.shared_input_lift,
            warmup_base_epochs=args.warmup_base_epochs,
            residual_scale_init=args.residual_scale_init,
        )
        cache_residual_laplacian(model)
    else:
        model = GCN(
            in_channels=features.shape[1],
            hidden_channels=args.hidden,
            out_channels=num_classes,
            num_layers=args.layers,
            dropout=args.dropout,
            use_bn=not args.no_bn,
        )

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
    )

    model = model.to(device)
    features = features.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    edge_index = edge_index.to(device)
    train_mask = train_mask.clone().detach().to(device)
    val_mask = val_mask.clone().detach().to(device)
    test_mask = test_mask.clone().detach().to(device)
    mask = [train_mask, val_mask, test_mask]

    best_val_loss = float("inf")
    test_acc = 0
    test_ece = test_mce = test_brier = 0.0
    quad_metrics = None
    bad_epoch = 0
    run_time = []

    for epoch in range(args.epochs):
        t0 = time.time()
        train(model, optimizer, edge_index, features, labels, mask, epoch,
              lambda_reg=args.lambda_reg, is_hybrid=is_hybrid)
        run_time.append(time.time() - t0)

        [train_acc, val_acc, tmp_test_acc], [train_loss, val_loss, tmp_test_loss], logits = test(
            model, edge_index, features, labels, mask
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            test_ece = compute_ece(logits[test_mask], labels[test_mask]).item()
            test_mce = compute_mce(logits[test_mask], labels[test_mask]).item()
            test_brier = compute_brier(logits[test_mask], labels[test_mask]).item()
            if is_hybrid:
                quad_metrics = quad_test(model, edge_index, features, labels, test_mask)
            bad_epoch = 0
        else:
            bad_epoch += 1

        if bad_epoch == args.patience:
            break

    result = {
        "mean": (test_acc, test_ece, test_mce, test_brier),
        "quad": quad_metrics,
        "best_val_loss": best_val_loss,
        "run_time": run_time,
    }
    del model, optimizer, edge_index, features, labels, train_mask, val_mask, test_mask, mask
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description="DSS-Hybrid (GCN base + DSS residual) calibration protocol")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="cora, citeseer, pubmed, texas, cornell, wisconsin, chameleon, squirrel, cs, "
             "roman-empire, amazon-ratings, minesweeper, tolokers, questions",
    )
    parser.add_argument("--model", type=str, default="hybrid", choices=["hybrid", "gcn"],
                        help="hybrid: GCN base + DSS residual; gcn: plain GCN base only")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--runs", type=int, default=10)

    parser.add_argument("--K_lp", type=int, default=3, help="Chebyshev degree low-pass (DSS residual)")
    parser.add_argument("--K_hp", type=int, default=2, help="Chebyshev degree high-pass (DSS residual)")
    parser.add_argument("--P", type=int, default=1, help="Chaos truncation order")
    parser.add_argument("--S", type=int, default=4, help="Number of Gauss-Hermite quadrature nodes")
    parser.add_argument("--lambda_max", type=float, default=2.0)
    parser.add_argument("--lambda_reg", type=float, default=0.01, help="Regularization on higher-order channels")
    parser.add_argument("--warmup_base_epochs", type=int, default=50,
                        help="Epochs training the GCN base alone before the DSS residual activates")
    parser.add_argument("--residual_scale_init", type=float, default=0.1,
                        help="Initial value of the learnable residual logit scale")
    parser.add_argument("--use_random_gates", action="store_true",
                        help="Stochastic gates in the DSS residual (default: deterministic scalar gates)")
    parser.add_argument("--P_gate", type=int, default=1,
                        help="Chaos order for gate expansion (when use_random_gates)")
    parser.add_argument("--shared_input_lift", action="store_true",
                        help="Share the input lift across chaos channels in the DSS residual")

    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--no_bn", action="store_true", help="Disable BatchNorm in the GCN base")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=float, default=5e-4)

    parser.add_argument("--full", type=bool, default=True)
    parser.add_argument("--random_split", type=bool, default=True)

    args = parser.parse_args()
    args.train_rate = 0.6 if args.full else 0.025
    args.val_rate = 0.2 if args.full else 0.025

    print(args)

    all_mean = {"acc": [], "ece": [], "mce": [], "brier": []}
    all_quad = {"acc": [], "ece": [], "mce": [], "brier": [], "disagree": []}
    time_results = []

    for i in tqdm(range(args.runs)):
        result = run(args, args.dataset, args.full, args.random_split, i)
        test_acc, test_ece, test_mce, test_brier = result["mean"]
        all_mean["acc"].append(test_acc.item())
        all_mean["ece"].append(test_ece)
        all_mean["mce"].append(test_mce)
        all_mean["brier"].append(test_brier)
        time_results.append(result["run_time"])
        print(f"run_{i+1}  mean_head  acc: {test_acc:.4f}  ECE: {test_ece:.4f}  "
              f"MCE: {test_mce:.4f}  Brier: {test_brier:.4f}")
        if result["quad"] is not None:
            q_acc, q_ece, q_mce, q_brier, q_disagree = result["quad"]
            all_quad["acc"].append(q_acc.item())
            all_quad["ece"].append(q_ece)
            all_quad["mce"].append(q_mce)
            all_quad["brier"].append(q_brier)
            all_quad["disagree"].append(q_disagree)
            print(f"run_{i+1}  quad_head  acc: {q_acc:.4f}  ECE: {q_ece:.4f}  "
                  f"MCE: {q_mce:.4f}  Brier: {q_brier:.4f}  disagree: {q_disagree:.4f}")

    run_sum = sum(sum(t) for t in time_results)
    epoch_count = sum(len(t) for t in time_results)
    print(f"avg time per run: {run_sum / args.runs:.2f}s")
    print(f"avg time per epoch: {1000 * run_sum / epoch_count:.1f}ms")
    print(f"mean_head test acc mean (%) = {np.mean(all_mean['acc']) * 100:.2f} ± {np.std(all_mean['acc']) * 100:.2f}")
    print(f"mean_head test ECE mean = {np.mean(all_mean['ece']):.4f} ± {np.std(all_mean['ece']):.4f}")
    print(f"mean_head test MCE mean = {np.mean(all_mean['mce']):.4f} ± {np.std(all_mean['mce']):.4f}")
    print(f"mean_head test Brier mean = {np.mean(all_mean['brier']):.4f} ± {np.std(all_mean['brier']):.4f}")
    if all_quad["acc"]:
        print(f"quad_head test acc mean (%) = {np.mean(all_quad['acc']) * 100:.2f} ± {np.std(all_quad['acc']) * 100:.2f}")
        print(f"quad_head test ECE mean = {np.mean(all_quad['ece']):.4f} ± {np.std(all_quad['ece']):.4f}")
        print(f"quad_head test MCE mean = {np.mean(all_quad['mce']):.4f} ± {np.std(all_quad['mce']):.4f}")
        print(f"quad_head test Brier mean = {np.mean(all_quad['brier']):.4f} ± {np.std(all_quad['brier']):.4f}")
        print(f"quad_head disagree mean = {np.mean(all_quad['disagree']):.4f} ± {np.std(all_quad['disagree']):.4f}")


if __name__ == "__main__":
    main()
