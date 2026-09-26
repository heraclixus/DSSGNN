"""
DS-GNN training script for node classification.

Reuses data loading and utilities from tfe_utils.
"""

import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
import time
import argparse
from tqdm import tqdm

from tfe_utils import set_seed, accuracy, load_data, propagate_adj, random_walk_adj, compute_ece, compute_mce, compute_brier
from dssgnn.chebyshev import build_rescaled_laplacian
from dssgnn.model import DSSGNN, DSSGNNTFE
from dssgnn.dsconv_tfe_propfirst import DSSGNNTFEPropFirst


def _gate_reg_loss(model, use_tfe):
    """L_op: penalize higher-order gate coefficients (r>=1) for S+G."""
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    if getattr(model, "use_random_gates", False):
        if hasattr(model, "a_coeffs") and model.a_coeffs is not None:
            loss = loss + (model.a_coeffs[:, 1:] ** 2).sum() + (model.b_coeffs[:, 1:] ** 2).sum()
    for m in model.modules():
        if hasattr(m, "a_coeffs") and m.a_coeffs is not None and m is not model:
            loss = loss + (m.a_coeffs[:, 1:] ** 2).sum() + (m.b_coeffs[:, 1:] ** 2).sum()
    return loss


def _filter_reg_loss(model):
    """L_filter: penalize higher-order filter coefficients (r>=1) for S+F."""
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    if getattr(model, "use_random_filter", False):
        if hasattr(model, "gamma_lp") and model.gamma_lp is not None:
            loss = loss + (model.gamma_lp[:, 1:] ** 2).sum() + (model.gamma_hp[:, 1:] ** 2).sum()
    for m in model.modules():
        if hasattr(m, "gamma_lp") and m.gamma_lp is not None and m is not model:
            loss = loss + (m.gamma_lp[:, 1:] ** 2).sum() + (m.gamma_hp[:, 1:] ** 2).sum()
    return loss


def train(model, optimizer, graph_input, x, y, mask, lambda_reg=0.01, lambda_op=0.0, lambda_filter=0.0, scaler=None, use_tfe=False, loss_type="mean"):
    """
    Train one epoch.
    loss_type:
      "mean"  — L_sup(Z_0, Y) + λ_reg * chaos_energy  (current default)
      "quad"  — sum_s w_s * L(z^(s), Y) + λ_reg * chaos_energy  (quadrature-averaged)
    """
    model.train()
    optimizer.zero_grad()
    use_amp = scaler is not None
    with torch.cuda.amp.autocast(enabled=use_amp):
        if loss_type == "quad" and not use_tfe and hasattr(model, "forward_with_sample_logits"):
            # Quadrature-averaged loss: exact integration over chaos realizations
            logits, sample_logits, weights, uncertainty = model.forward_with_sample_logits(graph_input, x)
            # sample_logits: (S, N, C), weights: (S,)
            S = sample_logits.shape[0]
            train_mask = mask[0]
            # Compute per-sample loss and average with quadrature weights
            per_sample_loss = torch.stack([
                F.cross_entropy(F.log_softmax(sample_logits[s][train_mask], dim=1), y[train_mask])
                for s in range(S)
            ])
            w = weights.to(per_sample_loss.device, dtype=per_sample_loss.dtype)
            loss = (w * per_sample_loss).sum()
            loss = loss + lambda_reg * uncertainty.mean()
        else:
            # Standard mean-logit loss
            if use_tfe:
                adj_lp, adj_hp = graph_input
                logits, uncertainty = model(adj_lp, adj_hp, x, return_uncertainty=True)
            else:
                logits, uncertainty = model(graph_input, x, return_uncertainty=True)
            out = F.log_softmax(logits, dim=1)
            loss = F.cross_entropy(out[mask[0]], y[mask[0]])
            loss = loss + lambda_reg * uncertainty.mean()
        if lambda_op > 0:
            loss = loss + lambda_op * _gate_reg_loss(model, use_tfe)
        if lambda_filter > 0:
            loss = loss + lambda_filter * _filter_reg_loss(model)
    if use_amp:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()


def test(model, graph_input, x, y, mask, use_tfe=False):
    """Evaluate on train/val/test splits."""
    model.eval()
    with torch.no_grad():
        if use_tfe:
            adj_lp, adj_hp = graph_input
            logits = model(adj_lp, adj_hp, x)
        else:
            logits = model(graph_input, x)
    logits = F.log_softmax(logits, dim=1)
    accs, losses = [], []
    for i in range(3):
        acc = accuracy(logits[mask[i]], y[mask[i]])
        loss = F.cross_entropy(logits[mask[i]], y[mask[i]])
        accs.append(acc)
        losses.append(loss)
    return accs, losses, logits


def quad_predictive(model, graph_input, x, use_tfe=False, S=4):
    """Quadrature-averaged predictive p_bar_i = sum_s w_s softmax(z_i^(s)) (Eq. 8)."""
    model.eval()
    with torch.no_grad():
        if use_tfe:
            adj_lp, adj_hp = graph_input
            _, sample_logits, weights, _ = model.forward_with_sample_logits(adj_lp, adj_hp, x, S=S)
        else:
            _, sample_logits, weights, _ = model.forward_with_sample_logits(graph_input, x)
        w = weights.to(device=sample_logits.device, dtype=sample_logits.dtype)
        p_bar = torch.einsum("s,snc->nc", w, F.softmax(sample_logits, dim=-1))
        p_bar = p_bar.clamp_min(0)
        p_bar = p_bar / p_bar.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return p_bar


# Large datasets that may cause CUDA OOM; use reduced config when --reduced_memory
LARGE_DATASETS = {"physics", "cs", "cora-full", "tolokers", "questions", "roman-empire", "amazon-ratings"}
# Heterophilous datasets: use TFE-style propagation and stronger settings for better performance
HETEROPHILOUS_DATASETS = {"chameleon", "squirrel", "texas", "cornell", "wisconsin",
                          "roman-empire", "amazon-ratings", "minesweeper", "tolokers", "questions"}
# Squirrel: larger, noisier; needs stronger regularization and longer training
SQUIRREL_CHAMELEON = {"chameleon", "squirrel"}
# Squirrel-only: most aggressive settings (highest dropout, longest patience)
SQUIRREL_ONLY = {"squirrel"}


def run(args, dataset, full, random_split, i):
    if args.random_split:
        set_seed(args.seed)
    else:
        set_seed(i)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    # Apply reduced-memory config for large datasets
    hidden = args.hidden
    K_lp, K_hp, P = args.K_lp, args.K_hp, args.P
    pro_dropout = args.pro_dropout
    use_tfe = args.propagation == "tfe" or (args.heterophily_tfe and dataset in HETEROPHILOUS_DATASETS)
    use_prop_first = args.propagate_first and (use_tfe or dataset in HETEROPHILOUS_DATASETS)

    if args.reduced_memory and dataset in LARGE_DATASETS:
        hidden = min(hidden, 32)
        K_lp, K_hp = min(K_lp, 2), min(K_hp, 2)
        P = min(P, 1)
    elif (use_tfe or use_prop_first) and dataset in HETEROPHILOUS_DATASETS:
        # TFE-style: deeper filters, larger hidden, higher dropout (align with TFE-GNN)
        K_lp, K_hp = 6, 5
        hidden = max(hidden, 512) if use_prop_first else max(hidden, 128)
        pro_dropout = max(pro_dropout, 0.8) if dataset in SQUIRREL_ONLY else (
            max(pro_dropout, 0.7) if dataset in SQUIRREL_CHAMELEON else max(pro_dropout, 0.6))

    # Layers: optional override for heterophilous datasets (squirrel, cornell, PyG heterophilous)
    HETEROPHILOUS_LAYERS = {"squirrel", "cornell", "roman-empire", "amazon-ratings", "minesweeper", "tolokers", "questions"}
    num_layers = (args.layers_heterophilous if getattr(args, "layers_heterophilous", None) is not None
                  and dataset in HETEROPHILOUS_LAYERS else args.layers)

    # Reuse load_data from tfe_utils
    adj, features, labels, train_mask, val_mask, test_mask = load_data(
        dataset, full, random_split, args.train_rate, args.val_rate, i
    )

    num_classes = int(max(labels)) + 1

    # Patience: longer for heterophilous (squirrel/chameleon need more epochs)
    patience = getattr(args, "patience_heterophilous", args.patience)
    if (use_tfe or use_prop_first) and dataset in SQUIRREL_ONLY:
        patience = max(patience, 400)
    elif (use_tfe or use_prop_first) and dataset in SQUIRREL_CHAMELEON:
        patience = max(patience, 300)

    if use_prop_first:
        # TFE's propagate-once structure: prop -> combine -> MLP (best for heterophily)
        combine = args.combine if dataset in HETEROPHILOUS_DATASETS else "sum"
        model = DSSGNNTFEPropFirst(
            input_dim=features.shape[1],
            hidden_dim=hidden,
            out_dim=num_classes,
            num_layers=num_layers,
            K_lp=K_lp,
            K_hp=K_hp,
            P=P,
            dropout=(pro_dropout, args.lin_dropout),
            activation=True,
            combine=combine,
            use_ense_coe=getattr(args, "use_ense_coe", False),
            use_random_gates=getattr(args, "use_random_gates", False),
            use_random_filter=getattr(args, "use_random_filter", False),
            P_gate=getattr(args, "P_gate", 1),
            P_filter=getattr(args, "P_filter", 1),
            use_bn=getattr(args, "use_bn", False),
        )
        gf = getattr(args, "gf", "sym")
        if gf == "rw":
            adj_lp = random_walk_adj(adj, "low", -1.0).to(device)
            adj_hp = random_walk_adj(adj, "high", -1.0).to(device)
        else:
            adj_lp = propagate_adj(adj, "low", -0.5, -0.5).to(device)
            adj_hp = propagate_adj(adj, "high", args.eta, args.eta).to(device)
        graph_input = (adj_lp, adj_hp)
    elif use_tfe:
        model = DSSGNNTFE(
            input_dim=features.shape[1],
            hidden_dim=hidden,
            out_dim=num_classes,
            num_layers=num_layers,
            K_lp=K_lp,
            K_hp=K_hp,
            P=P,
            dropout=(pro_dropout, args.lin_dropout),
            activation=True,
            use_random_gates=getattr(args, "use_random_gates", False),
            use_random_filter=getattr(args, "use_random_filter", False),
            P_gate=getattr(args, "P_gate", 1),
            P_filter=getattr(args, "P_filter", 1),
            use_bn=getattr(args, "use_bn", False),
        )
        adj_lp = propagate_adj(adj, "low", -0.5, -0.5).to(device)
        adj_hp = propagate_adj(adj, "high", args.eta, args.eta).to(device)
        graph_input = (adj_lp, adj_hp)
    else:
        model = DSSGNN(
            input_dim=features.shape[1],
            hidden_dim=hidden,
            out_dim=num_classes,
            num_layers=num_layers,
            K_lp=K_lp,
            K_hp=K_hp,
            P=P,
            dropout=(pro_dropout, args.lin_dropout),
            activation=True,
            use_random_gates=getattr(args, "use_random_gates", False),
            P_gate=getattr(args, "P_gate", 1),
            S=args.S,
            act_fn=getattr(args, "act_fn", "relu"),
            use_bn=getattr(args, "use_bn", False),
        )
        graph_input = build_rescaled_laplacian(adj, lambda_max=args.lambda_max).to(device)

    # Filter-branch isolation (--branch lp|hp): zero and freeze the other branch's
    # combine gates so only one of the low/high-pass branches contributes. Gradients
    # cannot reach the disabled branch's filter coefficients through a zero gate.
    if getattr(args, "branch", "dual") != "dual":
        kill = "beta" if args.branch == "lp" else "alpha"
        killed = 0
        for m in model.modules():
            p = getattr(m, kill, None)
            if isinstance(p, torch.nn.Parameter):
                with torch.no_grad():
                    p.zero_()
                p.requires_grad_(False)
                killed += 1
        assert killed > 0, f"--branch {args.branch}: no '{kill}' gate parameters found"

    # TFE-style optimizer: higher lr for spectral coeffs when using propagate-first
    opt_name = getattr(args, "optimizer_prop", "Adam")
    if use_prop_first:
        gate_params = (
            [model.alpha, model.beta] if model.alpha is not None
            else ([model.a_coeffs, model.b_coeffs] if getattr(model, "use_random_gates", False) else [model.ense_coe])
        )
        filter_params = (
            [model.gamma_lp, model.gamma_hp] if getattr(model, "use_random_filter", False)
            else [model.coeffs_lp, model.coeffs_hp]
        )
        param_groups = [
            {"params": filter_params[0], "lr": args.lr_adaptive, "weight_decay": args.wd_adaptive},
            {"params": filter_params[1], "lr": args.lr_adaptive, "weight_decay": args.wd_adaptive},
            {"params": gate_params, "lr": args.lr_adaptive, "weight_decay": args.wd_adaptive},
            {"params": model.layers.parameters(), "lr": args.lr_lin, "weight_decay": args.wd_lin},
        ]
        if opt_name == "RMSprop":
            optimizer = optim.RMSprop(param_groups)
        else:
            optimizer = optim.Adam(param_groups)
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.wd,
        )

    model = model.to(device)
    features = features.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    train_mask = train_mask.clone().detach().to(device)
    val_mask = val_mask.clone().detach().to(device)
    test_mask = test_mask.clone().detach().to(device)
    use_tfe_forward = use_tfe or use_prop_first  # both use adj_lp, adj_hp
    if not use_tfe_forward:
        graph_input = graph_input.to(device)
    mask = [train_mask, val_mask, test_mask]

    best_val_loss = float("inf")
    test_acc = 0
    bad_epoch = 0
    run_time = []
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp) if torch.cuda.is_available() and args.amp else None

    quad_supported = hasattr(model, "forward_with_sample_logits") and getattr(args, "eval_quad_head", True)
    quad_metrics = {
        "acc": float("nan"), "ece": float("nan"), "mce": float("nan"),
        "brier": float("nan"), "disagree": float("nan"),
    }
    # Validation-split metrics at the selected (best-validation-loss) epoch; the
    # validation Brier is the model-selection criterion reported in the paper's
    # per-dataset selection table.
    val_metrics = {"acc": float("nan"), "brier": float("nan"), "loss": float("nan")}

    for epoch in range(args.epochs):
        t0 = time.time()
        train(model, optimizer, graph_input, features, labels, mask,
              lambda_reg=args.lambda_reg,
              lambda_op=args.lambda_op if getattr(args, "use_random_gates", False) else 0.0,
              lambda_filter=args.lambda_filter if getattr(args, "use_random_filter", False) else 0.0,
              scaler=scaler, use_tfe=use_tfe_forward,
              loss_type=getattr(args, "loss_type", "mean"))
        run_time.append(time.time() - t0)

        [train_acc, val_acc, tmp_test_acc], [train_loss, val_loss, tmp_test_loss], logits = test(
            model, graph_input, features, labels, mask, use_tfe=use_tfe_forward
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            test_ece = compute_ece(logits[test_mask], labels[test_mask]).item()
            test_mce = compute_mce(logits[test_mask], labels[test_mask]).item()
            test_brier = compute_brier(logits[test_mask], labels[test_mask]).item()
            val_metrics["acc"] = float(val_acc)
            val_metrics["brier"] = compute_brier(logits[val_mask], labels[val_mask]).item()
            val_metrics["loss"] = float(val_loss)
            if quad_supported:
                p_bar = quad_predictive(model, graph_input, features, use_tfe=use_tfe_forward, S=args.S)
                quad_metrics["acc"] = accuracy(p_bar[test_mask], labels[test_mask]).item()
                quad_metrics["ece"] = compute_ece(p_bar[test_mask], labels[test_mask]).item()
                quad_metrics["mce"] = compute_mce(p_bar[test_mask], labels[test_mask]).item()
                quad_metrics["brier"] = compute_brier(p_bar[test_mask], labels[test_mask]).item()
                quad_metrics["disagree"] = (
                    (p_bar.argmax(dim=1) != logits.argmax(dim=1))[test_mask].float().mean().item()
                )
            bad_epoch = 0
        else:
            bad_epoch += 1

        if bad_epoch == patience:
            break

    del model, optimizer, graph_input, features, labels, train_mask, val_mask, test_mask, mask
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return test_acc, test_ece, test_mce, test_brier, best_val_loss, run_time, quad_metrics, val_metrics


def main():
    parser = argparse.ArgumentParser(description="DSS-GNN node classification (method.tex Algorithm 1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="cora, citeseer, pubmed, texas, cornell, wisconsin, chameleon, squirrel, roman-empire, amazon-ratings, minesweeper, tolokers, questions",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--runs", type=int, default=10)

    parser.add_argument("--K_lp", type=int, default=4, help="Chebyshev degree low-pass")
    parser.add_argument("--K_hp", type=int, default=4, help="Chebyshev degree high-pass")
    parser.add_argument("--P", type=int, default=2, help="Chaos truncation order")
    parser.add_argument("--branch", type=str, default="dual", choices=["dual", "lp", "hp"],
                        help="Filter-branch isolation: lp = low-pass only (high-pass gates zeroed+frozen), hp = high-pass only")
    parser.add_argument("--S", type=int, default=4, help="Number of Gauss-Hermite quadrature nodes")
    parser.add_argument("--loss_type", type=str, default="mean", choices=["mean", "quad"],
                        help="Training loss: mean (loss on Z_0) or quad (quadrature-averaged loss)")
    parser.add_argument("--lambda_max", type=float, default=2.0)
    parser.add_argument("--lambda_reg", type=float, default=0.01, help="Regularization on higher-order channels")

    parser.add_argument("--pro_dropout", type=float, default=0.5)
    parser.add_argument("--lin_dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--lr_adaptive", type=float, default=0.1, help="TFE-style lr for spectral coeffs (prop-first)")
    parser.add_argument("--wd_adaptive", type=float, default=0.05)
    parser.add_argument("--lr_lin", type=float, default=0.005, help="TFE-style lr for linear layers (prop-first)")
    parser.add_argument("--wd_lin", type=float, default=0.0)

    parser.add_argument("--full", type=bool, default=True)
    parser.add_argument("--random_split", type=bool, default=True)
    parser.add_argument("--use_bn", action="store_true",
                        help="Per-chaos-channel BatchNorm1d between hidden layers "
                             "(opt-in; default off preserves the published architecture)")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision (FP16) to reduce memory")
    parser.add_argument("--reduced_memory", action="store_true",
                        help="Use smaller hidden/K/P for large datasets (physics, cs, cora-full)")
    parser.add_argument("--propagation", type=str, default="chebyshev", choices=["chebyshev", "tfe"],
                        help="Graph propagation: chebyshev (Laplacian) or tfe (adjacency, aligns with TFE-GNN)")
    parser.add_argument("--heterophily_tfe", action="store_true",
                        help="Use TFE-style propagation on heterophilous datasets (chameleon, squirrel, etc.)")
    parser.add_argument("--propagate_first", action="store_true",
                        help="Use TFE propagate-once structure (prop -> combine -> MLP) on heterophilous datasets")
    parser.add_argument("--combine", type=str, default="con", choices=["sum", "con"],
                        help="TFE combine mode for prop-first: sum or con (concat)")
    parser.add_argument("--eta", type=float, default=-0.3, help="Exponent for TFE high-pass (propagate_adj)")
    parser.add_argument("--gf", type=str, default="sym", choices=["sym", "rw"],
                        help="Graph filter: sym (symmetric) or rw (random walk), for prop-first")
    parser.add_argument("--patience_heterophilous", type=int, default=200,
                        help="Patience for heterophilous; overridden to 300 for squirrel/chameleon")
    parser.add_argument("--optimizer_prop", type=str, default="Adam", choices=["Adam", "RMSprop"],
                        help="Optimizer for prop-first: Adam or RMSprop (TFE uses both)")
    parser.add_argument("--use_ense_coe", action="store_true",
                        help="Use TFE ense_coe (2 params) instead of per-order alpha/beta for prop-first")
    parser.add_argument("--use_random_gates", action="store_true",
                        help="S+G variant: stochastic alpha_n(omega), beta_n(omega) for epistemic uncertainty")
    parser.add_argument("--P_gate", type=int, default=1,
                        help="Chaos order for gate expansion (when use_random_gates)")
    parser.add_argument("--lambda_op", type=float, default=0.01,
                        help="Regularization on gate stochastic coefficients (S+G)")
    parser.add_argument("--use_random_filter", action="store_true",
                        help="S+F variant: stochastic c_k(omega) for graph filter coefficients")
    parser.add_argument("--P_filter", type=int, default=1,
                        help="Chaos order for filter expansion (when use_random_filter)")
    parser.add_argument("--lambda_filter", type=float, default=0.01,
                        help="Regularization on filter stochastic coefficients (S+F)")
    parser.add_argument("--layers_heterophilous", type=int, default=None,
                        help="Override num_layers for squirrel/cornell (e.g. 3 for deeper MLP)")
    parser.add_argument("--act_fn", type=str, default="relu", choices=["relu", "gelu", "tanh"],
                        help="Activation function for DSS layers")
    parser.add_argument("--no_quad_head", dest="eval_quad_head", action="store_false",
                        help="Skip quadrature-averaged predictive (Eq. 8) evaluation")
    parser.set_defaults(eval_quad_head=True)

    args = parser.parse_args()
    args.train_rate = 0.6 if args.full else 0.025
    args.val_rate = 0.2 if args.full else 0.025

    print(args)

    all_test_accs = []
    all_test_eces = []
    all_test_mces = []
    all_test_briers = []
    all_quad = {"acc": [], "ece": [], "mce": [], "brier": [], "disagree": []}
    all_val = {"acc": [], "brier": [], "loss": []}
    time_results = []

    for i in tqdm(range(args.runs)):
        test_acc, test_ece, test_mce, test_brier, _, run_time, quad_metrics, val_metrics = run(args, args.dataset, args.full, args.random_split, i)
        for k in all_val:
            all_val[k].append(val_metrics[k])
        all_test_accs.append(test_acc.item())
        all_test_eces.append(test_ece)
        all_test_mces.append(test_mce)
        all_test_briers.append(test_brier)
        for k in all_quad:
            all_quad[k].append(quad_metrics[k])
        time_results.append(run_time)
        print(f"run_{i+1}  acc: {test_acc:.4f}  ECE: {test_ece:.4f}  MCE: {test_mce:.4f}  Brier: {test_brier:.4f}")
        print(f"run_{i+1}  val acc: {val_metrics['acc']:.4f}  val Brier: {val_metrics['brier']:.4f}  val loss: {val_metrics['loss']:.4f}")
        if args.eval_quad_head:
            print(f"run_{i+1}  quad acc: {quad_metrics['acc']:.4f}  quad ECE: {quad_metrics['ece']:.4f}  "
                  f"quad MCE: {quad_metrics['mce']:.4f}  quad Brier: {quad_metrics['brier']:.4f}  "
                  f"disagree: {quad_metrics['disagree']:.4f}")

    run_sum = sum(sum(t) for t in time_results)
    epoch_count = sum(len(t) for t in time_results)
    print(f"avg time per run: {run_sum / args.runs:.2f}s")
    print(f"avg time per epoch: {1000 * run_sum / epoch_count:.1f}ms")
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"peak GPU memory allocated: {peak_mb:.1f} MiB")
    print(f"test acc mean (%) = {np.mean(all_test_accs) * 100:.2f} ± {np.std(all_test_accs) * 100:.2f}")
    print(f"test ECE mean = {np.mean(all_test_eces):.4f} ± {np.std(all_test_eces):.4f}")
    print(f"test MCE mean = {np.mean(all_test_mces):.4f} ± {np.std(all_test_mces):.4f}")
    print(f"test Brier mean = {np.mean(all_test_briers):.4f} ± {np.std(all_test_briers):.4f}")
    print(f"val acc mean (%) = {np.mean(all_val['acc']) * 100:.2f} ± {np.std(all_val['acc']) * 100:.2f}")
    print(f"val Brier mean = {np.mean(all_val['brier']):.4f} ± {np.std(all_val['brier']):.4f}")
    if args.eval_quad_head:
        print(f"test quad acc mean (%) = {np.mean(all_quad['acc']) * 100:.2f} ± {np.std(all_quad['acc']) * 100:.2f}")
        print(f"test quad ECE mean = {np.mean(all_quad['ece']):.4f} ± {np.std(all_quad['ece']):.4f}")
        print(f"test quad MCE mean = {np.mean(all_quad['mce']):.4f} ± {np.std(all_quad['mce']):.4f}")
        print(f"test quad Brier mean = {np.mean(all_quad['brier']):.4f} ± {np.std(all_quad['brier']):.4f}")
        print(f"quad vs mean argmax disagreement (%) = {np.mean(all_quad['disagree']) * 100:.2f} ± {np.std(all_quad['disagree']) * 100:.2f}")


if __name__ == "__main__":
    main()
