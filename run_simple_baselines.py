"""
Simple GNN baselines (GCN, GAT, GraphTransformer) for calibration comparison.
Reports accuracy, ECE, MCE, Brier -- same protocol as dssgnn_training.py.

Usage:
  python run_simple_baselines.py --model gcn --dataset cora --runs 10
  python run_simple_baselines.py --model gat --dataset cora --runs 10
  python run_simple_baselines.py --model transformer --dataset cora --runs 10
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
from torch_geometric.utils import from_scipy_sparse_matrix

from tfe_utils import set_seed, accuracy, load_data, compute_ece, compute_mce, compute_brier


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, out_dim))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, heads=8, dropout=0.6):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_dim, hidden_dim // heads, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout))
        self.convs.append(GATConv(hidden_dim, out_dim, heads=1, concat=False, dropout=dropout))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GraphTransformer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, heads=4, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(TransformerConv(in_dim, hidden_dim // heads, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout))
        self.convs.append(TransformerConv(hidden_dim, out_dim, heads=1, concat=False, dropout=dropout))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(model, optimizer, x, edge_index, labels, train_mask):
    model.train()
    optimizer.zero_grad()
    logits = model(x, edge_index)
    loss = F.cross_entropy(logits[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, x, edge_index, labels, mask):
    model.eval()
    logits = model(x, edge_index)
    probs = F.softmax(logits, dim=-1)
    acc = accuracy(logits[mask], labels[mask])
    ece = compute_ece(probs[mask].cpu(), labels[mask].cpu())
    mce = compute_mce(probs[mask].cpu(), labels[mask].cpu())
    brier = compute_brier(probs[mask].cpu(), labels[mask].cpu())
    loss = F.cross_entropy(logits[mask], labels[mask]).item()
    return acc, ece, mce, brier, loss


@torch.no_grad()
def predict_probs(model, x, edge_index):
    model.eval()
    return F.softmax(model(x, edge_index), dim=-1)


@torch.no_grad()
def mc_dropout_probs(model, x, edge_index, n_samples):
    # dropout stays active at inference; average the predictive distribution
    model.train()
    probs = None
    for _ in range(n_samples):
        p = F.softmax(model(x, edge_index), dim=-1)
        probs = p if probs is None else probs + p
    return probs / n_samples


def metrics_from_probs(probs, labels, mask):
    preds = probs.argmax(dim=-1)
    acc = (preds[mask] == labels[mask]).float().mean().item()
    ece = compute_ece(probs[mask].cpu(), labels[mask].cpu())
    mce = compute_mce(probs[mask].cpu(), labels[mask].cpu())
    brier = compute_brier(probs[mask].cpu(), labels[mask].cpu())
    return float(acc), float(ece), float(mce), float(brier)


def build_model(args, in_dim, num_classes):
    if args.model == "gcn":
        return GCN(in_dim, args.hidden, num_classes, num_layers=args.layers, dropout=args.dropout)
    elif args.model == "gat":
        return GAT(in_dim, args.hidden, num_classes, num_layers=args.layers, heads=args.heads, dropout=args.dropout)
    elif args.model == "transformer":
        return GraphTransformer(in_dim, args.hidden, num_classes, num_layers=args.layers, heads=args.heads, dropout=args.dropout)
    else:
        raise ValueError(f"Unknown model: {args.model}")


def train_to_best(model, args, x, edge_index, labels, train_mask, val_mask, device):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    for epoch in range(args.epochs):
        train_epoch(model, optimizer, x, edge_index, labels, train_mask)
        _, _, _, _, val_loss = evaluate(model, x, edge_index, labels, val_mask)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model


def run_single(args, run_idx):
    set_seed(42 + run_idx)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    adj, features, labels, train_mask, val_mask, test_mask = load_data(
        args.dataset, full=True, random_split=True,
        train_rate=0.6, val_rate=0.2, i=run_idx,
    )

    # Convert adj to edge_index -- adj may be scipy sparse or torch sparse
    import scipy.sparse as sp
    if isinstance(adj, torch.Tensor):
        if adj.is_sparse:
            adj_coo = adj.coalesce()
            edge_index = adj_coo.indices()
        else:
            edge_index, _ = from_scipy_sparse_matrix(sp.csr_matrix(adj.numpy()))
    else:
        edge_index, _ = from_scipy_sparse_matrix(adj)
    x = features.to(device) if isinstance(features, torch.Tensor) else torch.tensor(features, dtype=torch.float32).to(device)
    labels = labels.to(device) if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.long).to(device)
    edge_index = edge_index.to(device)
    train_mask = train_mask.to(device) if isinstance(train_mask, torch.Tensor) else torch.tensor(train_mask, dtype=torch.bool).to(device)
    val_mask = val_mask.to(device) if isinstance(val_mask, torch.Tensor) else torch.tensor(val_mask, dtype=torch.bool).to(device)
    test_mask = test_mask.to(device) if isinstance(test_mask, torch.Tensor) else torch.tensor(test_mask, dtype=torch.bool).to(device)

    in_dim = x.shape[1]
    num_classes = int(labels.max().item()) + 1

    model = build_model(args, in_dim, num_classes).to(device)
    model = train_to_best(model, args, x, edge_index, labels, train_mask, val_mask, device)

    if args.ensemble_size > 1:
        # deep ensemble: average softmax probabilities of independently
        # initialized and trained members (first member is the model above)
        probs_sum = predict_probs(model, x, edge_index)
        for member in range(1, args.ensemble_size):
            set_seed(1000 * (run_idx + 1) + member)
            m = build_model(args, in_dim, num_classes).to(device)
            m = train_to_best(m, args, x, edge_index, labels, train_mask, val_mask, device)
            probs_sum = probs_sum + predict_probs(m, x, edge_index)
        probs = probs_sum / args.ensemble_size
        return metrics_from_probs(probs, labels, test_mask)

    if args.mc_samples > 0:
        probs = mc_dropout_probs(model, x, edge_index, args.mc_samples)
        return metrics_from_probs(probs, labels, test_mask)

    test_acc, test_ece, test_mce, test_brier, _ = evaluate(model, x, edge_index, labels, test_mask)
    return float(test_acc), float(test_ece), float(test_mce), float(test_brier)


def main():
    parser = argparse.ArgumentParser(description="Simple GNN baselines for calibration")
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "gat", "transformer"])
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8, help="Attention heads for GAT/Transformer")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--mc_samples", type=int, default=0,
                        help="If >0, MC-dropout at test: average probs over this many stochastic passes")
    parser.add_argument("--ensemble_size", type=int, default=1,
                        help="If >1, deep ensemble: average probs of this many independently trained members")
    args = parser.parse_args()
    print(args)

    accs, eces, mces, briers = [], [], [], []
    for run in range(args.runs):
        acc, ece, mce, brier = run_single(args, run)
        accs.append(acc)
        eces.append(ece)
        mces.append(mce)
        briers.append(brier)
        print(f"  run {run}: acc={acc*100:.2f}% ece={ece:.4f} brier={brier:.4f}")

    accs_pct = [a * 100 for a in accs]
    method = args.model
    if args.mc_samples > 0:
        method += f"_mc{args.mc_samples}"
    if args.ensemble_size > 1:
        method += f"_ens{args.ensemble_size}"
    print(f"\nRESULT model={method} dataset={args.dataset} "
          f"acc_mean={np.mean(accs_pct):.2f} acc_std={np.std(accs_pct):.2f} "
          f"brier_mean={np.mean(briers):.4f} brier_std={np.std(briers):.4f} "
          f"ece_mean={np.mean(eces):.4f} ece_std={np.std(eces):.4f}")
    print(f"test acc mean (%) = {np.mean(accs_pct):.2f} \u00b1 {np.std(accs_pct):.2f}")
    print(f"test ECE mean = {np.mean(eces):.4f} \u00b1 {np.std(eces):.4f}")
    print(f"test MCE mean = {np.mean(mces):.4f} \u00b1 {np.std(mces):.4f}")
    print(f"test Brier mean = {np.mean(briers):.4f} \u00b1 {np.std(briers):.4f}")


if __name__ == "__main__":
    main()
