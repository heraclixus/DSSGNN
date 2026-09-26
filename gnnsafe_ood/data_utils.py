"""Data utilities for GNNSafe OOD pipeline."""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

from .scores import score_variant_suffix


def rand_splits(node_idx, train_prop=0.5, valid_prop=0.25):
    n = node_idx.size(0)
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)
    perm = torch.as_tensor(np.random.permutation(n))
    return {
        "train": node_idx[perm[:train_num]],
        "valid": node_idx[perm[train_num : train_num + valid_num]],
        "test": node_idx[perm[train_num + valid_num :]],
    }


def to_sparse_tensor(edge_index, edge_feat, num_nodes):
    """Convert edge_index + edge_feat to SparseTensor for ogbn-proteins."""
    try:
        from torch_sparse import SparseTensor
    except ImportError:
        raise ImportError("torch_sparse required for ogbn-proteins. Install: pip install torch-sparse")
    row, col = edge_index
    N, E = num_nodes, edge_index.size(1)
    perm = (col * N + row).argsort()
    row, col = row[perm], col[perm]
    value = edge_feat[perm]
    adj_t = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N), is_sorted=True)
    adj_t.storage.rowptr()
    adj_t.storage.csr2csc()
    return adj_t


def stable_cumsum(arr, rtol=1e-5, atol=1e-8):
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError("cumsum unstable")
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if pos_label is None:
        pos_label = 1
    y_true = (y_true == pos_label)
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    recall = tps / tps[-1]
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    recall, fps, tps = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0]
    cutoff = np.argmin(np.abs(recall - recall_level))
    return fps[cutoff] / max(np.sum(np.logical_not(y_true)), 1), None


def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos).reshape((-1, 1))
    neg = np.array(_neg).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[: len(pos)] += 1
    auroc = roc_auc_score(labels, examples)
    aupr = average_precision_score(labels, examples)
    fpr, _ = fpr_and_fdr_at_recall(labels, examples, recall_level)
    return auroc, aupr, fpr, None


def eval_acc(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    if y_pred.shape != y_true.shape:
        y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()
    is_labeled = y_true == y_true
    correct = y_true[is_labeled] == y_pred[is_labeled]
    return float(np.sum(correct)) / max(len(correct), 1)


def eval_rocauc(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        y_pred = F.softmax(y_pred, dim=-1)[:, 1].unsqueeze(1).detach().cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()
    rocauc_list = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])
            rocauc_list.append(score)
    if len(rocauc_list) == 0:
        raise RuntimeError("No positively labeled data for ROC-AUC.")
    return sum(rocauc_list) / len(rocauc_list)


@torch.no_grad()
def evaluate_classify(model, dataset, eval_func, criterion, args, device):
    model.eval()
    train_idx, valid_idx, test_idx = dataset.splits["train"], dataset.splits["valid"], dataset.splits["test"]
    y = dataset.y
    out = model(dataset, device).cpu()
    train_score = eval_func(y[train_idx], out[train_idx])
    valid_score = eval_func(y[valid_idx], out[valid_idx])
    test_score = eval_func(y[test_idx], out[test_idx])
    if args.dataset in ("proteins", "ppi"):
        valid_loss = criterion(out[valid_idx], y[valid_idx].to(torch.float))
    else:
        valid_out = F.log_softmax(out[valid_idx], dim=1)
        valid_loss = criterion(valid_out, y[valid_idx].squeeze(1))
    return train_score, valid_score, test_score, valid_loss


def evaluate_detect(model, dataset_ind, dataset_ood, criterion, eval_func, args, device, return_score=False):
    model.eval()
    with torch.no_grad():
        test_ind_score = model.detect(dataset_ind, dataset_ind.splits["test"], device, args).cpu()

    if isinstance(dataset_ood, list):
        result = []
        for d in dataset_ood:
            with torch.no_grad():
                test_ood_score = model.detect(d, d.node_idx, device, args).cpu()
            auroc, aupr, fpr, _ = get_measures(test_ind_score, test_ood_score)
            result += [auroc, aupr, fpr]
    else:
        with torch.no_grad():
            test_ood_score = model.detect(dataset_ood, dataset_ood.node_idx, device, args).cpu()
        auroc, aupr, fpr, _ = get_measures(test_ind_score, test_ood_score)
        result = [auroc, aupr, fpr]

    # no_grad + scalar valid_loss: this runs every epoch and the result list is
    # kept by the logger, so a graph-attached valid_loss retains the full GPU
    # forward graph per epoch (OOMs any GPU on large graphs / wide backbones).
    with torch.no_grad():
        out = model(dataset_ind, device).cpu()
    test_idx = dataset_ind.splits["test"]
    test_score = eval_func(dataset_ind.y[test_idx], out[test_idx])
    valid_idx = dataset_ind.splits["valid"]
    if args.dataset in ("proteins", "ppi"):
        valid_loss = criterion(out[valid_idx], dataset_ind.y[valid_idx].to(torch.float)).item()
    else:
        valid_out = F.log_softmax(out[valid_idx], dim=1)
        valid_loss = criterion(valid_out, dataset_ind.y[valid_idx].squeeze(1)).item()
    # Brier score on ID test nodes
    test_probs = F.softmax(out[test_idx], dim=1)
    test_labels = dataset_ind.y[test_idx].squeeze(1)
    one_hot = F.one_hot(test_labels, num_classes=test_probs.shape[1]).float()
    test_brier = ((test_probs - one_hot) ** 2).sum(dim=1).mean().item()
    result += [test_score, valid_loss, test_brier]

    if return_score:
        return result, test_ind_score, test_ood_score if not isinstance(dataset_ood, list) else None
    return result


def save_result(results, args):
    import os
    os.makedirs("results", exist_ok=True)
    if args.dataset in ("cora", "amazon-photo", "coauthor-cs"):
        filename = f"results/{args.dataset}-{args.ood_type}.csv"
    else:
        filename = f"results/{args.dataset}.csv"
    if args.method == "gnnsafe":
        reg_score_type = getattr(args, "reg_score_type", "energy")
        if args.use_reg and reg_score_type == "chaos":
            name = "chaos_safe" if args.use_prop else "chaos_ft"
        elif args.use_reg and reg_score_type == "pred_entropy":
            name = "pred_entropy_safe" if args.use_prop else "pred_entropy_ft"
        elif args.use_reg and reg_score_type == "mutual_info":
            name = "mutual_info_safe" if args.use_prop else "mutual_info_ft"
        else:
            name = "gnnsafe++" if (args.use_prop and args.use_reg) else "gnnsafe" if args.use_prop else "energy_ft" if args.use_reg else "energy"
    else:
        name = args.method
    name = f"{name}_{args.backbone}"
    suffix = score_variant_suffix(args)
    if suffix:
        name = f"{name}_{suffix}"
    with open(filename, "a+") as f:
        f.write(f"{name}\n")
        for k in range(results.shape[1] // 3):
            r = results[:, k * 3]
            f.write(f"OOD Test {k + 1} AUROC: {r.mean():.2f} ")
            r = results[:, k * 3 + 1]
            f.write(f"AUPR: {r.mean():.2f} ")
            r = results[:, k * 3 + 2]
            f.write(f"FPR: {r.mean():.2f}\n")
        r = results[:, -1]
        f.write(f"IND Test Score: {r.mean():.2f}\n")
