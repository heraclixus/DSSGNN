"""
OOD dataset loading for GNNSafe pipeline.

Ported from GraphOOD-GNNSafe. Returns (dataset_ind, dataset_ood_tr, dataset_ood_te).
"""

import csv
import json
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, Twitch
from torch_geometric.data import Data
from torch_geometric.utils import stochastic_blockmodel_graph, subgraph


def load_dataset(args):
    """
    Returns:
        dataset_ind: in-distribution (train/val/test splits)
        dataset_ood_tr: OOD training (for GNNSafe++ regularization)
        dataset_ood_te: OOD test (single Data or list)
    """
    if args.dataset == "twitch":
        return _load_twitch(args.data_dir)
    if args.dataset == "arxiv":
        return _load_arxiv(args.data_dir)
    if args.dataset == "proteins":
        return _load_proteins(args.data_dir)
    if args.dataset in ("cora", "citeseer", "pubmed", "amazon-photo", "amazon-computer", "coauthor-cs", "coauthor-physics"):
        return _load_graph_dataset(args.data_dir, args.dataset, args.ood_type)
    raise ValueError(f"Invalid dataset: {args.dataset}")


@contextmanager
def _ogb_torch_load_compat():
    """Temporarily restore torch.load(weights_only=False) for trusted OGB caches."""
    original_torch_load = torch.load

    def compat_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = compat_torch_load
    try:
        yield
    finally:
        torch.load = original_torch_load


def _load_twitch(data_dir):
    transform = T.NormalizeFeatures()
    local_root = Path(__file__).resolve().parents[1] / "new_data2" / "twitch"
    if local_root.exists():
        return _load_twitch_from_local_files(local_root, transform)

    subgraph_names = ["DE", "EN", "ES", "FR", "RU"]
    train_idx, valid_idx = 0, 1
    dataset_ood_te = []
    for i, name in enumerate(subgraph_names):
        dset = Twitch(root=f"{data_dir}Twitch", name=name, transform=transform)
        data = dset[0]
        data.node_idx = torch.arange(data.num_nodes)
        if i == train_idx:
            dataset_ind = data
        elif i == valid_idx:
            dataset_ood_tr = data
        else:
            dataset_ood_te.append(data)
    return dataset_ind, dataset_ood_tr, dataset_ood_te


def _load_twitch_from_local_files(root: Path, transform):
    subgraph_names = ["DE", "ENGB", "ES", "FR", "RU"]
    train_idx, valid_idx = 0, 1
    dataset_ood_te = []
    for i, name in enumerate(subgraph_names):
        data = _load_local_twitch_subgraph(root / name, name)
        data = transform(data)
        data.node_idx = torch.arange(data.num_nodes)
        if i == train_idx:
            dataset_ind = data
        elif i == valid_idx:
            dataset_ood_tr = data
        else:
            dataset_ood_te.append(data)
    return dataset_ind, dataset_ood_tr, dataset_ood_te


def _load_local_twitch_subgraph(folder: Path, name: str) -> Data:
    labels = []
    node_ids = []
    uniq_ids = set()
    target_path = folder / f"musae_{name}_target.csv"
    with target_path.open("r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            node_id = int(row[5])
            if node_id in uniq_ids:
                continue
            uniq_ids.add(node_id)
            labels.append(int(row[2] == "True"))
            node_ids.append(node_id)

    node_ids = np.asarray(node_ids, dtype=np.int64)
    num_nodes = node_ids.shape[0]

    src = []
    dst = []
    edge_path = folder / f"musae_{name}_edges.csv"
    with edge_path.open("r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            src.append(int(row[0]))
            dst.append(int(row[1]))

    features = np.zeros((num_nodes, 3170), dtype=np.float32)
    feature_path = folder / f"musae_{name}_features.json"
    with feature_path.open("r") as f:
        feat_dict = json.load(f)
    for node, feats in feat_dict.items():
        node_idx = int(node)
        if node_idx >= num_nodes:
            continue
        features[node_idx, np.asarray(feats, dtype=np.int64)] = 1.0

    inv_node_ids = {node_id: idx for idx, node_id in enumerate(node_ids)}
    reorder_node_ids = np.zeros_like(node_ids)
    for i in range(num_nodes):
        reorder_node_ids[i] = inv_node_ids[i]

    labels = np.asarray(labels, dtype=np.int64)[reorder_node_ids]
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long).view(-1, 1)
    return Data(x=x, edge_index=edge_index, y=y)


def _load_arxiv(data_dir, time_bound=(2015, 2017), inductive=True):
    from ogb.nodeproppred import NodePropPredDataset

    with _ogb_torch_load_compat():
        ogb_dataset = NodePropPredDataset(name="ogbn-arxiv", root=f"{data_dir}ogb")
    edge_index = torch.as_tensor(ogb_dataset.graph["edge_index"])
    node_feat = torch.as_tensor(ogb_dataset.graph["node_feat"])
    label = torch.as_tensor(ogb_dataset.labels).reshape(-1, 1)
    year = torch.as_tensor(ogb_dataset.graph["node_year"], dtype=torch.long)

    year_min, year_max = time_bound[0], time_bound[1]
    test_year_bound = [2017, 2018, 2019, 2020]

    center_node_mask = (year <= year_min).squeeze()
    if inductive:
        ind_edge_index, _ = subgraph(center_node_mask, edge_index)
    else:
        ind_edge_index = edge_index

    dataset_ind = Data(x=node_feat, edge_index=ind_edge_index, y=label)
    idx = torch.arange(label.size(0))
    dataset_ind.node_idx = idx[center_node_mask]

    center_node_mask = ((year <= year_max) & (year > year_min)).squeeze()
    if inductive:
        all_node_mask = (year <= year_max).squeeze()
        ood_tr_edge_index, _ = subgraph(all_node_mask, edge_index)
    else:
        ood_tr_edge_index = edge_index

    dataset_ood_tr = Data(x=node_feat, edge_index=ood_tr_edge_index, y=label)
    dataset_ood_tr.node_idx = idx[center_node_mask]

    dataset_ood_te = []
    for i in range(len(test_year_bound) - 1):
        center_node_mask = ((year <= test_year_bound[i + 1]) & (year > test_year_bound[i])).squeeze()
        if inductive:
            all_node_mask = (year <= test_year_bound[i + 1]).squeeze()
            ood_te_edge_index, _ = subgraph(all_node_mask, edge_index)
        else:
            ood_te_edge_index = edge_index
        data = Data(x=node_feat, edge_index=ood_te_edge_index, y=label)
        data.node_idx = idx[center_node_mask]
        dataset_ood_te.append(data)

    return dataset_ind, dataset_ood_tr, dataset_ood_te


def _load_proteins(data_dir, inductive=True):
    from ogb.nodeproppred import NodePropPredDataset

    with _ogb_torch_load_compat():
        ogb_dataset = NodePropPredDataset(name="ogbn-proteins", root=f"{data_dir}ogb")
    edge_index = torch.as_tensor(ogb_dataset.graph["edge_index"])
    edge_feat = torch.as_tensor(ogb_dataset.graph["edge_feat"])
    label = torch.as_tensor(ogb_dataset.labels)
    node_species = torch.as_tensor(ogb_dataset.graph["node_species"])

    from .data_utils import to_sparse_tensor
    edge_index_ = to_sparse_tensor(edge_index, edge_feat, ogb_dataset.graph["num_nodes"])
    node_feat = edge_index_.mean(dim=1)

    species = [0] + node_species.unique().tolist()
    ind_species_min, ind_species_max = species[0], species[3]
    ood_tr_species_min, ood_tr_species_max = species[3], species[5]
    ood_te_species = [species[i] for i in range(5, 8)]

    center_node_mask = (node_species <= ind_species_max).squeeze(1) * (node_species > ind_species_min).squeeze(1)
    if inductive:
        all_node_mask = (node_species <= ind_species_max).squeeze(1)
        ind_edge_index, _ = subgraph(all_node_mask, edge_index)
    else:
        ind_edge_index = edge_index

    dataset_ind = Data(x=node_feat, edge_index=ind_edge_index, y=label)
    idx = torch.arange(label.size(0))
    dataset_ind.node_idx = idx[center_node_mask]

    center_node_mask = (node_species <= ood_tr_species_max).squeeze(1) * (node_species > ood_tr_species_min).squeeze(1)
    if inductive:
        all_node_mask = (node_species <= ood_tr_species_max).squeeze(1)
        ood_tr_edge_index, _ = subgraph(all_node_mask, edge_index)
    else:
        ood_tr_edge_index = edge_index

    dataset_ood_tr = Data(x=node_feat, edge_index=ood_tr_edge_index, y=label)
    dataset_ood_tr.node_idx = idx[center_node_mask]

    dataset_ood_te = []
    for sp in ood_te_species:
        center_node_mask = (node_species == sp).squeeze(1)
        data = Data(x=node_feat, edge_index=edge_index, y=label)
        data.node_idx = idx[center_node_mask]
        dataset_ood_te.append(data)

    return dataset_ind, dataset_ood_tr, dataset_ood_te


def _create_sbm_dataset(data, p_ii=1.5, p_ij=0.5):
    n = data.num_nodes
    d = data.edge_index.size(1) / data.num_nodes / max(data.num_nodes - 1, 1)
    num_blocks = int(data.y.max()) + 1
    p_ii, p_ij = p_ii * d, p_ij * d
    block_size = n // num_blocks
    block_sizes = [block_size] * (num_blocks - 1) + [block_size + n % num_blocks]
    edge_probs = torch.ones((num_blocks, num_blocks)) * p_ij
    edge_probs[torch.arange(num_blocks), torch.arange(num_blocks)] = p_ii
    edge_index = stochastic_blockmodel_graph(block_sizes, edge_probs)
    dataset = Data(x=data.x, edge_index=edge_index, y=data.y)
    dataset.node_idx = torch.arange(dataset.num_nodes)
    return dataset


def _create_feat_noise_dataset(data):
    x, n = data.x, data.num_nodes
    idx = torch.randint(0, n, (n, 2))
    weight = torch.rand(n).unsqueeze(1)
    x_new = x[idx[:, 0]] * weight + x[idx[:, 1]] * (1 - weight)
    dataset = Data(x=x_new, edge_index=data.edge_index, y=data.y)
    dataset.node_idx = torch.arange(n)
    return dataset


def _load_graph_dataset(data_dir, dataname, ood_type):
    transform = T.NormalizeFeatures()
    if dataname in ("cora", "citeseer", "pubmed"):
        dset = Planetoid(root=f"{data_dir}Planetoid", split="public", name=dataname, transform=transform)
        data = dset[0]
        idx = torch.arange(data.num_nodes)
        data.splits = {
            "train": idx[data.train_mask],
            "valid": idx[data.val_mask],
            "test": idx[data.test_mask],
        }
    elif dataname == "amazon-photo":
        dset = Amazon(root=f"{data_dir}Amazon", name="Photo", transform=transform)
        data = dset[0]
    elif dataname == "amazon-computer":
        dset = Amazon(root=f"{data_dir}Amazon", name="Computers", transform=transform)
        data = dset[0]
    elif dataname == "coauthor-cs":
        dset = Coauthor(root=f"{data_dir}Coauthor", name="CS", transform=transform)
        data = dset[0]
    elif dataname == "coauthor-physics":
        dset = Coauthor(root=f"{data_dir}Coauthor", name="Physics", transform=transform)
        data = dset[0]
    else:
        raise NotImplementedError(dataname)

    data.node_idx = torch.arange(data.num_nodes)
    dataset_ind = data

    if ood_type == "structure":
        dataset_ood_tr = _create_sbm_dataset(data, p_ii=1.5, p_ij=0.5)
        dataset_ood_te = _create_sbm_dataset(data, p_ii=1.5, p_ij=0.5)
    elif ood_type == "feature":
        dataset_ood_tr = _create_feat_noise_dataset(data)
        dataset_ood_te = _create_feat_noise_dataset(data)
    elif ood_type == "label":
        class_t = {"cora": 3, "amazon-photo": 4, "coauthor-cs": 4}.get(dataname, 3)
        label = data.y
        center_node_mask_ind = (label > class_t).squeeze()
        idx = torch.arange(label.size(0))
        dataset_ind.node_idx = idx[center_node_mask_ind]
        if dataname in ("cora", "citeseer", "pubmed"):
            split_idx = data.splits
            tensor_split_idx = {}
            for key in split_idx:
                mask = torch.zeros(label.size(0), dtype=torch.bool)
                mask[torch.as_tensor(split_idx[key])] = True
                tensor_split_idx[key] = idx[mask & center_node_mask_ind]
            dataset_ind.splits = tensor_split_idx

        dataset_ood_tr = Data(x=data.x, edge_index=data.edge_index, y=data.y)
        dataset_ood_te = Data(x=data.x, edge_index=data.edge_index, y=data.y)
        center_node_mask_ood_tr = (label == class_t).squeeze()
        center_node_mask_ood_te = (label < class_t).squeeze()
        dataset_ood_tr.node_idx = idx[center_node_mask_ood_tr]
        dataset_ood_te.node_idx = idx[center_node_mask_ood_te]
    else:
        raise NotImplementedError(ood_type)

    return dataset_ind, dataset_ood_tr, dataset_ood_te
