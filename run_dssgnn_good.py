#!/usr/bin/env python3
"""
Run GCN / GraphSAGE / APPNP / GPR-GNN / DSS-GNN on GOOD (Graph OOD) benchmark datasets.

Loads GOOD datasets via GOOD's load_dataset, converts to our format (adj, features, masks),
and runs a deterministic GCN baseline or DSS-GNN training. Reports Acc, ECE, MCE, Brier on
id_val, id_test, val, test splits.

Usage:
  # Via GOOD config (recommended)
  python run_dssgnn_good.py --config_path GOOD_clean/configs/GOOD_configs/GOODCora/word/concept/ERM.yaml --model gcn
  python run_dssgnn_good.py --config_path GOOD_clean/configs/GOOD_configs/GOODCora/word/concept/ERM.yaml --model dssgnn
  python run_dssgnn_good.py --config_path GOOD_clean/configs/GOOD_configs/GOODArxiv/time/concept/ERM.yaml --model appnp_dssres
  python run_dssgnn_good.py --config_path GOOD_clean/configs/GOOD_configs/GOODArxiv/time/concept/ERM.yaml --model gpr_dssres

  # Via explicit args (when GOOD is installed)
  python run_dssgnn_good.py --dataset_name GOODCora --domain word --shift_type concept --model gcn

Requires: GOOD importable from either GOOD_clean/ or GOOD/, or installed in the environment.
"""

import argparse
import importlib
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim

# Add project root
PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ_ROOT)
for good_root in (
    os.path.join(PROJ_ROOT, "GOOD_clean"),
    os.path.join(PROJ_ROOT, "GOOD"),
):
    if os.path.isdir(good_root):
        sys.path.insert(0, good_root)

from tfe_utils import set_seed, accuracy, compute_ece, compute_mce, compute_brier
from dssgnn.chebyshev import build_rescaled_laplacian
from dssgnn.model import DSSGNN


class SimpleGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        from torch_geometric.nn import GCNConv

        if num_layers < 2:
            raise ValueError("GCN baseline requires at least 2 layers.")
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.convs = nn.ModuleList(
            GCNConv(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)

    @staticmethod
    def _linear_only(conv, x):
        x = conv.lin(x)
        if conv.bias is not None:
            x = x + conv.bias
        return x

    def forward_with_graph_ebm_views(self, x, edge_index):
        """
        Match the GNNSafe-side Graph-EBM interface for the plain GOOD GCN.

        We expose propagated and unpropagated penultimate activations/logits so
        the lightweight post-hoc Graph-EBM scorer can be reused unchanged.
        """
        h_prop = x
        h_unprop = x

        if len(self.convs) == 1:
            logits = self.convs[0](x, edge_index)
            logits_unpropagated = self._linear_only(self.convs[0], x)
            return {
                "embeddings": x,
                "embeddings_unpropagated": x,
                "logits": logits,
                "logits_unpropagated": logits_unpropagated,
            }

        for conv in self.convs[:-1]:
            h_prop = conv(h_prop, edge_index)
            h_unprop = self._linear_only(conv, h_unprop)
            h_prop = F.relu(h_prop)
            h_unprop = F.relu(h_unprop)
            h_prop = F.dropout(h_prop, p=self.dropout, training=self.training)
            h_unprop = F.dropout(h_unprop, p=self.dropout, training=self.training)

        logits = self.convs[-1](h_prop, edge_index)
        logits_unpropagated = self._linear_only(self.convs[-1], h_unprop)
        return {
            "embeddings": h_prop,
            "embeddings_unpropagated": h_unprop,
            "logits": logits,
            "logits_unpropagated": logits_unpropagated,
        }


def build_model(model_name, input_dim, hidden_dim, out_dim, args):
    if model_name == "dssgnn":
        return DSSGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=args.layers,
            K_lp=args.K_lp,
            K_hp=args.K_hp,
            P=args.P,
            use_random_gates=args.use_random_gates,
            P_gate=args.P_gate,
            S=args.quadrature_nodes,
            shared_input_lift=args.shared_input_lift,
            dropout=(args.pro_dropout, args.lin_dropout),
            activation=True,
            use_bn=getattr(args, "dss_bn", False),
        )
    if model_name == "gcn_dssres":
        from gnnsafe_ood.backbone import GCNDSSResidualEncoder

        return GCNDSSResidualEncoder(
            in_channels=input_dim,
            hidden_channels=hidden_dim,
            out_channels=out_dim,
            num_layers=args.layers,
            dropout=args.pro_dropout,
            use_bn=True,
            K_lp=args.K_lp,
            K_hp=args.K_hp,
            P=args.P,
            lambda_max=args.lambda_max,
            use_random_gates=args.use_random_gates,
            P_gate=args.P_gate,
            quadrature_nodes=args.quadrature_nodes,
            shared_input_lift=args.shared_input_lift,
            warmup_base_epochs=args.warmup_base_epochs,
            residual_scale_init=args.residual_scale_init,
        )
    if model_name == "sage_dssres":
        from gnnsafe_ood.backbone import SAGEDSSResidualEncoder

        return SAGEDSSResidualEncoder(
            in_channels=input_dim,
            hidden_channels=hidden_dim,
            out_channels=out_dim,
            num_layers=args.layers,
            dropout=args.pro_dropout,
            use_bn=True,
            K_lp=args.K_lp,
            K_hp=args.K_hp,
            P=args.P,
            lambda_max=args.lambda_max,
            use_random_gates=args.use_random_gates,
            P_gate=args.P_gate,
            quadrature_nodes=args.quadrature_nodes,
            shared_input_lift=args.shared_input_lift,
            warmup_base_epochs=args.warmup_base_epochs,
            residual_scale_init=args.residual_scale_init,
        )
    if model_name == "appnp_dssres":
        from gnnsafe_ood.backbone import APPNPDSSResidualEncoder

        return APPNPDSSResidualEncoder(
            in_channels=input_dim,
            hidden_channels=hidden_dim,
            out_channels=out_dim,
            num_layers=args.layers,
            dropout=args.pro_dropout,
            use_bn=True,
            K_lp=args.K_lp,
            K_hp=args.K_hp,
            P=args.P,
            lambda_max=args.lambda_max,
            use_random_gates=args.use_random_gates,
            P_gate=args.P_gate,
            quadrature_nodes=args.quadrature_nodes,
            shared_input_lift=args.shared_input_lift,
            warmup_base_epochs=args.warmup_base_epochs,
            residual_scale_init=args.residual_scale_init,
            appnp_K=args.appnp_K,
            appnp_alpha=args.appnp_alpha,
        )
    if model_name == "gpr_dssres":
        from gnnsafe_ood.backbone import GPRDSSResidualEncoder

        return GPRDSSResidualEncoder(
            in_channels=input_dim,
            hidden_channels=hidden_dim,
            out_channels=out_dim,
            num_layers=args.layers,
            dropout=args.pro_dropout,
            use_bn=True,
            K_lp=args.K_lp,
            K_hp=args.K_hp,
            P=args.P,
            lambda_max=args.lambda_max,
            use_random_gates=args.use_random_gates,
            P_gate=args.P_gate,
            quadrature_nodes=args.quadrature_nodes,
            shared_input_lift=args.shared_input_lift,
            warmup_base_epochs=args.warmup_base_epochs,
            residual_scale_init=args.residual_scale_init,
            gpr_K=args.gpr_K,
            gpr_alpha=args.gpr_alpha,
        )
    if model_name == "gcn":
        return SimpleGCN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=args.layers,
            dropout=args.pro_dropout,
        )
    if model_name == "sage":
        from gnnsafe_ood.backbone import GraphSAGE

        return GraphSAGE(
            in_channels=input_dim,
            hidden_channels=hidden_dim,
            out_channels=out_dim,
            num_layers=args.layers,
            dropout=args.pro_dropout,
            use_bn=True,
        )
    if model_name == "appnp":
        from gnnsafe_ood.backbone import APPNPBackbone

        return APPNPBackbone(
            in_channels=input_dim,
            hidden_channels=hidden_dim,
            out_channels=out_dim,
            num_layers=args.layers,
            dropout=args.pro_dropout,
            use_bn=True,
            appnp_K=args.appnp_K,
            appnp_alpha=args.appnp_alpha,
        )
    if model_name == "gpr":
        from gnnsafe_ood.backbone import GPRBackbone

        return GPRBackbone(
            in_channels=input_dim,
            hidden_channels=hidden_dim,
            out_channels=out_dim,
            num_layers=args.layers,
            dropout=args.pro_dropout,
            use_bn=True,
            gpr_K=args.gpr_K,
            gpr_alpha=args.gpr_alpha,
        )
    if model_name == "gduq":
        from gnnsafe_ood.backbone import GDUQEncoder

        return GDUQEncoder(
            in_channels=input_dim,
            hidden_channels=hidden_dim,
            out_channels=out_dim,
            num_layers=args.layers,
            dropout=args.pro_dropout,
            use_bn=True,
            K_lp=args.K_lp,
            K_hp=args.K_hp,
            P=args.P,
            lambda_max=args.lambda_max,
            n_anchors=getattr(args, "gduq_n_anchors", 5),
        )
    raise ValueError(f"Unsupported model: {model_name}")


def load_good_config(
    *,
    config_path=None,
    dataset_name="GOODCora",
    domain="word",
    shift_type="concept",
    dataset_root=None,
    hidden=64,
    seed=42,
    generate=False,
):
    if config_path:
        from GOOD.utils.config_reader import load_config, process_configs
        from munch import munchify

        config, _, _ = load_config(config_path)
        config = munchify(config)
        process_configs(config)
        config.model.model_level = "node"
        config.random_seed = seed
        if dataset_root:
            config.dataset.dataset_root = dataset_root
        if generate:
            config.dataset.generate = True
        return config

    from types import SimpleNamespace

    root = dataset_root or os.path.join(PROJ_ROOT, "GOOD_clean", "storage", "datasets")
    if not os.path.isdir(os.path.dirname(root)):
        root = dataset_root or os.path.join(PROJ_ROOT, "GOOD", "storage", "datasets")
    class _MetricStub:
        """Minimal stub so GOOD's load_dataset can call config.metric methods."""
        def set_score_func(self, metric):
            self.score_func = metric
        def set_loss_func(self, task):
            self.loss_func = task

    return SimpleNamespace(
        dataset=SimpleNamespace(
            dataset_name=dataset_name,
            domain=domain,
            shift_type=shift_type,
            dataset_root=root,
            generate=generate,
        ),
        model=SimpleNamespace(model_level="node", dim_hidden=hidden),
        metric=_MetricStub(),
        random_seed=seed,
    )


def ensure_good_dataset_registered(dataset_name: str) -> None:
    """Import the GOOD dataset module so it registers itself with GOOD.register."""
    dataset_modules = {
        "GOODArxiv": "GOOD.data.good_datasets.good_arxiv",
        "GOODCBAS": "GOOD.data.good_datasets.good_cbas",
        "GOODCora": "GOOD.data.good_datasets.good_cora",
        "GOODTwitch": "GOOD.data.good_datasets.good_twitch",
        "GOODWebKB": "GOOD.data.good_datasets.good_webkb",
        "GOODHIV": "GOOD.data.good_datasets.good_hiv",
        "GOODPCBA": "GOOD.data.good_datasets.good_pcba",
        "GOODZINC": "GOOD.data.good_datasets.good_zinc",
    }
    module_name = dataset_modules.get(dataset_name)
    if module_name is None:
        return
    importlib.import_module(module_name)


def _good_data_to_dssgnn_format(dataset, config, normalize_features=False):
    """
    Convert GOOD node-level dataset to (adj, features, labels, masks) for DSS-GNN.

    Args:
        normalize_features: if True, L1 row-normalize features, matching the
            preprocessing in tfe_utils.load_data used by the calibration
            pipeline. Raw GOOD features (e.g. CoraFull bag-of-words) have large
            row norms that saturate the DSS input lift and kill all ReLUs,
            collapsing standalone DSS-GNN logits to a constant.

    Returns:
        adj: scipy sparse (with self-loops)
        edge_index: torch LongTensor (2, E)
        features: torch Float (N, F)
        labels: torch Long (N,)
        masks: dict with train_mask, val_mask, test_mask, id_val_mask, id_test_mask
    """
    from torch_geometric.utils import to_scipy_sparse_matrix, add_self_loops

    graph = dataset[0]
    edge_index = graph.edge_index
    if edge_index.numel() == 0:
        n = graph.num_nodes
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=graph.x.device)
    edge_index, _ = add_self_loops(edge_index, num_nodes=graph.num_nodes)
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=graph.num_nodes).tocsr()

    features = graph.x.float()
    if normalize_features:
        rowsum = features.abs().sum(dim=1, keepdim=True)
        features = features / rowsum.clamp_min(1e-12)
    labels = graph.y.long().squeeze()
    if labels.dim() > 1:
        labels = labels.argmax(dim=1)

    masks = {
        "train_mask": graph.train_mask,
        "val_mask": graph.val_mask,
        "test_mask": graph.test_mask,
    }
    if hasattr(graph, "id_val_mask") and graph.id_val_mask is not None:
        masks["id_val_mask"] = graph.id_val_mask
    if hasattr(graph, "id_test_mask") and graph.id_test_mask is not None:
        masks["id_test_mask"] = graph.id_test_mask

    return adj, edge_index.cpu(), features, labels, masks


def forward_model(model, model_name, graph_input, edge_index, x):
    if model_name == "dssgnn":
        logits, uncertainty = model(graph_input, x, return_uncertainty=True)
        return logits, uncertainty
    if hasattr(model, "forward_with_uncertainty"):
        logits, uncertainty = model.forward_with_uncertainty(x, edge_index, uncertainty_type="chaos")
        return logits, uncertainty
    logits = model(x, edge_index)
    zero_uncertainty = logits.new_zeros(logits.shape[0])
    return logits, zero_uncertainty


def forward_predictive_stats(model, model_name, graph_input, edge_index, x):
    if model_name == "dssgnn":
        return model.forward_with_predictive_stats(graph_input, x)
    if hasattr(model, "forward_with_predictive_stats"):
        return model.forward_with_predictive_stats(x, edge_index)
    raise ValueError(f"Predictive stats are not available for {model_name}")


def restore_best_state(model, best_state, best_epoch):
    """Restore both parameters and the warmup phase of phased models."""
    if best_state is not None:
        model.load_state_dict(best_state)
    if hasattr(model, "set_train_epoch"):
        model.set_train_epoch(best_epoch)
    return model


def build_transport_graph(edge_index, train_mask, num_nodes):
    """Build transport graph: edges between training nodes with unit weights."""
    train_idx = train_mask.nonzero(as_tuple=True)[0]
    # Map global node ID -> local training index (-1 if not training)
    global_to_local = torch.full((num_nodes,), -1, dtype=torch.long)
    global_to_local[train_idx] = torch.arange(len(train_idx))

    src, dst = edge_index[0], edge_index[1]
    local_src = global_to_local[src]
    local_dst = global_to_local[dst]
    # Keep edges where both endpoints are training nodes, and src < dst (undirected)
    valid = (local_src >= 0) & (local_dst >= 0) & (src < dst)
    pairs_t = torch.stack([local_src[valid], local_dst[valid]], dim=1)
    weights_t = torch.ones(pairs_t.shape[0])
    return {"pairs": pairs_t, "weights": weights_t, "num_train": len(train_idx)}


def tar_transport_step(node_risk, transport_graph, beta=0.1, step_size=0.5, inner_steps=3):
    """Run graph-local transport on node risk to produce reweighting coefficients."""
    num_train = transport_graph["num_train"]
    if num_train <= 1:
        return node_risk.new_ones((num_train,))
    pairs = transport_graph["pairs"].to(node_risk.device)
    weights = transport_graph["weights"].to(node_risk.device, dtype=node_risk.dtype)
    if pairs.numel() == 0:
        return node_risk.new_ones((num_train,))

    q = node_risk.new_full((num_train,), 1.0 / num_train)
    eps = 1e-8
    for _ in range(inner_steps):
        qi = q[pairs[:, 0]].clamp_min(eps)
        qj = q[pairs[:, 1]].clamp_min(eps)
        ri = node_risk[pairs[:, 0]]
        rj = node_risk[pairs[:, 1]]
        velocity = ri - rj + beta * (torch.log(qj) - torch.log(qi))
        xi = torch.where(velocity > 0, qj, qi)
        raw_flux = step_size * weights * velocity * xi
        capacity = 0.5 * torch.minimum(qi, qj)
        flux = torch.clamp(raw_flux, min=-capacity, max=capacity)
        delta = q.new_zeros(q.shape)
        delta.index_add_(0, pairs[:, 0], flux)
        delta.index_add_(0, pairs[:, 1], -flux)
        q = (q + delta).clamp_min(eps)
        q = q / q.sum()
    return (num_train * q).detach()


def spectral_node_risk(per_node_loss, uncertainty_full, train_mask, graph_input, eta0=0.2, eta_lp=0.2, eta_hp=0.1):
    """Augment per-node loss with spectrally filtered DSS uncertainty.

    Args:
        per_node_loss: [num_train] loss per training node
        uncertainty_full: [N] uncertainty for ALL nodes (needed for Chebyshev filter on full graph)
        train_mask: [N] boolean mask for training nodes
        graph_input: rescaled Laplacian [N, N] (sparse)
    """
    from dssgnn.chebyshev import chebyshev_filter
    node_risk = per_node_loss.detach()
    if uncertainty_full is None or (eta0 == 0 and eta_lp == 0 and eta_hp == 0):
        return node_risk
    unc_train = uncertainty_full[train_mask].detach()
    unc_normed = unc_train / unc_train.mean().clamp_min(1e-8)
    if eta0 > 0:
        node_risk = node_risk + eta0 * unc_normed
    if eta_lp > 0 or eta_hp > 0:
        device, dtype = uncertainty_full.device, uncertainty_full.dtype
        K_lp, K_hp = 3, 2
        coeffs_lp = torch.zeros(K_lp + 1, device=device, dtype=dtype)
        for k in range(K_lp + 1):
            coeffs_lp[k] = 0.5 ** k
        coeffs_hp = -coeffs_lp.clone()
        coeffs_hp[0] = coeffs_hp[0] + 1.0
        # Filter on FULL graph, then extract training nodes
        unc_full_col = uncertainty_full.detach().unsqueeze(-1)
        lp_full = chebyshev_filter(graph_input, unc_full_col, coeffs_lp).squeeze(-1)
        hp_full = chebyshev_filter(graph_input, unc_full_col, coeffs_hp).squeeze(-1).abs()
        lp = lp_full[train_mask]
        hp = hp_full[train_mask]
        if eta_lp > 0:
            lp_normed = lp / lp.abs().mean().clamp_min(1e-8)
            node_risk = node_risk + eta_lp * lp_normed
        if eta_hp > 0:
            hp_normed = hp / hp.mean().clamp_min(1e-8)
            node_risk = node_risk + eta_hp * hp_normed
    return node_risk


def train_epoch(model, model_name, optimizer, graph_input, edge_index, x, y, train_mask,
                lambda_reg=0.01, train_mode="erm", transport_graph=None, tar_args=None,
                loss_type="mean"):
    model.train()
    optimizer.zero_grad()

    if loss_type == "quad" and model_name in ("dssgnn",) and hasattr(model, "forward_with_sample_logits"):
        logits, sample_logits, weights, uncertainty = model.forward_with_sample_logits(graph_input, x)
        S = sample_logits.shape[0]
        per_sample_loss = torch.stack([
            F.cross_entropy(F.log_softmax(sample_logits[s][train_mask], dim=1), y[train_mask])
            for s in range(S)
        ])
        w = weights.to(per_sample_loss.device, dtype=per_sample_loss.dtype)
        sup_loss = (w * per_sample_loss).sum()
    elif loss_type == "quad" and hasattr(model, "forward_with_predictive_stats"):
        stats = model.forward_with_predictive_stats(x, edge_index)
        logits = stats["logits"]
        sample_logits = stats["sample_logits"]
        weights = stats["weights"]
        uncertainty = stats.get("chaos_energy", logits.new_zeros(logits.shape[0]))
        S = sample_logits.shape[0]
        per_sample_loss = torch.stack([
            F.cross_entropy(F.log_softmax(sample_logits[s][train_mask], dim=1), y[train_mask])
            for s in range(S)
        ])
        w = weights.to(per_sample_loss.device, dtype=per_sample_loss.dtype)
        sup_loss = (w * per_sample_loss).sum()
    else:
        logits, uncertainty = forward_model(model, model_name, graph_input, edge_index, x)
        sup_loss = F.cross_entropy(logits[train_mask], y[train_mask])

    if train_mode == "erm":
        loss = sup_loss
        loss = loss + lambda_reg * uncertainty.mean()
    elif train_mode == "tar_adaptive":
        per_node_loss = F.cross_entropy(logits[train_mask], y[train_mask], reduction="none")
        # Base TAR: loss-only reweighting
        rho_base = tar_transport_step(per_node_loss.detach(), transport_graph)
        # Spectral TAR: uncertainty-augmented node risk (needs full-graph uncertainty for Chebyshev)
        unc_full = uncertainty if uncertainty.sum() > 0 else None
        node_risk_spectral = spectral_node_risk(
            per_node_loss.detach(), unc_full, train_mask, graph_input,
            eta0=tar_args.get("eta0", 0.2),
            eta_lp=tar_args.get("eta_lp", 0.2),
            eta_hp=tar_args.get("eta_hp", 0.1),
        )
        rho_spectral = tar_transport_step(node_risk_spectral, transport_graph)
        # Adaptive interpolation
        lam = tar_args.get("adaptive_lambda", 0.5)
        rho = (1.0 - lam) * rho_base + lam * rho_spectral
        loss = torch.mean(rho * per_node_loss)
        loss = loss + lambda_reg * uncertainty.mean()
    else:
        raise ValueError(f"Unknown train_mode: {train_mode}")

    loss.backward()
    optimizer.step()


def evaluate(model, model_name, graph_input, edge_index, x, y, mask):
    model.eval()
    with torch.no_grad():
        logits, _ = forward_model(model, model_name, graph_input, edge_index, x)
    log_probs = F.log_softmax(logits, dim=1)
    acc = accuracy(log_probs[mask], y[mask])
    ece = compute_ece(log_probs[mask], y[mask]).item()
    mce = compute_mce(log_probs[mask], y[mask]).item()
    brier = compute_brier(log_probs[mask], y[mask]).item()
    return acc.item(), ece, mce, brier, logits


def run(config, args):
    set_seed(config.random_seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load GOOD dataset
    from GOOD.data import load_dataset

    ensure_good_dataset_registered(config.dataset.dataset_name)
    dataset = load_dataset(config.dataset.dataset_name, config)
    if config.model.model_level != "node":
        raise ValueError("DSS-GNN GOOD benchmark supports node-level datasets only.")

    if args.normalize_features == "on":
        normalize_features = True
    elif args.normalize_features == "off":
        normalize_features = False
    else:  # auto: standalone DSS-GNN needs it; leave published hybrid/baseline behavior untouched
        normalize_features = args.model == "dssgnn"
    adj, edge_index, features, labels, masks = _good_data_to_dssgnn_format(
        dataset, config, normalize_features=normalize_features
    )
    train_mask = masks["train_mask"]
    val_mask = masks["val_mask"]
    test_mask = masks["test_mask"]
    id_val_mask = masks.get("id_val_mask", val_mask)
    id_test_mask = masks.get("id_test_mask", test_mask)

    num_classes = int(labels.max()) + 1
    input_dim = features.shape[1]

    # Build rescaled Laplacian
    graph_input = build_rescaled_laplacian(adj, lambda_max=args.lambda_max).to(device)
    edge_index = edge_index.to(device)

    model = build_model(args.model, input_dim, args.hidden, num_classes, args)
    # Initialize anchor distribution for GDUQ
    if hasattr(model, "set_anchor_dist"):
        train_feats = features[train_mask.cpu()]
        model.set_anchor_dist(train_feats.mean(dim=0), train_feats.std(dim=0).clamp_min(1e-6))
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    id_val_mask = id_val_mask.to(device)
    id_test_mask = id_test_mask.to(device)

    # Build transport graph for TAR modes
    transport_graph = None
    tar_args = None
    if args.train_mode.startswith("tar"):
        transport_graph = build_transport_graph(edge_index.cpu(), train_mask.cpu(), features.shape[0])
        tar_args = {
            "adaptive_lambda": args.tar_adaptive_lambda,
            "eta0": args.tar_eta0,
            "eta_lp": args.tar_eta_lp,
            "eta_hp": args.tar_eta_hp,
        }

    # Training loop
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(args.epochs):
        if hasattr(model, "set_train_epoch"):
            model.set_train_epoch(epoch)
        train_epoch(
            model, args.model, optimizer, graph_input, edge_index, features, labels, train_mask,
            lambda_reg=args.lambda_reg,
            train_mode=args.train_mode,
            transport_graph=transport_graph,
            tar_args=tar_args,
            loss_type=args.loss_type,
        )
        _, _, _, _, logits = evaluate(model, args.model, graph_input, edge_index, features, labels, val_mask)
        val_loss = F.cross_entropy(logits[val_mask], labels[val_mask]).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    # Restore best and evaluate
    restore_best_state(model, best_state, best_epoch)
    model = model.to(device)

    splits = [
        ("train", train_mask),
        ("id_val", id_val_mask),
        ("id_test", id_test_mask),
        ("val", val_mask),
        ("test", test_mask),
    ]
    results = {}
    for name, mask in splits:
        acc, ece, mce, brier, _ = evaluate(model, args.model, graph_input, edge_index, features, labels, mask)
        results[name] = {"acc": acc, "ece": ece, "mce": mce, "brier": brier}
        print(f"  {name}: Acc={acc:.4f} ECE={ece:.4f} MCE={mce:.4f} Brier={brier:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="GCN / GraphSAGE / APPNP / GPR-GNN / DSS-GNN on GOOD benchmark")
    parser.add_argument("--config_path", type=str, default=None, help="Path to GOOD YAML config")
    parser.add_argument("--dataset_name", type=str, default="GOODCora", help="GOOD dataset name")
    parser.add_argument("--domain", type=str, default="word", help="Domain (word, degree, etc.)")
    parser.add_argument("--shift_type", type=str, default="concept", help="Shift (no_shift, covariate, concept)")
    parser.add_argument("--dataset_root", type=str, default=None, help="GOOD dataset root")
    parser.add_argument(
        "--model",
        type=str,
        default="dssgnn",
        choices=["gcn", "sage", "appnp", "gpr", "dssgnn", "gcn_dssres", "sage_dssres", "appnp_dssres", "gpr_dssres", "gduq"],
        help="Model to train",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--K_lp", type=int, default=3)
    parser.add_argument("--K_hp", type=int, default=2)
    parser.add_argument("--P", type=int, default=1)
    parser.add_argument("--pro_dropout", type=float, default=0.5)
    parser.add_argument("--lin_dropout", type=float, default=0.0)
    parser.add_argument("--lambda_reg", type=float, default=0.01)
    parser.add_argument("--lambda_max", type=float, default=2.0)
    parser.add_argument("--use_random_gates", action="store_true")
    parser.add_argument("--P_gate", type=int, default=1)
    parser.add_argument("--quadrature_nodes", type=int, default=4)
    parser.add_argument("--shared_input_lift", action="store_true")
    parser.add_argument("--appnp_K", type=int, default=10)
    parser.add_argument("--appnp_alpha", type=float, default=0.1)
    parser.add_argument("--gpr_K", type=int, default=10)
    parser.add_argument("--gpr_alpha", type=float, default=0.1)
    parser.add_argument("--gduq_n_anchors", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generate", action="store_true", help="Regenerate GOOD dataset locally when supported")
    parser.add_argument("--warmup_base_epochs", type=int, default=25)
    parser.add_argument("--residual_scale_init", type=float, default=0.1)
    parser.add_argument("--train_mode", type=str, default="erm", choices=["erm", "tar_adaptive"],
                        help="Training mode: erm (default) or tar_adaptive (hybrid loss+spectral reweighting)")
    parser.add_argument("--tar_adaptive_lambda", type=float, default=0.5,
                        help="Interpolation: 0=pure-loss TAR, 1=fully spectral TAR")
    parser.add_argument("--tar_eta0", type=float, default=0.2, help="Raw uncertainty weight in spectral node risk")
    parser.add_argument("--tar_eta_lp", type=float, default=0.2, help="Low-pass uncertainty weight")
    parser.add_argument("--tar_eta_hp", type=float, default=0.1, help="High-pass uncertainty weight")
    parser.add_argument("--loss_type", type=str, default="mean", choices=["mean", "quad"],
                        help="Training loss: mean (loss on Z_0) or quad (quadrature-averaged loss)")
    parser.add_argument("--dss_bn", action="store_true",
                        help="per-chaos-channel BatchNorm between hidden DSS layers (opt-in)")
    parser.add_argument("--normalize_features", type=str, default="auto", choices=["auto", "on", "off"],
                        help="L1 row-normalize input features (auto: on for model=dssgnn only, "
                             "matching the calibration pipeline; published hybrid runs used raw features)")
    args = parser.parse_args()

    # Load config
    config = load_good_config(
        config_path=args.config_path,
        dataset_name=args.dataset_name,
        domain=args.domain,
        shift_type=args.shift_type,
        dataset_root=args.dataset_root,
        hidden=args.hidden,
        seed=args.seed,
        generate=args.generate,
    )

    print(
        f"# {args.model.upper()} on GOOD: "
        f"{config.dataset.dataset_name} / {config.dataset.domain} / {config.dataset.shift_type}"
    )
    t0 = time.time()
    _ = run(config, args)
    print(f"# Time: {time.time() - t0:.1f}s")
    print("# Done.")


if __name__ == "__main__":
    main()
