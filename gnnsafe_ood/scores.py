"""Score utilities for OOD detection across backbones and methods."""

import torch


DEFAULT_SCORE_TYPE = {
    "msp": "msp",
    "gnnsafe": "energy",
    "gduq": "std",
    "graph_ebm": "graph_ebm",
}

CHAOS_SCORE_TYPES = {
    "chaos",
    "chaos_norm",
    "chaos_layernorm",
    "chaos_ratio",
    "chaos_layerratio",
}
HYBRID_SCORE_TYPES = {
    "energy_chaos_ratio": "chaos_ratio",
    "energy_chaos_layerratio": "chaos_layerratio",
}
PREDICTIVE_SCORE_TYPES = {
    "pred_entropy": "predictive_entropy",
    "mutual_info": "mutual_info",
}


def default_score_type(args):
    """Default detection score for the current training setup."""
    if (
        getattr(args, "method", None) == "gnnsafe"
        and getattr(args, "use_reg", False)
        and (
            getattr(args, "reg_score_type", "energy") in CHAOS_SCORE_TYPES
            or getattr(args, "reg_score_type", "energy") in PREDICTIVE_SCORE_TYPES
        )
    ):
        return getattr(args, "reg_score_type", "chaos")
    return DEFAULT_SCORE_TYPE.get(args.method, "msp")


def resolve_score_type(args):
    """Resolve the effective detection score from CLI args."""
    score_type = getattr(args, "score_type", "auto")
    if score_type == "auto":
        return default_score_type(args)
    return score_type


def score_variant_suffix(args):
    """
    Suffix for result naming when detection score diverges from the method default.

    Examples:
    - msp+dssgnn+chaos        -> chaos
    - msp+dssgnn+chaos+prop   -> chaos_prop
    - gnnsafe+dssgnn default  -> ""
    """
    score_type = resolve_score_type(args)
    default_type = default_score_type(args)
    parts = []
    if getattr(args, "score_type", "auto") != "auto" or score_type != default_type:
        parts.append(score_type)
    if getattr(args, "use_prop", False) and not (args.method == "gnnsafe" and score_type == default_type):
        parts.append("prop")
    return "_".join(parts)


def propagate_scalar_score(score, edge_index, prop_layers=1, alpha=0.5):
    """Propagate a scalar node score using the GNNSafe belief-propagation rule."""
    from torch_geometric.utils import degree

    score = score.unsqueeze(1)
    num_nodes = score.shape[0]
    row, col = edge_index
    deg = degree(col, num_nodes).float()
    norm = 1.0 / deg[col]
    norm = torch.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)
    value = torch.ones_like(row, dtype=torch.float, device=score.device) * norm
    try:
        from torch_sparse import SparseTensor, matmul

        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes))
        for _ in range(prop_layers):
            score = score * alpha + matmul(adj, score) * (1 - alpha)
    except ImportError:
        # Same D^-1 A aggregation without torch_sparse (environments where
        # the compiled extension is unavailable).
        adj = torch.sparse_coo_tensor(
            torch.stack([col, row]), value, (num_nodes, num_nodes)
        ).coalesce()
        for _ in range(prop_layers):
            score = score * alpha + torch.sparse.mm(adj, score) * (1 - alpha)
    return score.squeeze(1)


def _msp_score(logits, dataset_name):
    if dataset_name in ("proteins", "ppi"):
        pred = torch.sigmoid(logits).unsqueeze(-1)
        pred = torch.cat([pred, 1 - pred], dim=-1)
        return pred.max(dim=-1)[0].sum(dim=1)
    return torch.softmax(logits, dim=-1).max(dim=1)[0]


def _energy_score(logits, temperature, dataset_name):
    if dataset_name in ("proteins", "ppi"):
        logits = torch.stack([logits, torch.zeros_like(logits)], dim=2)
        return temperature * torch.logsumexp(logits / temperature, dim=-1).sum(dim=1)
    return temperature * torch.logsumexp(logits / temperature, dim=-1)


def _chaos_score(encoder, x, edge_index, uncertainty_type="chaos"):
    return -_chaos_uncertainty(encoder, x, edge_index, uncertainty_type=uncertainty_type)


def _chaos_uncertainty(encoder, x, edge_index, uncertainty_type="chaos"):
    if not hasattr(encoder, "forward_with_uncertainty"):
        raise ValueError("chaos score requires a backbone with forward_with_uncertainty().")
    _, uncertainty = encoder.forward_with_uncertainty(x, edge_index, uncertainty_type=uncertainty_type)
    return uncertainty


def _std_uncertainty(encoder, x, edge_index):
    try:
        _, std = encoder(x, edge_index, return_std=True)
    except TypeError as exc:
        raise ValueError("std score requires a backbone that supports return_std=True.") from exc
    return std.mean(dim=1)


def _std_score(encoder, x, edge_index):
    std = _std_uncertainty(encoder, x, edge_index)
    # Higher uncertainty should mean more OOD, so we negate it to match
    # the convention that larger scores indicate more in-distribution.
    return -std


def _predictive_stats(encoder, x, edge_index, eps=1e-12):
    if not hasattr(encoder, "forward_with_predictive_stats"):
        raise ValueError("predictive uncertainty scores require a backbone with forward_with_predictive_stats().")
    return encoder.forward_with_predictive_stats(x, edge_index, eps=eps)


def compute_backbone_uncertainty_penalty(encoder, dataset, node_idx, device, args):
    """
    Optional DSS-style IND regularizer used during supervised backbone training.

    This mirrors the standalone DSS-GNN training script, where IND training also
    penalizes higher-order uncertainty on labeled nodes.
    """
    penalty_weight = getattr(args, "dss_lambda_reg", 0.0)
    if penalty_weight <= 0 or not hasattr(encoder, "forward_with_uncertainty"):
        return torch.tensor(0.0, device=device)

    score_type = getattr(args, "dss_reg_score_type", "chaos")
    x = dataset.x.to(device)
    edge_index = dataset.edge_index.to(device)
    _, uncertainty = encoder.forward_with_uncertainty(x, edge_index, uncertainty_type=score_type)
    return penalty_weight * uncertainty[node_idx].mean()


def compute_regularizer_signal(encoder, dataset, device, args, score_type=None):
    """
    Compute an OOD-oriented regularizer signal where larger means more OOD-like.

    Supported score types:
    - energy: conventional energy, E(x) = -T logsumexp(logits / T)
    - chaos: DSS-GNN chaos uncertainty, sum_{n>0} ||Z_n||^2
    """
    x = dataset.x.to(device)
    edge_index = dataset.edge_index.to(device)
    score_type = score_type or getattr(args, "reg_score_type", "energy")

    if score_type == "energy":
        logits = encoder(x, edge_index)
        return -_energy_score(logits, args.T, args.dataset)
    if score_type in CHAOS_SCORE_TYPES:
        return _chaos_uncertainty(encoder, x, edge_index, uncertainty_type=score_type)
    if score_type in PREDICTIVE_SCORE_TYPES:
        stats = _predictive_stats(encoder, x, edge_index)
        return stats[PREDICTIVE_SCORE_TYPES[score_type]]
    if score_type == "std":
        return _std_uncertainty(encoder, x, edge_index)
    raise ValueError(f"Unknown regularizer score_type: {score_type}")


def compute_detection_score(encoder, dataset, device, args):
    """
    Compute the scalar OOD detection score for each node.

    Convention: larger score => more in-distribution.
    """
    x = dataset.x.to(device)
    edge_index = dataset.edge_index.to(device)
    score_type = resolve_score_type(args)
    predictive_stats = None

    if score_type == "msp":
        logits = encoder(x, edge_index)
        score = _msp_score(logits, args.dataset)
    elif score_type == "energy":
        logits = encoder(x, edge_index)
        score = _energy_score(logits, args.T, args.dataset)
    elif score_type in HYBRID_SCORE_TYPES:
        logits = encoder(x, edge_index)
        energy = _energy_score(logits, args.T, args.dataset)
        chaos = _chaos_uncertainty(encoder, x, edge_index, uncertainty_type=HYBRID_SCORE_TYPES[score_type])
        score = energy - getattr(args, "chaos_weight", 1.0) * chaos
    elif score_type in PREDICTIVE_SCORE_TYPES:
        predictive_stats = _predictive_stats(encoder, x, edge_index)
        score = -predictive_stats[PREDICTIVE_SCORE_TYPES[score_type]]
    elif score_type == "energy_mutual_info":
        predictive_stats = _predictive_stats(encoder, x, edge_index)
        energy = _energy_score(predictive_stats["logits"], args.T, args.dataset)
        score = energy - getattr(args, "mi_weight", 1.0) * predictive_stats["mutual_info"]
    elif score_type in CHAOS_SCORE_TYPES:
        score = _chaos_score(encoder, x, edge_index, uncertainty_type=score_type)
    elif score_type == "std":
        score = _std_score(encoder, x, edge_index)
    else:
        raise ValueError(f"Unknown score_type: {score_type}")

    if getattr(args, "use_prop", False):
        score = propagate_scalar_score(score, edge_index, args.K, args.alpha)

    return score
