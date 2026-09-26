"""GNNSafe: Energy-based OOD detection with optional energy propagation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import GCN, DSSGNNEncoder, TFEEncoder, GDUQEncoder, GCNDSSResidualEncoder
from .scores import (
    CHAOS_SCORE_TYPES,
    PREDICTIVE_SCORE_TYPES,
    compute_backbone_uncertainty_penalty,
    compute_detection_score,
    compute_regularizer_signal,
    propagate_scalar_score,
)


def _make_encoder(d, c, args):
    if args.backbone == "gcn":
        return GCN(
            in_channels=d,
            hidden_channels=args.hidden_channels,
            out_channels=c,
            num_layers=args.num_layers,
            dropout=args.dropout,
            use_bn=args.use_bn,
        )
    if args.backbone == "dssgnn":
        return DSSGNNEncoder(
            in_channels=d,
            hidden_channels=args.hidden_channels,
            out_channels=c,
            num_layers=args.num_layers,
            dropout=args.dropout,
            use_bn=args.use_bn,
            K_lp=getattr(args, "K_lp", 3),
            K_hp=getattr(args, "K_hp", 2),
            P=getattr(args, "P", 1),
            lambda_max=getattr(args, "lambda_max", 2.0),
            use_random_gates=getattr(args, "use_random_gates", False),
            P_gate=getattr(args, "P_gate", 1),
            quadrature_nodes=getattr(args, "S", 4),
            shared_input_lift=getattr(args, "shared_input_lift", False),
            dss_use_bn=getattr(args, "dss_bn", False),
        )
    if args.backbone == "gcn_dssres":
        return GCNDSSResidualEncoder(
            in_channels=d,
            hidden_channels=args.hidden_channels,
            out_channels=c,
            num_layers=args.num_layers,
            dropout=args.dropout,
            use_bn=args.use_bn,
            K_lp=getattr(args, "K_lp", 3),
            K_hp=getattr(args, "K_hp", 2),
            P=getattr(args, "P", 1),
            lambda_max=getattr(args, "lambda_max", 2.0),
            use_random_gates=getattr(args, "use_random_gates", False),
            P_gate=getattr(args, "P_gate", 1),
            quadrature_nodes=getattr(args, "S", 4),
            shared_input_lift=getattr(args, "shared_input_lift", False),
            warmup_base_epochs=getattr(args, "warmup_base_epochs", 50),
            residual_scale_init=getattr(args, "residual_scale_init", 0.1),
        )
    if args.backbone == "tfe":
        return TFEEncoder(
            in_channels=d,
            hidden_channels=args.hidden_channels,
            out_channels=c,
            num_layers=args.num_layers,
            dropout=args.dropout,
            use_bn=args.use_bn,
            hop_lp=getattr(args, "hop_lp", 2),
            hop_hp=getattr(args, "hop_hp", 2),
            combine=getattr(args, "tfe_combine", "sum"),
            eta=getattr(args, "eta", 0.5),
        )
    if args.backbone == "gduq":
        return GDUQEncoder(
            in_channels=d,
            hidden_channels=args.hidden_channels,
            out_channels=c,
            num_layers=args.num_layers,
            dropout=args.dropout,
            use_bn=args.use_bn,
            K_lp=getattr(args, "K_lp", 3),
            K_hp=getattr(args, "K_hp", 2),
            P=getattr(args, "P", 1),
            lambda_max=getattr(args, "lambda_max", 2.0),
            n_anchors=getattr(args, "gduq_n_anchors", 5),
            use_random_gates=getattr(args, "use_random_gates", False),
            P_gate=getattr(args, "P_gate", 1),
            quadrature_nodes=getattr(args, "S", 4),
            shared_input_lift=getattr(args, "shared_input_lift", False),
        )
    raise NotImplementedError(f"backbone {args.backbone}")


class GNNSafe(nn.Module):
    """
    Energy-based OOD detection.
    use_reg + use_prop: GNNSafe++
    use_prop only: GNNSafe
    use_reg only: Energy FT
    neither: Energy
    """

    def __init__(self, d, c, args):
        super().__init__()
        self.encoder = _make_encoder(d, c, args)
        self.use_reg = getattr(args, "use_reg", False)
        self.use_prop = getattr(args, "use_prop", False)
        self.T = getattr(args, "T", 1.0)
        self.K = getattr(args, "K", 2)
        self.alpha = getattr(args, "alpha", 0.5)

    def set_anchor_dist(self, mean, std):
        if hasattr(self.encoder, "set_anchor_dist"):
            self.encoder.set_anchor_dist(mean, std)

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def set_train_epoch(self, epoch):
        if hasattr(self.encoder, "set_train_epoch"):
            self.encoder.set_train_epoch(epoch)

    def forward(self, dataset, device):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        return self.encoder(x, edge_index)

    def propagation(self, e, edge_index, prop_layers=1, alpha=0.5):
        """Score propagation shared with other detectors."""
        return propagate_scalar_score(e, edge_index, prop_layers, alpha)

    def detect(self, dataset, node_idx, device, args):
        score = compute_detection_score(self.encoder, dataset, device, args)
        return score[node_idx]

    def _regularizer_loss(self, signal_in, signal_out, args):
        reg_score_type = getattr(args, "reg_score_type", "energy")
        min_n = min(signal_in.shape[0], signal_out.shape[0])
        signal_in = signal_in[:min_n]
        signal_out = signal_out[:min_n]

        if reg_score_type == "energy":
            return torch.mean(F.relu(signal_in - args.m_in) ** 2 + F.relu(args.m_out - signal_out) ** 2)
        if reg_score_type in CHAOS_SCORE_TYPES or reg_score_type in PREDICTIVE_SCORE_TYPES:
            margin = getattr(args, "chaos_margin", 0.01)
            return torch.mean(F.relu(margin + signal_in - signal_out) ** 2)
        raise ValueError(f"Unsupported reg_score_type: {reg_score_type}")

    def loss_compute(self, dataset_ind, dataset_ood, criterion, device, args):
        x_in, edge_index_in = dataset_ind.x.to(device), dataset_ind.edge_index.to(device)
        logits_in = self.encoder(x_in, edge_index_in)

        train_in_idx = dataset_ind.splits["train"]

        if args.dataset in ("proteins", "ppi"):
            sup_loss = criterion(logits_in[train_in_idx], dataset_ind.y[train_in_idx].to(device).to(torch.float))
        else:
            pred_in = F.log_softmax(logits_in[train_in_idx], dim=1)
            sup_loss = criterion(pred_in, dataset_ind.y[train_in_idx].squeeze(1).to(device))

        dss_penalty = compute_backbone_uncertainty_penalty(self.encoder, dataset_ind, train_in_idx, device, args)

        if args.use_reg:
            train_ood_idx = dataset_ood.node_idx
            signal_in = compute_regularizer_signal(self.encoder, dataset_ind, device, args)
            signal_out = compute_regularizer_signal(self.encoder, dataset_ood, device, args)
            if args.use_prop:
                signal_in = self.propagation(signal_in, edge_index_in, args.K, args.alpha)[train_in_idx]
                signal_out = self.propagation(signal_out, dataset_ood.edge_index.to(device), args.K, args.alpha)[train_ood_idx]
            else:
                signal_in = signal_in[train_in_idx]
                signal_out = signal_out[train_ood_idx]

            reg_loss = self._regularizer_loss(signal_in, signal_out, args)
            return sup_loss + dss_penalty + args.lamda * reg_loss
        return sup_loss + dss_penalty
