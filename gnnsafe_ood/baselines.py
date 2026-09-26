"""Baseline OOD methods: MSP, Graph-EBM, GDUQ, MC-Dropout, and Deep Ensemble."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import GCN, DSSGNNEncoder, TFEEncoder, GDUQEncoder, GCNDSSResidualEncoder
from .graph_ebm import GraphEBMPosthocScorer
from .scores import (
    _energy_score,
    compute_backbone_uncertainty_penalty,
    compute_detection_score,
    propagate_scalar_score,
)


class MSP(nn.Module):
    """MSP baseline: use max softmax probability as OOD score."""

    def __init__(self, d, c, args):
        super().__init__()
        self.encoder = _make_encoder(d, c, args)

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

    def detect(self, dataset, node_idx, device, args):
        score = compute_detection_score(self.encoder, dataset, device, args)
        return score[node_idx]

    def loss_compute(self, dataset_ind, dataset_ood, criterion, device, args):
        train_idx = dataset_ind.splits["train"]
        x = dataset_ind.x.to(device)
        edge_index = dataset_ind.edge_index.to(device)
        logits_in = self.encoder(x, edge_index)[train_idx]
        if args.dataset in ("proteins", "ppi"):
            sup_loss = criterion(logits_in, dataset_ind.y[train_idx].to(device).to(torch.float))
        else:
            pred_in = F.log_softmax(logits_in, dim=1)
            sup_loss = criterion(pred_in, dataset_ind.y[train_idx].squeeze(1).to(device))
        dss_penalty = compute_backbone_uncertainty_penalty(self.encoder, dataset_ind, train_idx, device, args)
        return sup_loss + dss_penalty


class GDUQ(nn.Module):
    """GDUQ: use negative predictive std as OOD score (higher = more in-distribution)."""

    def __init__(self, d, c, args):
        super().__init__()
        self.encoder = _make_encoder(d, c, args)
        assert args.backbone == "gduq", "GDUQ method requires backbone=gduq"

    def set_anchor_dist(self, mean, std):
        self.encoder.set_anchor_dist(mean, std)

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def set_train_epoch(self, epoch):
        if hasattr(self.encoder, "set_train_epoch"):
            self.encoder.set_train_epoch(epoch)

    def forward(self, dataset, device):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        return self.encoder(x, edge_index, return_std=False)

    def detect(self, dataset, node_idx, device, args):
        score = compute_detection_score(self.encoder, dataset, device, args)
        return score[node_idx]

    def loss_compute(self, dataset_ind, dataset_ood, criterion, device, args):
        train_idx = dataset_ind.splits["train"]
        x, edge_index = dataset_ind.x.to(device), dataset_ind.edge_index.to(device)
        logits_in = self.encoder(x, edge_index, return_std=False)[train_idx]
        if args.dataset in ("proteins", "ppi"):
            sup_loss = criterion(logits_in, dataset_ind.y[train_idx].to(device).to(torch.float))
        else:
            pred_in = F.log_softmax(logits_in, dim=1)
            sup_loss = criterion(pred_in, dataset_ind.y[train_idx].squeeze(1).to(device))
        dss_penalty = compute_backbone_uncertainty_penalty(self.encoder, dataset_ind, train_idx, device, args)
        return sup_loss + dss_penalty


class GraphEBM(nn.Module):
    """Graph-EBM post-hoc detector on top of a supervised backbone."""

    def __init__(self, d, c, args):
        super().__init__()
        if args.backbone != "gcn":
            raise ValueError("graph_ebm currently supports backbone=gcn only.")
        self.encoder = _make_encoder(d, c, args)
        self.scorer = GraphEBMPosthocScorer(
            covariance_type=getattr(args, "graph_ebm_covariance_type", "diagonal"),
            tied_covariance=getattr(args, "graph_ebm_tied_covariance", False),
            gamma_correction=getattr(args, "graph_ebm_gamma_correction", 1.0),
            lambda_independent_energy=getattr(args, "graph_ebm_lambda_independent_energy", 1.0),
            lambda_local_energy=getattr(args, "graph_ebm_lambda_local_energy", 1.0),
            lambda_group_energy=getattr(args, "graph_ebm_lambda_group_energy", 1.0),
            alpha=getattr(args, "graph_ebm_alpha", 0.5),
            num_diffusion_steps=getattr(args, "graph_ebm_num_diffusion_steps", 10),
            aggregation=getattr(args, "graph_ebm_aggregation", "sum"),
        )

    def set_anchor_dist(self, mean, std):
        del mean, std

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.scorer._fitted = False

    def set_train_epoch(self, epoch):
        if hasattr(self.encoder, "set_train_epoch"):
            self.encoder.set_train_epoch(epoch)

    def forward(self, dataset, device):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        return self.encoder(x, edge_index)

    def _fit_posthoc(self, dataset_ind, device):
        if not hasattr(dataset_ind, "splits") or "train" not in dataset_ind.splits:
            raise ValueError("graph_ebm requires an IND dataset with a train split for fitting.")
        x = dataset_ind.x.to(device)
        edge_index = dataset_ind.edge_index.to(device)
        y = dataset_ind.y.to(device)
        if y.dim() > 1:
            y = y.squeeze(-1)
        train_mask = torch.zeros(dataset_ind.num_nodes, dtype=torch.bool, device=device)
        train_mask[dataset_ind.splits["train"].to(device)] = True
        views = self.encoder.forward_with_graph_ebm_views(x, edge_index)
        self.scorer.fit(
            logits=views["logits"],
            embeddings=views["embeddings"],
            edge_index=edge_index,
            y=y,
            mask=train_mask,
        )

    def detect(self, dataset, node_idx, device, args):
        del args
        if hasattr(dataset, "splits") and "train" in dataset.splits:
            self._fit_posthoc(dataset, device)
        if not self.scorer.fitted:
            raise ValueError("graph_ebm scorer has not been fitted yet. Detect on the IND dataset first.")
        x = dataset.x.to(device)
        edge_index = dataset.edge_index.to(device)
        views = self.encoder.forward_with_graph_ebm_views(x, edge_index)
        uncertainty = self.scorer.get_uncertainty(
            logits_unpropagated=views["logits_unpropagated"],
            embeddings_unpropagated=views["embeddings_unpropagated"],
            edge_index=edge_index,
        )
        score = -uncertainty
        return score[node_idx]

    def loss_compute(self, dataset_ind, dataset_ood, criterion, device, args):
        del dataset_ood
        train_idx = dataset_ind.splits["train"]
        x = dataset_ind.x.to(device)
        edge_index = dataset_ind.edge_index.to(device)
        logits_in = self.encoder(x, edge_index)[train_idx]
        if args.dataset in ("proteins", "ppi"):
            sup_loss = criterion(logits_in, dataset_ind.y[train_idx].to(device).to(torch.float))
        else:
            pred_in = F.log_softmax(logits_in, dim=1)
            sup_loss = criterion(pred_in, dataset_ind.y[train_idx].squeeze(1).to(device))
        dss_penalty = compute_backbone_uncertainty_penalty(self.encoder, dataset_ind, train_idx, device, args)
        return sup_loss + dss_penalty


def _log_mean_softmax(sample_logits):
    """
    Log of the mean softmax across sampled logits.

    Returned values behave like logits for downstream evaluation: argmax matches
    the mean-softmax prediction, softmax recovers the mean probabilities, and
    log_softmax gives the ensemble/MC predictive log-likelihood.
    """
    log_probs = F.log_softmax(sample_logits, dim=-1)
    return torch.logsumexp(log_probs, dim=0) - math.log(sample_logits.shape[0])


def _bayesian_detection_score(sample_logits, edge_index, args):
    """
    Detection score from sampled logits (larger = more in-distribution).

    Supported score types:
    - auto/energy: energy of the mean logits, T * logsumexp(mean_logits / T)
    - pred_entropy: negative predictive entropy of the mean softmax
    - msp: max of the mean softmax
    """
    score_type = getattr(args, "score_type", "auto")
    if score_type in ("auto", "energy"):
        mean_logits = sample_logits.mean(dim=0)
        score = _energy_score(mean_logits, args.T, args.dataset)
    elif score_type == "pred_entropy":
        probs = torch.softmax(sample_logits, dim=-1).mean(dim=0)
        score = (probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    elif score_type == "msp":
        probs = torch.softmax(sample_logits, dim=-1).mean(dim=0)
        score = probs.max(dim=-1)[0]
    else:
        raise ValueError(f"score_type {score_type} not supported for method {args.method}")
    if getattr(args, "use_prop", False):
        score = propagate_scalar_score(score, edge_index, args.K, args.alpha)
    return score


class MCDropout(nn.Module):
    """
    MC-dropout baseline: one backbone trained normally, detection via M
    stochastic forward passes with dropout kept active. Default OOD score is
    the energy of the mean logits across passes.
    """

    def __init__(self, d, c, args):
        super().__init__()
        self.encoder = _make_encoder(d, c, args)
        self.mc_samples = getattr(args, "mc_samples", 20)
        if getattr(args, "dropout", 0.0) <= 0.0:
            print("WARNING: mc_dropout with dropout=0 has no stochasticity; pass --dropout > 0.")

    def set_anchor_dist(self, mean, std):
        if hasattr(self.encoder, "set_anchor_dist"):
            self.encoder.set_anchor_dist(mean, std)

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def set_train_epoch(self, epoch):
        if hasattr(self.encoder, "set_train_epoch"):
            self.encoder.set_train_epoch(epoch)

    def _sample_logits(self, x, edge_index):
        """Run mc_samples stochastic passes with dropout active and BatchNorm frozen."""
        was_training = self.encoder.training
        self.encoder.train()
        for module in self.encoder.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eval()
        sample_logits = torch.stack([self.encoder(x, edge_index) for _ in range(self.mc_samples)], dim=0)
        self.encoder.train(was_training)
        return sample_logits

    def forward(self, dataset, device):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        return _log_mean_softmax(self._sample_logits(x, edge_index))

    def detect(self, dataset, node_idx, device, args):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        score = _bayesian_detection_score(self._sample_logits(x, edge_index), edge_index, args)
        return score[node_idx]

    def loss_compute(self, dataset_ind, dataset_ood, criterion, device, args):
        train_idx = dataset_ind.splits["train"]
        x = dataset_ind.x.to(device)
        edge_index = dataset_ind.edge_index.to(device)
        logits_in = self.encoder(x, edge_index)[train_idx]
        if args.dataset in ("proteins", "ppi"):
            sup_loss = criterion(logits_in, dataset_ind.y[train_idx].to(device).to(torch.float))
        else:
            pred_in = F.log_softmax(logits_in, dim=1)
            sup_loss = criterion(pred_in, dataset_ind.y[train_idx].squeeze(1).to(device))
        dss_penalty = compute_backbone_uncertainty_penalty(self.encoder, dataset_ind, train_idx, device, args)
        return sup_loss + dss_penalty


class Ensemble(nn.Module):
    """
    Deep-ensemble baseline: ensemble_size independently initialized backbones.

    Members are trained jointly by the runner's single optimizer; since members
    share no parameters, their gradients stay independent and this matches
    training each member separately from a different random init. Default OOD
    score is the energy of the member-averaged logits.
    """

    def __init__(self, d, c, args):
        super().__init__()
        self.encoders = nn.ModuleList(
            [_make_encoder(d, c, args) for _ in range(getattr(args, "ensemble_size", 5))]
        )

    def set_anchor_dist(self, mean, std):
        for encoder in self.encoders:
            if hasattr(encoder, "set_anchor_dist"):
                encoder.set_anchor_dist(mean, std)

    def reset_parameters(self):
        for encoder in self.encoders:
            encoder.reset_parameters()

    def set_train_epoch(self, epoch):
        for encoder in self.encoders:
            if hasattr(encoder, "set_train_epoch"):
                encoder.set_train_epoch(epoch)

    def _member_logits(self, x, edge_index):
        return torch.stack([encoder(x, edge_index) for encoder in self.encoders], dim=0)

    def forward(self, dataset, device):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        return _log_mean_softmax(self._member_logits(x, edge_index))

    def detect(self, dataset, node_idx, device, args):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        score = _bayesian_detection_score(self._member_logits(x, edge_index), edge_index, args)
        return score[node_idx]

    def loss_compute(self, dataset_ind, dataset_ood, criterion, device, args):
        train_idx = dataset_ind.splits["train"]
        x = dataset_ind.x.to(device)
        edge_index = dataset_ind.edge_index.to(device)
        losses = []
        for encoder in self.encoders:
            logits_in = encoder(x, edge_index)[train_idx]
            if args.dataset in ("proteins", "ppi"):
                sup_loss = criterion(logits_in, dataset_ind.y[train_idx].to(device).to(torch.float))
            else:
                pred_in = F.log_softmax(logits_in, dim=1)
                sup_loss = criterion(pred_in, dataset_ind.y[train_idx].squeeze(1).to(device))
            dss_penalty = compute_backbone_uncertainty_penalty(encoder, dataset_ind, train_idx, device, args)
            losses.append(sup_loss + dss_penalty)
        return torch.stack(losses).mean()


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
