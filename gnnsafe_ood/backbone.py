"""Backbone encoders for GNNSafe: GCN, GraphSAGE, APPNP, GPR-GNN, DSS-GNN, hybrid residual, and GDUQ."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import APPNP as PyGAPPNP, GCNConv, SAGEConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import to_scipy_sparse_matrix, add_self_loops


def _edge_index_to_adj(edge_index, num_nodes):
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()
    return adj


class GCN(nn.Module):
    """GCN backbone matching GNNSafe interface: forward(x, edge_index) -> logits."""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, use_bn=True):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_channels)])
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    @staticmethod
    def _linear_only(conv, x):
        x = conv.lin(x)
        if conv.bias is not None:
            x = x + conv.bias
        return x

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

    def forward_with_graph_ebm_views(self, x, edge_index):
        """
        Return propagated and unpropagated penultimate embeddings/logits.

        The "unpropagated" path matches Graph-EBM's convention: use the same layer
        weights but skip the message-passing aggregation itself.
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

        for i, conv in enumerate(self.convs[:-1]):
            h_prop = conv(h_prop, edge_index)
            h_unprop = self._linear_only(conv, h_unprop)
            if self.use_bn:
                h_prop = self.bns[i](h_prop)
                h_unprop = self.bns[i](h_unprop)
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


class GraphSAGE(nn.Module):
    """GraphSAGE backbone matching GNNSafe interface: forward(x, edge_index) -> logits."""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, use_bn=True):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_channels)])
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    @staticmethod
    def _linear_only(conv, x):
        out = conv.lin_l(x)
        if getattr(conv, "root_weight", False) and getattr(conv, "lin_r", None) is not None:
            out = out + conv.lin_r(x)
        return out

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

    def forward_with_graph_ebm_views(self, x, edge_index):
        """
        Return propagated and unpropagated penultimate embeddings/logits.

        For GraphSAGE, the unpropagated path applies the layer-local linear maps
        without neighbor aggregation, analogous to the GCN helper above.
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

        for i, conv in enumerate(self.convs[:-1]):
            h_prop = conv(h_prop, edge_index)
            h_unprop = self._linear_only(conv, h_unprop)
            if self.use_bn:
                h_prop = self.bns[i](h_prop)
                h_unprop = self.bns[i](h_unprop)
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


class APPNPBackbone(nn.Module):
    """APPNP backbone matching GNNSafe interface: forward(x, edge_index) -> logits."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        dropout=0.5,
        use_bn=True,
        appnp_K=10,
        appnp_alpha=0.1,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("APPNP backbone requires at least 2 layers.")
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_channels)])
        for _ in range(num_layers - 2):
            self.linears.append(nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.out_lin = nn.Linear(hidden_channels, out_channels)
        self.propagation = PyGAPPNP(K=appnp_K, alpha=appnp_alpha)
        self.dropout = dropout
        self.use_bn = use_bn

    def reset_parameters(self):
        for lin in self.linears:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.out_lin.reset_parameters()
        self.propagation.reset_parameters()

    def _encode_hidden(self, x):
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x, edge_index):
        x = self._encode_hidden(x)
        logits_unpropagated = self.out_lin(x)
        return self.propagation(logits_unpropagated, edge_index)

    def forward_with_graph_ebm_views(self, x, edge_index):
        """
        Return propagated and unpropagated penultimate embeddings/logits.

        For APPNP, the MLP produces unpropagated logits and APPNP performs the
        final propagation step, so the propagated/unpropagated penultimate
        embeddings are identical while the logits differ after diffusion.
        """
        embeddings = self._encode_hidden(x)
        logits_unpropagated = self.out_lin(embeddings)
        logits = self.propagation(logits_unpropagated, edge_index)
        return {
            "embeddings": embeddings,
            "embeddings_unpropagated": embeddings,
            "logits": logits,
            "logits_unpropagated": logits_unpropagated,
        }


class GPRPropagation(MessagePassing):
    """Learned generalized PageRank propagation over a normalized graph."""

    def __init__(self, K=10, alpha=0.1):
        super().__init__(aggr="add")
        self.K = int(K)
        self.alpha = float(alpha)
        self.gamma = nn.Parameter(torch.empty(self.K + 1))
        self.reset_parameters()

    @staticmethod
    def _ppr_coeffs(K, alpha, device=None, dtype=None):
        coeffs = [alpha * (1.0 - alpha) ** k for k in range(K)]
        coeffs.append((1.0 - alpha) ** K)
        return torch.tensor(coeffs, device=device, dtype=dtype)

    def reset_parameters(self):
        with torch.no_grad():
            self.gamma.copy_(self._ppr_coeffs(self.K, self.alpha, device=self.gamma.device, dtype=self.gamma.dtype))

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, edge_weight = gcn_norm(
            edge_index,
            edge_weight,
            num_nodes=x.size(0),
            add_self_loops=True,
            dtype=x.dtype,
        )
        hidden = self.gamma[0] * x
        x_k = x
        for k in range(self.K):
            x_k = self.propagate(edge_index, x=x_k, edge_weight=edge_weight, size=None)
            hidden = hidden + self.gamma[k + 1] * x_k
        return hidden

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j


class GPRBackbone(nn.Module):
    """GPR-GNN backbone matching GNNSafe interface: forward(x, edge_index) -> logits."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        dropout=0.5,
        use_bn=True,
        gpr_K=10,
        gpr_alpha=0.1,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("GPR backbone requires at least 2 layers.")
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_channels)])
        for _ in range(num_layers - 2):
            self.linears.append(nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.out_lin = nn.Linear(hidden_channels, out_channels)
        self.propagation = GPRPropagation(K=gpr_K, alpha=gpr_alpha)
        self.dropout = dropout
        self.use_bn = use_bn

    def reset_parameters(self):
        for lin in self.linears:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.out_lin.reset_parameters()
        self.propagation.reset_parameters()

    def _encode_hidden(self, x):
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x, edge_index):
        x = self._encode_hidden(x)
        logits_unpropagated = self.out_lin(x)
        return self.propagation(logits_unpropagated, edge_index)

    def forward_with_graph_ebm_views(self, x, edge_index):
        """
        Return propagated and unpropagated penultimate embeddings/logits.

        As in APPNP, the MLP produces the unpropagated logits and the learned
        propagation step produces the final propagated logits.
        """
        embeddings = self._encode_hidden(x)
        logits_unpropagated = self.out_lin(embeddings)
        logits = self.propagation(logits_unpropagated, edge_index)
        return {
            "embeddings": embeddings,
            "embeddings_unpropagated": embeddings,
            "logits": logits,
            "logits_unpropagated": logits_unpropagated,
        }


class DSSGNNEncoder(nn.Module):
    """DSS-GNN backbone adapter: forward(x, edge_index) -> logits. Builds L_rescaled from edge_index."""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, use_bn=True,
                 K_lp=3, K_hp=2, P=1, lambda_max=2.0, use_random_gates=False, P_gate=1, quadrature_nodes=4,
                 shared_input_lift=False, dss_use_bn=False):
        super().__init__()
        from dssgnn.model import DSSGNN
        from dssgnn.chebyshev import build_rescaled_laplacian
        self._build_rescaled_laplacian = build_rescaled_laplacian
        self.lambda_max = lambda_max
        self.dssgnn = DSSGNN(
            input_dim=in_channels,
            hidden_dim=hidden_channels,
            out_dim=out_channels,
            num_layers=num_layers,
            K_lp=K_lp,
            K_hp=K_hp,
            P=P,
            dropout=(dropout, 0.0),
            activation=True,
            use_random_gates=use_random_gates,
            P_gate=P_gate,
            S=quadrature_nodes,
            shared_input_lift=shared_input_lift,
            use_bn=dss_use_bn,
        )
        # use_bn is the GCN-style flag; it is intentionally IGNORED here so existing
        # configs (incl. gcn_dssres --use_bn) keep the published DSS branch unchanged.
        # dss_use_bn is the separate opt-in that enables per-channel BN inside DSSGNN.
        self.use_bn = use_bn

    def reset_parameters(self):
        self.dssgnn.reset_parameters()

    def _build_graph_input(self, x, edge_index):
        num_nodes = x.size(0)
        adj = _edge_index_to_adj(edge_index, num_nodes)
        L_rescaled = self._build_rescaled_laplacian(adj, lambda_max=self.lambda_max)
        return L_rescaled.to(x.device)

    def forward(self, x, edge_index):
        L_rescaled = self._build_graph_input(x, edge_index)
        return self.dssgnn(L_rescaled, x)

    def forward_with_uncertainty(self, x, edge_index, uncertainty_type="chaos"):
        L_rescaled = self._build_graph_input(x, edge_index)
        return self.dssgnn(L_rescaled, x, return_uncertainty=True, uncertainty_type=uncertainty_type)

    def forward_with_predictive_stats(self, x, edge_index, eps=1e-12):
        L_rescaled = self._build_graph_input(x, edge_index)
        return self.dssgnn.forward_with_predictive_stats(L_rescaled, x, eps=eps)


class GCNDSSResidualEncoder(nn.Module):
    """
    Hybrid low-label backbone: a label-efficient GCN base plus a DSS residual branch.

    The base GCN is trained first for a configurable warmup period. After warmup,
    DSS contributes a residual logit correction and uncertainty signal.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        dropout=0.5,
        use_bn=True,
        K_lp=3,
        K_hp=2,
        P=1,
        lambda_max=2.0,
        use_random_gates=False,
        P_gate=1,
        quadrature_nodes=4,
        shared_input_lift=False,
        warmup_base_epochs=50,
        residual_scale_init=0.1,
    ):
        super().__init__()
        self.base_gcn = GCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            use_bn=use_bn,
        )
        self.residual_dss = DSSGNNEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            use_bn=use_bn,
            K_lp=K_lp,
            K_hp=K_hp,
            P=P,
            lambda_max=lambda_max,
            use_random_gates=use_random_gates,
            P_gate=P_gate,
            quadrature_nodes=quadrature_nodes,
            shared_input_lift=shared_input_lift,
        )
        self.warmup_base_epochs = max(int(warmup_base_epochs), 0)
        self.residual_scale_init = float(residual_scale_init)
        self.residual_scale = nn.Parameter(torch.tensor(self.residual_scale_init))
        self._train_epoch = 0
        self._set_phase_requires_grad()

    def _residual_active(self):
        return self._train_epoch >= self.warmup_base_epochs

    def _set_phase_requires_grad(self):
        residual_active = self._residual_active()
        for param in self.base_gcn.parameters():
            param.requires_grad_(True)
        for param in self.residual_dss.parameters():
            param.requires_grad_(residual_active)
        self.residual_scale.requires_grad_(residual_active)

    def set_train_epoch(self, epoch):
        self._train_epoch = int(epoch)
        self._set_phase_requires_grad()

    def load_base_state_dict(self, state_dict):
        self.base_gcn.load_state_dict(state_dict)

    def reset_parameters(self):
        self.base_gcn.reset_parameters()
        self.residual_dss.reset_parameters()
        with torch.no_grad():
            self.residual_scale.fill_(self.residual_scale_init)
        self.set_train_epoch(0)

    def forward(self, x, edge_index):
        base_logits = self.base_gcn(x, edge_index)
        if not self._residual_active():
            return base_logits
        residual_logits = self.residual_dss(x, edge_index)
        return base_logits + self.residual_scale * residual_logits

    def forward_with_uncertainty(self, x, edge_index, uncertainty_type="chaos"):
        base_logits = self.base_gcn(x, edge_index)
        if not self._residual_active():
            zeros = torch.zeros(x.size(0), device=x.device, dtype=base_logits.dtype)
            return base_logits, zeros
        residual_logits, uncertainty = self.residual_dss.forward_with_uncertainty(
            x, edge_index, uncertainty_type=uncertainty_type
        )
        scale = self.residual_scale.to(dtype=base_logits.dtype)
        logits = base_logits + scale * residual_logits
        return logits, scale.square() * uncertainty

    def forward_with_predictive_stats(self, x, edge_index, eps=1e-12):
        from dssgnn.model import _predictive_stats_from_sample_logits

        base_logits = self.base_gcn(x, edge_index)
        if not self._residual_active():
            ref_conv = self.residual_dss.dssgnn.convs[-1]
            weights = ref_conv._w.to(device=x.device, dtype=base_logits.dtype)
            sample_logits = base_logits.unsqueeze(0).expand(weights.shape[0], -1, -1)
            stats = _predictive_stats_from_sample_logits(sample_logits, weights, eps=eps)
            stats["logits"] = base_logits
            return stats

        residual_stats = self.residual_dss.forward_with_predictive_stats(x, edge_index, eps=eps)
        scale = self.residual_scale.to(dtype=base_logits.dtype)
        logits = base_logits + scale * residual_stats["logits"]
        sample_logits = base_logits.unsqueeze(0) + scale * residual_stats["sample_logits"]
        stats = _predictive_stats_from_sample_logits(sample_logits, residual_stats["weights"], eps=eps)
        stats["logits"] = logits
        return stats


class SAGEDSSResidualEncoder(nn.Module):
    """
    Hybrid low-label backbone: a GraphSAGE base plus a DSS residual branch.

    The training schedule mirrors GCNDSSResidualEncoder: the deterministic base
    is trained alone during warmup, and the DSS residual branch is activated
    afterward.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        dropout=0.5,
        use_bn=True,
        K_lp=3,
        K_hp=2,
        P=1,
        lambda_max=2.0,
        use_random_gates=False,
        P_gate=1,
        quadrature_nodes=4,
        shared_input_lift=False,
        warmup_base_epochs=50,
        residual_scale_init=0.1,
    ):
        super().__init__()
        self.base_sage = GraphSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            use_bn=use_bn,
        )
        self.residual_dss = DSSGNNEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            use_bn=use_bn,
            K_lp=K_lp,
            K_hp=K_hp,
            P=P,
            lambda_max=lambda_max,
            use_random_gates=use_random_gates,
            P_gate=P_gate,
            quadrature_nodes=quadrature_nodes,
            shared_input_lift=shared_input_lift,
        )
        self.warmup_base_epochs = max(int(warmup_base_epochs), 0)
        self.residual_scale_init = float(residual_scale_init)
        self.residual_scale = nn.Parameter(torch.tensor(self.residual_scale_init))
        self._train_epoch = 0
        self._set_phase_requires_grad()

    def _residual_active(self):
        return self._train_epoch >= self.warmup_base_epochs

    def _set_phase_requires_grad(self):
        residual_active = self._residual_active()
        for param in self.base_sage.parameters():
            param.requires_grad_(True)
        for param in self.residual_dss.parameters():
            param.requires_grad_(residual_active)
        self.residual_scale.requires_grad_(residual_active)

    def set_train_epoch(self, epoch):
        self._train_epoch = int(epoch)
        self._set_phase_requires_grad()

    def load_base_state_dict(self, state_dict):
        self.base_sage.load_state_dict(state_dict)

    def reset_parameters(self):
        self.base_sage.reset_parameters()
        self.residual_dss.reset_parameters()
        with torch.no_grad():
            self.residual_scale.fill_(self.residual_scale_init)
        self.set_train_epoch(0)

    def forward(self, x, edge_index):
        base_logits = self.base_sage(x, edge_index)
        if not self._residual_active():
            return base_logits
        residual_logits = self.residual_dss(x, edge_index)
        return base_logits + self.residual_scale * residual_logits

    def forward_with_uncertainty(self, x, edge_index, uncertainty_type="chaos"):
        base_logits = self.base_sage(x, edge_index)
        if not self._residual_active():
            zeros = torch.zeros(x.size(0), device=x.device, dtype=base_logits.dtype)
            return base_logits, zeros
        residual_logits, uncertainty = self.residual_dss.forward_with_uncertainty(
            x, edge_index, uncertainty_type=uncertainty_type
        )
        scale = self.residual_scale.to(dtype=base_logits.dtype)
        logits = base_logits + scale * residual_logits
        return logits, scale.square() * uncertainty

    def forward_with_predictive_stats(self, x, edge_index, eps=1e-12):
        from dssgnn.model import _predictive_stats_from_sample_logits

        base_logits = self.base_sage(x, edge_index)
        if not self._residual_active():
            ref_conv = self.residual_dss.dssgnn.convs[-1]
            weights = ref_conv._w.to(device=x.device, dtype=base_logits.dtype)
            sample_logits = base_logits.unsqueeze(0).expand(weights.shape[0], -1, -1)
            stats = _predictive_stats_from_sample_logits(sample_logits, weights, eps=eps)
            stats["logits"] = base_logits
            return stats

        residual_stats = self.residual_dss.forward_with_predictive_stats(x, edge_index, eps=eps)
        scale = self.residual_scale.to(dtype=base_logits.dtype)
        logits = base_logits + scale * residual_stats["logits"]
        sample_logits = base_logits.unsqueeze(0) + scale * residual_stats["sample_logits"]
        stats = _predictive_stats_from_sample_logits(sample_logits, residual_stats["weights"], eps=eps)
        stats["logits"] = logits
        return stats


class APPNPDSSResidualEncoder(nn.Module):
    """
    Hybrid low-label backbone: an APPNP base plus a DSS residual branch.

    The training schedule mirrors the GCN and GraphSAGE residual hybrids: the
    deterministic APPNP base is trained alone during warmup, and the DSS
    residual branch is activated afterward.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        dropout=0.5,
        use_bn=True,
        K_lp=3,
        K_hp=2,
        P=1,
        lambda_max=2.0,
        use_random_gates=False,
        P_gate=1,
        quadrature_nodes=4,
        shared_input_lift=False,
        warmup_base_epochs=50,
        residual_scale_init=0.1,
        appnp_K=10,
        appnp_alpha=0.1,
    ):
        super().__init__()
        self.base_appnp = APPNPBackbone(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            use_bn=use_bn,
            appnp_K=appnp_K,
            appnp_alpha=appnp_alpha,
        )
        self.residual_dss = DSSGNNEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            use_bn=use_bn,
            K_lp=K_lp,
            K_hp=K_hp,
            P=P,
            lambda_max=lambda_max,
            use_random_gates=use_random_gates,
            P_gate=P_gate,
            quadrature_nodes=quadrature_nodes,
            shared_input_lift=shared_input_lift,
        )
        self.warmup_base_epochs = max(int(warmup_base_epochs), 0)
        self.residual_scale_init = float(residual_scale_init)
        self.residual_scale = nn.Parameter(torch.tensor(self.residual_scale_init))
        self._train_epoch = 0
        self._set_phase_requires_grad()

    def _residual_active(self):
        return self._train_epoch >= self.warmup_base_epochs

    def _set_phase_requires_grad(self):
        residual_active = self._residual_active()
        for param in self.base_appnp.parameters():
            param.requires_grad_(True)
        for param in self.residual_dss.parameters():
            param.requires_grad_(residual_active)
        self.residual_scale.requires_grad_(residual_active)

    def set_train_epoch(self, epoch):
        self._train_epoch = int(epoch)
        self._set_phase_requires_grad()

    def load_base_state_dict(self, state_dict):
        self.base_appnp.load_state_dict(state_dict)

    def reset_parameters(self):
        self.base_appnp.reset_parameters()
        self.residual_dss.reset_parameters()
        with torch.no_grad():
            self.residual_scale.fill_(self.residual_scale_init)
        self.set_train_epoch(0)

    def forward(self, x, edge_index):
        base_logits = self.base_appnp(x, edge_index)
        if not self._residual_active():
            return base_logits
        residual_logits = self.residual_dss(x, edge_index)
        return base_logits + self.residual_scale * residual_logits

    def forward_with_uncertainty(self, x, edge_index, uncertainty_type="chaos"):
        base_logits = self.base_appnp(x, edge_index)
        if not self._residual_active():
            zeros = torch.zeros(x.size(0), device=x.device, dtype=base_logits.dtype)
            return base_logits, zeros
        residual_logits, uncertainty = self.residual_dss.forward_with_uncertainty(
            x, edge_index, uncertainty_type=uncertainty_type
        )
        scale = self.residual_scale.to(dtype=base_logits.dtype)
        logits = base_logits + scale * residual_logits
        return logits, scale.square() * uncertainty

    def forward_with_predictive_stats(self, x, edge_index, eps=1e-12):
        from dssgnn.model import _predictive_stats_from_sample_logits

        base_logits = self.base_appnp(x, edge_index)
        if not self._residual_active():
            ref_conv = self.residual_dss.dssgnn.convs[-1]
            weights = ref_conv._w.to(device=x.device, dtype=base_logits.dtype)
            sample_logits = base_logits.unsqueeze(0).expand(weights.shape[0], -1, -1)
            stats = _predictive_stats_from_sample_logits(sample_logits, weights, eps=eps)
            stats["logits"] = base_logits
            return stats

        residual_stats = self.residual_dss.forward_with_predictive_stats(x, edge_index, eps=eps)
        scale = self.residual_scale.to(dtype=base_logits.dtype)
        logits = base_logits + scale * residual_stats["logits"]
        sample_logits = base_logits.unsqueeze(0) + scale * residual_stats["sample_logits"]
        stats = _predictive_stats_from_sample_logits(sample_logits, residual_stats["weights"], eps=eps)
        stats["logits"] = logits
        return stats


class GPRDSSResidualEncoder(nn.Module):
    """
    Hybrid low-label backbone: a GPR-GNN base plus a DSS residual branch.

    The training schedule mirrors the other residual hybrids: the deterministic
    GPR base is trained alone during warmup, and the DSS residual branch is
    activated afterward.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        dropout=0.5,
        use_bn=True,
        K_lp=3,
        K_hp=2,
        P=1,
        lambda_max=2.0,
        use_random_gates=False,
        P_gate=1,
        quadrature_nodes=4,
        shared_input_lift=False,
        warmup_base_epochs=50,
        residual_scale_init=0.1,
        gpr_K=10,
        gpr_alpha=0.1,
    ):
        super().__init__()
        self.base_gpr = GPRBackbone(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            use_bn=use_bn,
            gpr_K=gpr_K,
            gpr_alpha=gpr_alpha,
        )
        self.residual_dss = DSSGNNEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            use_bn=use_bn,
            K_lp=K_lp,
            K_hp=K_hp,
            P=P,
            lambda_max=lambda_max,
            use_random_gates=use_random_gates,
            P_gate=P_gate,
            quadrature_nodes=quadrature_nodes,
            shared_input_lift=shared_input_lift,
        )
        self.warmup_base_epochs = max(int(warmup_base_epochs), 0)
        self.residual_scale_init = float(residual_scale_init)
        self.residual_scale = nn.Parameter(torch.tensor(self.residual_scale_init))
        self._train_epoch = 0
        self._set_phase_requires_grad()

    def _residual_active(self):
        return self._train_epoch >= self.warmup_base_epochs

    def _set_phase_requires_grad(self):
        residual_active = self._residual_active()
        for param in self.base_gpr.parameters():
            param.requires_grad_(True)
        for param in self.residual_dss.parameters():
            param.requires_grad_(residual_active)
        self.residual_scale.requires_grad_(residual_active)

    def set_train_epoch(self, epoch):
        self._train_epoch = int(epoch)
        self._set_phase_requires_grad()

    def load_base_state_dict(self, state_dict):
        self.base_gpr.load_state_dict(state_dict)

    def reset_parameters(self):
        self.base_gpr.reset_parameters()
        self.residual_dss.reset_parameters()
        with torch.no_grad():
            self.residual_scale.fill_(self.residual_scale_init)
        self.set_train_epoch(0)

    def forward(self, x, edge_index):
        base_logits = self.base_gpr(x, edge_index)
        if not self._residual_active():
            return base_logits
        residual_logits = self.residual_dss(x, edge_index)
        return base_logits + self.residual_scale * residual_logits

    def forward_with_uncertainty(self, x, edge_index, uncertainty_type="chaos"):
        base_logits = self.base_gpr(x, edge_index)
        if not self._residual_active():
            zeros = torch.zeros(x.size(0), device=x.device, dtype=base_logits.dtype)
            return base_logits, zeros
        residual_logits, uncertainty = self.residual_dss.forward_with_uncertainty(
            x, edge_index, uncertainty_type=uncertainty_type
        )
        scale = self.residual_scale.to(dtype=base_logits.dtype)
        logits = base_logits + scale * residual_logits
        return logits, scale.square() * uncertainty

    def forward_with_predictive_stats(self, x, edge_index, eps=1e-12):
        from dssgnn.model import _predictive_stats_from_sample_logits

        base_logits = self.base_gpr(x, edge_index)
        if not self._residual_active():
            ref_conv = self.residual_dss.dssgnn.convs[-1]
            weights = ref_conv._w.to(device=x.device, dtype=base_logits.dtype)
            sample_logits = base_logits.unsqueeze(0).expand(weights.shape[0], -1, -1)
            stats = _predictive_stats_from_sample_logits(sample_logits, weights, eps=eps)
            stats["logits"] = base_logits
            return stats

        residual_stats = self.residual_dss.forward_with_predictive_stats(x, edge_index, eps=eps)
        scale = self.residual_scale.to(dtype=base_logits.dtype)
        logits = base_logits + scale * residual_stats["logits"]
        sample_logits = base_logits.unsqueeze(0) + scale * residual_stats["sample_logits"]
        stats = _predictive_stats_from_sample_logits(sample_logits, residual_stats["weights"], eps=eps)
        stats["logits"] = logits
        return stats


class TFEEncoder(nn.Module):
    """TFE-GNN backbone: forward(x, edge_index) -> logits. Uses low/high pass filters."""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, use_bn=True,
                 hop_lp=2, hop_hp=2, combine="sum", eta=0.5):
        super().__init__()
        from tfe_utils import propagate_adj, sparse_mx_to_torch_sparse_tensor
        from tfe_models import TFE_GNN
        self._propagate_adj = propagate_adj
        self._to_sparse = sparse_mx_to_torch_sparse_tensor
        self.tfe = TFE_GNN(
            input_dim=in_channels,
            hidden_dim=hidden_channels,
            out_dim=out_channels,
            num_layers=num_layers,
            dropout=(dropout, 0.0),
            activation=True,
            hop=(hop_lp, hop_hp),
            combine=combine,
        )
        self.eta = eta

    def reset_parameters(self):
        if hasattr(self.tfe, "init_parameter"):
            self.tfe.init_parameter()
        else:
            for m in self.tfe.modules():
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()

    def _get_adj_lp_hp(self, edge_index, num_nodes, device):
        adj = _edge_index_to_adj(edge_index, num_nodes)
        adj_lp = self._propagate_adj(adj, "low", -0.5, -0.5)
        adj_hp = self._propagate_adj(adj, "high", self.eta, self.eta)
        return adj_lp.to(device), adj_hp.to(device)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        adj_lp, adj_hp = self._get_adj_lp_hp(edge_index, num_nodes, x.device)
        return self.tfe(adj_hp, adj_lp, x)


class GDUQEncoder(nn.Module):
    """
    GDUQ backbone: DSS-GNN with anchor mechanism. forward(x, edge_index) -> logits.
    Requires mean, std from train features (set via set_anchor_dist before use).
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, use_bn=True,
                 K_lp=3, K_hp=2, P=1, lambda_max=2.0, n_anchors=5, use_random_gates=False, P_gate=1, quadrature_nodes=4,
                 shared_input_lift=False):
        super().__init__()
        from dssgnn.model import DSSGNN
        from dssgnn.chebyshev import build_rescaled_laplacian
        from dssgnn.dssgnn_gduq import DSSGNNWithGDUQAnchor
        self._build_rescaled_laplacian = build_rescaled_laplacian
        self.lambda_max = lambda_max
        self.n_anchors = n_anchors
        self.orig_input_dim = in_channels
        base_kwargs = dict(
            input_dim=in_channels,  # DSSGNNWithGDUQAnchor doubles for concat (x-xi, xi)
            hidden_dim=hidden_channels,
            out_dim=out_channels,
            num_layers=num_layers,
            K_lp=K_lp,
            K_hp=K_hp,
            P=P,
            dropout=(dropout, 0.0),
            activation=True,
            use_random_gates=use_random_gates,
            P_gate=P_gate,
            S=quadrature_nodes,
            shared_input_lift=shared_input_lift,
        )
        self.gduq = DSSGNNWithGDUQAnchor(
            base_model_class=DSSGNN,
            base_model_kwargs=base_kwargs,
            mean=torch.zeros(in_channels),
            std=torch.ones(in_channels),
            anchor_type="node",
        )

    def set_anchor_dist(self, mean, std):
        std_safe = std.clone()
        std_safe[std_safe == 0] = 1e-3
        self.gduq.mean = mean
        self.gduq.std = std_safe
        self.gduq.anchor_dist = torch.distributions.Normal(loc=mean, scale=std_safe)

    def reset_parameters(self):
        self.gduq.base_model.reset_parameters()

    def forward(self, x, edge_index, return_std=False):
        num_nodes = x.size(0)
        adj = _edge_index_to_adj(edge_index, num_nodes)
        L_rescaled = self._build_rescaled_laplacian(adj, lambda_max=self.lambda_max)
        L_rescaled = L_rescaled.to(x.device)
        out = self.gduq(L_rescaled, x, n_anchors=self.n_anchors, return_std=return_std)
        if return_std:
            return out[0], out[1]
        return out

    def forward_with_uncertainty(self, x, edge_index, uncertainty_type="chaos"):
        num_nodes = x.size(0)
        adj = _edge_index_to_adj(edge_index, num_nodes)
        L_rescaled = self._build_rescaled_laplacian(adj, lambda_max=self.lambda_max)
        L_rescaled = L_rescaled.to(x.device)
        logits, uncertainty = self.gduq(
            L_rescaled,
            x,
            n_anchors=self.n_anchors,
            return_uncertainty=True,
            uncertainty_type=uncertainty_type,
        )
        return logits, uncertainty
