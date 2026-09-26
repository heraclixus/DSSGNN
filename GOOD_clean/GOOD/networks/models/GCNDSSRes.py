r"""Minimal GOOD-native wrapper for the GCN + DSS residual encoder."""

import sys
from pathlib import Path

import torch

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gnnsafe_ood.backbone import GCNDSSResidualEncoder


@register.model_register
class GCN_DSSRes(GNNBasic):
    r"""GOOD-native node model that wraps our GCN+DSS residual encoder."""

    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)
        self.encoder = GCNDSSResidualEncoder(
            in_channels=config.dataset.dim_node,
            hidden_channels=config.model.dim_hidden,
            out_channels=config.dataset.num_classes,
            num_layers=config.model.model_layer,
            dropout=config.model.dropout_rate,
            use_bn=True,
            K_lp=getattr(config.model, "dss_K_lp", 3),
            K_hp=getattr(config.model, "dss_K_hp", 2),
            P=getattr(config.model, "dss_P", 1),
            lambda_max=getattr(config.model, "dss_lambda_max", 2.0),
            use_random_gates=getattr(config.model, "use_random_gates", False),
            P_gate=getattr(config.model, "dss_P_gate", 1),
            quadrature_nodes=getattr(config.model, "quadrature_nodes", 2),
            shared_input_lift=getattr(config.model, "shared_input_lift", False),
            warmup_base_epochs=getattr(config.model, "warmup_base_epochs", 100),
            residual_scale_init=getattr(config.model, "residual_scale_init", 0.05),
        )

    def forward(self, *args, **kwargs) -> torch.Tensor:
        x, edge_index, _edge_weight, _batch = self.arguments_read(*args, **kwargs)
        if hasattr(self.encoder, "set_train_epoch"):
            self.encoder.set_train_epoch(getattr(self.config.train, "epoch", 0))
        return self.encoder(x, edge_index)
