"""Tests for the lightweight Graph-EBM integration."""

from types import SimpleNamespace

import torch

from gnnsafe_ood.backbone import GCN
from gnnsafe_ood.baselines import GraphEBM
from gnnsafe_ood.graph_ebm import GraphEBMPosthocScorer


def test_graph_ebm_posthoc_scorer_shapes_and_finiteness():
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [-1.0, 0.0],
            [-0.8, -0.2],
        ],
        dtype=torch.float32,
    )
    logits = torch.tensor(
        [
            [4.0, 0.0],
            [3.0, 0.5],
            [0.0, 4.0],
            [0.5, 3.0],
        ],
        dtype=torch.float32,
    )
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 0, 2], [1, 0, 3, 2, 2, 0]],
        dtype=torch.long,
    )
    y = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    mask = torch.tensor([True, True, True, False])

    scorer = GraphEBMPosthocScorer()
    scorer.fit(logits=logits, embeddings=embeddings, edge_index=edge_index, y=y, mask=mask)
    uncertainty = scorer.get_uncertainty(
        logits_unpropagated=logits,
        embeddings_unpropagated=embeddings,
        edge_index=edge_index,
    )
    assert uncertainty.shape == (4,)
    assert torch.isfinite(uncertainty).all()


def test_gcn_graph_ebm_views_have_expected_shapes():
    model = GCN(in_channels=3, hidden_channels=5, out_channels=2, num_layers=2, dropout=0.0, use_bn=False)
    model.eval()
    x = torch.randn(4, 3)
    edge_index = torch.tensor([[0, 1, 2, 3, 0, 2], [1, 0, 3, 2, 2, 0]], dtype=torch.long)
    views = model.forward_with_graph_ebm_views(x, edge_index)
    assert views["embeddings"].shape == (4, 5)
    assert views["embeddings_unpropagated"].shape == (4, 5)
    assert views["logits"].shape == (4, 2)
    assert views["logits_unpropagated"].shape == (4, 2)


def test_graph_ebm_baseline_detect_runs_after_supervised_training():
    args = SimpleNamespace(
        backbone="gcn",
        hidden_channels=4,
        out_channels=2,
        num_layers=2,
        dropout=0.0,
        use_bn=False,
        dataset="cora",
        dss_lambda_reg=0.0,
        dss_reg_score_type="chaos",
        graph_ebm_covariance_type="diagonal",
        graph_ebm_tied_covariance=False,
        graph_ebm_gamma_correction=1.0,
        graph_ebm_lambda_independent_energy=1.0,
        graph_ebm_lambda_local_energy=1.0,
        graph_ebm_lambda_group_energy=1.0,
        graph_ebm_alpha=0.5,
        graph_ebm_num_diffusion_steps=3,
        graph_ebm_aggregation="sum",
    )
    model = GraphEBM(3, 2, args)
    model.eval()

    class _Dataset:
        pass

    dataset = _Dataset()
    dataset.x = torch.randn(5, 3)
    dataset.edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 0, 2], [1, 0, 3, 2, 0, 4, 4]],
        dtype=torch.long,
    )
    dataset.y = torch.tensor([[0], [0], [1], [1], [0]], dtype=torch.long)
    dataset.splits = {
        "train": torch.tensor([0, 1, 2], dtype=torch.long),
        "valid": torch.tensor([3], dtype=torch.long),
        "test": torch.tensor([4], dtype=torch.long),
    }
    dataset.num_nodes = dataset.x.size(0)

    score = model.detect(dataset, dataset.splits["test"], torch.device("cpu"), SimpleNamespace())
    assert score.shape == (1,)
    assert torch.isfinite(score).all()
