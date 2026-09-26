"""Tests for OOD score selection utilities."""

from types import SimpleNamespace

import numpy as np
import torch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnnsafe_ood.scores import (
    compute_backbone_uncertainty_penalty,
    compute_detection_score,
    compute_regularizer_signal,
    default_score_type,
    propagate_scalar_score,
    resolve_score_type,
    score_variant_suffix,
)
from gnnsafe_ood.fusion import build_feature_matrix, fit_logistic_score_fusion, score_with_fusion


class _DummyDataset:
    def __init__(self):
        self.x = torch.randn(3, 4)
        self.edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        self.splits = {"train": torch.tensor([0, 1], dtype=torch.long)}


class _LogitEncoder:
    def __call__(self, x, edge_index):
        return torch.tensor(
            [[2.0, 0.0], [0.0, 0.0], [-1.0, 3.0]],
            dtype=x.dtype,
            device=x.device,
        )


class _ChaosEncoder(_LogitEncoder):
    def forward_with_uncertainty(self, x, edge_index, uncertainty_type="chaos"):
        logits = self(x, edge_index)
        if uncertainty_type == "chaos":
            uncertainty = torch.tensor([0.1, 0.5, 1.2], dtype=x.dtype, device=x.device)
        elif uncertainty_type == "chaos_norm":
            uncertainty = torch.tensor([0.05, 0.25, 0.6], dtype=x.dtype, device=x.device)
        elif uncertainty_type == "chaos_layernorm":
            uncertainty = torch.tensor([0.2, 0.7, 1.5], dtype=x.dtype, device=x.device)
        elif uncertainty_type == "chaos_ratio":
            uncertainty = torch.tensor([0.25, 0.5, 0.75], dtype=x.dtype, device=x.device)
        elif uncertainty_type == "chaos_layerratio":
            uncertainty = torch.tensor([0.3, 0.55, 0.8], dtype=x.dtype, device=x.device)
        else:
            raise ValueError(f"unexpected uncertainty_type: {uncertainty_type}")
        return logits, uncertainty


class _PredictiveEncoder(_LogitEncoder):
    def __call__(self, x, edge_index):
        return torch.tensor(
            [[3.0, 0.0], [0.5, 0.5], [0.0, 2.0]],
            dtype=x.dtype,
            device=x.device,
        )

    def forward_with_predictive_stats(self, x, edge_index, eps=1e-12):
        logits = self(x, edge_index)
        sample_logits = torch.tensor(
            [
                [[4.0, 0.0], [3.0, 0.0], [0.0, 2.0]],
                [[2.0, 0.0], [0.0, 3.0], [0.0, 2.0]],
            ],
            dtype=x.dtype,
            device=x.device,
        )
        weights = torch.tensor([0.5, 0.5], dtype=x.dtype, device=x.device)
        sample_probs = torch.softmax(sample_logits, dim=-1)
        predictive_probs = (weights.view(-1, 1, 1) * sample_probs).sum(dim=0)
        predictive_probs = predictive_probs.clamp_min(eps)
        predictive_probs = predictive_probs / predictive_probs.sum(dim=-1, keepdim=True)
        predictive_entropy = -(predictive_probs * predictive_probs.log()).sum(dim=-1)
        expected_entropy = (
            weights.view(-1, 1)
            * -(sample_probs.clamp_min(eps) * sample_probs.clamp_min(eps).log()).sum(dim=-1)
        ).sum(dim=0)
        mutual_info = predictive_entropy - expected_entropy
        return {
            "logits": logits,
            "sample_logits": sample_logits,
            "weights": weights,
            "predictive_probs": predictive_probs,
            "predictive_entropy": predictive_entropy,
            "expected_entropy": expected_entropy,
            "mutual_info": mutual_info,
        }


class _StdEncoder(_LogitEncoder):
    def __call__(self, x, edge_index, return_std=False):
        logits = super().__call__(x, edge_index)
        if return_std:
            std = torch.tensor(
                [[0.2, 0.4], [0.5, 0.5], [1.0, 0.8]],
                dtype=x.dtype,
                device=x.device,
            )
            return logits, std
        return logits


class _FusionDataset:
    def __init__(self, x_values, splits=None):
        self.x = torch.tensor(x_values, dtype=torch.float32).unsqueeze(1)
        num_nodes = self.x.shape[0]
        src = torch.arange(num_nodes - 1, dtype=torch.long)
        dst = torch.arange(1, num_nodes, dtype=torch.long)
        self.edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
        self.node_idx = torch.arange(num_nodes, dtype=torch.long)
        self.splits = splits or {}
        self.y = torch.zeros((num_nodes, 1), dtype=torch.long)


class _FusionEncoder:
    def __call__(self, x, edge_index):
        base = x.squeeze(1)
        return torch.stack([base, torch.zeros_like(base)], dim=1)

    def forward_with_uncertainty(self, x, edge_index, uncertainty_type="chaos"):
        logits = self(x, edge_index)
        base = x.squeeze(1)
        chaos = torch.sigmoid(-base)
        if uncertainty_type in {"chaos", "chaos_ratio", "chaos_layerratio"}:
            return logits, chaos
        if uncertainty_type in {"chaos_norm", "chaos_layernorm"}:
            return logits, 0.5 * chaos
        raise ValueError(f"unexpected uncertainty_type: {uncertainty_type}")


def _args(**kwargs):
    base = dict(
        method="msp",
        score_type="auto",
        dataset="cora",
        T=1.0,
        use_prop=False,
        K=1,
        alpha=0.5,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_resolve_score_type_defaults():
    assert default_score_type(_args(method="msp")) == "msp"
    assert default_score_type(_args(method="gnnsafe", use_reg=True, reg_score_type="chaos")) == "chaos"
    assert default_score_type(_args(method="gnnsafe", use_reg=True, reg_score_type="pred_entropy")) == "pred_entropy"
    assert resolve_score_type(_args(method="msp")) == "msp"
    assert resolve_score_type(_args(method="gnnsafe")) == "energy"
    assert resolve_score_type(_args(method="gduq")) == "std"
    assert resolve_score_type(_args(method="gnnsafe", use_reg=True, reg_score_type="chaos")) == "chaos"
    assert resolve_score_type(_args(method="gnnsafe", use_reg=True, reg_score_type="pred_entropy")) == "pred_entropy"


def test_score_variant_suffix():
    assert score_variant_suffix(_args(method="gnnsafe")) == ""
    assert score_variant_suffix(_args(method="msp", score_type="chaos")) == "chaos"
    assert score_variant_suffix(_args(method="msp", score_type="chaos", use_prop=True)) == "chaos_prop"
    assert score_variant_suffix(_args(method="gnnsafe", use_reg=True, reg_score_type="chaos", use_prop=True)) == ""


def test_compute_chaos_score_is_negative_uncertainty():
    dataset = _DummyDataset()
    args = _args(score_type="chaos")
    score = compute_detection_score(_ChaosEncoder(), dataset, torch.device("cpu"), args)
    torch.testing.assert_close(score, torch.tensor([-0.1, -0.5, -1.2]))


def test_compute_std_score_is_negative_mean_std():
    dataset = _DummyDataset()
    args = _args(method="gduq", score_type="std")
    score = compute_detection_score(_StdEncoder(), dataset, torch.device("cpu"), args)
    torch.testing.assert_close(score, torch.tensor([-0.3, -0.5, -0.9]))


def test_compute_chaos_regularizer_signal_is_positive_uncertainty():
    dataset = _DummyDataset()
    args = _args(method="gnnsafe", use_reg=True, reg_score_type="chaos")
    score = compute_regularizer_signal(_ChaosEncoder(), dataset, torch.device("cpu"), args)
    torch.testing.assert_close(score, torch.tensor([0.1, 0.5, 1.2]))


def test_compute_predictive_entropy_regularizer_signal():
    dataset = _DummyDataset()
    encoder = _PredictiveEncoder()
    args = _args(method="gnnsafe", use_reg=True, reg_score_type="pred_entropy")
    score = compute_regularizer_signal(encoder, dataset, torch.device("cpu"), args)
    expected = encoder.forward_with_predictive_stats(dataset.x, dataset.edge_index)["predictive_entropy"]
    torch.testing.assert_close(score, expected)


def test_compute_backbone_uncertainty_penalty_matches_mean_uncertainty():
    dataset = _DummyDataset()
    args = _args(dss_lambda_reg=0.2, dss_reg_score_type="chaos")
    penalty = compute_backbone_uncertainty_penalty(
        _ChaosEncoder(),
        dataset,
        dataset.splits["train"],
        torch.device("cpu"),
        args,
    )
    torch.testing.assert_close(penalty, torch.tensor(0.06))


def test_compute_normalized_chaos_scores():
    dataset = _DummyDataset()
    args = _args(score_type="chaos_norm")
    score = compute_detection_score(_ChaosEncoder(), dataset, torch.device("cpu"), args)
    torch.testing.assert_close(score, torch.tensor([-0.05, -0.25, -0.6]))

    args = _args(score_type="chaos_layernorm")
    score = compute_detection_score(_ChaosEncoder(), dataset, torch.device("cpu"), args)
    torch.testing.assert_close(score, torch.tensor([-0.2, -0.7, -1.5]))

    args = _args(score_type="chaos_ratio")
    score = compute_detection_score(_ChaosEncoder(), dataset, torch.device("cpu"), args)
    torch.testing.assert_close(score, torch.tensor([-0.25, -0.5, -0.75]))

    args = _args(score_type="chaos_layerratio")
    score = compute_detection_score(_ChaosEncoder(), dataset, torch.device("cpu"), args)
    torch.testing.assert_close(score, torch.tensor([-0.3, -0.55, -0.8]))


def test_compute_hybrid_energy_chaos_ratio_score():
    dataset = _DummyDataset()
    args = _args(score_type="energy_chaos_ratio", chaos_weight=1.0)
    score = compute_detection_score(_ChaosEncoder(), dataset, torch.device("cpu"), args)
    energy = torch.tensor([2.1269, 0.6931, 3.0181])
    expected = energy - torch.tensor([0.25, 0.5, 0.75])
    torch.testing.assert_close(score, expected, atol=1e-4, rtol=1e-4)


def test_predictive_scores_follow_ind_score_convention():
    dataset = _DummyDataset()
    args = _args(score_type="pred_entropy")
    pred_entropy_score = compute_detection_score(_PredictiveEncoder(), dataset, torch.device("cpu"), args)
    assert pred_entropy_score[0].item() > pred_entropy_score[1].item()

    args = _args(score_type="mutual_info")
    mutual_info_score = compute_detection_score(_PredictiveEncoder(), dataset, torch.device("cpu"), args)
    assert mutual_info_score[0].item() > mutual_info_score[1].item()


def test_energy_mutual_info_score_matches_formula():
    dataset = _DummyDataset()
    encoder = _PredictiveEncoder()
    args = _args(score_type="energy_mutual_info", mi_weight=0.7)
    score = compute_detection_score(encoder, dataset, torch.device("cpu"), args)
    energy = compute_detection_score(encoder, dataset, torch.device("cpu"), _args(score_type="energy"))
    mutual_info = -compute_detection_score(encoder, dataset, torch.device("cpu"), _args(score_type="mutual_info"))
    expected = energy - 0.7 * mutual_info
    torch.testing.assert_close(score, expected)


def test_build_feature_matrix_for_fusion():
    dataset = _FusionDataset([2.0, 1.0, -1.0])
    args = _args(score_type="energy", chaos_weight=1.0)
    features = build_feature_matrix(
        _FusionEncoder(),
        dataset,
        torch.device("cpu"),
        args,
        feature_names=["energy", "chaos_ratio"],
    )
    assert features.shape == (3, 2)
    assert features[0, 0] > features[-1, 0]
    assert features[0, 1] > features[-1, 1]


def test_build_predictive_feature_matrix_for_fusion():
    dataset = _DummyDataset()
    args = _args(score_type="pred_entropy")
    features = build_feature_matrix(
        _PredictiveEncoder(),
        dataset,
        torch.device("cpu"),
        args,
        feature_names=["pred_entropy", "mutual_info"],
    )
    assert features.shape == (3, 2)
    assert np.isfinite(features).all()


def test_fit_logistic_score_fusion_separates_ind_from_ood():
    dataset_ind = _FusionDataset(
        [2.0, 1.5, 0.9, 0.7],
        splits={
            "train": torch.tensor([0, 1], dtype=torch.long),
            "valid": torch.tensor([2, 3], dtype=torch.long),
            "test": torch.tensor([0, 1], dtype=torch.long),
        },
    )
    dataset_ood = _FusionDataset([-0.2, -1.0, -1.5])
    args = _args(score_type="energy", chaos_weight=1.0)

    calibrator = fit_logistic_score_fusion(
        _FusionEncoder(),
        dataset_ind,
        dataset_ood,
        torch.device("cpu"),
        args,
        feature_names=["energy", "chaos_ratio"],
        ind_split="valid",
    )
    ind_score = score_with_fusion(_FusionEncoder(), dataset_ind, torch.device("cpu"), args, calibrator)
    ood_score = score_with_fusion(_FusionEncoder(), dataset_ood, torch.device("cpu"), args, calibrator)
    assert ind_score[dataset_ind.splits["test"]].mean().item() > ood_score.mean().item()


def test_propagate_scalar_score_simple_two_node_graph():
    try:
        score = torch.tensor([1.0, 3.0])
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        propagated = propagate_scalar_score(score, edge_index, prop_layers=1, alpha=0.5)
        torch.testing.assert_close(propagated, torch.tensor([2.0, 2.0]))
    except ImportError:
        # Propagation requires torch_sparse; skip in lighter environments.
        pass


def test_propagated_predictive_score_shape_and_finiteness():
    dataset = _DummyDataset()
    try:
        score = compute_detection_score(
            _PredictiveEncoder(),
            dataset,
            torch.device("cpu"),
            _args(score_type="pred_entropy", use_prop=True, K=1, alpha=0.5),
        )
        assert score.shape == (dataset.x.shape[0],)
        assert torch.isfinite(score).all()
    except ImportError:
        pass


if __name__ == "__main__":
    test_resolve_score_type_defaults()
    test_score_variant_suffix()
    test_compute_chaos_score_is_negative_uncertainty()
    test_compute_std_score_is_negative_mean_std()
    test_compute_backbone_uncertainty_penalty_matches_mean_uncertainty()
    test_predictive_scores_follow_ind_score_convention()
    test_energy_mutual_info_score_matches_formula()
    test_build_feature_matrix_for_fusion()
    test_build_predictive_feature_matrix_for_fusion()
    test_fit_logistic_score_fusion_separates_ind_from_ood()
    test_propagate_scalar_score_simple_two_node_graph()
    test_propagated_predictive_score_shape_and_finiteness()
    print("All OOD score tests passed.")
