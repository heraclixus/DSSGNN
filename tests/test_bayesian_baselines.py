"""Tests for the MC-dropout and deep-ensemble OOD baselines."""

from types import SimpleNamespace

import torch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnnsafe_ood.baselines import MCDropout, Ensemble, _log_mean_softmax
from gnnsafe_ood.scores import _energy_score


NUM_NODES = 100
NUM_FEATS = 8
NUM_CLASSES = 3


def _args(**kwargs):
    base = dict(
        method="mc_dropout",
        backbone="gcn",
        dataset="synthetic",
        score_type="auto",
        hidden_channels=16,
        num_layers=2,
        dropout=0.5,
        use_bn=False,
        T=1.0,
        use_prop=False,
        K=2,
        alpha=0.5,
        mc_samples=10,
        ensemble_size=3,
        dss_lambda_reg=0.0,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def _synthetic_dataset(seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(NUM_NODES, NUM_FEATS, generator=generator)
    edge_index = torch.randint(0, NUM_NODES, (2, 400), generator=generator)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    y = torch.randint(0, NUM_CLASSES, (NUM_NODES, 1), generator=generator)
    node_idx = torch.arange(NUM_NODES)
    dataset = SimpleNamespace(
        x=x,
        edge_index=edge_index,
        y=y,
        node_idx=node_idx,
        num_nodes=NUM_NODES,
        splits={
            "train": node_idx[:60],
            "valid": node_idx[60:80],
            "test": node_idx[80:],
        },
    )
    return dataset


def _train_few_epochs(model, dataset, args, epochs=5):
    device = torch.device("cpu")
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    losses = []
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = model.loss_compute(dataset, None, criterion, device, args)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def test_mc_dropout_trains_and_detects():
    torch.manual_seed(0)
    dataset = _synthetic_dataset()
    args = _args(method="mc_dropout")
    model = MCDropout(NUM_FEATS, NUM_CLASSES, args)
    losses = _train_few_epochs(model, dataset, args)
    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0]

    model.eval()
    with torch.no_grad():
        score = model.detect(dataset, dataset.splits["test"], torch.device("cpu"), args)
    assert score.shape == (dataset.splits["test"].shape[0],)
    assert torch.isfinite(score).all()
    # detect() must not leave the encoder in train mode.
    assert not model.encoder.training


def test_mc_dropout_passes_are_stochastic():
    torch.manual_seed(0)
    dataset = _synthetic_dataset()
    args = _args(method="mc_dropout", dropout=0.5)
    model = MCDropout(NUM_FEATS, NUM_CLASSES, args)
    model.eval()
    with torch.no_grad():
        sample_logits = model._sample_logits(dataset.x, dataset.edge_index)
    assert sample_logits.shape == (args.mc_samples, NUM_NODES, NUM_CLASSES)
    # With dropout active the MC passes must differ.
    assert sample_logits.std(dim=0).max().item() > 0


def test_mc_dropout_prediction_is_argmax_of_mean_softmax():
    torch.manual_seed(0)
    dataset = _synthetic_dataset()
    args = _args(method="mc_dropout")
    model = MCDropout(NUM_FEATS, NUM_CLASSES, args)
    model.eval()
    torch.manual_seed(7)
    with torch.no_grad():
        out = model(dataset, torch.device("cpu"))
    torch.manual_seed(7)
    with torch.no_grad():
        sample_logits = model._sample_logits(dataset.x, dataset.edge_index)
    mean_probs = torch.softmax(sample_logits, dim=-1).mean(dim=0)
    assert out.shape == (NUM_NODES, NUM_CLASSES)
    assert torch.equal(out.argmax(dim=-1), mean_probs.argmax(dim=-1))
    torch.testing.assert_close(torch.softmax(out, dim=-1), mean_probs)


def test_mc_dropout_pred_entropy_score():
    torch.manual_seed(0)
    dataset = _synthetic_dataset()
    args = _args(method="mc_dropout", score_type="pred_entropy")
    model = MCDropout(NUM_FEATS, NUM_CLASSES, args)
    model.eval()
    with torch.no_grad():
        score = model.detect(dataset, dataset.node_idx, torch.device("cpu"), args)
    assert score.shape == (NUM_NODES,)
    assert torch.isfinite(score).all()
    # Negative predictive entropy is bounded above by zero.
    assert score.max().item() <= 0


def test_ensemble_members_have_different_parameters():
    torch.manual_seed(0)
    dataset = _synthetic_dataset()
    args = _args(method="ensemble")
    model = Ensemble(NUM_FEATS, NUM_CLASSES, args)
    assert len(model.encoders) == args.ensemble_size

    def _first_weight(encoder):
        return encoder.convs[0].lin.weight.detach().clone()

    for i in range(len(model.encoders)):
        for j in range(i + 1, len(model.encoders)):
            assert not torch.allclose(_first_weight(model.encoders[i]), _first_weight(model.encoders[j]))

    # Different random inits must survive reset_parameters as well.
    model.reset_parameters()
    assert not torch.allclose(_first_weight(model.encoders[0]), _first_weight(model.encoders[1]))

    _train_few_epochs(model, dataset, args)
    for i in range(len(model.encoders)):
        for j in range(i + 1, len(model.encoders)):
            assert not torch.allclose(_first_weight(model.encoders[i]), _first_weight(model.encoders[j]))


def test_ensemble_trains_and_detects():
    torch.manual_seed(0)
    dataset = _synthetic_dataset()
    args = _args(method="ensemble")
    model = Ensemble(NUM_FEATS, NUM_CLASSES, args)
    losses = _train_few_epochs(model, dataset, args)
    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0]

    model.eval()
    with torch.no_grad():
        score = model.detect(dataset, dataset.splits["test"], torch.device("cpu"), args)
        out = model(dataset, torch.device("cpu"))
    assert score.shape == (dataset.splits["test"].shape[0],)
    assert torch.isfinite(score).all()
    assert out.shape == (NUM_NODES, NUM_CLASSES)
    assert torch.isfinite(out).all()


def test_ensemble_detect_matches_energy_of_mean_logits():
    torch.manual_seed(0)
    dataset = _synthetic_dataset()
    args = _args(method="ensemble")
    model = Ensemble(NUM_FEATS, NUM_CLASSES, args)
    model.eval()
    with torch.no_grad():
        score = model.detect(dataset, dataset.node_idx, torch.device("cpu"), args)
        mean_logits = model._member_logits(dataset.x, dataset.edge_index).mean(dim=0)
    expected = _energy_score(mean_logits, args.T, args.dataset)
    torch.testing.assert_close(score, expected)


def test_log_mean_softmax_matches_mean_probs():
    torch.manual_seed(0)
    sample_logits = torch.randn(5, 7, NUM_CLASSES)
    out = _log_mean_softmax(sample_logits)
    mean_probs = torch.softmax(sample_logits, dim=-1).mean(dim=0)
    torch.testing.assert_close(out.exp(), mean_probs)


def test_detect_with_propagation():
    torch.manual_seed(0)
    dataset = _synthetic_dataset()
    for model_cls, method in ((MCDropout, "mc_dropout"), (Ensemble, "ensemble")):
        args = _args(method=method, use_prop=True)
        model = model_cls(NUM_FEATS, NUM_CLASSES, args)
        model.eval()
        try:
            with torch.no_grad():
                score = model.detect(dataset, dataset.node_idx, torch.device("cpu"), args)
        except ImportError:
            # Propagation requires torch_sparse; skip in lighter environments.
            continue
        assert score.shape == (NUM_NODES,)
        assert torch.isfinite(score).all()


if __name__ == "__main__":
    test_mc_dropout_trains_and_detects()
    test_mc_dropout_passes_are_stochastic()
    test_mc_dropout_prediction_is_argmax_of_mean_softmax()
    test_mc_dropout_pred_entropy_score()
    test_ensemble_members_have_different_parameters()
    test_ensemble_trains_and_detects()
    test_ensemble_detect_matches_energy_of_mean_logits()
    test_log_mean_softmax_matches_mean_probs()
    test_detect_with_propagation()
    print("All Bayesian baseline tests passed.")
