"""Utilities for fitting small OOD score-fusion detectors on top of a backbone."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .scores import compute_detection_score


FUSION_FEATURE_SPECS = {
    "msp": {"score_type": "msp", "use_prop": False},
    "energy": {"score_type": "energy", "use_prop": False},
    "energy_prop": {"score_type": "energy", "use_prop": True},
    "pred_entropy": {"score_type": "pred_entropy", "use_prop": False},
    "pred_entropy_prop": {"score_type": "pred_entropy", "use_prop": True},
    "mutual_info": {"score_type": "mutual_info", "use_prop": False},
    "mutual_info_prop": {"score_type": "mutual_info", "use_prop": True},
    "chaos": {"score_type": "chaos", "use_prop": False},
    "chaos_prop": {"score_type": "chaos", "use_prop": True},
    "chaos_norm": {"score_type": "chaos_norm", "use_prop": False},
    "chaos_ratio": {"score_type": "chaos_ratio", "use_prop": False},
    "chaos_ratio_prop": {"score_type": "chaos_ratio", "use_prop": True},
    "chaos_layerratio": {"score_type": "chaos_layerratio", "use_prop": False},
    "chaos_layerratio_prop": {"score_type": "chaos_layerratio", "use_prop": True},
}


def _clone_args(args, **overrides):
    data = dict(vars(args))
    data.update(overrides)
    return SimpleNamespace(**data)


def _resolve_ind_idx(dataset_ind, ind_split):
    if ind_split == "train":
        return dataset_ind.splits["train"]
    if ind_split == "valid":
        return dataset_ind.splits["valid"]
    if ind_split in {"train+valid", "train_valid", "train-valid"}:
        return torch.cat([dataset_ind.splits["train"], dataset_ind.splits["valid"]], dim=0)
    raise ValueError(f"Unsupported fusion ind_split: {ind_split}")


@dataclass
class ScoreFusionCalibrator:
    feature_names: tuple[str, ...]
    mean_: np.ndarray
    scale_: np.ndarray
    coef_: np.ndarray
    intercept_: float
    c: float
    ind_split: str
    train_accuracy: float

    def transform(self, feature_matrix: np.ndarray) -> np.ndarray:
        scale = np.where(self.scale_ == 0, 1.0, self.scale_)
        return (feature_matrix - self.mean_) / scale

    def score_matrix(self, feature_matrix: np.ndarray) -> np.ndarray:
        transformed = self.transform(feature_matrix)
        return transformed @ self.coef_ + self.intercept_

    def to_dict(self) -> dict:
        return {
            "feature_names": list(self.feature_names),
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
            "coef": self.coef_.tolist(),
            "intercept": float(self.intercept_),
            "c": float(self.c),
            "ind_split": self.ind_split,
            "train_accuracy": float(self.train_accuracy),
        }


@torch.no_grad()
def build_feature_matrix(encoder, dataset, device, args, feature_names, node_idx=None):
    """Return an (N, D) numpy matrix of scalar score features for the given dataset."""
    if hasattr(encoder, "eval"):
        encoder.eval()
    columns = []
    for feature_name in feature_names:
        if feature_name not in FUSION_FEATURE_SPECS:
            raise ValueError(f"Unknown fusion feature: {feature_name}")
        spec = FUSION_FEATURE_SPECS[feature_name]
        feature_args = _clone_args(args, score_type=spec["score_type"], use_prop=spec["use_prop"])
        score = compute_detection_score(encoder, dataset, device, feature_args).detach().cpu()
        if node_idx is not None:
            index = node_idx.detach().cpu()
            score = score.index_select(0, index)
        columns.append(score.numpy())
    return np.column_stack(columns)


@torch.no_grad()
def fit_logistic_score_fusion(
    encoder,
    dataset_ind,
    dataset_ood,
    device,
    args,
    feature_names,
    ind_split="valid",
    c=1.0,
    max_iter=1000,
):
    """Fit a linear IND-vs-OOD detector over score features using OOD-train data."""
    ind_idx = _resolve_ind_idx(dataset_ind, ind_split)
    ood_idx = dataset_ood.node_idx
    x_ind = build_feature_matrix(encoder, dataset_ind, device, args, feature_names, node_idx=ind_idx)
    x_ood = build_feature_matrix(encoder, dataset_ood, device, args, feature_names, node_idx=ood_idx)

    x = np.concatenate([x_ind, x_ood], axis=0)
    y = np.concatenate([np.ones(x_ind.shape[0], dtype=np.int64), np.zeros(x_ood.shape[0], dtype=np.int64)], axis=0)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    clf = LogisticRegression(
        C=c,
        max_iter=max_iter,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(x_scaled, y)
    train_accuracy = float((clf.predict(x_scaled) == y).mean())

    return ScoreFusionCalibrator(
        feature_names=tuple(feature_names),
        mean_=scaler.mean_.astype(np.float64, copy=True),
        scale_=scaler.scale_.astype(np.float64, copy=True),
        coef_=clf.coef_[0].astype(np.float64, copy=True),
        intercept_=float(clf.intercept_[0]),
        c=float(c),
        ind_split=ind_split,
        train_accuracy=train_accuracy,
    )


@torch.no_grad()
def score_with_fusion(encoder, dataset, device, args, calibrator):
    """Apply a fitted score-fusion detector to all nodes in a dataset."""
    features = build_feature_matrix(encoder, dataset, device, args, calibrator.feature_names)
    score = calibrator.score_matrix(features)
    return torch.from_numpy(score).to(torch.float32)
