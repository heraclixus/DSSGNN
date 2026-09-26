"""Compatibility helpers for running GOOD datasets in modern PyTorch envs."""

from __future__ import annotations

import torch


def torch_load_compat(path):
    """Load GOOD processed dataset files across PyTorch versions.

    PyTorch 2.6 changed ``torch.load`` to default to ``weights_only=True``,
    which breaks loading PyG ``Data`` objects from GOOD's cached dataset
    splits. GOOD's processed files are trusted local artifacts in this
    workspace, so we explicitly opt back into full loading here.
    """

    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)
