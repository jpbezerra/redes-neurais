"""Utilitários gerais: seed e seleção de dispositivo (CPU/GPU)."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fixa as sementes de aleatoriedade para tornar o experimento reprodutível."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Retorna GPU (CUDA), Apple Metal (MPS) ou CPU, na ordem de preferência."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
